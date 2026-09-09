"""Standalone FlyLM dynamics inference. Requires numpy and compatible PyTorch."""
import argparse
import codecs
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

BOS, EOS, VOCAB = 256, 257, 258

class SparseEdges(torch.autograd.Function):

    @staticmethod
    def forward(ctx, h, weights, rowptr, col, dst, block):
        matrix = torch.sparse_csr_tensor(rowptr, col, weights, size=(h.shape[1], h.shape[1]), device=h.device, dtype=h.dtype)
        ctx.save_for_backward(h, weights, rowptr, col, dst)
        ctx.block = block
        return torch.sparse.mm(matrix, h.T.contiguous()).T.contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        h, weights, rowptr, col, dst = ctx.saved_tensors
        grad_h = torch.zeros_like(h) if ctx.needs_input_grad[0] else None
        grad_w = torch.empty_like(weights) if ctx.needs_input_grad[1] else None
        for lo in range(0, col.numel(), ctx.block):
            hi = min(col.numel(), lo + ctx.block)
            sources, targets = (col[lo:hi], dst[lo:hi])
            g = grad_output.index_select(1, targets)
            if grad_w is not None:
                grad_w[lo:hi] = (g * h.index_select(1, sources)).sum(0)
            if grad_h is not None:
                grad_h.index_add_(1, sources, g * weights[lo:hi])
        return (grad_h, grad_w, None, None, None, None)

class ConnectomeLM(nn.Module):

    def __init__(self, cfg, rowptr, src, counts, edge_sign, inputs, outputs, type_ids, device):
        super().__init__()
        self.n, self.k = (cfg['n_neurons'], cfg['brain_steps'])
        self.block, self.checkpoint_tokens = (cfg['edge_block'], cfg['checkpoint_tokens'])
        for name, value in [('rowptr', rowptr), ('src', src), ('inputs', inputs), ('outputs', outputs), ('type_ids', type_ids)]:
            self.register_buffer(name, torch.tensor(np.asarray(value), dtype=torch.long, device=device), persistent=False)
        assert len(type_ids) == self.n and min(type_ids) >= 0
        self.register_buffer('dst', torch.repeat_interleave(torch.arange(self.n, device=device), self.rowptr.diff()), persistent=False)
        sign = torch.tensor(np.asarray(edge_sign), dtype=torch.float32, device=device)
        self.register_buffer('edge_sign', sign, persistent=False)
        c = torch.tensor(np.asarray(counts), dtype=torch.float32, device=device).log1p()
        incoming = torch.zeros(self.n, device=device).index_add_(0, self.dst, c)
        self.register_buffer('prior', c / incoming[self.dst].clamp_min(1e-08), persistent=False)
        raw = torch.where(sign != 0, torch.full_like(sign, math.log(0.2 / 0.8)), torch.where(torch.rand_like(sign) < 0.5, -0.2554128, 0.2554128))
        self.theta = nn.Parameter(raw)
        ntypes = int(max(type_ids)) + 1
        low, high, initial = [torch.tensor(cfg[key], device=device)[:, None] for key in ['tau_min', 'tau_max', 'tau_initial']]
        assert torch.all((low > 0) & (initial > low) & (initial < high))
        self.register_buffer('tau_low', low)
        self.register_buffer('tau_high', high)
        fraction = (initial - low) / (high - low)
        self.raw_tau = nn.Parameter(torch.logit(fraction).expand(3, ntypes).clone())
        self.threshold = nn.Parameter(torch.zeros(ntypes, device=device))
        self.raw_adaptation = nn.Parameter(torch.full((ntypes,), math.log(0.1 / 0.9), device=device))
        d = cfg['embedding_dim']
        self.embedding = nn.Embedding(VOCAB, d, device=device)
        self.input_adapter = nn.Linear(d, len(inputs), bias=False, device=device)
        self.motor_norm = nn.LayerNorm(len(outputs), eps=1e-06, elementwise_affine=False, device=device)
        self.output_adapter = nn.Linear(len(outputs), d, bias=False, device=device)

    def effective_weights(self):
        magnitude = 0.25 + 3.75 * self.theta.sigmoid()
        signed = torch.where(self.edge_sign != 0, self.edge_sign * magnitude, 4 * self.theta.tanh())
        return self.prior * signed

    def dynamics(self):
        tau = self.tau_low + (self.tau_high - self.tau_low) * self.raw_tau.sigmoid()
        alpha = -torch.expm1(-1 / tau[:, self.type_ids])
        threshold = 0.5 * self.threshold.tanh()[self.type_ids]
        adaptation = self.raw_adaptation.sigmoid()[self.type_ids]
        return (alpha, threshold, adaptation)

    def initial_state(self, batch):
        return self.embedding.weight.new_zeros(3, batch, self.n)

    def advance(self, token, state, weights, alpha, threshold, adaptation):
        v, signal, slow = state.unbind(0)
        value = self.input_adapter(self.embedding(token))
        drive = torch.zeros_like(v).index_add(1, self.inputs, value)
        for _ in range(self.k):
            message = SparseEdges.apply(signal, weights, self.rowptr, self.src, self.dst, self.block)
            v = v + alpha[0] * (-v + message + drive - adaptation * slow)
            rate = torch.relu(torch.tanh(v - threshold))
            signal = signal + alpha[1] * (rate - signal)
            slow = slow + alpha[2] * (rate - slow)
        return torch.stack((v, signal, slow))

    def forward(self, tokens, state=None, active_mask=None):
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError('Nonempty tokens [B,T] required')
        state = self.initial_state(tokens.shape[0]) if state is None else state
        if state.shape != (3, tokens.shape[0], self.n):
            raise ValueError('State must be [3,B,N]: membrane, synaptic signal, adaptation')
        if active_mask is not None and (active_mask.shape != tokens.shape or active_mask.dtype != torch.bool):
            raise ValueError('active_mask must be bool [B,T]')
        weights = self.effective_weights()
        alpha, threshold, adaptation = self.dynamics()
        logits = []
        for t in range(tokens.shape[1]):
            previous = state
            args = (tokens[:, t], state, weights, alpha, threshold, adaptation)
            if self.training and self.checkpoint_tokens and torch.is_grad_enabled():
                state = checkpoint(self.advance, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                state = self.advance(*args)
            if active_mask is not None:
                state = torch.where(active_mask[:, t][None, :, None], state, previous)
            motor = self.motor_norm(state[1].index_select(1, self.outputs))
            logits.append(F.linear(self.output_adapter(motor), self.embedding.weight))
        return (torch.stack(logits, dim=1), state)

def inference_inputs(question):
    question = ' '.join(question.split())
    if not question:
        raise ValueError('Nonempty question required')
    return [BOS] + list(('화자1: ' + question + '\n화자2: ').encode('utf-8'))

@torch.no_grad()
def prefill_question(m, question):
    prefix = inference_inputs(question)
    state = None
    for start in range(0, len(prefix), m.inference_tbptt):
        ids = torch.tensor([prefix[start:start + m.inference_tbptt]], device=m.embedding.weight.device)
        logits, state = m(ids, state)
    return (logits, state)

def valid_utf8_tokens(decoder):
    state = decoder.getstate()
    probe = codecs.getincrementaldecoder('utf-8')('strict')
    allowed = []
    for value in range(256):
        probe.setstate(state)
        try:
            probe.decode(bytes([value]), final=False)
            allowed.append(value)
        except UnicodeDecodeError:
            pass
    if not state[0]:
        allowed.append(EOS)
    return allowed

@torch.no_grad()
def reply(m, question, max_new_bytes=180, temperature=0.8):
    if max_new_bytes < 1 or temperature < 0:
        raise ValueError('max_new_bytes >= 1, temperature >= 0 required')
    was_training = m.training
    m.eval()
    decoder = codecs.getincrementaldecoder('utf-8')('strict')
    pieces = []
    try:
        logits, state = prefill_question(m, question)
        for _ in range(max_new_bytes):
            allowed = torch.tensor(valid_utf8_tokens(decoder), device=logits.device)
            scores = logits[0, -1].index_select(0, allowed)
            if not torch.isfinite(scores).all():
                raise RuntimeError('Nonfinite generation logits')
            selected = scores.argmax() if temperature == 0 else torch.multinomial((scores / temperature).softmax(-1), 1)[0]
            token = int(allowed[selected])
            if token == EOS:
                break
            pieces.append(decoder.decode(bytes([token]), final=False))
            logits, state = m(torch.tensor([[token]], device=logits.device), state)
        return ''.join(pieces)
    finally:
        m.train(was_training)

def load_model(checkpoint_path, device=None):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if saved.get('architecture') != 'malecns_dynamics_v1':
        raise ValueError('이 노트북의 dynamics checkpoint만 복원할 수 있습니다.')
    folder = Path(checkpoint_path).parent / saved['graph_directory']
    info = json.loads((folder / 'complete.json').read_text())
    count = info['nnz']
    restored = ConnectomeLM(saved['config'], np.load(folder / 'crow.npy', mmap_mode='r'), np.load(folder / 'src.npy', mmap_mode='r')[:count], np.load(folder / 'count.npy', mmap_mode='r')[:count], np.load(folder / 'edge_sign.npy', mmap_mode='r'), np.load(folder / 'inputs.npy'), np.load(folder / 'outputs.npy'), np.load(folder / 'type_ids.npy'), device)
    restored.load_state_dict(saved['model_state'])
    restored.inference_tbptt = saved['config']['tbptt']
    restored.checkpoint_tokens = False
    restored.requires_grad_(False)
    return restored.eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(__file__).with_name('model.pt'))
    parser.add_argument('--question', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-new-bytes', type=int, default=180)
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.device)
    print(reply(model, args.question, max_new_bytes=args.max_new_bytes, temperature=args.temperature))
