
# FlyLM-2 SentencePiece 혼합 동역학 추론 모듈. 학습 노트북과 함께 내보낸다.
import codecs
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

def encode_text(tokenizer, text):
    # SentencePiece의 공백 표식과 실제 U+2581 문자를 구분해 원문을 보존한다.
    if '▁' not in text:
        return tokenizer.encode(text, out_type=int)
    parts = text.split('▁')
    ids = []
    literal = [tokenizer.piece_to_id(f'<0x{value:02X}>') for value in '▁'.encode('utf-8')]
    for i, part in enumerate(parts):
        if i:
            ids.extend(literal)
        ids.extend(tokenizer.encode(part, out_type=int))
    return ids


def render_prompt(question):
    return '화자1: ' + question + '\n화자2: '


def tokenizer_metadata(tokenizer):
    return dict(vocab_size=tokenizer.get_piece_size(), bos_id=tokenizer.bos_id(),
                eos_id=tokenizer.eos_id(), pad_id=tokenizer.pad_id(), unk_id=tokenizer.unk_id())


def validate_tokenizer(tokenizer, expected=None):
    meta = tokenizer_metadata(tokenizer)
    if (meta['bos_id'], meta['eos_id'], meta['pad_id'], meta['unk_id']) != (1, 2, 3, 0):
        raise ValueError('특수 토큰 설정이 다르다: BOS=1, EOS=2, PAD=3, UNK=0이어야 한다.')
    if not 260 <= meta['vocab_size'] <= 65536:
        raise ValueError('바이트 fallback과 uint16 캐시에 맞는 어휘 크기가 필요하다.')
    for value in range(256):
        piece = f'<0x{value:02X}>'
        index = tokenizer.piece_to_id(piece)
        if not tokenizer.is_byte(index) or tokenizer.id_to_piece(index) != piece:
            raise ValueError('모든 256개 바이트 fallback 토큰이 필요하다.')
    if expected is not None and any(expected[key] != value for key, value in meta.items()):
        raise ValueError('토크나이저의 어휘/특수 토큰 정보와 체크포인트 설정이 다르다.')
    return meta

class SparseEdges(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, weights, rowptr, col, dst, block):
        matrix = torch.sparse_csr_tensor(rowptr, col, weights, size=(h.shape[1], h.shape[1]),
                                         device=h.device, dtype=h.dtype)
        ctx.save_for_backward(h, weights, rowptr, col, dst)
        ctx.block = block
        return torch.sparse.mm(matrix, h.T.contiguous()).T.contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        h, weights, rowptr, col, dst = ctx.saved_tensors
        grad_h = torch.zeros_like(h) if ctx.needs_input_grad[0] else None
        grad_w = torch.empty_like(weights) if ctx.needs_input_grad[1] else None
        for lo in range(0, col.numel(), ctx.block):
            hi = min(col.numel(), lo+ctx.block)
            sources, targets = col[lo:hi], dst[lo:hi]
            g = grad_output.index_select(1, targets)
            if grad_w is not None:
                grad_w[lo:hi] = (g*h.index_select(1, sources)).sum(0)
            if grad_h is not None:
                grad_h.index_add_(1, sources, g*weights[lo:hi])
        return grad_h, grad_w, None, None, None, None

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage_gap, slope):
        ctx.save_for_backward(voltage_gap)
        ctx.slope = slope
        return (voltage_gap >= 0).to(voltage_gap.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (gap,) = ctx.saved_tensors
        # 순전파는 이산 사건이며, 역전파에만 빠른 시그모이드의 대리 미분을 쓴다.
        return grad_output / (1 + ctx.slope * gap.abs()).square(), None


def mixed_neuron_tick(v, signal, slow, refractory, current, alpha, threshold,
                      adaptation, spiking_mask, reset_voltage, refractory_ticks, surrogate_slope):
    candidate = v + alpha[0] * (-v + current - adaptation * slow)
    blocked = spiking_mask & (refractory > 0)
    candidate = torch.where(blocked, torch.full_like(candidate, reset_voltage), candidate)
    spikes = SurrogateSpike.apply(candidate - threshold, surrogate_slope)
    spikes = spikes * (spiking_mask & ~blocked).to(candidate.dtype)
    graded = torch.relu(torch.tanh(candidate - threshold))
    release = torch.where(spiking_mask, spikes, graded)
    # 리셋·불응기 사건의 이산 결정에는 별도 대리 미분을 적용하지 않는다.
    fired = spikes.detach() > 0
    v = torch.where(fired, torch.full_like(candidate, reset_voltage), candidate)
    refractory = torch.where(fired, torch.full_like(refractory, float(refractory_ticks)),
                             torch.clamp(refractory - 1, min=0))
    refractory = torch.where(spiking_mask, refractory, torch.zeros_like(refractory)).detach()
    signal = signal + alpha[1] * (release - signal)
    slow = slow + alpha[2] * (release - slow)
    return v, signal, slow, refractory, spikes


def validate_dynamics_config(cfg):
    if cfg['neuron_mode'] not in ('mixed', 'graded_control'):
        raise ValueError('neuron_mode은 mixed 또는 graded_control이어야 한다.')
    if cfg['unknown_activity'] != 'graded':
        raise ValueError('미확인 유형의 기본 동역학은 graded이다.')
    for key in ('spike_threshold_center', 'spike_threshold_radius', 'spike_reset', 'surrogate_slope'):
        if not math.isfinite(cfg[key]):
            raise ValueError(f'유한한 동역학 설정이 필요하다: {key}')
    if (cfg['spike_threshold_radius'] < 0 or cfg['surrogate_slope'] <= 0
            or cfg['spike_threshold_center'] - cfg['spike_threshold_radius'] <= max(0, cfg['spike_reset'])):
        raise ValueError('스파이크 역치는 항상 0과 리셋 전압보다 높아야 한다.')
    if (type(cfg['refractory_ticks']) is not int or cfg['refractory_ticks'] < 0
            or type(cfg['brain_steps']) is not int or cfg['brain_steps'] < 1):
        raise ValueError('불응기는 0 이상의 정수, 토큰당 tick 수는 양의 정수여야 한다.')


class ConnectomeLM(nn.Module):
    def __init__(self, cfg, rowptr, src, counts, edge_sign, inputs, outputs, type_ids, dynamics_ids, device):
        super().__init__()
        validate_dynamics_config(cfg)
        self.tbptt = cfg['tbptt']
        self.vocab_size = cfg['vocab_size']
        self.bos_id, self.eos_id = cfg['bos_id'], cfg['eos_id']
        self.pad_id, self.unk_id = cfg['pad_id'], cfg['unk_id']
        self.tokenizer = None
        self.reset_voltage = cfg['spike_reset']
        self.refractory_ticks = cfg['refractory_ticks']
        self.surrogate_slope = cfg['surrogate_slope']
        self.spike_threshold_center = cfg['spike_threshold_center']
        self.spike_threshold_radius = cfg['spike_threshold_radius']
        codes = np.asarray(dynamics_ids)
        if (codes.shape != (cfg['n_neurons'],) or codes.dtype != np.uint8
                or not np.isin(codes, [0, 1, 2]).all()):
            raise ValueError('뉴런 순서에 맞는 uint8 활동 코드 0/1/2가 필요하다.')
        mask = (codes == 2) & (cfg['neuron_mode'] == 'mixed')
        self.register_buffer('spiking_mask', torch.tensor(mask, dtype=torch.bool, device=device), persistent=False)
        self.n, self.k = cfg['n_neurons'], cfg['brain_steps']
        self.block, self.checkpoint_tokens = cfg['edge_block'], cfg['checkpoint_tokens']
        for name, value in [('rowptr', rowptr), ('src', src), ('inputs', inputs),
                            ('outputs', outputs), ('type_ids', type_ids)]:
            self.register_buffer(name, torch.tensor(np.asarray(value), dtype=torch.long,
                                                   device=device), persistent=False)
        assert len(type_ids) == self.n and min(type_ids) >= 0
        self.register_buffer('dst', torch.repeat_interleave(
            torch.arange(self.n, device=device), self.rowptr.diff()), persistent=False)
        sign = torch.tensor(np.asarray(edge_sign), dtype=torch.float32, device=device)
        self.register_buffer('edge_sign', sign, persistent=False)
        c = torch.tensor(np.asarray(counts), dtype=torch.float32, device=device).log1p()
        incoming = torch.zeros(self.n, device=device).index_add_(0, self.dst, c)
        self.register_buffer('prior', c / incoming[self.dst].clamp_min(1e-8), persistent=False)
        raw = torch.where(sign != 0, torch.full_like(sign, math.log(0.2 / 0.8)),
                          torch.where(torch.rand_like(sign) < 0.5, -0.2554128, 0.2554128))
        self.theta = nn.Parameter(raw)
        ntypes = int(max(type_ids)) + 1
        low, high, initial = [torch.tensor(cfg[key], device=device)[:, None]
                              for key in ['tau_min', 'tau_max', 'tau_initial']]
        assert torch.all((low > 0) & (initial > low) & (initial < high))
        self.register_buffer('tau_low', low)
        self.register_buffer('tau_high', high)
        fraction = (initial - low) / (high - low)
        self.raw_tau = nn.Parameter(torch.logit(fraction).expand(3, ntypes).clone())
        self.threshold = nn.Parameter(torch.zeros(ntypes, device=device))
        self.raw_adaptation = nn.Parameter(torch.full((ntypes,), math.log(0.1 / 0.9), device=device))
        d = cfg['embedding_dim']
        self.embedding = nn.Embedding(self.vocab_size, d, device=device)
        self.input_adapter = nn.Linear(d, len(inputs), bias=False, device=device)
        self.motor_norm = nn.LayerNorm(len(outputs), eps=1e-6, elementwise_affine=False, device=device)
        self.output_adapter = nn.Linear(len(outputs), d, bias=False, device=device)

    def effective_weights(self):
        magnitude = 0.25 + 3.75 * self.theta.sigmoid()
        signed = torch.where(self.edge_sign != 0, self.edge_sign * magnitude,
                             4 * self.theta.tanh())
        return self.prior * signed

    def dynamics(self):
        tau = self.tau_low + (self.tau_high - self.tau_low) * self.raw_tau.sigmoid()
        alpha = -torch.expm1(-1 / tau[:, self.type_ids])
        offset = self.threshold.tanh()[self.type_ids]
        threshold = torch.where(self.spiking_mask,
            self.spike_threshold_center + self.spike_threshold_radius * offset, 0.5 * offset)
        adaptation = self.raw_adaptation.sigmoid()[self.type_ids]
        return alpha, threshold, adaptation

    def initial_state(self, batch):
        return self.embedding.weight.new_zeros(4, batch, self.n)

    def advance(self, token, state, weights, alpha, threshold, adaptation, collect_spikes=False):
        v, signal, slow, refractory = state.unbind(0)
        value = self.input_adapter(self.embedding(token))
        drive = torch.zeros_like(v).index_add(1, self.inputs, value)
        counts = torch.zeros_like(v) if collect_spikes else None
        for _ in range(self.k):
            message = SparseEdges.apply(signal, weights, self.rowptr, self.src, self.dst, self.block)
            v, signal, slow, refractory, spikes = mixed_neuron_tick(
                v, signal, slow, refractory, message + drive, alpha, threshold, adaptation,
                self.spiking_mask, self.reset_voltage, self.refractory_ticks, self.surrogate_slope)
            if collect_spikes:
                counts = counts + spikes.detach()
        result = torch.stack((v, signal, slow, refractory))
        return (result, counts) if collect_spikes else result

    def forward(self, tokens, state=None, active_mask=None):
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError('Nonempty tokens [B,T] required')
        state = self.initial_state(tokens.shape[0]) if state is None else state
        if state.shape != (4, tokens.shape[0], self.n):
            raise ValueError('상태는 [4,B,N]: 막전위, 시냅스 신호, 적응, 남은 불응기여야 한다.')
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
        return torch.stack(logits, dim=1), state

import codecs


def inference_inputs(m, question):
    question = ' '.join(question.split())
    if not question:
        raise ValueError('비어 있지 않은 질문이 필요하다.')
    return [m.bos_id] + encode_text(m.tokenizer, render_prompt(question))


@torch.no_grad()
def prefill_question(m, question):
    prefix = inference_inputs(m, question)
    state = None
    for start in range(0, len(prefix), m.tbptt):
        ids = torch.tensor([prefix[start:start + m.tbptt]], device=m.embedding.weight.device)
        logits, state = m(ids, state)
    return logits, state


def valid_utf8_bytes(pending):
    if not pending:
        return list(range(0x80)) + list(range(0xC2, 0xF5))
    if len(pending) == 1:
        bounds = {0xE0: (0xA0, 0xC0), 0xED: (0x80, 0xA0),
                  0xF0: (0x90, 0xC0), 0xF4: (0x80, 0x90)}
        if pending[0] in bounds:
            return list(range(*bounds[pending[0]]))
    return list(range(0x80, 0xC0))


def generation_tables(tokenizer):
    regular, byte_ids, byte_values = [], {}, {}
    for index in range(tokenizer.get_piece_size()):
        if tokenizer.is_byte(index):
            value = int(tokenizer.id_to_piece(index)[3:5], 16)
            byte_ids[value] = index
            byte_values[index] = value
        elif not (tokenizer.is_control(index) or tokenizer.is_unknown(index) or tokenizer.is_unused(index)):
            regular.append(index)
    return regular, byte_ids, byte_values


def allowed_piece_ids(m, decoder, tables):
    regular, byte_ids, _ = tables
    pending = decoder.getstate()[0]
    # fallback 바이트로 시작한 문자가 완성되기 전에는 다른 subword나 EOS를 허용하지 않는다.
    allowed = [byte_ids[value] for value in valid_utf8_bytes(pending)]
    if not pending:
        allowed.extend(regular)
        allowed.append(m.eos_id)
    return allowed


@torch.no_grad()
def reply(m, question, max_new_tokens=128, temperature=0.8):
    if type(max_new_tokens) is not int or max_new_tokens < 1 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError('max_new_tokens는 양의 정수, temperature는 유한한 0 이상의 값이어야 한다.')
    was_training = m.training
    m.eval()
    decoder = codecs.getincrementaldecoder('utf-8')('strict')
    generated = []
    tables = generation_tables(m.tokenizer)
    try:
        logits, state = prefill_question(m, question)
        for _ in range(max_new_tokens):
            allowed = torch.tensor(allowed_piece_ids(m, decoder, tables), device=logits.device)
            scores = logits[0, -1].index_select(0, allowed)
            if not torch.isfinite(scores).all():
                raise RuntimeError('생성 점수에 유한하지 않은 값이 있다.')
            selected = scores.argmax() if temperature == 0 else torch.multinomial(
                (scores / temperature).softmax(-1), 1)[0]
            token = int(allowed[selected])
            if token == m.eos_id:
                break
            generated.append(token)
            if token in tables[2]:
                decoder.decode(bytes([tables[2][token]]), final=False)
            logits, state = m(torch.tensor([[token]], device=logits.device), state)
        # 출력 한도에서 끝나지 않은 fallback 문자는 해당 바이트 토큰만 생략한다.
        pending = decoder.getstate()[0]
        if pending:
            generated = generated[:-len(pending)]
        return m.tokenizer.decode(generated)
    finally:
        m.train(was_training)

def load_model(checkpoint_path, device=None):
    checkpoint_path = Path(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if saved.get('architecture') != 'malecns_mixed_sentencepiece_v2':
        raise ValueError('SentencePiece용 flylm-2 체크포인트가 필요하다. 바이트 모델은 직접 호환되지 않는다.')
    if saved.get('graph_directory') != 'graph':
        raise ValueError('이 버전은 체크포인트 옆의 graph 폴더만 사용한다.')
    tokenizer_bytes = (checkpoint_path.parent/'tokenizer.model').read_bytes()
    if hashlib.sha256(tokenizer_bytes).hexdigest() != saved['tokenizer_sha256']:
        raise ValueError('체크포인트와 토크나이저 모델의 해시가 다르다.')
    tokenizer = spm.SentencePieceProcessor(model_proto=tokenizer_bytes)
    validate_tokenizer(tokenizer, saved['config'])
    folder = checkpoint_path.parent / 'graph' 
    evidence_bytes = (folder/'dynamics_evidence.json').read_bytes()
    if hashlib.sha256(evidence_bytes).hexdigest() != saved['dynamics_sha256']:
        raise ValueError('체크포인트와 신경 활동 근거집의 해시가 다르다.')
    evidence = json.loads(evidence_bytes)
    if (evidence['mode'] != saved['config']['neuron_mode']
            or evidence['unknown_activity'] != saved['config']['unknown_activity']):
        raise ValueError('체크포인트와 근거집의 동역학 모드가 다르다.')
    codes = np.load(folder/'dynamics_ids.npy', allow_pickle=False)
    body_ids = np.load(folder/'body_ids.npy', allow_pickle=False)
    if hashlib.sha256(codes.tobytes()).hexdigest() != saved['dynamics_ids_sha256']:
        raise ValueError('체크포인트와 신경 활동 배정이 다르다.')
    if hashlib.sha256(body_ids.tobytes()).hexdigest() != saved['body_ids_sha256']:
        raise ValueError('체크포인트와 뉴런 순서가 다르다.')
    info = json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    count = info['nnz']
    if len(body_ids) != saved['config']['n_neurons']:
        raise ValueError('체크포인트와 그래프의 뉴런 수가 다르다.')
    target = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
    restored = ConnectomeLM(saved['config'], np.load(folder/'crow.npy', mmap_mode='r'),
        np.load(folder/'src.npy', mmap_mode='r')[:count],
        np.load(folder/'count.npy', mmap_mode='r')[:count],
        np.load(folder/'edge_sign.npy', mmap_mode='r'),
        np.load(folder/'inputs.npy'), np.load(folder/'outputs.npy'),
        np.load(folder/'type_ids.npy'), codes, target)
    restored.tokenizer = tokenizer
    restored.load_state_dict(saved['model_state'], strict=True)
    return restored.eval()

