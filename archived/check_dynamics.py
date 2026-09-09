"""Run the notebook's actual model, training loop and checkpoint path on a tiny graph.

python -X utf8 output/jupyter-notebook/check_dynamics.py
This checks implementation correctness, not full MaleCNS language performance.
"""
import ast
import gc
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# Silence only progress rendering; execute the notebook's numerical/training code unchanged.
def tqdm(iterable=None, **kwargs):
    return iterable if iterable is not None else nullcontext(SimpleNamespace(update=lambda count: None))


tqdm.write = print


def check_bootstrap(source):
    if '# End Pascal bootstrap.' in source:
        bootstrap = source[:source.index('# End Pascal bootstrap.')]
        good = dict(version='compatible', capability=[6, 0], architectures=['sm_60'])
        bad = dict(version='incompatible', capability=[6, 0], architectures=['sm_70'])
        other = dict(version='other', capability=[8, 9], architectures=['sm_89'])
        for disk, loaded, must_install, must_restart in [
            (bad, None, True, False), (bad, bad, True, True),
            (good, bad, False, True), (good, good, False, False),
            (good, None, False, False), (other, other, False, False),
        ]:
            module = None if loaded is None else SimpleNamespace(
                cuda=SimpleNamespace(get_arch_list=lambda: loaded['architectures']))
            replies = [SimpleNamespace(returncode=0, stdout=json.dumps(disk), stderr=''),
                       SimpleNamespace(returncode=0, stdout=json.dumps(good), stderr='')]
            with patch('subprocess.run', side_effect=replies), \
                 patch('subprocess.check_call') as install, patch.dict(sys.modules, {'torch': module}):
                restarted = False
                try:
                    exec(compile(bootstrap, '<Pascal bootstrap>', 'exec'), {})
                except RuntimeError as error:
                    assert '커널' in str(error)
                    restarted = True
                assert install.called == must_install
                assert restarted == must_restart
        # A failed repair must stop, not request endless restarts.
        reply = SimpleNamespace(returncode=0, stdout=json.dumps(bad), stderr='')
        with patch('subprocess.run', return_value=reply), patch('subprocess.check_call'):
            try:
                exec(compile(bootstrap, '<failed repair>', 'exec'), {})
            except RuntimeError as error:
                assert 'sm_60' in str(error)
            else:
                raise AssertionError('Unsupported repaired build was accepted')
        print('PASS: Pascal repair, fresh/stale kernels, compatible rerun, failed repair')
    if '# Validate current CUDA runtime by execution' in source:
        assert 'SystemExit' not in source
        if not torch.cuda.is_available():
            print('SKIP actual runtime probe: CUDA unavailable')
            return
        probe = source[source.index('# Validate current CUDA runtime by execution'):
                       source.index('# End CUDA runtime probe.')]
        with patch('subprocess.check_call') as install:
            exec(compile(probe, '<actual CUDA probe>', 'exec'), dict(torch=torch, sys=sys))
            # The gate must accept a different version label if actual kernels work.
            with patch.object(torch, '__version__', '2.6.0+kaggle-test'):
                exec(compile(probe, '<different version label>', 'exec'), dict(torch=torch, sys=sys))
            assert not install.called
        print('PASS: actual CUDA compatibility probe; no exact-version gate or forced reinstall')
        return
    if "loaded_torch = sys.modules.get('torch')" not in source:
        return
    bootstrap = source[:source.index('import importlib.util')]
    target = '2.10.0+cu126'
    for installed, loaded, must_install, must_restart in [
        ('old', None, True, False),
        (target, target, False, False),
        ('old', 'old', True, True),
        (target, 'old', False, True),
    ]:
        module = None if loaded is None else SimpleNamespace(__version__=loaded)
        with patch('importlib.metadata.version', return_value=installed), \
             patch('subprocess.check_call') as install, patch.dict(sys.modules, {'torch': module}):
            restarted = False
            try:
                exec(compile(bootstrap, '<bootstrap>', 'exec'), {})
            except SystemExit:
                restarted = True
            assert install.called == must_install
            assert restarted == must_restart
    print('PASS: fresh install continues; only incompatible loaded torch requires kernel restart')


def main():
    torch.set_num_threads(1)
    path = (Path(sys.argv[1]) if len(sys.argv) > 1 else
            Path(__file__).with_name('flylm_malecns_dynamics_t4.ipynb'))
    notebook = json.loads(path.read_text(encoding='utf-8'))
    sources = [''.join(cell['source']) for cell in notebook['cells']]
    check_bootstrap(sources[2])
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(sources[i], f'{path.name}:cell{i}', 'exec')

    def definitions(index):
        tree = ast.parse(sources[index])
        tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        return compile(tree, f'{path.name}:cell{index}', 'exec')

    env = dict(globals(), BOS=256, EOS=257, VOCAB=258, IGNORE_INDEX=-100)
    config = next(node for node in ast.parse(sources[2]).body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CFG'
                                                         for t in node.targets))
    exec(compile(ast.Module(body=[config], type_ignores=[]), '<config>', 'exec'), env)
    cfg = env['CFG']
    cfg.update(n_neurons=6, embedding_dim=8, edge_block=3,
               train_updates=3, epochs=3, log_every=1, eval_every=1, save_every=1)
    batched = 'def batched_chunks' in sources[18]
    if batched:
        cfg.update(batch_size=2, max_batch_size=4)
    for index in [6, 12, 14, 16, 18, 19, 21]:
        exec(definitions(index), env)
    frame = pd.DataFrame(dict(type=['A', 'A', None, 'tbc'],
                              **{'class': ['x', 'x', 'y', None]},
                              superclass=['i', 'i', 'i', 'sensory']))
    codes, names = env['dynamics_types'](frame)
    assert [names[i] for i in codes] == ['type:A', 'type:A', 'class:y', 'superclass:sensory']
    if batched:
        # Verify exact search boundaries, including no fitting batch and a data-size cap.
        for limit in [1, 2, 7, 32]:
            for capacity in range(limit + 1):
                assert env['largest_fitting_batch'](limit, lambda b: b <= capacity) == capacity
    for backend in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        env['device'] = torch.device(backend)
        torch.manual_seed(42)
        env['numerical_checks']()
        print('Numerical checks:', backend)

    # Full application loop uses CUDA timing; fail explicitly if unavailable.
    if not torch.cuda.is_available():
        print('SKIP CUDA 64-token backward and training loop: CUDA unavailable')
        return
    import codecs
    env['codecs'] = codecs
    env.update(crow=np.arange(7, dtype=np.int64), src=np.array([5, 0, 1, 2, 3, 4]),
               counts=np.ones(6, dtype=np.int64), edge_sign=np.ones(6, dtype=np.float32),
               input_nodes=np.array([0, 1]), output_nodes=np.array([2, 3, 4]),
               type_ids=np.array([0, 0, 1, 1, 2, 2]), DATA_CFG={'text_field': 'form'})
    env['check_response_format']()
    # Every target occurs once, even when a long prompt crosses multiple TBPTT boundaries.
    for prompt in [2, 31, 32, 33, 64, 65, 100, 300]:
        length = prompt + 90
        spans = list(env['chunk_spans'](length, prompt))
        seen = [k for start, count in spans for k in range(start+1, start+count+1)]
        assert seen == list(range(1, length))
        first = next((start, count) for start, count in spans if start+count >= prompt)
        assert first[0] < prompt-1
        assert all(1 <= count <= 64 for _, count in spans)

    m = env['make_model']()
    env['full_backward_check'](m)
    if batched:
        # Batched forward and gradients must match independent samples, including padding.
        m.checkpoint_tokens = False
        x = torch.tensor([[256, 40, 60, 80], [256, 10, 0, 0]], device='cuda')
        active = torch.tensor([[True]*4, [True, True, False, False]], device='cuda')
        targets = torch.tensor([[40, 60, 80, 257], [10, 257, -100, -100]], device='cuda')
        m.zero_grad(set_to_none=True)
        batch_logits, batch_state = m(x, active_mask=active)
        F.cross_entropy(batch_logits.flatten(0, 1), targets.flatten(), ignore_index=-100).backward()
        batched_grads = [p.grad.clone() for p in m.parameters()]
        m.zero_grad(set_to_none=True)
        for lane, width in enumerate([4, 2]):
            independent, independent_state = m(x[lane:lane+1, :width])
            torch.testing.assert_close(batch_logits[lane:lane+1, :width], independent)
            torch.testing.assert_close(batch_state[:, lane:lane+1], independent_state)
            (F.cross_entropy(independent.flatten(0, 1), targets[lane, :width], reduction='sum')/6).backward()
        for expected, parameter in zip(batched_grads, m.parameters()):
            torch.testing.assert_close(expected, parameter.grad, rtol=1e-4, atol=1e-6)
        print('PASS: batched gradients equal individual samples; padding freezes each state')
    # Loss on continuation must backpropagate into the prefix state inside a chunk.
    prefix = torch.tensor([[256, 10, 20, 30]], device='cuda')
    _, state = m(prefix)
    state.retain_grad()
    logits, _ = m(torch.tensor([[40, 50]], device='cuda'), state)
    logits.square().mean().backward()
    assert state.grad is not None and state.grad.abs().sum() > 0
    del m, state, logits
    gc.collect()
    torch.cuda.empty_cache()

    with tempfile.TemporaryDirectory(prefix='flylm-dynamics-') as directory:
        folder = Path(directory)
        graph = folder / 'graph'
        graph.mkdir()
        for name, key in [('crow', 'crow'), ('src', 'src'), ('count', 'counts'),
                          ('edge_sign', 'edge_sign'), ('inputs', 'input_nodes'),
                          ('outputs', 'output_nodes'), ('type_ids', 'type_ids')]:
            np.save(graph / (name + '.npy'), env[key])
        (graph / 'complete.json').write_text(json.dumps({'nnz': 6}))
        if batched:
            # Run the new directory parser on two real JSON files, including a nested file.
            nested = folder/'json'/'nested'
            nested.mkdir(parents=True)
            json_files = [folder/'json'/'first.json', nested/'second.json']
            for i, filename in enumerate(json_files):
                filename.write_text(json.dumps({'document': [dict(id=f'doc{i}', utterance=[
                    dict(speaker_id='A', form=f'질문 {i}'), dict(speaker_id='B', form=f'답변 {i}')])]},
                    ensure_ascii=False), encoding='utf-8')
            env.update(JSON_FILES=json_files, DATA_ROOT=folder/'json',
                       token_path=folder/'pairs.uint16', index_path=folder/'pairs.json', data_spec={})
            env['DATA_CFG']['validation_fraction'] = 0.5
            manifest = env['prepare_pairs']()
            assert len(manifest['pairs']) == 2 and len(manifest['documents']) == 2
            assert {p['split'] for p in manifest['pairs']} == {'train', 'validation'}
            assert manifest['documents'][1]['member'] == 'nested/second.json'
            print('PASS: direct JSON directory parser and document split')
        rows, values = [], []
        examples = [('가' * 40, '안녕하세요'), ('어디로 가요?', '집으로 가요')]
        if batched:
            examples = [('가' * 40, '안녕하세요'), ('나', '좋아요' * 16),
                        ('짧은 질문', '네'), ('어디로 가요?', '집으로 가요')]
        for question, answer in examples:
            tokens, boundary = env['encode_pair'](question, answer)
            rows.append(dict(length=len(tokens), prompt_tokens=boundary, offset=len(values)))
            values.extend(tokens.tolist())
        validation_index = len(rows) - 1
        # Match production uint16 memmap: int64 fixtures hid unsigned -100 wrapping.
        fixture_tokens = folder / 'fixture.uint16'
        np.asarray(values, dtype='<u2').tofile(fixture_tokens)
        fixture_mm = np.memmap(fixture_tokens, mode='r+', dtype='<u2')
        env.update(RUN_DIR=folder, CACHE=folder, pairs=rows, tokens_mm=fixture_mm,
                   train_ids=list(range(validation_index)), val_ids=[validation_index],
                   eval_ids=[validation_index], data_spec={}, extract_spec={}, source_manifest={},
                   data_stats={'validation_response_tokens': rows[-1]['length']-rows[-1]['prompt_tokens']})
        if batched:
            found_x, found_y = [[], []], [[], []]
            for x, y, active, counts in env['batched_chunks']([0, 1]):
                assert x.shape == y.shape == active.shape and x.shape[0] == 2 and x.shape[1] <= 64
                assert y.dtype == torch.int64
                assert bool(((y == -100) | ((y >= 0) & (y < 258))).all())
                for lane in range(2):
                    found_x[lane].extend(x[lane][active[lane]].tolist())
                    found_y[lane].extend(y[lane][y[lane] != -100].tolist())
                    assert int((y[lane] != -100).sum()) == counts[lane]
            for lane in range(2):
                row = rows[lane]
                original = values[row['offset']:row['offset']+row['length']]
                assert found_x[lane] == original[:-1]
                assert found_y[lane] == original[row['prompt_tokens']:]
            fixture_mm[0] = 65436
            try:
                next(env['batched_chunks']([0]))
            except ValueError as error:
                assert 'Invalid token IDs' in str(error)
            else:
                raise AssertionError('Invalid IDs reached CUDA')
            finally:
                fixture_mm[0] = values[0]
            print('PASS: real uint16 memmap masks and CPU rejection of invalid IDs')
            batches = env['epoch_batches'](0)
            assert sorted(i for b in batches for i in b) == env['train_ids']
            assert max(map(len, batches)) == 2 and min(map(len, batches)) == 1
            # Exercise the actual CUDA tuner, but cap it at three small-graph samples.
            cfg['batch_size'] = 'auto'
            assert env['tune_batch_size']() == 3
            # Test OOM recovery without deliberately exhausting the user's GPU.
            original_probe = env['probe_batch']
            def simulated_oom_probe(model, optimizer, batch):
                if batch > 2:
                    raise torch.cuda.OutOfMemoryError('simulated test OOM')
                return dict(batch=batch, peak_reserved_GiB=0., simulated=True)
            env['probe_batch'] = simulated_oom_probe
            try:
                assert env['tune_batch_size']() == 2
            finally:
                env['probe_batch'] = original_probe
            cfg['batch_size'] = 2
        exec(compile(sources[19], '<actual training cell>', 'exec'), env)
        assert env['updates'] == 3
        for name in ['best.pt', 'latest.pt', 'history.json']:
            assert (folder / name).is_file()
        trained = env['model']
        restored = env['load_full_checkpoint'](folder / 'latest.pt')
        x = torch.tensor([[256, 65, 66]], device='cuda')
        with torch.no_grad():
            a, sa = trained(x)
            b, sb = restored(x)
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(sa, sb)
        for temperature in (0, 0.8):
            result = env['reply'](restored, '안녕하세요?', max_new_bytes=8, temperature=temperature)
            assert '\ufffd' not in result
            result.encode('utf-8', errors='strict')
        decoder = codecs.getincrementaldecoder('utf-8')('strict')
        assert 257 in env['valid_utf8_tokens'](decoder)
        decoder.decode(b'\xe3', final=False)
        valid = env['valid_utf8_tokens'](decoder)
        assert 257 not in valid and all(0x80 <= v <= 0xbf for v in valid)
        export_source = next((s for s in sources if s.startswith('INFERENCE_SOURCE =')), None)
        if export_source is not None:
            exec(compile(export_source, '<inference export cell>', 'exec'), env)
            import zipfile
            with zipfile.ZipFile(env['INFERENCE_ZIP']) as archive:
                assert {'flylm_inference/model.pt', 'flylm_inference/graph/crow.npy',
                        'flylm_inference/flylm_inference.py'} <= set(archive.namelist())
                archive.extractall(folder/'relocated')  # Only our own test-generated ZIP.
            # Run only the independent notebook loader in a fresh namespace.
            inference_notebook = json.loads(path.with_name('flylm_inference.ipynb').read_text(encoding='utf-8'))
            loader = ''.join(inference_notebook['cells'][2]['source'])
            loader = loader.replace("'/kaggle/input/your-model-dataset/flylm_inference'",
                                    repr(str(folder/'relocated'/'flylm_inference'))).replace('DEVICE = None', "DEVICE = 'cpu'")
            fresh = {}
            exec(compile(loader, '<standalone inference notebook>', 'exec'), fresh)
            best = env['load_full_checkpoint'](folder / 'best.pt')
            with torch.no_grad():
                expected, _ = best(x)
                actual, _ = fresh['model'](x.cpu())
            torch.testing.assert_close(actual, expected.cpu(), rtol=1e-4, atol=1e-5)
            result = subprocess.run([sys.executable, '-X', 'utf8',
                str(folder/'relocated'/'flylm_inference'/'flylm_inference.py'), '--device', 'cpu',
                '--question', '안녕하세요?', '--max-new-bytes', '8'],
                capture_output=True, text=True, encoding='utf-8', timeout=60)
            assert result.returncode == 0, result.stderr
            print('PASS: exported ZIP, independent CPU loader parity, fresh-process inference without training data')
        fixture_mm._mmap.close()
    print('PASS: TBPTT boundaries, 64-token CUDA backward/Adam, training/eval/save/load, UTF-8 generation')


if __name__ == '__main__':
    main()
