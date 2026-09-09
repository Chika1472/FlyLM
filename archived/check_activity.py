"""Regression checks for observing inference without changing it. No trained model required."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import torch

import flylm_activity as viz
import flylm_inference as runtime


def check(device='cpu'):
    torch.set_num_threads(1)
    torch.manual_seed(10)
    cfg = dict(n_neurons=6, brain_steps=4, edge_block=3, checkpoint_tokens=False,
               embedding_dim=8, tau_min=(0.5, 0.5, 8.0), tau_max=(32.0, 16.0, 512.0),
               tau_initial=(2.0, 2.0, 64.0))
    m = runtime.ConnectomeLM(cfg, np.arange(7), np.array([5, 0, 1, 2, 3, 4]),
                             np.ones(6), np.ones(6), np.array([0, 1]), np.array([4, 5]),
                             np.array([0, 0, 1, 1, 2, 2]), 'cpu')
    m.inference_tbptt = 64
    m.to(device)
    m.eval()
    weights = {k: v.clone() for k, v in m.state_dict().items()}
    question = '가나다'
    expected = runtime.reply(m, question, temperature=0, max_new_bytes=12)
    trace = viz.record_activity(m, runtime, question, temperature=0, max_new_bytes=12, max_frames=256)
    assert trace['answer'] == expected
    assert trace['activity'].shape[1] == 6 and trace['activity'].dtype == np.uint8
    assert 'advance' not in m.__dict__ and not m.training
    assert np.any(trace['activity'] > 0)
    for name, value in m.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    # Every recorded input frame must equal an independent actual model forward.
    state = None
    with torch.no_grad():
        for i, token in enumerate(runtime.inference_inputs(question), start=1):
            _, state = m(torch.tensor([[token]], device=device), state)
            quantized = state[1, 0].clamp(0, 1).sqrt().mul(255).round().to(torch.uint8).cpu().numpy()
            np.testing.assert_array_equal(quantized, trace['activity'][i])
    for temperature in (0.0, 0.8):
        torch.manual_seed(17)
        baseline = runtime.reply(m, question, temperature=temperature, max_new_bytes=12)
        torch.manual_seed(17)
        sparse_trace = viz.record_activity(m, runtime, question, temperature=temperature,
                                           max_new_bytes=12, max_frames=5)
        assert sparse_trace['answer'] == baseline
        assert len(sparse_trace['captions']) <= 5
        assert sparse_trace['captions'][-1]['phase'] == 'done'
    def broken_reply(*args, **kwargs):
        raise RuntimeError('forced test error')
    try:
        viz.record_activity(m, SimpleNamespace(reply=broken_reply, inference_inputs=runtime.inference_inputs), question)
    except RuntimeError as error:
        assert str(error) == 'forced test error'
    else:
        raise AssertionError('Expected test error')
    assert 'advance' not in m.__dict__
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        np.save(folder / 'body_ids.npy', np.arange(100, 106))
        annotations = folder / 'annotations.feather'
        feather.write_feather(pa.table({'bodyId': [105, 103, 101, 104, 102, 100],
            'somaLocation': [[3, 4, 5], [6, 7, 8], None, [20, 25, 35], [10, 11, 12], None]}), annotations)
        layout = viz.neuron_layout(m, folder, folder, annotation_path=annotations)
        assert layout['known'].tolist() == [False, False, True, True, True, True]
        assert layout['roles'].tolist() == [1, 1, 0, 0, 2, 2]
        assert layout['xy'].shape == (6, 2) and np.isfinite(layout['xy']).all()
        assert (layout['xy'][~layout['known'], 0] >= 0.76).all()
        assert (layout['xy'][layout['known'], 0] < 0.36).all()
        assert (layout['xy_front'][~layout['known']] == -2).all()
        assert (layout['xy_front'][layout['known'], 0] > 0.36).all()
        # Both views retain the same body-ID order; their vertical coordinates use Z and Y.
        assert layout['xy'][5, 0] == layout['xy'][layout['known'], 0].min()
        assert layout['xy_front'][5, 0] == layout['xy_front'][layout['known'], 0].min()
        assert not np.allclose(layout['xy'][layout['known'], 1], layout['xy_front'][layout['known'], 1])
        # Resolved coordinates can be in a different file order, and must include the right units.
        xyz = layout['xyz'].copy()
        xyz[:2] = [[8, 8, 8], [10, 12, 14]]
        source = np.array([2, 2, 1, 1, 1, 1], dtype=np.uint8)
        order = np.array([4, 1, 5, 0, 3, 2])
        coordinates = folder/'resolved.npz'
        np.savez_compressed(coordinates, body_ids=np.arange(100, 106)[order], xyz=xyz[order],
                            source=source[order], metadata=np.array(json.dumps(
                                dict(dataset='male-cns:v1.0', units='8nm_voxels'))))
        resolved = viz.neuron_layout(m, folder, folder, coordinate_path=coordinates)
        np.testing.assert_array_equal(resolved['xyz'], xyz)
        np.testing.assert_array_equal(resolved['coordinate_kind'], source)
        assert resolved['known'].all()
        assert (resolved['xy'][:, 0] < 0.5).all()
        assert (resolved['xy_front'][:, 0] > 0.5).all()
        np.savez_compressed(coordinates, body_ids=np.arange(100, 106), xyz=xyz, source=source,
                            metadata=np.array(json.dumps(dict(dataset='male-cns:v1.0', units='1nm'))))
        try:
            viz.neuron_layout(m, folder, folder, coordinate_path=coordinates)
        except ValueError as error:
            assert 'units' in str(error)
        else:
            raise AssertionError('Mixed coordinate units were accepted')
        attack = '</script><script>alert("test")</script>'
        trace['captions'][-1]['output'] = attack
        rendered = viz.activity_html(trace, layout)
        assert attack not in rendered
        payload = rendered.split('<script id="payload" type="application/json">')[1].split('</script>')[0]
        data = json.loads(payload)
        assert data['captions'][-1]['output'] == attack
        (folder/'activity.html').write_text(rendered, encoding='utf-8')
        import base64, gzip
        unpacked = np.frombuffer(gzip.decompress(base64.b64decode(data['activity'])), dtype=np.uint8)
        np.testing.assert_array_equal(unpacked.reshape(-1, 6), trace['activity'])
    print('PASS:', device, 'exact greedy/sampling parity, real recorded states, frame cap, restoration, ID-based layout, safe HTML')


if __name__ == '__main__':
    from build_neuron_coordinates import skeleton_representative
    swc = b'# fixture\n1 0 0 0 0 1 -1\n2 0 10 0 0 1 1\n3 0 -10 0 0 1 1\n4 0 10 10 0 1 2\n5 0 10 -10 0 1 2\n6 0 -10 10 0 1 3\n7 0 -10 -10 0 1 3\n'
    point, node, count = skeleton_representative(swc)
    np.testing.assert_array_equal(point, [0, 0, 0])
    assert node == 1 and count == 7
    try:
        skeleton_representative(b'1 0 0 0 0 1 -1\n1 0 10 0 0 1 1\n')
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate SWC IDs accepted')
    notebook = json.loads(Path(__file__).with_name('flylm_inference.ipynb').read_text(encoding='utf-8'))
    packed_cell = next(''.join(c['source']) for c in notebook['cells']
                       if ''.join(c['source']).startswith('# Official MaleCNS v1.0 coordinate snapshot'))
    namespace = {}
    exec(compile(packed_cell, '<bundled coordinate cell>', 'exec'), namespace)
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        coordinates = namespace['bundled_coordinates'](folder)
        assert coordinates.read_bytes() == Path(__file__).with_name('flylm_neuron_coordinates.npz').read_bytes()
        with np.load(coordinates, allow_pickle=False) as saved:
            assert len(saved['body_ids']) == 165122
            assert np.isfinite(saved['xyz']).all() and (saved['source'] > 0).all()
            np.save(folder/'body_ids.npy', saved['body_ids'])
        full = SimpleNamespace(n=165122, inputs=torch.tensor([0, 1]), outputs=torch.tensor([2]))
        resolved = viz.neuron_layout(full, folder, folder, coordinate_path=coordinates)
        assert resolved['known'].all() and resolved['xy_front'].shape == (165122, 2)
        assert np.bincount(resolved['coordinate_kind']).tolist() == [0, 140024, 25098]
    print('PASS: embedded coordinate file SHA-256, all 165122 body IDs, offline two-view layout')
    check()
    if torch.cuda.is_available():
        check('cuda')
