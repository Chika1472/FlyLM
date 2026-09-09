"""Resolve MaleCNS v1.0 soma gaps from public SWC geometry, without retaining large skeleton files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import threading
import time

import numpy as np
import pyarrow.feather as feather
import requests

SWC_BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/'
SOURCE_NAMES = ['unresolved', 'soma', 'skeleton_branch_representative', 'tosoma_annotation']
_thread = threading.local()


def skeleton_representative(content):
    # SWC columns: node ID, type, X, Y, Z, radius, parent ID. Official units are 8 nm.
    rows = np.loadtxt(io.BytesIO(content), comments='#', usecols=(0, 2, 3, 4, 6), ndmin=2)
    if len(rows) == 0 or not np.isfinite(rows).all():
        raise ValueError('Empty or nonfinite SWC')
    ids = rows[:, 0].astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate SWC node IDs')
    parents, counts = np.unique(rows[:, 4].astype(np.int64), return_counts=True)
    branch = np.isin(ids, parents[(parents >= 0) & (counts >= 2)])
    candidates = np.flatnonzero(branch) if branch.sum() >= 3 else np.arange(len(rows))
    points = rows[candidates, 1:4]
    center = np.median(points, axis=0)
    selected = candidates[np.argmin(np.square(points - center).sum(axis=1))]
    # Return a real skeleton vertex, not an average point that might lie outside the neuron.
    return rows[selected, 1:4].astype(np.float32), int(ids[selected]), len(rows)


def fetch_representative(body):
    if not hasattr(_thread, 'session'):
        _thread.session = requests.Session()
    last_error = ''
    for attempt in range(3):
        try:
            response = _thread.session.get(f'{SWC_BASE}{int(body)}.swc', timeout=(15, 45))
            if response.status_code == 404:
                return None, 0, 0, 0, '404'
            response.raise_for_status()
            xyz, node, vertices = skeleton_representative(response.content)
            return xyz, node, vertices, len(response.content), ''
        except (requests.RequestException, ValueError) as error:
            last_error = str(error)[:220]
            if attempt < 2:
                time.sleep(attempt + 1)
    return None, 0, 0, 0, last_error


def build(annotation_path, output_path, workers=16):
    frame = feather.read_table(annotation_path).to_pandas()
    frame = frame[frame.status == 'Traced'].sort_values('bodyId')
    body_ids = frame.bodyId.to_numpy(np.int64)
    xyz = np.full((len(frame), 3), np.nan, np.float32)
    source = np.zeros(len(frame), np.uint8)
    skeleton_node = np.zeros(len(frame), np.int32)
    for i, point in enumerate(frame.somaLocation):
        if point is not None and len(point) == 3 and np.isfinite(point).all():
            xyz[i], source[i] = point, 1
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as saved:
            if not np.array_equal(saved['body_ids'], body_ids):
                raise ValueError('Existing cache uses a different body-ID order')
            xyz, source, skeleton_node = saved['xyz'].copy(), saved['source'].copy(), saved['skeleton_node'].copy()
    wanted = np.flatnonzero(source == 0)
    errors = {}
    started, completed, transferred = time.monotonic(), 0, 0

    def persist():
        metadata = dict(dataset='male-cns:v1.0', units='8nm_voxels',
                        swc_base=SWC_BASE, source_names=SOURCE_NAMES,
                        method='nearest actual branch vertex to coordinate-wise branch median; all vertices if fewer than 3 branches')
        temporary = output_path.with_suffix('.partial.npz')
        np.savez_compressed(temporary, body_ids=body_ids, xyz=xyz, source=source,
                            skeleton_node=skeleton_node, metadata=np.array(json.dumps(metadata)))
        temporary.replace(output_path)
        output_path.with_suffix('.json').write_text(json.dumps(dict(
            counts={name: int((source == i).sum()) for i, name in enumerate(SOURCE_NAMES)},
            requested=len(wanted), completed=completed, transferred_MiB=transferred/2**20,
            seconds=time.monotonic()-started, errors=errors), indent=2), encoding='utf-8')

    print(f'Need coordinates for {len(wanted):,} neurons; workers={workers}', flush=True)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(fetch_representative, body_ids[i]): i for i in wanted}
            for job in as_completed(jobs):
                i = jobs[job]
                point, node, vertices, size, error = job.result()
                completed += 1
                transferred += size
                if point is not None:
                    xyz[i], source[i], skeleton_node[i] = point, 2, node
                else:
                    marker = frame.iloc[i].tosomaLocation
                    if marker is not None and len(marker) == 3 and np.isfinite(marker).all():
                        xyz[i], source[i] = marker, 3
                    else:
                        errors[str(body_ids[i])] = error
                if completed % 500 == 0 or completed == len(wanted):
                    persist()
                    print(json.dumps(dict(completed=completed, total=len(wanted),
                        remaining=int((source == 0).sum()), downloaded_MiB=round(transferred/2**20, 1),
                        seconds=round(time.monotonic()-started, 1))), flush=True)
    finally:
        persist()
    print('Saved:', output_path, 'Unresolved:', int((source == 0).sum()), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    build(args.annotations, args.output, args.workers)
