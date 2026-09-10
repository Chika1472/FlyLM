"""Reproduce the structural audit used by embodied-writing-plan.md.

Run from any directory. Requires numpy, pandas, pyarrow and scipy.
Uses the existing MaleCNS annotation cache; never modifies model files.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ANNOTATIONS = ROOT / "tmp/mb-research/body-annotations-male-cns-v1.0-minconf-0.5.feather"
GRAPH = ROOT / "flylm-2-model/graph"
annotation_hash = hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest()
manifest = json.loads((GRAPH / "complete.json").read_text(encoding="utf-8"))
assert annotation_hash == manifest["spec"]["source_sha256"]["annotations"]
df = pd.read_feather(ANNOTATIONS)
df = df.loc[df.status.eq("Traced")].sort_values("bodyId").reset_index(drop=True)
body_ids = np.load(GRAPH / "body_ids.npy", mmap_mode="r")
assert np.array_equal(df.bodyId.to_numpy(), body_ids)
src = np.load(GRAPH / "src.npy", mmap_mode="r")
crow = np.load(GRAPH / "crow.npy", mmap_mode="r")
inputs = np.load(GRAPH / "inputs.npy")
outputs = np.load(GRAPH / "outputs.npy")
assert len(src) == manifest["nnz"] == int(crow[-1])
assert len(crow) == len(df) + 1
assert int(src.min()) >= 0 and int(src.max()) < len(df)

# Original CSR rows are postsynaptic. Transpose for pre -> post traversal.
post_pre = csr_matrix((np.ones(len(src), dtype=np.uint8), src, crow), shape=(len(df), len(df)))
pre_post = post_pre.transpose().tocsr()
visual = df.superclass.eq("ol_sensory").to_numpy()
motor = df.superclass.eq("vnc_motor").to_numpy()
r16 = df.type.eq("R1-R6").to_numpy()
lamina = df.type.isin(["L1", "L2", "L3", "L4", "L5"]).to_numpy()
t1 = motor & df.somaNeuromere.eq("T1").to_numpy()

def reachability(mask):
    distances = dijkstra(pre_post, directed=True, indices=np.flatnonzero(mask),
                         unweighted=True, min_only=True)
    result = {}
    for label, target in [("all_output_ports", np.isin(np.arange(len(df)), outputs)),
                          ("vnc_motor", motor), ("T1_vnc_motor_candidates", t1)]:
        values = distances[target]
        finite = values[np.isfinite(values)]
        result[label] = {
            "total": int(target.sum()), "reachable": int(len(finite)),
            "minimum_hops": int(finite.min()) if len(finite) else None,
            "maximum_minimum_hops": int(finite.max()) if len(finite) else None,
            "minimum_hop_histogram": {str(int(h)): int((finite == h).sum()) for h in np.unique(finite)},
        }
    return result

visual_records = []
for name, group in df.loc[visual].groupby("type", dropna=False):
    visual_records.append({"type": str(name), "count": int(len(group)),
                           "hex1_nonnull": int(group.assignedOlHex1.notna().sum()),
                           "hex2_nonnull": int(group.assignedOlHex2.notna().sum())})
report = {
    "annotation_sha256": annotation_hash,
    "annotation_source_url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "selection": "status == Traced; sorted bodyId; exact equality with cached graph body_ids",
    "neurons": len(df), "directed_neuron_pair_edges": len(src),
    "input_ports": len(inputs), "output_ports": len(outputs),
    "visual_sensory": int(visual.sum()), "vnc_motor": int(motor.sum()),
    "cb_motor": int(df.superclass.eq("cb_motor").sum()),
    "visual_types": visual_records,
    "R1_R6_by_root_side": {str(k): int(v) for k, v in df.loc[r16].groupby("rootSide", dropna=False).size().items()},
    "R1_R6_to_L1_L5_pair_edges": int(post_pre[lamina][:, r16].nnz),
    "T1_vnc_motor_candidates": int(t1.sum()),
    "reachability_from_visual_sensory": reachability(visual),
    "reachability_from_R1_R6": reachability(r16),
    "limitations": [
        "Unweighted structural reachability ignores synapse strength, sign, physiological delay and trained dynamics.",
        "T1 vnc_motor is a candidate set containing non-leg muscles; this is not a validated front-leg atlas.",
        "Null hex coordinates in this flat table do not imply coordinates are unavailable in official spatial resources.",
        "The source edge-file hash is inherited from the existing graph manifest, not independently redownloaded and checked.",
    ],
}
(OUT / "connectome_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
columns = ["bodyId", "type", "superclass", "class", "subclass", "somaSide", "rootSide", "somaNeuromere",
           "entryNerve", "exitNerve", "assignedOlHex1", "assignedOlHex2", "mancType", "mancBodyid"]
df.loc[visual, columns].to_csv(OUT / "visual_sensory_inventory.csv", index=False)
df.loc[t1, columns].to_csv(OUT / "T1_motor_candidates.csv", index=False)
print(json.dumps(report, ensure_ascii=False, indent=2))
