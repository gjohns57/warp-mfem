# PokeFlex_Tracked

Temporally tracked surface meshes derived from the
[PokeFlex dataset](https://pokeflex.ait.ethz.ch/). Unlike the original
per-frame capture meshes, vertex `i` represents the same tracked material
point in every frame.

This research dataset currently contains four T1 poking episodes:

| episode | frames | canonical vertices | faces |
|---|---:|---:|---:|
| `FoamDice_T1` | 150 | 15,002 | 30,000 |
| `PlushOctopus_T1` | 155 | 15,002 | 30,000 |
| `PlushTurtle_T1` | 155 | 15,002 | 30,000 |
| `ToiletPaperRoll_T1` | 150 | 14,983 | 30,000 |

## Recommended files

For tracking, visualization, or machine learning, use these files inside each
episode directory:

| file | contents |
|---|---|
| `mesh_trajectories_canonical.npy` | `float32 [T,N,3]` positions in **meters**, in the PokeFlex world frame |
| `valid_mask_canonical.npy` | `bool [T,N]`; whether a vertex is directly supported by tracking observations |
| `template_canonical.obj` | Fixed triangle topology; OBJ coordinates are in **millimeters** |
| `template_canonical_colors.npy` | `uint8 [N,3]` RGB colors |
| `nviews_canonical.npz` | Number of real and virtual views supporting each vertex |
| `tool_trajectory.npz` | Synchronized tool pose, calibrated tip position, wrench, timestamps, and validity flags |
| `qc_canonical.json` | Tracking quality measurements |
| `manifest.json` | Episode provenance, parameters, and result records |

Frame `t` corresponds to PokeFlex capture frame `t + 1`. Positions are
absolute; compute displacement with:

```python
displacement = trajectory - trajectory[0:1]
```

Use the OBJ for its faces. Multiply its vertices by `1e-3` only if you need
them in meters; the trajectory array is already in meters.

## Minimal loader

```python
from pathlib import Path
import numpy as np


def obj_faces(path):
    faces = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("f "):
            faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    return np.asarray(faces, dtype=np.int64)


episode = Path("/path/to/PokeFlex_Tracked/FoamDice_T1")
trajectory = np.load(episode / "mesh_trajectories_canonical.npy", mmap_mode="r")
valid = np.load(episode / "valid_mask_canonical.npy", mmap_mode="r")
colors = np.load(episode / "template_canonical_colors.npy", mmap_mode="r")
faces = obj_faces(episode / "template_canonical.obj")

vertices = np.asarray(trajectory[60])
observed_vertices = vertices[valid[60]]
```

Load the synchronized tool trajectory with:

```python
with np.load(episode / "tool_trajectory.npz", allow_pickle=False) as tool:
    tool_transform = tool["tool_transform"]      # [T,4,4], tool to world
    tool_origin = tool_transform[:, :3, 3]       # [T,3], meters
    contact_position = tool["contact_position"] # [T,3], tool tip, meters
    timestamps = tool["timestamps"]              # [T], seconds
    force = tool["force"]                        # [T,3], newtons, world frame
    torque = tool["torque"]                      # [T,3], N m, world frame
```

All tool arrays are frame-aligned with `mesh_trajectories_canonical.npy`.
Use `force_valid` and `torque_valid` before consuming the calibrated wrench.

## Interpretation

- A `False` validity value means the finite position was inferred by the
  spatial/temporal model rather than directly observed. Use the mask for
  observation-only losses and metrics.
- The canonical surface fills capture holes and is not raw sensor ground
  truth in permanently unseen regions.
- `qc*.json` contains quality checks such as surface distance, observation
  coverage, distortion, temporal jitter, and self-intersections. Metric
  schemas evolved during development, so read the definitions in each file.
- The raw-template files (`mesh_trajectories.npy`, `valid_mask.npy`, and
  `nviews.npz`) are useful for auditing, but the canonical files above are the
  cleanest downstream interface.
- Files containing `physics`, `implicit`, `latent_contact`, `smoke`, `state`,
  or `failed` are experimental reconstructions or diagnostics. Do not use
  them as observation-derived labels without checking their `result.json` and
  `manifest.json` records.

## Viewer

From the dataset directory:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open <http://127.0.0.1:8000/index.html>. For a remote machine, create a tunnel
from your local computer:

```bash
ssh -L 8000:127.0.0.1:8000 user@remote-machine
```

`index.html` links to the method/results page and viewers.
`dataset_viewer.html` is self-contained and can usually also be opened
directly, but serving the directory is more reliable.

## Physics solver status

The reported physics reconstructions use the established **CPU reference
solver**: implicit nonlinear tetrahedral FEM with projected Newton iterations,
an assembled sparse tangent matrix, and a SuperLU direct solve. It includes
finite-strain elasticity, viscosity and crushable-foam plasticity, IPC-style
tool/table contact, friction, observation anchors, and a feasibility-preserving
line search that prevents inverted tetrahedra and invalid contact gaps.

An isolated matrix-free FP64 CUDA implementation also exists in
`pokeflex_tracking_gpu/`. It replaces sparse assembly and factorization with
preconditioned conjugate gradient, but is still **work in progress**. It has
not replaced the CPU solver or any released result and requires full A100
parity and episode-level validation before promotion.

## Method and provenance

The canonical tracks combine dense multi-view tracking, ray-cast lifting,
confidence-aware fusion, deformation and temporal regularization, geometric
re-registration, surface relaxation, and transfer to a fixed canonical
topology. Experimental physics results add finite-element simulation, contact,
material identification, held-out evaluation, and surface refinement.

Code: [tolesch/pokeflex-tracking](https://github.com/tolesch/pokeflex-tracking).
Detailed methods and results are in `results.html`; exact per-episode settings
are in `manifest.json`.

This directory is derived from PokeFlex and does not grant additional
redistribution rights. Obtain PokeFlex access and follow its license and
attribution requirements. Cite PokeFlex and the tracking/reconstruction method
associated with the released version.
