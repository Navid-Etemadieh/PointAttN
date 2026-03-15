# PointAttN Study Guide (README-Style Teaching Document)

This guide is a **teacher-style walkthrough** of the real `PointAttN` codebase in this repository.
I wrote it by reading the actual files and following function/class dependencies.
Whenever something is not explicitly documented by the authors and I deduce it from behavior, I mark it as **(inferred)**.

---

## 1. Project Overview

### What this project does overall
`PointAttN` is a point cloud completion project: it takes an incomplete 3D point cloud and predicts a completed point cloud.

### What problem it solves
In real scans, many object surfaces are missing because of occlusion/view limitations. This project learns to fill those missing regions.

### Main model/pipeline objective
The model is a coarse-to-fine attention pipeline:
1. Encode partial points with attention blocks and hierarchical sampling.
2. Produce a coarse completion.
3. Refine/upsample it in two stages.
4. Train with Chamfer Distance losses against ground truth complete points.

### Overall inputs
- Partial point clouds from either:
  - PCN data in `.pcd` folders (`dataset.PCN_pcd`).
  - Completion3D data in `.h5` (`dataset.C3D_h5`).
- Ground-truth complete point clouds for training/validation.

### Overall outputs
- During training: checkpoints (`network.pth`, `best_*_network.pth`) and logs.
- During validation/testing: Chamfer metrics (`cd_p`, `cd_t`, coarse variants).
- Optional prediction export:
  - `.obj` for PCN testing.
  - `.h5` for C3D testing.

### High-level repository organization
- `train.py`: training + validation loop.
- `test_pcn.py`, `test_c3d.py`: testing scripts.
- `dataset.py`: dataset readers + augmentations.
- `models/PointAttN.py`: all core model components.
- `utils/model_utils.py`: Chamfer-based metrics/loss helper.
- `utils/train_utils.py`: meters/checkpoint utilities.
- `cfgs/PointAttN.yaml`: experiment configuration.

---

## 2. Directory Structure

```text
PointAttN/
├── README.md                     # project usage instructions
├── train.py                      # training + validation entrypoint
├── test_pcn.py                   # test script for PCN benchmark
├── test_c3d.py                   # test script for Completion3D benchmark
├── dataset.py                    # PCN and C3D dataset classes
├── cfgs/
│   └── PointAttN.yaml            # hyperparameters and runtime settings
├── models/
│   ├── PointAttN.py              # main network definition
│   ├── PointAttN_README.md       # supplementary architecture explanation
│   └── __init__.py
├── utils/
│   ├── model_utils.py            # calc_cd() and chamfer/fscore bindings
│   ├── train_utils.py            # AverageValueMeter + save_model
│   ├── vis_utils.py              # point cloud plotting helpers
│   ├── ChamferDistancePytorch/   # third-party chamfer/fscore extension
│   └── mm3d_pn2/                 # third-party point ops (FPS/gather/etc.)
└── requirements.txt              # Python dependencies
```

Notes:
- `utils/mm3d_pn2` and `utils/ChamferDistancePytorch` are vendor/3rd-party operator code; PointAttN depends on them for geometry ops and Chamfer computation.

---

## 3. End-to-End Execution Flow

1. Parse config path (`-c PointAttN.yaml`) in `train.py`/`test_*.py`.
2. Load YAML into `args` (via `munch`), set visible GPU(s).
3. Build dataset:
   - `PCN_pcd` if `args.dataset == 'pcn'`.
   - `C3D_h5` if `args.dataset == 'c3d'`.
4. Build DataLoader(s) for train/val or test.
5. Dynamically import model module `models.PointAttN`.
6. Construct `Model(args)`, wrap with `torch.nn.DataParallel`, move to CUDA.
7. Training mode (`train.py`):
   - Forward: `out2, loss2, net_loss = net(inputs, gt)`.
   - Backward + optimizer step.
   - Logging every configured step interval.
8. Epoch-level routines:
   - Save periodic checkpoint (`network.pth`).
   - Run `val(...)` and compute validation metrics.
   - Save best checkpoint per metric (`best_cd_p_network.pth`, etc.).
9. Testing mode (`test_pcn.py`/`test_c3d.py`):
   - Load pretrained weights.
   - Run `net(inputs, gt, is_training=False)`.
   - Aggregate/report metrics.
   - Optionally export predictions.

Model internals in each forward pass:
1. `PCT_encoder` creates global feature + coarse output.
2. Concatenate coarse with input, sample 512 anchors using FPS.
3. `PCT_refine` stage-1 upsampling/refinement.
4. `PCT_refine` stage-2 upsampling/refinement.
5. Training: compute 3 Chamfer losses (coarse/intermediate/final).
6. Eval: return final & coarse metrics + outputs.

---

## 4. File-by-File Annotation

### `train.py`

#### Purpose of this file
Main training launcher. Handles config loading, dataset selection, model creation, optimizer/lr scheduling, train loop, validation loop, and checkpoint logic.

#### Important imports
- `from dataset import C3D_h5, PCN_pcd`: dataset dispatch.
- `from utils.train_utils import *`: `AverageValueMeter`, `save_model`.
- `importlib`: dynamically load model module by name.
- `munch`, `yaml`: config object construction.

#### Important classes/functions
- `train()`:
  - Builds datasets and dataloaders.
  - Initializes model and optimizer.
  - Runs epoch training, checkpointing, and validation calls.
- `val(net, curr_epoch_num, ...)`:
  - Switches model to eval mode.
  - Collects metrics from model output dict.
  - Tracks and saves best model per metric.

#### Code snippet with explanation
```python
model_module = importlib.import_module('.%s' % args.model_name, 'models')
net = torch.nn.DataParallel(model_module.Model(args))
net.cuda()
...
out2, loss2, net_loss = net(inputs, gt)
...
result_dict = net(inputs, gt, is_training=False)
```
This is the central control interface:
- Training call returns `(pred, stage_loss, total_loss)`.
- Validation call (with `is_training=False`) returns a dictionary of metrics and outputs.

```python
if args.lr_decay:
    if args.lr_decay_interval:
        if epoch > 0 and epoch % args.lr_decay_interval == 0:
            lr = lr * args.lr_decay_rate
```
Simple interval-based LR decay, with optional clip by `lr_clip`.

#### Inputs and outputs of this file
- Inputs: config YAML + dataset directory paths + optional checkpoint.
- Outputs: logs (`train.log`), periodic model, best-by-metric models.

#### Common confusion points
- `net_loss.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())` is unusual; explicit grad tensor is provided for DataParallel aggregation.
- `metrics` includes coarse and fine CD variants; best checkpoints are tracked independently.

---

### `dataset.py`

#### Purpose of this file
Defines two dataset classes:
- `PCN_pcd` for PCN `.pcd` layout.
- `C3D_h5` for Completion3D `.h5` layout.

#### Important imports
- `open3d` for `.pcd` reading.
- `h5py` for `.h5` reading.
- `transforms3d` for augmentation transforms.

#### Important classes/functions
1. `PCN_pcd.__init__(path, prefix)`
   - Chooses split folder (`train/val/test`).
   - Builds `input_data` list and label mapping.
2. `PCN_pcd.get_data(path)`
   - Traverses class/object/view directories and stores all partial paths.
3. `PCN_pcd.upsample(ptcloud, n_points)`
   - Tiles points when too few, then randomly chooses extras.
4. `PCN_pcd.get_transform(points)`
   - Applies random mirror (and optional scale) consistently to partial+complete.
5. `PCN_pcd.__getitem__(index)`
   - Randomly picks one partial view per object.
   - Reads partial/complete `.pcd`.
   - Upsamples partial to 2048 points.
   - Returns label + tensors (+ object id in test mode).

6. `C3D_h5.__getitem__(index)`
   - Reads partial from `.h5` key `data`.
   - In train/val, also reads GT from mirrored `gt` path and augments.
   - In test mode, returns `(label, partial, partial)` for interface compatibility **(inferred)**.

#### Code snippet with explanation
```python
partial_path = self.input_data[index]
n_sample = len(partial_path)
idx = random.randint(0, n_sample-1)
partial_path = partial_path[idx]
partial = self.read_pcd(partial_path)
partial = self.upsample(partial, 2048)
```
For PCN training, each object has multiple partial views; one view is sampled at runtime.

```python
gt_path = partial_path.replace('/'+partial_path.split('/')[-1],'.pcd')
gt_path = gt_path.replace('partial','complete')
```
GT path is derived by string substitution.

```python
if self.prefix == 'train' and self.sample:
    choice = np.random.permutation((partial.shape[0]))
    partial = partial[choice[:2048]]
```
C3D additionally random-subsamples to fixed 2048 points in training.

#### Inputs and outputs
- Inputs: filesystem tree with class/object files.
- Outputs: `(label, partial_tensor, complete_tensor)` or test variants.

#### Common confusion points
- `if prefix is not "test"` in `C3D_h5.__init__` uses identity comparison (`is not`) instead of value comparison (`!=`).
- Label map values are strings (`'0'...'8'`) not integers.

---

### `models/PointAttN.py`

#### Purpose of this file
Core architecture: attention modules, hierarchical encoder, two refinement stages, and forward logic for train/eval losses.

#### Important imports
- `calc_cd` from `utils.model_utils`: Chamfer metrics/loss.
- `furthest_point_sample`, `gather_points` from `utils.mm3d_pn2`: point sampling/indexing.

#### Important classes/functions
1. `cross_transformer`
   - Generic attention block using `nn.MultiheadAttention` + feedforward residual stack.
   - Input `(B,C,N)` to output `(B,C_out,N_query)`.

2. `PCT_encoder`
   - Extracts multi-scale features via repeated:
     - FPS downsampling (GDP in comments).
     - attention aggregation.
     - feature concatenation.
   - Produces global feature `x_g` and coarse point set.

3. `PCT_refine`
   - Takes coarse points + global feature.
   - Uses attention blocks + channel/shape reshaping.
   - Upsamples by `ratio` and predicts residual xyz offsets.

4. `Model`
   - Chooses `step1` and `step2` ratios by dataset.
   - Chains encoder + refine + refine1.
   - Training mode computes three Chamfer losses.
   - Eval mode returns outputs and CD metrics.

#### Code snippets with explanation
```python
src12 = self.multihead_attn1(query=src1, key=src2, value=src2)[0]
src1 = src1 + self.dropout12(src12)
src1 = self.norm12(src1)
src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
src1 = src1 + self.dropout13(src12)
```
Classic transformer-style block: attention update + FFN update, each with residual path.

```python
idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
x_g0 = gather_points(x0, idx_0)
points = gather_points(points, idx_0)
x1 = self.sa1(x_g0, x0).contiguous()
x1 = torch.cat([x_g0, x1], dim=1)
```
Hierarchical encoder step:
- Sample a subset of points with FPS.
- Use sampled features as queries and dense features as keys/values.
- Concatenate old/new features for richer representation.

```python
fine, feat_fine = self.refine(None, new_x, feat_g)
fine1, feat_fine1 = self.refine1(feat_fine, fine, feat_g)
...
loss3, _ = calc_cd(fine1, gt)
loss2, _ = calc_cd(fine, gt_fine1)
loss1, _ = calc_cd(coarse, gt_coarse)
total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()
```
Coarse-to-fine supervision: each stage compared against an appropriately sampled GT resolution.

#### Inputs and outputs
- Input to `Model.forward`:
  - `x`: partial cloud `(B,3,N)`.
  - `gt`: complete cloud `(B,N_gt,3)` (for loss/metrics).
- Output:
  - Training: `(fine, loss2, total_train_loss)`.
  - Eval: dict with `out1`, `out2`, `cd_t`, `cd_p`, and coarse metrics.

#### Common confusion points
- Variable naming: `fine` in encoder is actually treated as coarse by top model.
- In `PCT_refine.forward`, first argument `x` is not used.
- GT subsampling for multi-stage losses uses FPS on GT itself.

---

### `utils/model_utils.py`

#### Purpose
Defines Chamfer-distance helper `calc_cd` used for both training and evaluation.

#### Important imports
- `dist_chamfer_3D.chamfer_3DDist()` CUDA extension.
- `fscore` helper from third-party package.

#### Key function
- `calc_cd(output, gt, calc_f1=False)`:
  - Inputs: predicted points and GT points, shape `(B,N,3)`.
  - Returns:
    - `cd_p`: sqrt-based chamfer (point-wise average).
    - `cd_t`: raw squared-distance chamfer.
    - optional `f1`.

#### Snippet
```python
dist1, dist2, _, _ = cham_loss(gt, output)
cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
cd_t = (dist1.mean(1) + dist2.mean(1))
```
Two-way nearest-neighbor distances are combined to measure set similarity.

#### I/O
No disk I/O. Pure tensor metric computation.

#### Confusion point
`cham_loss` is instantiated inside function each call (may be slightly inefficient but functionally fine).

---

### `utils/train_utils.py`

#### Purpose
Small training helpers.

#### Important functions
- `AverageValueMeter`: running average tracker.
- `set_requires_grad`: toggles gradient flags.
- `save_model(path, net, net_d=None)`: saves state dict(s).

#### I/O
Writes checkpoint files via `torch.save`.

---

### `test_pcn.py`

#### Purpose
Evaluate trained model on PCN split and optionally save predicted point clouds as `.obj`.

#### Important functions
- `save_obj(point, path)`: writes vertices `v x y z`.
- `test()`: loops through test loader, computes metrics, aggregates category-wise values.

#### Key behavior
- Uses fixed `cat_num = 150` per category (assumed dataset count) **(inferred)**.
- Saves outputs to `.../all/<label>/<obj>.obj` when `save_vis=True`.

#### Inputs/outputs
- Input: checkpoint + PCN test dataset.
- Output: logs, optional `.obj` files.

---

### `test_c3d.py`

#### Purpose
Run inference on Completion3D-style test split and optionally save `.h5` predictions.

#### Important functions
- `save_h5(data, path)` writes predicted point tensor under key `data`.
- `test()` runs forward and optional save.

#### Inputs/outputs
- Input: checkpoint + C3D test dataset.
- Output: logs + optional per-sample `.h5` predictions.

---

### `cfgs/PointAttN.yaml`

#### Purpose
Holds run configuration for training and testing scripts.

#### Important parameters
- Data: `dataset`, `pcnpath`, `c3dpath`.
- Train loop: `batch_size`, `nepoch`, `start_epoch`.
- Optimizer: `lr`, `optimizer`, `betas`, `weight_decay`.
- LR schedule: `lr_decay*`, `lr_clip`.
- Logging/checkpoint intervals.
- Inference save toggle: `save_vis`.

---

### `README.md`

#### Purpose
User-facing quickstart:
- dependency installation,
- CUDA extension compilation,
- train/test commands,
- pretrained model links,
- citation.

---

### Supporting files (short notes)
- `utils/vis_utils.py`: plotting helper using matplotlib+open3d, not used in main train/test scripts.
- `utils/mm3d_pn2/__init__.py`: re-exports ops including FPS and gather functions used by model.
- `requirements.txt`: basic package versions.
- `models/PointAttN_README.md`: additional model walkthrough document.

---

## 5. Function-by-Function Reference (Core)

> Listed are the highest-impact functions for understanding execution.

1. `train()` — `train.py`
   - Purpose: full training lifecycle.
   - Inputs: global `args`.
   - Outputs: checkpoints/logs; no direct return.

2. `val(...)` — `train.py`
   - Purpose: validation metrics and best-model tracking.
   - Inputs: model, loaders, meters, best record dict.
   - Outputs: updates meters + writes best checkpoints.

3. `PCN_pcd.__getitem__(index)` — `dataset.py`
   - Purpose: yield one PCN sample.
   - Output tuple: train/val `(label, partial, complete)`; test adds `obj`.

4. `C3D_h5.__getitem__(index)` — `dataset.py`
   - Purpose: yield one C3D sample from `.h5`.

5. `cross_transformer.forward(src1, src2)` — `models/PointAttN.py`
   - Purpose: attention transform from source2 into source1.
   - Input shape: `(B,C,N)` tensors.
   - Output: transformed query features.

6. `PCT_encoder.forward(points)` — `models/PointAttN.py`
   - Purpose: hierarchical feature extraction + coarse generation.
   - Outputs: global feature token + coarse xyz.

7. `PCT_refine.forward(x, coarse, feat_g)` — `models/PointAttN.py`
   - Purpose: refine and upsample coordinates.
   - Output: refined xyz + intermediate features.

8. `Model.forward(x, gt=None, is_training=True)` — `models/PointAttN.py`
   - Purpose: orchestrate end-to-end prediction and losses/metrics.

9. `calc_cd(output, gt, calc_f1=False)` — `utils/model_utils.py`
   - Purpose: Chamfer metric computation.

10. `save_model(path, net, net_d=None)` — `utils/train_utils.py`
   - Purpose: checkpoint serialization.

---

## 6. Class-by-Class Reference

### `cross_transformer` (`models/PointAttN.py`)
- Purpose: reusable self/cross-attention block.
- Constructor params:
  - `d_model`, `d_model_out`, `nhead`, `dim_feedforward`, `dropout`.
- Internal modules:
  - `input_proj`, `multihead_attn1`, FFN (`linear11/12`), norms/dropouts.
- Forward behavior:
  - project -> (N,B,C) permutation -> attention -> residual+norm -> FFN residual -> back to (B,C,N).

### `PCT_encoder` (`models/PointAttN.py`)
- Purpose: hierarchical encoding + coarse generation.
- Key attributes:
  - Initial convs (`conv1`, `conv2`), multiple `sa*` attention blocks,
  - deconv seed generator (`ps`) and decoder attention (`sa*_d`).
- Outputs:
  - `x_g` global feature `(B,512,1)`.
  - coarse point set `(B,3,N//8)` (default pipeline).

### `PCT_refine` (`models/PointAttN.py`)
- Purpose: point upsampling and coordinate residual refinement.
- Constructor key param: `ratio` (upsampling factor).
- Output: `(B,3,N*ratio)`.

### `Model` (`models/PointAttN.py`)
- Purpose: top-level network wrapper.
- Constructor behavior:
  - selects refine ratios by dataset:
    - PCN: `step1=4`, `step2=8`.
    - C3D: `step1=1`, `step2=4`.
- Forward behavior:
  - encoder -> FPS selection -> refine1 -> refine2 -> loss/metrics.

### Dataset classes (`dataset.py`)
- `PCN_pcd`: filesystem-based `.pcd` loader with per-object multi-view partials.
- `C3D_h5`: `.h5` loader with optional train augmentations.

### Utility class (`utils/train_utils.py`)
- `AverageValueMeter`: stateful average calculator (`val`, `sum`, `count`, `avg`).

---

## 7. Tensor Shape Tracking (Main Model Path)

Assume batch `B`, input partial size `N=2048`.

1. Input to model: `x` = `(B, 3, 2048)`.
2. Encoder output coarse (named `coarse` in `Model`): `(B, 3, 256)` because `N//8`.
3. Concatenate with input: `torch.cat([x, coarse], dim=2)` -> `(B, 3, 2304)`.
4. FPS to 512 points: `new_x` -> `(B, 3, 512)`.
5. Refine stage-1:
   - PCN ratio=4: `(B, 3, 512*4=2048)`.
   - C3D ratio=1: `(B, 3, 512)`.
6. Refine stage-2:
   - PCN ratio=8: `(B, 3, 16384)`.
   - C3D ratio=4: `(B, 3, 2048)`.
7. Transpose before loss/metrics:
   - coarse/fine/final become `(B, N_points, 3)`.

Loss target alignment:
- `fine1` compared to full `gt`.
- `fine` compared to FPS-sampled GT matching `fine` point count.
- `coarse` compared to further sampled GT matching coarse count.

---

## 8. Attention Mechanism Explanation

### Where implemented
- File: `models/PointAttN.py`
- Class: `cross_transformer`

### Attention tensors involved
- Input query tensor `src1` and key/value tensor `src2` initially `(B,C,N)`.
- After permutation: `(N,B,C)` for `nn.MultiheadAttention`.

### How attention scores are formed
`nn.MultiheadAttention` internally computes scaled dot-product attention:
- scores = `Q K^T / sqrt(d)`
- softmax over key positions
- weighted sum over `V`

In this code, `Q` from `src1`, `K/V` from `src2`.
So each query point in `src1` gathers contextual info from all points in `src2`.

### How features are updated
1. Attention output is added to original query (residual).
2. LayerNorm is applied.
3. Feed-forward MLP is applied and added again (second residual).

### Multiple attention blocks in architecture
- Encoder uses several attention modules (`sa1`, `sa1_1`, `sa2`, ... `sa3_1`) for hierarchical representation.
- Decoder/seed side uses (`sa0_d`, `sa1_d`, `sa2_d`).
- Refine module uses (`sa1`, `sa2`, `sa3`) before coordinate regression.

### Methodology relationship
The project title says “you only need attention”; indeed, most feature interaction layers are built on attention blocks, not graph conv/point conv stacks.

---

## 9. Loss Function and Training Objective Explanation

### Where implemented
- Main loss assembly: `Model.forward(..., is_training=True)` in `models/PointAttN.py`.
- CD computation helper: `calc_cd` in `utils/model_utils.py`.

### Loss terms
1. `loss3` (final output vs full GT)
   - `loss3, _ = calc_cd(fine1, gt)`
2. `loss2` (intermediate refine vs sampled GT)
   - GT sampled to `fine` point count.
3. `loss1` (coarse vs sampled GT)
   - GT sampled again to coarse count.

### Total loss
`total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()`

So it is a simple sum of three CD terms (equal weights).

### Shapes
- `fine1`, `gt`: `(B,Nf,3)`.
- `fine`, `gt_fine1`: `(B,Ni,3)`.
- `coarse`, `gt_coarse`: `(B,Nc,3)`.

### Methodological meaning
Multi-stage supervision encourages:
- coarse stage learns global object structure,
- intermediate stage learns finer arrangement,
- final stage improves dense detail.

---

## 10. How the code maps to the methodology

- **Feature extraction**: `PCT_encoder` initial conv + attention hierarchy.
- **Encoder logic**: `PCT_encoder.forward`, especially FPS + `sa*` blocks.
- **Attention logic**: `cross_transformer.forward`.
- **Decoder/upsampling/refinement**: `PCT_refine.forward` (twice in `Model`).
- **Point generation**:
  - coarse generation at encoder output (`fine` variable from encoder).
  - refined dense generation at `fine`, `fine1` in `Model.forward`.
- **Loss computation**: `Model.forward` + `calc_cd`.
- **Metrics**: eval branch in `Model.forward`, aggregated in `val`/`test_pcn`.
- **Training logic**: `train.py::train`.
- **Evaluation logic**: `train.py::val`, `test_pcn.py`, `test_c3d.py`.

---

## 11. Parameter and Configuration Explanation

From `cfgs/PointAttN.yaml` and code usage:

### Data parameters
- `dataset`: selects PCN vs C3D pipeline.
- `pcnpath`, `c3dpath`: root dataset directories.
- `num_points`: declared but not heavily used directly in model flow **(inferred)**.

### Model-related
- `model_name`: module in `models/` to import.
- Refine ratios are not in YAML; derived inside model from `dataset`.

### Training hyperparameters
- `batch_size`, `workers`, `nepoch`, `start_epoch`.
- `optimizer`, `lr`, `betas`, `weight_decay`.
- LR schedule: `lr_decay`, `lr_decay_interval`, `lr_decay_rate`, `lr_clip`.

### Runtime/logging
- `device`: sets `CUDA_VISIBLE_DEVICES`.
- `work_dir`, `flag`, `loss`: used in experiment directory naming.
- `step_interval_to_print`, `epoch_interval_to_save`, `epoch_interval_to_val`.

### Evaluation/testing
- `load_model`: checkpoint path used in train resume or testing.
- `save_vis`: controls prediction export.

---

## 12. Inputs and Outputs of the Whole Codebase

### Expected input folders and formats
- PCN layout:
  - `train/partial/<class>/<obj>/<view>.pcd`
  - `train/complete/<class>/<obj>.pcd`
  - similar for `test`.
- C3D layout:
  - `train/partial/<class>/<obj>.h5`
  - `train/gt/<class>/<obj>.h5`

### What dataloaders return
- PCN train/val: `(label, partial, complete)`.
- PCN test: `(label, partial, complete, obj)`.
- C3D train/val: `(label, partial, complete)`.
- C3D test: `(label, partial, partial)`.

### What model returns
- Training mode: predicted intermediate cloud + stage-2 loss + total loss.
- Eval mode: coarse and final outputs with 4 CD metrics.

### What scripts save
- `train.py`: `.pth` checkpoints + `train.log`.
- `test_pcn.py`: `test.log`, optional `.obj` predictions.
- `test_c3d.py`: `test.log`, optional `.h5` predictions.

---

## 13. Important Calculations and Operations in Code

Only operations actually used in this project:

- `transpose(2,1)` / `permute(...)`: switch between `(B,N,3)` and `(B,3,N)`, and to `(N,B,C)` for multi-head attention.
- `reshape(...)`: remap channel-length dimensions, especially in refine upsampling.
- `torch.cat([...], dim=1 or 2)`: concatenate features or point sets.
- `repeat(1,1,ratio)`: duplicate points/features for upsampling base.
- FPS (`furthest_point_sample`): downsample points while preserving spread.
- `gather_points`: fetch sampled point features/coordinates.
- `adaptive_max_pool1d(...,1)`: global feature token.
- Residual additions (`a + b`) in attention blocks and coordinate refinement.
- Chamfer nearest-neighbor distances (via CUDA extension).

Example from refine:
- predict `delta_xyz`, then add repeated coarse coordinates: residual coordinate learning.

---

## 14. Important PyTorch Concepts Used Here

- `nn.Module`: every model block (`cross_transformer`, `PCT_encoder`, etc.).
- `forward`: defines computation graph per block.
- `DataParallel`: multi-GPU wrapping in train/test scripts.
- Train vs eval modes:
  - `net.module.train()` enables training behavior.
  - `net.module.eval()` + `torch.no_grad()` for evaluation.
- Backprop:
  - `loss.backward(...)` then `optimizer.step()`.
- Optimizer setup:
  - chosen by name from config (`Adam` default).
- Checkpointing:
  - `torch.save` in `save_model`, `load_state_dict` in scripts.
- Device placement:
  - tensors moved via `.cuda()`.
- Tensor broadcasting:
  - used in additions and repeats (e.g., residual coordinate addition).

---

## 15. Examples for Understanding

### Example A: data flow of one training batch (PCN)
1. Loader returns `partial` `(B,2048,3)`, `complete` `(B,M,3)`.
2. `train.py` transposes partial -> `(B,3,2048)`.
3. Model outputs `fine1` (final dense prediction).
4. Three CD losses are computed against GT at matching resolutions.
5. Loss sum is backpropagated.

### Example B: attention block intuition
Suppose query has 256 points and key/value has 1024 points:
- each query point computes similarity to all 1024 key points,
- then mixes key features with attention weights,
- result is a context-enriched feature for that query point.

### Example C: why GT is downsampled
When intermediate output has fewer points than full GT, code FPS-samples GT to same count, enabling consistent CD comparison at each stage.

---

## 16. Recommended Reading Order

1. `cfgs/PointAttN.yaml` (understand settings).
2. `train.py` (overall loop and lifecycle).
3. `dataset.py` (what tensors the model actually receives).
4. `models/PointAttN.py` (core architecture).
5. `utils/model_utils.py` (loss/metric math).
6. `test_pcn.py` and `test_c3d.py` (evaluation and export behavior).
7. `README.md` (usage context, compile/setup notes).
8. `utils/train_utils.py` and `utils/vis_utils.py`.
9. Optional deep dive: `utils/mm3d_pn2` and `utils/ChamferDistancePytorch` internals.

---

## 17. Study Notes

### 20 most important things to learn
1. Input/output tensor layouts are frequently transposed.
2. Model is coarse-to-fine, not single-shot.
3. Attention is the main interaction primitive.
4. FPS + gather are central to hierarchical processing.
5. Encoder returns both global token and coarse xyz.
6. Two refiner stages upscale by dataset-dependent ratios.
7. Final loss is sum of three Chamfer terms.
8. Validation saves best models per metric.
9. PCN and C3D data readers differ in format and test behavior.
10. GT sampling aligns stage outputs with supervision sizes.
11. `cross_transformer` wraps MultiheadAttention + FFN residual.
12. Refiner predicts xyz deltas added to repeated base points.
13. `calc_cd` returns both sqrt-based and squared CD variants.
14. The code heavily relies on compiled CUDA extensions.
15. DataParallel conventions affect saving/loading (`net.module`).
16. Config drives many runtime decisions.
17. Category-wise reporting appears in PCN test script.
18. Augmentation mainly mirror/scale in datasets.
19. Some naming is confusing (`fine` vs `coarse`).
20. Understanding shape flow is key to mastering this code.

### 20 study questions to ask yourself
1. Why does encoder use FPS at multiple scales?
2. What is the exact role of each `sa*` block?
3. Why concatenate old and new features after attention?
4. How does `ConvTranspose1d` create seed structure?
5. Why are two refinement stages needed?
6. How do refinement ratios differ between PCN and C3D?
7. Why compare intermediate outputs to sampled GT?
8. What do `cd_p` and `cd_t` each emphasize?
9. Which tensors are `(B,3,N)` vs `(B,N,3)` at each stage?
10. Why is explicit gradient passed to `backward`?
11. How does random partial-view selection affect training?
12. What assumptions are hardcoded about dataset folder structure?
13. What happens if FPS/Gather extensions are missing?
14. How much does augmentation help completion quality?
15. Is `if prefix is not "test"` potentially problematic?
16. Why is global max pooling used before seed decoding?
17. How does residual xyz prediction stabilize refinement?
18. Where would you add new loss terms if needed?
19. Which script would you modify for custom inference export?
20. What would break if point count N changes a lot?

### Top files to focus on first
1. `models/PointAttN.py`
2. `train.py`
3. `dataset.py`
4. `utils/model_utils.py`
5. `cfgs/PointAttN.yaml`

### Files to inspect next for helper logic/dependencies
1. `test_pcn.py`
2. `test_c3d.py`
3. `utils/train_utils.py`
4. `utils/mm3d_pn2/__init__.py`
5. `utils/ChamferDistancePytorch/chamfer3D/dist_chamfer_3D.py`

---

If you want, I can generate a **second companion guide** that only covers `models/PointAttN.py` line-by-line with pseudocode and hand-worked shape tables for every block.
