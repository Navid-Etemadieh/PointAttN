# PointAttN Dental Notebook — Teacher-Style Study Guide

Target notebook: `models/PointAttN_Dental.ipynb`

This guide is written as a **beginner-friendly, teacher-style reference**. It is based on reading the notebook cell-by-cell in execution order and checking closely related local files needed to understand the logic.

---

# 1. Notebook Overview

`models/PointAttN_Dental.ipynb` is a **full workflow notebook**, not only a model definition notebook.

It does all of these:

1. Environment setup and package installation.
2. CUDA/toolchain checks and native extension build preparation.
3. Building required geometric ops (mm3d point ops and Chamfer distance).
4. Creating new project files for the dental exact-size workflow (`PointAttN_exact.py`, `dataset_exact.py`, `train_exact.py`, `test_exact.py`, YAML config).
5. Training / resume training logic.
6. Testing and saving predictions.
7. Visualization (single case + batch triplets).
8. Quantitative evaluation and report generation.

So its role in the full pipeline is:
- **infrastructure + experiment authoring + training + evaluation + reporting**.

It exists because dental scans can have very large point counts, and the notebook introduces practical controls (for example point caps and sampled Chamfer evaluation) to make training feasible while still producing exact-size outputs.

---

# 2. High-Level Summary of the Notebook

## Major sections

- **Cells 0–22**: Build/runtime environment preparation.
- **Cell 27**: Central configuration (`ExactCfg`).
- **Cells 28–29**: Dependency checks.
- **Cells 30–33**: Writes core exact-pipeline files.
- **Cell 34**: Writes YAML config.
- **Cell 35**: Dataset shape sanity check.
- **Cells 36–37**: Resume patch + training run.
- **Cells 38–40**: Pick checkpoint + run testing + inspect outputs.
- **Cells 41–42**: Qualitative visualization.
- **Cells 43–44**: Quantitative evaluation and report plotting.
- **Cells 45–46**: empty.

## Overall data flow

`dataset file pair -> partial/gt tensors -> transpose to model layout -> encoder -> coarse -> refine -> refine -> exact tail -> output cloud -> Chamfer-based losses/metrics -> save outputs/reports`

## Major stages of computation

1. Build needed low-level operators.
2. Define parameters and exact-size model behavior.
3. Build dataset loader and model code.
4. Run training and validation.
5. Run test and save predicted point clouds.
6. Visualize and quantify results.

---

# 3. Imports and Dependencies

## Important import families and usage

### PyTorch imports
- `torch`, `torch.nn`, `torch.nn.functional`, `torch.optim`
- Used for model modules, attention, loss backprop, optimizer, dataloading.

### NumPy / plotting / utilities
- `numpy`, `pandas`, `matplotlib`
- `yaml`, `munch`, `pathlib`, `subprocess`, `re`, `shutil`, `time`
- Used for config IO, parsing logs, plotting, and shell orchestration.

### Project-local imports
- `from utils.mm3d_pn2 import furthest_point_sample, gather_points`
- `from utils.model_utils import calc_cd`
- `from utils.train_utils import AverageValueMeter, save_model`

These are critical. If they fail, core method cannot run.

## Which local files you should inspect next

1. `models/PointAttN.py` (baseline PointAttN design).
2. `utils/model_utils.py` (Chamfer calculation wrapper).
3. `utils/train_utils.py` (meters and checkpoint saving).
4. `train.py` (original project training style).
5. `dataset.py` (base dataset conventions).

---

# 4. Cell-by-Cell Annotation

> Note: The notebook has almost only code cells. There are no rich markdown explanation cells, so state is carried mainly through Python variables.

## Cells 0–3: Base package installation and CUDA torch setup
- **Purpose**: install compatible dependencies and force CUDA-enabled torch.
- **Code snippet**:
```python
run([PY, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel", "ninja", "pybind11", "packaging"])
...
run_py_cmd([
    sys.executable, "-m", "pip", "install", "-U",
    "torch==2.5.1+cu124",
    "torchvision==0.20.1+cu124",
    "torchaudio==2.5.1+cu124",
    "--index-url", "https://download.pytorch.org/whl/cu124"
])
```
- **Detailed explanation**: these cells aggressively normalize environment assumptions so later C++/CUDA extension compilation matches PyTorch ABI.
- **Inputs used**: current Python env, requirements file path.
- **Outputs produced**: installed packages and CUDA torch.
- **Why this matters**: all geometry kernels depend on this consistency.

## Cells 4–6: Repo discovery and reusable shell logging helpers
- **Purpose**: define utility functions and locate repo paths.
- **Snippet**:
```python
def run_bash_to_log(script: str, log_path: Path, ...): ...
REPO_ROOT = find_repo_root(Path.cwd())
MM3D = (REPO_ROOT / "utils" / "mm3d_pn2").resolve()
OPS = MM3D / "ops"
LOG_DIR = ensure_dir(REPO_ROOT / "_build_logs")
```
- **State created**: `REPO_ROOT`, `MM3D`, `OPS`, `LOG_DIR` used by many later cells.

## Cells 7–13: mm3d point op compatibility patch/build
- **Purpose**: prepare and compile FPS/gather ops.
- **Snippet**:
```python
PATCHES = [
  (re.compile(r'^\s*#\s*include\s*<\s*THC/THC\.h\s*>\s*$', re.M), ""),
  (re.compile(r'(\b\w+\b)\.type\(\)\.is_cuda\(\)'), r"\1.is_cuda()"),
]
...
python setup.py build_ext --inplace
```
- **Explanation**: removes old THC-era code patterns and builds shared objects.
- **Why matters**: model uses `furthest_point_sample` and `gather_points` everywhere.

## Cells 14–22: ChamferDistance clone/patch/build
- **Purpose**: ensure Chamfer extension works with current Python/CUDA toolchain.
- **Snippet**:
```python
git clone https://github.com/ThibaultGROUEIX/ChamferDistancePytorch.git
...
python setup.py build_ext --inplace
```
- **State used later**: Chamfer import path used by `utils/model_utils.py`.

## Cells 23–26: Empty
- No operation.

## Cell 27: Global experiment config (`ExactCfg`)
- **Purpose**: define all path/data/model/train/eval controls in one dataclass.
- **Important variables**:
  - `max_encoder_points`, `exact_cd_points`, `metric_cd_points`
  - `step1`, `step2`, `encoder_coarse_points`
  - `exact_target_from`, `exact_target_points`
- **Outputs**: `CFG`, `REPO_ROOT`, `DATA_ROOT`, `CFG_PATH`.

## Cell 28: Install common runtime packages
- **Purpose**: install scientific and IO libraries.

## Cell 29: Import check for critical custom ops
- **Purpose**: quick fail-fast check before writing/starting model code.

## Cell 30: Write exact model file (`models/PointAttN_exact.py`)
- **Purpose**: define architecture used in this notebook run.
- **Key blocks created**:
  - `fps_bcn`, `fps_bnc`
  - `cross_transformer`
  - `PCT_encoderExact`
  - `PCT_refine`
  - `ExactSizeHead`
  - top-level `Model`
- **Connection**: later `train_exact.py` imports this model via `args.model_name`.

## Cell 31: Write exact dataset file (`dataset_exact.py`)
- **Purpose**: dental pair dataset loader with normalization and optional augmentation.

## Cell 32: Write train script (`train_exact.py`)
- **Purpose**: training loop + validation + checkpoint save.

## Cell 33: Write test script (`test_exact.py`)
- **Purpose**: inference/evaluation and optional export of predicted clouds.

## Cell 34: Write YAML config (`cfgs/PointAttN_exact.yaml`)
- **Purpose**: serialize `CFG` dataclass values for scripts.

## Cell 35: Dataset smoke test
- **Purpose**: check one sample shape/dtype before expensive training.

## Cell 36: Patch train script for full resume checkpoints
- **Purpose**: add optimizer/epoch/lr/best-metric resume state.
- **Important**: this cell modifies behavior of `train_exact.py` created in cell 32.

## Cell 37: Auto-resume and launch training
- **Purpose**: infer resume epoch from log, update YAML, start training.

## Cell 38: Select checkpoint for testing
- **Purpose**: choose `best_cd_p_network.pth` or fallback `network.pth` and update YAML.

## Cell 39: Launch test script
- **Purpose**: run `test_exact.py` with selected checkpoint.

## Cell 40: Inspect test output folder
- **Purpose**: verify prediction files exist in `<ckpt_dir>/all`.

## Cell 41: Single sample qualitative visualization
- **Purpose**: load one item, run model, plot partial/pred/GT.

## Cell 42: Batch triplet visualization export
- **Purpose**: save many side-by-side plots for dataset items.

## Cell 43: Quantitative evaluation writer
- **Purpose**: compute per-case and overall metrics and write text report.

## Cell 44: Parse quant report and generate CSV/figures
- **Purpose**: convert text report to tables/plots for quick analysis.

## Cells 45–46
- Empty.

---

# 5. Full Line-by-Line Explanation (important blocks)

## 5.1 Model forward block (Cell 30 generated `Model.forward`)

```python
x_raw = x
x_enc = self._maybe_cap_encoder_input(x_raw)

feat_g, coarse = self.encoder(x_enc)

if self.use_input_for_seed_sampling:
    seed_source = torch.cat([x_raw, coarse], dim=2)
else:
    seed_source = torch.cat([x_enc, coarse], dim=2)

seed = fps_bcn(seed_source, min(self.base_seed_points, seed_source.shape[-1]))

fine, feat_fine = self.refine(None, seed, feat_g)
fine1, feat_fine1 = self.refine1(feat_fine, fine, feat_g)

target_points = self._resolve_target_points(x_raw, gt)
exact = self.exact_head(fine1, feat_g, target_points)
```

Line-by-line notes:
- `x_raw = x`: keep original full input for possible later use.
- `_maybe_cap_encoder_input`: optionally FPS-reduce points to control memory.
- `self.encoder(x_enc)`: generate global feature (`feat_g`) and coarse output.
- `seed_source` concat on point axis: combines observed partial and coarse estimate.
- `fps_bcn(...)`: selects seed points with good spatial coverage.
- `refine` then `refine1`: two-stage coarse-to-fine refinement.
- `_resolve_target_points`: decides exact final point count source (`gt`, `input`, or config).
- `exact_head`: forces exact final output point count.

Shape meaning:
- input expected `[B,3,N]`.
- outputs later converted to `[B,N,3]` for Chamfer and saving.

## 5.2 Attention block line-by-line (`cross_transformer.forward`)

```python
src1 = self.input_proj(src1)
src2 = self.input_proj(src2)

b, c, _ = src1.shape
src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

src1 = self.norm13(src1)
src2 = self.norm13(src2)

src12 = self.multihead_attn1(query=src1, key=src2, value=src2)[0]
src1 = src1 + self.dropout12(src12)
src1 = self.norm12(src1)

src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
src1 = src1 + self.dropout13(src12)
src1 = src1.permute(1, 2, 0)
```

Line-by-line purpose:
- Project channel size to attention dimension.
- Rearrange `[B,C,N] -> [N,B,C]` for `nn.MultiheadAttention`.
- Normalize both token streams.
- Cross-attention computes context-aware features for `src1` from `src2`.
- Residual + norm maintains stable gradients.
- FFN refines token features.
- Convert back to model’s channel-first layout.

## 5.3 Loss block line-by-line (training branch)

```python
pred_exact_cd = self._maybe_sample_for_cd(exact_bnc, self.exact_cd_points)
gt_exact_cd = gt
if pred_exact_cd.shape[1] != gt.shape[1]:
    gt_exact_cd = self._maybe_sample_for_cd(gt, pred_exact_cd.shape[1])

loss_exact, _ = calc_cd(pred_exact_cd, gt_exact_cd)

gt_fine1 = fps_bnc(gt, fine1_bnc.shape[1])
loss_fine1, _ = calc_cd(fine1_bnc, gt_fine1)

gt_fine = fps_bnc(gt_fine1, fine_bnc.shape[1])
loss_fine, _ = calc_cd(fine_bnc, gt_fine)

gt_coarse = fps_bnc(gt_fine, coarse_bnc.shape[1])
loss_coarse, _ = calc_cd(coarse_bnc, gt_coarse)
```

Explanation:
- `exact_cd_points` can downsample final cloud for feasible CD compute.
- GT is size-matched when needed.
- Intermediate outputs are supervised by FPS-aligned GT at each stage.
- This is deep supervision from coarse to exact output.

## 5.4 Dataset getitem block

```python
partial = load_points_any(partial_path)
gt = load_points_any(gt_path)

if self.normalize:
    partial, gt = normalize_pair(partial, gt)

if self.augment_train:
    partial, gt = random_pair_transform(partial, gt)

partial = torch.from_numpy(partial)
gt = torch.from_numpy(gt)

label = 0
return label, partial, gt, stem
```

Explanation:
- Loads paired clouds by stem.
- Optional normalization and augmentation are applied to both clouds consistently.
- Converts arrays to tensors, returns tuple consumed by DataLoader loops.

---

# 6. File-by-File Annotation for Related Dependencies

## `utils/model_utils.py`
- Why needed: defines Chamfer wrapper used by model loss and metrics.
- Key function: `calc_cd(output, gt, calc_f1=False)`.
- Connects to notebook: generated model imports it directly.

## `utils/train_utils.py`
- Why needed: helper meter and model-saving utilities.
- Key functions/classes: `AverageValueMeter`, `save_model`.
- Used by generated train/test scripts.

## `models/PointAttN.py`
- Why needed: baseline architecture pattern.
- Notebook-generated `PointAttN_exact.py` preserves much of this design but adds exact-size output controls.

## `dataset.py` and `train.py` (context files)
- Why useful: show project-wide conventions and naming style.

---

# 7. Function-by-Function Explanation

## In generated model (`PointAttN_exact.py`)

### `fps_bcn(points_bcn, npoints)`
- Input: `[B,C,N]`.
- Output: `[B,C,npoints]`.
- Purpose: FPS sampling in channel-first layout.

### `fps_bnc(points_bnc, npoints)`
- Input: `[B,N,C]`.
- Output: `[B,npoints,C]`.
- Purpose: FPS sampling in point-first layout.

### `cross_transformer.forward(src1, src2, if_act=False)`
- Purpose: cross/self attention feature update.
- Output layout: `[B,C,N]`.

### `PCT_encoderExact.forward(points)`
- Purpose: extract global context + generate coarse cloud.
- Key ops: staged FPS, gather, attention, global max pooling.

### `PCT_refine.forward(x, coarse, feat_g)`
- Purpose: upsample and refine point cloud by `ratio`.
- Output: refined coordinates and intermediate features.

### `ExactSizeHead.forward(base_points_bcn, feat_g, target_points)`
- Purpose: enforce exact output point count.
- Strategy: repeat -> residual correction -> FPS trim.

### `Model.forward(x, gt=None, is_training=True)`
- Training branch outputs: `(exact_bnc, loss_exact, total_train_loss)`.
- Eval branch outputs: dict with outputs and CD metrics.

## In generated dataset (`dataset_exact.py`)

### `load_points_any(path)`
- Loads `.npy` or `.ply`, returns float32 `(N,3)`.

### `normalize_pair(partial, gt)`
- Centers by GT centroid, scales by GT max radius.

### `random_pair_transform(partial, gt)`
- Applies shared geometric transform to both clouds.

### `PairPointCloudDataset.__getitem__(index)`
- Returns `(label, partial, gt, stem)`.

## In `utils/model_utils.py`

### `calc_cd(output, gt, calc_f1=False)`
- Computes Chamfer-based metrics (`cd_p`, `cd_t` and optional `f1`).

---

# 8. Class-by-Class Explanation

## `cross_transformer`
- Exists to perform context aggregation through multi-head attention.
- Internals: `input_proj`, `MultiheadAttention`, layer norms, FFN.
- Used repeatedly in encoder and refiners.

## `PCT_encoderExact`
- Purpose: compress large point set into learned global/contextual representation and coarse cloud.
- Why exists: initial feature extraction and geometric scaffold.

## `PCT_refine`
- Purpose: coarse-to-fine upsampling with attention-guided residual updates.
- Called twice with different ratios.

## `ExactSizeHead`
- Purpose: final exact point count control.
- Why exists: dental data can require very high and exact output sizes.

## `Model`
- Orchestrates all stages and computes training/evaluation outputs.

## `PairPointCloudDataset`
- Handles paired dental partial/GT reading, transforms, and output tuple.

---

# 9. End-to-End Execution Flow

1. Install/verify packages and CUDA torch.
2. Discover repo paths and create log helpers.
3. Patch and build mm3d ops.
4. Clone/patch/build Chamfer extension.
5. Define `ExactCfg` experiment parameters.
6. Write exact model/dataset/train/test scripts.
7. Write YAML config.
8. Run dataset smoke test.
9. Patch train script for full resume checkpoint behavior.
10. Launch training (new or resume).
11. Select checkpoint.
12. Run testing script.
13. Verify prediction files.
14. Generate qualitative plots.
15. Generate quantitative metrics/report tables and figures.

---

# 10. Inputs and Outputs of the Whole Notebook

## Expected inputs
- Dataset root with:
```text
<root>/<split>/partial/*.npy
<root>/<split>/gt/*.npy
```
for `split in {train,val,test}`.

- Valid CUDA/toolchain environment for extension builds.
- Correct path values in `ExactCfg` (`repo_root`, `data_root`).

## User-provided configuration values
- Data splits and normalization flags.
- Model size controls (`max_encoder_points`, `base_seed_points`, `step1`, `step2`).
- Training hyperparameters (lr, epochs, optimizer, workers).
- Loss weights and CD sampling controls.
- Save options (`save_vis`, `save_ext`).

## Notebook outputs
- Generated files: `models/PointAttN_exact.py`, `dataset_exact.py`, `train_exact.py`, `test_exact.py`, `cfgs/PointAttN_exact.yaml`.
- Checkpoints in `log/<exp_name>/`.
- Saved predictions in `<ckpt_dir>/all`.
- Quantitative text reports and CSV/PNG charts.
- Visualization PNGs for case triplets.

---

# 11. Tensor Shape Tracking

Assume a batch from loader:
- `partial`: `(B, Np, 3)`
- `gt`: `(B, Ng, 3)`

Model path:
1. transpose for model:
   - input `(B, Np, 3)` -> `(B, 3, Np)`
2. optional cap:
   - `(B, 3, Np)` -> `(B, 3, Nenc)` where `Nenc <= max_encoder_points`
3. encoder outputs:
   - `feat_g` roughly `(B, 512, 1)`
   - `coarse` `(B, 3, Ncoarse)`
4. seed source:
   - concat point dimension `(B,3,Np+Ncoarse)`
   - FPS seed `(B,3,Nseed)`
5. refine 1:
   - `(B,3,Nseed)` -> `(B,3,Nseed*step1)`
6. refine 2:
   - `(B,3,Nseed*step1)` -> `(B,3,Nseed*step1*step2)`
7. exact head:
   - -> `(B,3,Ntarget)`
8. transpose for losses/metrics/saving:
   - `(B,3,Ntarget)` -> `(B,Ntarget,3)`

Attention block shape (inside transformer):
- `[B,C,N] -> [N,B,C] -> [B,C,N]`.

---

# 12. Attention Mechanism Explanation

## Where implemented
- Notebook Cell 30 generated model code (`cross_transformer`).
- Baseline reference structure also appears in `models/PointAttN.py`.

## Attention code snippet

```python
src1 = self.input_proj(src1)
src2 = self.input_proj(src2)

b, c, _ = src1.shape
src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

src1 = self.norm13(src1)
src2 = self.norm13(src2)

src12 = self.multihead_attn1(query=src1, key=src2, value=src2)[0]
src1 = src1 + self.dropout12(src12)
src1 = self.norm12(src1)
```

## Careful explanation
- Projection aligns channel dimensions for attention.
- `query=src1`, `key=value=src2` means stream-1 gathers context from stream-2.
- Residual keeps original information path.
- Layer norm and FFN stabilize/strengthen token updates.

## Math idea in simple words
For each point token in `src1`, compute "how related" it is to each token in `src2`, use those relations as weights, and blend `src2` values into an updated representation for `src1`.

---

# 13. Loss Function / Training Objective Explanation

## Where loss is calculated
- In Cell 30 generated `Model.forward(..., is_training=True)`.
- Distance primitive from `utils/model_utils.py::calc_cd`.

## Loss terms
- `loss_coarse`: coarse output vs downsampled GT.
- `loss_fine`: first refined output vs downsampled GT.
- `loss_fine1`: second refined output vs downsampled GT.
- `loss_exact`: final exact output vs GT (or sampled GT if CD sampling is active).

## Total loss

```python
total_train_loss = (
    self.w_coarse * loss_coarse.mean()
    + self.w_fine * loss_fine.mean()
    + self.w_fine1 * loss_fine1.mean()
    + self.w_exact * loss_exact.mean()
)
```

This matches a coarse-to-fine supervision strategy: each stage is guided, not only the final stage.

## Shapes in loss path
- All CD calls expect point-first layout `(B, N, 3)`.
- Therefore model outputs are transposed before CD.

---

# 14. Other Important Methodology Blocks

## Data preprocessing
- Pair loading + optional normalize + optional augmentation.

## Normalization
- GT-centered and GT-scaled pair normalization.

## Cropping / sampling
- FPS used for:
  - encoder capping,
  - seed sampling,
  - GT alignment for stage losses,
  - optional CD metric sampling.

## Feature extraction / encoder
- Conv embedding + staged attention over downsampled sets.

## Context aggregation
- `feat_g` global latent from adaptive max pooling, reused in refinement/exact tail.

## Decoder and upsampling
- Two `PCT_refine` stages.

## Coarse-to-fine generation
- coarse -> fine -> fine1 -> exact.

## Refinement
- residual offsets added to repeated point sets.

## Evaluation metrics
- `cd_p`, `cd_t`, `cd_p_coarse`, `cd_t_coarse`.

## Visualization
- single-case interactive-like plot and bulk saved triplet figures.

---

# 15. Important Calculations and Operations

Operations actually used in this notebook/workflow:
- `transpose`, `permute`, `reshape`, `view`
- `torch.cat`
- `repeat`
- FPS sampling and indexed gather
- adaptive max pooling
- interpolation (`F.interpolate`)
- normalization by centroid/radius
- Chamfer distance nearest-neighbor based computation
- residual additions (`pred + repeated_base`)

Why important: these operations define geometry flow and how point counts and feature channels change.

---

# 16. Parameter and Configuration Explanation

Key parameters from `ExactCfg` (Cell 27):

- `max_encoder_points`:
  - caps encoder input size.
  - affects memory/runtime strongly.

- `encoder_coarse_points`:
  - coarse output points from encoder branch.

- `base_seed_points`:
  - points sent to first refinement stage.

- `step1`, `step2`:
  - refinement upsampling multipliers.

- `exact_target_from` / `exact_target_points`:
  - controls final output point count policy.

- `exact_cd_points`:
  - training-time CD sampling size.

- `metric_cd_points`:
  - eval-time CD sampling size.

- `merge_input_in_final`:
  - optional merge of observed partial points in final output before FPS trim.

- `w_coarse`, `w_fine`, `w_fine1`, `w_exact`:
  - stage loss weights.

- training settings (`lr`, `nepoch`, `optimizer`, `betas`, `batch_size`, `workers`):
  - affect optimization behavior and speed.

---

# 17. How This Notebook Maps to the Methodology

## How `models/PointAttN_Dental.ipynb` maps to the methodology

- **data preparation**: Cell 31 (`PairPointCloudDataset`, normalization, augmentation).
- **feature extraction**: Cell 30 (`PCT_encoderExact` conv + attention stack).
- **encoder logic**: staged FPS + gather + attention.
- **attention mechanism**: `cross_transformer` class.
- **context aggregation**: global pooled `feat_g`.
- **decoder logic**: two `PCT_refine` modules.
- **coarse-to-fine generation**: coarse -> fine -> fine1.
- **refinement**: residual coordinate updates in refiners.
- **output generation**: `ExactSizeHead` exact target point count.
- **loss computation**: multi-stage Chamfer in training branch.
- **metrics**: CD metrics in eval branch.
- **training logic**: generated train script + patch + launch cells.
- **evaluation logic**: generated test script + quant report cells.

---

# 18. Worked Example: One Input Passing Through the Notebook

Let’s choose a realistic example based on configuration defaults:
- `B = 1`
- partial input points `Np = 164610` (large dental scan)
- GT points `Ng = 164610`
- `max_encoder_points = 4096`
- `base_seed_points = 512`
- `step1 = 4`, `step2 = 8`
- `exact_target_from = 'gt'`

## Stage-by-stage walk

1. **Dataset output** from `PairPointCloudDataset`:
   - `partial`: `(164610, 3)`
   - `gt`: `(164610, 3)`

2. **Batching** in DataLoader:
   - `partial`: `(1, 164610, 3)`
   - `gt`: `(1, 164610, 3)`

3. **Transpose before model**:
   - `x = partial.transpose(2,1)` -> `(1, 3, 164610)`

4. **Encoder input cap**:
   - `x_enc = fps_bcn(x, 4096)` -> `(1, 3, 4096)`

5. **Encoder outputs**:
   - `feat_g ~ (1, 512, 1)`
   - `coarse ~ (1, 3, 256)` (using `encoder_coarse_points=256`)

6. **Seed source and seed selection**:
   - concat raw+coarse -> `(1, 3, 164866)`
   - FPS seed to 512 -> `(1, 3, 512)`

7. **Refine stage 1**:
   - output `fine` -> `(1, 3, 2048)`

8. **Refine stage 2**:
   - output `fine1` -> `(1, 3, 16384)`

9. **Exact head**:
   - target points from GT: `164610`
   - output `exact` -> `(1, 3, 164610)`

10. **Training/eval layout for CD**:
   - transpose `exact_bnc` -> `(1, 164610, 3)`

What values mean:
- Coordinates are normalized 3D positions in model coordinate space.
- Later denormalization is not shown in this notebook; saved outputs are model-space predictions.

---

# 19. Exact Progress Values for the Worked Example

## What can be exact from static code reading
From static reading, we can give exact **shape progression** and exact formulas. That is done above.

## About exact numeric intermediate tensor values
To provide true numeric values, code execution is required. I attempted to run minimal Python checks in this environment, but runtime packages are missing here (no `torch`, no `numpy` available in the current shell environment at this moment).

So in this section:
- **real execution values**: not available in this environment now.
- **illustrative exact-number mini-example**: provided below, hand-worked.

## Illustrative example values (hand-worked)

### A) Pair normalization mini-example (illustrative exact arithmetic)
Take:
- `gt = [[1,0,0],[0,1,0],[0,0,1]]`
- `partial = [[2,0,0],[0,2,0],[0,0,2]]`

1. GT centroid:
   - `center = [1/3, 1/3, 1/3]`
2. Centered GT first point:
   - `[1,0,0] - center = [2/3, -1/3, -1/3]`
3. Radius of that point:
   - `sqrt((2/3)^2 + (-1/3)^2 + (-1/3)^2) = sqrt(6/9) = sqrt(2/3)`
4. All 3 centered GT points have same norm, so `scale = sqrt(2/3)`.
5. Normalized GT first point:
   - `[2/3, -1/3, -1/3] / sqrt(2/3)`.

This is exactly the code logic used in `normalize_pair`.

### B) Tiny attention math mini-example (illustrative)
Using simple matrices:
- `Q = [[1,0],[0,1]]`
- `K = [[1,1],[1,0]]`
- `V = [[2,0],[0,2]]`

1. Scores:
- `S = QK^T / sqrt(2)`
- `QK^T = [[1,1],[1,0]]`
- `S = [[0.7071,0.7071],[0.7071,0.0000]]`

2. Softmax row-wise:
- row1 -> `[0.5,0.5]`
- row2 -> `[0.6698,0.3302]`

3. Output:
- row1 = `0.5*[2,0] + 0.5*[0,2] = [1,1]`
- row2 = `0.6698*[2,0] + 0.3302*[0,2] = [1.3396,0.6604]`

These are illustrative values of the same attention principle used by `MultiheadAttention`.

---

# 20. Common Confusion Points

## Confusion 1: “These files already exist in repo”
Snippet:
```python
model_path = REPO_ROOT / "models" / "PointAttN_exact.py"
model_path.write_text(model_code, encoding="utf-8")
```
Why confusing: file is created dynamically during notebook run.
Correct view: notebook is also a code generator.

## Confusion 2: Hidden state across cells
- `REPO_ROOT`, `CFG`, `CFG_PATH`, `CHAMFER_DIR` are created earlier and reused later.
- Running cells out of order can break logic.

## Confusion 3: Shape layout flips
- DataLoader gives `(B,N,3)` but model expects `(B,3,N)`.
- Many bugs come from forgetting this transpose.

## Confusion 4: Multiple CUDA-fix cells
- Cells 16/17/18 are alternative repair pathways in different environments.

## Confusion 5: Training script behavior may change after Cell 36
- Cell 36 patches `train_exact.py` text, so later behavior differs from original generated version.

## Confusion 6: CD can be sampled
- `exact_cd_points` and `metric_cd_points` mean reported/training CD may be on FPS subsets, not always full clouds.

---

# 21. Summary of Full Execution Flow

1. Prepare python/cuda/toolchain environment.
2. Build mm3d ops and Chamfer distance extension.
3. Define experiment configuration dataclass.
4. Generate model/dataset/train/test exact files.
5. Write YAML config.
6. Validate sample data shapes.
7. Patch and run training/resume.
8. Select checkpoint and run test.
9. Save/inspect predicted point clouds.
10. Create visualization images.
11. Compute quantitative metrics and export reports.

---

# 22. Study Notes

## 20 most important things to learn
1. The notebook is end-to-end, not only architecture.
2. It dynamically writes core scripts.
3. Native ops are mandatory (FPS/gather).
4. Chamfer extension is mandatory for loss.
5. Data is paired partial/GT by matching filename stems.
6. Pair normalization is GT-centered and GT-scaled.
7. Model input layout is channel-first `(B,3,N)`.
8. Attention block is `cross_transformer`.
9. Encoder alternates sampling and attention.
10. Decoder has two refinement stages.
11. Exact-size head enforces target point count.
12. Loss uses deep supervision at multiple scales.
13. GT is FPS-aligned per stage for loss.
14. CD computation can be subset-sampled for feasibility.
15. Resume logic is patched later in notebook.
16. Evaluation exports both aggregate and per-case results.
17. Visualization includes single-case and bulk modes.
18. Many cells depend on earlier global variables.
19. Empty cells exist and are safe to ignore.
20. Cell order is essential for reproducibility.

## 20 study questions
1. Why does the notebook compile native code before model training?
2. What breaks if CUDA toolkit version mismatches torch CUDA version?
3. Which variables from Cell 27 control output size behavior?
4. Why is `max_encoder_points` important for dental scans?
5. What is the role of `feat_g` in refinement and exact tail?
6. Why does refinement use residual addition?
7. How does FPS differ from random sampling here?
8. Why are there both `cd_p` and `cd_t`?
9. Why supervise coarse and fine outputs, not only final output?
10. How does `exact_target_from` change behavior?
11. When would `merge_input_in_final=True` help or hurt?
12. Why does the dataset enforce `.npy` pairing?
13. Which cells must be rerun after changing `CFG`?
14. What does Cell 36 add to checkpoint format?
15. How are best checkpoints selected in validation?
16. Where are predictions saved and how are they named?
17. How does Cell 43 differ from `test_exact.py` summary logging?
18. What operations change tensor shape but not semantic meaning?
19. Which operations change geometric values directly?
20. How would you verify whether CD is full-resolution or sampled?

## Next 5 files to inspect and why
1. `models/PointAttN.py` — baseline architecture to compare with exact variant.
2. `utils/model_utils.py` — exact Chamfer metric/loss implementation.
3. `utils/train_utils.py` — meter/checkpoint helpers used in train/test scripts.
4. `train.py` — original training flow and conventions.
5. `dataset.py` — baseline data pipeline for comparison with `dataset_exact.py`.

---

If you want, I can also produce a **“cell run order checklist”** version (short operational playbook) derived from this guide for practical notebook execution.
