# PointAttN Dental Notebook — Full Worked Example Teaching Note

Notebook target: `models/PointAttN_Dental.ipynb`

This note focuses on **one full worked example flow** through the notebook pipeline, with explicit stage mapping, shape tracking, and a clear split between:

1. **Real code execution values** (only what was actually executable in this shell)
2. **Simplified illustrative exact values** (small toy tensors with exact arithmetic)

---

## 0) Scope and honesty note

- The notebook itself defines and writes runtime files (`models/PointAttN_exact.py`, `dataset_exact.py`, `train_exact.py`, `test_exact.py`) that are part of the execution path.
- In this shell environment, `torch`/`numpy` are not available, so real numeric intermediate model activations could not be produced from live execution.
- Therefore:
  - **real executed numeric values** = environment checks and deterministic static values only.
  - **exact toy numeric values** = provided separately in a simplified worked example.

---

## 1) One realistic sample input (main worked path)

We choose one realistic dental sample matching the notebook defaults and code logic:

- Batch size: `B = 1`
- Partial points: `Np = 164610` (realistic high-resolution dental input)
- GT points: `Ng = 164610`
- Tensor from DataLoader (before model transpose):
  - `partial`: `(1, 164610, 3)`
  - `gt`: `(1, 164610, 3)`

Why this is realistic:
- Notebook config defaults and comments are designed for very large dental clouds.
- The exact pipeline adds controls (`max_encoder_points`, CD sampling caps) specifically for such scales.

Cells involved:
- Dataset definition: **Cell 31**
- Config defaults: **Cell 27**
- Training/test data loading use: **Cells 32, 33, 41, 43**

Related file called:
- `dataset_exact.py` (written by Cell 31)

---

## 2) Stage-by-stage full pipeline walkthrough (real model path)

## Stage A — Data load and preprocessing

### Code location
- Notebook Cell 31 writes `dataset_exact.py`
- Used by `train_exact.py` / `test_exact.py` and visualization/eval cells

### Key operations

```python
partial = load_points_any(partial_path)
gt = load_points_any(gt_path)

if self.normalize:
    partial, gt = normalize_pair(partial, gt)

if self.augment_train:
    partial, gt = random_pair_transform(partial, gt)

partial = torch.from_numpy(partial)
gt = torch.from_numpy(gt)
return label, partial, gt, stem
```

### Shape/result for our sample
- Loaded arrays: `(164610, 3)` each
- Batched by DataLoader: `(1, 164610, 3)` each

### Role of block
- Guarantees paired partial/GT sample by shared stem.
- Applies normalization consistently to both clouds.

---

## Stage B — Model input layout conversion

### Code location
- Training: Cell 32-generated `train_exact.py`
- Testing: Cell 33-generated `test_exact.py`

### Key operation

```python
inputs = inputs.transpose(2, 1).contiguous()
```

### Shape change
- Before: `(B, N, 3)` = `(1, 164610, 3)`
- After: `(B, 3, N)` = `(1, 3, 164610)`

### Role of block
- PointAttN modules expect channel-first point layout.

---

## Stage C — Encoder input cap and feature extraction

### Code location
- Cell 30-generated file `models/PointAttN_exact.py`
- `Model.forward` + `PCT_encoderExact.forward`

### Key operations

```python
x_raw = x
x_enc = self._maybe_cap_encoder_input(x_raw)  # uses fps_bcn if capped
feat_g, coarse = self.encoder(x_enc)
```

For our default-like config:
- `max_encoder_points = 4096`
- `encoder_coarse_points = 256`

### Shape changes (worked)
1. Input to model: `(1, 3, 164610)`
2. Capped encoder input: `(1, 3, 4096)`
3. Encoder outputs (expected by code design):
   - `feat_g`: approximately `(1, 512, 1)`
   - `coarse`: `(1, 3, 256)`

### Role of block
- Keep attention tractable by capping input point count.
- Produce global context and initial coarse reconstruction.

Cells involved:
- File generation: **Cell 30**
- Use in training/testing: **Cells 32/33/41/43**

Related files called:
- `models/PointAttN_exact.py`
- `utils.mm3d_pn2` functions `furthest_point_sample`, `gather_points`

---

## Stage D — Seed construction + coarse-to-fine refinement

### Code location
- `Model.forward` and `PCT_refine` in `models/PointAttN_exact.py` (Cell 30 output)

### Key operations

```python
seed_source = torch.cat([x_raw, coarse], dim=2)
seed = fps_bcn(seed_source, min(self.base_seed_points, seed_source.shape[-1]))

fine, feat_fine = self.refine(None, seed, feat_g)
fine1, feat_fine1 = self.refine1(feat_fine, fine, feat_g)
```

Config values used:
- `base_seed_points = 512`
- `step1 = 4`
- `step2 = 8`

### Shape changes (worked)
- `seed_source`: `(1, 3, 164610 + 256) = (1, 3, 164866)`
- `seed`: `(1, 3, 512)`
- `fine`: `(1, 3, 512*4) = (1, 3, 2048)`
- `fine1`: `(1, 3, 2048*8) = (1, 3, 16384)`

### Role of block
- Merge observed input geometry + coarse estimate.
- Upsample and refine in two stages (coarse-to-fine).

---

## Stage E — Exact-size output head

### Code location
- `ExactSizeHead.forward` in `models/PointAttN_exact.py`

### Key operations

```python
target_points = self._resolve_target_points(x_raw, gt)
exact = self.exact_head(fine1, feat_g, target_points)
```

If `exact_target_from='gt'`:
- `target_points = gt.shape[1] = 164610`

The head repeats, refines, then FPS-trims:

```python
rep = ceil(target_points / n)
up = base_points_bcn.repeat(1, 1, rep)
out = conv_out(...) + up
out = fps_bcn(out, target_points)
```

### Shape change
- Input to exact head: `(1, 3, 16384)`
- Output from exact head: `(1, 3, 164610)`

### Role of block
- Guarantees final output has exact requested point count.

---

## Stage F — Output formatting, losses, and metrics

### Code location
- `Model.forward` (training/eval branches)
- Chamfer function in `utils/model_utils.py`

### Format conversion

```python
exact_bnc = exact.transpose(1, 2).contiguous()
```

Shape:
- `(1, 3, 164610)` -> `(1, 164610, 3)`

### Loss branch (training)

```python
loss_exact, _ = calc_cd(pred_exact_cd, gt_exact_cd)
loss_fine1, _ = calc_cd(fine1_bnc, gt_fine1)
loss_fine, _ = calc_cd(fine_bnc, gt_fine)
loss_coarse, _ = calc_cd(coarse_bnc, gt_coarse)

total_train_loss = (
    w_coarse * loss_coarse.mean()
    + w_fine * loss_fine.mean()
    + w_fine1 * loss_fine1.mean()
    + w_exact * loss_exact.mean()
)
```

### Metrics branch (eval)

```python
cd_p, cd_t = calc_cd(pred_metric, gt_metric)
cd_p_coarse, cd_t_coarse = calc_cd(coarse_bnc, fps_bnc(gt, coarse_bnc.shape[1]))
```

### Role of block
- Training: optimize coarse + intermediate + exact outputs.
- Evaluation: report final and coarse Chamfer metrics.

Cells involved:
- Model creation: **Cell 30**
- Train loop usage: **Cell 32**
- Test loop usage: **Cell 33**
- Quant eval usage: **Cell 43**

Related file called:
- `utils/model_utils.py::calc_cd`

---

## Stage G — Saving, visualization, and reporting

### Where outputs are saved

1. **Predictions** (`test_exact.py`, Cell 33)
   - folder: `os.path.join(os.path.dirname(args.load_model), 'all')`
   - formats: `.npy` or `.ply`

2. **Single-case visualization** (Cell 41)
   - direct matplotlib display.

3. **Batch triplet visualization** (Cell 42)
   - saves `idxXXXX_<stem>.png` images in a visualization folder.

4. **Quantitative text report** (Cell 43)
   - `quant_eval_exact_<split>.txt`

5. **Report CSV + figures** (Cell 44)
   - `overview_exact.csv`, `per_case_exact.csv`
   - `fig_overview_exact.png`, `fig_cd_p_by_case.png`

---

## 3) Attention block details (worked)

### Implemented in
- Cell 30-generated `models/PointAttN_exact.py`
- Class: `cross_transformer`
- Function: `forward`

### Core attention code

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

### Shape explanation
- Input streams: `[B, C, N]`
- Internal attention format: `[N, B, C]`
- Output back to `[B, C, N]`

### Functional meaning
- `query=src1`, `key/value=src2`: stream 1 reads context from stream 2.
- Residual + FFN improves feature quality while preserving gradient flow.

---

## 4) Loss/metrics location summary

- **Loss computed in model**: `models/PointAttN_exact.py` (generated by Cell 30), training branch of `Model.forward`.
- **Chamfer primitive**: `utils/model_utils.py::calc_cd`.
- **Metrics aggregation in scripts**:
  - test script (Cell 33 file)
  - quantitative eval cell (Cell 43)

---

## 5) Real code execution values vs illustrative values

## 5.1 Real code execution values (available here)

### What was attempted
- Run minimal numeric checks for `torch` and `numpy` in current shell.

### Actual result
- `torch` import failed (`ModuleNotFoundError`).
- `numpy` import failed (`ModuleNotFoundError`).

Therefore **real numeric intermediate model tensors could not be executed in this shell session**.

### What is still real and exact from code
- Cell order, variable dependencies, file paths, and shape formulas are exact from notebook and related files.

---

## 5.2 Simplified illustrative values (exact toy example)

These are toy values to illustrate calculations exactly.

## A) Toy normalization example (exact arithmetic)

Use:
- `gt = [[1,0,0],[0,1,0],[0,0,1]]`
- `partial = [[2,0,0],[0,2,0],[0,0,2]]`

Compute exactly like `normalize_pair`:
1. `center = [1/3, 1/3, 1/3]`
2. `gt_centered[0] = [2/3, -1/3, -1/3]`
3. `scale = sqrt((2/3)^2 + (-1/3)^2 + (-1/3)^2) = sqrt(2/3)`
4. normalized first GT point:
   - `[2/3, -1/3, -1/3] / sqrt(2/3)`

This is an exact hand-worked mirror of notebook dataset normalization logic.

## B) Toy attention exact numbers

Define small matrices:
- `Q = [[1,0],[0,1]]`
- `K = [[1,1],[1,0]]`
- `V = [[2,0],[0,2]]`
- `d = 2`

1. score matrix:
- `QK^T = [[1,1],[1,0]]`
- `S = QK^T / sqrt(2) = [[0.7071,0.7071],[0.7071,0.0000]]`

2. row-wise softmax:
- row1: `[0.5, 0.5]`
- row2: `[0.6698, 0.3302]`

3. output:
- row1: `0.5*[2,0] + 0.5*[0,2] = [1.0000, 1.0000]`
- row2: `0.6698*[2,0] + 0.3302*[0,2] = [1.3396, 0.6604]`

This is a simplified exact numeric analogue of the notebook’s attention mechanism idea.

---

## 6) Notebook cells and called files per stage (quick map)

- Setup/build: Cells **0–22**
- Config root state: Cell **27**
- Write model: Cell **30** -> `models/PointAttN_exact.py`
- Write dataset: Cell **31** -> `dataset_exact.py`
- Write train script: Cell **32** -> `train_exact.py`
- Write test script: Cell **33** -> `test_exact.py`
- Write YAML: Cell **34** -> `cfgs/PointAttN_exact.yaml`
- Data smoke test: Cell **35**
- Resume patch/train launch: Cells **36–37**
- Checkpoint select/test launch: Cells **38–40**
- Visualization: Cells **41–42**
- Quant eval/report: Cells **43–44**

Essential related files called during training/eval:
- `utils/model_utils.py`
- `utils/train_utils.py`
- `utils/mm3d_pn2` ops
- Chamfer extension under `utils/ChamferDistancePytorch`

---

## 7) Final outputs for the worked sample

For one sample in eval path:
- model output dict includes:
  - `out0` (coarse cloud)
  - `out1` (intermediate refined cloud)
  - `out2` (final exact-size cloud)
  - `cd_p`, `cd_t`, `cd_p_coarse`, `cd_t_coarse`

Persisted artifacts (if enabled):
- prediction file in `<ckpt_dir>/all/<stem>.npy` or `.ply`
- visualization image(s)
- quantitative text + CSV + plot files

---

## 8) Teaching recap (single-sample mental model)

Think of the notebook pipeline for one dental sample as:

1. **Read and normalize paired clouds**
2. **Cap huge input for encoder feasibility**
3. **Extract global context + coarse structure**
4. **Refine geometry in two coarse-to-fine steps**
5. **Force exact target point count**
6. **Compute multi-level Chamfer supervision**
7. **Report metrics and save geometric outputs**

That is the full worked-path logic implemented by `models/PointAttN_Dental.ipynb`.
