# PointAttN vs PointAttN_Dental Notebook — Differences-Only Technical Note

Compared targets:
- Reference: `models/PointAttN.py`
- Compared variant source: `models/PointAttN_Dental.ipynb` (notebook cells that generate `PointAttN_exact.py`, `dataset_exact.py`, `train_exact.py`, `test_exact.py`)

This document is **differences-only** and focuses on precise technical changes.

---

## 1) Executive summary of differences

1. The baseline is a static model module; the notebook is an orchestration workflow that **writes and runs** multiple files.
2. Core attention/refine building blocks are mostly preserved.
3. The notebook path adds a new **exact-size output tail** (`ExactSizeHead`) not in baseline.
4. The notebook path adds **large-scale controls** (`max_encoder_points`, CD sampling caps).
5. Training objective is expanded from baseline 3-term loss to configurable weighted 4-term loss.
6. Data pipeline changes from generic PCN/C3D loaders to a paired dental exact loader.

---

## 2) Architectural differences

### Reference architecture (`models/PointAttN.py`)

```python
self.encoder = PCT_encoder()
self.refine = PCT_refine(ratio=step1)
self.refine1 = PCT_refine(ratio=step2)
```

### Notebook-generated architecture (Cell 30 code string)

```python
self.encoder = PCT_encoderExact(coarse_points=self.encoder_coarse_points)
self.refine = PCT_refine(ratio=self.step1)
self.refine1 = PCT_refine(ratio=self.step2)
self.exact_head = ExactSizeHead(hidden=int(getattr(args, "exact_tail_hidden", 128)))
```

### Diff interpretation
- Preserved: encoder + two refine stages concept.
- Added: `ExactSizeHead` (major functional change).
- Added: configurable encoder coarse count and exact-tail hidden size.

---

## 3) Attention implementation differences

### Reference (`models/PointAttN.py`)

```python
src12 = self.multihead_attn1(query=src1,
                             key=src2,
                             value=src2)[0]
src1 = src1 + self.dropout12(src12)
src1 = self.norm12(src1)
src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
```

### Notebook-generated exact model

```python
src12 = self.multihead_attn1(query=src1, key=src2, value=src2)[0]
src1 = src1 + self.dropout12(src12)
src1 = self.norm12(src1)
src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
```

### Diff interpretation
- No meaningful math difference detected in attention core.
- Same projections, MHA call pattern, residual+norm, FFN.
- Difference is mostly usage context (dental exact pipeline scale/controls).

---

## 4) Feature extraction differences

### Reference encoder key behavior
- Uses ratio-based FPS downsampling and reshape with `N // 8` assumption in decoder seed path.

```python
x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size,self.channel*4,N//8)
```

### Notebook exact encoder key behavior
- Includes comment and code adapting away from rigid size assumptions.
- Reshapes to fixed latent width then interpolates to configurable coarse points.

```python
x2_d = x2_d.reshape(batch_size, self.channel * 4, 256)
if x2_d.shape[-1] != self.coarse_points:
    x2_d = F.interpolate(x2_d, size=self.coarse_points, mode="linear", align_corners=False)
```

### Diff interpretation
- Functional change: output coarse count becomes configurable and less tied to baseline `N` assumptions.

---

## 5) Decoder / refinement differences

### Same
- Two `PCT_refine` stages remain.

### Different
- Reference ends at `fine1`.
- Notebook exact adds final tail:

```python
target_points = self._resolve_target_points(x_raw, gt)
exact = self.exact_head(fine1, feat_g, target_points)
```

And in `ExactSizeHead`:

```python
rep = int(math.ceil(float(target_points) / float(n)))
up = base_points_bcn.repeat(1, 1, rep)
out = self.conv_out(x) + up
out = fps_bcn(out, target_points)
```

### Diff interpretation
- Major functional change: explicit exact point-count enforcement.

---

## 6) Loss and metric differences

### Reference loss path (`models/PointAttN.py`)

```python
loss3, _ = calc_cd(fine1, gt)
loss2, _ = calc_cd(fine, gt_fine1)
loss1, _ = calc_cd(coarse, gt_coarse)
total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()
```

### Notebook exact loss path (generated model)

```python
loss_exact, _ = calc_cd(pred_exact_cd, gt_exact_cd)
loss_fine1, _ = calc_cd(fine1_bnc, gt_fine1)
loss_fine, _ = calc_cd(fine_bnc, gt_fine)
loss_coarse, _ = calc_cd(coarse_bnc, gt_coarse)

total_train_loss = (
    self.w_coarse * loss_coarse.mean()
    + self.w_fine * loss_fine.mean()
    + self.w_fine1 * loss_fine1.mean()
    + self.w_exact * loss_exact.mean()
)
```

### Metric diff
- Reference eval: final `fine1` + coarse metrics.
- Notebook eval: metrics computed on `exact` output (optionally sampled by `metric_cd_points`) plus coarse metrics.

### Shared dependency
- Both use `utils/model_utils.py::calc_cd`.

---

## 7) Data pipeline differences

### Reference (`dataset.py`)
- Uses `PCN_pcd` and `C3D_h5` classes.
- Supports `.pcd` and `.h5`-oriented formats and category mappings.
- Includes upsample/random sample logic around baseline dataset assumptions.

### Notebook-generated dental dataset (`dataset_exact.py` in cell text)
- Uses strict split folder pairing:
  - `<root>/<split>/partial/*.npy`
  - `<root>/<split>/gt/*.npy`
- Pairing by same filename stem.
- Pair normalization by GT center/scale.
- Optional paired augmentation.

### Diff interpretation
- Functional and domain-specific data contract change.

---

## 8) Configuration and hyperparameter differences

### Reference config (`cfgs/PointAttN.yaml`)
- Baseline keys: `dataset`, `num_points`, `lr_decay`, etc.

### Notebook config (`ExactCfg` in cell 27)
Adds key controls not in baseline:
- `max_encoder_points`
- `encoder_coarse_points`
- `base_seed_points`
- `exact_target_from`, `exact_target_points`
- `exact_cd_points`, `metric_cd_points`
- `w_coarse`, `w_fine`, `w_fine1`, `w_exact`
- `use_input_for_seed_sampling`, `merge_input_in_final`

### Diff interpretation
- Major expansion of controllable behavior.

---

## 9) Notebook workflow vs model-file workflow differences

### Reference workflow
- Static files: model + dataset + train script + config.

### Notebook workflow
- Stateful cells:
  - install and build deps
  - patch code
  - generate files
  - run train/test/vis/report

### Diff interpretation
- Notebook offers one-place orchestration but higher hidden-state risk.

---

## 10) Tensor shape differences

### Reference (typical)
- Input `(B,N,3)` -> `(B,3,N)`
- Seed fixed FPS 512
- Final eval output from `fine1`

### Notebook exact
- Input `(B,N,3)` -> `(B,3,N_raw)`
- Optional cap -> `(B,3,N_enc)`
- Seed configurable (`base_seed_points`)
- Final exact tail output `(B,3,N_target)` then transpose for CD

### Practical difference
- Final size in baseline is refinement-derived.
- Final size in notebook exact is explicitly target-controlled.

---

## 11) Output differences

### Reference eval output
```python
{'out1': coarse, 'out2': fine1, 'cd_t_coarse': ..., 'cd_p_coarse': ..., 'cd_p': ..., 'cd_t': ...}
```

### Notebook exact eval output
```python
{'out0': coarse_bnc, 'out1': fine1_bnc, 'out2': exact_bnc,
 'cd_t_coarse': ..., 'cd_p_coarse': ..., 'cd_p': ..., 'cd_t': ...}
```

Notebook adds explicit intermediate `out0/out1/out2` hierarchy and exact final cloud semantics.

---

## 12) Which changes are cosmetic vs functional

### Mostly cosmetic / organizational
- Notebook cell comments and orchestration wrappers.
- Minor formatting differences in attention code line wrapping.

### Functional
- `ExactSizeHead` addition.
- Target-point resolution policy.
- Encoder input capping.
- Weighted expanded loss terms.
- CD sampling controls.
- Dataset contract changes.

---

## 13) Which changes are minor vs major

### Minor
- Small code-style/layout changes.
- Logging/report formatting differences.

### Major
1. Exact-size output architecture addition.
2. Data pipeline switch to paired dental exact format.
3. Expanded objective and config controls.
4. Notebook end-to-end orchestration replacing simple static flow.

---

## 14) Which changes likely affect model behavior the most

1. **ExactSizeHead + target policy** (final output geometry cardinality).
2. **max_encoder_points** (information retained vs compute cost).
3. **exact_cd_points / metric_cd_points** (optimization/evaluation fidelity vs feasibility).
4. **Weighted loss coefficients** (`w_coarse`/`w_fine`/`w_fine1`/`w_exact`).
5. **Dental pair normalization/data contract**.

---

## 15) Code block mapping table

| Reference block (`models/PointAttN.py`) | Notebook counterpart | Diff status |
|---|---|---|
| `cross_transformer` | Cell 30-generated `cross_transformer` | Same/near-same |
| `PCT_refine` | Cell 30-generated `PCT_refine` | Same/near-same |
| `PCT_encoder` | `PCT_encoderExact` | Modified |
| `Model.forward` ending at `fine1` | `Model.forward` with `ExactSizeHead` | Expanded major |
| 3-term unweighted loss | 4-term weighted loss | Expanded major |
| Baseline dataset loaders | `PairPointCloudDataset` exact | Replaced major |
| Static script workflow | Cell-generated script workflow | Reorganized major |

---

## 16) Function-by-function diff

### `cross_transformer.forward`
- Reference: `models/PointAttN.py`
- Notebook exact: generated model
- Diff: negligible.

### `PCT_refine.forward`
- Diff: little to no core logic change.

### `PCT_encoder.forward` vs `PCT_encoderExact.forward`
- Diff: exact version introduces configurable coarse handling and shape adaptation.

### `Model.forward`
- Reference: returns baseline fine output path.
- Notebook exact: adds cap->refine->exact-tail and expanded loss/eval controls.

### `calc_cd`
- Shared function in `utils/model_utils.py`.
- Diff: none (same dependency).

### Dataset `__getitem__`
- Reference: `dataset.py` classes (PCN/C3D style).
- Notebook exact: paired exact dental style.
- Diff: major in file format contract and preprocessing behavior.

---

## 17) Class-by-class diff

### `cross_transformer`
- Same role and nearly same implementation.

### `PCT_refine`
- Same role and nearly same implementation.

### Encoder class
- `PCT_encoder` (reference) vs `PCT_encoderExact` (notebook exact): modified for variable/exact use.

### Top-level `Model`
- Baseline coarse-to-fine output vs exact-size extended output path.

### New class in notebook exact only
- `ExactSizeHead` (no equivalent in reference).

---

## 18) Worked input-flow comparison

Assume one realistic sample:
- partial `(1, 164610, 3)`
- gt `(1, 164610, 3)`

### Reference flow
1. transpose -> `(1,3,164610)`
2. encoder -> coarse/global
3. concat + FPS(512)
4. refine -> refine1
5. final eval output = `fine1`

### Notebook exact flow
1. transpose -> `(1,3,164610)`
2. optional cap (e.g., 4096)
3. encoder exact -> coarse configurable (e.g., 256)
4. refine -> refine1 (e.g., 2048 -> 16384)
5. exact head -> enforce 164610 points
6. final eval output = exact `out2`

### Real vs illustrative numeric values
- Real intermediate numeric activations cannot be provided from static reading alone.
- In this shell environment, runtime imports for torch/numpy are unavailable, so live model numeric execution could not be produced.
- Use illustrative toy attention math only for conceptual numeric understanding.

---

## 19) Common misunderstandings

1. “Attention changed a lot.”
   - Core attention block is almost unchanged.

2. “Only dataset changed.”
   - Architecture and objective changed too (exact tail + weighted loss + caps).

3. “Notebook is just a model variant file.”
   - It is a full build/write/train/test/report workflow.

4. “Output names imply same semantics.”
   - Baseline `out2` is refine output; notebook exact `out2` is exact-tail output.

5. “CD always full-resolution in dental path.”
   - CD can be subset-sampled via config for feasibility.

---

## 20) Study checklist

- [ ] I can identify exactly where `ExactSizeHead` is introduced.
- [ ] I can explain why `max_encoder_points` exists.
- [ ] I can explain reference vs dental final output semantics.
- [ ] I can list all new loss terms in dental exact path.
- [ ] I can explain which parts of attention stayed unchanged.
- [ ] I can describe data format assumptions in both pipelines.
- [ ] I can map baseline config keys vs dental added keys.
- [ ] I can explain why notebook cell order matters.
- [ ] I can trace where metrics are computed in each path.
- [ ] I can separate cosmetic differences from functional ones.
- [ ] I can identify the top 5 behavior-impacting changes.
- [ ] I can map reference code blocks to notebook-generated blocks.
- [ ] I can explain tensor shape flow differences from input to output.
- [ ] I can state where shared dependencies are unchanged (`calc_cd`, mm3d ops).
- [ ] I can explain why this is an adaptation (not total rewrite).
