# PointAttN Dental — Deep Code-to-Method Mapping

This document is a **methodology-focused companion** to `models/PointAttN_Dental.ipynb`.

It focuses only on: **how the notebook’s real code maps to the method** (data prep, encoder, attention, decoder/refinement, loss, metrics, and full data flow).

---

## 1) Exact mapping from notebook sections/cells to methodology blocks

The notebook is mostly code cells. The methodology mapping is:

### A. Environment + reproducible execution foundation
- **Cells 0–22**
- Method block: *infrastructure for reproducible geometric learning pipeline*.
- What it does in method terms:
  - Ensures PyTorch CUDA compatibility.
  - Builds required native kernels for FPS/gather and Chamfer distance.
  - Patches older CUDA/Python compatibility issues.

> Why this is methodological (not just setup): Point sampling and Chamfer are core math operations; if extension builds fail, the model method cannot run.

### B. Experiment configuration definition
- **Cell 27**
- Method block: *single source of truth for experiment semantics*.
- Defines data split names, output-point strategy, model scale controls, and loss weighting.

### C. Core architecture definition (exact-size PointAttN variant)
- **Cell 30** (writes `models/PointAttN_exact.py`)
- Method blocks:
  - Attention mechanism (`cross_transformer`)
  - Encoder feature extraction (`PCT_encoderExact`)
  - Decoder/refinement (`PCT_refine` × 2)
  - Exact-size generation tail (`ExactSizeHead`)
  - Multi-stage loss composition (`Model.forward` training branch)

### D. Dataset implementation
- **Cell 31** (writes `dataset_exact.py`)
- Method blocks:
  - Pairing logic of partial/GT point clouds
  - Pair normalization
  - Optional paired geometric augmentation

### E. Training / validation loops
- **Cell 32** (writes `train_exact.py`)
- **Cell 36** (patches resume/checkpoint behavior)
- **Cell 37** (auto-resume launch logic)
- Method blocks:
  - optimization loop
  - train/val split behavior
  - checkpointing strategy

### F. Test / prediction export
- **Cell 33** (writes `test_exact.py`)
- **Cells 38–40** (choose checkpoint, run test, inspect saved outputs)
- Method blocks:
  - held-out evaluation execution
  - prediction artifact persistence (`.npy`/`.ply`)

### G. Qualitative and quantitative analysis
- **Cells 41–42**: visualization
- **Cell 43**: quantitative per-case + overview metrics
- **Cell 44**: report parsing, CSV/plot generation
- Method blocks:
  - qualitative geometric comparison
  - metric logging / reporting

---

## 2) Data preparation and preprocessing implementation

This logic is created in notebook **Cell 31** (`dataset_exact.py`).

### 2.1 Dataset structure contract

```python
class PairPointCloudDataset(data.Dataset):
    """
    Expected folder structure:
      root/
        train/
          partial/
          gt/
        val/
          partial/
          gt/
        test/
          partial/
          gt/
    """
```

Method meaning:
- The method assumes **paired supervision**: each partial sample has a same-stem GT sample.
- Training and evaluation both rely on this one-to-one pairing.

### 2.2 Strict file matching behavior

```python
PREFERRED_EXT = ".npy"
partial_files = sorted(self.partial_dir.glob(f"*{PREFERRED_EXT}"))
for p in partial_files:
    stem = p.stem
    gt_match = self.gt_dir / f"{stem}{PREFERRED_EXT}"
    if gt_match.exists():
        items.append((stem, p, gt_match))
```

Method meaning:
- Only `.npy` is used by default in this exact pipeline.
- This avoids silent format ambiguity during training.

### 2.3 Pair normalization

```python
def normalize_pair(partial: np.ndarray, gt: np.ndarray):
    center = gt.mean(axis=0, keepdims=True)
    gt_n = gt - center
    partial_n = partial - center

    scale = np.sqrt((gt_n ** 2).sum(axis=1)).max()
    if scale > 1e-8:
        gt_n = gt_n / scale
        partial_n = partial_n / scale
```

Method meaning (plain):
- Use GT centroid as shared center.
- Scale both clouds by GT max radius.
- This puts both clouds in aligned normalized coordinates, helping stable loss optimization.

### 2.4 Optional train-time paired augmentation

```python
def random_pair_transform(partial, gt):
    # axis flips + y-axis rotation + uniform scale
    partial[:, :3] = np.dot(partial[:, :3], trfm_mat.T)
    gt[:, :3] = np.dot(gt[:, :3], trfm_mat.T)
    partial = partial * scale
    gt = gt * scale
```

Method meaning:
- Same transform applied to both partial and GT preserves correspondence relation.

---

## 3) Attention implementation

## 3.1 Where implemented
- Notebook source: **Cell 30** code string.
- Related architecture baseline: `models/PointAttN.py` (same block style).
- Class: `cross_transformer`
- Function: `cross_transformer.forward`

## 3.2 Code snippet

```python
class cross_transformer(nn.Module):
    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)
        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.activation1 = torch.nn.GELU()
        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def forward(self, src1, src2, if_act=False):
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

        return src1
```

## 3.3 Line-by-line explanation
1. `input_proj`: aligns channels into attention dimension.
2. `reshape + permute`: converts from `[B,C,N]` to `[N,B,C]` (PyTorch MHA format).
3. `norm13`: pre-norm before attention.
4. `multihead_attn1(query=src1,key=src2,value=src2)`: each token in `src1` reads context from `src2`.
5. Residual add + dropout + norm stabilizes training.
6. Feed-forward MLP (`linear11 -> GELU -> linear12`) refines each token.
7. Second residual add.
8. Convert back to `[B,C,N]`.

## 3.4 Tensor shape view
- Input: `src1, src2`: `[B, C_in, N]`
- After `input_proj`: `[B, C_attn, N]`
- For MHA: `[N, B, C_attn]`
- After attention + FFN: `[N, B, C_attn]`
- Returned: `[B, C_attn, N]`

## 3.5 Simple math meaning
- Attention computes weighted mixing of value features:
  - `weights = softmax(QK^T / sqrt(d))`
  - `context = weights * V`
- Here, `Q` comes from `src1`, `K,V` from `src2`.
- Intuition: "for each point feature in stream 1, retrieve most relevant context from stream 2."

---

## 4) Feature extraction / encoder implementation

Implemented in `PCT_encoderExact` (Cell 30 generated model).

### Core snippet

```python
x = self.relu(self.conv1(points))
x0 = self.conv2(x)

idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), max(1, N // 4))
x_g0 = gather_points(x0, idx_0)
points_1 = gather_points(points, idx_0)
x1 = self.sa1(x_g0, x0)
x1 = torch.cat([x_g0, x1], dim=1)
x1 = self.sa1_1(x1, x1)

# ... repeat GDP/SFA style stages ...

x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)
```

### Method explanation
- The encoder alternates:
  1. **Sampling (GDP-like)** with FPS to reduce point set while preserving coverage.
  2. **Attention aggregation (SFA-like)** to enrich sampled features with context.
- `x_g` is global latent summary (single token per sample).

### Shape intuition
- Input: `[B,3,N]`
- After stagewise downsampling: point count decreases.
- Feature channels increase by concat operations (`C -> 2C -> 4C -> 8C`).
- Global pooling produces `[B,512,1]` style feature.

---

## 5) Decoder / refinement / point generation implementation

The method uses a staged decoder:
1. coarse cloud from encoder
2. first refinement (`PCT_refine`, ratio=`step1`)
3. second refinement (`PCT_refine`, ratio=`step2`)
4. exact-size tail (`ExactSizeHead`)

### 5.1 Refinement block (`PCT_refine`)

```python
y = self.conv_x1(self.relu(self.conv_x(coarse)))
feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))
y0 = torch.cat([y, feat_g.repeat(1, 1, y.shape[-1])], dim=1)

y1 = self.sa1(y0, y0)
y2 = self.sa2(y1, y1)
y3 = self.sa3(y2, y2)
y3 = self.conv_ps(y3).reshape(batch_size, -1, N * self.ratio)

y_up = y.repeat(1, 1, self.ratio)
y_cat = torch.cat([y3, y_up], dim=1)
y4 = self.conv_delta(y_cat)

x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1, 1, self.ratio)
```

Meaning:
- Merge local coarse geometry + broadcast global context.
- Apply attention-based feature refinement.
- Increase point count by `ratio` and predict residual offsets.
- Add residual to repeated coarse points for stable geometric updates.

### 5.2 Exact-size head (`ExactSizeHead`)

```python
if target_points <= n:
    return fps_bcn(base_points_bcn, target_points)

rep = int(math.ceil(float(target_points) / float(n)))
up = base_points_bcn.repeat(1, 1, rep)

g = self.conv_g(feat_g).repeat(1, 1, up.shape[-1])
p = self.conv_p(up)
x = torch.cat([p, g], dim=1)
x = self.act(self.conv1(x))
x = self.act(self.conv2(x))
out = self.conv_out(x) + up
out = fps_bcn(out, target_points)
```

Meaning:
- If too many points: downsample with FPS.
- If too few points: repeat to exceed target, apply learned residual correction, trim to exact count.

---

## 6) Loss function

## 6.1 Where implemented
- In generated model `Model.forward(..., is_training=True)` from Cell 30.
- Distance primitive from `utils/model_utils.py::calc_cd`.

### Distance primitive snippet (`utils/model_utils.py`)

```python
def calc_cd(output, gt, calc_f1=False):
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, _, _ = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))
    return cd_p, cd_t
```

Plain meaning:
- `dist1`: GT-to-pred nearest-neighbor squared distances.
- `dist2`: pred-to-GT nearest-neighbor squared distances.
- `cd_t`: sum of mean squared distances.
- `cd_p`: mean of point-wise distances (sqrt form).

## 6.2 Training branch snippet and explanation

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

total_train_loss = (
    self.w_coarse * loss_coarse.mean()
    + self.w_fine * loss_fine.mean()
    + self.w_fine1 * loss_fine1.mean()
    + self.w_exact * loss_exact.mean()
)
```

Line-by-line meaning:
1. Optionally FPS-sample final prediction for feasible Chamfer compute.
2. Align GT size to sampled prediction size if needed.
3. Compute exact-branch CD.
4. For intermediate outputs, repeatedly FPS-downsample GT to matching sizes.
5. Compute stage losses at coarse/fine/fine1.
6. Weighted sum gives total training objective.

Interpretation of each term:
- `loss_coarse`: global shape scaffold quality.
- `loss_fine`: first-level detail consistency.
- `loss_fine1`: second-level detail consistency.
- `loss_exact`: final exact-size output alignment.

---

## 7) Metrics and evaluation

### Evaluation outputs from model (`is_training=False`)

```python
return {
    'out0': coarse_bnc,
    'out1': fine1_bnc,
    'out2': exact_bnc,
    'cd_t_coarse': cd_t_coarse,
    'cd_p_coarse': cd_p_coarse,
    'cd_p': cd_p,
    'cd_t': cd_t,
}
```

Metric meaning:
- `cd_p`, `cd_t`: final output quality metrics.
- `cd_*_coarse`: coarse-stage quality metrics.

### Script-level evaluation behavior
- `test_exact.py` (Cell 33): iterates test loader, averages metrics, optionally saves outputs.
- Cell 43: creates per-case metric rows and writes `quant_eval_exact_<split>.txt`.
- Cell 44: parses text -> dataframe -> CSV + bar plots.

Method takeaway:
- Evaluation is both aggregate and case-wise, supporting error localization by sample.

---

## 8) Data flow from raw input to final output

1. Load paired arrays from dataset folders.
2. Optional normalization/augmentation.
3. DataLoader batch gives `partial [B,Np,3]`, `gt [B,Ng,3]`.
4. Transpose partial to model format `x [B,3,Np]`.
5. Optional encoder cap via FPS (`max_encoder_points`).
6. Encoder extracts global feature + coarse cloud.
7. Seed construction from raw input + coarse output.
8. Refine stage 1 upsamples features/points.
9. Refine stage 2 upsamples again.
10. Exact head outputs exactly `target_points` points.
11. During training: compute multi-stage CD losses and backprop.
12. During eval: compute `cd_p/cd_t` metrics and export outputs.

---

## 9) Common misunderstandings

1. **“The notebook only trains a model.”**
   - Actually it also **writes model/data/train/test files**, patches scripts, and generates reports.

2. **“Exact output size is only from decoder ratio math.”**
   - Not true. Final exact size is enforced by `ExactSizeHead` + FPS trim.

3. **“Chamfer is always full-resolution.”**
   - No. `exact_cd_points` and `metric_cd_points` can downsample for computational feasibility.

4. **“GT is used as-is for all stage losses.”**
   - Intermediate stage losses use **FPS-downsampled GT** at matching point counts.

5. **“Any file format works for dataset.”**
   - Exact dataset implementation is strict: preferred `.npy` matching by stem.

6. **“Cell order doesn’t matter much.”**
   - It matters heavily. The notebook stores state and writes files consumed by later cells.

---

## 10) Study checklist

Use this checklist while re-reading `models/PointAttN_Dental.ipynb`:

- [ ] I can explain why cells 0–22 are required for method execution.
- [ ] I can point to where attention is implemented (`cross_transformer`).
- [ ] I can track shape format changes `[B,N,3]` ↔ `[B,3,N]`.
- [ ] I understand why FPS is used in 4 places (encoder cap, seed selection, loss alignment, metric sampling).
- [ ] I can describe encoder stage logic (sample → attention → concat).
- [ ] I can explain two-step refinement and what `step1/step2` do.
- [ ] I understand exact-size head repeat + residual + FPS trim.
- [ ] I can derive target point count source (`gt`/`input`/`config`).
- [ ] I can explain each loss term and why weights exist.
- [ ] I can locate where `calc_cd` computes `cd_p` and `cd_t`.
- [ ] I can explain how validation picks best checkpoints per metric.
- [ ] I can describe difference between test script metrics and quant-report cell output.
- [ ] I know where predictions are saved and in what formats.
- [ ] I can trace one sample from dataset row to saved prediction file.
- [ ] I can explain how Cell 36 changes resume/checkpoint behavior.
- [ ] I can identify optional vs required cells for my own environment.

---

If you want, the next step can be a **single-sample worked tensor walkthrough** (with concrete numbers for each intermediate tensor shape from one fake batch), but this document stays focused on strict code-to-method mapping.
