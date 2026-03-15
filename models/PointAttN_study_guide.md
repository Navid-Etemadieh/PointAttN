# PointAttN Code Study Guide (`models/PointAttN.py`)

> This is a teacher-style, beginner-friendly study reference for the file `models/PointAttN.py`.
> Everything below is based on reading the actual code in this repository.

---

## 1. File Overview

`models/PointAttN.py` defines the **main PointAttN network** used for point cloud completion. The model takes a partial point cloud and predicts a more complete point cloud in stages.

In this project pipeline, this file exists because it is the core model definition that:

- builds the encoder (`PCT_encoder`) to extract global geometric context,
- builds refinement modules (`PCT_refine`) to progressively upsample/refine points,
- computes training losses (Chamfer Distance terms) when `is_training=True`,
- returns evaluation metrics and outputs when `is_training=False`.

So, this file is not just architecture; it also includes **training-time objective composition** inside `Model.forward`.

The main module in this file is class `Model`, which glues together:

1. encoder output (`coarse`),
2. first refinement (`fine`),
3. second refinement (`fine1`, final dense output),
4. multi-stage Chamfer losses.

---

## 2. High-Level Summary of the File

### Main classes

- `cross_transformer`
  - A transformer-like cross-attention block (PyTorch `nn.MultiheadAttention`) with residual + FFN.
- `PCT_encoder`
  - Hierarchical encoder that downsamples via FPS, applies cross/self attention repeatedly, then generates a coarse point set.
- `PCT_refine`
  - Refinement/upsampling block that expands point count by a given `ratio` and predicts coordinate offsets.
- `Model`
  - Top-level network; picks stage ratios by dataset, runs encoder + two refinement stages, computes losses.

### Main functions

- Every class has `forward`; the top-level behavior is in `Model.forward`.
- Loss helper `calc_cd` is imported from `utils/model_utils.py`.

### Overall data flow

`input partial points -> encoder coarse output -> concatenate with input -> FPS to 512 seeds -> refine stage 1 -> refine stage 2 -> final prediction`

### Major stages

1. local feature projection,
2. hierarchical attention with FPS-guided pooling,
3. global feature pooling and seed generation,
4. iterative refinement/upsampling,
5. multi-scale Chamfer losses.

---

## 3. Imports and Dependencies

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import *
from utils.mm3d_pn2 import furthest_point_sample, gather_points
```

### Important imports and their roles

- `torch`, `torch.nn`, `torch.nn.functional`: core PyTorch ops, layers, pooling.
- `nn.MultiheadAttention`, `nn.LayerNorm`, `nn.Conv1d`, `nn.ConvTranspose1d`, `nn.Linear`: used to build attention blocks and feature decoders.
- `calc_cd` (from `utils/model_utils.py`): Chamfer Distance computation for training/eval.
- `furthest_point_sample`, `gather_points` (from `utils/mm3d_pn2`): geometric subsampling and indexing.

### PyTorch imports

- `torch`, `torch.nn as nn`, `torch.nn.functional as F` are PyTorch.

### Local project imports

- `utils/model_utils.py`
- `utils/mm3d_pn2/__init__.py` (exports CUDA ops)

### Next local files to inspect

1. `utils/model_utils.py` (defines `calc_cd` with Chamfer distance).
2. `utils/mm3d_pn2/__init__.py` and underlying op files for FPS/gather behavior.
3. `train.py` (how outputs/losses are consumed).
4. `test_pcn.py` / `test_c3d.py` (how inference outputs are saved/evaluated).
5. dataset files (`dataset.py`) for exact input shape expectations.

---

## 4. File-by-File Annotation for This File

Below, we break `models/PointAttN.py` into logical sections.

### Section A: Cross-attention block

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
```

**Explanation (plain language):**
- This class first projects incoming channels with 1x1 conv (`input_proj`).
- Then it applies cross-attention (`query=src1`, `key/value=src2`).
- Then it does a feed-forward network (Linear -> GELU -> Linear).
- It has residual connections and LayerNorm, similar to transformer blocks.

**Purpose:** reusable attention unit for both cross- and self-attention style usage.

**Connection to next block:** this block is heavily reused inside both encoder and refiner.

---

### Section B: `cross_transformer.forward`

```python
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

**Plain explanation:**
- Input starts as `(B, C, N)`.
- It is permuted to `(N, B, C)` because `nn.MultiheadAttention` expects sequence-first by default.
- Attention output is added back (residual), then normalized.
- FFN output is added back (second residual).
- Finally, shape returns to `(B, C, N)`.

**Important note:** `if_act` exists but is not used in this implementation.

---

### Section C: `PCT_refine` constructor

```python
class PCT_refine(nn.Module):
    def __init__(self, channel=128, ratio=1):
        ...
        self.sa1 = cross_transformer(channel*2, 512)
        self.sa2 = cross_transformer(512, 512)
        self.sa3 = cross_transformer(512, channel*ratio)
        ...
        self.conv_delta = nn.Conv1d(channel * 2, channel*1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel*ratio, channel*ratio, kernel_size=1)
```

**Plain explanation:**
- This module refines/upsamples coordinates by factor `ratio`.
- It creates features from coordinates + global feature.
- It applies three attention blocks.
- It predicts offsets and adds them to repeated coarse points.

**Purpose:** convert coarse points into denser, refined points.

---

### Section D: `PCT_refine.forward`

```python
def forward(self, x, coarse, feat_g):
    batch_size, _, N = coarse.size()

    y = self.conv_x1(self.relu(self.conv_x(coarse)))
    feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))
    y0 = torch.cat([y, feat_g.repeat(1,1,y.shape[-1])], dim=1)

    y1 = self.sa1(y0, y0)
    y2 = self.sa2(y1, y1)
    y3 = self.sa3(y2, y2)
    y3 = self.conv_ps(y3).reshape(batch_size, -1, N*self.ratio)

    y_up = y.repeat(1,1,self.ratio)
    y_cat = torch.cat([y3, y_up], dim=1)
    y4 = self.conv_delta(y_cat)

    x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1,1,self.ratio)

    return x, y3
```

**Plain explanation:**
- `coarse` xyz is encoded into feature `y`.
- Global latent `feat_g` is projected and repeated across points.
- Both are fused, then self-attention stacks refine features.
- Point count is increased (`reshape` to `N*ratio`).
- Final output is **residual xyz**: `predicted_offset + repeated_coarse`.

**Why this matters:** residual prediction stabilizes coordinate refinement.

---

### Section E: `PCT_encoder` constructor

```python
class PCT_encoder(nn.Module):
    def __init__(self, channel=64):
        ...
        self.sa1 = cross_transformer(channel, channel)
        self.sa1_1 = cross_transformer(channel*2, channel*2)
        self.sa2 = cross_transformer(channel*2, channel*2)
        self.sa2_1 = cross_transformer(channel*4, channel*4)
        self.sa3 = cross_transformer(channel*4, channel*4)
        self.sa3_1 = cross_transformer(channel*8, channel*8)
        ...
        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
```

**Plain explanation:**
- Encoder uses multiple attention blocks at increasing channel widths.
- It alternates geometric downsampling (FPS) and attention.
- Then it decodes from global pooled feature with transposed conv to seed points.

---

### Section F: `PCT_encoder.forward`

```python
def forward(self, points):
    batch_size, _, N = points.size()
    x = self.relu(self.conv1(points))
    x0 = self.conv2(x)

    idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
    x_g0 = gather_points(x0, idx_0)
    points = gather_points(points, idx_0)
    x1 = self.sa1(x_g0, x0).contiguous()
    x1 = torch.cat([x_g0, x1], dim=1)
    x1 = self.sa1_1(x1, x1).contiguous()

    idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
    x_g1 = gather_points(x1, idx_1)
    points = gather_points(points, idx_1)
    x2 = self.sa2(x_g1, x1).contiguous()
    x2 = torch.cat([x_g1, x2], dim=1)
    x2 = self.sa2_1(x2, x2).contiguous()

    idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
    x_g2 = gather_points(x2, idx_2)
    x3 = self.sa3(x_g2, x2).contiguous()
    x3 = torch.cat([x_g2, x3], dim=1)
    x3 = self.sa3_1(x3, x3).contiguous()

    x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)
    x = self.relu(self.ps_adj(x_g))
    x = self.relu(self.ps(x))
    x = self.relu(self.ps_refuse(x))

    x0_d = (self.sa0_d(x, x))
    x1_d = (self.sa1_d(x0_d, x0_d))
    x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size, self.channel*4, N//8)

    fine = self.conv_out(self.relu(self.conv_out1(x2_d)))

    return x_g, fine
```

**Plain explanation:**
- The encoder progressively reduces points (`N -> N/4 -> N/8 -> N/16`) using FPS.
- At each stage, sampled features attend to denser features (cross-attention), then self-attention fuses.
- Global descriptor `x_g` is max-pooled.
- A seed generator branch upsamples from global descriptor and outputs coarse xyz (`fine` variable name inside encoder, but conceptually coarse output for top-level model).

**Important naming note:** `fine` returned by encoder is used as `coarse` in `Model.forward`.

---

### Section G: Top-level `Model`

```python
class Model(nn.Module):
    def __init__(self, args):
        if args.dataset == 'pcn':
            step1 = 4
            step2 = 8
        elif args.dataset == 'c3d':
            step1 = 1
            step2 = 4
        ...
        self.encoder = PCT_encoder()
        self.refine = PCT_refine(ratio=step1)
        self.refine1 = PCT_refine(ratio=step2)
```

**Plain explanation:**
- Dataset controls upsampling ratios.
- For PCN, total refinement factor is larger (`4` then `8`).
- For C3D, first stage may keep size (`1`) then upsample by `4`.

---

### Section H: `Model.forward` with training/eval behavior

```python
def forward(self, x, gt=None, is_training=True):
    feat_g, coarse = self.encoder(x)

    new_x = torch.cat([x,coarse],dim=2)
    new_x = gather_points(new_x, furthest_point_sample(new_x.transpose(1, 2).contiguous(), 512))

    fine, feat_fine = self.refine(None, new_x, feat_g)
    fine1, feat_fine1 = self.refine1(feat_fine, fine, feat_g)

    coarse = coarse.transpose(1, 2).contiguous()
    fine = fine.transpose(1, 2).contiguous()
    fine1 = fine1.transpose(1, 2).contiguous()

    if is_training:
        loss3, _ = calc_cd(fine1, gt)
        gt_fine1 = gather_points(gt.transpose(1, 2).contiguous(), furthest_point_sample(gt, fine.shape[1])).transpose(1, 2).contiguous()

        loss2, _ = calc_cd(fine, gt_fine1)
        gt_coarse = gather_points(gt_fine1.transpose(1, 2).contiguous(), furthest_point_sample(gt_fine1, coarse.shape[1])).transpose(1, 2).contiguous()

        loss1, _ = calc_cd(coarse, gt_coarse)

        total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()

        return fine, loss2, total_train_loss
    else:
        cd_p, cd_t = calc_cd(fine1, gt)
        cd_p_coarse, cd_t_coarse = calc_cd(coarse, gt)

        return {'out1': coarse, 'out2': fine1, ...}
```

**Plain explanation:**
- Runs three output scales: coarse, fine, final fine1.
- During training, builds multi-stage GT subsets via FPS from GT, then computes 3 Chamfer losses.
- During eval, reports Chamfer metrics for coarse and final outputs.

---

## 5. Class-by-Class Explanation

### `cross_transformer`
- **Purpose:** feature interaction block using attention.
- **Why exists:** core reusable operator for both hierarchical encoder and refinement.
- **Constructor params:**
  - `d_model`: input channels before projection.
  - `d_model_out`: projected channel size and attention embedding size.
  - `nhead`: number of attention heads.
  - `dim_feedforward`: hidden width of FFN.
  - `dropout`: dropout probability.
- **Internal layers:** `input_proj`, `MultiheadAttention`, FFN, two LayerNorms, residual/dropout.
- **Returns:** tensor `(B, d_model_out, Nq)`.

### `PCT_encoder`
- **Purpose:** encode partial cloud into global latent + coarse completion.
- **Why exists:** creates context-rich representation and initial point generation.
- **Key params:** `channel=64` base width.
- **Internal modules:** input convs, staged attention blocks, seed-generation transposed conv branch.
- **Returns:**
  - `x_g` global feature `(B, 512, 1)` (inferred from channel progression),
  - `fine` coarse xyz `(B, 3, N/8)`.

### `PCT_refine`
- **Purpose:** upsample and refine coordinates.
- **Why exists:** coarse-to-fine improvement.
- **Params:**
  - `channel` feature width,
  - `ratio` upsampling multiplier.
- **Internal modules:** point encoder convs, attention stack (`sa1..sa3`), point-shuffle style reshape, delta head.
- **Returns:**
  - refined xyz `(B,3,N*ratio)`,
  - intermediate feature `(B,channel*ratio,N*ratio)`.

### `Model`
- **Purpose:** full training/eval model wrapper.
- **Why exists:** one callable entry point for scripts.
- **Params:** `args` with `dataset` controlling ratios.
- **Returns (training):** `(fine, loss2, total_train_loss)`.
- **Returns (eval):** dict with coarse/final outputs + CD metrics.

---

## 6. Function-by-Function Explanation

### `cross_transformer.with_pos_embed`
- Adds positional embedding if provided; currently not used in this file.

### `cross_transformer.forward(src1, src2, if_act=False)`
- `src1`: query source `(B,C1,N1)`.
- `src2`: key/value source `(B,C2,N2)`.
- Projects both to `d_model_out`, attends `src1` over `src2`, applies FFN.

### `PCT_encoder.forward(points)`
- Input: partial cloud `(B,3,N)`.
- Output: global latent + coarse point cloud.
- Internals: FPS + gather + attention + pooling + seed generation.

### `PCT_refine.forward(x, coarse, feat_g)`
- `x` is unused here (passed as `None` in top call).
- `coarse`: input points to refine `(B,3,N)`.
- `feat_g`: global feature `(B,512,1)`.
- Returns refined coordinates and intermediate features.

### `Model.forward(x, gt=None, is_training=True)`
- Runs complete pipeline.
- If training: computes three CD losses (`loss1` coarse, `loss2` intermediate fine, `loss3` final fine).
- If eval: returns outputs and metrics.

### Important imported helper

`calc_cd` in `utils/model_utils.py` uses `chamfer_3DDist()` to compute:
- `cd_p` = mean point-wise Euclidean Chamfer,
- `cd_t` = mean squared-distance Chamfer.

---

## 7. Forward Pass Walkthrough (Numbered)

1. **Input** partial point cloud `x` (shape `(B,3,N)`).
2. `encoder(x)` extracts global descriptor `feat_g` and a coarse output.
3. Concatenate original input and coarse points along point dimension.
4. Use FPS to pick exactly 512 seed points from concatenation.
5. `refine` stage upsamples/refines these seeds (ratio depends on dataset).
6. `refine1` stage further upsamples/refines the stage-1 output.
7. Transpose outputs to `(B, num_points, 3)` for CD computation.
8. If training: compute multi-scale CD losses with FPS-sampled GT subsets.
9. Return training tuple or eval dictionary.

---

## 8. Tensor Shape Tracking

Let input be `(B,3,N)`.

- After `conv1`: `(B,64,N)`.
- After `conv2`: `(B,64,N)` because `channel=64` default.
- After first FPS (`N/4`) + gather: feature/query approx `(B,64,N/4)`.
- After concat in stage1: `(B,128,N/4)`.
- After stage2 concat: `(B,256,N/8)`.
- After stage3 concat: `(B,512,N/16)`.
- Global pool `adaptive_max_pool1d(...,1)`: `(B,512,1)`.
- Seed generator branch + reshape in encoder ends at `x2_d`: `(B,256,N/8)`.
- Encoder xyz output (`coarse`): `(B,3,N/8)`.

Top-level:
- `new_x = cat([x, coarse], dim=2)` -> `(B,3,N + N/8)`.
- FPS to 512 => `(B,3,512)`.

Refinement stage with ratio `r`:
- Input coarse `(B,3,M)`.
- Output refined `(B,3,M*r)`.

So for PCN (inferred typical ratios):
- stage1 ratio 4: `512 -> 2048`.
- stage2 ratio 8: `2048 -> 16384` final.

(Exact `N` depends on dataset input size; these counts follow code ratios.)

---

## 9. Attention Mechanism Explanation

### Where attention is implemented

In class `cross_transformer`, inside `forward`, using:

```python
src12 = self.multihead_attn1(query=src1, key=src2, value=src2)[0]
```

### Line-by-line idea

1. `src1/src2` projected to same channel size.
2. Converted to `(sequence_len, batch, embed_dim)`.
3. LayerNorm applied.
4. `query=src1`, `key=value=src2`:
   - each query point in `src1` attends over all points in `src2`.
5. Attention output added back (residual).
6. FFN refines attended features.
7. Convert back to `(B,C,N)`.

### Mathematical intuition (simple)

- Each point feature asks: “which points in the other set are relevant to me?”
- Similarity scores create weights.
- Weighted sum of value features forms context-enhanced feature.
- Residual keeps original information.

### Shape view

- Query length `Nq`, key length `Nk`.
- Input to attention: `(Nq,B,C)`, `(Nk,B,C)`.
- Output: `(Nq,B,C)` -> back to `(B,C,Nq)`.

### Match to PointAttN methodology

The method relies on attention-based feature interaction and aggregation; this block is exactly the code mechanism enabling that.

---

## 10. Decoder / Upsampling / Refinement Explanation

Yes, this file contains both decoder-like seed generation and refinement.

### Encoder-side seed generation
- In `PCT_encoder.forward` after global pooling, `ConvTranspose1d` (`self.ps`) expands sequence length, then attention blocks decode features into xyz (`conv_out1`, `conv_out`).

### Refinement blocks (`PCT_refine`)
- Input: coarse points + global feature.
- Attention stack builds rich point features.
- `reshape(batch,-1,N*ratio)` upsamples point count.
- Final coordinates are predicted as offsets added to repeated base points.

So this file clearly implements coarse-to-fine generation + refinement.

---

## 11. Loss Function / Training Objective Explanation

Loss is **partly implemented in this file** (`Model.forward`, training branch).

- `calc_cd` is called three times:
  - `loss1`: coarse vs downsampled GT,
  - `loss2`: first refined output vs medium GT,
  - `loss3`: final refined output vs full GT.
- `total_train_loss = mean(loss1)+mean(loss2)+mean(loss3)`.

`calc_cd` itself is defined in `utils/model_utils.py`, not in this file.

How outputs connect to training:
- multi-stage supervision encourages geometry quality at each resolution.
- returned `loss2` is also logged as “fine loss” in `train.py`.

---

## 12. Inputs and Outputs of the Whole File

### Expected inputs

Top-level `Model.forward`:
- `x`: partial point cloud `(B,3,N)`.
- `gt`: ground-truth complete cloud `(B,Ng,3)` in training/eval metric mode.
- `is_training`: bool.

### Output format

- If `is_training=True`:
  - `fine` (stage-1 refined points, transposed to `(B,points,3)`),
  - `loss2` (per-batch vector),
  - `total_train_loss`.
- If `is_training=False`:
  - dict with:
    - `out1` coarse `(B,coarse_points,3)`,
    - `out2` final `(B,final_points,3)`,
    - `cd_t_coarse`, `cd_p_coarse`, `cd_p`, `cd_t`.

---

## 13. Important Calculations and Operations

Operations actually used here:

- `transpose/permute`: switch between `(B,C,N)` and `(N,B,C)` for attention.
- `reshape`: expand sequence length after feature projection in refiner.
- `cat`: fuse features or combine point sets.
- `repeat`: tile global feature or base coordinates for upsampling.
- `gather_points` + FPS indexing: geometric downsampling/sampling.
- `adaptive_max_pool1d`: get global descriptor.
- residual additions: `x + block(x)` pattern.
- point-wise conv (`Conv1d kernel=1`): channel mixing per point.
- `ConvTranspose1d`: sequence-length expansion in seed generator.

---

## 14. How `models/PointAttN.py` maps to the methodology

- **Feature extraction:** `conv1`, `conv2` in `PCT_encoder`.
- **Encoder logic:** staged FPS + cross/self attention (`sa1..sa3_1`).
- **Attention mechanism:** `cross_transformer` (`MultiheadAttention`).
- **Context aggregation:** global max pooling to `x_g`.
- **Decoder logic:** seed generator (`ps_adj -> ps -> ps_refuse -> sa*_d -> conv_out`).
- **Coarse-to-fine generation:** encoder output -> `refine` -> `refine1`.
- **Refinement:** residual offset prediction in `PCT_refine`.
- **Output generation:** final `fine1` and coarse `out1`.

---

## 15. Common Confusion Points

### Confusion 1: variable name `fine` inside encoder

```python
fine = self.conv_out(self.relu(self.conv_out1(x2_d)))
return x_g, fine
```

Why confusing: top-level uses this as `coarse`.

Correct interpretation: encoder returns an initial/coarse xyz set even though local name is `fine`.

### Confusion 2: unused argument `x` in `PCT_refine.forward`

```python
def forward(self, x, coarse, feat_g):
```

Why confusing: `x` is passed but not used.

Correct interpretation: current implementation only uses `coarse` and `feat_g`; `x` may be legacy interface.

### Confusion 3: `if_act` argument unused

`cross_transformer.forward(..., if_act=False)` ignores `if_act`.

Correct interpretation: placeholder parameter; no branch logic.

### Confusion 4: GT sampling style

```python
gt_fine1 = gather_points(gt.transpose(1,2), furthest_point_sample(gt, fine.shape[1]))
```

Why confusing: FPS runs on GT directly in `(B,N,3)` format, while `gather_points` expects channel-first source.

Correct interpretation: code transposes only for `gather_points`; indexing comes from original point format.

---

## 16. Simple Examples

### Example A: one shape change
- Suppose `coarse` is `(B,3,512)` and `ratio=4`.
- `coarse.repeat(1,1,4)` becomes `(B,3,2048)`.
- Refiner predicts offsets of same shape and adds them -> refined `(B,3,2048)`.

### Example B: attention perspective
- `src1=(B,64,256)`, `src2=(B,64,1024)`.
- Each of 256 query points aggregates information from 1024 key/value points.
- Output remains 256 points but with context-enhanced features.

### Example C: training outputs
- Model returns `(fine, loss2, total_loss)`.
- `fine` is stage-1 prediction used for intermediate supervision and logging.

---

## 17. Summary of Full Execution Flow

1. Read partial input points.
2. Encode with hierarchical FPS + attention.
3. Produce global latent and coarse point set.
4. Merge input + coarse and sample 512 seeds.
5. Run first refinement/upsampling stage.
6. Run second refinement/upsampling stage.
7. Convert outputs to `(B,N,3)`.
8. Compute multi-stage Chamfer losses (training) or metrics (eval).
9. Return outputs for optimization or evaluation.

---

## 18. Study Notes

### 15 most important things to learn

1. `cross_transformer` is reused everywhere.
2. Attention operates on sequence-first tensors.
3. Encoder alternates FPS and attention.
4. FPS reduces points while widening channels.
5. Concatenation doubles channels at each stage.
6. Global max pooling yields compact latent.
7. Seed generator uses transposed convolution.
8. Encoder output is coarse xyz.
9. Refinement is residual coordinate prediction.
10. `ratio` controls upsampling factor.
11. Two refinement stages are chained.
12. Training uses three Chamfer terms.
13. GT is downsampled by FPS for intermediate losses.
14. Eval returns both coarse and final metrics.
15. Dataset type changes refinement ratios.

### 15 study questions

1. Why is attention used instead of only MLPs?
2. What does cross-attention query/key choice imply?
3. Why apply LayerNorm before attention?
4. Why concatenate sampled and attended features?
5. How does FPS improve geometric coverage?
6. Why global max pool to a single token?
7. What does `ConvTranspose1d(...,128)` do to sequence length?
8. Why predict offsets, not absolute points?
9. How does `ratio` affect memory/time?
10. Why supervise multiple scales?
11. Why are GT subsets created with FPS?
12. Why return stage-1 `fine` during training?
13. What changes between PCN and C3D settings?
14. Which parts depend on CUDA custom ops?
15. Where could numerical instability occur?

### Next 5 files to inspect

1. `utils/model_utils.py` — Chamfer loss details (`calc_cd`).
2. `utils/mm3d_pn2/__init__.py` — where FPS/gather are exported from.
3. `utils/mm3d_pn2/ops/furthest_point_sample/furthest_point_sample.py` — FPS op behavior.
4. `utils/mm3d_pn2/ops/gather_points/gather_points.py` — gather indexing semantics.
5. `train.py` — exact training loop and how returned losses are used.

---

## Notes on inferred points

- Exact output point counts depend on input `N` and dataset conventions; examples like `512 -> 2048 -> 16384` are inferred from visible ratios and sampling code.
- The paper-level terminology mapping is inferred from code structure (e.g., comments `GDP`, `SFA`, and staged architecture), not from explicit equations in this file.
