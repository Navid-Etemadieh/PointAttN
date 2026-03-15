# PointAttN Worked Example Teaching Note (`models/PointAttN.py`)

This note focuses **only** on a full worked example for the code in `models/PointAttN.py`.

It gives:

- one realistic sample input,
- major intermediate tensors,
- important shape changes,
- the role of each block,
- attention block details,
- final outputs,
- and a strict separation between:
  1. real code execution values,
  2. simplified illustrative values.

---

## A. Scope and Ground Rules

- This note is grounded in the real pipeline from `models/PointAttN.py`.
- Shapes and block order are from the actual code.
- Numeric values are split into:
  - **Real execution values** (only if execution is available),
  - **Illustrative toy values** (exact arithmetic, pedagogical).

---

## B. Realistic Sample Input (Shape-Level)

We choose a realistic PCN-style scenario because the code has explicit dataset settings:

- `args.dataset = 'pcn'`
  - so `step1 = 4`, `step2 = 8` in `Model.__init__`.
- Sample batch:
  - partial input `x` shape: `(B, 3, N) = (1, 3, 2048)`.
  - ground truth `gt` shape for training/eval comparisons: `(1, 16384, 3)`.

Why this is realistic:

- The model first creates/refines from a fixed 512-point seed set.
- With PCN ratios `4` then `8`, final point count becomes `512 * 4 * 8 = 16384`.

---

## C. End-to-End Worked Example (Real Pipeline, with Shapes)

Below is the full flow from `Model.forward`.

---

### Step 1) Input enters top-level model

```python
def forward(self, x, gt=None, is_training=True):
    feat_g, coarse = self.encoder(x)
```

- Input `x`: `(1, 3, 2048)`.
- `encoder(x)` returns:
  - `feat_g`: global feature token,
  - `coarse`: initial generated point set.

Block role:
- Encoder extracts hierarchical context and creates first coarse geometry.

---

### Step 2) Inside encoder: initial feature lifting

```python
x = self.relu(self.conv1(points))  # (B,64,N)
x0 = self.conv2(x)                 # (B,64,N) when channel=64
```

Shape path:
- `points`: `(1,3,2048)`
- `conv1`: `(1,64,2048)`
- `conv2`: `(1,64,2048)`

Block role:
- Convert raw xyz into learnable per-point feature channels.

---

### Step 3) Encoder hierarchy, level 1 (GDP + SFA style)

```python
idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
x_g0 = gather_points(x0, idx_0)
points = gather_points(points, idx_0)
x1 = self.sa1(x_g0, x0)
x1 = torch.cat([x_g0, x1], dim=1)
x1 = self.sa1_1(x1, x1)
```

Shape path (with `N=2048`):
- `idx_0`: selects 512 points.
- `x_g0`: `(1,64,512)`.
- downsampled `points`: `(1,3,512)`.
- `sa1(x_g0, x0)`: `(1,64,512)`.
- concat -> `x1`: `(1,128,512)`.
- `sa1_1` keeps `(1,128,512)`.

Block role:
- Use sampled points as queries over denser features, then self-refine fused stage features.

---

### Step 4) Encoder hierarchy, level 2

```python
idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
x_g1 = gather_points(x1, idx_1)
points = gather_points(points, idx_1)
x2 = self.sa2(x_g1, x1)
x2 = torch.cat([x_g1, x2], dim=1)
x2 = self.sa2_1(x2, x2)
```

Shape path:
- `idx_1`: selects 256 points.
- `x_g1`: `(1,128,256)`.
- downsampled `points`: `(1,3,256)`.
- `sa2`: `(1,128,256)`.
- concat -> `(1,256,256)`.
- `sa2_1` keeps `(1,256,256)`.

Block role:
- Deeper context extraction at lower point count, higher channel capacity.

---

### Step 5) Encoder hierarchy, level 3

```python
idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
x_g2 = gather_points(x2, idx_2)
x3 = self.sa3(x_g2, x2)
x3 = torch.cat([x_g2, x3], dim=1)
x3 = self.sa3_1(x3, x3)
```

Shape path:
- `idx_2`: selects 128 points.
- `x_g2`: `(1,256,128)`.
- `sa3`: `(1,256,128)`.
- concat -> `(1,512,128)`.
- `sa3_1` keeps `(1,512,128)`.

Block role:
- Highest-level encoded feature map for global context aggregation.

---

### Step 6) Global token and seed-generation decoder in encoder

```python
x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)
x = self.relu(self.ps_adj(x_g))
x = self.relu(self.ps(x))
x = self.relu(self.ps_refuse(x))
x0_d = self.sa0_d(x, x)
x1_d = self.sa1_d(x0_d, x0_d)
x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size,self.channel*4,N//8)
fine = self.conv_out(self.relu(self.conv_out1(x2_d)))
return x_g, fine
```

Shape path:
- `x_g`: `(1,512,1)` global descriptor.
- decoder branch creates sequence-like seed features (length expanded by transposed conv).
- reshape aligns to `(1,256,256)` because `N//8 = 256`.
- `fine` (encoder output) -> `(1,3,256)`.

Important naming note:
- Encoder returns `fine`, but top-level assigns it to variable `coarse`.

Block role:
- Produce initial coarse coordinates from global context.

---

### Step 7) Build refinement seed set (top-level)

```python
new_x = torch.cat([x,coarse],dim=2)
new_x = gather_points(new_x, furthest_point_sample(new_x.transpose(1, 2).contiguous(), 512))
```

Shape path:
- `x`: `(1,3,2048)`
- `coarse`: `(1,3,256)`
- concat along point dim -> `(1,3,2304)`
- FPS to 512 -> `new_x: (1,3,512)`

Block role:
- Build robust 512-point seed set combining observed and generated geometry.

---

### Step 8) Refinement stage 1 (`ratio=4` for PCN)

```python
fine, feat_fine = self.refine(None, new_x, feat_g)
```

Inside `PCT_refine`:

```python
y = self.conv_x1(self.relu(self.conv_x(coarse)))
feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))
y0 = torch.cat([y,feat_g.repeat(1,1,y.shape[-1])],dim=1)

y1 = self.sa1(y0, y0)
y2 = self.sa2(y1, y1)
y3 = self.sa3(y2, y2)
y3 = self.conv_ps(y3).reshape(batch_size,-1,N*self.ratio)

y_up = y.repeat(1,1,self.ratio)
y_cat = torch.cat([y3,y_up],dim=1)
y4 = self.conv_delta(y_cat)

x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1,1,self.ratio)
```

Shape path for stage 1 (`N=512`, `ratio=4`):
- input coarse: `(1,3,512)`
- output `fine`: `(1,3,2048)`
- returned `feat_fine` (`y3`): `(1,128*4,2048)` i.e., `(1,512,2048)`

Block role:
- Upsample 4x and refine coordinates by residual correction.

---

### Step 9) Refinement stage 2 (`ratio=8` for PCN)

```python
fine1, feat_fine1 = self.refine1(feat_fine, fine, feat_g)
```

Shape path for stage 2 (`N=2048`, `ratio=8`):
- input coarse: `(1,3,2048)`
- output `fine1`: `(1,3,16384)`
- intermediate feature `feat_fine1`: returned but not used later.

Block role:
- Final densification and refinement to target dense output.

---

### Step 10) Output layout for losses/metrics

```python
coarse = coarse.transpose(1, 2).contiguous()
fine = fine.transpose(1, 2).contiguous()
fine1 = fine1.transpose(1, 2).contiguous()
```

Shape path:
- `coarse`: `(1,256,3)`
- `fine`: `(1,2048,3)`
- `fine1`: `(1,16384,3)`

Block role:
- Convert to layout expected in this code path for Chamfer helper calls.

---

### Step 11) Final outputs

#### Training mode

```python
loss3, _ = calc_cd(fine1, gt)
...
loss2, _ = calc_cd(fine, gt_fine1)
...
loss1, _ = calc_cd(coarse, gt_coarse)

total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()
return fine, loss2, total_train_loss
```

Returns:
- stage-1 output `fine` (shape `(1,2048,3)`),
- stage-1 loss tensor,
- total summed loss.

#### Eval mode

```python
return {
  'out1': coarse,
  'out2': fine1,
  'cd_t_coarse': cd_t_coarse,
  'cd_p_coarse': cd_p_coarse,
  'cd_p': cd_p,
  'cd_t': cd_t
}
```

Returns:
- `out1`: coarse `(1,256,3)`
- `out2`: final `(1,16384,3)`
- CD metrics.

---

## D. Attention Block Details (Focused)

### Location
- File: `models/PointAttN.py`
- Class: `cross_transformer`
- Function: `forward`

### Code

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

### Shape-level behavior

If `src1` is `(B, Cq, Nq)` and `src2` is `(B, Ck, Nk)`:
- after projection: both become `(B, C, N*)`
- for attention: `(Nq,B,C)` and `(Nk,B,C)`
- attention output: `(Nq,B,C)`
- return: `(B,C,Nq)`

### Method role

- query tokens in `src1` gather context from key/value tokens in `src2`.
- this enables sampled points to learn from denser feature sets.
- residual + FFN further refine features.

---

## E. Important Shape Changes at a Glance

For realistic PCN example `(B=1, N=2048)`:

1. Input: `(1,3,2048)`
2. Encoder coarse output: `(1,3,256)`
3. Merge + FPS512: `(1,3,512)`
4. Refine stage 1 (`x4`): `(1,3,2048)`
5. Refine stage 2 (`x8`): `(1,3,16384)`
6. Final transposed outputs:
   - coarse `(1,256,3)`
   - fine `(1,2048,3)`
   - fine1 `(1,16384,3)`

---

## F. Exact Values Section (Strict Separation)

## F1) Real code execution values

### Attempted real run

I attempted to run a real forward snippet to extract actual numeric intermediate values.

### Result

Real numeric execution is **not available in this environment** because PyTorch is missing.

Observed runtime error:
- `ModuleNotFoundError: No module named 'torch'`

Therefore:
- exact intermediate numeric values from this repository’s actual model execution cannot be provided here.

---

## F2) Simplified illustrative values (exact arithmetic, toy example)

This toy example is only to illustrate attention arithmetic exactly.

Given:
- query `q = [1, 0]`
- keys `k1=[1,0], k2=[0,1]`
- values `v1=[10,1], v2=[2,8]`

Scores (dot-product toy):
- `s1 = q·k1 = 1`
- `s2 = q·k2 = 0`

Softmax weights:
- `w1 = e^1 / (e^1 + e^0) = 2.718 / 3.718 = 0.731` (approx)
- `w2 = e^0 / (e^1 + e^0) = 1 / 3.718 = 0.269` (approx)

Output:
- `out = w1*v1 + w2*v2`
- first channel: `0.731*10 + 0.269*2 = 7.31 + 0.538 = 7.848`
- second channel: `0.731*1 + 0.269*8 = 0.731 + 2.152 = 2.883`
- so `out ≈ [7.848, 2.883]`

Interpretation:
- query matches `k1` more, so output is biased toward `v1`.
- this mirrors attention behavior in `cross_transformer`.

Label:
- These are **illustrative toy values**, not actual model-run values.

---

## G. Quick Recap (Worked Example)

- One realistic input `(1,3,2048)` passes through encoder + two refinement stages.
- Point counts move roughly: `2048 -> 256 -> 512 seed -> 2048 -> 16384`.
- Attention is used repeatedly for context transfer and self-refinement.
- Final outputs are coarse and dense predictions, with multi-scale Chamfer supervision in training.
- Real numeric internals were not executable here (missing torch), so exact toy arithmetic was provided separately and clearly labeled.

