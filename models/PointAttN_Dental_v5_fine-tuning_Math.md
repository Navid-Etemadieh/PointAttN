# Mathematical Foundations of PointAttN_Dental_v5_fine-tuning

## 1. Purpose of the model in mathematical terms

The notebook defines an **exact-size point-cloud completion** model for dental geometry. Its main supervised mapping is:

\[
\mathcal{F}_\theta:\; \mathbb{R}^{B\times 3\times N_{in}} \rightarrow \mathbb{R}^{B\times N_{out}\times 3}
\]

where each sample is a partial point set and the target is a complete point set.

- Input partial cloud (batch element \(b\)): \(X^{(b)}=\{x_i\}_{i=1}^{N_{in}}\subset\mathbb{R}^3\).
- Ground-truth complete cloud: \(Y^{(b)}=\{y_j\}_{j=1}^{N_{gt}}\subset\mathbb{R}^3\).
- Prediction: \(\hat Y^{(b)}=\{\hat y_k\}_{k=1}^{N_{out}}\subset\mathbb{R}^3\), with \(N_{out}\) selected to match a rule (`gt`, `input`, or fixed config count).

Training objective is weighted multi-stage Chamfer minimization:
\[
\min_\theta\;\mathbb{E}_{(X,Y)}\big[\lambda_c\mathcal L_c+\lambda_f\mathcal L_f+\lambda_{f1}\mathcal L_{f1}+\lambda_e\mathcal L_e\big].
\]

Fine-tuning (as implemented in this notebook) means continuing optimization from pretrained parameters (`load_model`) with the same architecture and loss family, but adapted to this dataset/split/hyperparameter regime.

---

## 2. Mathematical notation

- \(B\): batch size.
- \(N_{in}\): input partial point count.
- \(N_{gt}\): ground-truth count.
- \(N_{enc}\): encoder input count after optional cap (`max_encoder_points`).
- \(N_c\): encoder coarse output count (`encoder_coarse_points`).
- \(N_s\): base seed count (`base_seed_points`).
- \(r_1, r_2\): refinement ratios (`step1`, `step2`).
- \(N_{f}=N_s r_1\), \(N_{f1}=N_s r_1 r_2\).
- \(N_t\): exact target count resolved by `exact_target_from`.
- \(X\in\mathbb{R}^{B\times 3\times N}\): BCN coordinate tensor.
- \(\tilde X\in\mathbb{R}^{B\times N\times 3}\): BNC coordinate tensor.
- \(G\in\mathbb{R}^{B\times C_g\times 1}\): global latent feature (encoder pooled token).
- \(Q,K,V\): attention query/key/value tensors.
- \(\text{FPS}(\cdot, n)\): furthest-point sampling operator selecting \(n\) points.
- \(\hat Y_c,\hat Y_f,\hat Y_{f1},\hat Y_e\): coarse/fine/finer/exact predictions.
- \(\mathcal L_c,\mathcal L_f,\mathcal L_{f1},\mathcal L_e\): corresponding loss terms.

---

## 3. Geometric preprocessing and point-cloud representation

### 3.1 Representation
Each cloud is a set of 3D points only (xyz), loaded from `.npy` (or `.ply` fallback in loader utility).

### 3.2 Pair normalization
When `normalize_pair=True`, normalization uses **ground-truth statistics**:

\[
\mu = \frac{1}{N_{gt}}\sum_{j=1}^{N_{gt}} y_j,
\quad
Y' = Y-\mu,
\quad
X' = X-\mu
\]
\[
s = \max_j \|y'_j\|_2,
\quad
\bar Y = \frac{Y'}{s},\;\bar X = \frac{X'}{s}
\quad (s>10^{-8}).
\]

So both partial and gt are rigidly co-centered/scaled by gt-derived sphere radius.

### 3.3 Augmentation (train only when enabled)
A shared transform is applied to both clouds:
- optional axis reflections (x/z based on random branch),
- rotation about y-axis by random angle \(\alpha\in[0,2\pi)\),
- isotropic scaling \(\gamma\in[1/1.3,1]\).

Equivalent form:
\[
X_{aug}=\gamma R X,\qquad Y_{aug}=\gamma R Y.
\]

### 3.4 Furthest Point Sampling (FPS)
For many stages, the code uses CUDA FPS:
\[
\mathcal S = \{i_1,\dots,i_n\},\quad i_t=\arg\max_i\min_{j<t}\|p_i-p_{i_j}\|_2.
\]
This greedily maximizes set coverage in Euclidean space. `gather_points` then forms the sampled tensor.

---

## 4. Encoder mathematics

Encoder class: `PCT_encoderExact`.

### 4.1 Initial pointwise embedding
Input \(X\in\mathbb{R}^{B\times 3\times N_{enc}}\):

- `conv1`: \(3\to 64\), GELU.
- `conv2`: \(64\to C\) with \(C=64\).

Gives \(F_0\in\mathbb{R}^{B\times C\times N_{enc}}\).

### 4.2 GDP stage 1
- Sample indices \(I_0=\text{FPS}(X,\lfloor N_{enc}/4\rfloor)\).
- Gather \(F_{g0}=F_0[:, :, I_0]\), \(X_1=X[:, :, I_0]\).
- Cross-attend \(F_{g0}\) against full \(F_0\): \(A_1=\text{Attn}(F_{g0},F_0)\).
- Concatenate and self-attend: \([F_{g0};A_1]\to F_1\).

Shape trend: \(N_{enc}\to N_{enc}/4\), channel doubles through concat.

### 4.3 GDP stage 2
- FPS on \(X_1\) to roughly half: \(I_1\).
- Gather \(F_{g1}\), \(X_2\).
- Cross-attn \(F_{g1}\leftrightarrow F_1\), concat, self-attn \(\to F_2\).

### 4.4 GDP stage 3
- FPS on \(X_2\) to roughly half again: \(I_2\).
- Cross-attn/concat/self-attn \(\to F_3\).

### 4.5 Global token and coarse generation
- Global max pool:
\[
G = \max_{n} F_3[:, :, n] \in \mathbb{R}^{B\times 512\times 1}.
\]
- Seed generator path: `ps_adj` \(512\to512\), transposed conv `ps` \((k=128)\), `ps_refuse` back to 512, then 3 attention blocks.
- Explicit reshape used in code:
\[
\mathbb{R}^{B\times 512\times 128}\to\mathbb{R}^{B\times 256\times 256}
\]
(implementation-specific inferred remapping), then linear interpolation to \(N_c\), then \(1\times1\) convs to xyz.

Output: coarse \(\hat Y_c\in\mathbb{R}^{B\times 3\times N_c}\) and global feature \(G\).

---

## 5. Attention mathematics

Attention block class: `cross_transformer`.

Given source tensors \(S_1,S_2\in\mathbb{R}^{B\times C_{in}\times N}\):
1. 1x1 input projection: \(\phi(S)=W_p*S\) to \(C_{out}\).
2. Rearrangement to \((N,B,C_{out})\) for PyTorch MHA.
3. LayerNorm on both query and key/value streams.
4. Multi-head attention:
\[
\text{MHA}(Q,K,V)=\text{Concat}(h_1,\dots,h_H)W^O,
\]
\[
h_t=\text{softmax}\!\left(\frac{Q_tK_t^\top}{\sqrt{d_h}}\right)V_t.
\]
5. Residual + dropout + LayerNorm.
6. FFN: \(\text{Linear}\to\text{GELU}\to\text{Linear}\), residual add.

Notes:
- This block is used as self-attention when `src1==src2`, and cross-attention otherwise.
- Attention is over the **point/token axis**, not channel attention.

---

## 6. Coarse generation / seed generation / refinement mathematics

### 6.1 Base seed construction
Let input branch be either raw input or encoder-capped input (`use_input_for_seed_sampling`).
Optional pre-concat FPS on input branch (`seed_concat_use_input_fps`) produces \(X_s\).

Then:
\[
S_{src} = [X_s\;\Vert\;\hat Y_c] \quad\text{(concatenate on point axis)}
\]
\[
\hat Y_s = \text{FPS}(S_{src},\min(N_s,|S_{src}|)).
\]

### 6.2 Refinement block (`PCT_refine`)
Given coarse points \(P\in\mathbb{R}^{B\times 3\times N}\), global feature \(G\):
- Point feature path: \(\psi(P)\in\mathbb{R}^{B\times C\times N}\).
- Global broadcast path: \(\gamma(G)\in\mathbb{R}^{B\times C\times N}\).
- Fuse then attention stack \(\to Z\in\mathbb{R}^{B\times Cr\times N}\).
- Reshape to expanded point count: \(\mathbb{R}^{B\times C\times Nr}\).
- Residual coordinate regression:
\[
\hat P_{up}=\Delta(P,G)+\text{repeat}(P,r).
\]

This is used twice:
- `refine` with ratio \(r_1\): \(N_s\to N_f\).
- `refine1` with ratio \(r_2\): \(N_f\to N_{f1}\).

### 6.3 Exact-size head
From \(\hat Y_{f1}\), if target \(N_t>N_{f1}\):
- repeat by \(\rho=\lceil N_t/N_{f1}\rceil\),
- concatenate point and broadcast global features,
- predict residual offsets via MLP,
- add residual to repeated points,
- FPS-trim to exactly \(N_t\).

If \(N_t\le N_{f1}\), direct FPS downsample to \(N_t\).

Optional final merge (`merge_input_in_final`):
\[
\hat Y_e = \text{FPS}([\hat Y_e\Vert X],N_t).
\]

---

## 7. Exact-size or output-size matching mathematics

Target count resolver:

\[
N_t=
\begin{cases}
N_{gt}, & \text{if }\texttt{exact_target_from}=\texttt{"gt"} \\
N_{in}, & \text{if }\texttt{"input"} \\
N_{cfg}, & \text{if }\texttt{"config" and }N_{cfg}>0
\end{cases}
\]

The exact-size mechanism is **non-learned size control + learned coordinate refinement**:
- cardinality alignment by repeat + FPS trim,
- geometry adaptation by residual coordinate head.

So final predicted size is guaranteed to match \(N_t\) per sample.

---

## 8. Loss functions

### 8.1 Chamfer terms in `calc_cd`
Chamfer CUDA kernel returns nearest-neighbor squared distances:
- \(d_1(i)=\min_j\|y_i-\hat y_j\|_2^2\),
- \(d_2(j)=\min_i\|\hat y_j-y_i\|_2^2\).

Implemented metrics:
\[
\text{CD}_p = \frac{1}{2}\left(\frac{1}{N_y}\sum_i\sqrt{d_1(i)} + \frac{1}{N_{\hat y}}\sum_j\sqrt{d_2(j)}\right)
\]
\[
\text{CD}_t = \frac{1}{N_y}\sum_i d_1(i)+\frac{1}{N_{\hat y}}\sum_j d_2(j).
\]

### 8.2 Training losses
- Exact-stage: \(\mathcal L_e=\text{CD}_p(\hat Y_e^{sub},Y^{sub})\), where optional FPS subsets are used if `exact_cd_points>0`.
- Fine1: \(\mathcal L_{f1}=\text{CD}_p(\hat Y_{f1},Y_{f1})\), with \(Y_{f1}=\text{FPS}(Y,N_{f1})\).
- Fine: \(\mathcal L_f=\text{CD}_p(\hat Y_f,Y_f)\), \(Y_f=\text{FPS}(Y_{f1},N_f)\).
- Coarse: \(\mathcal L_c=\text{CD}_p(\hat Y_c,Y_c)\), \(Y_c=\text{FPS}(Y_f,N_c)\).

Combined loss:
\[
\mathcal L_{total}=w_c\mathcal L_c+w_f\mathcal L_f+w_{f1}\mathcal L_{f1}+w_e\mathcal L_e.
\]

Default notebook weights:
\(w_c=0.5, w_f=0.75, w_{f1}=1.0, w_e=2.0\).

### 8.3 Evaluation metrics
Reported: `cd_p`, `cd_t` on final output (optionally FPS-limited by `metric_cd_points`), plus `cd_p_coarse`, `cd_t_coarse` for coarse output.

---

## 9. Optimization mathematics

Training does standard backprop through all differentiable paths:
\[
\theta \leftarrow \theta - \eta\,\nabla_\theta\mathcal L_{total}.
\]

Default optimizer in notebook config: **AdamW** with betas \((0.9,0.999)\), lr \(3\times 10^{-4}\), weight decay \(10^{-5}\).

AdamW parameter update (per parameter \(\theta\)):
\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]
\[
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad \hat v_t=\frac{v_t}{1-\beta_2^t},
\]
\[
\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}-\eta\lambda\theta_{t-1}.
\]

`lr_decay` knobs exist in config but are off by default in this notebook setup.

---

## 10. Fine-tuning mathematics

Fine-tuning behavior in this notebook pipeline:
- If `load_model` is set, `net_state_dict` is loaded and optimization resumes.
- Checkpoint may also restore optimizer state and effective lr (`load_full_checkpoint`).
- All model parameters remain trainable (no freezing logic is implemented).

So fine-tuning is mathematically continuation from \(\theta_0=\theta_{pretrained}\) instead of random init, with same loss family but potentially different data distribution and exact-size settings.

---

## 11. Tensor shape and dimensional analysis

Using symbolic sizes:

1. **Input**: \(X\in\mathbb{R}^{B\times 3\times N_{in}}\).
2. **Encoder cap** (optional FPS): \(X_{enc}\in\mathbb{R}^{B\times 3\times N_{enc}}\), \(N_{enc}=\min(N_{in},\texttt{max_encoder_points})\) if cap active.
3. **Encoder coarse output**: \(\hat Y_c\in\mathbb{R}^{B\times 3\times N_c}\), global \(G\in\mathbb{R}^{B\times 512\times 1}\).
4. **Seed source concat**: \([X_s\Vert\hat Y_c]\in\mathbb{R}^{B\times 3\times(N_{xs}+N_c)}\).
5. **Base seed**: \(\hat Y_s\in\mathbb{R}^{B\times 3\times N_s}\).
6. **Refine-1**: \(\hat Y_f\in\mathbb{R}^{B\times 3\times N_f}\), \(N_f=N_s r_1\).
7. **Refine-2**: \(\hat Y_{f1}\in\mathbb{R}^{B\times 3\times N_{f1}}\), \(N_{f1}=N_f r_2\).
8. **Exact head output BCN**: \(\hat Y_e^{BCN}\in\mathbb{R}^{B\times 3\times N_t}\).
9. **Final returned BNC**: \(\hat Y_e=\text{transpose}(\hat Y_e^{BCN})\in\mathbb{R}^{B\times N_t\times 3}\).
10. **GT tensor in training/eval path**: \(Y\in\mathbb{R}^{B\times N_{gt}\times 3}\).

Concrete counts with notebook defaults:
- \(N_c=4096\), \(N_s=5900\), \(r_1=4\), \(r_2=8\).
- \(N_f=23600\), \(N_{f1}=188800\) before exact-head resizing/trimming.
- `max_encoder_points = 131072 * 0.25` (stored as int) \(=32768\).

---

## 12. Mathematical meaning of key hyperparameters

- `max_encoder_points`: upper bound on tokens entering attention-heavy encoder; reduces \(O(N^2)\)-like attention memory/compute pressure.
- `encoder_coarse_points` (\(N_c\)): cardinality of encoder coarse hypothesis; larger \(N_c\) increases geometric support before refinement.
- `base_seed_points` (\(N_s\)): number of anchors after concatenation/FPS; sets base for subsequent multiplicative upsampling.
- `step1`, `step2` (\(r_1,r_2\)): expansion factors in two refine blocks; final pre-exact count scales as \(N_s r_1 r_2\).
- `exact_target_from`/`exact_target_points`: defines cardinality constraint set-point \(N_t\).
- `exact_tail_hidden`: hidden width of exact residual head; controls capacity of final coordinate correction.
- `seed_concat_use_input_fps` and related input-point selectors: regulate balance between observed input branch and predicted coarse branch before seed FPS.
- `merge_input_in_final`: enforces a union-then-FPS post-step to preserve observed geometry in output.
- `exact_cd_points`, `metric_cd_points`: optional FPS downsampling for Chamfer computation; trades unbiased full-set distance for tractable complexity.
- `w_coarse,w_fine,w_fine1,w_exact`: stage-wise objective weighting; larger weight increases gradient emphasis on that scale.
- `nhead` in attention blocks: number of heads in MHA (default 4 in `cross_transformer`), splitting feature subspaces for multi-relation modeling.

---

## 13. End-to-end mathematical summary

Overall composition:
\[
\hat Y = \mathcal F_\theta(X) = \mathcal H_{exact}\Big(\mathcal R_2(\mathcal R_1(\mathcal S(\mathcal E(X))))\Big),
\]
with:
- \(\mathcal E\): encoder producing global token and coarse set,
- \(\mathcal S\): seed construction by concat + FPS,
- \(\mathcal R_1,\mathcal R_2\): refinement/upsampling with residual coordinate prediction,
- \(\mathcal H_{exact}\): exact-size repeat/residual/trim head.

Learning pipeline:
1. Normalize pair (optional), augment pair (optional).
2. Encode partial cloud with hierarchical FPS+attention.
3. Generate coarse cloud and global latent.
4. Build seed from input/coarse union.
5. Two-stage refinement increases point density.
6. Exact-size head enforces target cardinality.
7. Multi-scale Chamfer losses supervise coarse-to-final geometry jointly.

---

## 14. Implementation-grounded notes

### A) Directly explicit in code
- Pair normalization center/scale from gt cloud.
- FPS-based sampling at encoder and loss alignment stages.
- Two refine blocks with ratios `step1` and `step2`.
- Exact head uses repeat + residual offset + FPS trim.
- Total training loss is weighted sum of four CD terms.

### B) Inferred from implementation structure
- Attention equations are standard scaled dot-product MHA via `nn.MultiheadAttention`.
- `x2_d.reshape(B, channel*4, 256)` acts as fixed-token remapping from `512×128` feature volume.
- When CD subset sizing differs from gt count, gt is FPS-sampled to matching cardinality before distance computation.

### C) Standard assumptions tied to used layers
- 1x1 Conv acts as shared pointwise affine map across points.
- GELU nonlinearity provides smooth gating in feature transforms.
- AdamW decouples weight decay from adaptive moment update.

### Uncertainty disclosures
- Notebook embeds key model/dataset/train code as strings written at runtime; if users modify those cells before execution, mathematics may differ.
- No explicit additional regularizers (e.g., normal consistency, repulsion, attention entropy penalties) are present in the inspected implementation.
