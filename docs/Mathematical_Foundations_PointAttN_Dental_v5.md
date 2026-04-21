# Mathematical Foundations of PointAttN_Dental_v5_fine-tuning

## 1. Problem formulation

The notebook implements supervised point-cloud completion for dental geometry.

For each sample, the model receives a partial point cloud and predicts a completed cloud:

$$
\hat{Y} = F(X;\theta)
$$

Where:
- $X \in \mathbb{R}^{3 \times N_{\text{in}}}$ is the partial input cloud (channel-first in model forward).
- $\hat{Y} \in \mathbb{R}^{N_{\text{out}} \times 3}$ is the predicted complete cloud.
- $\theta$ denotes all trainable model parameters.

The supervision target is a complete cloud $Y \in \mathbb{R}^{N_{\text{gt}} \times 3}$, and the objective is multi-stage Chamfer-based minimization.

Fine-tuning is implemented as optimization continuation from a saved checkpoint (`load_model`) rather than random initialization. Mathematically, this changes initialization from $\theta_0 \sim \mathcal{D}_{\text{init}}$ to $\theta_0 = \theta_{\text{pretrained}}$, while preserving the same forward mapping family and weighted multi-loss objective.

---

## 2. Notation

Let the batch size be $B$.

- $N_{\text{in}}$: input partial point count.
- $N_{\text{enc}}$: encoder input count after optional capping (`max_encoder_points`).
- $N_{\text{gt}}$: ground-truth point count.
- $N_c$: encoder coarse output count (`encoder_coarse_points`).
- $N_s$: base seed count (`base_seed_points`).
- $r_1, r_2$: refinement expansion factors (`step1`, `step2`).
- $N_{f}=N_s r_1$: first refinement count.
- $N_{f1}=N_s r_1 r_2$: second refinement count.
- $N_t$: exact target output count chosen by `exact_target_from`.
- $X^{(b)}=\{x_i\}_{i=1}^{N_{\text{in}}}$: input point set for batch item $b$.
- $Y^{(b)}=\{y_j\}_{j=1}^{N_{\text{gt}}}$: ground-truth set.
- $\hat{Y}_c,\hat{Y}_s,\hat{Y}_f,\hat{Y}_{f1},\hat{Y}_e$: coarse, seed, first refine, second refine, and exact outputs.
- $G$: global encoder feature token.
- $F_\ell$: intermediate feature tensor at stage $\ell$.
- $Q,K,V$: attention query, key, value tensors.
- $\mathrm{FPS}(\cdot,n)$: furthest-point sampling to $n$ points.
- $\mathcal{L}_c,\mathcal{L}_f,\mathcal{L}_{f1},\mathcal{L}_e$: coarse/fine/fine1/exact losses.
- $w_c,w_f,w_{f1},w_e$: stage loss weights.
- $\theta$: trainable parameters of encoder, refinement blocks, exact-size head, and projections.

Tensor layout conventions used in this implementation:
- BCN: $\mathbb{R}^{B\times C\times N}$.
- BNC: $\mathbb{R}^{B\times N\times C}$.

---

## 3. Point-cloud representation and geometric preprocessing

### 3.1 Point-cloud representation

Each cloud is represented as an unordered set of Euclidean coordinates in $\mathbb{R}^3$. The dataset loader keeps only the first three columns of loaded arrays.

### 3.2 Pair normalization

If `normalize_pair=True`, normalization is performed with statistics from the ground-truth cloud.

$$
\mu = \frac{1}{N_{\text{gt}}} \sum_{j=1}^{N_{\text{gt}}} y_j
$$

Where:
- $\mu \in \mathbb{R}^{3}$ is the GT centroid.

Both clouds are centered with the same centroid:

$$
X' = X - \mu, \qquad Y' = Y - \mu
$$

Where:
- $X',Y'$ are centered point sets.

Scale is the maximum GT radial norm:

$$
s = \max_{j} \lVert y'_j \rVert_2
$$

Where:
- $s$ is a scalar normalization radius.

If $s>10^{-8}$:

$$
\bar{X}=\frac{X'}{s}, \qquad \bar{Y}=\frac{Y'}{s}
$$

Where:
- $\bar{X},\bar{Y}$ are normalized clouds used by the model.

### 3.3 Pair augmentation (optional; train split only)

When `augment_train=True`, the same rigid/scale transform is applied to both partial and GT clouds.

$$
X_{\text{aug}} = \gamma R X, \qquad Y_{\text{aug}} = \gamma R Y
$$

Where:
- $R \in \mathbb{R}^{3\times 3}$ is built from random axis reflections and random rotation about the $y$ axis.
- $\gamma \in [1/1.3,1]$ is isotropic random scaling.

### 3.4 Furthest Point Sampling (FPS)

The model repeatedly uses CUDA FPS (`furthest_point_sample` + `gather_points`). Conceptually:

$$
\mathcal{S} = \{i_1,\dots,i_n\},\quad
i_t = \arg\max_i \min_{k<t} \lVert p_i - p_{i_k} \rVert_2
$$

Where:
- $\{p_i\}$ is an input point set.
- $\mathcal{S}$ is the selected index set of size $n$.
- At each step, FPS picks the point farthest from the currently selected set.

No explicit k-NN/radius-neighborhood grouping is used in the forward model path for this notebook’s exact pipeline.

---

## 4. Encoder mathematics

Encoder class used by the notebook-defined exact model: `PCT_encoderExact`.

### 4.1 Input lifting

Input to encoder after optional cap is $X_{\text{enc}}\in\mathbb{R}^{B\times 3\times N_{\text{enc}}}$.

Pointwise 1x1 convolutions:

$$
F_0 = \phi_2\big(\sigma(\phi_1(X_{\text{enc}}))\big)
$$

Where:
- $\phi_1:3\to 64$ (1x1 conv), $\phi_2:64\to 64$ (1x1 conv).
- $\sigma$ is GELU.
- $F_0\in\mathbb{R}^{B\times 64\times N_{\text{enc}}}$.

### 4.2 GDP stage 1 (cross + self attention on sampled support)

Sampling:

$$
I_0 = \mathrm{FPS}(X_{\text{enc}},\lfloor N_{\text{enc}}/4\rfloor)
$$

Where:
- $I_0$ are sampled indices.

Gather sampled features/points:

$$
F_{g0} = \mathrm{Gather}(F_0,I_0),\quad X_1 = \mathrm{Gather}(X_{\text{enc}},I_0)
$$

Cross-attention then concatenation and self-attention:

$$
A_1 = \mathrm{Attn}(F_{g0},F_0),\quad
F_1 = \mathrm{Attn}\big([F_{g0};A_1],[F_{g0};A_1]\big)
$$

Where:
- $[\cdot;\cdot]$ denotes channel concatenation.
- Point support is reduced from $N_{\text{enc}}$ to roughly $N_{\text{enc}}/4$.

### 4.3 GDP stage 2

$$
I_1 = \mathrm{FPS}(X_1,\lfloor |X_1|/2\rfloor), \quad
F_{g1}=\mathrm{Gather}(F_1,I_1), \quad X_2=\mathrm{Gather}(X_1,I_1)
$$

$$
A_2 = \mathrm{Attn}(F_{g1},F_1),\quad
F_2 = \mathrm{Attn}([F_{g1};A_2],[F_{g1};A_2])
$$

Where:
- Support size is approximately halved again.

### 4.4 GDP stage 3

$$
I_2 = \mathrm{FPS}(X_2,\lfloor |X_2|/2\rfloor),\quad F_{g2}=\mathrm{Gather}(F_2,I_2)
$$

$$
A_3 = \mathrm{Attn}(F_{g2},F_2),\quad
F_3 = \mathrm{Attn}([F_{g2};A_3],[F_{g2};A_3])
$$

### 4.5 Global feature and coarse generation

Global feature token by adaptive max pooling:

$$
G = \max_{n} F_3[:,:,n] \in \mathbb{R}^{B\times 512\times 1}
$$

Where:
- Max is over point index.

Then a transposed-convolution seed generator plus attention is applied, ending with an implementation-specific reshape and interpolation.

**Inferred from implementation.**

$$
\mathbb{R}^{B\times 512\times 128}
\xrightarrow{\text{reshape}}
\mathbb{R}^{B\times 256\times 256}
\xrightarrow{\text{linear interpolate to }N_c}
\mathbb{R}^{B\times 256\times N_c}
$$

Where:
- The reshape is hard-coded in the model (`reshape(batch_size, channel*4, 256)` with `channel=64`).
- Final 1x1 projection maps to xyz coarse points $\hat{Y}_c\in\mathbb{R}^{B\times 3\times N_c}$.

---

## 5. Attention mathematics

The model attention block is `cross_transformer`, built around `nn.MultiheadAttention`.

### 5.1 Query/key/value construction

Given two feature maps $S_1,S_2 \in \mathbb{R}^{B\times C_{\text{in}}\times N}$:

$$
\tilde{S}_1 = W_p * S_1, \qquad \tilde{S}_2 = W_p * S_2
$$

Where:
- $W_p$ is a shared 1x1 projection (`input_proj`) from $C_{\text{in}}$ to $C_{\text{out}}$.
- $*$ denotes pointwise convolution over channels.

After permutation to token-major format $(N,B,C_{\text{out}})$ and layer normalization:

$$
Q = \mathrm{LN}(\tilde{S}_1), \quad K = \mathrm{LN}(\tilde{S}_2), \quad V=\mathrm{LN}(\tilde{S}_2)
$$

Where:
- Attention is applied over the **point/token dimension**.

### 5.2 Multi-head attention equation

**Standard formula used by imported layer (`nn.MultiheadAttention`).**

$$
\mathrm{MHA}(Q,K,V) = \mathrm{Concat}(H_1,\dots,H_h)W^O
$$

$$
H_t = \mathrm{softmax}\!\left(\frac{Q_tK_t^\top}{\sqrt{d_h}}\right)V_t
$$

Where:
- $h$ is number of heads (default `nhead=4` in this block).
- $d_h$ is per-head key dimension.
- $W^O$ is the output projection in MHA.

### 5.3 Residual and feedforward structure

Block output sequence:

$$
Z_1 = \mathrm{LN}\big(Q + \mathrm{Dropout}(\mathrm{MHA}(Q,K,V))\big)
$$

$$
Z_2 = Z_1 + \mathrm{Dropout}\big(W_2\,\mathrm{GELU}(W_1 Z_1)\big)
$$

Where:
- $W_1,W_2$ are linear layers of the FFN.
- This matches transformer-style attention + FFN residual structure.

---

## 6. Coarse prediction, seed generation, and refinement

### 6.1 Input capping and branch selection

Before encoding:

$$
X_{\text{enc}} =
\begin{cases}
\mathrm{FPS}(X, N_{\max}), & N_{\max}>0 \\
X, & \text{otherwise}
\end{cases}
$$

Where:
- $N_{\max}=\texttt{max\_encoder\_points}$.

Seed-construction input branch:
- Raw input $X$ if `use_input_for_seed_sampling=True`.
- Else capped input $X_{\text{enc}}$.

Optional pre-concatenation FPS on this input branch chooses count from mode (`full`, `config`, `match_coarse`, `ratio_to_coarse`).

### 6.2 Base seed generation

After obtaining coarse prediction $\hat{Y}_c$:

$$
S_{\text{src}} = [X_{\text{seed}}\,\Vert\,\hat{Y}_c]
$$

$$
\hat{Y}_s = \mathrm{FPS}(S_{\text{src}},\min(N_s, |S_{\text{src}}|))
$$

Where:
- Concatenation is along point axis.
- $\hat{Y}_s\in\mathbb{R}^{B\times 3\times N_s}$ is the base seed cloud.

### 6.3 Refinement block mapping (`PCT_refine`)

Given input points $P\in\mathbb{R}^{B\times 3\times N}$ and global token $G\in\mathbb{R}^{B\times 512\times1}$:

$$
U = \psi(P), \qquad G_b = \gamma(G)\text{ broadcast to }N
$$

$$
T_0 = [U;G_b], \quad
T_1=\mathrm{Attn}(T_0,T_0),\; T_2=\mathrm{Attn}(T_1,T_1),\; T_3=\mathrm{Attn}(T_2,T_2)
$$

$$
\Delta P = \rho\big(T_3,\mathrm{repeat}(U,r)\big), \qquad
\hat{P}=\mathrm{repeat}(P,r)+\Delta P
$$

Where:
- $r$ is block ratio (`step1` for first block, `step2` for second).
- $\psi,\gamma,\rho$ denote pointwise conv pipelines implemented in the block.
- Output point count becomes $Nr$.

### 6.4 Two-stage refinement count transitions

$$
N_s \xrightarrow{r_1} N_f=N_s r_1 \xrightarrow{r_2} N_{f1}=N_s r_1 r_2
$$

Where:
- With notebook defaults: $N_s=5900$, $r_1=4$, $r_2=8$, so $N_{f1}=188800$ before exact-size head.

### 6.5 Optional final merge with partial input

If `merge_input_in_final=True`:

$$
\hat{Y}_e \leftarrow \mathrm{FPS}([\hat{Y}_e\Vert X],N_t)
$$

Where:
- This preserves some observed points in final support before enforcing exact count.

---

## 7. Exact-size head or output-size matching

Exact-size logic is implemented by `ExactSizeHead` and target-resolution resolver.

### 7.1 Target point count selection

$$
N_t =
\begin{cases}
N_{\text{gt}}, & \texttt{exact\_target\_from} = \texttt{gt} \\
N_{\text{in}}, & \texttt{exact\_target\_from} = \texttt{input} \\
N_{\text{cfg}}, & \texttt{exact\_target\_from} = \texttt{config},\; N_{\text{cfg}}>0
\end{cases}
$$

Where:
- $N_{\text{cfg}}=\texttt{exact\_target\_points}$.

Fallback in code uses GT count if available, otherwise input count.

### 7.2 Exact head mechanics

If $N_t \le N_{f1}$, output is direct FPS downsampling:

$$
\hat{Y}_e = \mathrm{FPS}(\hat{Y}_{f1},N_t)
$$

If $N_t > N_{f1}$, repeat-and-correct pipeline:

$$
\rho = \left\lceil \frac{N_t}{N_{f1}} \right\rceil,
\quad
P_{\text{rep}} = \mathrm{repeat}(\hat{Y}_{f1},\rho)
$$

$$
\Delta = h\big([\phi_p(P_{\text{rep}});\phi_g(G)]\big),
\quad
P_{\text{corr}} = P_{\text{rep}} + \Delta
$$

$$
\hat{Y}_e = \mathrm{FPS}(P_{\text{corr}},N_t)
$$

Where:
- $\phi_p$ is point-feature projection, $\phi_g$ is broadcast global-feature projection.
- $h$ is the residual MLP stack in the exact head.
- Mechanism combines feature-conditioned residual refinement and sampling-based exact cardinality enforcement.

---

## 8. Loss functions

The notebook’s exact model uses Chamfer-derived losses from `utils/model_utils.py::calc_cd`.

### 8.1 Chamfer nearest-neighbor distances

For predicted set $\hat{Y}$ and target set $Y$:

$$
d_1(i)=\min_j \lVert y_i - \hat{y}_j \rVert_2^2,
\qquad
d_2(j)=\min_i \lVert \hat{y}_j - y_i \rVert_2^2
$$

Where:
- $d_1$ maps GT points to nearest predicted points.
- $d_2$ maps predicted points to nearest GT points.

### 8.2 Point-wise Chamfer variant used for training losses (`cd_p`)

$$
\mathrm{CD}_p(Y,\hat{Y})
= \frac{1}{2}
\left(
\frac{1}{|Y|}\sum_i \sqrt{d_1(i)}
+
\frac{1}{|\hat{Y}|}\sum_j \sqrt{d_2(j)}
\right)
$$

Where:
- This is exactly how `calc_cd` computes `cd_p` from Chamfer kernel outputs.

### 8.3 Squared Chamfer variant for reporting (`cd_t`)

$$
\mathrm{CD}_t(Y,\hat{Y})
=
\frac{1}{|Y|}\sum_i d_1(i)
+
\frac{1}{|\hat{Y}|}\sum_j d_2(j)
$$

Where:
- `cd_t` is computed and reported in evaluation.

### 8.4 Stage losses used in training forward

Exact-stage loss (optionally with FPS subset using `exact_cd_points`):

$$
\mathcal{L}_e = \mathrm{CD}_p\big(\hat{Y}_e^{\text{sub}},Y^{\text{sub}}\big)
$$

Where:
- If `exact_cd_points <= 0`, no subset is applied.
- If subset is used, GT is sampled to matching count when needed.

Refinement-stage losses:

$$
\mathcal{L}_{f1} = \mathrm{CD}_p\big(\hat{Y}_{f1},\mathrm{FPS}(Y,N_{f1})\big)
$$

$$
\mathcal{L}_f = \mathrm{CD}_p\big(\hat{Y}_{f},\mathrm{FPS}(\mathrm{FPS}(Y,N_{f1}),N_f)\big)
$$

$$
\mathcal{L}_c = \mathrm{CD}_p\big(\hat{Y}_{c},\mathrm{FPS}(\mathrm{FPS}(\mathrm{FPS}(Y,N_{f1}),N_f),N_c)\big)
$$

Where:
- Each stage compares prediction with GT resampled to that stage’s point count.

### 8.5 Total weighted objective

$$
\begin{aligned}
\mathcal{L}_{\text{total}}
&= w_c\mathcal{L}_c + w_f\mathcal{L}_f + w_{f1}\mathcal{L}_{f1} + w_e\mathcal{L}_e
\end{aligned}
$$

Where:
- $w_c=\texttt{w\_coarse}$, $w_f=\texttt{w\_fine}$, $w_{f1}=\texttt{w\_fine1}$, $w_e=\texttt{w\_exact}$.
- Notebook defaults: $(w_c,w_f,w_{f1},w_e)=(0.5,0.75,1.0,2.0)$.

No L1/L2/MSE auxiliary coordinate loss, feature-space loss, or explicit regularization term is used in this exact training forward path.

---

## 9. Optimization mathematics

Training minimizes expected total loss over dataset distribution:

$$
\min_{\theta} \; \mathbb{E}_{(X,Y)}\left[\mathcal{L}_{\text{total}}(X,Y;\theta)\right]
$$

Where:
- Gradients backpropagate through encoder attention blocks, refinement modules, exact-size head, and all pointwise projections.
- FPS index selection itself is non-differentiable, but gradients flow through gathered predicted coordinates used by Chamfer.

Default optimizer is AdamW (`optimizer: AdamW`, `lr=3\times10^{-4}`, `betas=(0.9,0.999)`, `weight_decay=10^{-5}`).

**Standard optimizer formula.**

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,
\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t},
\qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} - \eta\lambda\theta_{t-1}
$$

Where:
- $g_t=\nabla_\theta \mathcal{L}_{\text{total}}$.
- $\eta$ is learning rate.
- $\lambda$ is decoupled weight decay.

Learning-rate decay knobs are present in config, but default notebook values keep decay disabled (`lr_decay=False`).

---

## 10. Fine-tuning mathematics

Fine-tuning behavior in this notebook pipeline:

1. Load pretrained weights into model parameters.
2. Optionally load optimizer state and resume epoch/lr from full checkpoint.
3. Continue minimizing the same $\mathcal{L}_{\text{total}}$ on dental exact dataset.

Mathematically:

$$
\theta_0 = \theta_{\text{pretrained}},
\qquad
\theta_{t+1} = \mathrm{OptStep}(\theta_t,\nabla_\theta\mathcal{L}_{\text{total}})
$$

Where:
- No parameter freezing is implemented in this exact training script.
- Therefore all parameters are updated during fine-tuning.

Difference from scratch training is initialization and possibly resumed optimizer moments, not the architecture/loss definition.

---

## 11. Tensor shape walkthrough

This section traces the exact forward path used in `Model.forward`.

### 11.1 Input and encoder cap

- Input tensor: $X \in \mathbb{R}^{B\times 3\times N_{\text{in}}}$.
- Optional cap: $X_{\text{enc}}\in\mathbb{R}^{B\times 3\times N_{\text{enc}}}$.

With defaults, $N_{\text{enc}}=\min(N_{\text{in}},32768)$.

### 11.2 Encoder output

- Global feature: $G\in\mathbb{R}^{B\times 512\times 1}$.
- Coarse points (BCN): $\hat{Y}_c\in\mathbb{R}^{B\times 3\times N_c}$, default $N_c=4096$.

### 11.3 Seed generation

- Chosen input branch for seed concat: $X_{\text{seed}}\in\mathbb{R}^{B\times 3\times N_{\text{seed-in}}}$.
- Concatenation: $S_{\text{src}}\in\mathbb{R}^{B\times 3\times (N_{\text{seed-in}}+N_c)}$.
- Base seed after FPS: $\hat{Y}_s\in\mathbb{R}^{B\times 3\times N_s}$, default $N_s=5900$.

### 11.4 Refinement stages

- First refine output: $\hat{Y}_f\in\mathbb{R}^{B\times 3\times (N_s r_1)}$.
- Second refine output: $\hat{Y}_{f1}\in\mathbb{R}^{B\times 3\times (N_s r_1 r_2)}$.

Default counts:
- $N_s r_1 = 5900\times4 = 23600$.
- $N_s r_1 r_2 = 23600\times8 = 188800$.

### 11.5 Exact-size head

- Exact output BCN: $\hat{Y}_{e,\text{BCN}}\in\mathbb{R}^{B\times 3\times N_t}$.
- Returned final output BNC: $\hat{Y}_e\in\mathbb{R}^{B\times N_t\times 3}$.

### 11.6 Training/evaluation conversion

- Coarse, seed, refine outputs are transposed to BNC for Chamfer computation and reporting.
- Ground truth is expected as $Y\in\mathbb{R}^{B\times N_{\text{gt}}\times3}$.

---

## 12. Mathematical meaning of key hyperparameters

### 12.1 Point-count and sampling controls

- `max_encoder_points`: caps token count for encoder attention complexity.
  - Increasing it raises geometric coverage and memory/compute cost.
- `encoder_coarse_points` ($N_c$): coarse support density from encoder.
  - Larger $N_c$ improves coarse geometric granularity but increases downstream cost.
- `base_seed_points` ($N_s$): anchor size before multiplicative refinement.
  - Larger $N_s$ scales all later counts via $N_s r_1 r_2$.
- `step1`, `step2` ($r_1,r_2$): upsampling/refinement ratios.
  - Control point-count growth and refinement resolution.

### 12.2 Exact-head and feature dimensions

- `exact_tail_hidden`: hidden channel size of exact residual head.
  - Controls capacity of final feature-conditioned coordinate correction.
- Encoder/refine internal channels (`channel=64`, refine channel default 128, global token 512) determine latent representation rank and attention projection sizes.

### 12.3 Attention heads

- `nhead` in `cross_transformer` (default 4): number of parallel attention subspaces.
  - More heads increase relational subspace decomposition at fixed total width.

### 12.4 Seed concatenation balance controls

- `seed_concat_use_input_fps`, `seed_concat_input_points_from`, `seed_concat_input_points`, `seed_concat_input_ratio_to_coarse` regulate how much partial-input geometry versus coarse-generated geometry contributes to seed source before FPS.

### 12.5 Loss and metric subsampling

- `exact_cd_points`: if positive, exact-stage training CD uses FPS subset of that size.
- `metric_cd_points`: analogous subset for evaluation metrics.

These parameters change the empirical loss surface by replacing full-set Chamfer with subsampled Chamfer.

### 12.6 Loss weights

- `w_coarse`, `w_fine`, `w_fine1`, `w_exact` scale gradient contribution from each supervision stage.
  - Increasing one weight prioritizes that stage during optimization.

---

## 13. End-to-end mathematical summary

Full mapping:

$$
\hat{Y} = F(X;\theta)
$$

Where:
- $X$ is partial cloud input.
- $\hat{Y}$ is exact-size completed output.

Decomposition used in this notebook-defined exact model:

$$
\hat{Y}
=
\mathcal{H}_{\text{exact}}
\circ
\mathcal{R}_2
\circ
\mathcal{R}_1
\circ
\mathcal{S}
\circ
\mathcal{E}
\,(X)
$$

Where:
- $\mathcal{E}$: encoder with FPS-reduced hierarchical attention, returning $(G,\hat{Y}_c)$.
- $\mathcal{S}$: seed construction from concatenated input/coarse support + FPS.
- $\mathcal{R}_1,\mathcal{R}_2$: two refinement modules with ratio-based expansion and residual coordinate prediction.
- $\mathcal{H}_{\text{exact}}$: exact-size head (repeat, feature-conditioned residual correction, FPS trim).

Training objective:

$$
\min_{\theta}
\; w_c\mathcal{L}_c + w_f\mathcal{L}_f + w_{f1}\mathcal{L}_{f1} + w_e\mathcal{L}_e
$$

Where:
- Each $\mathcal{L}_{\cdot}$ is a Chamfer-based term aligned to that stage’s point count.

---

## 14. Implementation-grounded notes

### 14.1 Directly explicit in code

- GT-centered and GT-scaled pair normalization.
- FPS-based sampling for encoder reduction, seed building, and size matching.
- Two refinement blocks with multiplicative ratios (`step1`, `step2`).
- Exact-size head logic: conditional repeat, residual correction, FPS trim.
- Chamfer calculations (`cd_p`, `cd_t`) and weighted total loss formula.
- AdamW optimizer selection and default hyperparameters in config cell.

### 14.2 Inferred from implementation

- Transformer equations for attention internals are inferred from use of `nn.MultiheadAttention` and surrounding residual/FFN code.
- The reshape transition from $[B,512,128]$ to $[B,256,256]$ is a hard-coded tensor remapping; geometric interpretation is not explicitly documented by code comments beyond author-compatibility intent.
- Gradient statement for FPS: index selection is non-differentiable while gathered coordinates remain in differentiable downstream paths.

### 14.3 Standard formula used by an imported layer or method

- Scaled dot-product multi-head attention equation (imported PyTorch module behavior).
- AdamW update equations (standard optimizer definition).
- Chamfer nearest-neighbor distance decomposition from imported CUDA Chamfer operator.

### 14.4 Uncertainty disclosures

- The notebook embeds key exact-model scripts as runtime-generated strings (`model_code`, `dataset_code`, `train_code`). Mathematical behavior documented here matches those embedded definitions as inspected; behavior can differ if notebook cells are edited before writing files.
- No additional regularizers (e.g., normal-consistency or repulsion) are present in this exact pipeline’s inspected forward/loss path.
