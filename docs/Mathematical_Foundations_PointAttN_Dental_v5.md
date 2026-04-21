# Mathematical Foundations of PointAttN\_Dental\_v5\_fine-tuning

## 1. Problem formulation

This notebook defines a supervised mapping from a partial dental point cloud to a completed dental point cloud.

Directly explicit in code.

$$
\hat{Y} = F(X;\theta)
$$

Where:
- $X \in \mathbb{R}^{3 \times N_{\text{in}}}$ is the partial input point cloud.
- $\hat{Y} \in \mathbb{R}^{N_{\text{out}} \times 3}$ is the predicted output point cloud.
- $\theta$ is the set of trainable model parameters.

What it computes:
- A completed geometric point set from a partial observation.

Why used:
- Point cloud completion is the core task in this notebook.

Role in the framework:
- This is the full end-to-end forward objective that all submodules implement.

Fine-tuning is checkpoint-based continuation rather than fresh initialization.

Directly explicit in code.

$$
\theta_0 = \theta_{\text{pretrained}}
$$

Where:
- $\theta_0$ is the initialization used at the start of the current run.
- $\theta_{\text{pretrained}}$ is loaded from `load_model`.

What it computes:
- Initial parameter state for optimization.

Why used:
- Reuses prior learned geometry and optimization state.

Role in the framework:
- Changes the effective starting point of the same mathematical objective.

Overall optimized quantity:

Directly explicit in code.

$$
\min_{\theta}\;\mathbb{E}_{(X,Y)}\left[\mathcal{L}_{\text{total}}(X,Y;\theta)\right]
$$

Where:
- $Y \in \mathbb{R}^{N_{\text{gt}} \times 3}$ is the complete ground-truth cloud.
- $\mathcal{L}_{\text{total}}$ is the weighted stage-wise Chamfer objective.

What it computes:
- The expected training objective across dataset pairs.

Why used:
- Supervises geometric fidelity at multiple reconstruction stages.

Role in the framework:
- Governs all gradient updates.

## 2. Notation

The following symbols are used consistently:

- $B$: batch size.
- $N_{\text{in}}$: number of input points.
- $N_{\text{enc}}$: encoder input size after optional capping by `max_encoder_points`.
- $N_{\text{gt}}$: number of ground-truth points.
- $N_t$: exact target output size used by the exact-size head.
- $X^{(b)} = \{x_i\}_{i=1}^{N_{\text{in}}}$: input points for sample $b$.
- $Y^{(b)} = \{y_j\}_{j=1}^{N_{\text{gt}}}$: ground-truth points.
- $\hat{Y}$: final prediction.
- $\hat{Y}_c$: encoder coarse prediction.
- $\hat{Y}_s$: base seed points.
- $\hat{Y}_f$: first refinement output.
- $\hat{Y}_{f1}$: second refinement output.
- $\hat{Y}_e$: exact-size output.
- $F_\ell$: intermediate feature tensor at stage $\ell$.
- $G$: global feature token.
- $L$: local feature tensor.
- $Q,K,V$: attention query, key, value.
- $\mathcal{S}$: sampled subset index set.
- $\mathcal{N}(i)$: neighborhood of point $i$.
- $\theta$: trainable parameters.
- $\mathcal{L}_c,\mathcal{L}_f,\mathcal{L}_{f1},\mathcal{L}_e$: stage losses.
- $\mathcal{L}_{\text{total}}$: weighted total training loss.

Layout notation:
- BCN means $\mathbb{R}^{B \times C \times N}$.
- BNC means $\mathbb{R}^{B \times N \times C}$.

Neighborhood note:
- This exact pipeline does not apply explicit ball-query or kNN grouping inside model forward. The symbol $\mathcal{N}(i)$ is included for completeness only.

## 3. Point-cloud representation and geometric preprocessing

### 3.1 Input representation

Directly explicit in code.

$$
X = \{x_i\in\mathbb{R}^3\}_{i=1}^{N_{\text{in}}},\qquad
Y = \{y_j\in\mathbb{R}^3\}_{j=1}^{N_{\text{gt}}}
$$

Where:
- $x_i,y_j$ are 3D coordinates.

What it computes:
- Set-valued geometric representation.

Why used:
- Dental surfaces are represented as unstructured point clouds.

Role in the framework:
- Baseline geometry input and supervision target.

### 3.2 Pair centering and scaling

Directly explicit in code.

$$
\mu = \frac{1}{N_{\text{gt}}}\sum_{j=1}^{N_{\text{gt}}} y_j
$$

Where:
- $\mu$ is the centroid of the ground-truth cloud.

What it computes:
- Shared center for both clouds.

Why used:
- Aligns partial and target in a common frame.

Role in the framework:
- Stabilizes the learning domain.

Directly explicit in code.

$$
X' = X-\mu,\qquad Y' = Y-\mu
$$

Where:
- $X',Y'$ are centered clouds.

Directly explicit in code.

$$
s = \max_j \lVert y'_j \rVert_2
$$

Where:
- $s$ is the maximal GT radius.

Directly explicit in code.

$$
\bar{X} = \frac{X'}{s},\qquad \bar{Y} = \frac{Y'}{s},\quad s>10^{-8}
$$

Where:
- $\bar{X},\bar{Y}$ are normalized clouds.

What it computes:
- Scale normalization to unit-like bounding sphere from GT.

Why used:
- Reduces scale variability across cases.

Role in the framework:
- Preconditioning for stable optimization and comparable Chamfer values.

### 3.3 Train-time random geometric transform

Directly explicit in code.

$$
X_{\text{aug}} = \gamma R X,\qquad Y_{\text{aug}} = \gamma R Y
$$

Where:
- $R$ is random reflection-plus-rotation matrix.
- $\gamma \in [1/1.3,1]$ is isotropic scaling.

What it computes:
- Paired augmentation preserving correspondence structure.

Why used:
- Improves invariance and generalization.

Role in the framework:
- Expands the effective training distribution.

### 3.4 Furthest point sampling

Standard formula associated with imported method.

$$
\mathcal{S} = \{i_1,\dots,i_n\},\qquad
i_t = \arg\max_i \min_{k<t} \lVert p_i-p_{i_k} \rVert_2
$$

Where:
- $\{p_i\}$ is an input point set.
- $\mathcal{S}$ is the selected FPS index set.

What it computes:
- A subset with broad spatial coverage.

Why used:
- Controls point count while preserving geometric extent.

Role in the framework:
- Used in encoder downsampling, seed construction, loss size matching, and exact-size trimming.

Nearest-neighbor grouping and ball query:
- Not used by this exact model forward path.
- Uncertainty note: utility libraries contain grouping operators, but they are not called in the inspected exact model forward function.

## 4. Encoder mathematics

The encoder maps $X_{\text{enc}}$ to global feature $G$ and coarse points $\hat{Y}_c$.

### 4.1 Input capping before encoder

Directly explicit in code.

$$
X_{\text{enc}}=
\begin{cases}
\mathrm{FPS}(X,N_{\max}), & N_{\max}>0 \\
X, & \text{otherwise}
\end{cases}
$$

Where:
- $N_{\max}$ is `max_encoder_points`.

What it computes:
- Optional fixed-size encoder token set.

Why used:
- Keeps attention cost manageable.

Role in the framework:
- Defines encoder computational budget.

### 4.2 Pointwise feature lifting

Directly explicit in code.

$$
F_0 = \phi_2\big(\sigma(\phi_1(X_{\text{enc}}))\big)
$$

Where:
- $\phi_1$ is 1x1 convolution $3\to64$.
- $\phi_2$ is 1x1 convolution $64\to64$.
- $\sigma$ is GELU.

What it computes:
- Per-point feature embedding.

Why used:
- Lifts raw coordinates into a learned latent space.

Role in the framework:
- Entry representation for hierarchical attention blocks.

### 4.3 Hierarchical encoder stages

Directly explicit in code.

$$
I_0=\mathrm{FPS}(X_{\text{enc}},\lfloor N_{\text{enc}}/4\rfloor),\quad
F_{g0}=\mathrm{Gather}(F_0,I_0)
$$

Directly explicit in code.

$$
A_1=\mathrm{Attn}(F_{g0},F_0),\quad
F_1=\mathrm{Attn}([F_{g0};A_1],[F_{g0};A_1])
$$

Where:
- $[\cdot;\cdot]$ is channel concatenation.

What it computes:
- Cross-scale interaction from sampled support to denser context, then self-refinement.

Why used:
- Captures long-range dependencies with reduced token count.

Role in the framework:
- First hierarchical abstraction stage.

Directly explicit in code.

$$
I_1=\mathrm{FPS}(X_1,\lfloor |X_1|/2\rfloor),\quad
F_{g1}=\mathrm{Gather}(F_1,I_1)
$$

Directly explicit in code.

$$
A_2=\mathrm{Attn}(F_{g1},F_1),\quad
F_2=\mathrm{Attn}([F_{g1};A_2],[F_{g1};A_2])
$$

Directly explicit in code.

$$
I_2=\mathrm{FPS}(X_2,\lfloor |X_2|/2\rfloor),\quad
F_{g2}=\mathrm{Gather}(F_2,I_2)
$$

Directly explicit in code.

$$
A_3=\mathrm{Attn}(F_{g2},F_2),\quad
F_3=\mathrm{Attn}([F_{g2};A_3],[F_{g2};A_3])
$$

What it computes:
- Progressive feature abstraction with decreasing support size.

Why used:
- Balances context capture and computational feasibility.

Role in the framework:
- Produces high-level features for global token and coarse prediction.

### 4.4 Global pooling and coarse-point decoding

Directly explicit in code.

$$
G = \max_{n} F_3[:,:,n]
$$

Where:
- Max is over point axis.
- $G\in\mathbb{R}^{B\times512\times1}$.

What it computes:
- Global shape descriptor.

Why used:
- Supplies global conditioning for refinement and exact-size head.

Role in the framework:
- Global latent control signal.

Inferred from implementation.

$$
\mathbb{R}^{B\times512\times128}
\xrightarrow{\text{reshape}}
\mathbb{R}^{B\times256\times256}
\xrightarrow{\text{interpolate to }N_c}
\mathbb{R}^{B\times256\times N_c}
$$

Where:
- The reshape is hard-coded in model code.

What it computes:
- Token remapping before final coordinate projection.

Why used:
- Keeps compatibility with original author-style coarse decoding path.

Role in the framework:
- Produces coarse coordinate prediction $\hat{Y}_c$.

## 5. Attention mathematics

Attention appears in `cross_transformer` and is used both as self-attention and cross-attention.

### 5.1 Query, key, and value construction

Directly explicit in code.

$$
\tilde{S}_1 = W_p * S_1,\qquad
\tilde{S}_2 = W_p * S_2
$$

Where:
- $W_p$ is shared 1x1 projection.
- $S_1,S_2$ are input feature maps.

Directly explicit in code.

$$
Q=\mathrm{LN}(\tilde{S}_1),\quad
K=\mathrm{LN}(\tilde{S}_2),\quad
V=\mathrm{LN}(\tilde{S}_2)
$$

Where:
- Layer normalization is applied before attention.
- Attention axis is point/token axis after permutation to token-major format.

What it computes:
- Feature-dependent query-key-value triplet.

Why used:
- Enables adaptive weighting of contextual points.

Role in the framework:
- Core mechanism for long-range geometric dependency modeling.

### 5.2 Attention equation

Standard formula associated with imported method.

$$
\mathrm{MHA}(Q,K,V)=\mathrm{Concat}(H_1,\dots,H_h)W^O
$$

$$
H_t = \mathrm{softmax}\left(\frac{Q_tK_t^\top}{\sqrt{d_h}}\right)V_t
$$

Where:
- $h$ is number of heads (`nhead`, default 4 in this block).
- $d_h$ is per-head feature dimension.
- $W^O$ is output projection matrix.

What it computes:
- Weighted context aggregation per token.

Why used:
- Captures nonlocal geometric dependencies.

Role in the framework:
- Drives encoder abstraction and refinement feature propagation.

### 5.3 Residual and feedforward path

Directly explicit in code.

$$
Z_1 = \mathrm{LN}\left(Q + \mathrm{Dropout}(\mathrm{MHA}(Q,K,V))\right)
$$

Directly explicit in code.

$$
Z_2 = Z_1 + \mathrm{Dropout}\left(W_2\,\mathrm{GELU}(W_1 Z_1)\right)
$$

Where:
- $W_1,W_2$ are feedforward linear layers.

What it computes:
- Transformer-style residual update with nonlinear channel mixing.

Why used:
- Stabilizes optimization and increases representational capacity.

Role in the framework:
- Repeated attention block used across encoder and refinement modules.

## 6. Coarse prediction, seed generation, and refinement

### 6.1 Seed-source construction

Directly explicit in code.

$$
S_{\text{src}} = [X_{\text{seed}}\Vert\hat{Y}_c]
$$

Where:
- $X_{\text{seed}}$ is selected from raw input or capped input, with optional pre-concat FPS.

What it computes:
- Combined observed and generated support set.

Why used:
- Prevents purely generated seeds from drifting away from observed geometry.

Role in the framework:
- Input to base seed sampling.

### 6.2 Base seed points

Directly explicit in code.

$$
\hat{Y}_s = \mathrm{FPS}(S_{\text{src}},\min(N_s,|S_{\text{src}}|))
$$

Where:
- $N_s$ is `base_seed_points`.

What it computes:
- Fixed-size seed set for downstream upsampling.

Why used:
- Provides controllable anchor count for progressive refinement.

Role in the framework:
- Bridge between coarse prediction and dense refinement.

### 6.3 Refinement mapping

Inferred from implementation.

$$
\hat{P} = \mathrm{repeat}(P,r) + \Delta(P,G)
$$

Where:
- $P$ is input point set of current refinement stage.
- $r$ is stage ratio (`step1` or `step2`).
- $\Delta(P,G)$ is offset predicted from local point features and broadcast global feature.

What it computes:
- Geometric upsampling via residual displacement.

Why used:
- Increases point density while preserving coarse geometry.

Role in the framework:
- Produces progressively finer reconstructions.

### 6.4 Point-count transition

Directly explicit in code.

$$
N_s \xrightarrow{r_1} N_f = N_s r_1 \xrightarrow{r_2} N_{f1}=N_s r_1 r_2
$$

Where:
- $r_1$ is `step1`, $r_2$ is `step2`.

What it computes:
- Deterministic cardinality growth through refinement stages.

Why used:
- Controls geometric resolution before exact-size head.

Role in the framework:
- Establishes dense candidate cloud for final exact-size projection.

### 6.5 Optional final input merge

Directly explicit in code.

$$
\hat{Y}_e \leftarrow \mathrm{FPS}([\hat{Y}_e\Vert X],N_t)
$$

Where:
- Applied only if `merge_input_in_final` is enabled.

What it computes:
- Re-injection of observed input points before exact trimming.

Why used:
- Preserves observed details.

Role in the framework:
- Optional final geometric anchoring step.

## 7. Exact-size matching or output-size head

### 7.1 Target count resolution

Directly explicit in code.

$$
N_t=
\begin{cases}
N_{\text{gt}}, & \text{if target source is GT} \\
N_{\text{in}}, & \text{if target source is input} \\
N_{\text{cfg}}, & \text{if target source is config and }N_{\text{cfg}}>0
\end{cases}
$$

Where:
- $N_{\text{cfg}}$ is `exact_target_points`.

What it computes:
- Required final output cardinality.

Why used:
- Enforces exact point-count compatibility with the selected target convention.

Role in the framework:
- Governs exact-size head output dimension.

### 7.2 Exact-size head mechanics

Directly explicit in code.

$$
\hat{Y}_e = \mathrm{FPS}(\hat{Y}_{f1},N_t),\quad \text{if }N_t\le N_{f1}
$$

Directly explicit in code.

$$
\rho = \left\lceil\frac{N_t}{N_{f1}}\right\rceil,
\quad P_{\text{rep}}=\mathrm{repeat}(\hat{Y}_{f1},\rho),
\quad P_{\text{corr}}=P_{\text{rep}}+\Delta_{\text{exact}}(P_{\text{rep}},G)
$$

Directly explicit in code.

$$
\hat{Y}_e = \mathrm{FPS}(P_{\text{corr}},N_t),\quad \text{if }N_t>N_{f1}
$$

Where:
- $\Delta_{\text{exact}}$ is the exact-head residual offset prediction.

What it computes:
- Exact cardinality output with feature-conditioned coordinate correction.

Why used:
- Supports arbitrary target sizes while preserving geometric quality.

Role in the framework:
- Final output-size control stage.

Mechanism type summary:
- Learned prediction: yes, via residual offsets.
- Sampling: yes, via FPS trimming.
- Interpolation: no in exact head itself.
- Truncation by slicing: no.
- Masking: no.

## 8. Loss functions

### 8.1 Chamfer nearest-neighbor distances

Standard formula associated with imported method.

$$
d_1(i)=\min_j \lVert y_i-\hat{y}_j\rVert_2^2,
\qquad
d_2(j)=\min_i \lVert \hat{y}_j-y_i\rVert_2^2
$$

Where:
- $d_1$ maps GT points to nearest predicted points.
- $d_2$ maps predicted points to nearest GT points.

What it computes:
- Bidirectional set discrepancy.

Why used:
- Permutation-invariant distance between point sets.

Role in the framework:
- Base geometric metric used at all supervised stages.

### Role in the framework
- Establishes alignment pressure from prediction to target and target to prediction.

### 8.2 Point-based Chamfer variant used for training

Directly explicit in code.

$$
\mathrm{CD}_p(Y,\hat{Y}) = \frac{1}{2}
\left(
\frac{1}{|Y|}\sum_i \sqrt{d_1(i)}
+
\frac{1}{|\hat{Y}|}\sum_j \sqrt{d_2(j)}
\right)
$$

Where:
- $\mathrm{CD}_p$ is the per-sample value returned by `calc_cd` and used for training losses.

What it computes:
- Symmetric average Euclidean nearest-neighbor distance.

Why used:
- More directly tied to physical distance units than squared form.

Role in the framework:
- Supervises coarse, refinement, and exact outputs.

### Role in the framework
- Encourages geometric closeness at each stage while respecting set permutation invariance.

### 8.3 Squared Chamfer variant used in metrics

Directly explicit in code.

$$
\mathrm{CD}_t(Y,\hat{Y}) = \frac{1}{|Y|}\sum_i d_1(i) + \frac{1}{|\hat{Y}|}\sum_j d_2(j)
$$

Where:
- $\mathrm{CD}_t$ is the squared-distance metric reported in evaluation outputs.

What it computes:
- Symmetric squared set discrepancy.

Why used:
- Common benchmark-style Chamfer reporting.

Role in the framework:
- Evaluation metric alongside $\mathrm{CD}_p$.

### Role in the framework
- Provides a squared-distance diagnostic sensitive to larger deviations.

### 8.4 Stage-wise losses

Directly explicit in code.

$$
\mathcal{L}_e = \mathrm{CD}_p\left(\hat{Y}_e^{\text{sub}},Y^{\text{sub}}\right)
$$

Where:
- Optional subset sampling is controlled by `exact_cd_points`.

What it computes:
- Final-output supervision.

Why used:
- Directly optimizes exact-size prediction quality.

Role in the framework:
- Primary final-stage objective.

### Role in the framework
- Forces the exact-size head output to match target geometry.

Directly explicit in code.

$$
\mathcal{L}_{f1} = \mathrm{CD}_p\left(\hat{Y}_{f1},\mathrm{FPS}(Y,N_{f1})\right)
$$

What it computes:
- Supervision for second refinement output.

Why used:
- Stabilizes deep refinement by direct stage supervision.

Role in the framework:
- Intermediate dense-stage geometric constraint.

### Role in the framework
- Prevents the final stage from carrying all reconstruction burden.

Directly explicit in code.

$$
\mathcal{L}_f = \mathrm{CD}_p\left(\hat{Y}_{f},\mathrm{FPS}(\mathrm{FPS}(Y,N_{f1}),N_f)\right)
$$

What it computes:
- Supervision for first refinement output.

Why used:
- Guides early upsampling geometry.

Role in the framework:
- Intermediate constraint between seed and dense prediction.

### Role in the framework
- Encourages meaningful geometric structure before the last refine stage.

Directly explicit in code.

$$
\mathcal{L}_c = \mathrm{CD}_p\left(\hat{Y}_{c},\mathrm{FPS}(\mathrm{FPS}(\mathrm{FPS}(Y,N_{f1}),N_f),N_c)\right)
$$

What it computes:
- Supervision for coarse prediction.

Why used:
- Anchors global structure at low resolution.

Role in the framework:
- First-stage geometric scaffold loss.

### Role in the framework
- Encourages coherent coarse topology that later refinements can densify.

### 8.5 Weighted total loss

Directly explicit in code.

$$
\mathcal{L}_{\text{total}} = w_c\mathcal{L}_c + w_f\mathcal{L}_f + w_{f1}\mathcal{L}_{f1} + w_e\mathcal{L}_e
$$

Where:
- $w_c=$ `w_coarse`.
- $w_f=$ `w_fine`.
- $w_{f1}=$ `w_fine1`.
- $w_e=$ `w_exact`.
- Default values are $(0.5,0.75,1.0,2.0)$.

What it computes:
- Multi-scale weighted geometric objective.

Why used:
- Balances global scaffold accuracy and final exact-size accuracy.

Role in the framework:
- Scalar loss used in backpropagation.

### Role in the framework
- Controls gradient allocation across reconstruction stages.

Other losses:
- L1, L2, MSE coordinate losses are not used in this exact model forward.
- Feature-space losses are not used.
- Explicit regularization terms are not added in loss expression.

## 9. Optimization mathematics

Total optimization:

Directly explicit in code.

$$
\min_{\theta}\;\mathbb{E}[\mathcal{L}_{\text{total}}]
$$

Where:
- Expectation is over training batches.

Gradient propagation:
- Gradients flow through attention blocks, 1x1 convolutions, refinement residual predictors, and exact head.
- Inferred from implementation: FPS index selection is non-differentiable; gradients flow through the selected coordinates and subsequent differentiable operations.

Optimizer:
- `AdamW` is configured in the notebook-generated training script.

Standard formula associated with imported method.

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
- $g_t=\nabla_{\theta}\mathcal{L}_{\text{total}}$.
- $(\beta_1,\beta_2)=(0.9,0.999)$.
- $\eta=3\times10^{-4}$ by default.
- $\lambda=10^{-5}$ by default.

Why appropriate:
- Adaptive moment scaling helps with heterogeneous gradient magnitudes from multi-stage Chamfer supervision.

Scheduler effect:
- Config fields for decay exist, but default run disables decay (`lr_decay=False`).

## 10. Fine-tuning mathematics

Fine-tuning in this framework means continuing optimization from pretrained weights and optionally pretrained optimizer state.

Directly explicit in code.

$$
\theta_0=\theta_{\text{pretrained}},
\qquad
\theta_{t+1}=\mathrm{AdamWStep}(\theta_t,\nabla_{\theta}\mathcal{L}_{\text{total}})
$$

Where:
- $\theta_{\text{pretrained}}$ comes from checkpoint `net_state_dict`.

What it computes:
- Parameter trajectory initialized at a learned point in parameter space.

Why used:
- Faster convergence and better low-data adaptation.

Role in the framework:
- Adapts a pretrained PointAttN-style completion model to the dental exact dataset.

Frozen layers:
- None are explicitly frozen in the inspected exact training script.
- Consequence: all parameters participate in optimization.

Compared with scratch:
- Same architecture and loss family.
- Different initialization and potentially resumed optimizer moments.

## 11. Tensor shape walkthrough

This is the forward path used by the exact model generated in the notebook.

1. Input:
   - $X\in\mathbb{R}^{B\times3\times N_{\text{in}}}$.
   - Axis meanings: batch, coordinate-channel, point index.

2. Encoder cap:
   - $X_{\text{enc}}\in\mathbb{R}^{B\times3\times N_{\text{enc}}}$.
   - $N_{\text{enc}}=\min(N_{\text{in}},\texttt{max\_encoder\_points})$ if cap active.

3. Encoder outputs:
   - $G\in\mathbb{R}^{B\times512\times1}$.
   - $\hat{Y}_c\in\mathbb{R}^{B\times3\times N_c}$.

4. Seed source concat:
   - $S_{\text{src}}\in\mathbb{R}^{B\times3\times(N_{\text{seed-in}}+N_c)}$.

5. Base seed:
   - $\hat{Y}_s\in\mathbb{R}^{B\times3\times N_s}$.

6. First refinement:
   - $\hat{Y}_f\in\mathbb{R}^{B\times3\times (N_s r_1)}$.

7. Second refinement:
   - $\hat{Y}_{f1}\in\mathbb{R}^{B\times3\times (N_s r_1 r_2)}$.

8. Exact head output BCN:
   - $\hat{Y}_{e,\text{BCN}}\in\mathbb{R}^{B\times3\times N_t}$.

9. Returned output BNC:
   - $\hat{Y}_e\in\mathbb{R}^{B\times N_t\times3}$ after transpose.

10. Ground truth format in training:
    - $Y\in\mathbb{R}^{B\times N_{\text{gt}}\times3}$.

Default numeric transitions in notebook config:
- $N_c=4096$.
- $N_s=5900$.
- $r_1=4$.
- $r_2=8$.
- $N_s r_1 r_2=188800$ before exact-size matching.

## 12. Mathematical meaning of key hyperparameters

### `max_encoder_points`
- Definition: upper bound on encoder token count.
- Mathematical effect: caps $N_{\text{enc}}$, reducing attention complexity.
- Why it matters: controls memory and compute feasibility.
- Increase: better raw coverage, higher cost.
- Decrease: lower cost, possible information loss.

### `encoder_coarse_points`
- Definition: coarse output cardinality $N_c$.
- Mathematical effect: size of coarse scaffold.
- Why it matters: controls global structure resolution.
- Increase: denser scaffold, more compute.
- Decrease: sparser scaffold, harder refinement burden.

### `base_seed_points`
- Definition: seed cardinality $N_s$.
- Mathematical effect: base for multiplicative refinement growth.
- Why it matters: directly scales later point counts.
- Increase: denser refinement input, higher memory/cost.
- Decrease: cheaper but potentially under-detailed refinement.

### Refinement step sizes (`step1`, `step2`)
- Definition: expansion ratios $r_1,r_2$.
- Mathematical effect: $N_{f1}=N_s r_1 r_2$.
- Why it matters: controls pre-exact density.
- Increase: denser predictions, more compute.
- Decrease: coarser intermediate outputs.

### `exact_tail_hidden`
- Definition: hidden width in exact-size head.
- Mathematical effect: capacity of residual offset map $\Delta_{\text{exact}}$.
- Why it matters: controls expressiveness of final correction.
- Increase: more capacity and cost.
- Decrease: less capacity, possibly underfitting fine details.

### Latent dimensions
- Definition: internal channel widths (for example 64, 128, 512 in this exact architecture).
- Mathematical effect: rank/capacity of learned feature transforms.
- Why it matters: affects representational power.

### Neighborhood sizes
- In this exact forward path, explicit neighborhood-size hyperparameters are not used.
- Uncertainty note: neighborhood operators exist in utility libraries but are not invoked by the inspected model forward.

### Attention head counts
- Definition: `nhead` in cross-transformer blocks.
- Mathematical effect: splits feature space into $h$ attention subspaces.
- Why it matters: controls diversity of relational patterns.
- Increase: potentially richer relations, higher cost.

### Loss weights (`w_coarse`, `w_fine`, `w_fine1`, `w_exact`)
- Definition: coefficients in $\mathcal{L}_{\text{total}}$.
- Mathematical effect: rescales gradient contributions by stage.
- Why it matters: sets training emphasis across coarse and fine outputs.
- Increase one weight: prioritizes that stage.

## 13. End-to-end mathematical summary

Directly explicit in code.

$$
\hat{Y} = F(X;\theta)
$$

Where:
- $X$ is partial input.
- $\hat{Y}$ is exact-size completed output.

Inferred from implementation.

$$
F = \mathcal{H}_{\text{exact}} \circ \mathcal{R}_2 \circ \mathcal{R}_1 \circ \mathcal{S} \circ \mathcal{E}
$$

Where:
- $\mathcal{E}$: encoder producing $(G,\hat{Y}_c)$.
- $\mathcal{S}$: seed-source concatenation and FPS seed extraction.
- $\mathcal{R}_1,\mathcal{R}_2$: progressive refinement blocks.
- $\mathcal{H}_{\text{exact}}$: exact-size repeat-correct-trim head.

Why each stage is needed:
- $\mathcal{E}$ provides global coarse structure.
- $\mathcal{S}$ injects observed geometry into refinement initialization.
- $\mathcal{R}_1,\mathcal{R}_2$ densify and improve local geometry.
- $\mathcal{H}_{\text{exact}}$ enforces exact output cardinality.

## 14. Implementation-grounded notes

### Directly explicit in code
- GT-based centering and scaling normalization.
- Optional paired geometric augmentation.
- FPS-based sampling for encoder, seeds, and output sizing.
- Two refinement stages with ratios `step1` and `step2`.
- Exact-size head logic with repeat, residual correction, and FPS trimming.
- Chamfer-based losses $\mathcal{L}_c,\mathcal{L}_f,\mathcal{L}_{f1},\mathcal{L}_e$ and weighted total loss.
- AdamW configuration and checkpoint-based fine-tuning behavior.

### Inferred from implementation
- Conceptual form of refinement offset map $\Delta(P,G)$ and exact-head residual map $\Delta_{\text{exact}}$.
- Interpretation of hard-coded reshape as compatibility remapping.
- Gradient-flow statement that FPS indices are non-differentiable while downstream selected coordinates are differentiable.

### Standard formula associated with imported method
- Multi-head scaled dot-product attention equation.
- FPS greedy farthest-point criterion.
- AdamW update equations.
- Bidirectional Chamfer nearest-neighbor decomposition.

Uncertainty disclosure:
- The notebook writes key exact-model files from embedded code strings. The mathematics here corresponds to those inspected embedded definitions and directly imported utilities. If those cells are edited before writing files, behavior can change.
