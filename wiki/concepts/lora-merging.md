---
type: concept
aliases: [LoRA Merging, LoRA Fusion, LoRA Merge, Gradient Fusion, Weight Fusion, ED-LoRA, ZipLoRA, LoRAHub]
tags: [lora-merging, low-rank-adaptation, multi-concept-customization, subject-driven-generation, generative-models]
related:
  - "[[low-rank-adaptation]]"
  - "[[multi-concept-customization]]"
  - "[[subject-driven-generation]]"
  - "[[controllable-generation]]"
  - "[[latent-diffusion]]"
summaries:
  - "[[summaries/2023-custom-diffusion]]"
  - "[[summaries/2023-mix-of-show]]"
  - "[[summaries/2024-ziplora]]"
updated: 2026-06-25
---

# LoRA Merging / Fusion（複数 LoRA の重みマージ／融合）

**LoRA merging / fusion（LoRA のマージ／融合）** とは、**別々に学習された複数の [[low-rank-adaptation]] LoRA（低ランク適応, 重み変化を $\Delta W=BA$ で表す軽量 fine-tune）を、1 つの重みにまとめて 1 枚の画像に複数概念を同時生成する**ための手法群である。コミュニティに無数の単一概念 LoRA（特定キャラ・物体・画風など）が共有された結果、それらを「**重みのレベルで合成する**」ことが自然な課題になった。本ページは [[multi-concept-customization]]（多概念カスタマイズ）の中でも特に **(a) 重みマージ／融合系統**を細粒度に扱う。重みを混ぜずに推論時に合成する系統（注意制御・復号中心）は [[multi-concept-customization]] を参照。本 wiki のランドマークは **Mix-of-Show**（[[summaries/2023-mix-of-show]]）と **ZipLoRA**（[[summaries/2024-ziplora]]）。

## なぜ素朴なマージは破綻するのか

最も素朴な手法は **LoRA Merge（線形和）**：複数 LoRA の重み変化を加重平均してベースに差す。

$$
W'=W_0+\sum_i w_i\,\Delta W_i=W_0+\sum_i w_i B_i A_i,\qquad \sum_i w_i=1
$$

これには 2 つの典型的な破綻がある。

- **identity loss（同一性損失）**：$n$ 概念を平均すると各概念の寄与が $\frac1n$ に薄まり、個々の identity が失われる。Mix-of-Show が分析した課題。
- **signal interference（信号干渉）**：合成対象の LoRA の**列方向（出力次元）の cosine 類似度が高い**と、直和したときに互いの信号が干渉して破綻する。ZipLoRA が分析した課題（content と style を素朴に足すと両方劣化する）。

さらに LoRA 数が増えるほど不安定になり細部が崩れる（[[multi-concept-customization]] で言う concept vanishing / confusion）。マージ系統の研究は「**どう混ぜれば各 LoRA の挙動を保てるか**」を競う。

## 系統と代表手法

### (0) 源流：Custom Diffusion の閉形式マージ

LoRA を対象にする以前に、「**別々に学習した概念の重みを最小二乗で整合させて混ぜる**」発想を最初に示したのが **Custom Diffusion**（[[summaries/2023-custom-diffusion]]）である。各概念で別々に fine-tune した cross-attention の $W^k,W^v$ を、次の制約付き最小二乗で結合する：

$$
\hat W=\operatorname*{arg\,min}_{W}\|WC_{\text{reg}}^{\top}-W_0C_{\text{reg}}^{\top}\|_F\quad\text{s.t. }WC^{\top}=V
$$

「正則化キャプション $C_{\text{reg}}$ では元モデル $W_0$ の出力を保ち、対象概念の単語 $C$ は各概念の fine-tune 済み value $V$ に一致させる」という目的で、Lagrange 乗数法により**閉形式**で解ける（$\hat W=W_0+\mathbf v^\top\mathbf d$、約 2 秒）。LoRA の $BA$ ではなく cross-attention 重みそのものを対象にする点が後続と異なるが、「単独挙動を保ちつつ重みを 1 つに畳む」という目的関数は (2) Mix-of-Show の gradient fusion と本質的に同型で、重みマージ系統の出発点にあたる。

### (1) 素朴な線形和（LoRA Merge / weighted sum）

Ryu（kohya 系）由来の重み付き和 $W'=W_0+\sum_i w_i\Delta W_i$。実装は容易だが上記 identity loss / interference に直撃され、2〜3 個までが実用上の限界。後続手法のベースラインになる。

### (2) Gradient Fusion（推論挙動を整合させる融合）— Mix-of-Show

**Mix-of-Show**（[[summaries/2023-mix-of-show]]）は線形和をやめ、「**融合モデルが各概念を単独 LoRA と同じように推論する**」よう融合重みを最適化する。各概念をサンプリングして各層の入出力特徴 $X_i$ を集め、層ごとに次の最小二乗を解く（LBFGS）：

$$
W=\operatorname*{arg\,min}_{W}\sum_{i=1}^{n}\|(W_{0}+\Delta W_{i})X_{i}-WX_{i}\|^{2}_{F}
$$

すなわち各概念の単独出力 $(W_0+\Delta W_i)X_i$ を 1 枚の重み $W$ で再現する。重み平均が薄める identity を大幅に保ち（image-alignment の融合後劣化を −0.094→−0.025）、**理論上無制限**の概念融合を可能にする。

あわせて Mix-of-Show は **ED-LoRA（embedding-decomposed LoRA, 埋め込み分解 LoRA）** を提案する。通常の LoRA は概念の identity を LoRA 重みに過度に押し込むため、意味的に近い embedding が別概念に射影され **concept conflict（概念衝突）** を起こす。ED-LoRA は概念トークンを layer-wise embedding と multi-word 表現 $V=V_{rand}^{+}V_{class}^{+}$ に分解し、in-domain essence を embedding 側に残して衝突を防ぐ。さらに多概念サンプリングの属性結合を解く **regionally controllable sampling（領域制御可能サンプリング, region-aware cross-attention）** も導入し、これは [[summaries/2024-lora-composer]] の領域注入の源流になった。

### (3) 学習係数マージ（列の干渉を最小化）— ZipLoRA

**ZipLoRA**（[[summaries/2024-ziplora]]）は **content LoRA ＋ style LoRA** を安全にマージし「任意被写体×任意スタイル」を実現する（SDXL ベース）。2 つの観察に基づく：(1) LoRA の $\Delta W$ は**疎**（90% を 0 にしても品質維持）、(2) **列の cosine 類似度が高いと直和が破綻**する。そこで層・列ごとに学習可能な **merger 係数 $m_c,m_s$** を導入し、

$$
\Delta W_m=m_c\otimes\Delta W_c+m_s\otimes\Delta W_s
$$

ベースと個別 LoRA は凍結したまま係数のみを最適化する。損失は「個別 LoRA の挙動を保ちつつ、両 LoRA が同じ列を使わない（直交化する）」よう設計される：

$$
\mathcal L_{merge}=\|(D{\oplus}L_m)(x_c,p_c)-(D{\oplus}L_c)(x_c,p_c)\|_2+\|(D{\oplus}L_m)(x_s,p_s)-(D{\oplus}L_s)(x_s,p_s)\|_2+\lambda\sum_i|m_c^{(i)}\cdot m_s^{(i)}|
$$

hyperparameter-free・約 100 step・joint 学習比で 10× 高速。直和マージ・joint training・StyleDrop を subject/style fidelity で上回る。

### (4) 係数学習による汎用融合 — LoRAHub ほか

LoRAHub は下流タスクに合わせて複数 LoRA の結合係数を（勾配フリー最適化で）学習する汎用的な重みベース融合。ZipLoRA の「係数を学習する」発想と同系統だが、画像の content×style 特化ではなくタスク適応寄り。

## 「重みを混ぜない」系統との対比

LoRA を 1 枚に合成する手法は、重みマージ（本ページ）以外に 2 系統ある（詳細は [[multi-concept-customization]]）：

- **(b) 訓練不要の注意制御**：LoRA-Composer（[[summaries/2024-lora-composer]]）。重みを混ぜず、推論時に U-Net の注意を領域ごとに操作。
- **(c) 復号中心の合成**：Multi-LoRA Composition（[[summaries/2024-multi-lora-composition]]）の LoRA Switch / Composite。各ノイズ除去ステップで LoRA を切替・平均。

重みマージは「**1 度融合すれば追加推論コストがない**（推論は 1 モデル）」のが利点で、(b)(c) は「**重みを保つので元 LoRA を壊さず柔軟**」だが推論が重い、というトレードオフがある。

## 既存知識との接続

- [[low-rank-adaptation]]：マージ対象は単一概念の LoRA。$\Delta W=BA$ がプラグ&プレイで共有・加算可能だからこそマージが成立する。ED-LoRA は LoRA の派生。
- [[multi-concept-customization]]：本ページは多概念合成の「(a) 重みマージ／融合」系統の詳細版。注意制御・復号中心系は親ページに。
- [[subject-driven-generation]]：マージ対象の単一概念は DreamBooth/Textual Inversion 系の personalization で作られる。ZipLoRA は subject（content）と style を別々に学習して合成する。
- [[controllable-generation]]：Mix-of-Show の regionally controllable sampling は空間条件制御を多概念マージに組み合わせたもの。
- [[latent-diffusion]]：Mix-of-Show は Stable Diffusion 系（実写 Chilloutmix・アニメ Anything-v4）、ZipLoRA は SDXL 上で動く。

## 参考文献（summaries）

- [[summaries/2023-custom-diffusion]] — Custom Diffusion（cross-attention K/V の閉形式制約付き最小二乗マージ。重みマージ系の源流）
- [[summaries/2023-mix-of-show]] — Mix-of-Show（ED-LoRA＋gradient fusion、分散型多概念カスタマイズ）
- [[summaries/2024-ziplora]] — ZipLoRA（content+style LoRA の学習係数マージ）
- [[summaries/2024-multi-lora-composition]] — Multi-LoRA Composition（重みマージのベースライン批判・decoding-centric）
- [[summaries/2024-lora-composer]] — LoRA-Composer（重みを混ぜない注意制御系）
