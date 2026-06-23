---
type: summary
source_path: raw/papers/Score-Based Generative Modeling through Stochastic Differential Equations.md
source_kind: paper
title: "Score-Based Generative Modeling through Stochastic Differential Equations"
authors: [Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole]
year: 2021
venue: ICLR 2021 (Outstanding Paper Award)
ingested: 2026-06-23
tags: [score-based-generative-models, probability-flow-ode, diffusion-sampling, controllable-generation, generative-models, score-sde]
translation: "[[translations/2021-score-sde]]"
---

# Score-Based Generative Modeling through SDEs（Score-SDE）

> 原典: [[translations/2021-score-sde]] ・ `raw/papers/Score-Based Generative Modeling through Stochastic Differential Equations.md`（arXiv:2011.13456）
> 著者・年・会議: Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole・2021・ICLR 2021（Outstanding Paper Award）
> コード: https://github.com/yang-song/score_sde

## 一言まとめ

拡散モデルの 2 大系統だった **SMLD（NCSN, [[score-based-generative-models]]）と DDPM（[[denoising-diffusion]]）を、連続時間の確率微分方程式（SDE, stochastic differential equation）として一つの枠組みに統一**した理論的支柱の論文。データにノイズを加える順方向 SDE と、それを逆向きにたどる**逆時間 SDE**（スコア ∇ₓlog pₜ(x) だけで定まる）を中心に据え、(1) SMLD=VE-SDE / DDPM=VP-SDE という対応、(2) **predictor-corrector サンプラー**、(3) SDE と同じ分布をたどる決定論的な **確率フロー ODE**（[[probability-flow-ode]]、厳密尤度・可逆 encode を可能に）、(4) 条件付き逆時間 SDE による**可制御生成**（[[controllable-generation]]）を導いた。CIFAR-10 で FID 2.20 / IS 9.89 の当時 SOTA、初の 1024² 生成を達成。

## 背景と問題意識

拡散モデルには見かけ上別個の 2 系統があった：

- **SMLD / NCSN**（Song & Ermon）：各ノイズスケールで**スコア**（対数密度の勾配）を推定し、**Langevin 動力学**でサンプリング（[[score-based-generative-models]]）。
- **DDPM**（Ho ら）：ノイズ破損の各ステップを逆転する確率モデルを学習（[[denoising-diffusion]]）。

両者は「複数ノイズスケールでデータを摂動し、逆転を学ぶ」点で似ており、連続状態空間では DDPM も暗黙にスコアを学んでいる。だが**統一的理解・サンプリングの自由度・厳密尤度・可制御生成**が欠けていた。本論文は「ノイズスケールを有限個から**無限個（連続時間）** へ一般化する」ことでこれらを一挙に与える。

## 提案手法 / 主張

### 順方向 SDE と逆時間 SDE

データにノイズを連続的に加える順方向 SDE：$\mathrm{d}\mathbf{x}=\mathbf{f}(\mathbf{x},t)\mathrm{d}t+g(t)\mathrm{d}\mathbf{w}$（$\mathbf{f}$=ドリフト、$g$=拡散係数、$\mathbf{w}$=ブラウン運動）。生成は **Anderson の逆時間 SDE**：

$$
\mathrm{d}\mathbf{x}=[\mathbf{f}(\mathbf{x},t)-g(t)^2\nabla_\mathbf{x}\log p_t(\mathbf{x})]\mathrm{d}t+g(t)\mathrm{d}\bar{\mathbf{w}}
$$

これは**周辺分布のスコア $\nabla_\mathbf{x}\log p_t(\mathbf{x})$ だけで定まる**。スコアは時間依存ネット $\mathbf{s}_\theta(\mathbf{x},t)$ を連続スコアマッチング（式7）で学習して推定する。

### SMLD = VE-SDE、DDPM = VP-SDE、新提案 sub-VP

ノイズの加え方で SDE が分類される（[[score-based-generative-models]]）：

- **VE-SDE（Variance Exploding）= SMLD**：分散が時間とともに爆発。
- **VP-SDE（Variance Preserving）= DDPM**：分散が有界（単位分散保存）。
- **sub-VP-SDE**（新提案）：分散が常に VP 以下。尤度に強い。

これにより「SMLD と DDPM は同じ枠組みの異なる SDE の離散化」と明快に位置づけられた。

### サンプリング：predictor-corrector（PC）

逆時間 SDE を解く新サンプラー（[[diffusion-sampling]]）：
- **predictor**：数値 SDE ソルバー（逆拡散サンプラー＝順過程と同じ離散化。DDPM の祖先的サンプリングは逆時間 VP-SDE の一離散化にすぎないと示す）。
- **corrector**：スコアベース MCMC（Langevin）で各ステップの周辺分布を補正。
- **PC** はこの 2 つを交互適用。同計算量で predictor のみ・corrector のみを上回る。

### 確率フロー ODE（[[probability-flow-ode]]）

任意の拡散 SDE には、**同じ周辺分布をたどる決定論的 ODE** が対応する：

$$
\mathrm{d}\mathbf{x}=\Big[\mathbf{f}(\mathbf{x},t)-\tfrac{1}{2}g(t)^2\nabla_\mathbf{x}\log p_t(\mathbf{x})\Big]\mathrm{d}t
$$

これがニューラル ODE であることから、**厳密な対数尤度計算**（瞬間変数変換＋Skilling-Hutchinson トレース推定）、**一意に識別可能な encode**、潜在補間、black-box ODE ソルバーによる高速サンプリング（評価回数を 90% 削減可）が得られる。**DDIM（[[diffusion-sampling]]）の ODE 視点の理論的根拠**にあたる。

### 可制御生成（[[controllable-generation]]）

条件 $\mathbf{y}$ 付き逆時間 SDE はスコアを足すだけで作れる：$\nabla_\mathbf{x}\log p_t(\mathbf{x}\mid\mathbf{y})=\nabla_\mathbf{x}\log p_t(\mathbf{x})+\nabla_\mathbf{x}\log p_t(\mathbf{y}\mid\mathbf{x})$。time-dependent classifier を学習すれば class-conditional 生成＝**[[classifier-guidance]] の連続時間における一般化／理論的根源**。inpainting・着色は無条件モデルで解ける逆問題として実現。

## 実験結果と知見

- **CIFAR-10 無条件**：NCSN++ cont. (deep, VE) が **FID 2.20 / IS 9.89**（当時 SOTA、条件付き BigGAN すら凌駕）。
- **尤度**：DDPM++ cont. (deep, sub-VP) が **2.99 bits/dim**（一様逆量子化 CIFAR-10 で当時最高、最尤学習なしでも）。確率フロー ODE による厳密尤度。
- **高解像度**：スコアベースモデルから初の **CelebA-HQ 1024×1024** 生成。
- **サンプラー比較**（表1）：PC（PC1000）が同計算量で predictor のみ（P2000）・corrector のみ（C2000）を一貫して上回る。
- アーキ改良（FIR、BigGAN 残差ブロック、スキップ再スケール、深層化、連続時間条件付けの Fourier 特徴）が効く。

## 限界・批判的視点

- **GAN より遅い**：PC・ODE サンプラーで改善しても逐次評価で GAN の一発生成に及ばない。安定学習と高速サンプリングの両立は今後の課題。
- **ハイパーパラメータが多い**：predictor/corrector の種類・ステップ配分・signal-to-noise ratio $r$・$\epsilon$ など、サンプラーの自由度が高く調整負担が大きい。
- **SDE の選択依存**：VE はサンプル品質、VP/sub-VP は尤度に強く、領域ごとに使い分けが要る。確率フロー ODE サンプラーは VE で品質が落ちやすい。
- 一般逆問題のスコア $\nabla\log p_t(\mathbf{y}\mid\mathbf{x})$ は近似が必要。

## 用語と略称

- **SDE** = Stochastic Differential Equation（確率微分方程式）・**ODE** = 常微分方程式
- **SMLD** = Score Matching with Langevin Dynamics（= NCSN, [[score-based-generative-models]]）
- **DDPM** = Denoising Diffusion Probabilistic Models（[[denoising-diffusion]]）
- **VE / VP / sub-VP SDE** = Variance Exploding / Preserving / sub-Variance Preserving SDE
- **score（スコア）** = $\nabla_\mathbf{x}\log p_t(\mathbf{x})$、データ対数密度の勾配
- **reverse-time SDE（逆時間 SDE）** = データ生成のためにノイズから戻る SDE（Anderson）
- **probability flow ODE** = SDE と同じ周辺分布をたどる決定論 ODE（[[probability-flow-ode]]）
- **PC sampler** = Predictor-Corrector サンプラー（[[diffusion-sampling]]）
- **Langevin dynamics** = スコアに沿った確率的サンプリング（corrector に使用）
- **Fokker-Planck / Kolmogorov 前進方程式** = 周辺密度 $p_t$ の時間発展を支配する PDE
- **bits/dim** = 次元あたりビット数（対数尤度の指標、低いほど良い）・**FID / IS** = 生成品質指標

## 関連ページ

- [[concepts/score-based-generative-models]] — SDE 統一枠組み（VE/VP/sub-VP）の本拠（本論文が主要原典）
- [[concepts/probability-flow-ode]] — 確率フロー ODE・厳密尤度・可逆 encode（本論文が主要原典）
- [[concepts/diffusion-sampling]] — predictor-corrector・逆拡散・ODE サンプラー
- [[concepts/controllable-generation]] — 条件付き逆時間 SDE による class-conditional・inpainting・着色・逆問題
- [[concepts/denoising-diffusion]] — DDPM = VP-SDE の離散化
- [[concepts/classifier-guidance]] — 条件付き逆時間 SDE（time-dependent classifier）の一般化関係
