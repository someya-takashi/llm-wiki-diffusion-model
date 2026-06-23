---
type: summary
source_path: raw/papers/Stochastic Interpolants_ A Unifying Framework for Flows and Diffusions.md
source_kind: paper
title: "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions"
authors: [Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden]
year: 2023
venue: arXiv preprint (NYU)
ingested: 2026-06-24
tags: [stochastic-interpolants, flow-matching, score-based-generative-models, probability-flow-ode, generative-models]
translation: "[[translations/2024-stochastic-interpolants]]"
---

# 確率的補間：フローと拡散のための統一的枠組み

> 原典: [[translations/2024-stochastic-interpolants]] ・ `raw/papers/Stochastic Interpolants_ ...md`
> 著者・年・会議: Albergo, Boffi, Vanden-Eijnden（NYU）・2023・arXiv:2303.08797

## 一言まとめ

**フロー（決定論的 ODE 生成）と拡散（確率的 SDE 生成）を 1 つの数学的枠組みに統一する理論論文**。任意の 2 密度 $\rho_0,\rho_1$ を**有限時間**で厳密に橋渡しする確率過程「**stochastic interpolant（確率的補間）**」$x_t=I(t,x_0,x_1)+\gamma(t)z$ を定義し、その時間依存密度 $\rho(t)$ が輸送方程式（→ODE 生成）と拡散係数を自在に調整できる前向き・後ろ向き Fokker-Planck 方程式（→SDE 生成）の両方を満たすことを示す。速度場 $b$ とスコア $s$ はいずれも**単純な二乗回帰**の唯一の最小化子で、[[flow-matching]]・rectified flow・[[score-based-generative-models]]・Schrödinger bridge をすべて特別な場合として内包する。

## 背景と問題意識

生成モデリングでは、基底密度 $\rho_0$ から目標密度 $\rho_1$ へサンプルを ODE/SDE で連続変形し、その速度場を回帰で学ぶ手法が中心になった。だが既存の主流——**スコアベース拡散（SBDM, [[score-based-generative-models]]）**——には不満があった：

- **片方がガウス必須**：OU 過程でデータを標準ガウスに写すため、$\rho_0$ はガウスに限られる。任意の 2 密度を結べない。
- **無限時間**：写像が双方向とも理論上無限時間を要し、実際には切り詰め（truncate）が必要。これがノイズスケジュール・時間パラメータ化の煩雑な調整を生む。
- **ODE vs SDE の優劣が未解明**：確率フロー ODE（[[probability-flow-ode]]）と逆時間 SDE は分布レベルで等価だが、近似スコアを使うと両者の密度はずれる。決定論的生成と確率的生成のどちらが良いかは未解決の重要問題。

本論文は、Albergo & Vanden-Eijnden（2022）の確率的補間に**潜在変数 $\gamma(t)z$** と **2 密度のカップリング $\nu$** を加えて拡張し、これらの不満を一掃する。

## 提案手法 / 主張

### 確率的補間と「密度の設計」と「サンプリング法」の分離

2 密度を結ぶ補間を

$$
x_t = I(t,x_0,x_1) + \gamma(t)z,\qquad t\in[0,1]
$$

で定義する（$I$ は $I(0,\cdot)=x_0,\ I(1,\cdot)=x_1$ を満たす補間関数、$\gamma(0)=\gamma(1)=0$、$z\sim\mathsf{N}(0,\mathrm{Id})$）。一例は $x_t=(1-t)x_0+tx_1+\sqrt{2t(1-t)}\,z$。構成上 $x_{t=0}\sim\rho_0$, $x_{t=1}\sim\rho_1$ を厳密に満たす。

核心の主張は「**橋渡しする時間依存密度 $\rho(t)$ の設計**」と「**それをどうサンプルするか**」を分離できること。$\rho(t)$ は次を同時に満たす：

- **1 階輸送方程式** $\partial_t\rho+\nabla\cdot(b\rho)=0$ → 決定論的 **確率フロー ODE** $\dot X_t=b(t,X_t)$ で生成。
- **前向き／後ろ向き Fokker-Planck 方程式**（拡散係数 $\epsilon(t)\geq0$ を任意に選べる）→ **SDE** $dX_t=b_{\mathsf F}dt+\sqrt{2\epsilon}\,dW_t$ で生成。

同じ $\rho(t)$ を ODE でも SDE でも実現でき、両者は**同じ速度 $b$ とスコア $s$ の推定**に依拠する（図1）。

### 速度とスコアは二乗回帰の最小化子

ドリフト係数はすべて条件付き期待値で、単純な二乗目的の唯一の最小化子：

- **速度** $b(t,x)=\mathbb{E}[\dot x_t\mid x_t=x]$ → $\mathcal{L}_b[\hat b]=\int_0^1\mathbb{E}(\tfrac12|\hat b|^2-(\partial_t I+\dot\gamma z)\cdot\hat b)\,dt$ を最小化。
- **スコア** $s(t,x)=\nabla\log\rho=-\gamma^{-1}(t)\,\mathbb{E}(z\mid x_t=x)$ → 新しい二乗目的 $\mathcal{L}_s$ を最小化（[[score-based-generative-models]] の denoising score matching のベクトル場版）。
- **ノイズ除去器** $\eta_z(t,x)=\mathbb{E}(z\mid x_t=x)$ を学べば $s=-\gamma^{-1}\eta_z$。$\gamma^{-1}$ を含まない $\eta_z$ の目的は端点で扱いやすい。

$(x_0,x_1)\sim\nu$ のサンプルさえあれば $x_t$ を任意の $t$ で生成でき、シミュレーション不要で回帰できる（[[flow-matching]] と同じ利点）。

### 尤度の制御（ODE と SDE の差）

二乗目的の最小化が、**SDE ベース生成モデルの尤度を制御**する（KL を上から抑える）一方、**ODE ベースではドリフト回帰だけでは不十分**で、追加で Fisher ダイバージェンスも最小化せねばならない（より厳格）。SDE の拡散係数 $\epsilon(t)$ は尤度を最大化するよう最適調整できる。これは「近似が不完全なとき確率的生成（SDE）の方が誤差に頑健」という実践的含意を持つ。

### 既存手法の統一・内包

- **スコアベース拡散（SBDM）**：時間の再パラメータ化で**片側補間**（$\rho_0$=ガウス）として書き直せ、有限区間に圧縮するときの特異性を除去（§5.1）。
- **rectified flow / flow matching**：[41]（rectified flow）・[39]（flow matching）は本枠組みの特別な場合。本論文はノイズを**バイアスなく**（潜在変数＋調整可能拡散係数で）入れる点が違う。§5.3 で rectified flow の整流をバイアスのない形に再構成。
- **Schrödinger bridge**：補間関数 $I$ を明示的に最適化すると、2 密度間の Schrödinger bridge（エントロピー正則化最適輸送）を復元する（§3.4、Benamou-Brenier 定式化）。ただし任意の固定補間でバイアスのない生成モデルになるので、この最適化は実際には不要。
- **denoising（ノイズ除去）**：補間からノイズ除去器のベイズ最適推定量を導け、反復して生成モデルにできる（§5.2）。

## 実験結果と知見

- **2D・128 次元ガウス混合**（§7.1–7.2）：決定論的（ODE）と確率的（SDE）モデルを比較。潜在変数 $\gamma(t)z$ と拡散係数 $\epsilon$ が密度推定・サンプル品質に与える役割を定量化。ガウス混合ではドリフトが解析的に計算でき（付録A）、学習した場との比較ができる。
- **画像生成**（§7.3）：ガウス基底からの生成と、$\rho_0=\rho_1$ の**ミラー補間**を実証。
- 一般に、**確率的（SDE）ダイナミクスはスコアの近似誤差に頑健**で、適切な $\epsilon$ で品質が上がる。決定論的 ODE は厳密尤度・適応積分という利点を持つ。両者の使い分けの理論的基盤を与えた。

## 限界・批判的視点

- **重い理論論文**：測度論・PDE・確率解析が前提で、付録B の証明（約 1000 行）は非常に高密度。実装の即効性より統一的理解を与える論文。
- **ドリフトの解析解はガウス混合に限る**：一般の密度では速度・スコアを数値的に学ぶ必要があり、学習誤差の伝播が品質を左右する。
- **Schrödinger bridge 最適化は追加コスト**：理論的には美しいが、実用上は固定補間で十分という立場。
- 本枠組みは「設計の自由度」が大きい反面、補間 $I$・$\gamma$・$\epsilon$ の良い選び方は経験則に依る部分が残る。

## 用語と略称

- **stochastic interpolant（確率的補間）** = 2 密度を有限時間で結ぶ過程 $x_t=I(t,x_0,x_1)+\gamma(t)z$。
- **transport equation（TE, 輸送方程式）** = $\partial_t\rho+\nabla\cdot(b\rho)=0$。決定論的生成（確率フロー ODE）に対応。
- **Fokker-Planck equation（FPE）** = 拡散項を含む密度発展方程式。確率的生成（SDE）に対応。$\epsilon(t)$ = 調整可能拡散係数。
- **velocity $b$ / score $s$ / denoiser $\eta_z$** = 速度場 / スコア $\nabla\log\rho$ / ノイズ除去器 $\mathbb{E}(z|x_t)$。いずれも二乗回帰で学習。
- **one-sided / mirror interpolant** = 片側補間（$\rho_0$=ガウス、SBDM に対応）／ ミラー補間（$\rho_0=\rho_1$）。
- **SBDM** = Score-Based Diffusion Models（[[score-based-generative-models]]）。**OU 過程** = Ornstein-Uhlenbeck（ガウスへ劣化させる拡散）。
- **Schrödinger bridge** = エントロピー正則化最適輸送（2 密度を結ぶ最小コスト拡散）。**Doob の h 変換** = 確率的橋の構成に通常要る（本枠組みは回避）。
- **probability flow（確率フロー）ODE** = [[probability-flow-ode]]。**Fisher ダイバージェンス** = スコアの差の二乗期待。**Benamou-Brenier** = 最適輸送の流体力学的定式化。

## 既存知識との接続

- [[stochastic-interpolants]]：本論文がその概念ページのランドマーク原典。flows と diffusions を統一する枠組みを定義した。
- [[flow-matching]]：flow matching・rectified flow は確率的補間の特別な場合（潜在変数なし・ガウス基底）。本論文はそれらを一般化し、ノイズをバイアスなく入れる。SD3（[[summaries/2024-sd3]]）の rectified flow もこの一族。
- [[score-based-generative-models]]：SBDM は片側補間として再構成でき、無限時間・ガウス必須・特異性という制約を有限時間・任意基底で解消する。スコアの二乗回帰目的も新しく与える。
- [[probability-flow-ode]]：輸送方程式から出る確率フロー ODE は決定論的生成。本論文は ODE（決定論）と SDE（確率）を同じ密度の異なる実現として明示的に統一し、尤度制御の差を示した。
- [[denoising-diffusion]]：ノイズ除去器 $\eta_z$ とスコアの関係は拡散の denoising と整合。

## 関連ページ

- [[concepts/stochastic-interpolants]] — 確率的補間（本論文がランドマーク）
- [[concepts/flow-matching]] — flow matching / rectified flow（特別な場合）
- [[concepts/score-based-generative-models]] — SBDM（片側補間として内包）
- [[concepts/probability-flow-ode]] — 確率フロー ODE（決定論的生成）
- [[summaries/2023-flow-matching]] — Flow Matching（関連する一般化）
- [[summaries/2024-sd3]] — SD3（rectified flow を大規模実用化）
- [[summaries/2021-score-sde]] — Score-SDE（SBDM の連続時間定式化）
