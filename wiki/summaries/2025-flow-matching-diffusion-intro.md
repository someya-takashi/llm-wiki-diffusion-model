---
type: summary
source_path: raw/papers/An Introduction to Flow Matching and Diffusion Models.md
source_kind: article
title: "An Introduction to Flow Matching and Diffusion Models"
authors: [Peter Holderrieth, Ezra Erives]
year: 2025
venue: "MIT 6.S184 講義ノート (arXiv:2506.02070)"
ingested: 2026-06-25
tags: [flow-matching, score-based-generative-models, probability-flow-ode, denoising-diffusion, generative-models]
translation: "[[translations/2025-flow-matching-diffusion-intro]]"
---

# Flow Matching と拡散モデル入門（MIT 6.S184 講義ノート）

> 原典: [[translations/2025-flow-matching-diffusion-intro]] ・ `raw/papers/An Introduction to Flow Matching and Diffusion Models.md`
> 著者・年・出典: Peter Holderrieth, Ezra Erives（MIT 6.S184: Generative AI With Stochastic Differential Equations, 2025）, arXiv:2506.02070
> コース: https://diffusion.csail.mit.edu/

## 一言まとめ

flow matching（フローマッチング）と denoising diffusion models（ノイズ除去拡散モデル）を、**「ノイズ → データ」の変換を ODE/SDE のシミュレーションとして捉える 1 つの枠組み**から自己完結的に導く教科書的な講義ノート。個別の手法論文ではなく、本 wiki の [[flow-matching]]・[[score-based-generative-models]]・[[probability-flow-ode]]・[[denoising-diffusion]]・[[stochastic-interpolants]]・[[classifier-free-guidance]]・[[diffusion-model-architecture]] を横断する**統一的リファレンス**。

## 背景と問題意識

拡散モデルの古典的な提示（DDPM・score-SDE 等）は、順過程（forward process）・時間反転・ELBO・離散時間マルコフ連鎖など、論文ごとに記法も枠組みもバラバラで、初学者には全体像が掴みにくい。本ノートは、近年の最先端モデル（Stable Diffusion 3・Movie Gen）が実際に採る **flow matching / stochastic interpolants の現代的な定式化**を出発点に据え、確率微分方程式（SDE）の最小限の道具立て（連続の方程式・Fokker-Planck 方程式・Langevin 動力学）だけで、拡散モデルをその特別な場合として導く。「データからノイズを作るのは簡単、ノイズからデータを作るのが生成モデリング」（Song ら）という標語が貫く。

前提知識は学部後半〜修士の確率論程度で、Appendix A に確率論の復習（ランダムベクトル・条件付き密度と期待値・タワー性質）、Appendix B に Fokker-Planck 方程式の自己完結的な証明を備える。

## 提案手法 / 主張（4 章の流れ）

### §2 flow モデルと拡散モデル＝ODE/SDE の数値シミュレーション

- **flow モデル**：ベクトル場 $u_t^\theta$ を持つ ODE $\frac{d}{dt}X_t=u_t^\theta(X_t)$ を $X_0\sim p_{\rm init}$ から **Euler 法**でシミュレーションし、$X_1\sim p_{\rm data}$ を目指す。ニューラルネットは（フローでなく）**ベクトル場**を表す。
- **拡散モデル**：ODE に Brownian motion（ブラウン運動＝連続ランダムウォーク）由来の確率項を足した **SDE** $\mathrm{d}X_t=u_t^\theta(X_t)\mathrm{d}t+\sigma_t\mathrm{d}W_t$ を **Euler-Maruyama 法**でシミュレーション。Ornstein-Uhlenbeck 過程が代表例。$\sigma_t=0$ の拡散モデルは flow モデル。

### §3 学習ターゲットの構成（核心：周辺化トリック）

直接は扱えない学習ターゲットを、手で導ける条件付き量から組み立てる。

- **conditional probability path** $p_t(\cdot|z)$：1 つのデータ点 $z$ とノイズ $p_{\rm init}$ を補間する分布（例: ガウスパス $\mathcal N(\alpha_t z,\beta_t^2 I)$）。これを $z\sim p_{\rm data}$ で周辺化すると **marginal probability path** $p_t$ が得られ、$p_0=p_{\rm init}$、$p_1=p_{\rm data}$ を補間する。
- **周辺化トリック（定理 10）**：手で導ける **conditional vector field** $u_t^{\rm target}(x|z)$ を確率パスで重み付き周辺化した **marginal vector field** が、marginal path を生成する（**連続の方程式**で証明）。
- **SDE 拡張トリック（定理 13）**：marginal score $\nabla\log p_t$ を足せば、同じ確率パスに従う SDE に拡張できる（**Fokker-Planck 方程式**で証明）。静的パスの特別な場合が **Langevin 動力学**。

<figure>

![](../../raw/assets/2025-flow-matching-diffusion-intro/conditional_vs_marginal.png)

<figcaption>図5（再掲）: 上＝条件付きパス（1 データ点 z へ収束）、下＝周辺パス（データ分布全体＝チェス盤へ収束）。条件付きを z で周辺化したものが周辺パス、というのが §3 の核心。</figcaption>
</figure>

### §4 学習＝条件付きターゲットへの回帰

- **Flow Matching（定理 18）**：扱えない marginal vector field への回帰 $\mathcal L_{\rm FM}$ は、手で書ける conditional vector field への回帰 $\mathcal L_{\rm CFM}$ と**定数差で勾配が一致**する。よって「データ点とノイズを引いて二乗誤差を取る」だけで学習でき、ガウス CondOT パス（$\alpha_t=t,\beta_t=1-t$）では損失が $\|u_t^\theta(tz+(1-t)\epsilon)-(z-\epsilon)\|^2$ という単純形に。Stable Diffusion 3・Movie Gen がこれを使う。
- **Score Matching（定理 20）**：同じ論法で score network を **denoising score matching** で学習。ガウスパスでは $s_t^\theta$ と $u_t^\theta$ は**互いに変換可能**（命題 1）で、別々に学習する必要はなく、ノイズ予測 $\epsilon_t^\theta$ パラメータ化（DDPM の $\|\epsilon_t^\theta-\epsilon\|^2$）に帰着する。marginal vector field の式が **確率フロー ODE**。
- **§4.3 文献ガイド**：離散時間 vs 連続時間（ELBO は連続時間でタイト）、「順過程」＝ガウス確率パスの特定の作り方、時間反転 vs Fokker-Planck 直接構成、flow matching と stochastic interpolants の関係を整理。**拡散モデルはガウス初期分布・ガウスパスに限られるが、flow matching / stochastic interpolants は任意の $p_{\rm init}\to p_{\rm data}$ を許す**点を強調。

### §5 画像生成器の構築

- **guidance**：条件 $y$ を強める **classifier-free guidance（CFG）** を、$\tilde u_t(x|y)=(1-w)u_t^{\rm target}(x|\varnothing)+w\,u_t^{\rm target}(x|y)$ として導出（$\varnothing$＝無条件ラベル、$w>1$＝ガイダンススケール）。Bayes 則で「分類器」$\nabla\log p_t(y|x)$ の寄与を増幅する解釈。
- **アーキテクチャ**：[[diffusion-model-architecture]] の **U-Net**・**DiT**・**MM-DiT**、潜在空間で動く **latent diffusion**（[[latent-diffusion]]）、CLIP/T5 によるテキスト埋め込み。
- **大規模モデル概観**：Stable Diffusion 3（CFM＋MM-DiT＋3 テキストエンコーダ、8B）と Movie Gen Video（CondOT＋temporal autoencoder＋DiT、30B）。

## 実験結果と知見

教科書のため新規実験はないが、要点として：(1) flow matching の条件付き回帰が周辺ターゲットと同じ勾配を与えるという**等式**（連続時間ゆえ下界でなく等式）、(2) ガウスパスで flow と score が変換可能ゆえ、学習後に決定論的（確率フロー ODE）・確率的（SDE）サンプリングを選べる、(3) CFG は $w>1$ でデータ分布から逸れるが経験的に条件忠実度が上がる（図11・12）。

## 限界・批判的視点

- 連続データ専用で、テキストのような離散データ（自己回帰 LM）は扱わない。
- 拡散モデル（denoising diffusion）はガウス初期分布・ガウス確率パスに限定される——一般の $p_{\rm init}$／確率パスを扱うには flow matching / stochastic interpolants（[[stochastic-interpolants]]）が必要。
- 教科書的入門であり、最適輸送パスの厳密理論・高次サンプラー（[[diffusion-sampling]] の Heun・DPM-Solver 等）・ノイズスケジュール設計（[[noise-schedule]] の EDM 等）の詳細には踏み込まない。これらは個別の原典（[[summaries/2023-flow-matching]]・[[summaries/2024-stochastic-interpolants]]・[[summaries/2022-edm]] 等）を参照。

## 既存 wiki との接続

本ノートは wiki の理論系ページを束ねる「地図」として機能する。§3 の conditional→marginal 構成と §4.1 の CFM は [[flow-matching]]（Lipman ら [[summaries/2023-flow-matching]]）の定式化そのもの、ODE↔SDE 統一と Langevin 拡張は [[stochastic-interpolants]]（[[summaries/2024-stochastic-interpolants]]）の見方、§4.2 の denoising score matching・確率フロー ODE は [[score-based-generative-models]] と [[probability-flow-ode]]、§4.3 の順過程・時間反転は [[denoising-diffusion]]（DDPM）の古典的提示に対応する。§5 の CFG は [[classifier-free-guidance]]、U-Net/DiT/MM-DiT は [[diffusion-model-architecture]]、latent space は [[latent-diffusion]]、大規模モデルは [[text-to-image-generation]]（[[summaries/2024-sd3]] SD3）に繋がる。

## 用語と略称

- **ODE / SDE** = Ordinary / Stochastic Differential Equation（常微分方程式／確率微分方程式）。flow は ODE、diffusion は SDE のシミュレーション。
- **vector field（ベクトル場） $u_t$**：各時刻・各点での速度。ニューラルネットが表す対象。
- **flow（フロー） $\psi_t$**：ODE の解＝初期点を時刻 $t$ の位置へ写す写像。
- **Brownian motion（ブラウン運動, Wiener 過程） $W_t$**：連続なランダムウォーク。SDE の確率項を駆動。
- **Ornstein-Uhlenbeck（OU）過程**：線形ドリフト＋定数拡散の SDE。ガウスへ収束。
- **conditional / marginal probability path**：1 データ点を補間する条件付きパス $p_t(\cdot|z)$ と、それを周辺化した周辺パス $p_t$。
- **連続の方程式（continuity equation）**：ODE での確率質量保存則 $\partial_t p_t=-\mathrm{div}(p_t u_t)$。
- **Fokker-Planck 方程式**：SDE 版。ラプラシアン項 $\frac{\sigma_t^2}{2}\Delta p_t$ を加える。$\sigma_t=0$ で連続の方程式。
- **score function（スコア関数）** $\nabla\log p_t$：対数密度の勾配。SDE 拡張に必要。
- **Langevin 動力学**：静的分布 $p$ に対するスコアベースの SDE。MCMC の基礎。
- **Flow Matching / CFM** = (Conditional) Flow Matching。ベクトル場を二乗誤差回帰する学習。
- **(denoising) score matching**：スコア／ノイズを回帰する学習。DDPM の損失 $\|\epsilon_t^\theta-\epsilon\|^2$ に帰着。
- **CondOT path**：$\alpha_t=t,\beta_t=1-t$ の直線的ガウスパス（最適輸送に対応）。
- **CFG** = Classifier-Free Guidance（分類器なしガイダンス）。$\varnothing$ ラベルで条件/無条件を 1 モデルに同居させ条件を増幅。
- **ELBO** = Evidence Lower Bound（証拠下界）。離散時間拡散の損失近似。連続時間ではタイト。
- **DDPM** = Denoising Diffusion Probabilistic Models。ガウスパスの拡散モデル。
- **U-Net / DiT / MM-DiT**：拡散の代表アーキテクチャ（畳み込み U 字／Transformer／マルチモーダル Transformer）。
- **CLIP / T5**：テキスト条件付けに使う事前学習済み埋め込みモデル。

## 関連ページ

- [[concepts/flow-matching]]
- [[concepts/score-based-generative-models]]
- [[concepts/probability-flow-ode]]
- [[concepts/stochastic-interpolants]]
- [[concepts/denoising-diffusion]]
- [[concepts/classifier-free-guidance]]
- [[concepts/diffusion-model-architecture]]
- [[summaries/2023-flow-matching]] ・ [[summaries/2024-stochastic-interpolants]] ・ [[summaries/2024-sd3]]
- [[translations/2025-flow-matching-diffusion-intro]]
