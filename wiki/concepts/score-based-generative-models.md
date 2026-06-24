---
type: concept
aliases: [Score-Based Generative Models, スコアベース生成モデル, score matching, NCSN]
tags: [score-based-generative-models, denoising-diffusion, generative-models]
related:
  - "[[denoising-diffusion]]"
  - "[[probability-flow-ode]]"
  - "[[diffusion-sampling]]"
  - "[[flow-matching]]"
  - "[[stochastic-interpolants]]"
  - "[[noise-schedule]]"
summaries:
  - "[[summaries/2020-ddpm]]"
  - "[[summaries/2021-score-sde]]"
  - "[[summaries/2023-flow-matching]]"
  - "[[summaries/2024-stochastic-interpolants]]"
  - "[[summaries/2022-edm]]"
  - "[[summaries/2025-flow-matching-diffusion-intro]]"
updated: 2026-06-25
---

# Score-Based Generative Models（スコアベース生成モデル）

**Score-Based Generative Models（スコアベース生成モデル）** とは、データ分布そのものではなく、その**スコア（score）**＝対数確率密度の勾配 $\nabla_{\mathbf{x}}\log p(\mathbf{x})$ を学習し、それを使ってサンプルを生成する生成モデルの一群である。スコアは「データ密度が高くなる方向」を各点で指すベクトル場であり、これが分かればノイズから出発して密度の高い領域へ「坂を登る」ように進めば本物らしいサンプルが得られる。

このページは、[[denoising-diffusion]]（ノイズ除去拡散モデル）と理論的に等価な双子としてのスコアベース生成モデルを俯瞰し、さらに両者を連続時間の確率微分方程式（SDE）として統一した **Score-SDE 枠組み**（Song ら 2021, [[summaries/2021-score-sde]]）までを扱う。両者の橋渡しは DDPM 論文（[[summaries/2020-ddpm]]）でも示され、Score-SDE がそれを連続時間へ一般化して完成させた。

## 2 つの構成要素

1. **スコアの学習：denoising score matching（ノイズ除去スコアマッチング）**
   スコア $\nabla_{\mathbf{x}}\log p(\mathbf{x})$ を直接推定するのは難しいが、「データに既知のガウスノイズを乗せ、そのノイズ（に比例する量）を予測する」タスクに置き換えると、推定量がスコアに一致することが知られている。つまり**ノイズ除去がそのままスコア推定になる**。データ 1 点だけではスコアが定まらない低密度領域をカバーするため、Song & Ermon の **NCSN（Noise Conditional Score Network）** は**複数のノイズスケール**でスコアを学習する。

2. **サンプリング：Langevin dynamics（ランジュバン動力学）**
   学習したスコア $\mathbf{s}_\theta(\mathbf{x})\approx\nabla_{\mathbf{x}}\log p(\mathbf{x})$ を使い、
   $$
   \mathbf{x}\leftarrow\mathbf{x}+\tfrac{\eta}{2}\,\mathbf{s}_\theta(\mathbf{x})+\sqrt{\eta}\,\mathbf{z},\quad \mathbf{z}\sim\mathcal{N}(0,\mathbf{I})
   $$
   という「スコア方向への勾配上昇＋ノイズ注入」を繰り返す。大きなノイズスケールから小さなスケールへ徐々に下げていく **annealed Langevin dynamics（アニーリング・ランジュバン動力学）** により、多様かつ高品質なサンプルが得られる。

## 拡散モデルとの等価性

DDPM の **ε 予測**（$\mathbf{x}_t$ に乗ったノイズを予測する）パラメータ化は、上記の denoising score matching と数学的に一致する。実際、予測ノイズ $\boldsymbol\epsilon_\theta$ とスコアは

$$
\mathbf{s}_\theta(\mathbf{x}_t,t)=\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t)\approx-\frac{\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{1-\bar\alpha_t}}
$$

の関係で結ばれる（ノイズ予測 ⇔ スコア推定）。さらに DDPM のサンプリング手続き（逆過程）は、学習スコアを使った Langevin 的な確率的サンプリングに対応する。したがって：

- **拡散モデル（[[denoising-diffusion]]）** … 変分推論で逆過程（マルコフ連鎖）を学習する離散時間の視点
- **スコアベース生成モデル** … スコアを学習し Langevin 動力学でサンプリングする視点

の **2 つは同じ生成原理の異なる定式化**である。

## 連続時間への一般化：Score-SDE 枠組み（VE / VP / sub-VP）

Song ら（Score-SDE, ICLR 2021, [[summaries/2021-score-sde]]）は、ノイズスケールを有限個から**無限個（連続時間）** へ一般化し、スコアベースモデルと拡散モデルを単一の **確率微分方程式（SDE, Stochastic Differential Equation）** の枠組みに統一した。これが現在のスコアベース生成モデルの標準的な定式化である。

- **順方向 SDE**：データにノイズを連続的に加える $\mathrm{d}\mathbf{x}=\mathbf{f}(\mathbf{x},t)\mathrm{d}t+g(t)\mathrm{d}\mathbf{w}$（$\mathbf{f}$=ドリフト、$g$=拡散係数、$\mathbf{w}$=ブラウン運動）。
- **逆時間 SDE（Anderson）**：生成は $\mathrm{d}\mathbf{x}=[\mathbf{f}(\mathbf{x},t)-g(t)^2\nabla_\mathbf{x}\log p_t(\mathbf{x})]\mathrm{d}t+g(t)\mathrm{d}\bar{\mathbf{w}}$。**周辺分布のスコア $\nabla_\mathbf{x}\log p_t(\mathbf{x})$ だけで定まる**点が核心。スコアは時間依存ネット $\mathbf{s}_\theta(\mathbf{x},t)$ を連続スコアマッチングで学習して推定する。

ノイズの加え方によって既存手法が SDE として分類される：

- **VE-SDE（Variance Exploding, 分散爆発）= SMLD / NCSN**：分散が時間とともに発散する。
- **VP-SDE（Variance Preserving, 分散保存）= DDPM**：分散が有界（初期が単位分散なら一定）。
- **sub-VP-SDE**（Score-SDE の新提案）：分散が常に対応する VP より小さく、尤度に強い。

この統一により、(1) [[diffusion-sampling]] の predictor-corrector サンプラー・逆拡散サンプラー、(2) SDE と同じ周辺分布をたどる決定論的な **確率フロー ODE**（[[probability-flow-ode]]、厳密尤度・可逆 encode を可能に）、(3) 条件付き逆時間 SDE による [[controllable-generation]]（可制御生成）が導かれた。[[diffusion-sampling]] の DDIM は、確率フロー ODE（VE-SDE 版）の Euler 離散化として理解できる。

## 既存知識との接続

- [[denoising-diffusion]]：DDPM の ε 予測目的関数＝denoising score matching、逆過程サンプリング＝Langevin 動力学という対応で本概念と結ばれる。
- DDPM 論文は NCSN との差分（U-Net＋自己注意、$\sqrt{1-\beta_t}$ によるスケーリング、信号を完全に破壊する順過程、$\beta_t$ から厳密に導いたサンプラー係数）も整理しており、サンプル品質改善の要因を明らかにしている（[[summaries/2020-ddpm]] 付録 C）。
- [[flow-matching]]：denoising score matching（スコア回帰）はフローマッチングのベクトル場回帰の特別な場合とみなせる。拡散の VE/VP パスは FM のガウス条件付きパスとして内包され、FM はスコアマッチングのより安定な代替を与える。
- [[stochastic-interpolants]]：確率的補間（[[summaries/2024-stochastic-interpolants]]）はスコアベース拡散を「片側補間（$\rho_0$=ガウス）」として時間再パラメータ化で再構成し、無限時間・ガウス必須・有限区間圧縮時の特異性という SBDM の制約を、有限時間・任意基底で解消する。スコア $s=-\gamma^{-1}\mathbb{E}(z|x_t)$ の新しい二乗回帰目的も与え、拡散（SDE）とフロー（ODE）を同じ密度の異なる実現として統一する。
- [[noise-schedule]]：EDM（[[summaries/2022-edm]]）は Score-SDE の VP/VE を σ(t)・s(t)・preconditioning という独立成分に分解して整理し、denoiser $D(\boldsymbol{x};\sigma)$ とスコア $\nabla\log p=(D-\boldsymbol{x})/\sigma^2$ の関係を中心に据えた。これにより VP/VE/iDDPM を 1 つの表で比較でき、ノイズスケジュールを第一級の設計軸として最適化できるようにした。
- [[classifier-guidance]]：「ノイズ予測 $\epsilon_\theta$ はスコアのリスケール $-\sqrt{1-\bar\alpha_t}\,\nabla_x\log p(x_t)$」というスコア視点が、ADM（[[summaries/2021-adm]]）の決定論的（DDIM）版分類器ガイダンスの導出根拠になっている。

## 参考文献（summaries）

- [[summaries/2020-ddpm]] — Denoising Diffusion Probabilistic Models（拡散モデルとスコアマッチングの等価性を示した）
- [[summaries/2021-score-sde]] — Score-Based Generative Modeling through SDEs（SMLD=VE / DDPM=VP として連続時間 SDE に統一）
- [[summaries/2023-flow-matching]] — Flow Matching（スコア回帰をベクトル場回帰へ一般化、拡散パスを内包）
- [[summaries/2024-stochastic-interpolants]] — Stochastic Interpolants（SBDM を片側補間として内包し flows と diffusions を統一）
- [[summaries/2022-edm]] — EDM（VP/VE を σ(t)/s(t)/preconditioning で統一、denoiser↔score を中心に設計空間を整理）
- [[summaries/2025-flow-matching-diffusion-intro]] — Flow Matching と拡散モデル入門（MIT 6.S184 講義ノート。denoising score matching・score↔ベクトル場の変換・確率フロー ODE を教科書的に導く）
