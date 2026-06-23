---
type: concept
aliases: [Score-Based Generative Models, スコアベース生成モデル, score matching, NCSN]
tags: [score-based-generative-models, denoising-diffusion, generative-models]
related:
  - "[[denoising-diffusion]]"
summaries:
  - "[[summaries/2020-ddpm]]"
updated: 2026-06-23
---

# Score-Based Generative Models（スコアベース生成モデル）

**Score-Based Generative Models（スコアベース生成モデル）** とは、データ分布そのものではなく、その**スコア（score）**＝対数確率密度の勾配 $\nabla_{\mathbf{x}}\log p(\mathbf{x})$ を学習し、それを使ってサンプルを生成する生成モデルの一群である。スコアは「データ密度が高くなる方向」を各点で指すベクトル場であり、これが分かればノイズから出発して密度の高い領域へ「坂を登る」ように進めば本物らしいサンプルが得られる。

このページは、[[denoising-diffusion]]（ノイズ除去拡散モデル）と理論的に等価な双子としてのスコアベース生成モデルを俯瞰する軽めの概念ページである。両者の橋渡しを明確にしたのが DDPM 論文（[[summaries/2020-ddpm]]）の主要貢献の一つである。

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

の **2 つは同じ生成原理の異なる定式化**である。この統一的な見方は、後に連続時間の **SDE（Stochastic Differential Equation, 確率微分方程式）／ODE（常微分方程式）** による定式化（Song ら 2021 のスコア SDE）へと一般化され、拡散モデルの理論的基盤となった。各拡散 SDE には同じ周辺分布をたどる決定論的な **確率フロー ODE**（[[probability-flow-ode]]）が対応し、[[diffusion-sampling]] の DDIM はその VE-SDE 版の特殊例＝Euler 離散化として理解できる。

## 既存知識との接続

- [[denoising-diffusion]]：DDPM の ε 予測目的関数＝denoising score matching、逆過程サンプリング＝Langevin 動力学という対応で本概念と結ばれる。
- DDPM 論文は NCSN との差分（U-Net＋自己注意、$\sqrt{1-\beta_t}$ によるスケーリング、信号を完全に破壊する順過程、$\beta_t$ から厳密に導いたサンプラー係数）も整理しており、サンプル品質改善の要因を明らかにしている（[[summaries/2020-ddpm]] 付録 C）。

## 参考文献（summaries）

- [[summaries/2020-ddpm]] — Denoising Diffusion Probabilistic Models（拡散モデルとスコアマッチングの等価性を示した）
