---
type: concept
aliases: [Probability Flow ODE, 確率フロー ODE, probability flow, SDE, ODE]
tags: [probability-flow-ode, diffusion-sampling, score-based-generative-models, generative-models]
related:
  - "[[diffusion-sampling]]"
  - "[[score-based-generative-models]]"
  - "[[denoising-diffusion]]"
  - "[[flow-matching]]"
summaries:
  - "[[summaries/2021-score-sde]]"
  - "[[summaries/2021-ddim]]"
  - "[[summaries/2023-flow-matching]]"
updated: 2026-06-23
---

# Probability Flow ODE（確率フロー ODE）

**Probability Flow ODE（確率フロー ODE）** とは、拡散過程を**連続時間**で記述したときに現れる、生成と同じ確率の流れを持つ**決定論的な常微分方程式（ODE）**である。拡散モデルは離散ステップのマルコフ連鎖としても、連続時間の確率微分方程式（SDE, Stochastic Differential Equation）としても定式化できる。Song ら（Score-SDE, 2021）は、ノイズを加える拡散 SDE には、各時刻で**同じ周辺分布を持つ決定論的 ODE**（確率フロー ODE）が対応することを示した。この ODE を逆向きに解けば、ノイズから決定論的にデータを生成できる。

このページは確率フロー ODE / 連続時間定式化（SDE・ODE）を扱う。主要原典は Song らの **Score-SDE 論文**（[[summaries/2021-score-sde]]）で、そこで確率フロー ODE が初めて一般的に導出された。[[score-based-generative-models]]（SDE 統一枠組み）と表裏一体の概念である。

## SDE / ODE 定式化の考え方

- **拡散 SDE**：データに連続時間でノイズを加える過程を確率微分方程式で書く。ノイズの加え方（分散の増え方）で **VE-SDE（Variance Exploding, 分散爆発型。SMLD/NCSN）** や **VP-SDE（Variance Preserving, 分散保存型。DDPM）** などに分類される（[[score-based-generative-models]]）。
- **逆 SDE**：生成は対応する逆向きの SDE を解くこと。これには各時刻のデータ対数密度の勾配＝**スコア $\nabla_\mathbf{x}\log p_t(\mathbf{x})$** が必要で、ノイズ予測ネットワーク $\epsilon_\theta$ と $\nabla_\mathbf{x}\log p_t \approx -\epsilon_\theta/\sigma_t$ の関係で結ばれる。
- **確率フロー ODE**：逆 SDE と同じ周辺分布 $\{p_t(\mathbf{x})\}$ をたどる決定論的 ODE。Score-SDE が示した一般形は

$$
\mathrm{d}\mathbf{x}=\Big[\mathbf{f}(\mathbf{x},t)-\tfrac{1}{2}g(t)^2\nabla_\mathbf{x}\log p_t(\mathbf{x})\Big]\mathrm{d}t
$$

（逆時間 SDE のスコア項を $\tfrac12$ にし、拡散項を落とした形）。これは Fokker-Planck 方程式から、SDE と同じ $p_t$ を持つ決定論過程として導かれる。ランダム項を含まないため、(1) 決定論的に生成でき、(2) データ ↔ 潜在の**可逆な対応**（encode / 再構成）が得られ、(3) ニューラル ODE であることから**厳密な対数尤度計算**（瞬間変数変換＋Skilling-Hutchinson トレース推定）が可能になる。

## 主要原典：Score-SDE（Song ら 2021）が示したこと

[[summaries/2021-score-sde]] は確率フロー ODE で以下を実現した：

- **厳密尤度**：CIFAR-10 で 2.99 bits/dim（当時最高、最尤学習なしでも）。
- **一意に識別可能な encode**：順方向 SDE が学習可能パラメータを持たないため、完全に推定されたスコアの下で同じ入力は同じ潜在コードに写る（異なるアーキテクチャの 2 モデルでもほぼ一致、付録 D.5）。
- **潜在操作**：補間・温度スケーリング（正規化フロー的）。
- **高速サンプリング**：black-box ODE ソルバー（RK45）で適応的に積分でき、許容誤差を緩めると関数評価回数を 90% 以上削減可。ただし corrector なしの ODE サンプラーは SDE サンプラーより FID が悪くなりやすく、特に VE-SDE で顕著。

## DDIM との関係

DDIM（[[diffusion-sampling]]）の決定論サンプリング更新は、変数変換 $\bar{\mathbf{x}}=\mathbf{x}/\sqrt{\alpha}$、$\sigma=\sqrt{(1-\alpha)/\alpha}$ を施すと、次の ODE の**オイラー法（Euler 法）による離散化**になる（DDIM 論文 式14）。

$$
\frac{\mathrm{d}\bar{\mathbf{x}}(t)}{\mathrm{d}t}=\frac{\mathrm{d}\sigma(t)}{\mathrm{d}t}\,\epsilon_\theta^{(t)}\!\left(\frac{\bar{\mathbf{x}}(t)}{\sqrt{\sigma^2(t)+1}}\right)
$$

DDIM 論文は **命題 1** で、この ODE が Song らの **VE-SDE の確率フロー ODE の特殊例と等価**であることを証明した（ただし離散化の取り方＝$\mathrm{d}\sigma$ で刻むか $\mathrm{d}t$ で刻むか、は両者で異なり、少ステップでは差が出る）。この「サンプリング＝ODE の数値積分」という視点が、DDIM の以下の性質を説明する。

- **決定論性**：ランダム項がないので $\mathbf{x}_T$ から $\mathbf{x}_0$ が一意に決まる。
- **encode / 再構成**：ODE を順逆に解いて $\mathbf{x}_0 \leftrightarrow \mathbf{x}_T$ を行き来でき、低い誤差で再構成できる（neural ODE / 正規化フロー的性質）。
- **少ステップ化**：ODE ソルバーの離散化ステップを粗くすれば少ステップ生成になる。高次の ODE ソルバーを使えばさらに少ステップで高精度化できる（→ [[diffusion-sampling]] の発展）。

## 既存知識との接続

- [[score-based-generative-models]]：確率フロー ODE はスコア $\nabla_\mathbf{x}\log p_t$ で駆動される。SDE 統一枠組みの決定論的な対であり、Score-SDE が両者を同時に提示した。
- [[diffusion-sampling]]：DDIM の決定論サンプリングは確率フロー ODE のオイラー離散化。ODE 視点は高速・高精度サンプラー（高次 ODE ソルバー等）設計の基盤。
- [[denoising-diffusion]]：DDPM は VP-SDE の離散版に対応し、DDIM の ODE は VE-SDE の確率フロー ODE と等価。
- [[flow-matching]]：確率フロー ODE もまた CNF（連続正規化フロー）の一種。フローマッチングは「拡散の確率フロー ODE の VF を、確率パスを直接指定して回帰する」一般化にあたり、拡散の条件付き VF が確率フロー ODE の VF と一致する（[[summaries/2023-flow-matching]] 付録 D）。

## 参考文献（summaries）

- [[summaries/2021-score-sde]] — Score-Based Generative Modeling through SDEs（確率フロー ODE の一般導出・厳密尤度・一意 encode、主要原典）
- [[summaries/2021-ddim]] — Denoising Diffusion Implicit Models（DDIM の ODE 形と VE-SDE 確率フロー ODE との等価性を証明）
- [[summaries/2023-flow-matching]] — Flow Matching（CNF を確率パス回帰で学習、拡散 VF が確率フロー ODE の VF と一致）
