---
type: concept
aliases: [Probability Flow ODE, 確率フロー ODE, probability flow, SDE, ODE]
tags: [probability-flow-ode, diffusion-sampling, score-based-generative-models, generative-models]
related:
  - "[[diffusion-sampling]]"
  - "[[score-based-generative-models]]"
  - "[[denoising-diffusion]]"
summaries:
  - "[[summaries/2021-ddim]]"
updated: 2026-06-23
---

# Probability Flow ODE（確率フロー ODE）

**Probability Flow ODE（確率フロー ODE）** とは、拡散過程を**連続時間**で記述したときに現れる、生成と同じ確率の流れを持つ**決定論的な常微分方程式（ODE）**である。拡散モデルは離散ステップのマルコフ連鎖としても、連続時間の確率微分方程式（SDE, Stochastic Differential Equation）としても定式化できる。Song ら（Score-SDE, 2021）は、ノイズを加える拡散 SDE には、各時刻で**同じ周辺分布を持つ決定論的 ODE**（確率フロー ODE）が対応することを示した。この ODE を逆向きに解けば、ノイズから決定論的にデータを生成できる。

このページは確率フロー ODE / 連続時間定式化（SDE・ODE）の俯瞰と、DDIM との関係を中心に扱う軽めの概念ページである。主要原典である Song らの Score-SDE 論文はまだ本 wiki に未取り込みのため、ここでは [[summaries/2021-ddim]]（DDIM）で示された関係を起点に記述する。

## SDE / ODE 定式化の考え方

- **拡散 SDE**：データに連続時間でノイズを加える過程を確率微分方程式で書く。ノイズの加え方（分散の増え方）で **VE-SDE（Variance Exploding, 分散爆発型）** や VP-SDE（Variance Preserving, 分散保存型。DDPM に対応）などに分類される。
- **逆 SDE**：生成は対応する逆向きの SDE を解くこと。これには各時刻のデータ対数密度の勾配＝**スコア $\nabla_\mathbf{x}\log p_t(\mathbf{x})$**（[[score-based-generative-models]]）が必要で、これはノイズ予測ネットワーク $\epsilon_\theta$ と $\nabla_\mathbf{x}\log p_t \approx -\epsilon_\theta/\sigma_t$ の関係で結ばれる。
- **確率フロー ODE**：逆 SDE と同じ周辺分布をたどる決定論的 ODE。ランダム項を含まないため、(1) 決定論的に生成でき、(2) データ ↔ 潜在の**可逆な対応**（encode / 再構成、尤度計算）が得られる。

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

- [[diffusion-sampling]]：DDIM の決定論サンプリングは確率フロー ODE のオイラー離散化。ODE 視点は高速・高精度サンプラー設計の基盤。
- [[score-based-generative-models]]：確率フロー ODE はスコア $\nabla_\mathbf{x}\log p_t$ で駆動される。スコアベース生成モデルの連続時間一般化が SDE/ODE 定式化。
- [[denoising-diffusion]]：DDPM は VP-SDE の離散版に対応し、DDIM の ODE は VE-SDE の確率フロー ODE と等価。

## 参考文献（summaries）

- [[summaries/2021-ddim]] — Denoising Diffusion Implicit Models（DDIM の ODE 形と VE-SDE 確率フロー ODE との等価性を証明）

> 注: 確率フロー ODE / Score-SDE の主要原典（Song et al., "Score-Based Generative Modeling through SDEs", 2021）は未取り込み。取り込み時に本ページを大幅拡充する。
