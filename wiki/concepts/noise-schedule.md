---
type: concept
aliases: [Noise Schedule, ノイズスケジュール, noise schedule, sigma schedule, σ schedule, time discretization, noise distribution]
tags: [noise-schedule, diffusion-sampling, denoising-diffusion, score-based-generative-models, generative-models]
related:
  - "[[diffusion-sampling]]"
  - "[[denoising-diffusion]]"
  - "[[score-based-generative-models]]"
  - "[[probability-flow-ode]]"
  - "[[flow-matching]]"
summaries:
  - "[[summaries/2022-edm]]"
  - "[[summaries/2025-flux-kontext]]"
updated: 2026-08-17
---

# Noise Schedule（ノイズスケジュール）

**Noise Schedule（ノイズスケジュール）** とは、拡散モデルで**各時刻に加えるノイズ量（ノイズレベル $\sigma$ あるいは $\beta_t,\bar\alpha_t$）をどう決めるか**という設計選択の総称である。同じネットワーク・同じ目的関数でも、ノイズスケジュールを変えると学習の効率もサンプリングの品質・速度も大きく変わる。ノイズスケジュールは 2 つの側面を持つ：

1. **学習時のノイズ分布**：どのノイズレベル $\sigma$ を重点的に学習するか（$p_{\rm train}(\sigma)$）。中間ノイズレベルが最も「学びがい」があり、極端なレベル（ほぼ無ノイズ／ほぼ純ノイズ）は学習が無意味になりがち。
2. **推論時の時間離散化**：サンプリングの $N$ ステップを $\sigma$ 軸上にどう配置するか（$\{\sigma_i\}$ 列＝時刻 $\{t_i\}$）。打ち切り誤差を最小化する刻み方が品質と少ステップ化を左右する。

本 wiki のランドマークは **EDM（Karras ら 2022・[[summaries/2022-edm]]）** で、ノイズスケジュールを「理論的便宜」でなく**実用性能を決める第一級の設計軸**として体系的に分析した。

## なぜ重要か

拡散モデルは「データ→ノイズ」の順過程と「ノイズ→データ」の逆過程からなる（[[denoising-diffusion]]）。順過程のノイズの増やし方（スケジュール）は、(a) 学習データの分布と test 分布の整合、(b) 逆過程の軌道の曲率（＝サンプリングの難しさ）、(c) どのノイズレベルにモデル容量を割くか、を同時に決める。EDM が示したように、スケジュールの選択は**サンプラー・ネットワーク・学習目的とほぼ独立に**最適化できる成分であり、ここを正しく選ぶだけで少ステップ・高品質が得られる。

## ノイズスケジュールの系譜（既存ページの代表手法を横断）

本 wiki に取り込み済みの各手法のスケジュールを、この 1 ページから俯瞰する。

### 学習時のノイズスケジュール（順過程）

- **DDPM 線形 β スケジュール**（Ho ら 2020・[[denoising-diffusion]]）：離散時刻で $\beta_t$ を線形に増やす。$\bar\alpha_t=\prod(1-\beta_s)$。最も基本的だが端点付近が非効率。
- **cosine スケジュール**（IDDPM, Nichol & Dhariwal 2021・[[diffusion-model-architecture]] 内で言及）：$\bar\alpha_t=\cos^2(\cdots)$。線形より中間に情報を残し、少ステップで改善。
- **VE / VP**（Score-SDE, Song ら 2021・[[score-based-generative-models]]）：Variance Exploding（$\sigma$ を指数的に増やす、NCSN 系）と Variance Preserving（DDPM 連続版）。連続時間 SDE のドリフト・拡散係数として表現。
- **LDM-linear**（Stable Diffusion・[[latent-diffusion]]）：DDPM スケジュールの修正版を潜在空間で使用。
- **EDM の対数正規ノイズ分布**：学習時に $\ln\sigma\sim\mathcal{N}(P_{\rm mean},P_{\rm std}^2)$（$P_{\rm mean}=-1.2,P_{\rm std}=1.2$）。中間ノイズレベルに学習を集中させる「どの σ を学ぶか」の最適化（[[summaries/2022-edm]]）。
- **SD3 の logit-normal / mode / CosMap サンプラー**（[[flow-matching]] / [[summaries/2024-sd3]]）：rectified flow の時刻 $t$ の分布を、中間時刻に重みを置くよう選ぶ。EDM のノイズ分布思想を flow matching へ移したもの（rf/lognorm(0,1) が最良）。

### 推論時の時間離散化（逆過程のステップ配置）

- **EDM の ρ スケジュール**：$\sigma_{i}=(\sigma_{\rm max}^{1/\rho}+\frac{i}{N-1}(\sigma_{\rm min}^{1/\rho}-\sigma_{\rm max}^{1/\rho}))^{\rho}$、**ρ=7**。低ノイズ側のステップを短くして打ち切り誤差を抑える。**σ(t)=t** を採れば σ と時刻が交換可能になり軌道がほぼ直線化（[[diffusion-sampling]] の Heun サンプラーと組で少 NFE 化）。
- **VE の幾何級数 / VP の線形時刻 / iDDPM の漸化式**：表1（[[translations/2022-edm]]）で EDM が各手法の時刻離散化を統一表記。
- **SDXL の解像度依存 timestep shift**（[[summaries/2023-sdxl]]）：高解像度ほど信号破壊に多くのノイズが要るので、解像度に応じて時刻スケジュールを log-SNR シフトする。

## ランドマーク：EDM のノイズスケジュール（Karras ら 2022）

EDM は noise schedule を含む拡散の設計空間を分解し（[[summaries/2022-edm]]）：

- **σ(t)=t, s(t)=1**：スケジュールを最も単純化し、ODE 解軌道を直線に近づける。
- **ρ=7 の時刻離散化**：付録 D.1 の打ち切り誤差分析から、低ノイズ側を短くするのが品質に効くと示し ρ=7 を推奨。
- **対数正規の学習ノイズ分布**（$P_{\rm mean}=-1.2,P_{\rm std}=1.2$）：学習後のごと σ 損失を見ると中間レベルでのみ有意な改善が可能と分かるため、そこに学習を集中。
- **損失重み $\lambda(\sigma)=1/c_{\rm out}(\sigma)^2$**：全ノイズレベルで実効損失を均一化（preconditioning と一体、[[diffusion-model-architecture]]）。

これらにより、サンプラー（[[diffusion-sampling]]）と組み合わせて少 NFE・高品質を実現した。

## タイムステップの「シフト」と logit-normal の等価性

高解像度で学習するとき、SD3 系では時刻サンプリングを係数 $\alpha$ で**シフト**させる操作が使われる（解像度を $256^2$ から $1024^2$ へ上げる際は $\alpha=3.0$ が良いと報告された）。一方で本ページで見たとおり、SD3 は時刻を **logit-normal** からサンプルする。この 2 つは別々の道具に見えるが、**同じものである**ことを FLUX.1 Kontext（[[summaries/2025-flux-kontext]] Appendix A.2）が示した。

導出は短い。rectified flow の順過程の log-SNR（対数信号対雑音比）は $\lambda_t^{0,1}=-2\,\mathrm{logit}(t)$ と書け、任意の $\mu,\sigma$ では $\lambda_t^{\mu,\sigma}=\sigma\lambda_t^{0,1}-2\mu$ になる。他方 $\alpha$ シフトは $\lambda_t^{\alpha}=\lambda_t^{0,1}-2\log\alpha$ である。両者を見比べれば、$\sigma=1.0$ のとき

$$
\mu=\log\alpha
$$

と同定できる。つまり **$\alpha=3.0$ のシフトは、$\mu=\log 3.0=1.0986$・$\sigma=1.0$ の logit-normal 分布と同じ**である。さらに一般化したシフト後の時刻は

$$
t'=\frac{e^{\mu}}{e^{\mu}+(1/t-1)^{\sigma}}
$$

で表せ、$\sigma=1.0,\ \mu=\log\alpha$ とすれば SD3 の再配分関数 $t'=\frac{\alpha t}{1+(\alpha-1)t}$ を復元する。

実務的な含意は 2 つある。第一に、**「解像度に応じてシフト量を変える」という操作は「logit-normal のモードをずらす」ことに他ならない**——ノイズスケジュール設計の 2 つの語彙が 1 つに統合される。第二に、$\sigma$ も動かせる一般形が得られるため、学習時だけでなく**推論時の時刻の刻み方**にも同じ式を使える。実際 FLUX.1 Kontext は学習データの解像度に応じてモード $\mu$ を変えている。

## 既存知識との接続

- [[diffusion-sampling]]：推論時の時間離散化（$\{\sigma_i\}$）はサンプラーの一部。EDM の ρ スケジュールは Heun サンプラーと一体で少ステップ化を実現する。
- [[denoising-diffusion]]：DDPM の β スケジュールが最も基本的なノイズスケジュール。学習目的の重み付け（λ(σ)）もここに関わる。
- [[score-based-generative-models]]：VE/VP は連続時間 SDE としてのノイズスケジュール。Score-SDE がこれらを統一し、EDM が σ(t)/s(t) として整理した。
- [[probability-flow-ode]]：σ(t)=t の選択は確率フロー ODE の軌道を直線化し、少 NFE の決定論サンプリングを可能にする。
- [[flow-matching]]：SD3 の logit-normal/mode サンプラーは「時刻分布をどう選ぶか」という同じ問題で、EDM のノイズ分布思想を rectified flow へ移したもの。

## 参考文献（summaries）

- [[summaries/2025-flux-kontext]] — FLUX.1 Kontext（タイムステップの α シフトが logit-normal の μ=log α と等価であることを導出。Appendix A.2）

- [[summaries/2022-edm]] — EDM（σ(t)=t・ρ=7 時刻離散化・対数正規ノイズ分布・損失重み、ノイズスケジュールのランドマーク）
- [[summaries/2021-score-sde]] — Score-SDE（VE/VP の連続時間スケジュール）
- [[summaries/2020-ddpm]] — DDPM（線形 β スケジュール）
- [[summaries/2024-sd3]] — SD3（rectified flow の logit-normal/mode 時刻分布）
- [[summaries/2023-sdxl]] — SDXL（解像度依存 timestep shift）
