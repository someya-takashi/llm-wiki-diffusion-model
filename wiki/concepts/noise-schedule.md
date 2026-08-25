---
type: concept
aliases: [Noise Schedule, ノイズスケジュール, noise schedule, sigma schedule, σ schedule, time discretization, noise distribution]
tags: [noise-schedule, diffusion-sampling, denoising-diffusion, score-based-generative-models, generative-models, flux2, ai-toolkit]
related:
  - "[[diffusion-sampling]]"
  - "[[denoising-diffusion]]"
  - "[[score-based-generative-models]]"
  - "[[probability-flow-ode]]"
  - "[[flow-matching]]"
  - "[[ai-toolkit]]"
summaries:
  - "[[summaries/2022-edm]]"
  - "[[summaries/2025-flux-kontext]]"
  - "[[summaries/2026-hidream-o1-image]]"
  - "[[summaries/2025-wan]]"
  - "[[summaries/2025-z-image]]"
  - "[[summaries/2024-sana]]"
  - "[[summaries/2025-flux2]]"
  - "[[summaries/2026-ai-toolkit]]"
updated: 2026-08-26
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

## 学習目標のパラメータ化とスケジュールは切り離せない

もう 1 つ、本ページの守備範囲に隣接する論点がある——**同じスケジュールでも、モデルに何を予測させるか（$\epsilon$／$x_0$／$v$）で数値的な性質が変わる**。

**Sana**（[[summaries/2024-sana]]）の付録は、Tweedie の公式を使ってこれを明示する。$t \approx T$（ほぼ純ノイズ）では $x_0$ と $x_t$ がほぼ独立になるため、**ノイズ予測モデルの最適解は $x_t$ の線形関数に退化し、データ予測モデルの最適解はほぼ定数 $\mathbb{E}[x_0]$ に近づく**。積分の離散化誤差は前者の方が大きく、しかも $t=T$ の誤差は以降の全ステップに伝播する。したがって**スケジュールの端点付近をどう扱うかは、予測対象の選択と一体で決まる**。Sana はこの分析から Flow-DPM-Solver を導いている（[[diffusion-sampling]]）。

同じ論文は、**Flow Matching と DDPM を 120K ステップという同一条件で直接比較**した数少ない例でもある（FID 19.5 → 16.9、CLIP 24.6 → 25.7）。$\epsilon$ 予測から $v$／$x_0$ 予測へ移ることが収束の速さそのものを変える、という本節の主張と整合する結果である（[[flow-matching]]）。

## シフト量をステップ数にも依存させる（FLUX.2）

上で導出した「$\alpha$ シフト ≡ $\mu=\log\alpha$ の logit-normal」という等価性は、**FLUX.2 の実装にそのまま $\mu$ という変数名で現れる**（[[summaries/2025-flux2]]）。

```python
def generalized_time_snr_shift(t, mu, sigma):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
```

$\sigma=1$ ならこれは標準のシフト $t'=st/(1+(s-1)t)$ と代数的に等価で、**シフト係数は $s=e^{\mu}$** である。理論的な同定が実装の設計変数として定着していることの確認になる。

**新しいのは $\mu$ の決め方である。** FLUX.1 の `base_shift` / `max_shift` の線形補間は廃止され、$\mu$ は **`(image_seq_len, num_steps)` の 2 次元の経験フィット**になった。系列長について 2 本の直線（10 ステップ用と 200 ステップ用）を引き、それを**ステップ数で線形補間する**。

本ページはこれまでシフトを**解像度への対処**として扱ってきた（多解像度で SNR が変わるため）。FLUX.2 はそこに**ステップ数という第 2 の軸**を加える。1360×768（4080 トークン）で計算すると：

| ステップ数 | シフト係数 $s=e^\mu$ |
| --- | --- |
| 50 | ≈ 7.6 |
| **4** | **≈ 9.9** |

**少ステップほどシフトを大きくする**——つまり高ノイズ域に時間予算を厚く配る。少ステップ生成では大域的な構図を決める初期段階に十分な刻みを割かないと破綻するので、**スケジュール側から蒸留を補完している**と読める（[[concepts/diffusion-distillation]]）。蒸留が「モデルを作り替える」のに対し、こちらは「同じモデルでも刻み方を変える」ぶんだけ直交した対処である。

## 事前学習と微調整でスケジュールを変える（2026）

もう 1 つ、[[summaries/2026-hidream-o1-image]] が持ち込んだ実務的な知見がある。SFT（Supervised Fine-Tuning, 高品質データでの教師あり微調整）の段階で、**事前学習で使っていた logit-normal サンプリングを一様サンプリングに切り替える**。

理由は本ページの中心的な論点から素直に導ける。logit-normal は中間タイムステップに学習を集中させる——ノイズと画像が拮抗する「一番情報量のある」領域に予算を割く戦略である。これは**ゼロから生成の骨格を学ぶ事前学習**には理にかなう。しかし SFT の目的は美的品質・写実性・細部の詰めであり、**それらが決まるのは $t$ が小さい後期のノイズ除去ステップ**（すでに大まかな構図はできていて、質感や細かい形状を整える段階）である。一様サンプリングに戻せば、後期ステップにも均等に学習の重みが回る。

「スケジュールは 1 つ選んで固定するもの」という暗黙の前提を崩し、**学習フェーズごとに使い分ける**という発想であり、EDM が「学習時のノイズ分布」と「推論時の時間離散化」を分離したのと同じ種類の細分化と見ることができる。

## 実務では「モデルの事前学習と違う分布」で学習される

本ページは学習時のノイズ分布を設計変数として追ってきた——SD3 の logit-normal、HiDream-O1 が SFT で一様へ切り替えたこと、FLUX の解像度依存シフト、Sana の Flow-DPM-Solver。実際の学習ツールではこれが **`timestep_type` という 1 つの設定**に集約されている（[[concepts/ai-toolkit]]・[[summaries/2026-ai-toolkit]]）。

| 設定値 | 本ページの記述との対応 |
| --- | --- |
| `sigmoid`（全体の既定） | 中央に厚い。SD3 の logit-normal と同じ動機 |
| `linear` | 一様。HiDream-O1 が SFT で採った方針 |
| `weighted` | **スケジュールは linear、損失側で重み付け** |
| `shift` / `flux_shift` | **推論の動的シフトに合わせる** |
| `one_step` / `four_step` / `eight_step` | 少ステップ蒸留モデル向けに時刻を固定 |

理論側の選択肢がほぼそのまま設定値になっている点は素直だが、**2 つ引っかかる事実がある**。

**第一に、FLUX.2 の既定が `weighted`（＝時刻は一様）であり、推論側の解像度依存シフトを学習では使わない。** 本ページは「$\alpha$ シフト ≡ $\mu=\log\alpha$ の logit-normal」という同定を記録し、[[summaries/2025-flux2]] ではそれがコードに $\mu$ という変数名で現れることまで確認した。ところが**その機構は学習では働いていない**。学習時の時刻分布はモデルの事前学習・推論の分布と一致していない。

**第二に、`weighted` が掛ける損失重みは 1003 行のハードコード表で、出自が別モデルである。** 実装のコメントが明かす。

> these weights were calculated using **flex.1-alpha**. A similar weighing scheme has been seen with other flowmatch models as well.

**flow matching モデルなら似た重み付けでよい、という経験則で使い回されている。** 本ページが積み上げてきた「どの時刻に予算を配るかが品質を決める」という主張に対し、実務側はモデル横断の経験則に落ち着いている。どちらが妥当かを判定する材料は本 wiki にはないが、**論文の設定をそのまま再現しようとしても、ツールの既定は別のことをしている**という点は把握しておく価値がある。

## 計算して捨てられる $\mu$ — Z-Image（2026-08-26 追記）

前節では FLUX.2 について「**推論は解像度依存シフト、学習は一様＋別モデル由来の重み表**」という不一致を記録した。Z-Image のリファレンス実装（[[summaries/2025-z-image]]）は、**同じ不一致を逆向きに持っている**。

パイプラインは SD3・FLUX と**同じ定数**で $\mu$ を計算する。

```python
# src/zimage/pipeline.py L23–33, L194–200
BASE_IMAGE_SEQ_LEN, MAX_IMAGE_SEQ_LEN = 256, 4096
BASE_SHIFT, MAX_SHIFT = 0.5, 1.15
mu = calculate_shift(image_seq_len, 256, 4096, 0.5, 1.15)
scheduler_kwargs = {"mu": mu}
```

ところが受け取る側は

```python
# src/zimage/scheduler.py L85–88
if self.use_dynamic_shifting:
    sigmas = self.time_shift(mu, 1.0, sigmas)   # ← ここに入らない
else:
    sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
```

で、**既定は `use_dynamic_shifting = False`・`shift = 3.0`**（`src/config/model.py` L39–40）。**$\mu$ は毎回計算されて、毎回捨てられている。** diffusers のスケジューラをほぼそのまま流用した結果、呼び出し側だけが SD3/FLUX の作法を引きずっている——実装の由来が読み取れる残骸である。

**代わりに使われるのは静的な $\text{shift}=3.0$** である。本ページの記法に直すと、$\sigma \mapsto \dfrac{s\sigma}{1+(s-1)\sigma}$ に $s=3$ を入れた形で、$\mu = \log 3 \approx 1.10$ の logit-normal に相当する。参考までに、捨てられているほうの $\mu$ は 1024×1024（4096 トークン）で $1.15$、256 トークンで $0.5$ ——**1024px 付近では偶然ほぼ同じ値**になり、差が出るのは低解像度側である。

**注目すべきは、こちらでは学習と推論が揃っていること。** [[concepts/ai-toolkit]]（[[summaries/2026-ai-toolkit]]）の Z-Image 実装も学習用スケジューラに**同じ `use_dynamic_shifting: False, shift: 3.0`** を使う（`extensions_built_in/diffusion_models/z_image/z_image.py` L42–45）。

| | 学習時のシフト | 推論時のシフト | 一致 |
| --- | --- | --- | --- |
| FLUX.2 | 一様（`weighted`）＋ flex.1-alpha 由来の損失重み | 解像度・ステップ数依存の動的シフト | ❌ |
| **Z-Image** | **静的 $s=3.0$** | **静的 $s=3.0$** | ✅ |

**2 つの独立した実装が同じ値を使っている**ので、これは片方の設定漏れではない。本ページが積み上げてきた「解像度によって SNR が変わるからシフトが要る」という議論に対し、**推奨解像度域を 512–2048px に限れば静的シフトで足りる**という実務上の答えが提示されていることになる。ただし**それを支持する測定は公開されていない**——Z-Image の報告書は「FLUX 由来の動的な時間シフト」を使うと述べており、**報告書と実装が食い違っている**（[[questions/z-image-figure10-architecture-walkthrough]] の §4 に同じ食い違いを記録した）。報告書の記述が学習の話で実装が推論の話だとしても、上記のとおり学習側も静的である。

## 既存知識との接続

- [[diffusion-sampling]]：推論時の時間離散化（$\{\sigma_i\}$）はサンプラーの一部。EDM の ρ スケジュールは Heun サンプラーと一体で少ステップ化を実現する。
- [[denoising-diffusion]]：DDPM の β スケジュールが最も基本的なノイズスケジュール。学習目的の重み付け（λ(σ)）もここに関わる。
- [[score-based-generative-models]]：VE/VP は連続時間 SDE としてのノイズスケジュール。Score-SDE がこれらを統一し、EDM が σ(t)/s(t) として整理した。
- [[probability-flow-ode]]：σ(t)=t の選択は確率フロー ODE の軌道を直線化し、少 NFE の決定論サンプリングを可能にする。
- [[flow-matching]]：SD3 の logit-normal/mode サンプラーは「時刻分布をどう選ぶか」という同じ問題で、EDM のノイズ分布思想を rectified flow へ移したもの。
- [[diffusion-sampling]]：Sana の Flow-DPM-Solver はタイムステップのシフト係数 $s$ を $\tilde{\sigma}_t = s\sigma_t/(1+(s-1)\sigma_t)$ としてソルバー内部に組み込む。本ページのシフトと同じ式が推論側でも使われる例。

## 参考文献（summaries）

- [[summaries/2025-wan]] — Wan（動画でも SD3 由来の logit-normal タイムステップサンプリングをそのまま使う）

- [[summaries/2026-hidream-o1-image]] — HiDream-O1-Image（事前学習の logit-normal を SFT で一様サンプリングへ切り替え、後期ステップに学習の重みを回す）

- [[summaries/2025-flux-kontext]] — FLUX.1 Kontext（タイムステップの α シフトが logit-normal の μ=log α と等価であることを導出。Appendix A.2）

- [[summaries/2022-edm]] — EDM（σ(t)=t・ρ=7 時刻離散化・対数正規ノイズ分布・損失重み、ノイズスケジュールのランドマーク）
- [[summaries/2021-score-sde]] — Score-SDE（VE/VP の連続時間スケジュール）
- [[summaries/2020-ddpm]] — DDPM（線形 β スケジュール）
- [[summaries/2024-sd3]] — SD3（rectified flow の logit-normal/mode 時刻分布）
- [[summaries/2023-sdxl]] — SDXL（解像度依存 timestep shift）
