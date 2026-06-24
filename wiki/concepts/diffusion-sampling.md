---
type: concept
aliases: [Diffusion Sampling, サンプリング, サンプラー, DDIM, ancestral sampling]
tags: [diffusion-sampling, denoising-diffusion, probability-flow-ode, generative-models]
related:
  - "[[denoising-diffusion]]"
  - "[[probability-flow-ode]]"
  - "[[latent-diffusion]]"
  - "[[flow-matching]]"
  - "[[stochastic-interpolants]]"
  - "[[noise-schedule]]"
summaries:
  - "[[summaries/2021-ddim]]"
  - "[[summaries/2021-score-sde]]"
  - "[[summaries/2023-flow-matching]]"
  - "[[summaries/2021-adm]]"
  - "[[summaries/2022-repaint]]"
  - "[[summaries/2022-edm]]"
updated: 2026-06-24
---

# Diffusion Sampling（拡散モデルのサンプリング）

**Diffusion Sampling（拡散モデルのサンプリング）** とは、学習済みの拡散モデル（ノイズ予測ネットワーク $\epsilon_\theta$）から実際に画像を生成する手続き＝**サンプラー／ソルバー**を指す。拡散モデルの学習（[[denoising-diffusion]]）と生成は分離して考えられ、**同じ学習済みモデルでもサンプラーを変えると生成速度と品質が大きく変わる**。拡散モデル実用化の最大の障壁だった「生成の遅さ」を解く中心的なテーマであり、ランドマーク手法が **DDIM（Denoising Diffusion Implicit Models, Song ら 2021）** である。

## なぜサンプリングが問題になるのか

[[denoising-diffusion]]（DDPM）の標準的な生成は **ancestral sampling（祖先的サンプリング）**：純粋なノイズ $\mathbf{x}_T$ から始め、逆過程 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ を $t=T,\dots,1$ と**逐次に**適用する。各ステップでニューラルネットを 1 回評価するため、$T=1000$ ステップなら 1 枚の生成にネットを 1000 回通す。GAN がネットワーク 1 回の順伝播で生成するのと比べ桁違いに遅く（2080 Ti で 256×256 を 5 万枚生成するのに約 1000 時間）、これが拡散モデルの実用化を阻んでいた。

なぜ $T$ を大きく取る必要があったか？ DDPM の変分的正当化は「各ステップのノイズ増分が小さければ逆過程もガウスで近似でき、$T$ が大きいほど近似が良い」というもの。DDIM はこの前提に疑問を投げ、**サンプリングのステップ数を学習から切り離す**。

## サンプラーの設計軸

- **確率性（stochastic ↔ deterministic）**：各ステップでランダムノイズを加えるか。DDPM は確率的、DDIM は決定論的（$\sigma=0$）。中間は連続パラメータ $\eta$ で補間できる。
- **ステップ数 / スケジュール**：何ステップで・どのタイムステップを使って生成するか。少ステップ化が高速化の鍵。
- **離散化の次数**：サンプリングを ODE の数値積分とみなすと、オイラー法（1 次）か高次法かでステップ数あたりの精度が変わる（[[probability-flow-ode]]）。

## 代表手法：DDIM（Song ら 2021）

DDIM の核心は、**DDPM の学習目的が周辺分布 $q(\mathbf{x}_t|\mathbf{x}_0)$ にしか依存しない**という観察。同じ周辺を持つ**非マルコフ的**な順過程の族（$\sigma$ でパラメータ化）を作っても、変分目的は DDPM と同じ $L_1$ に帰着する（定理 1: $J_\sigma=L_\gamma+C$）。したがって**学習済み DDPM をそのまま使い、サンプラーだけ差し替えられる**（再学習不要）。

サンプリング更新（式12）は「予測 $\mathbf{x}_0$」＋「$\mathbf{x}_t$ への方向」＋「ランダムノイズ $\sigma_t\epsilon$」の 3 項。

$$
\mathbf{x}_{t-1}=\sqrt{\alpha_{t-1}}\Big(\tfrac{\mathbf{x}_t-\sqrt{1-\alpha_t}\,\epsilon_\theta(\mathbf{x}_t)}{\sqrt{\alpha_t}}\Big)+\sqrt{1-\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(\mathbf{x}_t)+\sigma_t\epsilon_t
$$

- **$\sigma_t=0$（決定論）＝DDIM**：第 3 項が消え、$\mathbf{x}_T$ から $\mathbf{x}_0$ が一意に決まる暗黙生成モデルになる。
- **加速**：全 $T$ ではなく**タイムステップの部分列 $\tau$（$S\ll T$）**だけで生成できる。これも再学習不要。
- **成果**：CIFAR10・CelebA で **20〜100 ステップで 1000 ステップ DDPM 並みの FID**（10〜50× 高速化）。少ステップでは DDPM が急速に劣化するのに対し DDIM は一貫して高品質。
- **consistency**：同じ $\mathbf{x}_T$ ならステップ数を変えても高レベル特徴が一致 → $\mathbf{x}_T$ が情報的な潜在エンコーディングとして働き、**潜在空間での意味的補間**（slerp）が可能（DDPM は不可）。
- **ODE 視点**：DDIM はある ODE のオイラー積分＝確率フロー ODE の特殊例（[[probability-flow-ode]]）。決定論ゆえ encode/再構成も可能。

詳細・実験・限界は [[summaries/2021-ddim]] を参照。

## 代表手法：Predictor-Corrector・逆拡散サンプラー（Score-SDE, Song ら 2021）

連続時間 SDE の枠組み（[[score-based-generative-models]]）からは、逆時間 SDE を数値的に解く一連のサンプラーが導かれる（[[summaries/2021-score-sde]]）。

- **逆拡散サンプラー（reverse diffusion sampler）**：順方向 SDE と同じ離散化方式で逆時間 SDE を離散化する汎用サンプラー。DDPM の祖先的サンプリングは逆時間 VP-SDE の一離散化にすぎないと示され、連続枠組みに統一された。
- **Predictor-Corrector（PC）サンプラー**：各ステップで「**predictor**（数値 SDE ソルバーで次状態を予測）」と「**corrector**（スコアベース MCMC＝Langevin 動力学で周辺分布を補正）」を交互適用する。同じ計算量（スコア評価回数）で、predictor のみ・corrector のみを一貫して上回る。SMLD（恒等 predictor＋Langevin corrector）と DDPM（祖先的 predictor＋恒等 corrector）を特別な場合として一般化する。
- **確率フロー ODE サンプラー**：決定論的 ODE（[[probability-flow-ode]]）を black-box ODE ソルバーで解く。高速・適応的だが、corrector なしでは品質がやや劣りやすい。

これらにより、サンプリングは「確率的（SDE）↔ 決定論的（ODE）」「predictor のみ ↔ PC」という設計軸で整理された。

## ガイダンスとサンプラーの組合せ（classifier-guided DDIM）

サンプラーは [[classifier-guidance]] や [[classifier-free-guidance]] と直交して組み合わせられる。ADM（[[summaries/2021-adm]]）は、決定論的な DDIM に分類器ガイダンスを効かせる版（Algorithm 2）を示した。確率的サンプリング向けの「平均をシフトする」導出は DDIM に使えないため、**ノイズ予測自体を修正**する：

$$
\hat\epsilon(\mathbf{x}_t)=\epsilon_\theta(\mathbf{x}_t)-\sqrt{1-\bar\alpha_t}\,\nabla_{\mathbf{x}_t}\log p_\phi(y\mid\mathbf{x}_t)
$$

この $\hat\epsilon$ を通常の DDIM 更新にそのまま差し込む。ADM はこの classifier-guided DDIM で、**わずか 25 ステップ**でも BigGAN 級の FID を達成できることを示し、ガイダンスが少ステップ高速サンプリングと両立することを実証した（ImageNet 256 で ADM-G 25 ステップ = FID 5.44）。

## サンプリング時の調和：resampling（RePaint）

サンプラーの工夫は高速化だけでなく、**条件付き生成の品質**にも使える。**RePaint**（[[summaries/2022-repaint]]）の resampling は、推論時 inpainting（[[training-free-conditioning]]）で既知領域と生成領域を調和させるサンプリング時戦略である。各時刻で得た $\mathbf{x}_{t-1}$ を**前方拡散で $\mathbf{x}_t$ に戻し**（$\mathbf{x}_t\sim\mathcal N(\sqrt{1-\beta_t}\mathbf{x}_{t-1},\beta_t\mathbf I)$）、再度デノイズする操作を **jump length $j$・$r$ 回**繰り返す。拡散時刻は全体として減少しつつ局所的に何度も上方ジャンプする鋸歯状になる。

重要なのは、これが **slowing down（拡散を遅くする＝1 ステップの変化を小さくして総ステップ数を増やす）とは別物**だという点。slowing down は同じ計算予算を費やしても境界の不調和を解けないが、resampling は予算を「調和」に使う（[[summaries/2022-repaint]] 表2 のアブレーション）。学習（モデル）とサンプリング（生成手続き）を分離できる拡散モデルだからこそ可能な、後付けの品質改善である。

## 代表手法：EDM の Heun サンプラーと churn（Karras ら 2022）

DDIM の「サンプリング＝ODE 数値積分」という視点を体系化したのが **EDM**（[[summaries/2022-edm]]）である。拡散の設計空間をサンプリング・前処理・学習に分解し、**サンプラーは学習と直交する**（事前学習モデルにサンプラーだけ差し替えてよい）という主張を実証した。サンプリング側の主要な貢献：

- **Heun 2 次法（決定論サンプラー）**：オイラー法（1 次、局所打ち切り誤差 $\mathcal{O}(h^2)$）に台形補正を加え、1 回追加の denoiser 評価で $\mathcal{O}(h^3)$ を得る。打ち切り誤差/NFE のトレードオフが優秀で、同じ FID に遥かに少ない NFE で到達。
- **ρ=7 の時刻離散化**：$\sigma_{i}=(\sigma_{\rm max}^{1/\rho}+\frac{i}{N-1}(\sigma_{\rm min}^{1/\rho}-\sigma_{\rm max}^{1/\rho}))^{\rho}$ で低ノイズ側のステップを短くする（詳細は [[noise-schedule]]）。**σ(t)=t** を採ると σ と時刻が交換可能になり ODE 軌道がほぼ直線化し、少ステップ化に効く。
- **churn（確率サンプラー, Algorithm 2）**：決定論的 ODE ステップに Langevin 的な「ノイズ追加→ODE で除去」を交互適用し、先のステップの誤差を**自己訂正**する。$S_{\rm churn}/S_{\rm tmin}/S_{\rm tmax}/S_{\rm noise}$ で確率性の量を制御。過剰だと detail loss・過飽和。
- **成果**：事前学習 VP/VE/DDIM にサンプラーを差し替えるだけで NFE を **7.3×/300×/3.2×** 削減。CIFAR-10 を 35 NFE で SOTA、ImageNet-64 をサンプラーのみで FID 2.07→1.55。

EDM の Heun サンプラーは後続の標準になり、**SD3**（[[summaries/2024-sd3]]）や **Stochastic Interpolants**（[[summaries/2024-stochastic-interpolants]]）が採用する。なお、より少ステップ・高精度なソルバー（DPM-Solver 等の高次法）や、サンプリング過程を蒸留する consistency models へと発展する流れもある（これらは未取り込み）。

## 既存知識との接続

- [[denoising-diffusion]]：DDIM は DDPM の学習済みモデルを再学習なしに高速サンプリングする手法。学習は DDPM、生成は DDIM という分業。
- [[probability-flow-ode]]：DDIM の決定論サンプリングは確率フロー ODE のオイラー離散化として理解できる。
- [[latent-diffusion]]：LDM / Stable Diffusion は標準サンプラーとして DDIM を採用し、数十〜250 ステップで生成する。
- [[score-based-generative-models]]：祖先的サンプリングや Langevin 動力学（多ステップ）に対し、DDIM は決定論的に少ステップ化する。連続時間 SDE 枠組みが predictor-corrector・逆拡散・ODE サンプラーを統一的に与える。
- [[flow-matching]]：FM（特に OT パス）はノイズ→データを直線で結ぶため、既製 ODE ソルバーで拡散より少ない NFE（関数評価回数）で生成でき、サンプリング効率の新基準を与えた。
- [[stochastic-interpolants]]：確率的補間（[[summaries/2024-stochastic-interpolants]]）は同じ補間密度を、決定論的 ODE（$\epsilon=0$）でも拡散係数 $\epsilon(t)$ を持つ SDE でもサンプルできることを示す。サンプリング法（ODE か SDE か・$\epsilon$ の大小）を学習後に選べ、ODE は厳密尤度・適応積分、SDE はスコア近似誤差への頑健性というトレードオフを与える。
- [[noise-schedule]]：サンプラーが使う時刻離散化 $\{\sigma_i\}$（EDM の ρ=7 等）はノイズスケジュールの推論側の側面。サンプラー（Heun）とスケジュール（ρ）は一体で少ステップ・高品質を決める。

## 参考文献（summaries）

- [[summaries/2021-ddim]] — Denoising Diffusion Implicit Models（決定論サンプラー DDIM、Song, Meng, Ermon, ICLR 2021）
- [[summaries/2021-score-sde]] — Score-Based Generative Modeling through SDEs（predictor-corrector・逆拡散・確率フロー ODE サンプラー）
- [[summaries/2023-flow-matching]] — Flow Matching（OT パスによる少 NFE・既製 ODE ソルバーでの高速サンプリング）
- [[summaries/2021-adm]] — Diffusion Models Beat GANs on Image Synthesis（classifier-guided DDIM・25 ステップ高速サンプリング）
- [[summaries/2022-repaint]] — RePaint（resampling＝時刻を行き来して調和させるサンプリング時戦略）
- [[summaries/2022-edm]] — EDM（Heun 2 次サンプラー・ρ=7 時刻離散化・churn 確率サンプラー、設計空間の分解）
