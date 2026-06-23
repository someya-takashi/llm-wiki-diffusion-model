---
type: concept
aliases: [Flow Matching, FM, フローマッチング, Conditional Flow Matching, CFM, CNF, Continuous Normalizing Flow]
tags: [flow-matching, probability-flow-ode, score-based-generative-models, generative-models, optimal-transport]
related:
  - "[[probability-flow-ode]]"
  - "[[score-based-generative-models]]"
  - "[[denoising-diffusion]]"
  - "[[diffusion-sampling]]"
  - "[[stochastic-interpolants]]"
summaries:
  - "[[summaries/2023-flow-matching]]"
  - "[[summaries/2024-sd3]]"
  - "[[summaries/2024-stochastic-interpolants]]"
updated: 2026-06-24
---

# Flow Matching（フローマッチング）

**Flow Matching（フローマッチング, FM）** とは、ノイズからデータへ至る連続的な変形を表す**ベクトル場（vector field）** を、あらかじめ固定した目標パスに**回帰（regression）** するだけで学習する生成モデリングの枠組みである。拡散モデル（[[denoising-diffusion]]）が「ノイズ除去」を学ぶのに対し、FM は「各点・各時刻でどちらへどれだけ進むか」という速度場を直接学ぶ。拡散とは別系統に見えるが、**拡散パスを特別な場合として内包**し、さらに**最適輸送（Optimal Transport, OT）パス**という直線的でより高速・安定な代替を与える。ランドマークは **Flow Matching（Lipman ら, ICLR 2023）** で、Stable Diffusion 3 など後続の大規模生成モデルの標準的な学習目的になった。

## 前提：連続正規化フロー（CNF）

FM が学習する対象は **CNF（Continuous Normalizing Flows, 連続正規化フロー, Chen ら 2018）** である。CNF は、ニューラルネットでパラメータ化した時間依存ベクトル場 $v_t$ を常微分方程式

$$
\frac{d}{dt}\phi_t(x)=v_t(\phi_t(x)),\qquad \phi_0(x)=x
$$

で積分してフロー $\phi_t$ を作り、単純な事前分布 $p_0$（ノイズ）を複雑なデータ分布 $p_1$ へ連続変形する**決定論的**生成モデルである。これは [[probability-flow-ode]] と同じ「ODE による決定論生成」の系譜にある（拡散の確率フロー ODE も CNF の一種）。CNF は表現力が高い一方、従来の最尤学習は順・逆に高価な ODE シミュレーションを要し、高次元画像へスケールしなかった。FM はこの学習をシミュレーション不要にする。

## Flow Matching の仕組み

### FM 目的と、その扱いにくさ

目標確率パス $p_t$（$p_0$=ノイズ、$p_1$≈データ）を生成するベクトル場 $u_t$ に $v_t$ を回帰する：

$$
\mathcal{L}_{FM}(\theta)=\mathbb{E}_{t,p_t(x)}\|v_t(x)-u_t(x)\|^2
$$

しかし所望の $p_t$ を生成する $u_t$ は閉形式で分からず、このままでは使えない。

### Conditional Flow Matching（CFM）— 核心

データ 1 点 $x_1$ ごとの**条件付きパス** $p_t(x|x_1)$ と**条件付きベクトル場** $u_t(x|x_1)$ なら閉形式で扱える。これらを $q(x_1)$ で周辺化すると、正しい周辺パス $p_t$・周辺 VF $u_t$ が得られる（**定理1**）。そして条件付き量だけで回帰する

$$
\mathcal{L}_{CFM}(\theta)=\mathbb{E}_{t,q(x_1),p_t(x|x_1)}\|v_t(x)-u_t(x|x_1)\|^2
$$

は、**扱いにくい FM 目的と勾配が完全に一致する**（**定理2**）。つまり周辺量に一切触れず、サンプルごとの単純な二乗誤差回帰で CNF を学習できる。これは [[score-based-generative-models]] の denoising score matching が「スコア」を条件付きで回帰したのを、「ベクトル場」へ一般化したものにあたる。

### ガウス条件付きパスと拡散の内包

条件付きパスを $p_t(x|x_1)=\mathcal{N}(x|\mu_t(x_1),\sigma_t(x_1)^2 I)$ と置き、平均 $\mu_t$・標準偏差 $\sigma_t$ を自由に設計する（定理3 で対応 VF が閉形式に決まる）。拡散の **VE / VP パス**はこの族の特別な場合であり、その条件付き VF は [[probability-flow-ode]] の決定論 ODE の VF と一致する（[[summaries/2023-flow-matching]] 付録 D）。重要なのは、**拡散パスでも FM 目的で学習するとスコアマッチングより安定・高性能**な点で、FM は「拡散の確率的構築を経由せず、確率パスを直接指定する」別の見方を与える。

## 代表手法：Optimal Transport（OT）パス

FM が拡散を超える鍵が **OT パス**。平均・std を時間に**線形**に変える：

$$
\mu_t(x_1)=t\,x_1,\qquad \sigma_t(x_1)=1-(1-\sigma_{\min})t
$$

これは標準ガウスとデータ点ガウスの間の**最適輸送の変位写像**で、粒子が**直線・等速**で移動する。拡散パスがサンプリング中に最終サンプルをオーバーシュートして曲がるのに対し、OT パスは直進するため、

- **少ステップ生成**：既製の ODE ソルバーで、拡散の約 60% の NFE（関数評価回数）で同じ誤差に到達。
- **速い学習**：FID をより速く下げ、サンプリングコストは学習中ほぼ一定。
- **良い汎化・高品質**：ImageNet 32/64/128 で拡散ベースライン（DDPM/SM/ScoreFlow）を NLL・FID・NFE すべてで上回り、超解像でも SR3 を FID/IS で上回る。

詳細・実験・限界は [[summaries/2023-flow-matching]] を参照。

## 代表手法：Rectified Flow と大規模化（Stable Diffusion 3, Esser ら 2024）

**Rectified Flow（整流フロー, RF）** は、OT パスと本質的に同じ「直線・線形補間」を採る FM のインスタンスで、データ $x_0$ とノイズ $\epsilon$ を

$$
z_t=(1-t)x_0+t\epsilon
$$

で結び、速度場 $v_\Theta$ を CFM 目的で回帰する（$w_t^{\text{RF}}=\frac{t}{1-t}$）。直線ゆえ少ステップ・低曲率で、拡散の確率フロー ODE より速く解ける（[[probability-flow-ode]]）。

これを大規模 text-to-image で初めて決定的に確立したのが **Stable Diffusion 3（SD3）**（[[summaries/2024-sd3]]）である。鍵は、一様時刻サンプリング $\mathcal{U}(t)$ を**中間時刻に重みを置く分布**に替える点：速度予測目標 $\epsilon-x_0$ は端点では単なる平均で簡単、中間ほど難しいので、**logit-normal**（logit 変換した時刻を正規分布でサンプル、$\pi_{\text{ln}}(t;m,s)$）や **mode** サンプラーで中間を厚くする。これは重み付き損失 $w_t^\pi=\frac{t}{1-t}\pi(t)$ と等価。61 定式化の大規模比較で **rf/lognorm(0.00, 1.00)** が一貫して最良で、EDM・LDM-Linear・一様 RF を上回り、特に少ステップで強い。SD3 はこれを **MM-DiT**（[[diffusion-model-architecture]]）と組み合わせ 8B までスケールした。SD3 により rectified flow / flow matching は SDXL までの拡散定式化に代わる実用標準になった。

## 既存知識との接続

- [[probability-flow-ode]]：FM が学習する CNF は決定論的 ODE 生成。拡散の確率フロー ODE は FM の拡散パスの VF と一致し、FM はそれを「確率パスを直接指定する」視点へ一般化する。
- [[score-based-generative-models]]：CFM は denoising score matching（スコア回帰）のベクトル場版。拡散パスは FM ガウス族の特別な場合で、FM はスコアマッチングの安定な代替になる。
- [[denoising-diffusion]]：VE/VP 拡散を FM のガウス条件付きパスとして内包。
- [[diffusion-sampling]]：FM-OT は直線パスゆえ既製 ODE ソルバーで少 NFE 生成でき、サンプリング効率の新基準を与えた。

## さらなる一般化：Stochastic Interpolants

FM・rectified flow をさらに一般化し、**flows（決定論 ODE）と diffusions（確率 SDE）を 1 つの枠組みに統一**するのが [[stochastic-interpolants]]（Albergo–Boffi–Vanden-Eijnden）である。有限時間で 2 分布を結ぶ補間 $x_t=\alpha_t x_0+\beta_t x_1(+\gamma_t z)$ を定義し、速度場 $b$ とスコア $s$ を二乗損失で学習する。FM の条件付きパス・rectified flow はこの枠組みの特別な場合で、潜在変数 $\gamma_t z$ と拡散係数 $\epsilon$ を足すと SDE 生成（拡散）に、外すと ODE 生成（フロー）になる。OT パスの「条件付き最適」を周辺レベルへ近づける mini-batch OT も関連する一般化。

## 参考文献（summaries）

- [[summaries/2023-flow-matching]] — Flow Matching for Generative Modeling（Lipman, Chen, Ben-Hamu, Nickel, Le, ICLR 2023）
- [[summaries/2024-sd3]] — Scaling Rectified Flow Transformers（SD3。rectified flow＋改良サンプラーを大規模 text-to-image で確立）
- [[summaries/2024-stochastic-interpolants]] — Stochastic Interpolants（flows と diffusions を統一する一般化枠組み）
