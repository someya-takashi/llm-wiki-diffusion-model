---
type: summary
source_path: raw/papers/Flow Matching for Generative Modeling.md
source_kind: paper
title: "Flow Matching for Generative Modeling"
authors: [Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le]
year: 2023
venue: ICLR 2023
ingested: 2026-06-23
tags: [flow-matching, probability-flow-ode, score-based-generative-models, generative-models, optimal-transport, cnf]
translation: "[[translations/2023-flow-matching]]"
---

# Flow Matching for Generative Modeling（フローマッチング）

> 原典: [[translations/2023-flow-matching]] ・ `raw/papers/Flow Matching for Generative Modeling.md`（arXiv:2210.02747）
> 著者・年・会議: Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le（Meta AI FAIR & Weizmann Institute）・2023・ICLR 2023

## 一言まとめ

「ノイズからデータへ至る**ベクトル場**を、固定した目標パスに**回帰するだけ**で学習する」という、拡散とは別系統のシンプルな生成パラダイム（[[flow-matching]]）。連続正規化フロー（CNF）を**シミュレーション不要**で大規模学習でき、拡散パスを特別な場合として内包しつつ、より高速・安定。特に**最適輸送（Optimal Transport, OT）パス**（ノイズとデータを直線でつなぐ）を使うと、拡散の曲がった軌道より少ステップで生成でき、ImageNet で拡散ベースラインを尤度・FID・サンプリングコストすべてで上回る。Stable Diffusion 3 など後続の大規模モデルの標準的な学習目的になった。

## 背景と問題意識

**CNF（Continuous Normalizing Flows）** は、ニューラルネットでパラメータ化したベクトル場 $v_t$ を ODE $\frac{d}{dt}\phi_t(x)=v_t(\phi_t(x))$ で積分し、ノイズ $p_0$ をデータ $p_1$ へ連続変形する決定論的生成モデル（[[probability-flow-ode]] と同じ ODE 生成の系譜）。表現力は高いが、従来の最尤学習は**高価な ODE シミュレーション**を順・逆に走らせる必要があり、高次元画像へスケールしなかった。一方、拡散モデル（[[denoising-diffusion]] / [[score-based-generative-models]]）は denoising score matching で効率学習できるが、**単純な拡散過程に縛られ**、確率パスの設計自由度が低く、サンプリングが遅い・専用高速化手法が要る。

問いは「拡散過程を経由せず、CNF を**シミュレーション不要かつ不偏勾配**で、任意の確率パスについて大規模学習できないか？」である。

## 提案手法 / 主張

### Flow Matching（FM）目的

目標確率パス $p_t$（$p_0$=ノイズ、$p_1$≈データ）を生成するベクトル場 $u_t$ に、ニューラル $v_t$ を回帰する：

$$
\mathcal{L}_{FM}(\theta)=\mathbb{E}_{t,p_t(x)}\|v_t(x)-u_t(x)\|^2
$$

だが周辺の $u_t$ も $p_t$ も閉形式で分からず、そのままでは扱えない。

### Conditional Flow Matching（CFM）— 核心

データ 1 点 $x_1$ ごとの**条件付きパス** $p_t(x|x_1)$ と**条件付き VF** $u_t(x|x_1)$ なら閉形式で扱える。これらを $q(x_1)$ で周辺化すると正しい周辺 $p_t$・$u_t$ になる（**定理1**）。そして条件付き量だけで回帰する CFM 目的

$$
\mathcal{L}_{CFM}(\theta)=\mathbb{E}_{t,q(x_1),p_t(x|x_1)}\|v_t(x)-u_t(x|x_1)\|^2
$$

は、**FM と勾配が完全に一致する**（**定理2**）。つまり扱いにくい周辺量に一切触れずに、サンプルごとの単純な回帰で CNF を学習できる。これは denoising score matching が「スコア」を条件付きで回帰したのを、「ベクトル場」の回帰へ一般化したものにあたる。

### ガウス条件付きパスの一般族と拡散の内包

条件付きパスを $p_t(x|x_1)=\mathcal{N}(x|\mu_t(x_1),\sigma_t(x_1)^2 I)$ とし、平均 $\mu_t$・std $\sigma_t$ を自由に設計（定理3 で対応する VF が閉形式で出る）。拡散の **VE / VP パス**はこの族の特別な場合で、その条件付き VF は [[probability-flow-ode]] の決定論 ODE の VF と一致する（付録 D）。**拡散パスでも FM を使うとスコアマッチングより安定・高性能**。

### Optimal Transport（OT）パス — ランドマーク

平均・std を時間に**線形**に変える $\mu_t=tx_1,\ \sigma_t=1-(1-\sigma_{\min})t$。これは 2 つのガウス間の**最適輸送の変位写像**で、粒子が**直線・等速**で動く。拡散パスがサンプリング中にオーバーシュートして曲がるのに対し、OT パスは直進するため学習・生成が速く汎化も良い。VF が時間方向に一定の向きを持ち、回帰タスクが単純になる（図2・3）。

## 実験結果と知見

- **ImageNet 32/64/128**：同一アーキ・同一設定で、FM-OT が拡散ベースライン（DDPM・SM・ScoreFlow）を **NLL（bits/dim）・FID・NFE（関数評価回数）すべてで上回る**（表1）。CIFAR-10 でも FM-OT が最良の NLL 2.99。ImageNet-128 で FID 20.9（IC-GAN を除き当時 SOTA級）。
- **少 NFE サンプリング**：既製の ODE ソルバー（dopri5 等）で高速生成。同じ誤差に達するのに拡散の約 60% の NFE。低 NFE でも良い FID（図7）。
- **速い学習**：FM-OT は FID をより速く下げ、サンプリングコストは学習中ほぼ一定（拡散は変動）。ImageNet-128 で先行研究の約 1/9 の画像スループットで到達。
- **超解像（64→256）**：FM-OT が SR3 を FID（3.4 vs 5.2）・IS で上回る（表2）。

## 限界・批判的視点

- **逐次 ODE 積分は依然必要**：少ステップ化はしても、GAN の一発生成のような単一評価ではない。
- **本論文は主に等方ガウス条件付きパス**：非等方ガウスやより一般のカーネルは将来課題（著者ら自身が言及）。
- **OT は「条件付き」最適**：条件付きフローは OT だが、周辺 VF が最適輸送解とは限らない（ミニバッチ OT 等の後続研究につながる）。
- ベンチマークは主に ImageNet 系の無条件生成。大規模テキスト条件付けの検証は後続（Stable Diffusion 3 等）に委ねられた。

## 用語と略称

- **FM** = Flow Matching（フローマッチング、本論文の枠組み）
- **CFM** = Conditional Flow Matching（条件付きフローマッチング、扱える目的関数）
- **CNF** = Continuous Normalizing Flows（連続正規化フロー、ベクトル場 ODE で定義される決定論的生成モデル）
- **vector field（ベクトル場）$v_t$** = 各点・各時刻の「進む向きと速さ」。ODE で積分するとフロー $\phi_t$ になる
- **probability path（確率パス）$p_t$** = 時間とともに変化する確率密度の列（$p_0$=ノイズ → $p_1$=データ）
- **OT** = Optimal Transport（最適輸送）。displacement map / interpolant = 変位写像／補間（直線パス）
- **VE / VP path** = Variance Exploding / Preserving 拡散パス（FM のガウス族の特別な場合）
- **NFE** = Number of Function Evaluations（サンプリング時のネット評価回数、低いほど高速）
- **BPD** = Bits Per Dimension（次元あたりビット数、NLL の指標）・**FID / IS** = 生成品質指標
- **simulation-free** = シミュレーション不要（学習時に ODE を積分し直さずに済む）

## 関連ページ

- [[concepts/flow-matching]] — FM/CFM の仕組みと OT パス（本論文の中核）
- [[concepts/probability-flow-ode]] — CNF/決定論 ODE 生成。拡散の確率フロー ODE は FM 拡散パスの VF と一致
- [[concepts/score-based-generative-models]] — 拡散パスは FM ガウス族の特別な場合。FM はスコアマッチングの安定な代替
- [[concepts/denoising-diffusion]] — VE/VP 拡散パスを内包
- [[concepts/diffusion-sampling]] — 既製 ODE ソルバーによる少 NFE サンプリング
- [[concepts/super-resolution]] — FM-OT による超解像（SR3 比較）
