---
type: concept
aliases: [DDPM, Denoising Diffusion, 拡散モデル, diffusion model]
tags: [denoising-diffusion, generative-models, image-generation]
related:
  - "[[score-based-generative-models]]"
  - "[[latent-diffusion]]"
  - "[[diffusion-sampling]]"
  - "[[diffusion-model-architecture]]"
summaries:
  - "[[summaries/2020-ddpm]]"
  - "[[summaries/2021-adm]]"
updated: 2026-06-23
---

# Denoising Diffusion（ノイズ除去拡散モデル）

**Denoising Diffusion（ノイズ除去拡散モデル）** とは、データに少しずつノイズを加えて最終的に純粋なノイズへと壊していく過程（**順過程 / forward process**）を、逆向きにたどってノイズからデータを少しずつ復元する過程（**逆過程 / reverse process**）を学習する生成モデルの枠組みである。逆過程の 1 ステップは「いまの画像に乗っているノイズを少し取り除く（denoising）」操作に対応するため、この名前がつく。

画像生成における拡散モデルの実用化を決定づけたのが **DDPM（Denoising Diffusion Probabilistic Models, Ho ら 2020）** であり、本ページではこの概念の定式化と、ランドマーク手法としての DDPM を中心に解説する。理論的な双子である [[score-based-generative-models]]（スコアベース生成モデル）、効率化版の [[latent-diffusion]]（潜在拡散＝Stable Diffusion 系）、生成を高速化する [[diffusion-sampling]]（サンプラー、DDIM 等）と密接に関連する。

## 全体像：なぜ「ノイズを足して、引く」で生成できるのか

直感的には次の通り。

1. きれいな画像に、ほんの少しずつガウスノイズを何百〜千回と足していくと、最後はテレビの砂嵐のような**完全なノイズ**になる（順過程）。各ステップの変化はごく小さいので、「1 ステップ前に戻す」操作は比較的単純な確率分布（ガウス分布）で表せる。
2. そこで、「ノイズだらけの画像を、1 ステップだけきれいにする」関数をニューラルネットで学習しておけば、**純粋なノイズから出発して、その関数を何百回も繰り返す**ことで新しい画像を生成できる（逆過程）。

「いきなり完璧な画像を作る」のではなく「少しずつ汚す／少しずつ直す」を多数ステップに分解する点が、GAN（一発生成）や VAE と異なる拡散モデルの本質である。

## 定式化（DDPM）

### 順過程（forward process）

データ $\mathbf{x}_0$ に分散スケジュール $\beta_1,\dots,\beta_T$ に従ってガウスノイズを加えるマルコフ連鎖：

$$
q(\mathbf{x}_t|\mathbf{x}_{t-1})=\mathcal{N}\!\big(\mathbf{x}_t;\sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\,\beta_t\mathbf{I}\big)
$$

学習パラメータを持たず**固定**される。重要な性質として、$\alpha_t:=1-\beta_t$、$\bar\alpha_t:=\prod_{s=1}^{t}\alpha_s$ とおくと、任意の時刻 $t$ のノイズ画像を**閉形式で一発サンプリング**できる：

$$
q(\mathbf{x}_t|\mathbf{x}_0)=\mathcal{N}\!\big(\mathbf{x}_t;\sqrt{\bar\alpha_t}\,\mathbf{x}_0,\,(1-\bar\alpha_t)\mathbf{I}\big)
\quad\Longleftrightarrow\quad
\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\;\;\boldsymbol\epsilon\sim\mathcal{N}(0,\mathbf{I})
$$

この式のおかげで、学習時に途中の $t-1$ 個のステップを順に計算する必要がなく、ランダムな $t$ を選んで直接 $\mathbf{x}_t$ を作れる。

### 逆過程（reverse process）

ノイズ $\mathbf{x}_T\sim\mathcal{N}(0,\mathbf{I})$ から出発し、学習されたガウス遷移でデータを復元する：

$$
p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)=\mathcal{N}\!\big(\mathbf{x}_{t-1};\boldsymbol\mu_\theta(\mathbf{x}_t,t),\,\boldsymbol\Sigma_\theta(\mathbf{x}_t,t)\big)
$$

DDPM は分散 $\boldsymbol\Sigma_\theta=\sigma_t^2\mathbf{I}$ を固定し、平均 $\boldsymbol\mu_\theta$ のみを学習する。

### 学習目的：変分下界からノイズ予測へ

負の対数尤度の**変分下界（VLB, Variational Lower Bound, 対数尤度を下から押さえる量）** は、各ステップの KL ダイバージェンスの和に分解できる（$L_T+\sum L_{t-1}+L_0$）。DDPM の鍵となる工夫は、平均 $\boldsymbol\mu_\theta$ を直接予測する代わりに「$\mathbf{x}_t$ に乗っているノイズ $\boldsymbol\epsilon$」を予測するネットワーク $\boldsymbol\epsilon_\theta$ にパラメータ化し直したこと（**ε 予測**）。これにより目的関数は、重み係数を捨てた極めて単純な二乗誤差になる：

$$
L_\text{simple}(\theta)=\mathbb{E}_{t,\mathbf{x}_0,\boldsymbol\epsilon}\Big[\big\lVert\boldsymbol\epsilon-\boldsymbol\epsilon_\theta(\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\,t)\big\rVert^2\Big]
$$

学習は「画像にランダムな時刻 $t$ のノイズを乗せ、そのノイズを当てさせる」だけ（原典アルゴリズム 1）。生成は、ノイズから始めて各ステップで $\boldsymbol\epsilon_\theta$ の予測を使い $\mathbf{x}_{t-1}$ を計算する操作を $t=T,\dots,1$ と繰り返す（アルゴリズム 2、[[diffusion-sampling]]）。

## 代表手法：DDPM（Ho ら 2020）

- **位置づけ**：拡散モデルが GAN に匹敵する画像を生成できることを初めて実証し、現在の画像生成 AI ブームの起点となったランドマーク。詳細・実験・限界は要約 [[summaries/2020-ddpm]] を参照。
- **成果**：無条件 CIFAR10 で **FID 3.17 / IS 9.46**（当時の最先端）、LSUN・CelebA-HQ 256×256 で ProgressiveGAN 級。
- **アーキテクチャ**：時刻 $t$ を正弦波位置埋め込みで条件付けた **U-Net**（$16\times16$ 解像度に自己注意）。
- **ε 予測＋$L_\text{simple}$** がサンプル品質に最も効く、というアブレーション結果。

### アーキテクチャ改良：IDDPM・ADM

DDPM の定式化（ε 予測・$L_\text{simple}$）はそのままに、ノイズ予測ネットワーク自身を強くする流れが続いた。**IDDPM（Nichol & Dhariwal 2021）** は分散 $\boldsymbol\Sigma_\theta$ を固定せず学習し（learned variance）、cosine ノイズスケジュールとハイブリッド目的 $L_\text{simple}+\lambda L_\text{vlb}$ を導入。続く **ADM（Dhariwal & Nichol 2021, [[summaries/2021-adm]]）** は、多解像度 attention・BigGAN residual block・**AdaGN（時刻とクラス埋め込みを GroupNorm のスケール／シフトとして注入）** といった U-Net 改良だけで FID を大きく押し下げ、拡散モデルが GAN を超える土台を築いた。これらアーキテクチャ設計の系譜は [[diffusion-model-architecture]] に整理している。

DDPM 以降、専用ページ化に足る派生（DDIM、改良ノイズスケジュール、classifier-free guidance、潜在拡散など）が続くが、それらは本ページや [[diffusion-sampling]]・[[latent-diffusion]] 等の概念ページ内で扱う方針（CLAUDE.md §1）。

## 既存知識との接続

- [[score-based-generative-models]]：DDPM の ε 予測目的関数は denoising score matching と等価で、サンプリングは学習スコアによる Langevin 動力学に対応する。拡散モデルとスコアベースモデルは同じものの二つの顔。連続時間で見ると **DDPM は VP-SDE（分散保存型 SDE）の離散化**にあたる（[[summaries/2021-score-sde]]）。
- [[diffusion-sampling]]：DDPM の弱点である「$T=1000$ ステップで遅い」を解く DDIM 等の高速サンプラー。DDIM は学習済み DDPM をそのまま使い、再学習なしに 10〜50× 高速化する。
- [[latent-diffusion]]：ピクセル空間で重い拡散を、オートエンコーダの潜在空間で行うことで高解像度・テキスト条件付き生成（Stable Diffusion）を実用化。
- [[flow-matching]]：DDPM の拡散パス（VP/VE）は、フローマッチングのガウス条件付きパスの特別な場合として内包される。FM は拡散の確率的構築を経由せず確率パスを直接指定する一般化された見方を与える。
- [[diffusion-model-architecture]]：ε 予測ネットワーク（U-Net）の設計。ADM の改良 U-Net・AdaGN が拡散モデルの標準アーキテクチャを確立した。
- [[classifier-guidance]]：DDPM の枠組みに「条件忠実度↔多様性」のノブを与えた手法（ADM が導入）。
- [[training-free-conditioning]]：学習済み無条件 DDPM を凍結したまま推論時だけ条件付ける応用（RePaint の inpainting 等）。順過程の閉形式 $q(x_t\mid x_0)$ をそのまま活用する。
- [[subject-driven-generation]]：DreamBooth は拡散の二乗誤差損失の上に prior 保存項を足して少数画像で fine-tune し、特定被写体をモデルに埋め込む personalization。

## 参考文献（summaries）

- [[summaries/2020-ddpm]] — Denoising Diffusion Probabilistic Models（Ho, Jain, Abbeel, NeurIPS 2020）
- [[summaries/2021-adm]] — Diffusion Models Beat GANs on Image Synthesis（改良 U-Net・AdaGN・classifier guidance）
