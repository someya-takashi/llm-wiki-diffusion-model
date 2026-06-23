---
type: concept
aliases: [Controllable Generation, 可制御生成, inverse problems, 逆問題, conditional generation, ControlNet, zero convolution, spatial conditioning]
tags: [controllable-generation, score-based-generative-models, classifier-guidance, latent-diffusion, generative-models]
related:
  - "[[score-based-generative-models]]"
  - "[[classifier-guidance]]"
  - "[[latent-diffusion]]"
  - "[[image-inpainting]]"
  - "[[super-resolution]]"
summaries:
  - "[[summaries/2021-score-sde]]"
  - "[[summaries/2023-controlnet]]"
  - "[[summaries/2021-adm]]"
  - "[[summaries/2022-repaint]]"
  - "[[summaries/2023-anydoor]]"
  - "[[summaries/2023-mix-of-show]]"
  - "[[summaries/2023-sdxl]]"
updated: 2026-06-24
---

# Controllable Generation（可制御生成 / 逆問題）

**Controllable Generation（可制御生成）** とは、拡散モデルの生成を、ベースのテキストプロンプトを超える条件 $\mathbf{y}$（クラスラベル・破損していない部分・低解像度版・エッジ・深度・姿勢など）に従わせることである。これを実現する手法は大きく 2 系統に分けられる。

1. **推論時のスコア操作（学習不要〜軽量、逆問題アプローチ）** — 無条件に学習したモデルの**スコアに条件項を足す**だけで $\mathbf{x}\sim p(\mathbf{x}\mid\mathbf{y})$ を生成する。class-conditional 生成・[[image-inpainting]]（欠損補完）・[[super-resolution]]（超解像）・着色（colorization）が、**1 つの統一原理＝逆問題（inverse problem）** として扱える。これを連続時間 SDE 上で定式化したのが [[score-based-generative-models]] の Score-SDE 論文（[[summaries/2021-score-sde]]）。
2. **アーキテクチャ／ファインチューニング（アダプタアプローチ）** — 事前学習済みモデルに**追加の学習可能モジュール**を付けて、空間的な条件画像（エッジ・深度・姿勢・セグメント等）への追従を end-to-end で学ぶ。代表が **ControlNet**（[[summaries/2023-controlnet]]）。元モデルを凍結し品質を保ったまま、新しい制御能力を獲得する。

前者は「既存モデルをそのまま使う数学的な条件付け」、後者は「条件タイプごとに軽量に追加学習する実用的な制御」で、現代の画像生成（Stable Diffusion, [[latent-diffusion]]）では両者が併用される。

なお前者（推論時・学習不要）はさらに 2 つのスタイルに分かれる。**(1a) スコア勾配**を足す guidance（本ページの主題、$\nabla_\mathbf{x}\log p_t(\mathbf{y}\mid\mathbf{x})$ を加える）と、**(1b) 置き換え／射影ベース**——観測と整合する成分をサンプリング中に直接差し替える——である。後者の代表が **RePaint**（inpainting で既知領域を毎ステップ上書き＋resampling）で、学習不要条件付けの体系は [[training-free-conditioning]] にまとめた。

## アプローチ1：推論時のスコア操作（逆問題）

拡散モデルの生成は、各時刻のスコア $\nabla_\mathbf{x}\log p_t(\mathbf{x})$ で駆動される逆過程である（[[score-based-generative-models]]）。条件 $\mathbf{y}$ を課したいとき、ベイズの定理 $p_t(\mathbf{x}\mid\mathbf{y})\propto p_t(\mathbf{x})\,p_t(\mathbf{y}\mid\mathbf{x})$ から、**条件付きスコアは無条件スコアに尤度項の勾配を足すだけ**で得られる：

$$
\nabla_\mathbf{x}\log p_t(\mathbf{x}\mid\mathbf{y})=\underbrace{\nabla_\mathbf{x}\log p_t(\mathbf{x})}_{\text{無条件モデル}}+\underbrace{\nabla_\mathbf{x}\log p_t(\mathbf{y}\mid\mathbf{x})}_{\text{条件の項}}
$$

これを逆時間 SDE に差し込めば、無条件モデルを**再学習せずに**条件付き生成ができる。$\nabla_\mathbf{x}\log p_t(\mathbf{y}\mid\mathbf{x})$ をどう与えるかが各タスクの肝になる。

## 代表的なタスク（Score-SDE, Song ら 2021）

### class-conditional 生成

$\mathbf{y}$ がクラスラベルのとき、ノイズの乗った $\mathbf{x}(t)$ 上で **time-dependent classifier** $p_t(\mathbf{y}\mid\mathbf{x}(t))$ を学習し、その勾配を足す。順方向 SDE が扱いやすいので学習データ $(\mathbf{x}(t),\mathbf{y})$ は容易に作れる。これは **[[classifier-guidance]]（分類器ガイダンス）の連続時間における一般化であり、理論的根源**にあたる（分類器ガイダンス＝この尤度項勾配を強度 $w$ 倍して足す操作）。このアプローチを大規模画像生成で本格的に実用化したランドマークが **ADM（Dhariwal & Nichol 2021・[[summaries/2021-adm]]）** で、ノイズ画像分類器の勾配スケールを調整するだけで多様性↔忠実度を制御し、ImageNet で拡散モデルが初めて GAN を上回る原動力になった。詳細は [[classifier-guidance]] を参照。

### inpainting / imputation（[[image-inpainting]]）

既知部分 $\Omega(\mathbf{y})$ を条件に未知部分を埋める。未知次元だけの拡散過程を定義し、既知次元を各時刻でノイズ化したサンプルで「貼り付け」ながら逆過程を回すことで、**無条件モデルだけで**補完できる。

### colorization（着色）

グレースケール（既知）からカラー（未知）を復元。色チャネルを直交線形変換で分離すれば inpainting と同じ手続きに帰着する。直交変換なのでウィーナー過程の性質が保たれる。

### 一般逆問題

順過程 $p(\mathbf{y}\mid\mathbf{x})$ が分かる任意の逆問題（$\mathbf{y}$ から $\mathbf{x}$ を復元）に拡張できる。$\nabla_\mathbf{x}\log p_t(\mathbf{y}\mid\mathbf{x})$ を近似する一般的手法が示されており、補助モデルの学習なしに解ける場合がある。

## アプローチ2：アダプタ／ファインチューニング

### 代表手法：ControlNet（Zhang ら 2023）

スコア操作が「学習済みモデルをそのまま使う数学的な条件付け」なのに対し、**ControlNet**（[[summaries/2023-controlnet]]）は「事前学習済みモデルに学習可能モジュールを足し、条件タイプごとに end-to-end で追従を学ぶ」アダプタアプローチの代表である。エッジ・深度・姿勢・セグメンテーションといった**空間的な条件画像**で Stable Diffusion（[[latent-diffusion]]）を精密制御する。

核心は **zero convolution（ゼロ初期化 1×1 畳み込み）**。事前学習ブロック $\mathcal{F}(\cdot;\Theta)$ を凍結し、その学習可能コピー $\Theta_c$ を作り、両者を zero convolution $\mathcal{Z}$ で接続する：

$$
\bm{y}_c=\mathcal{F}(\bm{x};\Theta)+\mathcal{Z}\big(\mathcal{F}(\bm{x}+\mathcal{Z}(\bm{c};\Theta_{z1});\Theta_c);\Theta_{z2}\big)
$$

学習開始時は $\mathcal{Z}=0$ で $\bm{y}_c=\bm{y}$（元の出力と一致）なので、有害なノイズを加えず強力な事前学習バックボーンを壊さない。勾配が流れるにつれ重みがゼロから徐々に成長し、条件追従を獲得する（ある時点で急に従い始める「sudden convergence」）。条件ごとのデータが LAION-5B の 5 万分の 1 程度しかなくても過学習・破滅的忘却を起こさず、単一 GPU・小データでも産業モデル級の制御を達成する。複数 ControlNet の出力を単純加算すれば複数条件を同時適用でき、SD のトポロジーを変えないのでコミュニティ派生モデルへも再学習なしで転用できる。

ControlNet は同時期の T2I-Adapter や、LoRA・IP-Adapter などと並ぶ「アダプタ型条件制御」の系譜にあり、画像生成 AI の制御性を実用面で一変させた。詳細・実験・限界は [[summaries/2023-controlnet]] を参照。

このアダプタ型の条件付けは、テキストや空間マップだけでなく**参照画像（物体）**にも使える。**AnyDoor**（[[image-composition]] / [[summaries/2023-anydoor]]）は、参照物体の高周波マップを ControlNet スタイルの UNet エンコーダに通して detail map を作り、Stable Diffusion のデコーダ特徴に concat することで、物体をシーンの指定位置に合成する。条件が「テキスト→空間マップ→参照画像」へと広がっている例である。

## 既存知識との接続

- [[score-based-generative-models]]：可制御生成は逆時間 SDE のスコアに条件項を足すだけで実現される。Score-SDE がこの統一定式化を与えた。
- [[classifier-guidance]]：class-conditional 生成（time-dependent classifier の勾配を足す）は分類器ガイダンスそのもので、Score-SDE の条件付き逆時間 SDE がその一般形。さらに分類器を使わない [[classifier-free-guidance]] へと発展した。
- [[image-inpainting]] / [[super-resolution]]：これらは「既知の観測 $\mathbf{y}$ から未知を復元する」逆問題として、可制御生成の特別な場合。[[latent-diffusion]] では条件を連結（concat）して同じタスクを解く。
- [[latent-diffusion]]：ControlNet は Stable Diffusion（LDM）を制御対象とし、その U-Net エンコーダのコピーを zero convolution で接続する。[[classifier-free-guidance]] と組み合わせ、CFG 解像度重み付け（CFG-RW）でガイダンス強度を調整する。
- [[image-composition]]：AnyDoor の detail extractor は ControlNet スタイルのアダプタで、参照物体の高周波マップを条件にして物体をシーンへ合成する。
- [[multi-concept-customization]]：LoRA-Composer のレイアウト＋注意制御や Multi-LoRA Composition の復号制御も、重みを訓練し直さず推論時に複数概念を制御する系統。
- [[lora-merging]]：Mix-of-Show（[[summaries/2023-mix-of-show]]）の **regionally controllable sampling** は、global＋領域プロンプトを region-aware cross-attention（マスク $M_i$ で cross-attention 出力を差し替え）で注入し、多概念生成の属性結合を解く。ControlNet/T2I-Adapter の空間条件制御を多概念マージに組み合わせた例で、LoRA-Composer の領域注入の源流。
- [[latent-diffusion]]：SDXL（[[summaries/2023-sdxl]]）の **micro-conditioning** は、元画像サイズ・crop 座標・アスペクト比という**学習時メタデータ**を Fourier 埋め込みで条件化し timestep embedding に足す、別系統の可制御生成。スコア勾配ガイダンスや空間条件マップとは異なり、crop 条件で生成物のフレーミング（頭切れ回避・object-centered）を制御できる。

## 参考文献（summaries）

- [[summaries/2021-score-sde]] — Score-Based Generative Modeling through SDEs（§5・付録 I で可制御生成／逆問題を定式化）
- [[summaries/2023-controlnet]] — Adding Conditional Control to Text-to-Image Diffusion Models（ControlNet, zero convolution による空間条件制御）
- [[summaries/2021-adm]] — Diffusion Models Beat GANs on Image Synthesis（class-conditional 生成＝分類器ガイダンスの大規模実用化）
- [[summaries/2022-repaint]] — RePaint（置き換え／射影ベースの推論時条件付け、[[training-free-conditioning]]）
- [[summaries/2023-anydoor]] — AnyDoor（参照画像を ControlNet スタイルで条件付ける物体合成）
- [[summaries/2023-mix-of-show]] — Mix-of-Show（regionally controllable sampling：多概念生成の領域別 cross-attention 制御）
- [[summaries/2023-sdxl]] — SDXL（size/crop/aspect-ratio の micro-conditioning：学習時メタデータによる条件付け）
