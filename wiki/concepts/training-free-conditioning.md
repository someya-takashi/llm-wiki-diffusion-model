---
type: concept
aliases: [Training-Free Conditioning, 学習不要の条件付け, 推論時条件付け, inference-time conditioning, replacement-based conditioning]
tags: [training-free-conditioning, image-inpainting, denoising-diffusion, diffusion-sampling, generative-models]
related:
  - "[[image-inpainting]]"
  - "[[controllable-generation]]"
  - "[[diffusion-sampling]]"
  - "[[denoising-diffusion]]"
summaries:
  - "[[summaries/2022-repaint]]"
updated: 2026-06-24
---

# Training-Free Conditioning（学習不要・推論時の条件付け生成）

**Training-Free Conditioning（学習不要の条件付け）** とは、**事前学習済みの無条件拡散モデルを「画像の prior（事前分布）」として固定したまま、再学習せずにサンプリング過程だけを操作して条件 $\mathbf{y}$ に従わせる**パラダイムである。条件タイプ（マスク・低解像度版・参照画像など）ごとにモデルを学習し直す必要がないため、(1) 任意の条件・タスクに即座に汎化でき、(2) 強力な無条件モデルの生成能力（意味的に整合した高品質生成）をそのまま借りられる、という利点がある。ランドマーク手法は **RePaint（Lugmayr ら 2022・[[summaries/2022-repaint]]）** で、凍結した無条件 DDPM だけで任意マスクの [[image-inpainting]] を実現した。

## なぜ「学習不要」が嬉しいのか

条件付き生成の素朴な方法は「条件付きモデルを学習する」ことだが（例：マスク条件付き inpainting モデル、[[latent-diffusion]] の concat 方式）、これには 2 つの代償がある。第一に、**条件の種類ごとにデータと再学習が要る**——マスク形状の分布で学習すると未知マスクに弱い、といった過学習が起こる。第二に、大規模な無条件モデルを毎回ファインチューニングするのは高コスト。

学習不要アプローチは「**無条件モデルはそのまま、条件は推論時に注入する**」。1 つの汎用 prior を、inpainting・超解像・着色・参照画像ガイドなど多様なタスクに使い回せる。

## 条件付けの 3 系統（本 wiki での整理）

拡散モデルを推論時に条件付ける方法は、大きく次に分けられる。本ページは特に (C) の置き換え／射影ベースを主題とし、(A)(B) は関連ページに委ねる。

- **(A) スコア勾配による guidance**：条件のスコア（対数尤度勾配）をベース拡散スコアに足す。classifier guidance（[[classifier-guidance]]）や、逆問題としての条件付き逆時間 SDE（[[controllable-generation]] / [[score-based-generative-models]]）がこれ。条件項 $\nabla_\mathbf{x}\log p_t(\mathbf{y}\mid\mathbf{x})$ をどう与えるかが肝。
- **(B) アダプタ／ファインチューニング**：軽量モジュールを追加学習する。ControlNet（[[controllable-generation]]）など。**これは学習を伴う**ので厳密には training-free ではない。
- **(C) 置き換え／射影ベース（replacement / projection）**：観測と整合する成分をサンプリング中に**直接差し替える**。スコアに項を足すのでも追加学習でもなく、「既知部分を毎ステップ正しいノイズレベルの観測で上書きする」。RePaint の inpainting がこれ。学習も勾配計算も不要で最も手軽。

## 代表手法：RePaint（Lugmayr ら 2022）

[[image-inpainting]] のために、凍結した無条件 DDPM（[[denoising-diffusion]]）を prior に使う（実際には [[summaries/2021-adm]] の guided-diffusion 事前学習モデル）。

### 置き換えによる条件付け

各逆拡散ステップで、マスク $m$ により既知/未知を合成する：

$$
x_{t-1}=m\odot x_{t-1}^{\text{known}}+(1-m)\odot x_{t-1}^{\text{unknown}}
$$

既知領域 $x_{t-1}^{\text{known}}$ は真の入力 $x_0$ を時刻 $t$ 相当まで前方拡散して差し込み（順過程の閉形式 $\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)\mathbf I)$）、未知領域 $x_{t-1}^{\text{unknown}}$ は DDPM が普通に生成する。逆ステップが $x_t$ にしか依存しないため、既知部を毎ステップ「正しいノイズレベルの本物」で上書きしても分布の整合が崩れない。

### Resampling（調和のための時刻の行き来）

単純な上書きだけだと境界が不調和になる（既知部のノイズ化が生成中の内容を見ていないため）。RePaint は得られた $x_{t-1}$ を**前方拡散で $x_t$ に戻して再デノイズ**する操作を、jump length $j$・$r$ 回繰り返し、DDPM の「データ分布に整合させる」性質で既知/生成領域を調和させる。これは少ステップ化（slowing down）とは別物のサンプリング時戦略で、詳細は [[diffusion-sampling]] と [[summaries/2022-repaint]] を参照。

### 成果と限界

- 任意マスク（thin/thick/extreme）に汎化し、CelebA-HQ・ImageNet・Places2 で GAN・自己回帰手法をユーザー調査で上回る。
- **限界**：画像ごとに DDPM を多数回評価するため遅い。観測 prior に偏る（ImageNet では犬が出やすい）。

## 関連手法（本 wiki 未取り込み）

- **ILVR**：参照画像の低周波成分で逆過程を誘導する置き換え系。inpainting には直接使えない（マスク部は高低周波とも欠落するため）。
- **SDEdit**：誘導画像を中間時刻からデノイズし直して編集する。RePaint は SDEdit より inpainting で優位（[[summaries/2022-repaint]] 表4）。
- **DDRM など**：線形逆問題を無条件拡散 prior で解く後続研究。

## 既存知識との接続

- [[image-inpainting]]：RePaint は学習不要 inpainting の代表。学習型の LDM-inpainting（concat）と対照的。
- [[controllable-generation]]：スコア勾配 guidance（A）とアダプタ（B）を扱う姉妹概念。本ページは置き換え系（C）を担当し、3 系統で推論時条件付けを俯瞰する。
- [[diffusion-sampling]]：resampling はサンプラー側の工夫。学習（モデル）とサンプリング（生成手続き）の分離という拡散モデルの基本性質を最大限に活かす。
- [[denoising-diffusion]]：順過程の閉形式 $q(x_t\mid x_0)$ と無条件逆過程を、そのまま条件付けの土台に使う。

## 参考文献（summaries）

- [[summaries/2022-repaint]] — RePaint: Inpainting using DDPM（置き換え＋resampling による学習不要 inpainting）
- [[summaries/2021-adm]] — Diffusion Models Beat GANs（RePaint が prior に使う事前学習モデル）
