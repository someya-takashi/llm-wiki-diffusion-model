---
type: summary
source_path: raw/papers/RePaint_ Inpainting using Denoising Diffusion Probabilistic Models.md
source_kind: paper
title: "RePaint: Inpainting using Denoising Diffusion Probabilistic Models"
authors: [Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, Luc Van Gool]
year: 2022
venue: CVPR 2022
ingested: 2026-06-24
tags: [image-inpainting, training-free-conditioning, denoising-diffusion, diffusion-sampling, generative-models]
translation: "[[translations/2022-repaint]]"
---

# RePaint: Inpainting using DDPM

> 原典: [[translations/2022-repaint]] ・ `raw/papers/RePaint_ Inpainting using Denoising Diffusion Probabilistic Models.md`
> 著者・年・会議: Lugmayr, Danelljan, Romero, Yu, Timofte, Van Gool（ETH Zürich）・2022・CVPR 2022（arXiv 2201.09865）

## 一言まとめ

**事前学習済みの無条件 DDPM（ノイズ除去拡散確率モデル）をそのまま prior として使い、再学習せずに任意マスクの inpainting（欠損補完）を行う**手法。生成過程を一切学習し直さず、(1) 逆拡散の各ステップで既知領域を入力画像から差し込み、(2) 拡散時刻を**行き来する resampling** で既知/生成領域を調和させる、という 2 つの推論時テクニックだけで実現する。

## 背景と問題意識

**画像 inpainting（[[image-inpainting]]）** は、マスク（欠けた領域を示す二値画像）の内側を、周囲と自然につながるもっともらしい内容で埋めるタスク。従来の最先端は GAN や自己回帰モデルだが、2 つの弱点があった。

1. **マスク分布への過学習**：特定のマスク形状（細いブラシ・四角など）で学習するため、学習時と違うマスク（極端に大きい欠損など）にうまく汎化できない。
2. **意味の欠如**：画素ごとの損失や perceptual 損失で学習すると、欠損部に「意味のある内容」ではなく周囲テクスチャの単純な延長を生みがち。

一方、**DDPM（[[denoising-diffusion]]）** は多様で高品質な画像を生成でき、Dhariwal & Nichol の ADM（[[summaries/2021-adm]]）が示したように GAN を超える品質を持つ。RePaint の発想は「**inpainting 専用モデルを学習せず、強力な無条件 DDPM を画像の prior として借り、条件付けは推論時だけ行う**」。これにより、どんなマスクにも汎化でき、DDPM の強い生成能力（意味的なハルシネーション）をそのまま使える。RePaint は ADM が公開した guided-diffusion の事前学習モデルを利用する。

## 提案手法 / 主張

### (1) 既知領域への条件付け（§4.1）

学習済み無条件 DDPM を**固定**したまま、逆拡散の各ステップで画像をマスク $m$ で 2 つに分けて合成する：

$$
x_{t-1}=m\odot x_{t-1}^{\text{known}}+(1-m)\odot x_{t-1}^{\text{unknown}}
$$

- **既知領域** $x_{t-1}^{\text{known}}\sim\mathcal N(\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)\mathbf I)$：真の入力 $x_0$ を「時刻 $t$ 相当のノイズ」まで前方拡散して差し込む（順過程の閉形式を利用）。
- **未知領域** $x_{t-1}^{\text{unknown}}\sim\mathcal N(\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))$：DDPM が普通に 1 ステップ逆拡散して生成する。

逆ステップが $x_t$ にしか依存しないので、既知領域を「正しいノイズレベルの本物」で毎ステップ上書きしても分布の性質が保たれる、という点がミソ。学習は一切不要。

### (2) Resampling（リサンプリング, §4.2）——本手法の名前の由来

単純な上書きだけだと、既知領域のノイズ化が「生成中の内容」を考慮しないため、境界が**不調和（disharmonious）**になる。各ステップで分散スケジュールにより訂正できる量も小さく、間に合わない。そこで RePaint は、いったん得た $x_{t-1}$ を**前方拡散で $x_t$ に戻し**（$x_t\sim\mathcal N(\sqrt{1-\beta_t}x_{t-1},\beta_t\mathbf I)$）、再度デノイズし直す操作を繰り返す。戻してもノイズに生成情報の一部が残るので、DDPM の「データ分布に整合させる」性質が既知領域と生成領域を**調和**させる。

- **jump length $j$**：何ステップ分まとめて戻すか。$j=1$ だとぼやけやすく、$j=10$ が最良。
- **resample 回数 $r$**：何回戻して直すか。多いほど整合性が上がり、$r\approx10$ で飽和（図2）。
- 拡散時刻 $t$ は、全体として減少しつつ局所的に何度も上方ジャンプする**鋸歯状スケジュール**になる（図9）。
- **「slowing down（拡散を遅くする）」とは本質的に別物**：slowing down は 1 ステップの変化を小さくするだけで調和問題を解かない。同じ計算予算なら resampling が圧勝（表2）。最終設定 T=250, j=10, r=10。

<figure>

![](../../raw/assets/2022-repaint/x1.png)

<figcaption>図1（再掲, [[translations/2022-repaint]] より）: RePaint の概観。各逆ステップで、既知領域（上の経路）は入力をノイズ化して差し込み、未知領域（下の経路）は DDPM 出力を使い、マスクで合成する。</figcaption>
</figure>

## 実験結果と知見

- **データ/マスク**：CelebA-HQ・ImageNet（付録で Places2）。6 種のマスク Wide / Narrow / Super-Resolve 2× / Alternating Lines / Half / Expand。評価は **LPIPS（Learned Perceptual Image Patch Similarity, 知覚的距離・低いほど良い）** と、人手の**ユーザー調査（どちらが現実的か）**。
- **比較対象**：自己回帰（DSI, ICT）・GAN（DeepFillv2, AOT, LaMa）。RePaint は 6 マスク中 5 つ以上でユーザー票が最多。特に thin マスク（SR2×・Alternating Lines）で 73〜99% の票を獲得し圧倒。
- **LPIPS の注意点**：extreme マスク（Half・Expand）では LaMa が LPIPS で勝つことがあるが、これは RePaint が GT と違う「妥当で多様な」補完を生むため。LPIPS は「GT への近さ」を測るので、多様生成にはむしろ不適という議論。
- **ablation**：resampling ≫ slowing down（同予算, 表2）、$j=10$ が最良（表3）。
- **vs SDEdit**：同じ「拡散を一部戻して直す」系の SDEdit より全マスクでほぼ優位（表4, SR で LPIPS 53% 超減）。
- **多様性**：確率的サンプリングゆえ 1 入力から多様な補完を生成可能（Diversity Score でも細マスクで圧勝, 表6）。クラス条件付き生成にも対応（図5）。

## 限界・批判的視点

- **遅い**：画像ごとに DDPM を T×(j×r) 回も評価するため、GAN/自己回帰より桁違いに遅くリアルタイム困難。
- **評価の難しさ**：extreme マスクでは GT と大きく違う妥当解を生むため LPIPS が不適。FID は 1000 枚以上要し計算コストが現実的でない。
- **data bias**：ImageNet モデルは犬を過剰に inpaint しやすい（学習データ偏り, 図12）。失敗時は意味的文脈を取り違え非整合な物体を混ぜる。
- **位置づけ**：RePaint は「置き換え＋resampling」という推論時条件付けの代表。後続の拡散 inpainting（学習型の Palette や、Stable Diffusion の inpainting fine-tune）とは別系統で、無条件 prior をそのまま使う柔軟さが持ち味。

## 用語と略称

- **DDPM** = Denoising Diffusion Probabilistic Models（ノイズ除去拡散確率モデル）。
- **inpainting / free-form inpainting** = 任意形状マスクの欠損補完。
- **resampling（jump back/forth）** = 出力を前方拡散で戻して再デノイズする調和操作。**jump length $j$** = 一度に戻す時刻幅、**resample 回数 $r$**。
- **LPIPS** = Learned Perceptual Image Patch Similarity（学習された知覚的距離。低いほど GT に近い）。**FID** = Fréchet Inception Distance。**Diversity Score (DS)** = 妥当多様体内での多様性指標。
- **mask-agnostic** = マスク種別に依存しない（学習不要ゆえ任意マスクに汎化）。
- **SDEdit / ILVR** = 比較・関連する推論時条件付け拡散手法（本 wiki 未取り込み）。

## 既存知識との接続

- [[training-free-conditioning]]：RePaint は「事前学習済み拡散 prior をサンプリング時だけ条件付ける」学習不要パラダイムの代表。本ページの核心がこの概念。
- [[image-inpainting]]：RePaint は inpainting のランドマーク。LDM-inpainting（マスク条件付きモデルを**学習**して concat）と対照的に、**凍結した無条件 DDPM＋推論時条件付け**でマスク非依存。
- [[denoising-diffusion]]：RePaint は DDPM の順過程の閉形式と逆過程をそのまま使う（新規学習なし）。
- [[diffusion-sampling]]：resampling は学習ではなくサンプリング時の戦略。slowing down（少ステップ化）との違いが要点。
- [[controllable-generation]]：スコア勾配で条件付ける classifier guidance / 逆問題（Score-SDE）とは別系統の「置き換え／射影ベース」条件付け。
- [[summaries/2021-adm]]：RePaint が prior に使う guided-diffusion 事前学習モデルの原典。

## 関連ページ

- [[concepts/training-free-conditioning]] — 学習不要・推論時の条件付け（RePaint が代表）
- [[concepts/image-inpainting]] — 画像補完（RePaint をランドマークとして追記）
- [[concepts/denoising-diffusion]] / [[concepts/diffusion-sampling]] / [[concepts/controllable-generation]]
- [[summaries/2021-adm]] — Diffusion Models Beat GANs（RePaint が使う事前学習モデル）
- [[summaries/2022-latent-diffusion]] — LDM（学習型 inpainting の対比対象）
