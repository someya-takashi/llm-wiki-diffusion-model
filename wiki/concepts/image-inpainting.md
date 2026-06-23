---
type: concept
aliases: [Inpainting, Image Inpainting, 画像補完, 欠損補完]
tags: [image-inpainting, latent-diffusion, generative-models, conditional-generation]
related:
  - "[[latent-diffusion]]"
  - "[[denoising-diffusion]]"
summaries:
  - "[[summaries/2022-latent-diffusion]]"
updated: 2026-06-23
---

# Image Inpainting（画像 inpainting / 欠損補完）

**Image Inpainting（画像 inpainting）** とは、画像のマスクされた（欠けた・隠した）領域を、周囲と自然につながるもっともらしい内容で埋めるタスクである。用途は、破損・欠損部分の修復、写っている不要な物体の除去（object removal）、画像編集など。「与えられた周囲の文脈に条件付けて、欠損領域を生成する」という意味で、条件付き画像生成の一種である。

本ページは inpainting という概念の俯瞰と、拡散ベースの代表手法を解説する。

## 技術的な要点

inpainting では、入力として「マスクされた画像」と「マスク（どこを埋めるかを示す二値画像）」が与えられ、モデルはマスク領域の内容を生成する。鍵となる性質：

- **条件付け**：マスク画像とマスクを生成モデルに与える。拡散モデルでは、これらを潜在表現に**連結（concat）**して U-Net に入力するのが簡便で効果的（[[latent-diffusion]]）。
- **多様性**：GAN ベースや回帰ベースの手法は単一の（平均的になりがちな）解を出すのに対し、拡散モデルのような生成的アプローチは**同じ入力に対して複数の多様な補完候補**を生成できる。これが拡散ベース inpainting の強み。
- **評価**：補完結果の品質を **FID** で、元画像との知覚的近さを **LPIPS（Learned Perceptual Image Patch Similarity）** で測る。ベンチマークは **Places** データセットが代表的。

## 代表手法

### Latent Diffusion（LDM-inpainting, Rombach ら 2022）

[[latent-diffusion]] は、マスク画像を潜在に連結する形で inpainting に適用し、Places ベンチマークで当時の**最先端（SOTA）FID** を達成した。特化アーキテクチャの LaMa（Fast Fourier Convolution ベース）を FID で上回り、ユーザー調査でも好まれた。LaMa が単一の平均的な解を返しがちなのに対し、LDM は多様な補完を生成できる点が評価された（[[summaries/2022-latent-diffusion]] 表7、図11・21）。注意なし VQ 正則化第一段階で大きなモデルを学習し $512^2$ でファインチューニングした構成（big, w/o attn, w/ ft）が最良スコア。

<figure>

![](../../raw/assets/2022-latent-diffusion/img_object_removal_input_000007.jpg)

<figcaption>図11（再掲, [[summaries/2022-latent-diffusion]] より）: LDM の big・ft あり inpainting モデルによる物体除去（object removal）の定性結果。</figcaption>
</figure>

## 既存知識との接続

- [[latent-diffusion]]：マスク画像を潜在へ連結する条件付けで inpainting に適用し SOTA を達成した代表手法。
- [[denoising-diffusion]]：inpainting 拡散モデルの生成エンジンとなる拡散の基礎。
- [[super-resolution]]：同じく「空間的に整列した条件を連結する」タイプの密な条件付きタスクで、LDM では共通の枠組みで扱われる。

## 参考文献（summaries）

- [[summaries/2022-latent-diffusion]] — Latent Diffusion Models（Places で inpainting SOTA を達成）
