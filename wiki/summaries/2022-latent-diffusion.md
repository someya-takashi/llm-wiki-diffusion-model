---
type: summary
source_path: raw/papers/High-Resolution Image Synthesis with Latent Diffusion Models.md
source_kind: paper
title: "High-Resolution Image Synthesis with Latent Diffusion Models"
authors: [Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer]
year: 2022
venue: CVPR 2022
ingested: 2026-06-23
tags: [latent-diffusion, text-to-image-generation, super-resolution, image-inpainting, generative-models, stable-diffusion]
translation: "[[translations/2022-latent-diffusion]]"
---

# High-Resolution Image Synthesis with Latent Diffusion Models（LDM / Stable Diffusion）

> 原典: [[translations/2022-latent-diffusion]] ・ `raw/papers/High-Resolution Image Synthesis with Latent Diffusion Models.md`（arXiv:2112.10752）
> 著者・年・会議: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer（LMU Munich & Runway ML）・2022・CVPR 2022
> コード: https://github.com/CompVis/latent-diffusion

## 一言まとめ

拡散モデルをピクセル空間ではなく**オートエンコーダの潜在空間（latent space）で動かす**ことで、品質をほぼ落とさずに学習・推論コストを一桁下げ、さらに **cross-attention（クロスアテンション）** による条件付けで text-to-image・inpainting・超解像・layout-to-image を 1 つの枠組みで実現した論文。**Stable Diffusion の基盤**となり、画像生成 AI を一般ユーザーが扱える計算量へと「民主化」した。

## 背景と問題意識

[[denoising-diffusion]]（DDPM）以降、拡散モデルは GAN を上回る画質を出せるようになったが、**ピクセル空間で直接拡散する**ため計算が極めて重い。最強クラスの拡散モデルの学習は数百 GPU 日（150〜1000 V100 日）、5 万枚の生成に A100 で約 5 日かかる。これは (1) 一部の潤沢な計算資源を持つ機関しか学習できず大きなカーボンフットプリントを残す、(2) 学習済みモデルの推論も逐次評価で高価、という 2 つの問題を生む。

著者らの着眼は「尤度ベースモデルの学習は**知覚的圧縮**（高周波の細部を捨てる段階）と**意味的圧縮**（データの意味・構成を学ぶ段階）に分けられ、ビットの大半は知覚できない細部に費やされている」点（図 2）。ならば細部の圧縮は別の安価なモデルに任せ、拡散モデルは意味的な部分だけを低次元空間で学習すればよい。

## 提案手法 / 主張

**2 段階アーキテクチャ**（図 3）：

1. **第一段階：知覚的圧縮オートエンコーダ** — エンコーダ $\mathcal{E}$ が画像を係数 $f$（=4 や 8）でダウンサンプルした潜在 $z=\mathcal{E}(x)$ に圧縮し、デコーダ $\mathcal{D}$ が復元する。知覚的損失＋パッチ単位の敵対的損失で学習し、ぼやけない忠実な再構成を得る。潜在空間の暴走を防ぐため **KL 正則化**（VAE 風の弱い KL ペナルティ）か **VQ 正則化**（ベクトル量子化）を軽くかける。この段階は一度学習すれば多数のタスクに再利用できる。
2. **第二段階：潜在拡散モデル（LDM）** — 圧縮された潜在空間で [[denoising-diffusion]] と同じノイズ予測目的 $L_{LDM}=\mathbb{E}_{\mathcal{E}(x),\epsilon,t}[\lVert\epsilon-\epsilon_\theta(z_t,t)\rVert_2^2]$ を時間条件付き U-Net で最適化する。ピクセル空間より遥かに低次元なので学習・サンプリングが軽い。

**cross-attention による汎用条件付け**：テキスト・意味マップ・バウンディングボックスなどの条件 $y$ を、ドメイン特化エンコーダ $\tau_\theta$（テキストなら Transformer）で中間表現に変換し、U-Net の中間層に **cross-attention**（クエリ＝U-Net 特徴、キー/値＝$\tau_\theta(y)$）で注入する。これにより 1 つの枠組みで多モダリティ条件付けが可能になった。空間的に整列した条件（低解像度画像・意味マップ）は連結（concat）で与え、畳み込み的に評価すると学習解像度を超える $1024^2$ 級まで生成できる。

圧縮率の選択（$f$）が肝で、**LDM-4 / LDM-8** が効率と品質の最適点（$f$ が小さすぎる＝LDM-1 はピクセル拡散で遅い、大きすぎる＝LDM-32 は情報損失で品質頭打ち、図 6・7）。

## 実験結果と知見

- **無条件生成**：CelebA-HQ で **FID 5.11**（当時 SOTA、GAN・尤度ベース双方を凌駕）、FFHQ 4.98、LSUN-Bedrooms 2.95 など（表 1）。
- **クラス条件付き ImageNet**：classifier-free guidance 併用の **LDM-4-G が FID 3.60**で、ADM（拡散 SOTA）をパラメータ・計算量を大幅削減して上回る（表 3）。
- **text-to-image**：LAION-400M で 14.5 億パラメータの LDM を学習、MS-COCO で評価。classifier-free guidance 併用の LDM-KL-8-G が GLIDE（6B）等に匹敵しつつ大幅に少パラメータ（表 2、図 5）。→ [[text-to-image-generation]]
- **inpainting**：Places で **SOTA**（LaMa を上回る FID、多様なサンプル生成、ユーザー調査でも好まれる）。→ [[image-inpainting]]
- **超解像**：ImageNet $4\times$ で SR3 を FID で上回る（表 5）。→ [[super-resolution]]
- **効率**：ピクセルベースに対し inpainting で $2.7\times$ 以上の高速化・FID $1.6\times$ 改善。全実験が単一 A100 で実行可能。

## 限界・批判的視点

- **逐次サンプリングは依然 GAN より遅い**。本論文は DDIM（[[diffusion-sampling]]）で 50〜250 ステップに削減しているが、一発生成の GAN には及ばない。
- **オートエンコーダの再構成がボトルネック**。$f=4$ でも画素レベルの厳密な精度が要るタスク（細密な超解像など）では第一段階の再構成限界が効く。
- **社会的影響**：ディープフェイク・学習データの露出・データバイアスの再生産。著者ら自身が「アクセスを民主化する一方で悪用も容易にする諸刃の剣」と明記。
- **classifier-free guidance への依存**：text2img・class-cond の高品質化は CFG（[[classifier-free-guidance]]）に依存。CFG 自体は Ho & Salimans の別論文の貢献。

## 関連ページ

- [[concepts/latent-diffusion]] — 潜在拡散の考え方と LDM/Stable Diffusion の詳細（本論文の中核）
- [[concepts/denoising-diffusion]] — LDM が潜在空間で用いる拡散の基礎（DDPM）
- [[concepts/text-to-image-generation]] — LDM を代表手法とするテキストからの画像生成
- [[concepts/image-inpainting]] — LDM が SOTA を達成した欠損補完
- [[concepts/super-resolution]] — LDM-SR による超解像
