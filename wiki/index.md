# Index — Diffusion Model LLM Wiki

全ページのカタログ。1 行 = 1 ページ（`- [[<slug>]] — <一行の説明>`）。ingest / query で新規ページを作るたびに更新する。

## Overview

- [[overview]] — 拡散モデル全体の総括

## Summaries

### paper
- [[summaries/2020-ddpm]] — Denoising Diffusion Probabilistic Models（DDPM, Ho ら 2020・NeurIPS）。拡散モデルで初めて GAN 級の画像品質を達成した起点論文
- [[summaries/2022-latent-diffusion]] — High-Resolution Image Synthesis with Latent Diffusion Models（LDM, Rombach ら 2022・CVPR）。Stable Diffusion の基盤。潜在空間拡散＋cross-attention 条件付け
- [[summaries/2021-ddim]] — Denoising Diffusion Implicit Models（DDIM, Song ら 2021・ICLR）。非マルコフ過程による決定論サンプリングで 10〜50× 高速化
- [[summaries/2022-classifier-free-guidance]] — Classifier-Free Diffusion Guidance（CFG, Ho & Salimans 2022）。分類器なしで条件忠実度↔多様性を制御する標準技術

## Translations

- [[translations/2020-ddpm]] — DDPM 全文翻訳（本文＋Appendix A–D）
- [[translations/2022-latent-diffusion]] — LDM 全文翻訳（本文＋Appendix A–H）
- [[translations/2021-ddim]] — DDIM 全文翻訳（本文＋Appendix A–D）
- [[translations/2022-classifier-free-guidance]] — CFG 全文翻訳（本文＋Appendix A）

## Concepts

- [[denoising-diffusion]] — ノイズ除去拡散モデルの定式化と DDPM（順過程・逆過程・ε予測・$L_\text{simple}$）
- [[score-based-generative-models]] — スコアベース生成モデル（denoising score matching＋Langevin 動力学、拡散モデルとの等価性）
- [[latent-diffusion]] — 潜在拡散と LDM/Stable Diffusion（知覚的圧縮＋潜在拡散＋cross-attention 条件付け）
- [[diffusion-sampling]] — 拡散モデルのサンプリング／ソルバー（DDIM による決定論化・少ステップ高速化）
- [[probability-flow-ode]] — 確率フロー ODE / 連続時間（SDE/ODE）定式化（DDIM の ODE 視点）
- [[classifier-free-guidance]] — 分類器なしガイダンス（条件付き／無条件スコアの線形結合で条件忠実度↔多様性を制御）
- [[classifier-guidance]] — 分類器ガイダンス（CFG の先行手法、分類器勾配を加える）
- [[text-to-image-generation]] — テキストからの画像生成（cross-attention 条件付け、classifier-free guidance、LAION/MS-COCO）
- [[image-inpainting]] — 画像 inpainting（欠損補完・物体除去、LDM が Places で SOTA）
- [[super-resolution]] — 超解像（LDM-SR / LDM-BSR、SR3 比較）

略称リダイレクト：
- DDPM → [[denoising-diffusion]]
- DDIM → [[diffusion-sampling]]
- CFG → [[classifier-free-guidance]]
- score matching / NCSN → [[score-based-generative-models]]
- LDM / Stable Diffusion → [[latent-diffusion]]
- T2I → [[text-to-image-generation]] ・ SR → [[super-resolution]]
- SDE / ODE / probability flow → [[probability-flow-ode]]

## Questions

<!-- query で得た比較表・分析等 -->
