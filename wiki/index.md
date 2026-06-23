# Index — Diffusion Model LLM Wiki

全ページのカタログ。1 行 = 1 ページ（`- [[<slug>]] — <一行の説明>`）。ingest / query で新規ページを作るたびに更新する。

## Overview

- [[overview]] — 拡散モデル全体の総括

## Summaries

### paper
- [[summaries/2020-ddpm]] — Denoising Diffusion Probabilistic Models（DDPM, Ho ら 2020・NeurIPS）。拡散モデルで初めて GAN 級の画像品質を達成した起点論文
- [[summaries/2022-latent-diffusion]] — High-Resolution Image Synthesis with Latent Diffusion Models（LDM, Rombach ら 2022・CVPR）。Stable Diffusion の基盤。潜在空間拡散＋cross-attention 条件付け
- [[summaries/2021-ddim]] — Denoising Diffusion Implicit Models（DDIM, Song ら 2021・ICLR）。非マルコフ過程による決定論サンプリングで 10〜50× 高速化
- [[summaries/2021-adm]] — Diffusion Models Beat GANs on Image Synthesis（ADM, Dhariwal & Nichol 2021・NeurIPS）。改良 U-Net＋classifier guidance で拡散が初めて GAN を超えた
- [[summaries/2022-classifier-free-guidance]] — Classifier-Free Diffusion Guidance（CFG, Ho & Salimans 2022）。分類器なしで条件忠実度↔多様性を制御する標準技術
- [[summaries/2022-repaint]] — RePaint: Inpainting using DDPM（Lugmayr ら 2022・CVPR）。凍結した無条件 DDPM＋置き換え条件付け＋resampling で学習不要の任意マスク inpainting
- [[summaries/2021-score-sde]] — Score-Based Generative Modeling through SDEs（Score-SDE, Song ら 2021・ICLR Outstanding Paper）。SMLD/DDPM を連続時間 SDE に統一、確率フロー ODE・PC サンプラー・可制御生成
- [[summaries/2023-flow-matching]] — Flow Matching for Generative Modeling（FM, Lipman ら 2023・ICLR）。CNF をシミュレーション不要で学習する新パラダイム。拡散を内包し OT パスで高速・安定
- [[summaries/2023-controlnet]] — Adding Conditional Control to Text-to-Image Diffusion Models（ControlNet, Zhang ら 2023・ICCV Best Paper）。zero convolution で Stable Diffusion に空間条件制御を後付け
- [[summaries/2023-dreambooth]] — DreamBooth（Ruiz ら 2023・CVPR）。少数画像で T2I 拡散モデルを fine-tune し被写体を一意識別子に紐づける personalization（rare-token id＋prior preservation loss）

## Translations

- [[translations/2020-ddpm]] — DDPM 全文翻訳（本文＋Appendix A–D）
- [[translations/2022-latent-diffusion]] — LDM 全文翻訳（本文＋Appendix A–H）
- [[translations/2021-ddim]] — DDIM 全文翻訳（本文＋Appendix A–D）
- [[translations/2021-adm]] — ADM 全文翻訳（本文§1–8＋Appendix A–M）
- [[translations/2022-classifier-free-guidance]] — CFG 全文翻訳（本文＋Appendix A）
- [[translations/2022-repaint]] — RePaint 全文翻訳（本文§1–8＋Appendix A–I）
- [[translations/2023-dreambooth]] — DreamBooth 全文翻訳（本文§1–6＋Supplementary Material）
- [[translations/2021-score-sde]] — Score-SDE 全文翻訳（本文＋Appendix A–I）
- [[translations/2023-flow-matching]] — Flow Matching 全文翻訳（本文＋Appendix A–F）
- [[translations/2023-controlnet]] — ControlNet 全文翻訳（本文§1–5、appendix なし）

## Concepts

- [[denoising-diffusion]] — ノイズ除去拡散モデルの定式化と DDPM（順過程・逆過程・ε予測・$L_\text{simple}$）
- [[diffusion-model-architecture]] — 拡散モデルのアーキテクチャ（ノイズ予測 U-Net の系譜、ADM の改良 U-Net・AdaGN）
- [[score-based-generative-models]] — スコアベース生成モデルと SDE 統一枠組み（denoising score matching＋Langevin、VE/VP/sub-VP SDE）
- [[probability-flow-ode]] — 確率フロー ODE / 連続時間（SDE/ODE）定式化（厳密尤度・可逆 encode、DDIM の ODE 視点）
- [[diffusion-sampling]] — 拡散モデルのサンプリング／ソルバー（DDIM・predictor-corrector・逆拡散・ODE サンプラー）
- [[latent-diffusion]] — 潜在拡散と LDM/Stable Diffusion（知覚的圧縮＋潜在拡散＋cross-attention 条件付け）
- [[controllable-generation]] — 可制御生成（スコア操作の逆問題＋ControlNet のアダプタ型空間条件制御）
- [[flow-matching]] — フローマッチング（CNF をシミュレーション不要で学習、CFM・ガウス条件付きパス・OT パス）
- [[classifier-guidance]] — 分類器ガイダンス（ADM が導入、分類器勾配でサンプリングを誘導し忠実度↔多様性を制御。CFG の先行手法）
- [[classifier-free-guidance]] — 分類器なしガイダンス（条件付き／無条件スコアの線形結合で条件忠実度↔多様性を制御）
- [[text-to-image-generation]] — テキストからの画像生成（cross-attention 条件付け、classifier-free guidance、LAION/MS-COCO）
- [[image-inpainting]] — 画像 inpainting（欠損補完・物体除去。LDM の学習型 ↔ RePaint の学習不要型）
- [[training-free-conditioning]] — 学習不要・推論時の条件付け（凍結した無条件 prior＋置き換え／resampling、RePaint が代表）
- [[subject-driven-generation]] — 被写体駆動生成 / personalization（少数画像で fine-tune、DreamBooth・Textual Inversion）
- [[super-resolution]] — 超解像（LDM-SR / LDM-BSR、SR3 比較）

略称リダイレクト：
- DDPM → [[denoising-diffusion]]
- DDIM → [[diffusion-sampling]]
- ADM / ADM-G / ADM-U → [[diffusion-model-architecture]] ・ [[classifier-guidance]]
- AdaGN（Adaptive Group Normalization） → [[diffusion-model-architecture]]
- classifier guidance / 分類器ガイダンス → [[classifier-guidance]]
- IDDPM → [[diffusion-model-architecture]]
- CFG → [[classifier-free-guidance]]
- score matching / NCSN / SMLD → [[score-based-generative-models]]
- VE/VP/sub-VP SDE / Score-SDE → [[score-based-generative-models]]
- LDM / Stable Diffusion → [[latent-diffusion]]
- T2I → [[text-to-image-generation]] ・ SR → [[super-resolution]]
- SDE / ODE / probability flow → [[probability-flow-ode]]
- PC sampler / predictor-corrector → [[diffusion-sampling]]
- FM / CFM / CNF / Optimal Transport path → [[flow-matching]]
- ControlNet / zero convolution / 空間条件制御 → [[controllable-generation]]
- RePaint → [[image-inpainting]] ・ [[training-free-conditioning]]
- resampling / jump back-and-forth → [[diffusion-sampling]]
- training-free / 推論時条件付け / 置き換えベース → [[training-free-conditioning]]
- DreamBooth / Textual Inversion / personalization / 個人化 → [[subject-driven-generation]]
- PPL / prior preservation loss / language drift → [[subject-driven-generation]]
- Imagen → [[text-to-image-generation]]（cascaded SR は [[super-resolution]]）

## Questions

<!-- query で得た比較表・分析等 -->
