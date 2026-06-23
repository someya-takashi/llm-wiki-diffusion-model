# Index — Diffusion Model LLM Wiki

全ページのカタログ。1 行 = 1 ページ（`- [[<slug>]] — <一行の説明>`）。ingest / query で新規ページを作るたびに更新する。

## Overview

- [[overview]] — 拡散モデル全体の総括

## Summaries

### paper
- [[summaries/2020-ddpm]] — Denoising Diffusion Probabilistic Models（DDPM, Ho ら 2020・NeurIPS）。拡散モデルで初めて GAN 級の画像品質を達成した起点論文
- [[summaries/2022-latent-diffusion]] — High-Resolution Image Synthesis with Latent Diffusion Models（LDM, Rombach ら 2022・CVPR）。Stable Diffusion の基盤。潜在空間拡散＋cross-attention 条件付け
- [[summaries/2021-ddim]] — Denoising Diffusion Implicit Models（DDIM, Song ら 2021・ICLR）。非マルコフ過程による決定論サンプリングで 10〜50× 高速化
- [[summaries/2022-lora]] — LoRA: Low-Rank Adaptation of Large Language Models（Hu ら 2022・ICLR）。凍結重みに低ランク更新 ΔW=BA を足す PEFT。拡散の軽量 personalization の基礎
- [[summaries/2021-adm]] — Diffusion Models Beat GANs on Image Synthesis（ADM, Dhariwal & Nichol 2021・NeurIPS）。改良 U-Net＋classifier guidance で拡散が初めて GAN を超えた
- [[summaries/2022-classifier-free-guidance]] — Classifier-Free Diffusion Guidance（CFG, Ho & Salimans 2022）。分類器なしで条件忠実度↔多様性を制御する標準技術
- [[summaries/2022-repaint]] — RePaint: Inpainting using DDPM（Lugmayr ら 2022・CVPR）。凍結した無条件 DDPM＋置き換え条件付け＋resampling で学習不要の任意マスク inpainting
- [[summaries/2021-score-sde]] — Score-Based Generative Modeling through SDEs（Score-SDE, Song ら 2021・ICLR Outstanding Paper）。SMLD/DDPM を連続時間 SDE に統一、確率フロー ODE・PC サンプラー・可制御生成
- [[summaries/2023-flow-matching]] — Flow Matching for Generative Modeling（FM, Lipman ら 2023・ICLR）。CNF をシミュレーション不要で学習する新パラダイム。拡散を内包し OT パスで高速・安定
- [[summaries/2023-controlnet]] — Adding Conditional Control to Text-to-Image Diffusion Models（ControlNet, Zhang ら 2023・ICCV Best Paper）。zero convolution で Stable Diffusion に空間条件制御を後付け
- [[summaries/2023-dreambooth]] — DreamBooth（Ruiz ら 2023・CVPR）。少数画像で T2I 拡散モデルを fine-tune し被写体を一意識別子に紐づける personalization（rare-token id＋prior preservation loss）
- [[summaries/2023-anydoor]] — AnyDoor（Chen ら 2023・ICCV）。参照物体をシーンの指定位置に zero-shot で合成する object teleportation（DINO-V2 ID＋高周波マップ detail）
- [[summaries/2023-dit]] — Scalable Diffusion Models with Transformers（DiT, Peebles & Xie 2023・ICCV）。U-Net を Transformer に置換し Gflops スケーリングで ImageNet SOTA（FID 2.27）
- [[summaries/2024-lora-composer]] — LoRA-Composer（Yang ら 2024）。訓練不要で複数の単一概念 LoRA を 1 枚に合成（cross-attn 注入＋self-attn 分離＋潜在再初期化）
- [[summaries/2024-multi-lora-composition]] — Multi-LoRA Composition（Zhong ら 2024・ICML）。decoding-centric な LoRA Switch / LoRA Composite と ComposLoRA・GPT-4V 評価
- [[summaries/2023-mix-of-show]] — Mix-of-Show（Gu ら 2023・NeurIPS）。分散型多概念カスタマイズ。ED-LoRA＋gradient fusion で複数 LoRA を identity 損失なく融合、regionally controllable sampling
- [[summaries/2024-ziplora]] — ZipLoRA（Shah ら 2023・ECCV 2024）。被写体 LoRA × 画風 LoRA を列ごとの学習係数で干渉なくマージ（SDXL、hyperparameter-free）
- [[summaries/2023-sdxl]] — SDXL（Podell ら 2023・Stability AI）。Stable Diffusion を 3× UNet＋2 テキストエンコーダ＋micro-conditioning＋base+refiner で強化、オープンで SOTA 級
- [[summaries/2024-sd3]] — Stable Diffusion 3（Esser ら 2024・ICML）。rectified flow＋改良ノイズサンプラー＋MM-DiT を 8B までスケール、SDXL・DALL-E 3 を上回る
- [[summaries/2024-stochastic-interpolants]] — Stochastic Interpolants（Albergo ら 2023）。flows（ODE）と diffusions（SDE）を有限時間で統一する理論枠組み

## Translations

- [[translations/2020-ddpm]] — DDPM 全文翻訳（本文＋Appendix A–D）
- [[translations/2022-latent-diffusion]] — LDM 全文翻訳（本文＋Appendix A–H）
- [[translations/2021-ddim]] — DDIM 全文翻訳（本文＋Appendix A–D）
- [[translations/2022-lora]] — LoRA 全文翻訳（本文§1–8＋Appendix A–H）
- [[translations/2024-lora-composer]] — LoRA-Composer 全文翻訳（本文§1–5＋Appendix 0.A–0.D）
- [[translations/2024-multi-lora-composition]] — Multi-LoRA Composition 全文翻訳（本文§1–5＋Appendix A）
- [[translations/2021-adm]] — ADM 全文翻訳（本文§1–8＋Appendix A–M）
- [[translations/2022-classifier-free-guidance]] — CFG 全文翻訳（本文＋Appendix A）
- [[translations/2022-repaint]] — RePaint 全文翻訳（本文§1–8＋Appendix A–I）
- [[translations/2023-dreambooth]] — DreamBooth 全文翻訳（本文§1–6＋Supplementary Material）
- [[translations/2023-anydoor]] — AnyDoor 全文翻訳（本文§1–5、appendix なし）
- [[translations/2023-dit]] — DiT 全文翻訳（本文§1–6＋Appendix A–D）
- [[translations/2021-score-sde]] — Score-SDE 全文翻訳（本文＋Appendix A–I）
- [[translations/2023-flow-matching]] — Flow Matching 全文翻訳（本文＋Appendix A–F）
- [[translations/2023-controlnet]] — ControlNet 全文翻訳（本文§1–5、appendix なし）
- [[translations/2023-mix-of-show]] — Mix-of-Show 全文翻訳（本文§1–5＋Appendix §6、HTML 表 8 個 markdown 化）
- [[translations/2024-ziplora]] — ZipLoRA 全文翻訳（本文§1–5、appendix なし、HTML 表 1 個 markdown 化）
- [[translations/2023-sdxl]] — SDXL 全文翻訳（本文§1–3＋Appendix B–J、Alg.1/App J コードはコードブロック保持）
- [[translations/2024-sd3]] — SD3 全文翻訳（本文§1–6＋Appendix A–E、HTML 表 markdown 化、MM-DiT 図は SVG）
- [[translations/2024-stochastic-interpolants]] — Stochastic Interpolants 全文翻訳（本文§1–8＋Appendix A–C、全証明逐次）

## Concepts

- [[denoising-diffusion]] — ノイズ除去拡散モデルの定式化と DDPM（順過程・逆過程・ε予測・$L_\text{simple}$）
- [[diffusion-model-architecture]] — 拡散モデルのバックボーン（改良 U-Net＝ADM/AdaGN → Transformer 化＝DiT/adaLN-Zero）
- [[score-based-generative-models]] — スコアベース生成モデルと SDE 統一枠組み（denoising score matching＋Langevin、VE/VP/sub-VP SDE）
- [[probability-flow-ode]] — 確率フロー ODE / 連続時間（SDE/ODE）定式化（厳密尤度・可逆 encode、DDIM の ODE 視点）
- [[diffusion-sampling]] — 拡散モデルのサンプリング／ソルバー（DDIM・predictor-corrector・逆拡散・ODE サンプラー）
- [[latent-diffusion]] — 潜在拡散と LDM/Stable Diffusion（知覚的圧縮＋潜在拡散＋cross-attention 条件付け）
- [[controllable-generation]] — 可制御生成（スコア操作の逆問題＋ControlNet のアダプタ型空間条件制御）
- [[flow-matching]] — フローマッチング（CNF をシミュレーション不要で学習、CFM・ガウス条件付きパス・OT パス・rectified flow）
- [[stochastic-interpolants]] — 確率的補間（flows と diffusions を有限時間で統一する枠組み、ODE/SDE を同じ密度の異なる実現に）
- [[classifier-guidance]] — 分類器ガイダンス（ADM が導入、分類器勾配でサンプリングを誘導し忠実度↔多様性を制御。CFG の先行手法）
- [[classifier-free-guidance]] — 分類器なしガイダンス（条件付き／無条件スコアの線形結合で条件忠実度↔多様性を制御）
- [[text-to-image-generation]] — テキストからの画像生成（cross-attention 条件付け、classifier-free guidance、LAION/MS-COCO）
- [[image-inpainting]] — 画像 inpainting（欠損補完・物体除去。LDM の学習型 ↔ RePaint の学習不要型）
- [[training-free-conditioning]] — 学習不要・推論時の条件付け（凍結した無条件 prior＋置き換え／resampling、RePaint が代表）
- [[subject-driven-generation]] — 被写体駆動生成 / personalization（少数画像で fine-tune、DreamBooth・Textual Inversion）
- [[low-rank-adaptation]] — 低ランク適応（LoRA, ΔW=BA）。拡散の軽量 personalization の標準
- [[multi-concept-customization]] — 複数の LoRA／概念を 1 枚に合成（LoRA-Composer・Multi-LoRA Composition）
- [[lora-merging]] — 複数 LoRA の重みマージ／融合（Mix-of-Show の gradient fusion・ZipLoRA の学習係数マージ）
- [[image-composition]] — 画像コンポジション / object teleportation（参照物体をシーンの指定位置に zero-shot 合成、AnyDoor）
- [[super-resolution]] — 超解像（LDM-SR / LDM-BSR、SR3 比較）

略称リダイレクト：
- DDPM → [[denoising-diffusion]]
- DDIM → [[diffusion-sampling]]
- ADM / ADM-G / ADM-U → [[diffusion-model-architecture]] ・ [[classifier-guidance]]
- AdaGN（Adaptive Group Normalization） → [[diffusion-model-architecture]]
- DiT / Diffusion Transformer / adaLN / adaLN-Zero / patchify → [[diffusion-model-architecture]]
- LoRA / 低ランク適応 / PEFT / ΔW=BA → [[low-rank-adaptation]]
- LoRA Switch / LoRA Composite / ComposLoRA / multi-concept / 複数概念合成 → [[multi-concept-customization]]
- LoRA-Composer / concept vanishing / concept confusion → [[multi-concept-customization]]
- Mix-of-Show / gradient fusion / weight fusion / LoRA Merge / regionally controllable sampling / region-aware cross-attention → [[lora-merging]]
- ZipLoRA / merger coefficient / content LoRA / style LoRA / signal interference / LoRA fusion → [[lora-merging]]
- ED-LoRA / embedding-decomposed LoRA → [[lora-merging]] ・ [[low-rank-adaptation]]
- SDXL / Stable Diffusion XL → [[latent-diffusion]]（UNet スケーリングは [[diffusion-model-architecture]] 併記）
- micro-conditioning / size conditioning / crop conditioning / multi-aspect training / refinement model / SDXL-VAE / pooled text embedding → [[latent-diffusion]]
- SDEdit / offset-noise → [[latent-diffusion]]
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
- rectified flow / RF / reflow → [[flow-matching]]
- SD3 / Stable Diffusion 3 / MM-DiT / Multimodal DiT / QK-normalization → [[diffusion-model-architecture]] ・ [[latent-diffusion]]
- logit-normal sampler / mode sampler / CosMap → [[flow-matching]]
- stochastic interpolant / 確率的補間 / one-sided interpolant / mirror interpolant / denoiser η_z → [[stochastic-interpolants]]
- Schrödinger bridge / シュレディンガー橋 / stochastic localization → [[stochastic-interpolants]]
- ControlNet / zero convolution / 空間条件制御 → [[controllable-generation]]
- RePaint → [[image-inpainting]] ・ [[training-free-conditioning]]
- resampling / jump back-and-forth → [[diffusion-sampling]]
- training-free / 推論時条件付け / 置き換えベース → [[training-free-conditioning]]
- DreamBooth / Textual Inversion / personalization / 個人化 → [[subject-driven-generation]]
- PPL / prior preservation loss / language drift → [[subject-driven-generation]]
- Imagen → [[text-to-image-generation]]（cascaded SR は [[super-resolution]]）
- AnyDoor / object teleportation / 物体合成 → [[image-composition]]
- Paint-by-Example / ObjectStitch → [[image-composition]]
- DINO-V2 / HF-map / 高周波マップ → [[image-composition]]

## Questions

<!-- query で得た比較表・分析等 -->
