# Overview — Diffusion Model（拡散モデル）

> このページは拡散モデル分野全体の総括。原典を ingest するたびに、新潮流・パラダイムシフトを随時反映する（現状は初期スケルトン）。

## このwikiが扱う範囲

**Diffusion Model（拡散モデル）** は、データに徐々にノイズを加える順過程（forward process）を学習し、それを逆向きにたどってノイズからデータを生成する生成モデルの一群。本 wiki は画像生成を中心に、以下を幅広く扱う。

- **生成タスク**：text-to-image 生成、画像編集・inpainting（欠損補完）・super-resolution（超解像）、video / audio 生成への拡張
- **定式化**：DDPM（Denoising Diffusion Probabilistic Models）、score-based generative models（スコアベース生成モデル）、SDE/ODE による連続時間定式化、flow matching
- **サンプリング**：DDIM 等の高速サンプラー／ソルバー
- **条件付け**：classifier-free guidance など
- **効率化**：latent diffusion（潜在空間での拡散）

## 主要な系譜（今後追記）

拡散モデルの系譜の起点が **DDPM（Denoising Diffusion Probabilistic Models, Ho ら 2020）** である。それ以前から拡散確率モデル（Sohl-Dickstein ら 2015）やスコアベース生成モデル（NCSN, Song & Ermon）は存在したが、DDPM が「ノイズ予測の二乗誤差で学習する」シンプルな定式化により初めて GAN 級の画像品質を達成し、現在の生成 AI ブームに火をつけた（[[denoising-diffusion]]）。DDPM はまた、拡散モデルとスコアマッチング＋Langevin 動力学が等価であることを示し、両系統を統一した（[[score-based-generative-models]]）。

今後 ingest が進むにつれ、以下の流れを概念ページへのリンク付きで整理していく：

- **起点**：[[denoising-diffusion]]（DDPM 2020）／[[score-based-generative-models]]（スコア SDE への一般化）
- **高速化**：DDIM 等の少ステップサンプラー（[[diffusion-sampling]]）。学習済みモデルを再学習なしに 10〜50× 高速化。決定論サンプリングは確率フロー ODE（[[probability-flow-ode]]）の離散化として理解できる。
- **効率化・実用化**：[[latent-diffusion]]（潜在拡散＝Stable Diffusion, Rombach ら 2022）。オートエンコーダの潜在空間で拡散することで計算量を一桁削減し、cross-attention 条件付けで多タスク化。画像生成 AI を一般に普及させた転換点。
- **条件付け・応用**：cross-attention による [[text-to-image-generation]]（テキストからの画像生成）、[[image-inpainting]]（欠損補完）、[[super-resolution]]（超解像）。高品質化は classifier-free guidance（[[classifier-free-guidance]]）に依存。

## 関連ページ

- [[denoising-diffusion]] — 拡散モデルの定式化と DDPM（系譜の起点）
- [[latent-diffusion]] — 潜在空間での拡散と LDM/Stable Diffusion（効率化・実用化）
- [[diffusion-sampling]] — サンプリング／ソルバー（DDIM による高速化）
- [[probability-flow-ode]] — 連続時間（SDE/ODE）定式化と確率フロー ODE
- [[score-based-generative-models]] — スコアマッチング／Langevin 動力学との等価性
- [[classifier-free-guidance]] / [[classifier-guidance]] — 条件忠実度↔多様性を制御する guidance
- [[text-to-image-generation]] / [[image-inpainting]] / [[super-resolution]] — LDM が切り開いた主要応用タスク
- [[summaries/2020-ddpm]]・[[summaries/2022-latent-diffusion]]・[[summaries/2021-ddim]]・[[summaries/2022-classifier-free-guidance]] — 取り込み済み原典の要約
