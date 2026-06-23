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

- **起点**：[[denoising-diffusion]]（DDPM 2020）／[[score-based-generative-models]]（SMLD/NCSN, スコア＋Langevin）
- **理論的統一**：[[score-based-generative-models]] の **Score-SDE**（Song ら 2021）が SMLD=VE-SDE / DDPM=VP-SDE を連続時間 SDE に統一。決定論的な [[probability-flow-ode]]（厳密尤度・可逆 encode）、predictor-corrector サンプラー（[[diffusion-sampling]]）、条件付き逆時間 SDE による [[controllable-generation]] を導いた。
- **高速化**：DDIM 等の少ステップサンプラー（[[diffusion-sampling]]）。学習済みモデルを再学習なしに 10〜50× 高速化。決定論サンプリングは確率フロー ODE（[[probability-flow-ode]]）の離散化として理解できる。
- **GAN を超えた点**：[[denoising-diffusion]] の改良＝**ADM**（Dhariwal & Nichol 2021）。改良 U-Net（多解像度 attention・BigGAN resblock・AdaGN, [[diffusion-model-architecture]]）と [[classifier-guidance]]（分類器ガイダンス）を組み合わせ、ImageNet で拡散モデルが初めて GAN（BigGAN-deep）を画像品質で上回った。分類器勾配のスケール 1 個で「忠実度↔多様性」を制御できるノブを与えた点が鍵。
- **効率化・実用化**：[[latent-diffusion]]（潜在拡散＝Stable Diffusion, Rombach ら 2022）。オートエンコーダの潜在空間で拡散することで計算量を一桁削減し、cross-attention 条件付けで多タスク化。画像生成 AI を一般に普及させた転換点。
- **条件付け・応用**：[[controllable-generation]]（可制御生成）を起点に、guidance の系譜は [[classifier-guidance]]（分類器ガイダンス＝ADM 2021, 分類器の勾配を足す）→ [[classifier-free-guidance]]（CFG 2022, 分類器を排し条件付き／無条件スコアの差で同じノブを実現）と進んだ。これに cross-attention を加えて [[text-to-image-generation]]・[[image-inpainting]]・[[super-resolution]] が実現する。さらに **ControlNet**（Zhang ら 2023）が zero convolution で Stable Diffusion を凍結したままエッジ・深度・姿勢などの空間条件を後付けし、制御性を実用面で一変させた。
- **学習不要の推論時条件付け**：[[training-free-conditioning]]。事前学習済みの無条件拡散モデルを prior として固定し、**サンプリング過程だけ**で条件を課す系統。**RePaint**（Lugmayr ら 2022）は凍結 DDPM＋既知領域の置き換え＋resampling で、再学習なしに任意マスクの [[image-inpainting]] を実現した。条件タイプごとの再学習が要らず、強力な無条件 prior をそのまま使い回せるのが利点。
- **personalization（被写体駆動生成）**：[[subject-driven-generation]]。逆に、少数画像で T2I モデルを **fine-tune** して「特定個体」をモデルに埋め込む系統。**DreamBooth**（Ruiz ら 2023）は 3〜5 枚で Imagen / Stable Diffusion を fine-tune し、一意識別子と prior preservation loss で被写体を新文脈に再生成。比較対象の **Textual Inversion** は凍結モデルの token 埋め込みのみ学習。テキスト条件が「何を」を、personalization が「どの個体を」を制御する。
- **拡散の一般化／別系統**：[[flow-matching]]（Lipman ら 2023）。連続正規化フロー（CNF）をシミュレーション不要で学習し、拡散パスを特別な場合として内包しつつ、最適輸送（OT）パスで直線的・高速・安定な生成を実現。拡散の確率フロー ODE（[[probability-flow-ode]]）を「確率パスを直接指定する」視点へ一般化。

## 関連ページ

- [[denoising-diffusion]] — 拡散モデルの定式化と DDPM（系譜の起点）
- [[diffusion-model-architecture]] — ノイズ予測ネットワーク（U-Net）の設計と ADM の改良 U-Net・AdaGN
- [[latent-diffusion]] — 潜在空間での拡散と LDM/Stable Diffusion（効率化・実用化）
- [[diffusion-sampling]] — サンプリング／ソルバー（DDIM による高速化）
- [[probability-flow-ode]] — 連続時間（SDE/ODE）定式化と確率フロー ODE
- [[score-based-generative-models]] — スコアマッチング／Langevin・SDE 統一枠組み（VE/VP/sub-VP）
- [[controllable-generation]] — 可制御生成（スコア操作の逆問題＋ControlNet のアダプタ型空間条件制御）
- [[classifier-free-guidance]] / [[classifier-guidance]] — 条件忠実度↔多様性を制御する guidance
- [[training-free-conditioning]] — 凍結した無条件 prior を推論時だけ条件付ける学習不要パラダイム（RePaint）
- [[subject-driven-generation]] — 少数画像で T2I モデルを fine-tune し特定被写体を埋め込む personalization（DreamBooth）
- [[flow-matching]] — CNF をシミュレーション不要で学習する新パラダイム（拡散を内包・OT パス）
- [[text-to-image-generation]] / [[image-inpainting]] / [[super-resolution]] — LDM が切り開いた主要応用タスク
- [[summaries/2020-ddpm]]・[[summaries/2021-score-sde]]・[[summaries/2021-ddim]]・[[summaries/2021-adm]]・[[summaries/2022-latent-diffusion]]・[[summaries/2022-classifier-free-guidance]]・[[summaries/2022-repaint]]・[[summaries/2023-flow-matching]]・[[summaries/2023-controlnet]]・[[summaries/2023-dreambooth]] — 取り込み済み原典の要約
