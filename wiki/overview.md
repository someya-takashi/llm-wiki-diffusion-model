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
- **設計空間の整理**：**EDM**（Karras ら 2022, [[summaries/2022-edm]]）。拡散の設計を「サンプリング・前処理・学習」のモジュール空間に分解し、Heun 2 次サンプラー＋ρ=7 の σ スケジュール（[[noise-schedule]]）＋ preconditioning（[[diffusion-model-architecture]]）＋対数正規ノイズ分布を体系化。事前学習モデルにサンプラーを差し替えるだけで NFE を一桁削減し、CIFAR-10 を 35 NFE で SOTA、ImageNet-64 を 1.36 に。後続の SDXL・SD3・Stochastic Interpolants が土台に参照する。
- **GAN を超えた点**：[[denoising-diffusion]] の改良＝**ADM**（Dhariwal & Nichol 2021）。改良 U-Net（多解像度 attention・BigGAN resblock・AdaGN, [[diffusion-model-architecture]]）と [[classifier-guidance]]（分類器ガイダンス）を組み合わせ、ImageNet で拡散モデルが初めて GAN（BigGAN-deep）を画像品質で上回った。分類器勾配のスケール 1 個で「忠実度↔多様性」を制御できるノブを与えた点が鍵。
- **効率化・実用化**：[[latent-diffusion]]（潜在拡散＝Stable Diffusion, Rombach ら 2022）。オートエンコーダの潜在空間で拡散することで計算量を一桁削減し、cross-attention 条件付けで多タスク化。画像生成 AI を一般に普及させた転換点。その大型化後継 **SDXL**（Podell ら 2023）は、3× UNet＋2 テキストエンコーダ＋micro-conditioning（size/crop/aspect-ratio）＋base+refiner の 2 段で実務的に強化し、**オープンモデルでブラックボックス SOTA（Midjourney 等）に匹敵**——同時に COCO zero-shot FID が美的品質と乖離することも示した（[[latent-diffusion]]）。
- **アーキテクチャの転回**：[[diffusion-model-architecture]] の **DiT（Diffusion Transformer, Peebles & Xie 2023）**。拡散の定番バックボーンだった U-Net を Transformer（ViT）に置き換え、「U-Net の帰納バイアスは不要」「計算量（Gflops）を増やすほど FID が下がる」スケーリング則を示し ImageNet で SOTA。これを text-to-image のマルチモーダル性に拡張したのが **MM-DiT**（Esser ら 2024, Stable Diffusion 3）で、テキストと画像に別重みを与え attention でのみ双方向結合する。アーキテクチャ系譜は ADM（改良 U-Net）→ SDXL（U-Net スケール）→ DiT（Transformer 化）→ MM-DiT（マルチモーダル化）と進む。
- **条件付け・応用**：[[controllable-generation]]（可制御生成）を起点に、guidance の系譜は [[classifier-guidance]]（分類器ガイダンス＝ADM 2021, 分類器の勾配を足す）→ [[classifier-free-guidance]]（CFG 2022, 分類器を排し条件付き／無条件スコアの差で同じノブを実現）と進んだ。これに cross-attention を加えて [[text-to-image-generation]]・[[image-inpainting]]・[[super-resolution]] が実現する。さらに **ControlNet**（Zhang ら 2023）が zero convolution で Stable Diffusion を凍結したままエッジ・深度・姿勢などの空間条件を後付けし、制御性を実用面で一変させた。
- **学習不要の推論時条件付け**：[[training-free-conditioning]]。事前学習済みの無条件拡散モデルを prior として固定し、**サンプリング過程だけ**で条件を課す系統。**RePaint**（Lugmayr ら 2022）は凍結 DDPM＋既知領域の置き換え＋resampling で、再学習なしに任意マスクの [[image-inpainting]] を実現した。条件タイプごとの再学習が要らず、強力な無条件 prior をそのまま使い回せるのが利点。
- **personalization（被写体駆動生成）**：[[subject-driven-generation]]。逆に、少数画像で T2I モデルを **fine-tune** して「特定個体」をモデルに埋め込む系統。**DreamBooth**（Ruiz ら 2023）は 3〜5 枚で Imagen / Stable Diffusion を fine-tune し、一意識別子と prior preservation loss で被写体を新文脈に再生成。同時期の **Textual Inversion**（Gal ら 2022・[[summaries/2022-textual-inversion]]）は逆にモデルを凍結し、テキスト埋め込み空間に概念を表す擬似単語の埋め込みだけを学ぶ——personalization タスクを提唱した、DreamBooth と並ぶもう一方の原典。三本目の **Custom Diffusion**（Kumari ら 2023・[[summaries/2023-custom-diffusion]]）は cross-attention の key/value 射影だけを fine-tune（両者の中間の表現力／コスト、75MB・約 6 分）し、さらに**複数の新概念を 1 枚に合成**するタスクを切り開いた多概念カスタマイズの源流。テキスト条件が「何を」を、personalization が「どの個体を」を制御する。
- **軽量化と複数概念合成**：personalization を軽くしたのが **[[low-rank-adaptation]]（LoRA, Hu ら 2022）**——低ランク更新 $\Delta W=BA$ だけを学習し数 MB で共有でき、拡散コミュニティの標準になった。共有 LoRA が増えた結果、**複数の LoRA／概念を 1 枚に合成する** [[multi-concept-customization]] が課題化——このタスクと閉形式の重みマージを最初に確立したのが **Custom Diffusion**（Kumari ら 2023）である。解は 3 系統に分かれる：(a) **重みマージ／融合**（[[lora-merging]]）——Custom Diffusion の閉形式マージを起点に、素朴な線形和は identity loss・列干渉で破綻するが、**Mix-of-Show**（Gu ら 2023, ED-LoRA＋gradient fusion で推論挙動を整合）や **ZipLoRA**（Shah ら 2023, content+style を学習係数で干渉最小化マージ）が克服した。(b) **LoRA-Composer**（注意制御）や (c) **Multi-LoRA Composition**（LoRA Switch / Composite, 復号過程で合成）は重みを混ぜず訓練不要の解を与えた。
- **画像コンポジション（object teleportation）**：[[image-composition]]。fine-tune せず、参照物体を**シーンの指定位置に zero-shot で合成**する系統。**AnyDoor**（Chen ら 2023）は DINO-V2 の ID 特徴と高周波マップの detail 特徴で物体を特徴づけ、Stable Diffusion に注入して仮想試着・物体移動などを実現。DreamBooth が「テキストで新文脈に生成」なら、AnyDoor は「与えられたシーン・位置に合成」。
- **拡散の一般化／別系統**：[[flow-matching]]（Lipman ら 2023）。連続正規化フロー（CNF）をシミュレーション不要で学習し、拡散パスを特別な場合として内包しつつ、最適輸送（OT）パスで直線的・高速・安定な生成を実現。拡散の確率フロー ODE（[[probability-flow-ode]]）を「確率パスを直接指定する」視点へ一般化。その直線インスタンス **rectified flow** を、改良ノイズサンプラー（logit-normal）と MM-DiT で大規模 text-to-image に確立したのが **Stable Diffusion 3**（Esser ら 2024）で、SDXL までの拡散定式化に代わる実用標準になった。
- **flows と diffusions の統一**：[[stochastic-interpolants]]（Albergo–Boffi–Vanden-Eijnden 2023）。任意の 2 密度を有限時間で結ぶ補間 $x_t=I(t,x_0,x_1)+\gamma(t)z$ を作り、その密度が輸送方程式（→決定論 ODE 生成）と調整可能な Fokker-Planck 方程式（→確率 SDE 生成）の両方を満たすことを示す。flow matching・rectified flow・スコアベース拡散・Schrödinger 橋をすべて特別な場合として内包し、決定論的生成（[[probability-flow-ode]]）と確率的生成（[[score-based-generative-models]]）を 1 つの枠組みに統一した。
- **統一的入門リファレンス**：**An Introduction to Flow Matching and Diffusion Models**（Holderrieth & Erives, MIT 6.S184, 2025・[[summaries/2025-flow-matching-diffusion-intro]]）。上記の理論系（[[flow-matching]]・[[stochastic-interpolants]]・[[probability-flow-ode]]・[[score-based-generative-models]]・[[denoising-diffusion]]）を、「ノイズ→データ＝ODE/SDE シミュレーション」と「conditional→marginal の周辺化トリック」という 1 本の筋で導く教科書的講義ノート。連続の方程式・Fokker-Planck・Langevin から CFG・U-Net/DiT/MM-DiT まで自己完結的にカバーし、本 wiki の理論ページを束ねる地図になる。

## 関連ページ

- [[denoising-diffusion]] — 拡散モデルの定式化と DDPM（系譜の起点）
- [[diffusion-model-architecture]] — ノイズ予測ネットワークの設計（ADM の改良 U-Net・AdaGN → DiT の Transformer 化）
- [[latent-diffusion]] — 潜在空間での拡散と LDM/Stable Diffusion（効率化・実用化）
- [[diffusion-sampling]] — サンプリング／ソルバー（DDIM による高速化）
- [[probability-flow-ode]] — 連続時間（SDE/ODE）定式化と確率フロー ODE
- [[noise-schedule]] — ノイズスケジュール（学習時のノイズ分布＋推論時の時間離散化、EDM の σ(t)=t・ρ=7・対数正規）
- [[score-based-generative-models]] — スコアマッチング／Langevin・SDE 統一枠組み（VE/VP/sub-VP）
- [[controllable-generation]] — 可制御生成（スコア操作の逆問題＋ControlNet のアダプタ型空間条件制御）
- [[classifier-free-guidance]] / [[classifier-guidance]] — 条件忠実度↔多様性を制御する guidance
- [[training-free-conditioning]] — 凍結した無条件 prior を推論時だけ条件付ける学習不要パラダイム（RePaint）
- [[subject-driven-generation]] — 少数画像で T2I モデルを fine-tune し特定被写体を埋め込む personalization（DreamBooth）
- [[low-rank-adaptation]] — 低ランク適応（LoRA）。拡散の軽量 personalization の標準
- [[multi-concept-customization]] — 複数の LoRA／概念を 1 枚に合成（LoRA-Composer・Multi-LoRA Composition）
- [[lora-merging]] — 複数 LoRA の重みマージ／融合（Mix-of-Show の gradient fusion・ZipLoRA の学習係数マージ）
- [[image-composition]] — 参照物体をシーンの指定位置に zero-shot で合成する object teleportation（AnyDoor）
- [[flow-matching]] — CNF をシミュレーション不要で学習する新パラダイム（拡散を内包・OT パス・rectified flow＝SD3）
- [[stochastic-interpolants]] — flows と diffusions を有限時間で統一する枠組み（ODE/SDE を同じ密度の異なる実現に）
- [[text-to-image-generation]] / [[image-inpainting]] / [[super-resolution]] — LDM が切り開いた主要応用タスク
- [[summaries/2020-ddpm]]・[[summaries/2021-score-sde]]・[[summaries/2021-ddim]]・[[summaries/2021-adm]]・[[summaries/2022-latent-diffusion]]・[[summaries/2022-classifier-free-guidance]]・[[summaries/2022-repaint]]・[[summaries/2022-lora]]・[[summaries/2023-flow-matching]]・[[summaries/2023-controlnet]]・[[summaries/2023-dreambooth]]・[[summaries/2023-anydoor]]・[[summaries/2023-dit]]・[[summaries/2024-lora-composer]]・[[summaries/2024-multi-lora-composition]]・[[summaries/2023-mix-of-show]]・[[summaries/2024-ziplora]]・[[summaries/2023-sdxl]]・[[summaries/2024-sd3]]・[[summaries/2024-stochastic-interpolants]]・[[summaries/2022-edm]]・[[summaries/2022-textual-inversion]]・[[summaries/2023-custom-diffusion]]・[[summaries/2025-flow-matching-diffusion-intro]] — 取り込み済み原典の要約
