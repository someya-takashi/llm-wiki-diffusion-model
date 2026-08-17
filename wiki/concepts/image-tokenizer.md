---
type: concept
aliases: [Image Tokenizer, 画像トークナイザ, VAE, Visual Tokenizer, Autoencoder, diffusability, 拡散可能性, 高圧縮VAE, High-Compression VAE]
tags: [image-tokenizer, latent-diffusion, diffusion-model-architecture, visual-text-rendering, generative-models]
related:
  - "[[latent-diffusion]]"
  - "[[diffusion-model-architecture]]"
  - "[[visual-text-rendering]]"
  - "[[denoising-diffusion]]"
  - "[[flow-matching]]"
summaries:
  - "[[summaries/2026-qwen-image-vae-2]]"
  - "[[summaries/2026-qwen-image-2]]"
  - "[[summaries/2025-qwen-image]]"
  - "[[summaries/2022-latent-diffusion]]"
updated: 2026-06-25
---

# Image Tokenizer（画像トークナイザ / 潜在空間を作るオートエンコーダ）

**Image Tokenizer（画像トークナイザ）** とは、[[latent-diffusion]]（潜在拡散）において**画像を圧縮した潜在表現へ変換し、また元へ戻す**部品——通常は VAE（Variational AutoEncoder, 変分オートエンコーダ）——と、その設計論を指す。拡散モデルの研究では長らく「脇役」として扱われてきたが、**潜在空間の形は下流の生成モデルが何を学べるかを規定する**ため、実際には生成品質の上限を決める中心的な構成要素である。本 wiki のランドマークは **Qwen-Image-VAE-2.0**（[[summaries/2026-qwen-image-vae-2]]）で、トークナイザだけを主題にした技術報告として設計上の論点を体系的に整理している。

## なぜトークナイザが重要なのか

[[latent-diffusion]] は「知覚的圧縮はオートエンコーダに任せ、拡散モデルは意味的圧縮に集中する」という分業で成功した。この分業は強力だが、裏を返せば**オートエンコーダが捨てた情報は、拡散モデルがどれだけ賢くても復元できない**ということでもある。

その最も分かりやすい実例が**小さな文字**である。拡散モデルが完璧に正しい潜在を生成しても、デコーダが読める文字に戻せなければ画像は判読不能になる——すなわち **[[visual-text-rendering]] の上限はトークナイザが決める**。初代 Qwen-Image（[[summaries/2025-qwen-image]]）が「デコーダだけをテキストが密なコーパスで微調整する」という手を打ったのは、まさにこの認識からである。

## 中心的な設計変数：圧縮率 $f$ とチャネル数 $C$

入力画像 $I\in\mathbb{R}^{H\times W\times 3}$ をトークナイザは潜在 $z\in\mathbb{R}^{\frac{H}{f}\times\frac{W}{f}\times C}$ へ写す。

- **$f$（空間圧縮率）**：縦横を何分の 1 にするか。$f8$（8 倍）が長らく業界標準だった。**$f16$**、**$f32$** と表記する。
- **$C$（潜在チャネル数）**：各位置が持つ次元。$f16c64$ は「16 倍圧縮・64 チャネル」を意味する。

$f$ を上げる動機は計算量である。拡散 Transformer（DiT, [[diffusion-model-architecture]]）の計算量は潜在トークン数 $L=HW/f^2$ に対して**二次的**に増えるため、全体では $\mathcal{O}(H^2W^2/f^4)$ となる。$f8\to f16$ で系列長は 1/4、計算量は約 1/16 になり、**ネイティブな高解像度生成が現実的になる**。

そして重要な原則が、再構成忠実度を支配するのは**総情報ボトルネック** $N(z)=CHW/f^2$ だという点である。$f$ を上げて失った容量は $C$ を増やせば補償できる。しかも **$C$ を増やしても DiT の計算量はほぼ変わらない**——DiT は最初に線形層で潜在を固定の隠れ次元へ射影するからである。したがって「$f$ を上げ、$C$ で埋める」が原理的に正しい方向になる。

## 三者間トレードオフ（tripartite trade-off）

ところが $C$ を増やすと第三の問題が現れる。Qwen-Image-VAE-2.0 が定式化した**三竦み**である。

| 求めるもの | 上げると何が起きるか |
| --- | --- |
| **① 圧縮率 $f$** | DiT は速くなるが、情報が入りきらず再構成が壊れる（とくに小さな文字） |
| **② 再構成忠実度** | 保つために $C$ を増やすと、潜在分布が複雑・非構造的になる |
| **③ 拡散可能性（diffusability）** | 潜在が複雑だと拡散モデルが学習しにくく、収束が遅れ生成品質が落ちる |

**diffusability（拡散可能性）** とは「その潜在空間がどれだけ容易に拡散モデルでモデル化できるか」という、**再構成品質とは独立した性質**である。再構成が完璧でも、潜在が拡散しにくければ下流の生成モデルは使い物にならない。トークナイザ設計とは、この 3 つを同時に立てる問題だと理解するのが本質的である。

## 代表手法：Qwen-Image-VAE-2.0（2026）

[[summaries/2026-qwen-image-vae-2]] は $f16$・$f32$ の VAE 群でこの三竦みを解く。

### アーキテクチャ

<figure>

![](../../raw/assets/2026-qwen-image-vae-2/x1.png)

<figcaption>図1（引用, [[summaries/2026-qwen-image-vae-2]] より）: スキップ接続の比較。NSC（なし）・LSC（局所）・GSC（大域）。GSC は Space-to-Channel でピクセル情報をチャネル方向へ折り畳み、初期ダウンサンプリングを迂回して潜在へ直送する。右のグラフ（f16c64 をゼロから学習）で GSC が明確に速く収束する。</figcaption>
</figure>

- **Global Skip Connection（GSC）**：高圧縮の敵は、エンコーダの初期ダウンサンプリングで高周波情報が落ちること。**space-to-channel**（空間情報をチャネル次元へ折り畳む）でピクセルから潜在への大域的な近道を作り、収束を大きく速める。
- **Attention-free バックボーン**：自己注意は計算もメモリも $\mathcal{O}(N^2)$ で高解像度では致命的。畳み込みは $\mathcal{O}(N\cdot k^2)$。外しても性能が落ちなかったので全廃した。
- **エンコーダ・デコーダの非対称化**：エンコーダ 76〜78M に対しデコーダ 248〜250M。理由が実務的で、**エンコーダは拡散モデルの学習中に毎回走る**ので軽い方が学習が速く、デコーダは推論で 1 回だけなので重くてよい。

### 学習：KL も GAN も外す

損失は $\mathcal{L}_{recon}$（L1）＋ $\mathcal{L}_{lpips}$（知覚損失）＋ $\mathcal{L}_{align}$（意味的整合）の 3 項のみ。VAE の教科書的な部品を 2 つとも捨てたことが主張である。

- **KL 損失の除去**：KL は潜在を正規分布へ寄せるが、(1) 容量を制約して再構成を損ない、(2) より重要なことに**意味的整合と競合する**——目標の意味特徴はガウス分布ではないので、正規事前分布と意味的多様体の両方を満たせと強いると整合が中途半端になり、下流 DiT の収束がかえって遅れる。
- **GAN 損失の除去**：学習予算が十分大きければ再構成損失＋LPIPS で十分鋭くなる。識別器を消すと最適化が単純になり安定・高速化する。初代 Qwen-Image も「品質が上がると識別器が有効な勾配を出せなくなる」と同趣旨の観察を報告している。

### 拡散可能性を作る：意味的整合

**潜在を事前学習済み視覚エンコーダの特徴へ揃える**（VA-VAE 由来）。ここでの知見が 2 つ：

- **DINOv2 が最良**（DINOv3・MAE・PE-Spatial との比較）。
- **最終層ではなく中間層**に揃えるのが良い。中間層の方が空間マップが滑らかで整合しやすく、生成に適した潜在になる。複数層を素朴に混ぜると逆効果。

損失は **Marginal Cosine Similarity**（潜在の向きを意味特徴に合わせる）と **Marginal Distance Matrix Similarity**（位置ペア間の類似度構造＝相対的な空間レイアウトを保つ）の 2 本立てで、いずれも ReLU でマージンを設ける。さらに**段階的整合**——初期は厳しく整合させて拡散可能性を確立し、後半は緩めて再構成品質とのバランスを取る。

### 成果

- **f16c128 が文書テキスト再構成で全 f8 VAE を上回る**（NED 0.9617 対 FLUX.1-dev 0.9546）。「f8 を超えるテキスト忠実度を達成した初の f16 オートエンコーダ」。
- **f32c192 が 4 倍高い圧縮率で f8 の Wan2.1 に匹敵**。
- 拡散可能性：ImageNet で SiT を学習した gFID/IS で高圧縮ベースラインを一貫して上回る。

## 評価：ピクセル指標では足りない

トークナイザの評価は伝統的に **PSNR / SSIM / LPIPS / FID** で行われてきたが、これらは**文字の判読性に鈍感**である。「orange」が「orango」になっても PSNR の低下は 0.5 dB 未満だが、意味は壊れている。

そこで Qwen-Image-VAE-2.0 は **OmniDoc-TokenBench** を提案する。約 3K 枚の文書画像（書籍・スライド・教科書・試験問題・論文・雑誌・財務報告・新聞・ノートの 9 カテゴリ、中英）を、**OCR ベースの NED（Normalized Edit Distance, 正規化編集距離）** で測る。設計上の白眉は、**正解アノテーションではなく「元画像を OCR した結果」を参照にする**こと——OCR 自体がクリーンな画像でも誤る（「rn」を「m」と読む等）ので、両方に同じ OCR をかければ系統誤差が相殺され、**VAE の劣化だけを切り分けられる**。

実際、ピクセル指標と NED は一致しない：Stepvideo-T2V は HunyuanImage-3.0 と SSIM 差はわずかなのに NED は大きく上回る（0.8838 対 0.7753）。**NED は独立に必要な指標である**。

## 既存知識との接続

- [[latent-diffusion]]：トークナイザは LDM の第一段階そのもの。本ページはその「第一段階」を独立した設計問題として扱う。LDM 原典（[[summaries/2022-latent-diffusion]]）が $f4$〜$f8$ を最適点としたのに対し、高解像度時代は $f16$・$f32$ へ移りつつある。
- [[diffusion-model-architecture]]：DiT の計算量が系列長に二次であることが、圧縮率を上げる直接の動機になる。トークナイザ側の改善がバックボーン側の制約を緩める関係。
- [[visual-text-rendering]]：小さな文字の再現はトークナイザが上限を決める。初代 Qwen-Image はデコーダ微調整で、VAE-2.0 は圧縮率を上げつつ設計で解いた。
- [[denoising-diffusion]] / [[flow-matching]]：潜在空間の「形」が拡散／フローの学習しやすさ（diffusability）を左右する。生成モデルとトークナイザの分業を問い直す視点。
- [[summaries/2026-qwen-image-2]]：VAE-2.0 の f16c64 を実際に採用し、ネイティブ 2K 生成を可能にしている。

## 参考文献（summaries）

- [[summaries/2026-qwen-image-vae-2]] — Qwen-Image-VAE-2.0（f16/f32 高圧縮 VAE。GSC・attention-free・非対称構成、KL/GAN 除去、DINOv2 中間層への意味的整合、OmniDoc-TokenBench）
- [[summaries/2026-qwen-image-2]] — Qwen-Image-2.0（f16c64 を採用しネイティブ 2K 生成を実現した基盤モデル）
- [[summaries/2025-qwen-image]] — Qwen-Image（動画対応 VAE のデコーダのみをテキスト特化微調整し、文字再現の上限を引き上げた先行例）
- [[summaries/2022-latent-diffusion]] — Latent Diffusion Models（知覚的圧縮と意味的圧縮の分業、$f$ の選択という枠組みを確立した原典）
