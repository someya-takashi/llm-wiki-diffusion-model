---
type: concept
aliases: [Diffusion Model Architecture, 拡散モデルのアーキテクチャ, ADM, AdaGN, U-Net for diffusion, DiT, Diffusion Transformer, adaLN, adaLN-Zero, patchify]
tags: [diffusion-model-architecture, denoising-diffusion, latent-diffusion, generative-models, image-generation, dit]
related:
  - "[[denoising-diffusion]]"
  - "[[latent-diffusion]]"
  - "[[classifier-guidance]]"
  - "[[diffusion-sampling]]"
summaries:
  - "[[summaries/2021-adm]]"
  - "[[summaries/2020-ddpm]]"
  - "[[summaries/2023-dit]]"
  - "[[summaries/2023-sdxl]]"
  - "[[summaries/2024-sd3]]"
  - "[[summaries/2022-edm]]"
updated: 2026-06-24
---

# Diffusion Model Architecture（拡散モデルのアーキテクチャ）

**Diffusion Model Architecture（拡散モデルのアーキテクチャ）** とは、拡散モデルの心臓部である**ノイズ予測ネットワーク $\epsilon_\theta(\mathbf{x}_t,t)$**——ノイズの乗った画像とその時刻（必要ならクラスやテキストなどの条件）を入力に、そこに乗っているノイズ（＝スコア）を出力するニューラルネット——の設計を指す。拡散モデルでは「学習目的」（[[denoising-diffusion]]）と「サンプラー」（[[diffusion-sampling]]）が分離して論じられるが、**生成品質を最終的に決めるのはこのネットワーク自身の表現力**でもある。実際、Dhariwal & Nichol の **ADM（"Diffusion Models Beat GANs", 2021）** は、サンプリングや目的関数を変えずに**アーキテクチャ改良だけで FID を大きく押し下げ**、拡散モデルが GAN を超える土台を作った（[[summaries/2021-adm]]）。

本ページは拡散モデルのバックボーン設計を 2 つの柱で扱う。(1) 長く標準だった **U-Net** 系の改良の系譜（DDPM → IDDPM → ADM）と、(2) それを覆した **Transformer への転回（DiT）**——U-Net の帰納バイアスは必須でなく、Transformer に置き換えるとスケーラビリティの恩恵を受けられる、という転換である。

## なぜ U-Net なのか

拡散モデルのノイズ予測は「画像と同じ解像度の出力（＝各ピクセルのノイズ）を返す」**画像→画像（dense prediction）**タスクである。これに適すのが **U-Net**：ダウンサンプリングで解像度を下げながら特徴を抽象化するエンコーダと、アップサンプリングで解像度を戻すデコーダを、同解像度の層同士を結ぶ**スキップ接続（skip connections）**でつないだ構造。スキップ接続が細部（高周波）情報を保ち、エンコーダ側で得た大域文脈と合流させられるため、ノイズ除去に向く。拡散 U-Net には共通して次の要素が入る：

- **時刻埋め込み（timestep embedding）**：時刻 $t$ を正弦波で埋め込み、各残差ブロックに注入する。「いまどれだけノイズが乗った段階か」をネットに伝える。
- **残差ブロック（residual blocks）**：各解像度に複数積む。
- **自己注意（self-attention）**：低〜中解像度の特徴マップに大域的な依存を入れる。

## 改良の系譜

### DDPM の U-Net（Ho ら 2020）

拡散モデルに U-Net を持ち込んだ起点（[[denoising-diffusion]], [[summaries/2020-ddpm]]）。残差ブロック＋ダウン/アップサンプリング畳み込みに、**16×16 解像度で単一ヘッドの大域 self-attention** を 1 つ置き、時刻埋め込みを各ブロックに加える、という比較的素朴な構成。

### IDDPM（Nichol & Dhariwal 2021）

分散 $\Sigma_\theta$ を定数固定せず**学習（learned variance）**し、cosine ノイズスケジュールと $L_\text{simple}+\lambda L_\text{vlb}$ のハイブリッド目的を導入。少ステップサンプリングを改善した（ADM はこの分散学習を継承）。専用ページは設けず本項に記す。

### ADM（Dhariwal & Nichol 2021）——本概念のランドマーク

ADM は ImageNet 128×128 で系統的なアブレーション（[[summaries/2021-adm]] 表1〜3・図2）を行い、U-Net を次のように作り替えた。これが「拡散が GAN を超えた」要因の半分（残り半分は [[classifier-guidance]]）。

- **多解像度アテンション（multi-resolution attention）**：self-attention を 16×16 のみ → **32×32 / 16×16 / 8×8** の複数解像度に拡張。
- **アテンションヘッドの最適化**：ヘッド数を増やすか、ヘッドあたりチャネルを減らすほど FID が改善。最終的に**ヘッドあたり 64 チャネル**を既定に（Transformer 流）。
- **BigGAN residual block** をアップ/ダウンサンプリングに採用。
- **深さ vs 幅**：深くすると効くが学習が遅いため不採用。残差接続の $1/\sqrt2$ リスケールは効かず不採用。
- **Adaptive Group Normalization（AdaGN, 適応的グループ正規化）**：時刻埋め込みとクラス埋め込みを線形射影して $y=[y_s,y_b]$ を作り、GroupNorm 後の活性 $h$ に

$$
\text{AdaGN}(h,y)=y_s\,\text{GroupNorm}(h)+y_b
$$

として注入する（FiLM・adaptive instance norm に類似）。単純な「加算＋GroupNorm」より FID が良い。条件（時刻・クラス）を正規化のスケール／シフトとして効かせるのがポイント。

この最終構成（可変幅・解像度あたり 2 残差ブロック・多解像度 attention・BigGAN resblock・AdaGN）が、その後の拡散モデルの事実上の標準アーキテクチャになった。

### SDXL（Podell ら 2023）——改良 U-Net をスケールする

DiT が U-Net を捨てる一方、**SDXL** は ADM 系の改良 U-Net を**そのままスケール**する道を採った（[[latent-diffusion]] の代表的後継モデル・[[summaries/2023-sdxl]]）。Stable Diffusion の U-Net を 860M → **2.6B**（約 3×）に大型化したが、単純に各レベルを厚くするのでなく、**transformer block を低レベル特徴に集中させる不均一配分** `[0, 2, 10]`（最高解像度レベルの transformer block を省略、最低レベル＝8× ダウンサンプリングを撤去）を採る（simple diffusion 流。channel mult. も `[1,2,4]`）。条件付けは CLIP ViT-L ＋ OpenCLIP ViT-bigG の 2 エンコーダ出力連結（context dim 2048）＋ pooled text embedding を timestep embedding に加算する形に拡張した。

注目すべきは、SDXL が探索段階で **UViT や DiT のような transformer ベースを試したが「即座の利点は見出せなかった」**とし、改良 U-Net に留まった点である（[[summaries/2023-sdxl]] §3）。同時期に DiT が「U-Net 不要」を示したのと対照的で、U-Net の作り込み（ADM→SDXL）と Transformer 化（DiT）が 2023 年時点で併存していたことを示す。

## Transformer への転回：DiT（Peebles & Xie 2023）——本概念の第 2 のランドマーク

ADM までの改良はすべて「U-Net をどう良くするか」だったが、**DiT（Diffusion Transformer）** はそもそも U-Net を捨てた。「拡散モデルに U-Net の帰納バイアスは本当に必要か？」という問いに対し、**バックボーンを Vision Transformer（ViT）に丸ごと置き換えても、むしろ Transformer 由来の優れたスケーリング性が得られる**ことを示した（[[summaries/2023-dit]]）。拡散の数学（ε／Σ 予測・$L_\text{simple}$）も CFG も [[latent-diffusion]] の VAE 潜在空間も既存のまま、**変えたのはネットワークだけ**である。

- **patchify**：VAE 潜在（例 32×32×4）をパッチサイズ $p$（=2/4/8）でパッチに切り、線形埋め込み＋ViT の正弦波位置埋め込みでトークン列にする。トークン数 $T=(I/p)^2$。$p$ を半分にすると $T$ が 4 倍＝計算量（Gflops）が 4 倍になるが、パラメータ数はほぼ不変。
- **DiT block の条件付け**：時刻 $t$・クラス $c$ の注入方式を 4 種比較（in-context / cross-attention / adaLN / **adaLN-Zero**）。**adaLN-Zero** が最良——adaLN（LayerNorm のスケール $\gamma$・シフト $\beta$ を条件から回帰）に加え、残差直前のスケール $\alpha$ も回帰し、**$\alpha$=0 初期化で各ブロックを恒等写像から開始**する（ADM 系 U-Net の「残差前の層をゼロ初期化」を Transformer に移植したもの）。ADM の AdaGN と同じ「条件を正規化のスケール／シフトで効かせる」発想が、ここでも最良の条件付けとして再発見されている。
- **スケーラビリティが主役**：モデル Gflops（深さ・幅 or トークン数）と FID に**強い負の相関**があり、パラメータ数ではなく **Gflops** が品質を決める。S/B/L/XL × patch 2/4/8 の系統的スイープでこれを実証。
- **成果**：DiT-XL/2 が class-conditional ImageNet 256 で **FID 2.27**（ADM・LDM を更新、当時 SOTA）、512 で 3.04。しかも 118.6 Gflops と、ADM（1120 Gflops）より遥かに compute 効率が良い。

<figure>

![](../../raw/assets/2023-dit/x3.png)

<figcaption>図3（引用, [[summaries/2023-dit]] より）: DiT アーキテクチャ。潜在を patchify → N 個の DiT ブロック → 線形デコードでノイズ＋共分散を出力。ブロックは adaLN-Zero（最良）・cross-attention・in-context の変種。</figcaption>
</figure>

DiT は「拡散のバックボーンは U-Net である必要はない」ことを決定づけ、その後の **Stable Diffusion 3・PixArt-α・Sora（動画）** など主要モデルが DiT 系アーキテクチャを採用する転換点になった。改良 U-Net（ADM）と Transformer（DiT）が、拡散モデルのバックボーン設計の 2 大系譜である。

### MM-DiT（Multimodal DiT, Esser ら 2024）——DiT のマルチモーダル拡張

DiT はクラス条件付き生成のために設計され、テキストのような系列条件は cross-attention で一方向に注入するのが通例だった。**MM-DiT（Multimodal Diffusion Transformer）** は、Stable Diffusion 3（[[summaries/2024-sd3]]）で導入された DiT の発展形で、**テキストと画像という 2 つのモダリティに別々の重みの組**を与える。

- **2 ストリーム＋共有 attention**：画像トークンとテキストトークンを各々別の重みで処理しつつ、**attention 演算のときだけ両系列を連結**して双方向に情報を混ぜる（実質「2 つの独立 transformer が attention で握手する」）。固定テキスト表現の一方向 cross-attention より、テキスト理解・タイポグラフィ・人間選好が向上。DiT は「全モダリティで 1 組の重みを共有する MM-DiT の特殊ケース」と位置づけられる。
- **条件付け**：DiT 同様、時刻 $t$ と pooled テキスト $c_{\text{vec}}$ を adaLN 風の変調に入れ、加えて系列テキスト $c_{\text{ctxt}}$ をトークンとして供給する（pooled だけでは粗いため）。
- **QK-normalization**：高解像度で mixed-precision 学習が発散する問題（attention logit の増大不安定性）を、attention 前に Q・K を RMSNorm することで防ぐ。識別的 ViT 文献（Dehghani ら）の知見の移植。
- **スケーリング**：深さ $d$ で hidden=$64d$・ヘッド数=$d$ とパラメータ化し 8B までスケール。検証損失が人間評価・ベンチマークと強く相関し飽和しない。

MM-DiT は DiT を text-to-image のマルチモーダル性に合わせて拡張したもので、本 wiki のアーキテクチャ系譜「改良 U-Net（ADM）→ SDXL（U-Net スケール）→ DiT（Transformer 化）→ MM-DiT（マルチモーダル化）」の最新地点にあたる。詳細は [[summaries/2024-sd3]]。

## 既存知識との接続

- [[denoising-diffusion]]：アーキテクチャはノイズ予測 $\epsilon_\theta$ の中身。DDPM の U-Net がこの系譜の起点。
- [[classifier-guidance]]：ADM は guidance と同時にこのアーキテクチャ改良を提案。両者あわせて拡散が GAN を超えた。分類器自身も「U-Net のダウンサンプリング部＋8×8 アテンションプール」というこのアーキテクチャの部分を流用する。
- [[latent-diffusion]]：Stable Diffusion の U-Net も ADM 系の改良 U-Net を踏襲し、cross-attention でテキスト条件を注入する。拡散をピクセル空間から潜在空間へ移すことで、同じアーキテクチャを高解像度に適用可能にした。その後継 SDXL は同じ改良 U-Net を 3× にスケールし transformer block 配分を最適化（[[summaries/2023-sdxl]]）、DiT は同じ VAE 潜在空間でバックボーンだけを Transformer 化したもの。U-Net スケール（SDXL）と Transformer 化（DiT）が 2023 年の 2 つの選択肢。
- [[diffusion-sampling]]：アーキテクチャ（モデルの中身）とサンプラー（生成手続き）は直交する設計軸。同じ ADM/DiT をどのサンプラー（DDPM/DDIM）で回すかは別問題。
- [[classifier-free-guidance]]：DiT は CFG で高品質化し（cfg=1.5 で SOTA）、部分チャネル CFG の知見も示した。アーキテクチャと guidance は独立した改善軸。
- [[low-rank-adaptation]]：LoRA はバックボーン本体を変えず、注意・線形層の重みに低ランク更新 $\Delta W=BA$ を後付けで施す適応手法。アーキテクチャ設計とは直交する。
- [[noise-schedule]]：EDM（[[summaries/2022-edm]]）の **preconditioning** は、バックボーンに依らないネット入出力の設計軸である。生のネット $F_\theta$ を $\sigma$ 依存スケーリングで包んで denoiser $D_\theta(\boldsymbol{x};\sigma)=c_{\rm skip}\boldsymbol{x}+c_{\rm out}F_\theta(c_{\rm in}\boldsymbol{x};c_{\rm noise})$ を作り、入力・学習目標を単位分散に保ち誤差増幅を最小化する $c_{\rm skip}/c_{\rm out}/c_{\rm in}/c_{\rm noise}$ を第一原理で導く。U-Net でも DiT でも適用でき、アーキ本体と直交して学習を安定化する。

## 参考文献（summaries）

- [[summaries/2021-adm]] — Diffusion Models Beat GANs on Image Synthesis（ADM＝改良 U-Net・AdaGN）
- [[summaries/2020-ddpm]] — Denoising Diffusion Probabilistic Models（拡散モデルへの U-Net 導入）
- [[summaries/2023-dit]] — Scalable Diffusion Models with Transformers（DiT＝Transformer バックボーン・adaLN-Zero・Gflops スケーリング）
- [[summaries/2023-sdxl]] — SDXL（改良 U-Net を 3× スケール・transformer block の不均一配分・transformer 化は当時見送り）
- [[summaries/2024-sd3]] — Stable Diffusion 3（MM-DiT＝DiT のマルチモーダル拡張・QK-normalization・rectified flow）
- [[summaries/2022-edm]] — EDM（preconditioning $c_{\rm skip}/c_{\rm out}/c_{\rm in}/c_{\rm noise}$＝ネット入出力の前処理設計軸）
- [[summaries/2025-flow-matching-diffusion-intro]] — Flow Matching と拡散モデル入門（MIT 6.S184 講義ノート。U-Net・DiT・MM-DiT と条件付け変数の符号化・潜在空間動作を概観）
