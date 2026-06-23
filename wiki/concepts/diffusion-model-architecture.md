---
type: concept
aliases: [Diffusion Model Architecture, 拡散モデルのアーキテクチャ, ADM, AdaGN, U-Net for diffusion]
tags: [diffusion-model-architecture, denoising-diffusion, generative-models, image-generation]
related:
  - "[[denoising-diffusion]]"
  - "[[latent-diffusion]]"
  - "[[classifier-guidance]]"
  - "[[diffusion-sampling]]"
summaries:
  - "[[summaries/2021-adm]]"
  - "[[summaries/2020-ddpm]]"
updated: 2026-06-23
---

# Diffusion Model Architecture（拡散モデルのアーキテクチャ）

**Diffusion Model Architecture（拡散モデルのアーキテクチャ）** とは、拡散モデルの心臓部である**ノイズ予測ネットワーク $\epsilon_\theta(\mathbf{x}_t,t)$**——ノイズの乗った画像とその時刻（必要ならクラスやテキストなどの条件）を入力に、そこに乗っているノイズ（＝スコア）を出力するニューラルネット——の設計を指す。拡散モデルでは「学習目的」（[[denoising-diffusion]]）と「サンプラー」（[[diffusion-sampling]]）が分離して論じられるが、**生成品質を最終的に決めるのはこのネットワーク自身の表現力**でもある。実際、Dhariwal & Nichol の **ADM（"Diffusion Models Beat GANs", 2021）** は、サンプリングや目的関数を変えずに**アーキテクチャ改良だけで FID を大きく押し下げ**、拡散モデルが GAN を超える土台を作った（[[summaries/2021-adm]]）。

本ページは、拡散モデルで標準となった **U-Net** 系アーキテクチャの俯瞰と、その改良の系譜（DDPM → IDDPM → ADM）を扱う。

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

## 既存知識との接続

- [[denoising-diffusion]]：アーキテクチャはノイズ予測 $\epsilon_\theta$ の中身。DDPM の U-Net がこの系譜の起点。
- [[classifier-guidance]]：ADM は guidance と同時にこのアーキテクチャ改良を提案。両者あわせて拡散が GAN を超えた。分類器自身も「U-Net のダウンサンプリング部＋8×8 アテンションプール」というこのアーキテクチャの部分を流用する。
- [[latent-diffusion]]：Stable Diffusion の U-Net も ADM 系の改良 U-Net を踏襲し、cross-attention でテキスト条件を注入する。拡散をピクセル空間から潜在空間へ移すことで、同じアーキテクチャを高解像度に適用可能にした。
- [[diffusion-sampling]]：アーキテクチャ（モデルの中身）とサンプラー（生成手続き）は直交する設計軸。同じ ADM をどのサンプラー（DDPM/DDIM）で回すかは別問題。

## 参考文献（summaries）

- [[summaries/2021-adm]] — Diffusion Models Beat GANs on Image Synthesis（ADM＝改良 U-Net・AdaGN）
- [[summaries/2020-ddpm]] — Denoising Diffusion Probabilistic Models（拡散モデルへの U-Net 導入）
