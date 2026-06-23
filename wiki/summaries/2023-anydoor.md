---
type: summary
source_path: raw/papers/AnyDoor_ Zero-shot Object-level Image Customization.md
source_kind: paper
title: "AnyDoor: Zero-shot Object-level Image Customization"
authors: [Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, Hengshuang Zhao]
year: 2023
venue: ICCV 2023
ingested: 2026-06-24
tags: [image-composition, subject-driven-generation, latent-diffusion, controllable-generation, generative-models]
translation: "[[translations/2023-anydoor]]"
---

# AnyDoor: Zero-shot Object-level Image Customization

> 原典: [[translations/2023-anydoor]] ・ `raw/papers/AnyDoor_ Zero-shot Object-level Image Customization.md`
> 著者・年・会議: Chen, Huang, Liu, Shen, Zhao, Zhao（HKU / Alibaba / Ant）・2023・ICCV 2023（arXiv 2307.09481）

## 一言まとめ

**参照物体の写真を、シーン画像のユーザー指定位置（ボックス）に、物体ごとの追加学習なし（zero-shot）で、同一性（ID）を保ちつつ調和的に合成する**拡散ベース生成器。著者の言う "object teleportation（物体テレポーテーション）"。鍵は、物体を **DINO-V2 の ID 特徴**と**高周波マップ（HF-map）の detail 特徴**で二重に特徴づけ、Stable Diffusion に注入すること、そして**動画データ＋適応的時刻ステップサンプリング**で学習すること。

## 背景と問題意識

「特定の物体を、特定のシーンの特定の場所に、見た目を保って置く」——画像コンポジション（[[image-composition]]）の核心問題。従来手法には 2 つの系統があり、それぞれ不足があった。

1. **参照ベース（reference-based）**：Paint-by-Example・ObjectStitch は対象画像を CLIP 画像エンコーダで埋め込んでガイドにするが、**CLIP は意味レベルの粗い情報しか持たず**、未学習カテゴリでは ID（同一性）が崩れる。
2. **チューニングベース（tuning-based）**：DreamBooth・Custom Diffusion・Cones（[[subject-driven-generation]]）は物体ごとに 4〜5 枚で約 1 時間 fine-tune する。高忠実だが**遅く、位置を指定できず、複数被写体だと属性が混ざる**。

AnyDoor は両者の良いとこ取りを狙う：**一度だけ学習して zero-shot で汎化し、位置も指定でき、複数被写体も扱える**。古典的な **image harmonization（画像調和）**（前景を貼って色・光だけ低レベルに調整）が構造・姿勢・影を作れないのとも対照的に、AnyDoor は姿勢・視点まで再生成する。

## 提案手法 / 主張

物体を「identity 特徴」と「detail 特徴」で表し、Stable Diffusion（[[latent-diffusion]]）に注入する（図2 のパイプライン）。

### (1) ID extractor — DINO-V2（§3.1）

CLIP の代わりに**自己教師ありモデル DINO-V2** を使う。自己教師あり学習はインスタンス検索能力を持ち、データ増強不変な識別的特徴を与えるため、CLIP より物体の同一性を保てる。**背景を除去**してから入れると特徴がクリーンになる。グローバルトークン＋パッチトークンを連結し、単一線形層で UNet 埋め込み空間の ID トークン（257×1024）に射影。

### (2) detail extractor — 高周波マップ（§3.2）

ID トークンは空間解像度を失うので細部を保てない。そこで物体を**そのまま貼る collage** で補おうとすると、今度は「コピー&ペースト」になって多様性が消える。解決策が **high-frequency map（HF-map）**＝Sobel フィルタで高周波（輪郭・テクスチャ）だけ抽出し、RGB と eroded mask（輪郭付近を除く収縮マスク）を掛けたもの：

$$
\mathbf{I}_h=(\mathbf{I}\otimes\mathbf{K}_h+\mathbf{I}\otimes\mathbf{K}_v)\odot\mathbf{I}\odot\mathbf{M}_\text{erode}
$$

これを指定位置に貼った collage を **ControlNet スタイルの UNet エンコーダ**（[[controllable-generation]]）に通し、階層的な detail map を得る。HF-map は**情報ボトルネック**として働き、細部は保ちつつ姿勢・照明・向きの局所変化を許す（collage 比較で fidelity↔diversity の最良トレードオフ、表5）。

### (3) feature injection（§3.3）

Stable Diffusion のテキスト埋め込みを **ID トークンに置換して cross-attention 注入**、detail map を**各解像度で UNet デコーダ特徴に concat**。UNet エンコーダは凍結（prior 保護）、デコーダのみ学習。損失は標準の拡散 MSE。

### (4) 学習戦略 — 動画 ＋ ATS（§3.4）

「同じ物体の異なるシーン」のペアは既存データにないので、**動画から 2 フレームを取り**、片方を対象物体、もう片方をシーン＋GT にする（図4）。14 データセット（動画・多視点画像・単一画像）で学習。動画は見た目変化に強いが低解像度、画像は高品質だが変化なし——両者を活かすため **adaptive timestep sampling（ATS）**：拡散の早期ステップは構造・姿勢を、後期ステップは細部を作るという性質を使い、**動画は大きい T（早期）、画像は小さい T（後期）**をサンプルしやすくする。

<figure>

![](../../raw/assets/2023-anydoor/x2.png)

<figcaption>図2（再掲, [[translations/2023-anydoor]] より）: AnyDoor のパイプライン。物体は背景除去→DINO-V2 ID extractor で ID トークンに、HF-map collage→ControlNet スタイル detail extractor で detail map になり、両者を Stable Diffusion の U-Net に注入してシーンの指定位置に合成する。</figcaption>
</figure>

## 実験結果と知見

- **ベンチマーク**：DreamBooth の 30 概念＋COCO の 80 シーン＝2400 枚。指標は **CLIP-Score / DINO-Score**（生成領域と対象物体の特徴類似度）と 15 人のユーザー調査（Quality/Fidelity/Diversity）。
- **vs 参照ベース**（表2）：Fidelity で AnyDoor 3.06 vs Paint-by-Example 2.10 / Graphit 2.11 と大差。Quality も最高。Diversity は「意味的一貫性しか保たない」手法が構造的に有利だが、AnyDoor は ID を保ったうえで競合的。
- **vs チューニングベース**（図6）：DreamBooth 等は新概念の忠実度は高いが「複数被写体の混乱」が出る。AnyDoor は zero-shot で複数被写体を破綻なく合成。
- **アブレーション**（表3）：CLIP ベースライン（73.8/31.5）→ +DINO-V2(Seg)（80.4/63.2）→ +HF-map（81.5/64.8）→ +ATS（82.1/67.8）と各要素が寄与。ID extractor 比較（表4）で DINO-V2(G+P)+Seg が最良、collage 比較（表5）で HF-map が好バランス。
- **応用**：virtual try-on（ボックスだけで上半身位置指定、人体パースマップ不要）、複数被写体合成、inpainting＋対話セグメンテーションと組み合わせた物体の移動・入れ替え・変形（図11）。

## 限界・批判的視点

- 本文に明示的な限界節は薄く（Conclusion のみ）、失敗例の体系的分析は手控えめ。
- **動画フレーム品質**が低い問題を ATS で緩和しているが、根本的には高品質な「同一物体・異シーン」ペアの不足に依存する。
- 「コピー&ペースト」回避（多様性）と ID 保持（忠実度）は本質的にトレードオフで、HF-map はその折衷。極端な再姿勢化や大幅な見た目変化は HF-map の制約に縛られうる。
- セグメンテーション・inpainting・対話モデルなど**外部モジュール依存**が多く、パイプラインが重い。
- 後続：参照ベースの object compositing は本手法以降も活発（IP-Adapter 等の image-prompt アダプタ系）。本 wiki 未取り込み。

## 用語と略称

- **object teleportation（物体テレポーテーション）** = 対象物体をシーンの指定位置にシームレスに配置・再生成する、本論文の課題設定。
- **ID（identity）/ ID extractor / ID tokens** = 物体の同一性とその抽出器・特徴トークン。AnyDoor は **DINO-V2**（Caron/Oquab らの自己教師あり ViT）を使う。
- **HF-map（high-frequency map, 高周波マップ）** = Sobel 高域フィルタ＋eroded mask で抽出した輪郭・テクスチャの地図。detail extractor の入力 collage。
- **detail extractor** = HF-map collage を ControlNet スタイル UNet エンコーダで処理し階層 detail map を出すモジュール。
- **ATS（Adaptive Timestep Sampling, 適応的時刻ステップサンプリング）** = データ種別ごとに拡散の時刻 T のサンプル分布を変える学習戦略（動画=早期、画像=後期）。
- **collage（コラージュ）** = 物体（または HF-map）をシーンの指定位置に貼り合わせた合成入力。
- **CLIP-Score / DINO-Score** = CLIP / DINO 特徴での生成画像と対象の類似度。**SD** = Stable Diffusion。**ControlNet** = 凍結拡散モデルに条件枝を足すアダプタ（[[controllable-generation]]）。

## 既存知識との接続

- [[image-composition]]：AnyDoor は本概念（参照物体をシーンに合成する object teleportation）の代表。Paint-by-Example・ObjectStitch と同系だが ID 保持で凌駕。
- [[subject-driven-generation]]：DreamBooth（tuning 型・テキスト駆動・新文脈生成）と対照的に、AnyDoor は zero-shot・参照ベース・位置指定。被写体/物体駆動という点では姉妹的。
- [[latent-diffusion]]：base 生成器は Stable Diffusion。U-Net エンコーダ凍結＋デコーダ学習で新タスクに適応。
- [[controllable-generation]]：detail extractor は ControlNet スタイル（凍結 UNet に条件枝を足す）。参照画像による領域制御。
- [[image-inpainting]]：AnyDoor はボックス領域を再生成する点で inpainting と隣接するが、「妥当な内容で埋める」のではなく「特定物体で埋める」。物体移動では inpainting で元位置を消す。
- [[denoising-diffusion]]：ATS は「早期ステップ＝構造/姿勢、後期＝テクスチャ/色」という拡散の時刻ステップの役割分担を利用する。

## 関連ページ

- [[concepts/image-composition]] — 画像コンポジション / object teleportation（AnyDoor が代表）
- [[concepts/subject-driven-generation]] — DreamBooth との tuning↔zero-shot 対比
- [[concepts/latent-diffusion]] / [[concepts/controllable-generation]] / [[concepts/image-inpainting]]
- [[summaries/2023-dreambooth]] — DreamBooth（評価指標 DINO/CLIP-Score の出典、tuning 型の比較対象）
- [[summaries/2023-controlnet]] — ControlNet（detail extractor のスタイル元）
