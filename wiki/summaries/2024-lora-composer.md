---
type: summary
source_path: raw/papers/LoRA-Composer_ Leveraging Low-Rank Adaptation for Multi-Concept Customization in Training-Free Diffusion Models.md
source_kind: paper
title: "LoRA-Composer: Leveraging Low-Rank Adaptation for Multi-Concept Customization in Training-Free Diffusion Models"
authors: [Yang Yang, Wen Wang, Liang Peng, Chaotian Song, Yao Chen, Hengjia Li, Xiaolong Yang, Qinglin Lu, Deng Cai, Boxi Wu, Wei Liu]
year: 2024
venue: arXiv 2403.11627
ingested: 2026-06-24
tags: [multi-concept-customization, low-rank-adaptation, subject-driven-generation, controllable-generation, generative-models]
translation: "[[translations/2024-lora-composer]]"
---

# LoRA-Composer: 訓練不要の多概念カスタマイズ

> 原典: [[translations/2024-lora-composer]] ・ `raw/papers/LoRA-Composer_ ...md`
> 著者・年: Yang ら・2024（arXiv 2403.11627）

## 一言まとめ

**複数の単一概念 LoRA を、再学習なし（training-free）で 1 枚の画像に合成する**枠組み。LoRA 融合行列を訓練する従来法（Mix-of-Show）の 2 大欠陥——**concept vanishing（概念が出ない）** と **concept confusion（属性が混ざる）**——を、U-Net の **cross-attention での領域別 LoRA 注入**と **self-attention での概念分離**、そして **潜在再初期化**で解決する。入力はレイアウト＋テキストのみで、スケッチ／ポーズ等の画像条件が不要。

## 背景と問題意識

**多概念カスタマイズ（[[multi-concept-customization]]）** とは、「自分の犬」「特定のアニメキャラ」「ある背景」のような複数の被写体（それぞれ単一概念 [[low-rank-adaptation]] LoRA で表せる）を、1 枚の画像に同時に・各々の同一性を保って描くタスク。素朴な解は複数 LoRA の重みを**融合行列で混ぜる**（Mix-of-Show の gradient fusion）ことだが、

- **concept vanishing**：意図した概念が画像に現れない（[[summaries/2024-lora-composer]] 図1 左）。
- **concept confusion**：属性が被写体間で混ざる（犬の色違い、眼鏡が別人に付く 等、図1 右）。
- さらに Mix-of-Show は組合せごとに**再訓練**が必要で、高品質生成にスケッチ／ポーズ等の**画像条件**を要する。

LoRA-Composer は「融合行列を訓練せず、推論時の注意操作だけで複数 LoRA を統合する」訓練不要アプローチでこれを解く。

## 提案手法 / 主張

事前学習済み Stable Diffusion（[[latent-diffusion]]）の U-Net 注意ブロックを **LoRA-Composer Block** に置き換える（図2）。単一概念 LoRA は ED-LoRA（Mix-of-Show 由来、$r=4$）を使う。3 つの構成要素：

### (1) Concept Injection Constraints（cross-attention）— concept vanishing 対策

- **Region-Aware LoRA Injection**：レイアウトマスク $M_n$ で領域ごとに、その概念の LoRA を組み込んだ射影で cross-attention を計算（$Q_n=M_n\odot W^Q_0(z)$, $K_n,V_n$ は概念 LoRA・局所プロンプト由来）。各概念を指定領域に注入する。融合ファインチューン不要。
- **Concept Enhancement Constraints**：cross-attention の高応答を box 内に閉じ込める損失 $\mathcal{L}_{ce}$（ガウス重み付き）＋ box 全体を満たす $\mathcal{L}_{fill}$。

### (2) Concept Isolation Constraints（self-attention）— concept confusion 対策

- **Concept Region Mask**：self-attention で概念領域間の query 相互作用を制限し、特徴の混ざりを防ぐ。
- **Region Perceptual Restriction**：前景・背景間の相互作用を最小化する損失 $\mathcal{L}_{region}$（down-sampling による特徴漏れを抑える）。

総損失 $\mathcal{L}=\mathcal{L}_{ce}+\alpha\mathcal{L}_{fill}+\beta\mathcal{L}_{region}$ で潜在を各ステップ更新（$z'_t\leftarrow z_t-\phi_t\nabla\mathcal{L}$、BoxDiff 流に $\phi_t$ 減衰）。

### (3) Latent Re-initialization

単一概念 LoRA は画像全体で訓練されるため局所領域生成に不向き。生成前に潜在を 1 ステップ更新→ cross-attention マップの最高スコア領域でマスクを置き直し→標準ガウスに正規化して、局所生成に適した空間 prior を作る。

<figure>

![](../../raw/assets/2024-lora-composer/x2.png)

<figcaption>図2（再掲, [[translations/2024-lora-composer]] より）: LoRA-Composer のパイプライン。(b) U-Net の self-attention に Concept Isolation、cross-attention に Concept Injection を入れ、概念間の漏れを防ぎつつ正確に配置する。</figcaption>
</figure>

## 実験結果と知見

- **データ**：写実・アニメ両スタイルの 16 被写体、2〜5 概念の組合せ。指標は CLIP の画像類似度（I）・テキスト類似度（T）。
- **定量**（表1）：画像類似度で全ベースライン（Cones2・Mix-of-Show・AnyDoor・Paint-by-Example）を上回る（Mean-I 0.779 vs Mix-of-Show 0.652）。**画像条件なし**でも高性能で、Mix-of-Show は画像条件がないと大きく落ちるのに対し頑健。
- **定性**（図5）：Cones2/Mix-of-Show は concept confusion・vanishing、inpainting 系（AnyDoor 等）は顔の細部が苦手。LoRA-Composer は全被写体を正しい特徴で合成。
- **アブレーション**（表2）：CE が概念活性化に最も寄与（Mean-I 約 +0.1）、CI が区別性、LR が配置精度。
- **ユーザー調査**（表3）：Text/Image 整合の両方で最高評価。

## 限界・批判的視点

- **概念境界の消失**：概念間の間隔が狭いと down-sampling で重なる。
- **レイアウト境界の超過**：前景画素がボックスを超えうる（SD の常識的生成のため）。
- **推論効率**：複数 LoRA チェックポイントの読み込みと潜在更新の逆計算で遅延。
- 各概念に**事前学習済み単一概念 LoRA とレイアウト（ボックス）が必要**。
- 同時期の [[summaries/2024-multi-lora-composition]]（Multi-LoRA Composition の LoRA Switch/Composite）とはアプローチが対照的：あちらは前景 1＋背景 1 の合成を復号過程で扱うのに対し、LoRA-Composer は複数前景キャラを注意制御で扱う。

## 用語と略称

- **multi-concept customization（多概念カスタマイズ）** = 複数の被写体／概念を 1 枚に同時生成するタスク。
- **concept vanishing / concept confusion** = 概念が出ない／属性が混ざる、という多概念合成の 2 大失敗。
- **LoRA** = Low-Rank Adaptation（[[low-rank-adaptation]]）。**ED-LoRA** = Mix-of-Show の層ごと分解埋め込み＋LoRA。
- **Mix-of-Show** = LoRA 融合行列を gradient fusion で訓練する先行手法（比較対象）。
- **Region-Aware LoRA Injection / Concept Injection・Isolation Constraints / Latent Re-initialization** = 本手法の構成要素。
- **training-free** = 合成のための追加学習をしない（推論時の注意操作のみ）。

## 既存知識との接続

- [[multi-concept-customization]]：本論文は「訓練不要・注意制御」系統の代表。
- [[low-rank-adaptation]]：合成対象の単一概念表現に LoRA（ED-LoRA）を使う。LoRA がプラグ&プレイで共有可能だからこそ成立。
- [[subject-driven-generation]]：単一概念の personalization（DreamBooth/LoRA）の多概念版。
- [[controllable-generation]]：レイアウト（ボックス）条件と注意操作による推論時制御。BoxDiff・Attend-and-Excite 系の流れ。
- [[latent-diffusion]]：Stable Diffusion の U-Net を凍結したまま注意を操作する。
- [[image-composition]]：AnyDoor/Paint-by-Example（画像ベース inpainting で多概念）を比較対象に含む。

## 関連ページ

- [[concepts/multi-concept-customization]] — 多概念カスタマイズ（本論文が代表）
- [[concepts/low-rank-adaptation]] — LoRA（合成対象の単一概念表現）
- [[summaries/2024-multi-lora-composition]] — Multi-LoRA Composition（同時期・対照的な decoding-centric 合成）
- [[summaries/2023-dreambooth]] — DreamBooth（単一概念 personalization）
