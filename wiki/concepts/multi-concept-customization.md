---
type: concept
aliases: [Multi-Concept Customization, 多概念カスタマイズ, Multi-LoRA Composition, LoRA Composition, LoRA Merge, LoRA Switch, LoRA Composite, ComposLoRA]
tags: [multi-concept-customization, low-rank-adaptation, subject-driven-generation, controllable-generation, generative-models]
related:
  - "[[low-rank-adaptation]]"
  - "[[lora-merging]]"
  - "[[subject-driven-generation]]"
  - "[[image-composition]]"
  - "[[controllable-generation]]"
  - "[[classifier-free-guidance]]"
summaries:
  - "[[summaries/2023-mix-of-show]]"
  - "[[summaries/2024-ziplora]]"
  - "[[summaries/2024-lora-composer]]"
  - "[[summaries/2024-multi-lora-composition]]"
updated: 2026-06-24
---

# Multi-Concept Customization / Multi-LoRA Composition（多概念カスタマイズ / 複数 LoRA 合成）

**Multi-Concept Customization（多概念カスタマイズ）** とは、**複数の被写体・要素（それぞれが単一概念の [[low-rank-adaptation]] LoRA で表せる）を、各々の同一性を保ったまま 1 枚の画像に同時に生成する**タスクである。例：「自分の犬」＋「特定のアニメキャラ」＋「ある背景」を 1 枚に、あるいはキャラクター LoRA ＋ 服 LoRA ＋ 画風 LoRA を重ねる。単一概念の personalization（[[subject-driven-generation]]）が広く使われ、コミュニティに無数の LoRA が共有されるようになった結果、それらを**組み合わせる**ことが自然な次の課題になった。本 wiki のランドマークは **LoRA-Composer**（[[summaries/2024-lora-composer]]）と **Multi-LoRA Composition**（[[summaries/2024-multi-lora-composition]]）。

## なぜ難しいのか

複数の LoRA を素朴に組み合わせると、2 つの失敗が起きやすい。

- **concept vanishing（概念消失）**：意図した概念の一部が画像に現れない。
- **concept confusion（概念混同）**：属性が被写体間で混ざる（犬の色が入れ替わる、眼鏡が別人に付く 等）。

加えて、LoRA の数が増えるほど合成が不安定になり、細部が崩れる。これらをどう抑えるかで手法が分かれる。

## 3 系統のアプローチ

### (a) 重みマージ／融合（weight merging）→ 詳細は [[lora-merging]]

複数 LoRA の重みを 1 つにまとめてベースモデルに差す系統。最も素朴な **LoRA Merge**（線形和）$W'=W_0+\sum_i w_i B_iA_i$ は identity loss（各概念が $\frac1n$ に薄まる）と signal interference（列の cosine 類似度が高いと干渉）で破綻する。これを克服する代表が、推論挙動を最小二乗で整合させる **Mix-of-Show** の gradient fusion（[[summaries/2023-mix-of-show]]）と、content+style の列単位の干渉を学習係数で最小化する **ZipLoRA**（[[summaries/2024-ziplora]]）。詳しい系統整理（LoRA Merge / gradient fusion / 学習係数マージ / LoRAHub・ED-LoRA）は **[[lora-merging]]** に委譲する。

- 利点：1 度融合すれば追加推論コストがない（推論は 1 モデル）。欠点：LoRA 数が増えると不安定化し detail が崩れやすい。Mix-of-Show は高品質生成にスケッチ／ポーズなどの**画像条件**（regionally controllable sampling）を併用し、後者の領域注入は [[summaries/2024-lora-composer]] の源流になった。ZipLoRA は被写体（content）×画風（style）の 2 LoRA 合成に特化。

### (b) 訓練不要の注意制御（training-free, attention control）

重みを混ぜず、**推論時に U-Net の注意を領域ごとに操作**して各概念を所定の場所に注入・分離する。代表が **LoRA-Composer**（[[summaries/2024-lora-composer]]）：

- cross-attention に **Region-Aware LoRA Injection**（レイアウトマスクで領域別に概念 LoRA を注入）＋ Concept Enhancement（box 内に応答を集中）で **concept vanishing** を抑える。
- self-attention に **Concept Isolation**（概念領域マスクで相互作用を制限）で **concept confusion** を抑える。
- **Latent Re-initialization** で局所生成に適した空間 prior を作る。
- レイアウト＋テキストだけで動き、複数前景キャラを扱える。融合訓練も画像条件も不要。

### (c) 復号過程での合成（decoding-centric）

LoRA 重みを一切いじらず、**ノイズ除去（復号）の各ステップ**で合成する。代表が **Multi-LoRA Composition**（[[summaries/2024-multi-lora-composition]]）の 2 手法：

- **LoRA Switch (LoRA-s)**：各 denoise ステップで 1 つの LoRA だけを有効化し、$\tau$ ステップごとに順番に切替（character→clothing→style→…）。
- **LoRA Composite (LoRA-c)**：各ステップで全 LoRA の条件付き／無条件スコアを個別に計算し平均してガイダンスにする（[[classifier-free-guidance]] の多 LoRA 拡張）。
- どちらも任意個の LoRA を統合でき、重み操作の不安定さを回避。評価用に **ComposLoRA** テストベッドと GPT-4V 評価を提案。LoRA 数が増えるほど LoRA Merge への優位が拡大（LoRA-s は composition 品質、LoRA-c は image 品質で勝る）。

## 関連手法・隣接概念

- **Custom Diffusion・Cones / Cones2**：複数概念を joint fine-tune したり概念ニューロンを見つける学習型の多概念手法。
- **AnyDoor・Paint-by-Example**（[[image-composition]]）：画像ベースの inpainting で複数物体を合成する別系統（LoRA を使わず参照画像で挿入）。
- 評価は CLIP 画像／テキスト類似度、ユーザー調査、GPT-4V など。標準指標が未確立な新しい領域。

## 既存知識との接続

- [[low-rank-adaptation]]：合成対象の単一概念が LoRA で表される。LoRA がプラグ&プレイで共有可能だからこそ「複数を合成する」課題が成立する。
- [[lora-merging]]：重みマージ／融合系統（Mix-of-Show・ZipLoRA）を細粒度に扱う子ページ。本ページの (a) を詳説する。
- [[subject-driven-generation]]：単一概念 personalization の多概念版。
- [[classifier-free-guidance]]：LoRA Composite は CFG のスコア平均を多 LoRA に拡張したもの。
- [[controllable-generation]]：LoRA-Composer のレイアウト＋注意制御、Multi-LoRA の復号制御は、いずれも推論時の制御。
- [[image-composition]]：AnyDoor 系の画像ベース多概念合成は、LoRA ベースとは別系統の解。

## 参考文献（summaries）

- [[summaries/2023-mix-of-show]] — Mix-of-Show（ED-LoRA＋gradient fusion、重みマージ系の代表）
- [[summaries/2024-ziplora]] — ZipLoRA（content+style の学習係数マージ）
- [[summaries/2024-lora-composer]] — LoRA-Composer（訓練不要・注意制御による多概念合成）
- [[summaries/2024-multi-lora-composition]] — Multi-LoRA Composition（LoRA Switch / Composite、decoding-centric）
