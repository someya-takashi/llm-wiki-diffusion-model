---
type: concept
aliases: [Multi-Concept Customization, 多概念カスタマイズ, Multi-LoRA Composition, LoRA Composition, LoRA Merge, LoRA Switch, LoRA Composite, ComposLoRA]
tags: [multi-concept-customization, low-rank-adaptation, subject-driven-generation, controllable-generation, generative-models, style-content-disentanglement, model-merging]
related:
  - "[[low-rank-adaptation]]"
  - "[[lora-merging]]"
  - "[[subject-driven-generation]]"
  - "[[image-composition]]"
  - "[[controllable-generation]]"
  - "[[classifier-free-guidance]]"
  - "[[character-consistency]]"
  - "[[style-content-disentanglement]]"
  - "[[model-merging]]"
summaries:
  - "[[summaries/2023-custom-diffusion]]"
  - "[[summaries/2023-mix-of-show]]"
  - "[[summaries/2024-ziplora]]"
  - "[[summaries/2024-lora-composer]]"
  - "[[summaries/2024-multi-lora-composition]]"
  - "[[summaries/2026-hidream-o1-image]]"
  - "[[summaries/2024-orthogonal-adaptation]]"
  - "[[summaries/2026-ssr-merge]]"
updated: 2026-08-19
---

# Multi-Concept Customization / Multi-LoRA Composition（多概念カスタマイズ / 複数 LoRA 合成）

**Multi-Concept Customization（多概念カスタマイズ）** とは、**複数の被写体・要素（それぞれが単一概念の [[low-rank-adaptation]] LoRA で表せる）を、各々の同一性を保ったまま 1 枚の画像に同時に生成する**タスクである。例：「自分の犬」＋「特定のアニメキャラ」＋「ある背景」を 1 枚に、あるいはキャラクター LoRA ＋ 服 LoRA ＋ 画風 LoRA を重ねる。単一概念の personalization（[[subject-driven-generation]]）が広く使われ、コミュニティに無数の LoRA が共有されるようになった結果、それらを**組み合わせる**ことが自然な次の課題になった。

このタスクと、その 2 つの基本解法（複数概念を一緒に学習する **joint training** と、各概念を別々に学習してから**重みレベルで結合する**マージ）を最初に確立した源流が **Custom Diffusion**（Kumari ら 2022・CVPR 2023・[[summaries/2023-custom-diffusion]]）である。本 wiki のランドマークは、この源流の Custom Diffusion と、後続の **LoRA-Composer**（[[summaries/2024-lora-composer]]）・**Multi-LoRA Composition**（[[summaries/2024-multi-lora-composition]]）。

## なぜ難しいのか

複数の LoRA を素朴に組み合わせると、2 つの失敗が起きやすい。

- **concept vanishing（概念消失）**：意図した概念の一部が画像に現れない。
- **concept confusion（概念混同）**：属性が被写体間で混ざる（犬の色が入れ替わる、眼鏡が別人に付く 等）。

加えて、LoRA の数が増えるほど合成が不安定になり、細部が崩れる。これらをどう抑えるかで手法が分かれる。

## 3 系統のアプローチ

### (a) 重みマージ／融合（weight merging）→ 詳細は [[lora-merging]]

複数の fine-tune 済み概念の重みを 1 つにまとめてベースモデルに差す系統。この発想自体の起点が **Custom Diffusion**（[[summaries/2023-custom-diffusion]]）の **closed-form constrained-optimization merge**——各概念で別々に学習した cross-attention の $W^k,W^v$ を、「正則化キャプションでは元モデル出力を保ち（最小二乗）、対象概念の単語は fine-tune 済み value に一致させる（制約 $WC^\top=V$）」という制約付き最小二乗で **Lagrange 乗数法により閉形式に**結合する（約 2 秒、$\hat W=W_0+\mathbf v^\top\mathbf d$）。LoRA を対象にした後続もこの「重みを最小二乗で整合させて混ぜる」発想を引き継ぐ。最も素朴な **LoRA Merge**（線形和）$W'=W_0+\sum_i w_i B_iA_i$ は identity loss（各概念が $\frac1n$ に薄まる）と signal interference（列の cosine 類似度が高いと干渉）で破綻する。これを克服する代表が、推論挙動を最小二乗で整合させる **Mix-of-Show** の gradient fusion（[[summaries/2023-mix-of-show]]）と、content+style の列単位の干渉を学習係数で最小化する **ZipLoRA**（[[summaries/2024-ziplora]]）。詳しい系統整理（LoRA Merge / gradient fusion / 学習係数マージ / LoRAHub・ED-LoRA）は **[[lora-merging]]** に委譲する。

- 利点：1 度融合すれば追加推論コストがない（推論は 1 モデル）。欠点：LoRA 数が増えると不安定化し detail が崩れやすい。Mix-of-Show は高品質生成にスケッチ／ポーズなどの**画像条件**（regionally controllable sampling）を併用し、後者の領域注入は [[summaries/2024-lora-composer]] の源流になった。ZipLoRA は被写体（content）×画風（style）の 2 LoRA 合成に特化。

**この系統には「事後に混ぜる」という共通の前提がある**が、**Orthogonal Adaptation**（[[summaries/2024-orthogonal-adaptation]]）はそこを外し、**そもそも干渉しない LoRA を学習させる**。LoRA の $\Delta\theta_i = A_i B_i^\top$ のうち $B_i$ を凍結し、全ユーザーが共有する直交行列からランダムに列を取ることで $B_i^\top B_j \approx 0$ を保証する——マージは素朴な線形和のままでよく、**1 秒未満**で済む。詳細は [[lora-merging]] の系統 (5) を参照。

同論文はこのタスクを **modular customization（モジュラー・カスタマイゼーション）** として定式化した点でも本ページに寄与する。(i) 独立したカスタマイゼーション（他人のデータに一切アクセスしない）、(ii) モジュラーな組合せ（推論時に任意の部分集合を追加学習なしに結合）、(iii) 共同合成、の 3 段階からなる。**$n$ 概念の組合せは指数的に増える**ので、Mix-of-Show のように組合せごとに融合を最適化する方式は原理的にスケールしない——という指摘が、本ページ冒頭の「LoRA の数が増えるほど不安定になる」という経験則に、計算量の側からの根拠を与えている。

ただし本手法にも上限がある。$B_i$ を $n$ 次元空間から $r$ 列取る以上、**厳密に直交できる概念数は高々 $\lfloor n/r \rfloor$**（SD v1.5 の最小次元 320・$r=20$ なら 16）。原典はこの点を定量化していない。

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

## 規模を上げると疎化が裏目に出る（2026）

本ページの (a)(b)(c) は 2〜4 概念を主戦場にしてきた。**SSR-Merge**（[[summaries/2026-ssr-merge]]）は $K$ 個（最大 21）のタスク LoRA を 1 モデルに詰める設定で、**concept vanishing が疎化ベースのマージで構造的に起きる**ことを示した。

複数被写体を 1 枚に生成させ、Grounding DINO で全部検出できた割合を成功率として測る（検出されなかった物体には類似度 0 を与える）。

| 手法 | DINO | 成功率 |
| --- | --- | --- |
| Task Arithmetic（素朴な和） | 0.4052 | 0.76 |
| TIES（刈り込み＋符号の選挙） | 0.4475 | **0.69** |
| DARE（ランダム疎化＋再スケール） | 0.5050 | **0.62** |
| SSR-Merge | **0.5704** | **0.91** |

**TIES と DARE の成功率が、素朴な和より低い。** 忠実度（DINO）は上がっているのに、**要求された概念が画像に現れない**確率が増える——疎化は衝突を避ける代わりにタスクそのものを落としている。本ページ冒頭で挙げた concept vanishing が、マージ手法の設計に起因して起きる例であり、「干渉を減らす」ことと「全概念を出す」ことが同じではないことを示している。詳細と汎用の重みマージの系譜は [[model-merging]] を参照。

## 関連手法・隣接概念

- **Custom Diffusion**（[[summaries/2023-custom-diffusion]]）：本領域の**源流**。Stable Diffusion の cross-attention の $W^k,W^v$（全体の約 5%・75MB）だけを数枚で fine-tune し、複数概念を (i) joint training か (ii) 上記の閉形式マージで合成する。LoRA-Composer/Multi-LoRA Composition が **複数の独立 LoRA を推論時に**合成するのに対し、Custom Diffusion は **概念を学習して重みに焼き込み**、必要なら閉形式で結合する学習型である。似カテゴリ（cat+dog）や 3 概念以上は苦手で、これが後続の動機になった。
- **Cones / Cones2**：概念ニューロンを見つける／追加する学習型の多概念手法（本 wiki 未取り込み）。
- **AnyDoor・Paint-by-Example**（[[image-composition]]）：画像ベースの inpainting で複数物体を合成する別系統（LoRA を使わず参照画像で挿入）。
- 評価は CLIP 画像／テキスト類似度、ユーザー調査、GPT-4V など。標準指標が未確立な新しい領域。

## 参照数のスケーラビリティ（2026）

本ページの (a)(b)(c) はいずれも「2〜3 概念を破綻なく 1 枚に載せる」ことを主戦場にしてきた。**HiDream-O1-Image**（[[summaries/2026-hidream-o1-image]]）は、その先——**参照が 10 個近くになったとき何が起きるか**——を測る UniSubject を提示した。1 人の人物被写体に 1〜10 個の参照物体（衣服・車・家具など）を組み合わせた 300 ケース・計 1,800 被写体で、Qwen-VL2.5-72B に Prompt Following（Q-PF）と Subject Consistency（Q-SC）を採点させる。

| 参照数 | Qwen-Image-Edit (27B) | HiDream-O1-Image (8B) |
| --- | --- | --- |
| 2–3 | 7.50 | 7.95 |
| 4–8 | 5.34 | 7.47 |
| 9–11 | **2.71** | **7.65** |

Qwen-Image-Edit の崩壊の仕方が示唆的である。本ページ冒頭で挙げた **concept vanishing（概念の消失）と concept confusion（概念の混同）** が、参照数の増加に対して**線形ではなく崖のように**現れることを意味する。著者らの説明は「分断されたエンコーダ設計では指示と複数の参照画像が互いに干渉するが、共有トークン空間では意味的概念が対応する視覚トークンへ精密に錨づけられる」——つまり (a)(b)(c) のように**合成の段階で干渉を捌く**のではなく、**そもそも同じ表現空間に置いて干渉を起こさない**という第 4 の立場になる。

ただし UniSubject は著者らの自作ベンチマークで、公開の有無も明記されていない。採点も VLM 依存であり、この崖が本当に一般的な現象かは独立検証を待つ必要がある。

## 既存知識との接続

- [[low-rank-adaptation]]：合成対象の単一概念が LoRA で表される。LoRA がプラグ&プレイで共有可能だからこそ「複数を合成する」課題が成立する。
- [[lora-merging]]：重みマージ／融合系統（Mix-of-Show・ZipLoRA）を細粒度に扱う子ページ。本ページの (a) を詳説する。
- [[model-merging]]：$K$ 個のタスクを 1 モデルに詰める汎用の重みマージ。本ページが「1 枚の画像に複数概念を載せる」のに対し、あちらは「1 つのモデルに複数能力を載せる」。SSR-Merge の RQ2 で両者が交差する。
- [[style-content-disentanglement]]：概念が「被写体 × 画風」の 2 成分に限られる特殊ケース。任意個の被写体へ広げると本ページになる。複数スタイルの合成は両ページの未解決の交差点。
- [[subject-driven-generation]]：単一概念 personalization の多概念版。
- [[classifier-free-guidance]]：LoRA Composite は CFG のスコア平均を多 LoRA に拡張したもの。
- [[controllable-generation]]：LoRA-Composer のレイアウト＋注意制御、Multi-LoRA の復号制御は、いずれも推論時の制御。
- [[image-composition]]：AnyDoor 系の画像ベース多概念合成は、LoRA ベースとは別系統の解。
- [[character-consistency]]：合成した各概念が**ターンをまたいで**同一に保たれるかという軸。本ページが「1 枚に複数概念を同時に載せる」空間方向の合成なら、あちらは「同じ概念を何枚にもわたって保つ」時間方向の一貫性で、FLUX.1 Kontext（[[summaries/2025-flux-kontext]]）が扱う。

## 参考文献（summaries）

- [[summaries/2026-hidream-o1-image]] — HiDream-O1-Image（UniSubject で 1〜10 参照物体の合成を評価。共有トークン空間で干渉を抑えるという第 4 の立場）

- [[summaries/2023-custom-diffusion]] — Custom Diffusion（cross-attention K/V 限定 fine-tune＋閉形式マージ。多概念カスタマイズの源流）
- [[summaries/2023-mix-of-show]] — Mix-of-Show（ED-LoRA＋gradient fusion、重みマージ系の代表）
- [[summaries/2024-ziplora]] — ZipLoRA（content+style の学習係数マージ）
- [[summaries/2024-lora-composer]] — LoRA-Composer（訓練不要・注意制御による多概念合成）
- [[summaries/2024-multi-lora-composition]] — Multi-LoRA Composition（LoRA Switch / Composite、decoding-centric）
- [[summaries/2024-orthogonal-adaptation]] — Orthogonal Adaptation（modular customization の定式化、crosstalk、$B$ 凍結＋共有直交基底で素朴な線形和を安全にする）
- [[summaries/2026-ssr-merge]] — SSR-Merge（$K$ 最大 21 のマージ。**疎化ベースの TIES・DARE が素朴な和より成功率を落とす**ことを実測し、concept vanishing がマージ設計に起因することを示した）
