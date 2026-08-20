---
type: question
asked: 2026-08-21
question: "Z-Image のモデルアーキテクチャの図解と、LoRA でファインチューニングする場合にどこへ LoRA を付けるべきかのアドバイス"
summaries_used:
  - "[[summaries/2025-z-image]]"
  - "[[summaries/2024-b-lora]]"
  - "[[summaries/2023-custom-diffusion]]"
  - "[[summaries/2024-orthogonal-adaptation]]"
  - "[[summaries/2025-k-lora]]"
  - "[[summaries/2025-np-lora]]"
  - "[[summaries/2026-hidream-o1-image]]"
  - "[[summaries/2025-wan]]"
---

# Z-Image のアーキテクチャと LoRA の当てどころ

> 質問: Z-Image のモデルアーキテクチャの図解と、LoRA でファインチューニングを行う場合どこに LoRA を付けるべきか
> 回答日: 2026-08-21
> 出典: [[summaries/2025-z-image]] ・ [[translations/2025-z-image]] ・ [[concepts/diffusion-model-architecture]]

> ⚠️ **この回答の性格**: 本 wiki は **Z-Image に LoRA を当てた実験の原典を持っていない**。以下は S3-DiT の構造（原典の図 10・表 2）と、他モデルで確立した知見からの**演繹**である。断定できる部分と推測の部分を区別して書いた。

## 1. アーキテクチャ

### 原典の図

<figure>

![](../../raw/assets/2025-z-image/architecture.png)

<figcaption>図10（引用, [[summaries/2025-z-image]] より）: S3-DiT の全体像。左下の各プロセッサ（Semantic / Image / Text / Timestep）を通ったトークンが 1 本の系列に連結され、以降は共有重みの Single-Stream ブロックを通る。右は Attention / FFN ブロックの内部。</figcaption>
</figure>

### 構成（原典 表2）

| 構成 | 値 |
| --- | --- |
| 総パラメータ数 | 6.15B |
| 層数 | 30 |
| 隠れ次元 | 3840 |
| 注意ヘッド数 | 32 |
| FFN 中間次元 | 10240 |
| U-RoPE の次元配分 $(d_t,d_h,d_w)$ | $(32,48,48)$ |

> **訳注**: 表の値をそのまま取ると head_dim $=3840/32=120$ だが、U-RoPE の次元配分は $32+48+48=128$ に足し合わさる。実装では head_dim$=128$（→ 30 ヘッド）か、RoPE を部分次元にのみ適用しているかのいずれかと思われる。**LoRA を当てるだけなら影響しないが、独自実装で RoPE を触るなら実チェックポイントで確認したほうがよい。**

### データフローの図解

```
入力                    モダリティ固有プロセッサ            単一ストリーム・バックボーン
                         （各 2 transformer ブロック）        （共有重み ×30）
────────────────       ──────────────────────       ────────────────────────

SigLip-2 ─────▶ [Semantic Processor] ─┐   ★C
（編集時のみ）                          │
                                       │
Flux VAE ─────▶ [Image Processor] ─────┤   ★C     ┌────────────────────────┐
（ノイズ付き潜在）        ▲             │           │ Single-Stream          │
                          │             ├──連結──▶ │   Attention Block  ★A  │
Qwen3-4B ─────▶ [Text Processor] ──────┘           ├────────────────────────┤ ×30
（凍結・★F）              │           [img…][txt…]  │ Single-Stream          │
                          │           [sem…]        │   FFN Block        ★B  │
timestep ─────▶ [Embed] ──┘                         └────────────────────────┘
                          │                                     │
                          │  条件ベクトル                        ▼
                          │  ┌──────────────────┐      [Output Projection]
                          └─▶│ 下方射影（全層共有）│ ★E            │
                             │        ↓          │               ▼
                             │ 上方射影（層ごと）  │ ★D    Predicted Velocity
                             └────────┬─────────┘         （rectified flow）
                                      │
                          Scale / Gate を全ブロックへ
```

### ブロック内部（Sandwich-Norm 構造）

```
  Single-Stream Attention Block              Single-Stream FFN Block
  ─────────────────────────────              ───────────────────────
    x ──┬───────────────────────┐              x ──┬──────────────────┐
        ▼                       │                  ▼                  │
    [RMS Norm]                  │              [RMS Norm]             │
        ▼                       │                  ▼                  │
    [Scale] ◀── 条件            │              [Scale] ◀── 条件       │
        ▼                       │                  ▼                  │
    [Wq] [Wk] [Wv]      ★A      │              [Feed Forward]  ★B     │
        ▼                       │              （3840 → 10240 → 3840）│
    [Q-Norm] [K-Norm]           │                  ▼                  │
        ▼                       │              [RMS Norm]             │
    ⊗ U-RoPE（Q と K に）        │                  ▼                  │
        ▼                       │              [Zero-init Gate]◀─ 条件 │
    [Multi-head Self-Attention] │                  ▼                  │
        ▼  （+ Wo ★A）          │                  ⊕ ◀────────────────┘
    [RMS Norm]                  │
        ▼                       │              ★ = LoRA の貼り付け候補
    [Zero-init Gate] ◀── 条件   │
        ▼                       │
        ⊕ ◀────────────────────┘
```

**読み取るべき 3 点**

1. **Attention ブロックと FFN ブロックが別々**に積まれている（1 つの transformer ブロック内に両方があるのではない）。
2. **入口のプロセッサを除き、テキスト・画像・意味トークンはすべて同じ重みを通る**（[[concepts/diffusion-model-architecture]] の「完全な単一ストリーム」）。
3. **条件注入は低ランクの対に分解され、下方射影は全 30 層で共有、上方射影だけが層ごと**。Wan が adaLN の MLP を全層共有した（[[summaries/2025-wan]]）のと同じ省パラメータの方向。

## 2. Z-Image で「使えなくなる」定番レシピ 2 つ

LoRA の当てどころを考える前に、**他モデルで定番の 2 つの処方箋が S3-DiT では構造的に成立しない**ことを押さえておく必要がある。

### ✗ 「cross-attention の K/V だけ」が存在しない

**Custom Diffusion**（[[summaries/2023-custom-diffusion]]）は cross-attention の $W^k, W^v$ のみを微調整する。これが効くのは、cross-attention において **K と V がテキスト由来**だからである。

S3-DiT には **cross-attention がない**。連結された 1 本の系列に対する self-attention だけで、$W_k$ と $W_v$ はテキストトークンも画像トークンも同じ重みで処理する。**「テキスト条件の入り口だけを触る」ことが原理的にできない。**

### ✗ 「ブロック 4 と 5 だけ」が使えない

**B-LoRA**（[[summaries/2024-b-lora]]）は SDXL の第 4 ブロックがコンテンツ、第 5 ブロックが色を支配することを見つけ、その 2 つだけを学習する。

しかし [[concepts/style-content-disentanglement]] に記録したとおり、これは **SDXL の UNet（ダウン／ミドル／アップという非対称構造）に固有の経験則**であり、**DiT のような均質な Transformer 積層で同じ役割分化が現れるかは未検証**である。S3-DiT はまさにその均質な積層で、しかも 30 層が完全に同型である。

**→ Z-Image で「特定の層だけ」をやりたいなら、B-LoRA の手順を自分で回して層の役割を測るしかない**（後述 §5）。

## 3. LoRA の貼り付け先：候補と評価

| | 対象 | 箇所数 | 何が変わるか | 推奨度 |
| --- | --- | --- | --- | --- |
| **★A** | 単一ストリーム Attention の $W_q,W_k,W_v,W_o$ | 30 層 | トークン間の関係の作り方。**最も実績のある標準的な位置** | ★★★ **第一選択** |
| **★B** | 単一ストリーム FFN（3840→10240→3840） | 30 層 | トークンごとの特徴変換。容量が大きく**質感・画風に効きやすい** | ★★☆ A で足りなければ追加 |
| **★C** | 入口の Image Processor（2 ブロック） | 2 箇所 | **唯一モダリティを分離できる場所**。画像表現だけに触れる | ★★☆ 被写体特化なら試す価値 |
| **★D** | 条件注入の**層ごと**上方射影 | 30 層 | 各ブロックの Scale / Gate ＝**寄与量**を層ごとに変調 | ★☆☆ 実験的 |
| **★E** | 条件注入の**共有**下方射影 | 1 箇所 | 全 30 層に一斉に効く。パラメータ極小 | ☆☆☆ 破綻しやすい |
| **★F** | Text Processor / Qwen3-4B | — | PE-aware SFT で作った整合を壊しうる（§4） | 非推奨 |
| **★G** | Flux VAE | — | 凍結すべき（[[concepts/image-tokenizer]]） | 非推奨 |

### ★A を第一選択にする理由

[[concepts/low-rank-adaptation]] にまとめたとおり、拡散モデルの LoRA は伝統的に注意・線形層に当てるのが既定で、**Orthogonal Adaptation**（[[summaries/2024-orthogonal-adaptation]]）も「Stable Diffusion アーキテクチャ内のすべての線形層」に適用している。S3-DiT は均質な積層なので、**層を選ぶ根拠が現時点で存在しない以上、全層に薄く当てるのが最も安全**である。

### ★C が Z-Image では特別な意味を持つ

単一ストリームであることの裏返しとして、**バックボーンに当てた LoRA はテキストトークンの表現も必ず動かす**。画風 LoRA がテキスト理解を副作用で歪める、という事故が起こりうる。

**Image Processor（2 ブロック）だけがモダリティ的に隔離された場所**である。「被写体の見た目だけを教えたい、プロンプト追従は一切触りたくない」という要求には、ここが構造的に正しい当てどころになる。ただし 2 ブロックしかないので容量は小さく、**単独で十分かは未検証**。

### ★D / ★E は Z-Image 固有の（そして危険な）標的

条件注入の低ランク分解は Z-Image の設計上の特徴で、標準的な LoRA レシピには対応物がない。

- **★D（層ごと上方射影）**: 各ブロックの寄与の**大きさ**を層ごとに変えられる。「モデルの計算内容は変えず、どの層をどれだけ効かせるか」だけを学習する、という珍しい介入になる。
- **★E（共有下方射影）**: **1 箇所いじると 30 層すべての変調が変わる**。パラメータ効率は極端に良いが、レバーが長すぎて制御が難しい。「モデル全体の雰囲気をわずかに動かす」用途の実験としては面白いが、被写体学習には向かない。

## 4. Z-Image 固有の注意点 5 つ

### (1) Sandwich-Norm と Zero-init Gate が振幅を握っている

ブロック出力は **RMSNorm を通ってからゲートで倍率が決まる**。つまり LoRA でブロック内部の重みを動かしても、出力の**「向き」は変わるが「大きさ」はゲートが決める**。

**実務上の含意**: 効きが弱いと感じたときに rank や学習率を上げる前に、★D（ゲート側）も候補に入れる価値がある。逆に、**学習済みのゲート値が小さい層に LoRA を当てても効きにくい**はずである。

### (2) 学習目的は velocity 予測、タイムステップ分布は変える価値がある

Z-Image は rectified flow で、出力は **Predicted Velocity**（DDPM の $\epsilon$ 予測ではない）。

事前学習は logit-normal の時刻サンプリングだが、[[concepts/noise-schedule]] に記録した **HiDream-O1-Image の知見**（[[summaries/2026-hidream-o1-image]]）が直接効く——**SFT の段階では logit-normal を一様サンプリングに切り替える**。理由は、美的品質・写実性・細部が決まるのは $t$ が小さい後期のノイズ除去ステップであり、logit-normal は中間時刻に予算を集中させてしまうからである。

**被写体の細部（ロゴ・質感・エッジ）を教える LoRA なら、一様サンプリング寄りに倒すのは理にかなう。**

### (3) ベース版で学習する（Turbo ではなく）

Z-Image-Turbo は Decoupled DMD による 8 NFE の蒸留版である。[[concepts/diffusion-distillation]] の既知の限界がそのまま出ており、**OneIG の Diversity がベース 0.194 → Turbo 0.139 まで落ちる**（原典はこの悪化を議論していない）。

蒸留済みモデルの上に LoRA を積むと、**蒸留で作られた少ステップの軌道と LoRA の変更が干渉しうる**。ベース版で学習し、必要なら Turbo へ移して確認するのが安全である。

### (4) キャプションをプロンプトエンハンサの出力分布に合わせる

**これが最も見落とされやすい。**

Z-Image は SFT の段階で**全入力プロンプトをプロンプトエンハンサ（PE）に通し、拡散側を PE の出力分布に合わせている**（[[concepts/prompt-enhancement]] の「Z-Image — 逆向きに合わせる」）。つまりモデルが期待する入力は、**PE が吐く形の詳細で構造化されたプロンプト**である。

LoRA の学習キャプションを短いユーザー風の文（"a photo of sks phone"）で書くと**分布がずれる**。推論時に PE を通すなら、**学習キャプションも PE に通してから使う**か、少なくとも PE が出すのと同じ粒度・構文で書くべきである。

### (5) 元から多様性が低いので過学習が早い

Z-Image は SFT で「**多様性最大化の体制から品質最大化の動作点へ移す**」と明言しており、OneIG-EN の Diversity 0.194 は比較対象中でもかなり低い（SD 1.5 が 0.429）。

LoRA は分布をさらに狭める方向に働くので、**過学習の兆候が出るのが早い**と見込んでおく。学習ステップは短めから、適用強度も控えめから始めるのが無難である。

> 補足: リリースされている重みは SFT の仕上げに**能力次元ごとに偏らせた複数の変種を線形補間したモデルマージの産物**である（[[concepts/model-merging]]）。$\Delta W$ の構造が単一の学習軌道の産物ではない点は、後で複数 LoRA をマージする際に頭の片隅に置いておく価値がある。

## 5. 「どの層が効くか」を自分で測る 2 つの方法

均質な積層なので層選択の既成レシピがない。測るなら 2 通りある。

### (a) プロンプト注入解析（B-LoRA の手順を移植）

[[summaries/2024-b-lora]] の §4.1 がそのまま使える。

1. 30 層のうち $i$ 番目のブロックにだけ**違うテキストプロンプト**を注入し、他の層には元のプロンプトを入れる
2. 生成画像とその違うプロンプトの **CLIP 類似度**を測る
3. 多数のプロンプト対（B-LoRA は 400 対）で平均する

「その層を変えると出力の何が変わるか」が層ごとに出る。B-LoRA は物体を変えるプロンプト対と色を変えるプロンプト対を用意して**コンテンツ担当層と色担当層を分離**した。同じことを Z-Image でやれば、**均質な積層でも役割分化が起きているのか**という未解決の問いにも答えが出る。

### (b) 学習済みゲートのノルムを見る

Zero-init Gate は学習で非ゼロになっているはずだが、**その大きさは層ごとに違う**と考えられる。チェックポイントから層ごとのゲートのノルムを出して大きい層を優先する、という選び方ができる。実装コストがほぼゼロなので、(a) の前に試す価値がある。

> どちらも本 wiki に Z-Image での実施例はない。**やれば新規の知見になる**。

## 6. 出発点として推奨するレシピ

```
対象:   全 30 層の Single-Stream Attention の Wq / Wk / Wv / Wo   （★A）
rank:   16〜32                （隠れ次元 3840 に対して）
alpha:  rank と同じ 〜 2 倍
ベース: Z-Image ベース版（Turbo ではない）
目的:   velocity 予測（rectified flow）
時刻:   細部重視なら一様サンプリング寄りに
data:   キャプションを PE の出力分布に合わせる
```

**足りないときの追加順**

1. **画風・質感が乗らない** → ★B（FFN）を追加。rank は A より下げてよい（中間次元 10240 で容量が大きい）
2. **プロンプト追従が壊れた** → ★A/B を外し、★C（Image Processor の 2 ブロック）だけで試す
3. **どの層が効いているか知りたい** → §5 の測定へ

## 7. 後で複数 LoRA を合成する予定があるなら

前景／背景や被写体×画風で**複数の LoRA を作って後で混ぜる**計画なら、[[questions/lora-foreground-background-composition]] を参照。Z-Image に関して 1 点だけ補足する。

**K-LoRA（[[summaries/2025-k-lora]]）と NP-LoRA（[[summaries/2025-np-lora]]）は、いずれも SDXL に加えて FLUX でも検証されている。** NP-LoRA は「射影ベースの設計が異なる拡散バックボーンにわたってよく汎化する」と明言している。FLUX は Z-Image と同じ DiT 系なので、**SDXL の UNet に固有だった B-LoRA より移植の見込みは高い**。

両手法とも**重みのレベル**で動作し（K-LoRA は層ごとの Top-K 和の比較、NP-LoRA は $\Delta W$ の SVD と零空間射影）、バックボーンの構造に依存する仮定を置いていない点も有利に働く。

ただし——**単一ストリームでは LoRA がテキストトークンにも作用する**という §2 の事実は融合時にも効く。2 つの LoRA を混ぜると、両方のテキスト側への副作用も混ざる。SDXL や MM-DiT での検証結果がそのまま当てはまるとは限らない。

## 関連ページ

- [[summaries/2025-z-image]]
- [[translations/2025-z-image]]
- [[concepts/diffusion-model-architecture]]
- [[concepts/low-rank-adaptation]]
- [[concepts/style-content-disentanglement]]
- [[concepts/prompt-enhancement]]
- [[concepts/noise-schedule]]
- [[concepts/diffusion-distillation]]
- [[concepts/position-embedding]]
- [[questions/lora-foreground-background-composition]]
