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

> 🔧 **2026-08-26 追記（コードによる検証）**: 公式リファレンス実装（`code_analysis/Z-Image/`・コミット `26f23ed`）と [[summaries/2026-ai-toolkit]] の Z-Image 実装を読み、本ページを更新した。変更は 4 点——(a) **注意ヘッド数を 32 → 30 に訂正**し、下の訳注が留保していた疑問を決着させた、(b) ★A〜★G の候補に**実モジュール名**を与えた（従来は構造からの演繹だけで、名前を持っていなかった）、(c) **ai-toolkit の既定が `layers` のみを標的にする**ため ★C の実行に設定変更が要ることを追記、(d) §4 の (2)(3) にコード／公式表からの裏づけを追加。詳細は [[summaries/2025-z-image]] の「コードから確認できたこと」節。

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
| 層数 | 30（＋入口のリファイナ 2×2 ＝ **総 34 ブロック**） |
| 隠れ次元 | 3840 |
| 注意ヘッド数 | ~~32~~ → **30**（訂正） |
| head_dim | **128** |
| FFN 中間次元 | 10240（SwiGLU・$\lfloor 3840/3\times8 \rfloor$） |
| U-RoPE の次元配分 $(d_t,d_h,d_w)$ | $(32,48,48)$ ・ theta $=256$ |

> **訳注（2026-08-26 決着）**: 表の値をそのまま取ると head_dim $=3840/32=120$ だが、U-RoPE の次元配分は $32+48+48=128$ に足し合わさる。実装では head_dim$=128$（→ 30 ヘッド）か、RoPE を部分次元にのみ適用しているかのいずれかと思われる、と留保していた。**リファレンス実装により前者が確定した**——`src/config/model.py` L28–32 が `N_HEADS=30`、`src/zimage/transformer.py` L338–339 が `assert head_dim == sum(axes_dims)` を実行時に強制する（32 ヘッドなら head_dim=120 で assert が落ちる）。**原典 表2 の「32」は誤りである。** 当初の見立てどおり LoRA を当てるだけなら影響しない。経緯は [[summaries/2025-z-image]] の「訂正の記録」に残した。

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

| | 対象 | **実モジュール名**（リファレンス実装） | 箇所数 | 何が変わるか | 推奨度 |
| --- | --- | --- | --- | --- | --- |
| **★A** | 単一ストリーム Attention の $W_q,W_k,W_v,W_o$ | `layers.{0..29}.attention.{to_q,to_k,to_v,to_out.0}` | 30 層 × 4 | トークン間の関係の作り方。**最も実績のある標準的な位置** | ★★★ **第一選択** |
| **★B** | 単一ストリーム FFN（3840→10240→3840） | `layers.{0..29}.feed_forward.{w1,w2,w3}` | 30 層 × 3 | トークンごとの特徴変換。容量が大きく**質感・画風に効きやすい** | ★★☆ A で足りなければ追加 |
| **★C** | 入口の Image Processor（2 ブロック） | `noise_refiner.{0,1}.*` | 2 ブロック | **唯一モダリティを分離できる場所**。画像表現だけに触れる | ★★☆ 被写体特化なら試す価値 |
| **★D** | 条件注入の**層ごと**上方射影 | `layers.{i}.adaLN_modulation.0`（Linear 256→15360） | 30 層 | 各ブロックの Scale / Gate ＝**寄与量**を層ごとに変調 | ★☆☆ 実験的 |
| **★E** | 条件注入の**共有**下方射影 | `t_embedder.mlp.{0,2}`（Linear 256→1024→256） | 1 箇所 | 全 34 ブロックに一斉に効く。パラメータ極小 | ☆☆☆ 破綻しやすい |
| **★F** | Text Processor / Qwen3-4B | `context_refiner.{0,1}.*` / `cap_embedder.1` | 2 ブロック | PE-aware SFT で作った整合を壊しうる（§4） | 非推奨 |
| **★G** | Flux VAE | （別モジュール） | — | 凍結すべき（[[concepts/image-tokenizer]]） | 非推奨 |
| — | 入出力の射影 | `all_x_embedder.2-1`（64→3840）, `all_final_layer.2-1.linear`（3840→64） | 各 1 | 潜在の読み書きそのもの。触らないのが既定 | 非推奨 |

> **モジュール名は `code_analysis/Z-Image/src/zimage/transformer.py` から読み取った**（`attention` は L95–98、`feed_forward` は L70–75、`adaLN_modulation` は L169、`noise_refiner` / `context_refiner` は L308–320、`t_embedder` / `cap_embedder` は L322–326）。`all_x_embedder` / `all_final_layer` のキー `"2-1"` は `patch_size-f_patch_size` の連結で、2D 画像では常にこの 1 つだけが存在する（L299–306）。**★E は「共有下方射影」という独立した層ではなく、タイムステップ埋め込み器そのもの**である点に注意——ここを触るとタイムステップの解釈自体が変わる。

### ★A を第一選択にする理由

[[concepts/low-rank-adaptation]] にまとめたとおり、拡散モデルの LoRA は伝統的に注意・線形層に当てるのが既定で、**Orthogonal Adaptation**（[[summaries/2024-orthogonal-adaptation]]）も「Stable Diffusion アーキテクチャ内のすべての線形層」に適用している。S3-DiT は均質な積層なので、**層を選ぶ根拠が現時点で存在しない以上、全層に薄く当てるのが最も安全**である。

### ★C が Z-Image では特別な意味を持つ

単一ストリームであることの裏返しとして、**バックボーンに当てた LoRA はテキストトークンの表現も必ず動かす**。画風 LoRA がテキスト理解を副作用で歪める、という事故が起こりうる。

**Image Processor（2 ブロック）だけがモダリティ的に隔離された場所**である。「被写体の見た目だけを教えたい、プロンプト追従は一切触りたくない」という要求には、ここが構造的に正しい当てどころになる。ただし 2 ブロックしかないので容量は小さく、**単独で十分かは未検証**。

### ★D / ★E は Z-Image 固有の（そして危険な）標的

条件注入の低ランク分解は Z-Image の設計上の特徴で、標準的な LoRA レシピには対応物がない。

- **★D（層ごと上方射影）**: 各ブロックの寄与の**大きさ**を層ごとに変えられる。「モデルの計算内容は変えず、どの層をどれだけ効かせるか」だけを学習する、という珍しい介入になる。
- **★E（共有下方射影）**: **1 箇所いじると 30 層すべての変調が変わる**。パラメータ効率は極端に良いが、レバーが長すぎて制御が難しい。「モデル全体の雰囲気をわずかに動かす」用途の実験としては面白いが、被写体学習には向かない。

### ツール側の既定は ★A・★B・★D しか触らない（2026-08-26 追記）

上の表は「構造上どこに当てられるか」であって、「ツールが既定で当てるか」ではない。[[concepts/ai-toolkit]]（[[summaries/2026-ai-toolkit]]）の Z-Image 実装は

```python
def get_transformer_block_names(self) -> Optional[List[str]]:
    return ["layers"]
```

と宣言する（`extensions_built_in/diffusion_models/z_image/z_image.py` L464–465）。ai-toolkit ではこのメソッドが **LoRA 標的・凍結範囲・量子化範囲の 3 つの境界を同時に決める**ので、既定では次のようになる。

| 候補 | ai-toolkit の既定 |
| --- | --- |
| ★A `layers.*.attention.*` | ✅ 対象 |
| ★B `layers.*.feed_forward.*` | ✅ 対象 |
| ★D `layers.*.adaLN_modulation.0` | ✅ 対象（`layers` 配下なので巻き込まれる） |
| **★C `noise_refiner`** | ❌ **対象外** |
| ★F `context_refiner` / `cap_embedder` | ❌ 対象外（望ましい） |
| ★E `t_embedder` | ❌ 対象外（望ましい） |
| 入出力射影 | ❌ 対象外（望ましい） |

**含意は 2 つある。** 第一に、**★C（唯一モダリティを分離できる場所）を試すには `only_if_contains` 等で標的を明示的に書き足す必要がある**——何もしなければ入口プロセッサには一切 LoRA が乗らない。第二に、**★D は既定で巻き込まれている**。上の表で「★☆☆ 実験的」と評価したゲート／スケールの変調は、`layers` を丸ごと標的にした時点で自動的に含まれる。標準的な attention＋FFN だけに絞りたいなら、むしろ `adaLN_modulation` を `ignore_if_contains` で外す側の操作が要る。

なお [[summaries/2025-flux2]] の FLUX.2 では同じメソッドが `["double_blocks","single_blocks"]` を返し、**共有変調の 4 行列は標的から外れる**。Z-Image では変調が `layers` の内側にあるため、**同じツールの同じ既定が、モデルによって変調を含めたり外したりする**。合成を予定するなら（§7）、この差は把握しておく価値がある。

## 4. Z-Image 固有の注意点 5 つ

### (1) Sandwich-Norm と Zero-init Gate が振幅を握っている

ブロック出力は **RMSNorm を通ってからゲートで倍率が決まる**。つまり LoRA でブロック内部の重みを動かしても、出力の**「向き」は変わるが「大きさ」はゲートが決める**。

**実務上の含意**: 効きが弱いと感じたときに rank や学習率を上げる前に、★D（ゲート側）も候補に入れる価値がある。逆に、**学習済みのゲート値が小さい層に LoRA を当てても効きにくい**はずである。

> 🔧 **2026-08-26 追記**: 実装ではゲートは `gate.tanh()`、スケールは `1.0 + scale` で、変調は **`scale_msa, gate_msa, scale_mlp, gate_mlp` の 4 つだけ（shift 項がない）**（`transformer.py` L180–184）。「ゼロ初期化ゲート」は厳密には **tanh で $(-1,1)$ に有界化されたゲート**である。この差は上の議論を強める方向に効く——ゲートは**構造的に $\pm1$ を超えられない**ので、★D で押し込める倍率には上限がある。★D は「効きを増幅する」よりも「**層ごとの寄与配分を組み替える**」介入だと考えたほうがよい。

### (2) 学習目的は velocity 予測、タイムステップ分布は変える価値がある

Z-Image は rectified flow で、出力は **Predicted Velocity**（DDPM の $\epsilon$ 予測ではない）。

事前学習は logit-normal の時刻サンプリングだが、[[concepts/noise-schedule]] に記録した **HiDream-O1-Image の知見**（[[summaries/2026-hidream-o1-image]]）が直接効く——**SFT の段階では logit-normal を一様サンプリングに切り替える**。理由は、美的品質・写実性・細部が決まるのは $t$ が小さい後期のノイズ除去ステップであり、logit-normal は中間時刻に予算を集中させてしまうからである。

**被写体の細部（ロゴ・質感・エッジ）を教える LoRA なら、一様サンプリング寄りに倒すのは理にかなう。**

> 🔧 **2026-08-26 追記（推論側のスケジュール）**: リファレンス実装は **静的な `shift = 3.0`** を使い、`use_dynamic_shifting` は `False` である（`src/config/model.py` L39–40, `src/zimage/scheduler.py` L85–88）。興味深いことに、パイプラインは SD3・FLUX と同じ定数（base_shift 0.5 / max_shift 1.15）から**解像度依存の $\mu$ を計算しておきながら、`use_dynamic_shifting=False` のため使わずに捨てている**（`src/zimage/pipeline.py` L194–202）。[[summaries/2026-ai-toolkit]] の学習側も同じ `shift: 3.0` / `use_dynamic_shifting: False` を使うので、**学習と推論でスケジュールが揃っている**——これは [[concepts/noise-schedule]] に記録した FLUX.2 の状況（推論は解像度依存シフト、学習は一様）と対照的で、Z-Image では**その不整合が存在しない**。時刻分布を触るなら、いじるのは `timestep_type` 側であってシフトではない。

### (3) ベース版で学習する（Turbo ではなく）

Z-Image-Turbo は Decoupled DMD による 8 NFE の蒸留版である。[[concepts/diffusion-distillation]] の既知の限界がそのまま出ており、**OneIG の Diversity がベース 0.194 → Turbo 0.139 まで落ちる**（原典はこの悪化を議論していない）。

蒸留済みモデルの上に LoRA を積むと、**蒸留で作られた少ステップの軌道と LoRA の変更が干渉しうる**。ベース版で学習し、必要なら Turbo へ移して確認するのが安全である。

> 🔧 **2026-08-26 追記（公式表による裏づけ）**: リポジトリの `README.md` の Model Zoo が **Turbo の「Fine-Tunability」を明示的に `N/A`** とし、Z-Image / Omni-Base / Edit を `Easy` としている。本節は多様性スコアからの演繹だったが、**公式の推奨として同じ結論が示されていた**。加えて非蒸留の Z-Image は 2026-01-27 に公開済みで、推奨設定は 28–50 ステップ・CFG 3.0–5.0・`cfg_normalization` は写実で True / 画風で False。**Omni-Base と Z-Image-Edit は 2026-08-26 時点で未公開**である。
>
> なお [[concepts/ai-toolkit]] には Z-Image 用の **assistant LoRA（蒸留打ち消しアダプタ、multiplier $=-1$）** が用意されている（[[concepts/diffusion-distillation]]）。「それでも Turbo で学習したい」という需要が実在することの傍証だが、**merging の実験では蒸留と干渉のどちらが原因かを分離できなくなる**ので、本ページの推奨は変わらない。

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
