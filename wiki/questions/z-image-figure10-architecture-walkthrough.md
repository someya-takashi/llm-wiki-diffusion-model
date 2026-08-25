---
type: question
asked: 2026-08-25
question: "Z-Image の図10 のアーキテクチャの入力・処理・出力の詳細。DiT はテキストを cross-attention で条件付けするはずだが self-attention でも条件付けになるのか。左下の SigLIP-2→Semantic Processor と VAE→Image Processor はなぜ入力するのか。推論時はどうデノイズするのか"
summaries_used:
  - "[[summaries/2025-z-image]]"
  - "[[summaries/2024-sd3]]"
  - "[[summaries/2025-wan]]"
  - "[[summaries/2025-qwen-image]]"
  - "[[summaries/2025-hunyuanimage-3]]"
  - "[[summaries/2025-flux-kontext]]"
---

# Z-Image 図10 の読み解き：入力・処理・出力と推論

> 質問: 図10 のアーキテクチャの入力・処理・出力の詳細／self-attention でも条件付けになるのか／左下の SigLIP-2 と VAE の 2 経路はなぜ要るのか／推論時のデノイズ
> 回答日: 2026-08-25
> 出典: [[translations/2025-z-image]]（§4.1・§4.3・§4.7）・[[summaries/2025-z-image]]・[[concepts/diffusion-model-architecture]]

> 🔧 **2026-08-26 追記（コードによる検証）**: 公式リファレンス実装（`code_analysis/Z-Image/`・コミット `26f23ed`）で本ページの記述を検証した。**§4 の「時刻の向きが逆」はコードで確定した**（実装が引数を反転し出力の符号を反転している）。一方で**図 10 左下の 2 経路——SigLIP-2 と参照画像の経路——は公開実装に存在しない**ため、§3 は報告書のみを根拠とする記述のままである。以下、該当箇所に 🔧 で追記した。

## 1. 図10 は 2 つのモデルを重ねて描いている

最初に押さえるべき点——**この図は Z-Image（T2I）と Z-Image-Edit の 2 モデルを 1 枚に重ねている**。右下と左下の破線ボックスがそれぞれのラベルになっている。

<figure>

![](../../raw/assets/2025-z-image/architecture.png)

<figcaption>図10（引用, [[summaries/2025-z-image]] より）: 上段が Z-Image（T2I）の経路、左下の Semantic Processor / Image Processor が Z-Image-Edit で追加される参照画像の経路。</figcaption>
</figure>

> 🔧 **2026-08-26 追記（何が検証済みで何がそうでないか）**: 公開されているリファレンス実装は **T2I 経路だけを実装している**。参照画像の入力、SigLIP-2 の意味エンコーダ、左下の semantic processor / image processor は**コードに一切現れない**（`src/zimage/transformer.py` の `forward` が受け取るのは潜在・タイムステップ・キャプション特徴の 3 つだけ）。`README.md` の Model Zoo によれば **Z-Image-Edit と Z-Image-Omni-Base は 2026-08-26 時点で未公開**である。
>
> したがって本ページの信頼度は節ごとに違う——**§1 上段・§2・§4 は実装で裏が取れており、§3（左下 2 経路）は報告書のみが根拠**である。
>
> 実装で確定した上段の構造は次のとおり。図が「軽量なモダリティ固有のプロセッサ」と描くものは、**画像側 2 ブロック（`noise_refiner`）とテキスト側 2 ブロック（`context_refiner`）の合計 4 ブロック**で、**テキスト側だけタイムステップ変調を受けない**（`transformer.py` L308–320）。統一バックボーンは 30 ブロックなので、**総ブロック数は 34** になる。

```
【Z-Image（T2I）で使う経路】

  "A charming white kitten lounging on a striped sofa..."
        │
        ▼
  [Qwen3-4B]（凍結）
        │ Qwen3-4B Embedding
        ▼
  [Text Processor]（transformer 2 ブロック）
        │                    ノイズ付き潜在 x_t（t ∈ [0,1]）
        │                          │
        │                          ▼
        │                    [Image Processor]（2 ブロック）◀── Embed(timestep)
        │                          │
        └────────┬─────────────────┘
                 ▼
        1 本の系列に連結   [ txt … ][ img … ]
                 │
                 ▼
      ┌──────────────────────────┐
      │ Single-Stream Attention  │ ◀── Scale / Gate（timestep 条件）
      ├──────────────────────────┤  × 30
      │ Single-Stream FFN        │ ◀── Scale / Gate
      └──────────────────────────┘
                 │
                 ▼
         [Output Projection]      ← 画像トークン位置だけを取り出す
                 │
                 ▼
          Predicted Velocity

【Z-Image-Edit で「追加される」経路】（左下）

  参照画像（クリーン, t = 1）
      ├─▶ [SigLip-2] ─▶ [Semantic Processor] ─┐  意味特徴
      └─▶ [Flux VAE] ─▶ [Image Processor] ────┤  再構成特徴
                                              └─▶ 同じ 1 本の系列に追加連結
```

### 条件付けの経路は 3 系統ある

| 条件 | 注入方法 |
| --- | --- |
| **テキスト** | **系列に連結**して self-attention で混ぜる |
| **タイムステップ** | **Scale / Gate による変調**（adaLN 系。全 30 ブロックへ） |
| **参照画像**（編集時） | **系列に連結**（VAE ＋ SigLIP-2 の 2 経路） |

出力は **Predicted Velocity**（速度場）。図の上部で青（＝画像トークン）だけが Output Projection へ向かっており、**画像トークンの位置だけを取り出して速度を予測する**構成になっている。

構成値は原典 表2 では 30 層・隠れ次元 3840・注意ヘッド 32・FFN 中間次元 10240・U-RoPE の次元配分 $(d_t,d_h,d_w)=(32,48,48)$ とされる。**🔧 ただし注意ヘッド数 32 は誤りで、実装は 30 ヘッド・head_dim 128 である**（[[summaries/2025-z-image]] の「訂正の記録」）。統一バックボーンが 30 層で、入口のリファイナ 2×2 を含めた総ブロック数は 34。

## 2. self-attention でも条件付けになるのか

**なる。連結系列への self-attention は cross-attention を部分行列として含んでいる。**

系列を $z = [\,z_{\text{txt}};\, z_{\text{img}}\,]$ と連結し、$Q=W_qz,\ K=W_kz,\ V=W_vz$ とすると、注意ロジット $QK^\top$ は次のようにブロック分解できる。

$$
QK^\top=
\begin{bmatrix}
Q_{\text{txt}}K_{\text{txt}}^\top & Q_{\text{txt}}K_{\text{img}}^\top\\[2pt]
\boxed{Q_{\text{img}}K_{\text{txt}}^\top} & Q_{\text{img}}K_{\text{img}}^\top
\end{bmatrix}
$$

**枠で囲った左下ブロック（画像クエリ × テキストキー）が、まさに cross-attention そのもの**である。「テキスト条件が画像に効く」経路は確かに存在する。

> 🔧 **2026-08-26 追記（実際の連結順は `[画像, テキスト]`）**: 上では説明の都合で $z=[\,z_{\text{txt}};\,z_{\text{img}}\,]$ と書いたが、実装は**画像トークンを先に置く**（`transformer.py` L552: `torch.cat([x[i][:x_len], cap_feats[i][:cap_len]])`）。FLUX 系がテキストを先に置く（[[summaries/2025-flux2]]）のと逆である。
>
> **ブロック分解の議論は順序に依らない**ので上の結論は変わらないが、順序は 2 つの点で効く。第一に、**RoPE の座標割り当てが順序と連動する**——テキストは第 1 軸を 1 から増分し、画像はテキストの直後の値を全トークンで共有する（L391–443）。第二に、**モデル出力から画像部分を切り出す位置が変わる**。FLUX.2 が `img[:, num_txt_tokens:]` で後半を取るのに対し、Z-Image は前半を取る。
>
> もう 1 点、**テキスト側の入口プロセッサ（`context_refiner`）はタイムステップ変調を受けない**（`modulation=False`, L317）。§2 (d) で「テキストも毎ステップ流れる」と書いたが、**入口の 2 ブロックに限れば出力はステップに依存しない**——原理的にはキャッシュ可能な部分である（[[concepts/inference-caching]]）。実装はキャッシュしていない。

### cross-attention との 4 つの違い

**(a) テキスト側の重みが専用でない。** cross-attention は $K,V$ をテキスト専用の $W_k^{\text{txt}}, W_v^{\text{txt}}$ で作るが、単一ストリームでは**同じ $W_k, W_v$** がテキストも画像も処理する。

**(b) 逆向きの経路が増える。** 右上ブロック $Q_{\text{txt}}K_{\text{img}}^\top$ は**テキストが画像を見る**経路で、cross-attention には存在しない。結果として**テキストの表現がタイムステップ依存・画像依存になる**。

**(c) softmax の「予算」を奪い合う。** ここが最も重要で、**softmax は行全体で正規化される**ので上の 4 ブロックは足し算ではなく**競合**する。画像トークンが数千個・テキストが数百個なら、注意の質量は自然と画像側へ偏る。

→ これが **Wan があえて cross-attention 型 DiT を選んだ**（[[summaries/2025-wan]]）理由の数学的な中身である。動画では視覚トークンが数十万に達するので、そこにテキスト 512 トークンを連結すると**テキストが視覚に飲み込まれる**。「連結して self-attention」が常に優れているわけではない。

**(d) テキストの K/V をキャッシュできない。** cross-attention ならテキストの $K,V$ は全ステップで使い回せるが、単一ストリームでは (b) によりテキスト表現が毎ステップ変わるため、**30 層すべてを毎ステップ再計算**する必要がある。実推論コストに効く。

### 系譜の中での位置

| 世代 | テキスト条件の入れ方 | 代表 |
| --- | --- | --- |
| DiT | クラス条件を adaLN で変調するのみ | DiT（[[summaries/2023-dit]]） |
| cross-attention 型 | 画像がテキストを見る一方向 | LDM/SD、Wan（[[summaries/2025-wan]]） |
| MM-DiT（二重ストリーム） | 連結して joint attention。**ただし重みは別** | SD3（[[summaries/2024-sd3]]） |
| **完全単一ストリーム** | 連結して joint attention。**重みも共有** | **Z-Image の S3-DiT** |

詳細は [[concepts/diffusion-model-architecture]]。

## 3. なぜ SigLIP-2 と VAE の両方を入れるのか

### 前提：この 2 経路は編集時（Z-Image-Edit）専用

原典が明記している。

> 編集タスクに限り、参照画像から抽象的な視覚的意味を捉えるために SigLIP 2 でアーキテクチャを増強する

つまり T2I では左下の 2 経路は使われない。

> 🔧 **2026-08-26 追記**: **本節の内容は報告書のみを根拠とする**。公開実装には SigLIP-2 も参照画像の経路も存在せず、Z-Image-Edit の重みも未公開である。以下の説明は原典の記述に忠実だが、**実装で確認された事実ではない**。

### 同じ参照画像を 2 通りに符号化して両方入れる

| 経路 | 何を持ち込むか |
| --- | --- |
| **Flux VAE → Image Processor** | **再構成的な特徴**。画素レベルの詳細・正確な色・テクスチャ。「元の見た目をそのまま保つ」ための情報 |
| **SigLIP-2 → Semantic Processor** | **意味的な特徴**。「何が写っているか」の抽象表現。指示文と照合して「どこを変えるべきか」を判断するための情報 |

これは本 wiki が **二重符号化（dual-encoding）** として記録してきたパターンそのものである（[[concepts/instruction-based-image-editing]]）。Qwen-Image（[[summaries/2025-qwen-image]]）が「MLLM（意味）と VAE（画素）の 2 経路」を採り、HunyuanImage 3.0（[[summaries/2025-hunyuanimage-3]]）も「VAE の潜在と ViT の特徴を両方連結する二重エンコーダ」を採る。**Z-Image も独立に同じ設計へ到達している。**

### なぜ片方では足りないのか

- **VAE だけ**だと、画素は正確に持てても「この画像に猫が写っている」という意味的な把握が弱く、**指示文と画像の対応が取りにくい**
- **SigLIP-2 だけ**だと、意味は分かっても画素の詳細が失われ、**編集していない領域まで作り直してしまう**

編集の要求は「**変えるべき所だけ変える**」なので、**変えない部分を保つ情報（VAE）と、どこを変えるか判断する情報（SigLIP-2）の両方**が要る。

### 参照画像と対象画像の区別

2 つの仕掛けで行う（[[concepts/position-embedding]]）。

- **3D Unified RoPE の時間次元で単位区間だけずらす**（空間座標は揃えたまま）
- **異なる時間条件付けの値**：参照は $t=1$（クリーン）、対象は $t\in[0,1]$（ノイズ付き）

FLUX.1 Kontext の**仮想タイムステップ**（[[summaries/2025-flux-kontext]]）と同型の解に独立に到達している。

## 4. 推論時のデノイズ

### ⚠️ 時刻の向きが一般的な流儀と逆

原典の定義は $x_t = t\cdot x_1 + (1-t)\cdot x_0$ で、**$x_0$ がガウスノイズ、$x_1$ が元画像**。つまり：

$$t=0 \;\Rightarrow\; \text{純ノイズ},\qquad t=1 \;\Rightarrow\; \text{クリーン画像}$$

[[concepts/flow-matching]] に記録した SD3 の流儀（$z_t=(1-t)x_0+t\epsilon$、$t=0$ がデータ）とは**逆向き**なので、実装時に混同しやすい。図10 で参照画像が $t=1$ とされているのは「クリーン」の意味である。

**FLUX.2 のコードでも SD3 側の規約が確認できる**（[[summaries/2025-flux2]]）——`get_schedule` は `torch.linspace(1, 0, ...)` で降順に刻み、更新式 `img = img + (t_prev - t_curr) * pred` の係数は負である。つまり **$t=1$ がノイズ・$t=0$ がクリーン**で、Z-Image だけが逆を向いている。2 つのモデルの実装を並べて触るときは特に注意が要る。

> 🔧 **2026-08-26 追記（実装はどう帳尻を合わせているか）**: 本ページで指摘したこの逆向きは、**Z-Image のコードそのものに現れる**。しかも興味深いことに、実装は**モデルの内部規約を変えず、パイプライン側で 2 回の反転をかけて標準的なスケジューラに載せている**。
>
> ```python
> # src/zimage/pipeline.py L223 — 引数を反転してモデルへ渡す
> timestep = (1000 - timestep) / 1000        # scheduler の t（1000→0）を モデルの t（0→1）へ
>
> # 同 L274 — 出力の符号を反転
> noise_pred = -noise_pred.squeeze(2)
>
> # src/zimage/scheduler.py L137–138 — 以降は diffusers 標準の Euler そのまま
> dt = sigma_next - sigma                     # dt < 0
> prev_sample = sample + dt * model_output
> ```
>
> スケジューラは HuggingFace diffusers の `FlowMatchEulerDiscreteScheduler` をほぼそのまま流用したもの（`scheduler.py` 冒頭のコメントが明記）で、$\sigma$ は $1 \to 0$ へ降下する SD3/FLUX の流儀である。**モデルだけが逆向きの速度場を学習しており、$\Delta t<0$ と $-v$ の符号が打ち消し合って正しい方向に進む。**
>
> ここから 2 つ言える。第一に、**定式化の向きの違いは実装の細部で完全に吸収できる**——エコシステム（diffusers のスケジューラ、既存のサンプラー）を共有したいなら、モデルの規約を変えるより境界で反転するほうが安い（[[concepts/flow-matching]]）。第二に、**その代償として符号の間違いが静かに壊れる**。$-$ を落としても実行は通り、生成が壊れるだけである。Z-Image の LoRA を自分で学習・評価するコードを書くなら、ここは最初に確認する箇所になる。
>
> **なお推論の時刻スケジュールは、本節「補足 4 点」の記述と食い違う。** 実装は**静的な `shift = 3.0`** を使い、解像度依存の動的シフトは使っていない——パイプラインは SD3・FLUX と同じ定数から $\mu$ を計算するが、`use_dynamic_shifting=False` のため**その値は捨てられる**（`pipeline.py` L194–202, `scheduler.py` L85–88）。詳細は [[concepts/noise-schedule]]。

### 学習目標

速度 $v_t = x_1 - x_0$（ノイズから画像へ向かうベクトル）を回帰する。

$$\mathcal{L}=\mathbb{E}_{t,x_0,x_1,y}\big[\|u(x_t,y,t;\theta)-(x_1-x_0)\|^2\big]$$

時刻サンプリングは SD3 に倣った **logit-normal**（中間タイムステップに学習を集中させる）＋ FLUX 由来の**動的な時間シフト**（解像度による SNR の変動を吸収）。

### 推論の手続き

速度場に沿った ODE を $t=0 \to 1$ へ積分するだけである。

```
1. x ← ガウスノイズ  （t = 0）
2. for t in [0 → 1] のスケジュール:
     v ← u_θ(x, y, t)                     # 30 層を通して速度を予測
     v ← v_uncond + w·(v_cond − v_uncond)  # CFG（ベース版のみ）
     x ← x + Δt · v                        # オイラー法
3. Flux VAE デコーダで潜在 → 画像
```

### 補足 4 点

**時刻の刻み方**は解像度に応じた動的シフトを使う、と原典は述べる。[[concepts/noise-schedule]] に記録したとおり、この $\alpha$ シフトは **$\mu=\log\alpha$ の logit-normal と等価**であることが FLUX.1 Kontext の付録で示されている。**🔧 ただし公開実装は静的な `shift=3.0` で、動的シフトを使っていない**（上記追記）。原典の記述は学習時のものと読むのが自然だが、**推論側でそれが有効になっていない**ことは記録しておく価値がある。

**CFG はベース版のみ。** Z-Image ベースは classifier-free guidance を使うので **1 ステップあたり 2 回のモデル評価**が要る。Z-Image-Turbo は Decoupled DMD の CFG-Augmentation により CFG 不要で **8 NFE** で完結する（[[concepts/diffusion-distillation]]）。

**テキストトークンも毎ステップ流れる。** §2 の (d) のとおり、テキストは系列の一部として 30 層すべてを毎ステップ通る。cross-attention 型なら K/V を 1 回計算して使い回せる部分が、ここでは再計算になる。

**編集時は参照画像トークンも毎ステップ流れる。** $t=1$ 固定なので中身は変わらないが、系列に居続けるため計算は毎回発生する。この冗長性は [[concepts/inference-caching]] が扱う最適化の余地そのものである。

## まとめ

| 問い | 答え |
| --- | --- |
| 入力 | テキスト（Qwen3-4B）＋ノイズ付き潜在（Flux VAE）＋タイムステップ。編集時のみ参照画像を VAE と SigLIP-2 の 2 経路で追加 |
| 処理 | 各モダリティを 2 ブロックのプロセッサに通し、**1 本の系列に連結**して共有重みの Attention/FFN ブロック ×30 |
| 出力 | **画像トークン位置の速度場**（Predicted Velocity） |
| self-attention は条件付けになるか | **なる**。$Q_{\text{img}}K_{\text{txt}}^\top$ ブロックが cross-attention そのもの。ただし softmax の予算を競合する点が本質的な差 |
| SigLIP-2 と VAE の 2 経路 | **編集専用の二重符号化**。VAE＝保つための画素情報、SigLIP-2＝変える場所を判断する意味情報 |
| 推論 | $t=0$（ノイズ）→ $t=1$（画像）へ速度場の ODE をオイラー法で積分。ベースは CFG あり、Turbo は 8 NFE で CFG なし |

## 関連ページ

- [[summaries/2025-z-image]]
- [[translations/2025-z-image]]
- [[concepts/diffusion-model-architecture]]
- [[concepts/instruction-based-image-editing]]
- [[concepts/flow-matching]]
- [[concepts/noise-schedule]]
- [[concepts/position-embedding]]
- [[concepts/inference-caching]]
- [[concepts/diffusion-distillation]]
- [[questions/z-image-architecture-and-lora-placement]]
