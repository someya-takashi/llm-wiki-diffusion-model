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

## 1. 図10 は 2 つのモデルを重ねて描いている

最初に押さえるべき点——**この図は Z-Image（T2I）と Z-Image-Edit の 2 モデルを 1 枚に重ねている**。右下と左下の破線ボックスがそれぞれのラベルになっている。

<figure>

![](../../raw/assets/2025-z-image/architecture.png)

<figcaption>図10（引用, [[summaries/2025-z-image]] より）: 上段が Z-Image（T2I）の経路、左下の Semantic Processor / Image Processor が Z-Image-Edit で追加される参照画像の経路。</figcaption>
</figure>

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

構成値は原典 表2 のとおり：30 層・隠れ次元 3840・注意ヘッド 32・FFN 中間次元 10240・U-RoPE の次元配分 $(d_t,d_h,d_w)=(32,48,48)$。

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

**時刻の刻み方**は解像度に応じた動的シフトを使う。[[concepts/noise-schedule]] に記録したとおり、この $\alpha$ シフトは **$\mu=\log\alpha$ の logit-normal と等価**であることが FLUX.1 Kontext の付録で示されている。

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
