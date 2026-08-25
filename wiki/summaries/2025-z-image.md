---
type: summary
source_path: raw/papers/Z-Image_ An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer.md
source_kind: paper
code_source_path: code_analysis/Z-Image/
code_commit: 26f23ed
code_verified: 2026-08-26
title: "Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer"
authors: [Z-Image Team (Alibaba Group / Tongyi-MAI)]
year: 2025
venue: arXiv:2511.22699（テクニカルレポート）
ingested: 2026-08-18
tags: [diffusion-model-architecture, diffusion-distillation, reinforcement-learning-for-diffusion, data-curation, prompt-enhancement, text-to-image-generation, visual-text-rendering, instruction-based-image-editing, large-scale-training-infrastructure, flow-matching, position-embedding, classifier-free-guidance, efficient-attention]
translation: "[[translations/2025-z-image]]"
---

# Z-Image: 単一ストリーム DiT で「規模至上主義」に挑む 6B の基盤モデル

> 原典: [[translations/2025-z-image]] ・ `raw/papers/Z-Image_ An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer.md`
> 著者・年: Z-Image Team（Alibaba Group / Tongyi-MAI）・2025 年 11 月・arXiv:2511.22699

## 一言まとめ

**60 億パラメータ・総額 63 万ドルで、人間の選好の Elo ランキング世界 4 位（オープンソース 1 位）に達したことを示すテクニカルレポート**。テキストも画像も 1 本の系列に流す **S3-DiT（Scalable Single-Stream DiT）** を柱に、データ基盤・学習カリキュラム・蒸留・事後学習をすべて「効率」の観点から設計し直している。

## 背景と問題意識

本 wiki が 2025–2026 年の基盤モデルを追ってきて見えたのは、**素直にパラメータを増やす**という一本道だった。Qwen-Image が 20B（[[summaries/2025-qwen-image]]）、HiDream-I1 が 17B（[[summaries/2025-hidream-i1]]）、FLUX.2 が 32B、Hunyuan-Image-3.0 に至っては 80B。著者らはこれを「**scale-at-all-costs（あらゆる代償を払ってスケールする）**」と呼び、正面から挑戦する。

同時にもう 1 つ、より批判的な指摘がある。資源に乏しい研究者が取りがちな近道——**プロプライエタリなモデルから合成データを蒸留する**——について、「**閉じたフィードバックループを作り、誤差の蓄積とデータの均質化を招き、教師モデルに既に存在するものを超えた新しい視覚的能力の創発を妨げる**」と述べる。Z-Image は他モデルからの蒸留を一切使わず、実世界データのみで学習したと明言する。

この 2 つの拒否——大規模化の拒否と合成データの拒否——が、レポート全体の設計を規定している。

そして本 wiki にとって重要なのは、**学習コストを金額で公開している**ことである。

| | 低解像度事前学習 | Omni 事前学習 | 事後学習 | 合計 |
| --- | --- | --- | --- | --- |
| H800 GPU 時間 | 147.5K | 142.5K | 24K | **314K** |
| USD（$2/GPU 時間） | $295K | $285K | $48K | **$628K** |

直前に取り込んだ Wan（[[summaries/2025-wan]]）の要約で「学習コストの総量が報告されない」を批判点として挙げたが、Z-Image はそこに段階別の内訳付きで答えている。**事前学習が全体の 92%、しかもその半分以上が 256² の低解像度段階**という配分は、それ自体が知見である。

## 提案手法 / 主張

### 1. S3-DiT — 二重ストリームを完全に畳む

<figure>

![](../../raw/assets/2025-z-image/architecture.png)

<figcaption>図10（引用）: S3-DiT のアーキテクチャ。左下：SigLIP-2 の意味埋め込み・VAE 埋め込み・Qwen3-4B のテキスト埋め込み・タイムステップが、それぞれ軽量なプロセッサ（各 2 ブロック）を通って 1 本の系列に連結される。右：Single-Stream Attention / FFN ブロックの内部（RMS Norm → Scale → 演算 → RMS Norm → ゼロ初期化ゲート、Q-Norm/K-Norm と U-RoPE）。左下の Z-Image-Edit では参照画像が t=1（クリーン）、対象が t=[0,1]（ノイズ付き）。</figcaption>
</figure>

本 wiki が追ってきたモダリティ融合の系譜が、ここで一周する。

| 世代 | 構成 | 代表 |
| --- | --- | --- |
| 二重ストリーム | テキストと画像に別重み、attention でのみ混ぜる | SD3 の MM-DiT（[[summaries/2024-sd3]]） |
| 二重 → 単一の二段 | 前半は別重み、後半は連結して 1 本 | FLUX.1（[[summaries/2025-flux-kontext]]）、HiDream-I1 |
| **完全な単一ストリーム** | **入口の 2 ブロックを除き最初から全部 1 本** | **Z-Image の S3-DiT** |

動機は「decoder-only の大規模言語モデルのスケーラビリティ」で、これは HiDream-O1-Image（[[summaries/2026-hidream-o1-image]]）と同じ言い分である。ただし**あちらが VAE ごと捨てた**（[[pixel-space-diffusion]]）のに対し、Z-Image は **Flux VAE をそのまま流用**し、潜在空間に留まる。「単一ストリーム化」と「ピクセル空間化」は独立した設計判断であり、Z-Image はその片方だけを取った——という位置づけになる。

構成要素は徹底して既製品である：テキストエンコーダは **Qwen3-4B**（軽量な二言語 LLM）、画像トークナイザは **Flux VAE**、編集時の参照画像だけ **SigLIP 2**。6.15B・30 層・隠れ次元 3840。

**位置符号化に 3D Unified RoPE** を使う。画像トークンは空間 2 次元に展開し、**テキストトークンは時間次元に沿って増分する**——テキストと画像を同じ座標系に載せるための工夫である。編集タスクでは、参照画像と対象画像に**空間座標は揃えたまま、時間次元で単位区間だけずらす**。FLUX.1 Kontext の「仮想タイムステップ」（[[summaries/2025-flux-kontext]]）と同型の解に、独立に到達している。さらに参照画像には $t=1$（クリーン）、対象画像には $t\in[0,1]$（ノイズ付き）という**異なる時間条件付けの値**を与えて区別する。

安定化は 3 点セット：**QK-Norm**（attention の query/key を正規化）、**Sandwich-Norm**（attention / FFN ブロックの入力**と出力の両方**を正規化して信号振幅を抑える）、そして全正規化が **RMSNorm**。図 10 を見ると出力側には**ゼロ初期化ゲート**が置かれており、学習開始時に各ブロックの寄与をゼロにする ControlNet 的な発想（[[controllable-generation]]）が層内部に持ち込まれている。

条件注入にも省パラメータの工夫がある。条件ベクトルからスケール／ゲートを作る射影を **低ランクの対**に分解し、**下方射影は全層で共有、上方射影だけ層ごと**にする。Wan が adaLN の MLP を全層共有した（[[summaries/2025-wan]]）のと同じ方向の——**条件付け機構は思ったより安く済ませてよい**——知見である。

### 2. Decoupled DMD — DMD の理解そのものを書き換える

本論文で最も再利用価値が高いのはここだろう。

**DMD（Distribution Matching Distillation）** は、生徒の生成分布を教師の分布に一致させる蒸留として理解されてきた（[[diffusion-distillation]]）。しかし著者らが実際に使ってみると、**高周波の細部が失われ、色調がずれる**という持続的なアーティファクトに遭遇した。原因を探った結果の主張が以下である。

> DMD の有効性は一枚岩の現象ではなく、**2 つの独立して協働する機構の結果**である。
> - **CFG-Augmentation（CA）**：蒸留を実際に駆動する**主要なエンジン**。生徒の少ステップ生成能力を築き上げているのはこちら。**この要因は先行文献でほとんど見過ごされてきた**。
> - **Distribution Matching（DM）**：主として**正則化子**として機能し、学習を安定させアーティファクトを除去する。

つまり「分布マッチング蒸留」という名前が示すものは、実は主役ではなかった、という主張である。両者を切り離し、**CA 項と DM 項に別々の再ノイズ付与スケジュール**を適用したのが Decoupled DMD で、これで細部の喪失と色ずれが解消する。結果として蒸留された 8 ステップの生徒は、**100 ステップの教師に匹敵するだけでなく、写実性と視覚的インパクトで教師を上回る**。

[[diffusion-distillation]] の「敵対的蒸留は例外的に教師を超えうる」という記述に、**分布マッチング系でも同じことが起きる**という事例が加わったことになる。

### 3. DMDR — 正則化子を RL に転用する

上の洞察の自然な帰結である。生成モデルへの RL は **報酬ハッキング**（報酬関数を悪用して高スコアだが視覚的に意味をなさない画像を作る）のリスクを抱え、通常は外から正則化を足す必要がある。ところが Decoupled DMD が示したのは「**DM 項はもともと正則化子である**」ということだった。ならば外から足す必要はなく、RL の目的関数とそのまま組める——これが **DMDR（Distribution Matching Distillation meets Reinforcement Learning）** である。

RL が人間の選好への整合を解き放ち、DM 項が報酬ハッキングを防ぐ制約として働く。[[diffusion-distillation]] と [[reinforcement-learning-for-diffusion]] が別々の工程ではなく、**同じ損失の中で役割分担する**形になっている。

### 4. データ基盤 — 4 つのエンジン

[[data-curation]] に詳述したが、要点は「**単位計算あたりの情報獲得を最大化する**」という発想でデータを動的に扱うことである。

- **Data Profiling Engine**：pHash による低レベル重複除去、圧縮アーティファクト（理想非圧縮サイズ／実サイズの比）、劣化（色かぶり・ぼけ・透かし・ノイズ）、**情報エントロピー**（縁のピクセル分散で単色背景を検出、JPEG 再符号化の BPP を複雑さの代理指標に）、美的スコア、**AIGC 検出**、意味タグと NSFW。
- **Cross-modal Vector Engine**：SD3 の重複除去を**グラフのコミュニティ検出**として再定式化し、`range_search` を k-NN 探索に置き換えて 8 台の H800 で **10 億件あたり 8 時間**まで高速化。失敗事例で検索を掛けて**原因となるデータクラスタを特定し刈り込む**という使い方もする。
- **World Knowledge Topological Graph**：Wikipedia 全エンティティから知識グラフを作り、**PageRank で孤立概念を刈り**、**VLM で「そもそも視覚化できない抽象概念」を刈る**。キャプション埋め込みの階層クラスタリングで補強し、親ノードは VLM が子を要約して命名する。応用は BM25 スコアと親子関係から**概念別のサンプリング重み**を計算すること。
- **Active Curation Engine**：Z-Image 自身を**診断用の生成事前分布**として使い、失敗する概念（難例）を自動サンプリングで特定する。人間と AI の二重検証を挟む human-in-the-loop の閉ループでキャプションモデルと報酬モデルを再学習する。

長尾問題の具体例が秀逸である——「**松鼠鳜鱼**」は実在する中華料理名だが、モデルがこの概念を持たないと「**リス（松鼠）＋魚（鳜鱼）**」に分解して構成的に推論し、誤った画像を生む。

編集ペアの構築にも独自の工夫がある。**グラフ表現**：1 枚の入力画像に対し $N$ 個の編集版を専門家モデルで作れば、任意のペアの組み合わせで $2\binom{N+1}{2}$ 組が**ゼロコストで**得られる。しかも 2 つの編集版を組み合わせれば混合編集データになり、逆向きに取れば「歪んだ画像 → 実在の歪みのない画像」という**品質の高いペア**が得られる。ほかに動画フレームからの自然なペア（CN-CLIP のコサイン類似度で絞る）、テキスト編集用の可制御レンダリング。

### 5. PE-aware SFT — プロンプトエンハンサを「凍結する」

[[prompt-enhancement]] にまとめたが、Z-Image の立場は他と逆である。

6B という規模ゆえ、Z-Image は世界知識・意図理解・複雑な推論に限界がある。しかし「**詳細なプロンプトを写実的な画像へ翻訳する強力なテキストデコーダ**」としては優秀だ、と割り切る。そこで認知的なギャップは外付けの VLM（プロンプトエンハンサ）に任せる。

決定的なのは**どちらを動かすか**である。Qwen-Image-2.0 は PE を下流の画像品質で RL 最適化し（[[summaries/2026-qwen-image-2]]）、HiDream-O1-Image は推論エージェント自体を報酬化した（[[summaries/2026-hidream-o1-image]]）。Z-Image は逆に **VLM を固定したまま、SFT の段階で全入力プロンプトを PE に通し、拡散側を PE の出力分布に合わせる**。LLM の追加学習コストがゼロになる。

**構造化された推論連鎖**が鍵だとも述べる。推論なしの PE は地理座標を与えられると座標のテキストをそのまま画像に描くが、推論ありなら場所（西湖）を推論して正しい風景を生成する。

### 6. 学習カリキュラムと事後学習

**低解像度事前学習**（256² のみ、総計算量の半分超）→ **Omni 事前学習**（任意解像度 ＋ T2I と I2I の共同学習 ＋ 多水準・二言語キャプション）→ **SFT** → **少ステップ蒸留** → **RLHF**。

SFT の記述が明快である。事前学習は分散の大きい分布を作るので、SFT の目的は局所的な修正ではなく「**生成分布を高忠実な部分多様体へ絞り込む**」ことにある——**多様性最大化の体制から品質最大化の動作点へ移す**。副作用の破滅的忘却は、知識グラフと BM25 の希少度スコアによる動的再サンプリングで抑える。仕上げに **モデルマージ**：能力次元ごとにわずかに偏らせた複数の SFT 変種を作り、重みを線形補間する（$\theta_{\text{final}}=\sum_i \alpha_i \theta_i$）。推論時のルーティングなしにパレート最適を狙う。

RLHF は 2 段階だが、**分担が明確**である。**DPO は客観的で VLM が二値判定できる次元だけ**（テキストレンダリングの正誤、物体の個数）に限定し、VLM で選好ペアを大量生成して人間が検証する。主観的な次元（美しさ・様式）は選好ペアの調達が遅いので DPO では扱わず、**GRPO のオンライン精錬**に回す。DPO 側にはカリキュラム学習も入れ、差異が中程度のペアから始めて徐々に難しくする。

## 実験結果と知見

- **Alibaba AI Arena の Elo**（表3）: Z-Image-Turbo が **1025 で世界 4 位、オープンソース 1 位**。Seedream 3.0（1012）・**Qwen-Image（1008、20B）**・GPT Image 1（986）・FLUX.1 Kontext Pro（950）を上回る。上位は Imagen 4 Ultra（1048）、gemini-2.5-flash-image（1046）、Seedream 4.0（1039）。
- **FLUX.2 dev との直接比較**（表4）: 222 サンプル・各 3 名の評価で、G+S Rate 87.4%（Bad 12.6%）。**6B 対 32B** での結果である。
- **テキスト描画**: CVTG-2K で Z-Image が平均 Word Accuracy **0.8671** と首位（GPT Image 1 の 0.8569、Qwen-Image の 0.8288 を上回る）。OneIG の Text スコアは英語 **0.987**・中国語 **0.988** で、Turbo に至っては英語 0.994。LongText-Bench は英 0.935 / 中 0.936 で Qwen-Image（0.943/0.946）に次ぐ 2〜3 位。
- **OneIG 総合**: 英語トラック **0.546 で 1 位**（Qwen-Image 0.539、GPT Image 1 0.533）。中国語トラックは 0.535 で 2 位。
- **GenEval** 0.84（2 位タイ）、**DPG-Bench** 88.14（3 位）。
- **PRISM-Bench**: 英語で **Turbo が 77.4 の 3 位**、ベース版（75.6）を上回る。中国語は Z-Image が 75.3 の 2 位。
- **編集**: ImgEdit 総合 4.30 で 3 位（Qwen-Image-Edit [2509] の 4.35 に僅差）、GEdit で 3 位。
- **効率**: 8 NFE、H800 でサブ秒、**16GB VRAM 未満**で動作。蒸留前の SFT モデルは CFG 込みで約 100 NFE を要していた。

## コードから確認できたこと（2026-08-26 追記）

公式のリファレンス実装（`code_analysis/Z-Image/`・コミット `26f23ed`・Apache-2.0）を読み、本ページの記述を検証した。**本 wiki で「報告書がある原典をコードで裏取りした」初めての例**である（[[summaries/2025-flux2]] は報告書が存在しないケース、[[summaries/2026-ai-toolkit]] はツールのケース）。以下、ファイル名と行番号を添える。

### 訂正の記録：注意ヘッド数は 32 ではなく 30

**本ページと [[questions/z-image-architecture-and-lora-placement]] が原典 表2 から引いていた「注意ヘッド数 32」は、公開チェックポイントの構成と一致しない。**

| | 原典 表2（[[translations/2025-z-image]]） | **リファレンス実装** |
| --- | --- | --- |
| 隠れ次元 | 3840 | 3840 ✅ |
| 層数 | 30 | 30 ✅（ただし下記の 34 ブロック） |
| **注意ヘッド数** | **32** | **30** ❌ |
| head_dim | （記載なし。32 なら 120） | **128** |

根拠は 2 つある。第一に `src/config/model.py` L28–32 が `DIM=3840` / `N_HEADS=30` / `N_KV_HEADS=30` を既定とし、`src/zimage/transformer.py` L338–339 が

```python
head_dim = dim // n_heads
assert head_dim == sum(axes_dims)      # 128 == 32 + 48 + 48
```

を**実行時に強制する**。ヘッド数が 32 なら head_dim は 120 になり、この assert が落ちる。U-RoPE の次元配分 $(d_t,d_h,d_w)=(32,48,48)$ が表 2 に載っている以上、**表 2 は自分自身と整合していない**。

第二に、**パラメータ数の算術が 30 ヘッド構成を支持する**。1 ブロックは attention $4\times3840^2=58.98\text{M}$ ＋ FFN $3\times3840\times10240=117.96\text{M}$ ＋ 変調 $3.95\text{M}$ ≈ 180.9M。変調ありブロック 32 個（統一 30 ＋ 画像側リファイナ 2）＋ 変調なし 2 個 ＋ `cap_embedder` 9.83M ＝ **6.153B** で、報告書の 6.15B にぴったり乗る。

**結論への影響**：head_dim が 120 か 128 かは LoRA を当てるだけなら影響しない（LoRA は $W_q$ 等の行列全体に当たり、ヘッドへの分割はその後で起きる）。影響するのは**独自に RoPE を実装する場合**だけである。[[questions/z-image-architecture-and-lora-placement]] の訳注が「実チェックポイントで確認したほうがよい」と留保していた点が、これで決着した。**訳文（[[translations/2025-z-image]]）は直さない**——原典に忠実であることが翻訳ページの役割であり、食い違いは原典の側にある。

### 確定した構成値

**「30 層」は統一バックボーンだけを指しており、総ブロック数は 34 である。**

| モジュール | 個数 | タイムステップ変調 | 処理する対象 |
| --- | --- | --- | --- |
| `noise_refiner` | 2 | **あり** | 画像トークンのみ |
| `context_refiner` | 2 | **なし**（`modulation=False`） | テキストトークンのみ |
| `layers` | 30 | あり | 連結された統一系列 |

本文で「入口の 2 ブロック」と書いたのは**モダリティごとに 2 ブロックずつ、合計 4 ブロック**の意味だった（`transformer.py` L308–320）。そして**テキスト側のリファイナはタイムステップ変調を一切受けない**（L317, L544–545）——テキストはノイズ除去の進行度に依存しないのだから自然だが、**報告書はこの非対称に触れていない**。

**「共有され層に依存しない下方射影」の正体は 256 次元である。** `ADALN_EMBED_DIM = 256`（`config/model.py` L3）で、`t_embedder` が 256 次元のベクトルを 1 つ出し（L322）、各ブロックが `adaLN_modulation = Linear(256 \to 4\cdot\text{dim})` で上方射影する（L169）。

$$\underbrace{t \to \mathbb{R}^{256}}_{\text{全層で 1 回}} \;\longrightarrow\; \underbrace{\mathbb{R}^{256} \to \mathbb{R}^{4\times3840}}_{\text{層ごと}}$$

節約量を計算できる。素朴に $\text{dim}\to4\cdot\text{dim}$ とすると 30 層で **17.70 億パラメータ**、実際の構成では **1.18 億**。**約 16.5 億パラメータの差**で、素朴実装なら 6.15B のモデルが約 7.8B に膨らむ計算になる。「パラメータのオーバーヘッドを減らすため」という報告書の一文が、**モデル全体の 2 割強**を意味していたことになる。

**変調は 4 分割で、shift 項がない。** `scale_msa, gate_msa, scale_mlp, gate_mlp` の 4 つだけで、標準的な adaLN-Zero（[[summaries/2023-dit]]）が持つ shift ／ scale ／ gate の 6 分割から**シフトが落とされている**（L180–184）。そしてゲートは

```python
gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
```

——本ページで図から読み取って「ゼロ初期化ゲート」と書いたものは、**厳密には tanh で $(-1,1)$ に有界化されたゲート**である。初期値が 0 付近なら実効的にゼロ初期化として働くが、機構としては「ゲートが暴れないことを恒等的に保証する」ものになっている。Sandwich-Norm はコードでも確認できた——`attention_norm1`（前）と `attention_norm2`（**attention 出力側**）、`ffn_norm1` / `ffn_norm2` の 4 つの RMSNorm が実在する（L163–166, L186–192）。

### 報告書に書かれていない実装事実

| 事実 | 場所 | なぜ重要か |
| --- | --- | --- |
| **トークン順は `[画像, テキスト]`**（画像が先） | `transformer.py` L552 | FLUX 系の `[テキスト, 画像]` と逆。単一ストリームでは順序が RoPE の座標割り当てを決める |
| RoPE の `theta = 256.0` | `config/model.py` L6 | 標準の 10000、FLUX.2 の 2000 と比べ**極端に小さい**（[[position-embedding]]） |
| 軸配分 `[32,48,48]`・軸長 `[1536,512,512]` | 同 L7–8 | **空間軸に厚く配分**。テキスト最大 1536 位置、画像は 1 辺 512 トークン＝8192px 相当 |
| テキストは第 1 軸を **1 から**増分、画像は全トークンが同一の第 1 軸値、**位置 0 はパディング専用** | `transformer.py` L391–443 | 報告書の「テキストは時間次元に沿って増分する」の具体形 |
| テキスト埋め込みは **`hidden_states[-2]`**（最終層の 1 つ手前） | `pipeline.py` L134 | 報告書は「Qwen3-4B」としか書かない。**どの層を取るか**が設計変数（[[summaries/2025-flux2]] は中間 3 層を連結） |
| プロンプトは **Qwen の chat template ＋ `enable_thinking=True`** で整形してから符号化（生成はしない） | `pipeline.py` L110–116 | LLM を「対話モデルとして」呼び出す形のまま埋め込みだけ取る（[[prompt-enhancement]]） |
| 系列を **32 の倍数にパディング**し、**学習済みの `x_pad_token` / `cap_pad_token`** を差し込む。パッドも attention に参加する | L328–329, L390, L427, L508 | 可変長の効率を捨ててカーネル整列を取る設計（[[efficient-attention]]） |
| モデルへは `(1000-t)/1000` を渡し、**出力を符号反転**して標準スケジューラに載せる | `pipeline.py` L223, L274 | 「Z-Image は時刻の向きが逆」という本 wiki の記述の**実装上の帳尻合わせ**が確認できた |
| `calculate_shift` で $\mu$ を計算するが `use_dynamic_shifting=False` のため**捨てられる**。静的 `shift=3.0` | `pipeline.py` L194–202 / `scheduler.py` L85–88 | SD3・FLUX と同じ定数（base 0.5 / max 1.15）を計算しておきながら使わない（[[noise-schedule]]） |
| `cfg_truncation`（終盤で CFG を切る）と `cfg_normalization`（guided 予測のノルム上限）という 2 つのつまみ | `pipeline.py` L227–268 | 報告書に記述なし（[[classifier-free-guidance]]） |
| **`0 < guidance_scale ≤ 1.0` は黙って無効**（`> 1.0` で初めて有効化） | `pipeline.py` L105 | 「弱い CFG を掛けたつもり」が何もしない罠 |
| FFN は SwiGLU・`hidden = dim/3*8 = 10240`・全線形層 bias なし。`n_kv_heads == n_heads` で **GQA は使っていない** | `transformer.py` L70–75, L161 | 報告書の 10240 が導出値であることが分かる |
| attention backend は FA2 / FA3 / varlen / **`mps_flash`（Apple Silicon）** / SDPA / math。デバイスは cuda → TPU → mps → cpu | `utils/attention.py`, `inference.py` L32–49 | 「コンシューマ級ハードウェア」という主張が実装で裏づけられる |

**独立確認**: [[summaries/2026-ai-toolkit]] の Z-Image 実装も学習用スケジューラに**同じ `use_dynamic_shifting: False, shift: 3.0`** を使っている（`extensions_built_in/diffusion_models/z_image/z_image.py` L42–45）。静的シフトであることが 2 つの独立した実装で支持される。

### 実装されていないもの

**このリポジトリは T2I 推論専用であり、編集経路が丸ごと存在しない。** 参照画像の入力、SigLIP-2 の意味エンコーダ、図 10 左下の semantic processor / image processor の 2 経路は**コードに一切現れない**。本ページ「1. S3-DiT」で説明した編集時の構成（参照画像に $t=1$、対象に $t\in[0,1]$）は、**報告書のみを根拠とする記述のままである**——[[questions/z-image-figure10-architecture-walkthrough]] でも同じ書き分けをした。

`README.md` の Model Zoo が公開状況を整理している（**報告書には無い情報**）。

| モデル | 事前学習 | SFT | RL | ステップ | CFG | 多様性 | **微調整のしやすさ** | 公開 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| Z-Image-Omni-Base | ✅ | ❌ | ❌ | 50 | ✅ | High | Easy | 未公開 |
| **Z-Image** | ✅ | ✅ | ❌ | 50 | ✅ | Medium | **Easy** | 2026-01-27 |
| **Z-Image-Turbo** | ✅ | ✅ | ✅ | 8 | ❌ | Low | **N/A** | 2025-11-26 |
| Z-Image-Edit | ✅ | ✅ | ❌ | 50 | ✅ | Medium | Easy | 未公開 |

**Turbo の Fine-Tunability が「N/A」と明示されている**点は、[[questions/lora-merging-base-model-selection]] と [[questions/z-image-architecture-and-lora-placement]] が「ベース版で学習せよ」と述べた根拠を公式の表で裏づける。非蒸留の Z-Image は推奨 28–50 ステップ・CFG 3.0–5.0・`cfg_normalization` は写実で True / 画風で False。

**学習コード・データ・蒸留の実装は一切ない。** Decoupled DMD も DMDR も、データ基盤の 4 エンジンも、コードからは何一つ検証できない。本ページの「限界・批判的視点」で挙げた未検証の主張は、**リファレンス実装の公開によっても解消していない**。

## 限界・批判的視点

- **アブレーションが 1 つもない**。S3-DiT（単一ストリーム）が二重ストリームに対して本当に優れているのか、直接比較が提示されていない。「パラメータ効率が高い」は設計上の理屈であって測定結果ではなく、**本論文の中心的な主張が未検証のまま**である。Sandwich-Norm・低ランク条件射影・3D Unified RoPE についても同様。
- **Decoupled DMD と DMDR の技術的詳細が本論文にない**。「完全な技術的詳細についてはそれぞれの学術論文を参照されたい」として外部の論文に投げている。本レポートで読めるのは主張の要約だけで、CA と DM をどう切り分けたか、再ノイズ付与スケジュールをどう変えたかは分からない。**最も興味深い貢献が、本レポート単体では検証も再現もできない**。
- **主要な評価指標が自社の Arena である**。Elo 4 位という看板の数字は Alibaba AI Arena によるもので、著者らと同じ Alibaba が運営する基盤である。「独立した（independent）」と本文は述べるが、利益相反の可能性は割り引いて読む必要がある。加えて、arXiv の後続版では Arena での順位が異なる記述に更新されている（本 wiki が取り込んだ版は「4 位」と記す）。
- **コストの比較対象がない**。$628K は透明だが、Qwen-Image や FLUX.2 が実際にいくら掛かったかは公表されていないので、「桁違いに少ない」という主張は推定に留まる。また 314K GPU 時間は**最終的な学習ラン**のコストであり、そこに至る試行錯誤・失敗した実験・データ基盤の構築コストは含まれていないと読むのが自然である。
- **Diversity スコアが低い**。OneIG-EN で Z-Image の Diversity は 0.194、**Turbo に至っては 0.139** で、比較対象の中でもかなり低い（SD 1.5 が 0.429、Seedream 3.0 が 0.277）。SFT で「多様性最大化から品質最大化へ移す」と明言している以上これは設計通りの帰結だが、**蒸留がそれをさらに悪化させている**（0.194 → 0.139）ことは本文で議論されない。[[diffusion-distillation]] の既知の限界がそのまま出ている。
- **GenEval の Position が 0.62** と低い（GPT Image 1 の 0.75、Janus-Pro-7B の 0.79）。空間関係の指示追従は弱点として残る。
- **合成データ蒸留を批判しながら、自身も VLM への依存が深い**。プロプライエタリなモデルからの蒸留は拒否するが、キャプション生成・DPO の選好ペア生成・知識グラフのノード命名・視覚的生成可能性の判定はすべて VLM が担う。「閉じたフィードバックループ」の批判は、**キャプションモデルと報酬モデルを自身の出力で再学習する** Active Curation Engine 自体にも一定程度当てはまりうる。
- **PE 込みの評価と PE なしの評価が混在している**。付録は「PE を無効にすれば図 1–3 を再現できる」と述べるが、ベンチマーク数値が PE ありなのか無しなのかは明示されない。6B の性能の何割が PE によるものかを切り分けられない。
- **報告書の構成表が公開チェックポイントと食い違う**（上記「訂正の記録」）。表 2 の注意ヘッド数 32 は、同じ表に載る U-RoPE の次元配分 $(32,48,48)$ と整合しない。**表 2 は自己完結的に検算できたはずの誤りを含んでいる**——アブレーションの欠如とは別種の、記述の精度そのものに関わる問題である。一次資料としてこの報告書を引く際は、構成値は実装で確認したほうがよい。
- **図が 3 枚欠落している**（図 7・8・11）。ar5iv・arXiv HTML のどちらでも生成されておらず、編集データ構築の図解（図 7）と学習パイプライン全体図（図 11）は本文の記述からしか追えない。

## 用語と略称

- **S3-DiT** = Scalable Single-Stream Diffusion Transformer。本論文のアーキテクチャ。テキスト・画像 VAE・視覚意味の各トークンを 1 本の系列に連結して共有重みで処理する。
- **MM-DiT** = Multimodal Diffusion Transformer。SD3 が導入した、テキストと画像に別重みを与える二重ストリーム設計。
- **NFE** = Number of Function Evaluations。画像 1 枚の生成にネットワークを呼ぶ回数。
- **DMD** = Distribution Matching Distillation。**Decoupled DMD**：DMD を CFG-Augmentation（駆動役）と Distribution Matching（正則化役）に分解する改良。**DMDR**：DM 項を RL の正則化子として使う枠組み。
- **CFG** = Classifier-Free Guidance。条件付きと無条件の予測を組み合わせて条件忠実度を上げる手法。推論コストが 2 倍になる。
- **DPO** = Direct Preference Optimization（選好ペアから直接方策を最適化）。**GRPO** = Group Relative Policy Optimization（グループ内の相対評価で優位性を測る）。**RLHF** = Reinforcement Learning from Human Feedback。**SFT** = Supervised Fine-Tuning。
- **reward hacking（報酬ハッキング）**：報酬関数を悪用して高スコアだが望ましくない出力を作る現象。
- **PE** = Prompt Enhancer。ユーザーのプロンプトをモデルが好む形式へ書き換える外付けモジュール。
- **QK-Norm**：attention の query と key を正規化して学習を安定させる。**Sandwich-Norm**：ブロックの入力と出力の両方を正規化する。**RMSNorm**：二乗平均平方根のみで正規化する軽量な変種。
- **RoPE** = Rotary Position Embedding。**3D Unified RoPE**：画像を空間 2 次元、テキストを時間 1 次元に配置して同じ座標系で扱う位置符号化。
- **VAE** = Variational Autoencoder。ここでは Flux VAE を流用。**SigLIP 2**：sigmoid 損失で学習する画像-テキスト対照学習モデル。編集時の参照画像の符号化に使用。
- **flow matching / rectified flow**：ノイズとデータを線形補間し、その速度 $v_t = x_1 - x_0$ を予測する定式化。**logit-normal サンプラー**：タイムステップを中間に集中させる（SD3 由来）。**動的時間シフト**：解像度に応じてノイズ水準をスケールする（Flux 由来）。
- **モデルマージ（model merging）**：複数の微調整済みチェックポイントの重みを線形補間して 1 つにまとめる手法。
- **pHash**（知覚ハッシュ）：画像の視覚的指紋。低レベル重複除去に使う。**BPP** = Bytes Per Pixel。画像の複雑さの代理指標。
- **PageRank**：グラフ上のノードの重要度を測るアルゴリズム。**BM25**：情報検索における語の重み付け指標。
- **AIGC** = AI-Generated Content。学習データから除外する対象。
- **FSDP2** = Fully Sharded Data Parallel（第 2 世代）。**torch.compile**：PyTorch の JIT コンパイラ。
- **CVTG-2K / LongText-Bench / OneIG / GenEval / DPG-Bench / TIIF / PRISM-Bench**：text-to-image の各種ベンチマーク。**ImgEdit / GEdit-Bench**：指示ベース編集のベンチマーク。
- **Elo レーティング**：一対一の勝敗から競技者の強さを推定する方式。ここでは人間のペアワイズ選好に適用。

## 関連ページ

- [[concepts/diffusion-model-architecture]] — S3-DiT が完成させる二重 → 単一ストリームの系譜
- [[concepts/diffusion-distillation]] — Decoupled DMD が書き換える DMD の理解
- [[concepts/reinforcement-learning-for-diffusion]] — DMDR と、客観 DPO ／ 主観 GRPO の分担
- [[concepts/data-curation]] — 4 つのエンジンから成るデータ基盤
- [[concepts/prompt-enhancement]] — PE を凍結して拡散側を合わせる立場
- [[concepts/large-scale-training-infrastructure]] — 314K GPU 時間・$628K という透明なコスト報告
- [[concepts/text-to-image-generation]] — 6B での Elo 4 位という効率の実証
- [[concepts/visual-text-rendering]] — CVTG-2K 首位・OneIG Text 0.987/0.988
- [[concepts/instruction-based-image-editing]] — Z-Image-Edit と編集ペアのグラフ表現
- [[concepts/lora-merging]] — SFT 変種のモデルマージ（LoRA ではなく全重みの線形補間）
- [[summaries/2026-hidream-o1-image]] — 同じ「decoder-only に倣う」動機から VAE ごと捨てた対照例
- [[summaries/2025-hidream-i1]] — 同時期に MoE で容量を増やす逆方向の答え
- [[summaries/2025-qwen-image]] / [[summaries/2026-qwen-image-2]] — 20B の同門。PE の扱いが正反対
- [[summaries/2025-flux-kontext]] — 編集における時間軸オフセットの同型の解
- [[summaries/2025-wan]] — 条件付け機構の省パラメータ化で同じ方向の知見
- [[summaries/2025-flux2]] — 同じく共有変調を採る対照例。コードを一次資料とする方法論の先例
- [[summaries/2026-ai-toolkit]] — Z-Image の LoRA 学習を実装するツール。静的シフトの独立確認元
- [[concepts/position-embedding]] — 3D Unified RoPE の実装値（theta=256・軸配分・パディング位置）
- [[concepts/classifier-free-guidance]] — `cfg_truncation` / `cfg_normalization`
- [[concepts/efficient-attention]] — 32 の倍数パディングと学習済みパッドトークン
- [[concepts/noise-schedule]] — 計算されて捨てられる $\mu$
- [[concepts/flow-matching]] — 時刻の向きの帳尻合わせ
- [[questions/z-image-architecture-and-lora-placement]] — 実モジュール名での LoRA 標的
- [[questions/z-image-figure10-architecture-walkthrough]] — 図 10 の読み解き（編集経路は報告書のみが根拠）
