---
type: summary
source_path: raw/papers/HunyuanImage 3.0 Technical Report.md
source_kind: paper
title: "HunyuanImage 3.0 Technical Report"
authors: [Tencent Hunyuan Foundation Model Team]
year: 2025
venue: arXiv:2509.23951（テクニカルレポート）
ingested: 2026-08-18
tags: [unified-multimodal-generation, mixture-of-experts-diffusion, position-embedding, diffusion-model-architecture, image-tokenizer, reinforcement-learning-for-diffusion, prompt-enhancement, data-curation, text-to-image-generation, latent-diffusion]
translation: "[[translations/2025-hunyuanimage-3]]"
---

# HunyuanImage 3.0: 自己回帰と拡散を 1 つの MoE LLM に同居させる 80B

> 原典: [[translations/2025-hunyuanimage-3]] ・ `raw/papers/HunyuanImage 3.0 Technical Report.md`
> 著者・年: Tencent Hunyuan Foundation Model Team・2025 年 9 月・arXiv:2509.23951

## 一言まとめ

**事前学習済みの MoE 大規模言語モデル（総 80B / トークンあたり活性 13B）を拡張し、テキストは自己回帰・画像は拡散という異なる目的関数を単一のバックボーンに同居させたネイティブなマルチモーダルモデル**。理解・生成・言語モデリングを 1 本の系列で扱い、Chain-of-Thought を内蔵する。

## 背景と問題意識

**この論文は本 wiki において特別な位置にある**。直前に取り込んだ Z-Image（[[summaries/2025-z-image]]）と ERNIE-Image（[[summaries/2026-ernie-image]]）は、どちらも冒頭で「**大規模化は限界収穫の逓減に直面する**」と論じ、その代表例として **Hunyuan-Image-3.0 の 80B** を名指ししていた。今回はその当事者の主張を直接読むことになる。

Tencent 側の言い分は、実は「大きさ」ではない。**LLM を出発点にする**という点にある。彼らの構図はこうだ——最先端の画像生成システム（Seedream 4.0、Nano Banana、GPT-Image）は大半がクローズドソースで、透明性と再現性を欠く。そこで既に存在する強力な MoE LLM（**Hunyuan-A13B**）を土台にし、視覚エンコーダと VAE を接ぎ木して画像を扱えるようにする。80B という数字は「画像モデルを大きく作った」結果ではなく、「**既存の言語モデルをそのまま使った**」結果である。

この違いは重要で、本 wiki が追ってきた 2 つの系譜の交点にあたる。

- **拡散側から LLM へ寄る**：Qwen-Image が凍結 MLLM を条件エンコーダに据え（[[summaries/2025-qwen-image]]）、HiDream-O1-Image が decoder-only LLM をバックボーンに転用し（[[summaries/2026-hidream-o1-image]]）、Z-Image が単一ストリーム化を進めた。
- **LLM 側から拡散へ寄る**：Transfusion / JanusFlow の系譜。**言語モデルはそのままに、画像トークンだけ拡散で扱う**。

HunyuanImage 3.0 は後者に属する。そして重要なのは、**MoE がここで初めて「言語モデル由来の遺産」として画像生成に持ち込まれた**ことである。HiDream-I1（[[summaries/2025-hidream-i1]]）は DiT の FFN を意図的に MoE 化したが、こちらは**もともと MoE だった LLM を使ったら結果的に MoE になった**——動機がまったく違う。

## 提案手法 / 主張

### 1. ハイブリッドな離散-連続モデリング

<figure>

![](../../raw/assets/2025-hunyuanimage-3/model.png)

<figcaption>図3（引用）: 1 つの Decoder-Only Transformer（Hunyuan-A13B）が 3 つのモードを担う。左＝画像理解（Gen. Encoder＝VAE と Und. Encoder＝ViT の両方が同じ画像を受け取り、次トークン予測で応答）、中央＝言語モデリング、右＝画像生成（テキストトークン＋ノイズを載せた Gen. Encoder から**速度（拡散予測）**を出し、Gen. Decoder で画像に戻す）。</figcaption>
</figure>

中核は 1 行で言える——**テキストトークンは自己回帰的な次トークン予測、画像トークンは拡散**。同じ Transformer の同じ重みが、トークンの種類によって違う目的関数で学習される。

これは本 wiki が見てきた「統一」とは質が違う。HiDream-O1-Image も Z-Image も、テキストと画像を同じ系列に置くところまでは同じだが、**学習目的は拡散ひとつ**だった。HunyuanImage 3.0 は**目的関数のレベルで異種混合**である。詳細は [[unified-multimodal-generation]] に整理した。

**バックボーン**は Hunyuan-A13B：**64 エキスパート・トークンあたり 8 個活性・共有 MLP 1 個**で、活性化パラメータ約 13B（[[mixture-of-experts-diffusion]]）。

### 2. VAE は f16・32 チャネル、パッチ化なし

[[image-tokenizer]] にとって明確な主張がある。従来の DiT 系は **f8 の VAE ＋ 2× のパッチ化層**を組み合わせて実効的に 16 倍の空間圧縮を得ていた。HunyuanImage 3.0 は「**f16 の VAE 単体の方が単純かつ効果的で、優れた画像生成品質をもたらす**」と主張する。潜在は 32 次元。

同じ実効圧縮率に 2 通りの到達路があり、**どちらで払うか**という設計判断である。Qwen-Image-VAE-2.0（[[summaries/2026-qwen-image-vae-2]]）が「$f8 \to f16$ へ圧縮を強め、失った容量はチャネル $C$ で補償する」と論じたのと同じ方向で、独立した支持になっている。

**条件画像には二重エンコーダ**を使う。VAE の潜在特徴と視覚エンコーダの特徴を**連結する**。これは従来の統一モデルが「理解には ViT 特徴、生成には VAE 特徴」とタスクごとに分離していたのと対照的で、両方を常に使うことで**理解と生成のパイプラインを切り替える必要がなくなる**。プロジェクタも 2 種類あり、VAE 側は**タイムステップ変調された残差ブロック**、ViT 側は 2 層 MLP。

### 3. Generalized Causal Attention — 「穴」の話

<figure>

![](../../raw/assets/2025-hunyuanimage-3/attention.png)

<figcaption>図4（引用）: (a) 通常のケース。Text 行は階段状（因果的）だが、Gen Image / Cond Image の行は同一画像セグメント内で塊状に埋まる（完全注意）。緑枠＝画像理解、青枠＝text-to-image。(b) 複数 Gen Image の学習時。後続のトークンから最初の Gen Image の列が見えない**「穴」**が下三角に空いている。</figcaption>
</figure>

規則は 2 行である。

- **テキストトークン**：先行するマルチモーダルトークンのみに注意（因果的）。
- **画像トークン**：先行するすべて **＋ 同じ画像セグメント内の後続する画像トークン**にも注意（そのセグメント内では完全注意）。

HiDream-O1-Image のハイブリッド注意（[[summaries/2026-hidream-o1-image]]）と同型の解に独立に到達しているが、HunyuanImage 3.0 は**多画像学習時の落とし穴**まで踏み込む。1 つの学習系列に複数の Gen Image があると、**文脈中の Gen Image は後続トークンから見えてはならない**——さもないと学習時と推論時で条件が食い違う。推論時には生成済み画像が Cond Image に変わるので、この状況は起こらない。結果として学習マスクの下三角に「穴」が空く。**学習と推論の非対称性を明示的に扱った記述**として実装上の価値が高い。

### 4. Generalized 2D RoPE — 判断基準が「LLM を壊さないこと」

<figure>

![](../../raw/assets/2025-hunyuanimage-3/rope_2d.png)

<figcaption>図5（引用）: 11 トークンの 1D 系列（テキスト 1–3、画像 4–9、テキスト 10–11）を 2D 位置へ写す。テキストは対角線上（1,1),(2,2),(3,3) …）に留まり、画像は 2 行 3 列の矩形ブロックとして 2 つのテキスト区間の中間に置かれる。</figcaption>
</figure>

[[position-embedding]] の中心的な事例である。1 次元位置 $n$ に対する RoPE は $[\cos(n\theta_0), \cos(n\theta_1), \dots]$ だが、これを 2 次元座標 $(x,y)$ へ**異方的に**一般化する：

$$[\cos(x\theta_{0}),\cos(y\theta_{1}),\dots,\sin(x\theta_{0}),\sin(y\theta_{1}),\dots]$$

**偶数番目の周波数は $x$ に、奇数番目は $y$ に割り当てる**わけである。テキストトークンは $x=y=n$（対角線上）に置かれるので、**画像がなければ厳密に元の 1D RoPE に退化する**。

ここが他と決定的に違う。Qwen-Image の MSRoPE も FLUX.1 Kontext の仮想タイムステップも Z-Image の 3D Unified RoPE も、「テキストと画像をどう同じ座標系に置くか」を解いていたが、**HunyuanImage 3.0 の設計基準は「事前学習済み LLM の言語能力を壊さないこと」**である。ゼロから学習するモデルには存在しない制約で、LLM から出発する設計に固有の要請といえる。

### 5. Automatic Resolution — 画像の形をモデルが決める

DiT 系は画像サイズとアスペクト比をユーザーが指定する前提だった。HunyuanImage 3.0 は語彙に `<img_size_256>`, `<img_size_512>`, … と `<img_ratio_0>` 〜 `<img_ratio_32>`（1:4 から 4:1）を足し、**モデルが文脈から適切な形状トークンを予測する**。ユーザーが「3:4」「縦向き」と書けばそちらに導かれる。予測された形状に基づいて 2D RoPE が組まれる。

「LLM の語彙を拡張して制御を埋め込む」という発想で、拡散モデル側の設計からは出てきにくい。

### 6. ネイティブ CoT — プロンプトエンハンサをモデルの中に入れる

[[prompt-enhancement]] で 4 通りの設計（Wan の凍結書き換え、Qwen-Image-2.0 の PE を RL、HiDream-O1 の推論品質を報酬化、Z-Image の拡散側を PE に合わせる SFT）を対比したが、これは**5 つ目の点**である——**外付けのモジュールを置かず、モデル自身が「思考」の局面を通る**。

引き出し方は 2 種類の推論データによる小規模な微調整である。**T2T**（テキスト→テキスト：プロンプトから精緻化された記述への段階的推論）と **T2TI**（テキスト→テキスト→画像：抽象的な概念から視覚的な顕現までのワークフロー全体）。重要な実装上の点として、**推論部分のトークンもまた自己回帰的な次トークン予測でモデル化される**——AR と拡散の同居が、ここで直接効いている。

### 7. 事後学習が 5 段階（本 wiki 最長）

| 段階 | 目的 | 手法の要点 |
| --- | --- | --- |
| **SFT** | 基礎的な品質 | 段階的により高品質なサンプルへ |
| **DPO** | **構造的な歪みの抑制** | SFT モデルの出力を高品質／低品質にアノテートして選好ペアを作る |
| **MixGRPO** | 整合・写実性・美的魅力 | **ハイブリッドな ODE–SDE サンプリング**で GRPO を flow モデルへ拡張 |
| **SRPO** | 写実性と明瞭さ | 潜在にノイズ事前分布を注入し**単一ステップで復元**、軌道の**初期区間**を最適化、**肯定・否定の両テキストガイダンス**から微分可能な報酬 |
| **ReDA** | 視覚品質 | 高品質画像集合が定義する**高報酬分布**との乖離を最小化 |

SRPO の設計思想が面白い。ノイズ除去の**初期区間**を狙うのは「そこがモデルの改善の自由度が最も大きい領域だから」であり、[[diffusion-sampling]] の「タイムステップによって仕事の内容が違う」という理解が RL の適用範囲の選択に使われている。また ReDA は「選好ペアを比べる」でも「スカラー報酬を最大化する」でもなく**分布どうしを近づける**ので、Z-Image の DMDR（分布マッチング項を正則化子に使う・[[summaries/2025-z-image]]）と発想が近い。

### 8. SSAE — CLIP Score 批判が具体的

既存ベンチマーク（T2I-CompBench、GenEval）への批判が 2 点。

1. **プロンプトが定型的で短い**（「a photo of a [object] with [attribute]」）。実世界の複雑な指示を負荷試験できない。
2. **CLIP Score が人間の判断の代理として貧弱**。ここに強い具体例が付く——CLIP Score は「**『少年の下の蜂』と『蜂の下の少年』を取り違えた画像を高く評価しうる**」。

SSAE は 500 プロンプトから **3,500 のキーポイント**を LLM で抽出し、**12 の細粒度フィールド**（主要／副次的な被写体の名詞・属性・動作、場面の名詞と属性、カメラのショット、様式、構図）に分類する。別の LLM が幻覚ポイントを除去し欠落を補完し、人間が修正する。以降のモデル比較ではこのポイント集合を**固定**して、MLLM が CoT で 0-1 照合する。

## 実験結果と知見

- **GSB（Good/Same/Bad）**：1,000 プロンプト、100 名超の専門評価者、**チェリーピッキングなしの 1 回推論**。HunyuanImage 2.1 に対し相対勝率 **14.10%**。Seedream 4.0 に **+1.17%**、Nano Banana に **+2.64%**、GPT-Image に **+5.00%**。
- **SSAE**：すべての細粒度フィールドで主要モデルと同等。
- **データ**：100 億枚超から **45% 未満**に絞り、最終的に約 **50 億枚**。加えて**インターリーブ関係の学習用に 1 億件超の画像ペア／マルチ画像**を構築。
- **エキスパート活性化の分析**（§5.3.1）——本 wiki にとって最も価値のある実験結果である。

### エキスパートは本当に専門化するのか

<figure>

![](../../raw/assets/2025-hunyuanimage-3/expert_activation.png)

<figcaption>図8（引用）: 左＝エキスパートのモダリティ選好のヒートマップ（濃いほど画像トークンに特化）。右＝各層における、画像で活性化されるエキスパート分布とテキストで活性化されるそれとの KL ダイバージェンス。層が深くなるほど KL が増大する。</figcaption>
</figure>

[[mixture-of-experts-diffusion]] は HiDream-I1 単独に依存しており、限界として「**エキスパートが何を専門化したかの分析がない**」「**原典が 1 本しかない**」の 2 点を挙げていた。HunyuanImage 3.0 はその両方を埋める。

1,000 プロンプトで text-to-image 生成を実行し、各層・各エキスパートについて画像トークンとテキストトークンの活性化回数を数える。結果は明快で、**層が深くなるほど画像分布とテキスト分布の KL ダイバージェンスが増大し、エキスパートがモダリティごとに専門化していく**。著者らの解釈は「MoE は異なるモダリティに対する責任を専門化されたエキスパートに分散させることでマルチモーダルモデリングを強化しうる」。

「専門化が起きるはずだ」という期待が、**測定によって支持された最初の事例**である。ただし分かるのは「モダリティで分かれる」ところまでで、HiDream-I1 が期待していた「スタイルごと・被写体ごとに専門家が育つ」という細粒度の専門化は依然として検証されていない。

## 限界・批判的視点

- **アブレーションが 1 つもない**。f16 単体が f8＋パッチ化より良い、二重エンコーダが分離型より良い、Generalized 2D RoPE が有効——いずれも主張として述べられるだけで、比較実験が示されない。「f16 VAE 単体の方が優れた品質をもたらすことを**実証する**」と書きながら、その実証は本文に存在しない。
- **推論コストが一切報告されない**。80B の重みを載せる必要があり、13B が活性化される。**MoE はメモリを減らさない**（[[mixture-of-experts-diffusion]]）ので、実際に動かすためのハードウェア要件は相当に厳しいはずだが、レイテンシも VRAM も学習コストも書かれていない。Z-Image が 8.19GB VRAM・$628K を公開したのと対照的で、両者の「効率」論争は**片方だけが数字を出している**状態にある。
- **GSB の勝率が僅差である**。Seedream 4.0 に +1.17%、Nano Banana に +2.64% は、1,000 サンプルの評価では統計的に有意かどうか怪しい水準である。信頼区間も検定も示されない。「クローズドソースに匹敵する」という結論自体は妥当だが、「上回る」と読むのは慎重であるべきだろう。
- **SSAE も自作ベンチマークである**。CLIP Score 批判は正当だが、代わりに置いた指標も設計者の選択（12 フィールドの分類体系、キーポイントの抽出方針）を含む。図 6 で「すべてのフィールドで同等」と述べるのみで、**数値表がない**。
- **公開されるのは画像生成モジュールのみ**。「ネイティブなマルチモーダルモデル」を謳いながら、理解能力も image-to-image も公開されない（後者は「進行中」）。統一モデルの主張の大半が、公開された成果物では検証できない。
- **MixGRPO・SRPO・ReDA の詳細が外部に投げられている**。5 段階の事後学習は本 wiki 最長だが、各手法の説明は数文ずつで、MixGRPO の ODE–SDE ハイブリッドが具体的にどうサンプリングを分けるのか、SRPO の「単一ステップ復元」がどう微分可能な報酬を通すのかは読み取れない。Z-Image が Decoupled DMD を外部論文に投げたのと同じ問題である。
- **CoT の効果が定量化されていない**。「マルチモーダルの性能を有意に改善する」と結論で述べるが、CoT あり／なしの比較数値はない。[[prompt-enhancement]] で見た ERNIE-Image の PE ありなし分離報告（[[summaries/2026-ernie-image]]）と比べると後退している。
- **「45% 未満を保持」の内訳が粗い**。3 段階のうち最終段階の重複除去が 0.5% と明示される一方、第 1・第 2 段階の除去率は書かれない。
- **後続論文からの評価**。Z-Image と ERNIE-Image はどちらも本モデルを「素朴なスケーリングの例」として批判の対象に挙げる。ERNIE-Image は 8B で GenEval 0.89 を達成しており、**80B が必要だったのかという問いは残る**。ただし HunyuanImage 3.0 の主張は「画像モデルを大きくした」ではなく「既存の LLM を使った」なので、批判が正確に的を射ているとは限らない。

## 用語と略称

- **MoE** = Mixture-of-Experts（混合エキスパート）。複数の並列 FFN からルーターがトークンごとに一部だけを選んで通す仕組み。ここでは 64 エキスパート中 8 個＋共有 MLP 1 個。
- **活性化パラメータ（activated parameters）**：総パラメータのうち、1 トークンの処理で実際に使われる分。ここでは 80B 中 13B。
- **LLM** = Large Language Model。**Hunyuan-A13B**：本モデルのバックボーンとなる decoder-only の MoE LLM。
- **VAE** = Variational Autoencoder。**f16**：空間解像度を 16 分の 1 にする圧縮率。**パッチ化（patchification）**：潜在をさらにパッチにまとめて系列長を減らす操作。
- **ViT** = Vision Transformer。ここでは画像理解のための視覚エンコーダ。
- **AR** = Autoregressive（自己回帰）。**次トークン予測**：直前までのトークンから次を予測する言語モデルの標準的な学習目的。
- **Generalized Causal Attention**：テキストは因果マスク、画像は同一セグメント内で完全注意という混合の注意機構。
- **RoPE** = Rotary Position Embedding。回転行列で相対位置を注意に埋め込む位置符号化。**Generalized 2D RoPE**：周波数を $x$ と $y$ に交互に割り当てて 2 次元化し、画像がなければ 1D RoPE に退化する設計。
- **CoT** = Chain-of-Thought（思考の連鎖）。**T2T / T2TI**：テキスト→テキスト／テキスト→テキスト→画像の推論データ。
- **T2I / LM / MMU / INTL**：text-to-image 生成／言語モデリング／マルチモーダル理解／インターリーブされたテキスト-画像モデリング。
- **SFT** = Supervised Fine-Tuning。**DPO** = Direct Preference Optimization。**GRPO** = Group Relative Policy Optimization。
- **MixGRPO**：ハイブリッドな ODE–SDE サンプリングで GRPO を flow モデルへ拡張した手法。**SRPO**：軌道の初期区間を狙う勾配誘導型の RL。**ReDA** = Reward Distribution Alignment（高報酬分布との乖離を最小化）。
- **AIGC** = AI-Generated Content。学習データから除外する対象。
- **SSAE**：本論文が提案する構造化された意味的整合の評価指標。**GSB** = Good/Same/Bad（2 モデルの相対比較による人間評価）。
- **CLIP Score**：CLIP の埋め込みでテキストと画像の整合を測る自動指標。本論文の批判対象。
- **Transfusion / JanusFlow**：言語モデルに拡散ベースの画像モデリングを統合した先行研究。
- **KL ダイバージェンス**：2 つの確率分布の隔たりを測る量。ここではエキスパート活性化分布のモダリティ間の差を測る。

## 関連ページ

- [[concepts/unified-multimodal-generation]] — 理解と生成を単一モデルに統合する系統。本論文が最も明確な事例
- [[concepts/position-embedding]] — Generalized 2D RoPE と、テキスト・画像を同じ座標系に置く 5 通りの答え
- [[concepts/mixture-of-experts-diffusion]] — **エキスパートの専門化を実測した初の事例**
- [[concepts/image-tokenizer]] — f16 単体 対 f8＋パッチ化という到達路の選択
- [[concepts/diffusion-model-architecture]] — Generalized Causal Attention と二重エンコーダ
- [[concepts/reinforcement-learning-for-diffusion]] — MixGRPO・SRPO・ReDA の 5 段階事後学習
- [[concepts/prompt-enhancement]] — ネイティブ CoT＝プロンプトエンハンサの内部化
- [[concepts/data-curation]] — 100 億枚から 50 億枚へ、AIGC をソース単位で除去、階層的キャプションスキーマ
- [[concepts/text-to-image-generation]] — SSAE による CLIP Score 批判
- [[summaries/2025-hidream-i1]] — DiT の FFN を意図的に MoE 化した対照例
- [[summaries/2026-hidream-o1-image]] — 同じくハイブリッド注意に到達したが、目的関数は拡散ひとつ
- [[summaries/2025-z-image]] / [[summaries/2026-ernie-image]] — 本モデルを「大規模化の代表例」として批判する側
