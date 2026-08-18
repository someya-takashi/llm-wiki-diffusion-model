---
type: concept
aliases: [Text-to-Image, T2I, テキストからの画像生成, text2img]
tags: [text-to-image-generation, latent-diffusion, generative-models, conditional-generation, prompt-enhancement, data-curation, aesthetic-scoring, efficient-attention]
related:
  - "[[latent-diffusion]]"
  - "[[denoising-diffusion]]"
  - "[[controllable-generation]]"
  - "[[subject-driven-generation]]"
  - "[[visual-text-rendering]]"
  - "[[instruction-based-image-editing]]"
  - "[[reinforcement-learning-for-diffusion]]"
  - "[[diffusion-distillation]]"
  - "[[character-consistency]]"
  - "[[pixel-space-diffusion]]"
  - "[[mixture-of-experts-diffusion]]"
  - "[[video-diffusion]]"
  - "[[prompt-enhancement]]"
  - "[[data-curation]]"
  - "[[aesthetic-scoring]]"
  - "[[unified-multimodal-generation]]"
  - "[[efficient-attention]]"
summaries:
  - "[[summaries/2022-latent-diffusion]]"
  - "[[summaries/2023-controlnet]]"
  - "[[summaries/2023-dreambooth]]"
  - "[[summaries/2022-textual-inversion]]"
  - "[[summaries/2023-sdxl]]"
  - "[[summaries/2024-sd3]]"
  - "[[summaries/2025-qwen-image]]"
  - "[[summaries/2026-qwen-image-2]]"
  - "[[summaries/2025-flux-kontext]]"
  - "[[summaries/2025-hidream-i1]]"
  - "[[summaries/2026-hidream-o1-image]]"
  - "[[summaries/2025-wan]]"
  - "[[summaries/2025-z-image]]"
  - "[[summaries/2026-ernie-image]]"
  - "[[summaries/2025-hunyuanimage-3]]"
  - "[[summaries/2024-sana]]"
updated: 2026-08-19
---

# Text-to-Image Generation（テキストからの画像生成）

**Text-to-Image Generation（テキストからの画像生成, T2I）** とは、「a painting of the last supper by Picasso」のような自然言語の記述（テキストプロンプト）を入力として、それに合致する画像を生成するタスクである。拡散モデルの時代に入って爆発的に実用化し、Stable Diffusion・DALL·E 2・Imagen といったモデルで一般に広く知られるようになった、現在の画像生成 AI の中心的応用である。

本ページは text-to-image 生成という概念の俯瞰と、代表手法（特に拡散ベース）を解説する。テキスト条件をどうモデルに与えるか（条件付け機構）と、条件への忠実度をどう高めるか（guidance）が技術的な核になる。

## 技術的な要点

text-to-image を成立させるには 2 つの要素が要る。

1. **テキスト条件付け（conditioning）**：テキストを画像生成モデルに「効かせる」仕組み。テキストをテキストエンコーダ（Transformer や CLIP テキストエンコーダ）で埋め込みに変換し、生成側へ注入する。拡散モデルでは [[latent-diffusion]] が導入した **cross-attention（クロスアテンション）** が標準的で、U-Net の中間特徴をクエリ、テキスト埋め込みをキー・値として注意を取る。
2. **ガイダンス（guidance）**：テキスト条件への忠実度を高める手法。**classifier-free guidance（CFG, 分類器なしガイダンス）** が事実上の標準で、条件付き予測と無条件予測の差を増幅して条件への従い具合を強める。CFG の強さ（スケール $s$）を上げると条件忠実度は上がるが多様性は下がるトレードオフがある（[[classifier-free-guidance]]）。

評価は **FID（Fréchet Inception Distance, 生成画像の品質）** と **IS（Inception Score）** を、テキスト・画像ペアのベンチマーク **MS-COCO** 検証セット上で測るのが慣例。学習には大規模な画像・テキストペアデータ（**LAION-400M / LAION-5B** など）が使われる。

## 代表手法

### Latent Diffusion / Stable Diffusion（Rombach ら 2022）

[[latent-diffusion]] は、潜在空間での拡散に cross-attention でテキスト条件を注入することで text-to-image を実現した。LAION-400M で学習した 14.5 億パラメータの LDM-KL-8 を、classifier-free guidance 併用（LDM-KL-8-G）で MS-COCO で評価すると、GLIDE（6B）など遥かに大きいモデルに匹敵する FID を、大幅に少ないパラメータで達成した（[[summaries/2022-latent-diffusion]] 表2、図5）。これが **Stable Diffusion** として一般公開され、text-to-image を一般ユーザーの手元に届けた。

<figure>

![](../../raw/assets/2022-latent-diffusion/img_cr_text2img_sign_sample-43.jpg)

<figcaption>図5（再掲, [[summaries/2022-latent-diffusion]] より）: LAION で学習した LDM-8 (KL) による、ユーザー定義テキストプロンプトからの生成サンプル。200 DDIM ステップ・無条件ガイダンス s=10.0。</figcaption>
</figure>

Stable Diffusion の高解像度後継が **SDXL**（[[summaries/2023-sdxl]]）で、テキスト条件付けを **2 つのテキストエンコーダ**（CLIP ViT-L ＋ OpenCLIP ViT-bigG の出力連結、context dim 2048）＋ **pooled text embedding** に強化し、~1024² 級の高解像 T2I をオープンモデルで実現した。詳細は [[latent-diffusion]]。

### その他の代表手法（今後の ingest で拡充）

LDM と同時期に、拡散ベースの **GLIDE**・**Imagen**（いずれも大規模テキストエンコーダ＋ピクセル空間カスケード拡散）や、unCLIP ベースの **DALL·E 2** が登場した。**Imagen**（Saharia ら 2022）は T5-XXL 言語モデルでテキストを埋め込み、64×64 のベース拡散モデル＋2 段の超解像（[[super-resolution]]）でカスケード生成する。これらは専用原典としては未取り込みだが、DreamBooth（[[summaries/2023-dreambooth]]）の土台モデルとして言及されている。専用の記述は今後の原典取り込み時に本ページへ追記する。

なお、これら text-to-image 拡散モデルのバックボーンも U-Net 一択から変わりつつある。**DiT（Diffusion Transformer, [[diffusion-model-architecture]]）** の登場後、Stable Diffusion 3・Sora など後続の主要な text-to-image／動画モデルが Transformer バックボーンを採用している（DiT 自体はクラス条件付き生成だが、テキスト条件への drop-in を著者が予見していた）。**Stable Diffusion 3**（[[summaries/2024-sd3]]）はこれを **MM-DiT**（テキストと画像に別重み・attention で双方向結合）として実装し、3 テキストエンコーダ（CLIP×2＋T5）と組み合わせてテキスト理解・タイポグラフィを大きく改善、学習定式化も rectified flow（[[flow-matching]]）に切り替えた。

### 大きくしない方へ振る — Sana（NVIDIA / MIT 2024）

2024 年の text-to-image は「大きくすれば強くなる」路線にあった。PixArt-α の 0.6B から SD3 の 8B、FLUX の 12B、Playground v3 の 24B へ。**Sana**（[[summaries/2024-sana]]）はこの潮流に対する明確な反対提案で、**0.6B のまま FLUX-12B に匹敵する**ことを目指す。

やり方は「拡散モデルの計算量＝トークン数 × トークンあたりのコスト × ステップ数」という分解に沿って、**3 つとも同時に削る**というものである。

- **トークン数**：$f32$ の深圧縮オートエンコーダ（AE-F32C32P1）。同じトークン数でも「$f8$ VAE ＋ 4×4 パッチ化」より収束も品質も良いことをアブレーションで示した（[[image-tokenizer]]）。
- **トークンあたりのコスト**：ReLU 線形注意で $O(N^2) \to O(N)$。劣化は Mix-FFN の 3×3 depthwise conv で補償（[[efficient-attention]]）。
- **ステップ数**：Flow-DPM-Solver で 10–20 ステップ（[[diffusion-sampling]]）。

結果は 4096×4096 で FLUX-dev 比 **104 倍**のスループット、GenEval 0.64（FLUX-dev 0.67）、**16GB のラップトップ GPU で 1024px を 1 秒未満**。ただし FLUX-schnell の 0.71 には届かず、「12B と同等」は総合スコアに限った話である。

本ページの文脈で Sana が重要なのは、**同じ「効率で勝つ」という主張でも、Z-Image（[[summaries/2025-z-image]]）や ERNIE-Image（[[summaries/2026-ernie-image]]）とは削る場所が違う**ことである。後者はデータとポストトレーニングを磨いてパラメータ数を抑えるが、注意も VAE も標準的なままだ。Sana は逆に、**表現の粒度そのものを粗くする**方向で効率を稼ぐ。前者の路線は主流になり、後者の線形注意は主流にならなかった——ただし Sana のもう 1 つの提案、**デコーダのみの LLM をテキストエンコーダに使う**という点は、この直後に Qwen-Image・HunyuanImage 3.0・ERNIE-Image が全面採用する潮流の先駆けになっている。

### Qwen-Image（Qwen Team 2025）— MLLM をテキストエンコーダに据える

**Qwen-Image**（[[summaries/2025-qwen-image]]）は、SD3 が確立した MM-DiT＋rectified flow の路線を引き継ぎつつ、**テキスト条件付けの部品自体を differentiate した**世代である。CLIP や T5 のような専用テキストエンコーダを並べる代わりに、**凍結したマルチモーダル LLM（Qwen2.5-VL, 7B）1 本**を条件エンコーダに据える。視覚と言語がすでに整合済みのモデルを使うことで T2I に適した表現が得られ、しかも**画像入力を受けられるため編集タスク（[[instruction-based-image-editing]]）へ地続きに拡張できる**のが設計上の狙いである。バックボーンは 20B の MMDiT で、テキストを画像の対角線上に配置する新しい位置符号化 **MSRoPE** を導入する（[[diffusion-model-architecture]]）。

Qwen-Image がとくに押し進めたのが **[[visual-text-rendering]]（画像内テキストの描画）** で、VAE デコーダのテキスト特化微調整・3 種のデータ合成・非テキスト→テキストのカリキュラム学習を積み上げ、**中国語のような表語文字で他モデルを大きく引き離した**（ChineseWord 総合 58.30 対 GPT Image 1 の 36.14）。さらに事前学習の後に **DPO と Flow-GRPO による強化学習**（[[reinforcement-learning-for-diffusion]]）を回し、GenEval を 0.87→0.91 へ押し上げている。オープンソースながら人手評価（AI Arena, Elo）でトップ 3 に入り、T2I 基盤モデルが「事前学習だけでなく事後学習まで含めた総合設計」の段階に入ったことを示す。

その後継 **Qwen-Image-2.0**（[[summaries/2026-qwen-image-2]]）は、この総合設計をさらに推し進めて **生成と編集を単一モデルに統一**した（[[instruction-based-image-editing]]）。条件エンコーダを Qwen3-VL に更新し、**16 倍圧縮のトークナイザ**（[[image-tokenizer]]）でネイティブ 2K 生成を可能にし、学習は「解像度と編集データ比率を同時に上げる」カリキュラム（256/512 → 512/1024/2048、T2I:TI2I を 9:1 → 7:3）で回す。仕上げに **5 種の報酬モデルによる RLHF**（[[reinforcement-learning-for-diffusion]]）と **DMD 蒸留で 4 NFE 化**（[[diffusion-distillation]]）を重ね、LMArena で ELO 1168（世界 9 位）に達した。

**プロンプトの前処理も設計対象になった**点も新しい。複雑なレイアウト（インフォグラフィック・ポスター）では生成品質がプロンプトの具体性に強く依存するため、Qwen-Image-2.0 は **Prompt Enhancer（PE）**——曖昧なユーザー指示を構造化された詳細プロンプトへ書き換える 9B のモジュール——を前段に挟む。学習データの作り方が巧妙で、**詳細なアノテーションをわざと「劣化」させて口語的な短い指示を作り、その逆操作を思考連鎖（CoT）として記録する**逆工学パイプラインを使う。さらに書き換え結果を凍結生成器に通し、視覚的一貫性・美的品質で GRPO 報酬を与えることで、**下流の画像品質そのものでプロンプト書き換えを最適化**する。

### HiDream-I1 / HiDream-O1-Image（HiDream.ai 2025–2026）— 効率をアーキテクチャで買う

同じ「基盤モデル」の枠内でも、HiDream.ai の 2 本は**効率の稼ぎ方**で本 wiki に新しい軸を持ち込んだ。

- **HiDream-I1**（17B・[[summaries/2025-hidream-i1]]）は、FFN を疎な MoE（混合エキスパート）に置き換えて「容量は増やすが 1 トークンあたりの計算量は据え置く」を狙う（[[mixture-of-experts-diffusion]]）。テキスト条件は Long-CLIP 2 種＋T5-XXL＋Llama 3.1 の中間層という 4 系統の**足し算**で、Qwen-Image の**引き算**（凍結 MLLM 1 本）と正反対の答えを出している。HPSv2.1 平均 33.82 で全カテゴリ 1 位。ただし GenEval の Position は 0.60、DPG-Bench の Global は 76.44 と、**空間関係と大域的なトーン指定に明確な穴**が残った。
- **HiDream-O1-Image**（8B / 200B+・[[summaries/2026-hidream-o1-image]]）は、その穴を別の方向から埋める。VAE も外部テキストエンコーダも捨て、生ピクセル・テキスト・条件を単一の共有トークン空間に置く（[[pixel-space-diffusion]]）。**8B で GenEval 総合 0.90、Position 0.93**——Qwen-Image の 0.76、FLUX.2 [Dev] の 0.73 を大きく上回り、前作の DPG Global も 76.44 → 95.15 へ改善した。著者らはこれを「言語と視覚が同じトークン空間にあるので、意味的概念が空間領域へ精密に接地される」ためだと説明する。

この 2 本が並ぶと、2025→2026 の T2I 基盤モデルが「**パラメータを増やす**」から「**表現空間の断片化を減らす**」へ重心を移したことが見える。

### テキストエンコーダ論争に測って答える — Wan（2025）

上で見た通り、条件エンコーダの設計は 2025 年に真っ二つに割れた——Qwen-Image の「凍結 MLLM 1 本に集約する」引き算と、HiDream-I1 の「Long-CLIP×2＋T5-XXL＋Llama 3.1 中間層を混ぜる」足し算である。どちらも自説を主張するだけで、直接比較はなかった。

動画側から出た **Wan**（[[summaries/2025-wan]]・[[video-diffusion]]）が、この論点を**アブレーションで比べている**。候補は umT5（5.3B, encoder-only）・Qwen2.5-7B-Instruct・GLM-4-9B。結果は **umT5 が勝つ**。理由として引かれるのは HunyuanVideo の観察で、**decoder-only の LLM は因果注意（各トークンは前だけを見る）だが、umT5 は双方向注意なので拡散モデルに適する**——拡散の条件付けは「プロンプト全体を一度に見て意味を固める」作業なので、後ろを見られない表現は不利だ、という理屈である。HunyuanVideo に倣って双方向の token refiner をアダプタとして足しても umT5 が勝ったとも報告される。MLLM（Qwen-VL-7B）との比較では FID がほぼ互角（42.91 対 43.01）だがモデルが大きく、コスト対効果で umT5 を採る。

**「LLM の方が新しくて強いのだから条件エンコーダも LLM にすべき」という直観を、測って否定した数少ない事例**である。ただし動画・二言語という条件下での結果なので、そのまま画像へ一般化はできない——ここが決着したと見るのは早い。

### 規模を増やさずに勝つ — Z-Image（Alibaba 2025）

ここまで見た 2025–2026 年の基盤モデルはいずれもパラメータを増やす方向で進んだ。**Z-Image**（[[summaries/2025-z-image]]）はこれを「**scale-at-all-costs（あらゆる代償を払ってスケールする）**」と名指しして正面から挑戦する。

看板の数字は **60 億パラメータで Alibaba AI Arena の Elo 1025・世界 4 位（オープンソース 1 位）** である。**20B の Qwen-Image（1008）**、GPT Image 1（986）、FLUX.1 Kontext Pro（950）を上回り、上位 3 つのクローズドソース（Imagen 4 Ultra 1048、gemini-2.5-flash-image 1046、Seedream 4.0 1039）に迫る。32B の FLUX.2 dev との直接ユーザースタディでも G+S Rate 87.4% を得ている。しかも **8 NFE・サブ秒・16GB VRAM 未満**で動く。

手段は 4 つの柱の総合である——完全な単一ストリームの **S3-DiT**（[[diffusion-model-architecture]]）、作り込んだ**データ基盤**（[[data-curation]]）、**PE で認知的ギャップを外付けする**設計（[[prompt-enhancement]]）、そして **Decoupled DMD / DMDR** による蒸留と事後学習（[[diffusion-distillation]]）。

本ページの文脈で最も重要なのは、**学習コストを金額で公開している**ことである。総計 **314K H800 GPU 時間 ≒ $628K**、内訳は低解像度事前学習 147.5K / omni 事前学習 142.5K / 事後学習 24K。基盤モデルのコストがここまで具体的に開示された例は少なく、「原理に基づく設計が力任せのスケーリングに匹敵しうる」という主張の検証可能な足場になっている（[[large-scale-training-infrastructure]]）。

もう 1 つ、方法論上の拒否も記録に値する。資源に乏しい研究でよく取られる近道——**プロプライエタリなモデルから合成データを蒸留する**——について、著者らは「**閉じたフィードバックループを作り、誤差の蓄積とデータの均質化を招き、教師モデルに既に存在するものを超えた新しい視覚的能力の創発を妨げる**」と批判し、実世界データのみで学習したと明言する。

弱点も明確である。**OneIG の Diversity が 0.194（Turbo は 0.139）** と低い——SFT で「多様性最大化から品質最大化へ移す」と明言している以上これは設計通りだが、蒸留がそれをさらに悪化させている点は説明されない。GenEval の Position も 0.62 と弱い。そして**アーキテクチャのアブレーションが 1 つもない**ため、6B での成功が S3-DiT のおかげなのかデータ基盤のおかげなのか PE のおかげなのかは切り分けられていない。

### 8B で追う — ERNIE-Image（Baidu 2026）

Z-Image の翌年、**ERNIE-Image**（[[summaries/2026-ernie-image]]）が同じ「効率志向」の路線を 8B で引き継ぐ。診断は Z-Image をそのまま名指しする——大規模化は限界収穫の逓減に直面するが、**6B の Z-Image は複雑な指示追従と中国語のテキストレンダリングで明確な限界を示す**。そこで 8B で両者の間を狙う。

構成要素はいずれも**外から借りて小さくする**方向に振れている：VAE は FLUX.2 VAE を流用、テキストエンコーダは意図的に小さい **Ministral-3（3B）**、認知的なギャップは外付けの PE で埋める（[[prompt-enhancement]]）。

結果は本ページの流れの中で見ると意味が明確になる。

- **GenEval で 0.89（PE なし）と最高**。とくに **Position が 0.86** で、Qwen-Image 0.76、Z-Image 0.62 を大きく上回る。本ページで繰り返し弱点として現れてきた**空間関係の指示追従**が、ここで明確に前進した。
- **人間評価で総合 2 位・オープンソース 1 位**（Nano Banana 2.0 に次ぐ）。8B が 20B の Qwen-Image 系を上回る。
- **OneIG-EN で 0.575** とクローズドソースの Nano Banana 2.0（0.578）に肉薄。

もっとも、本ページにとって最も価値があるのは順位ではなく **PE ありなしを分けて報告している**ことだろう。Z-Image の項で「評価が PE 込みか否かが曖昧」と書いたが、ERNIE-Image は両方を出し、しかも**方向が一様でない**——GenEval では PE なしの方が高く（0.89 対 0.87）、OneIG の Reasoning では PE ありが大幅に上（0.295 → 0.357）。**プロンプト追従の評価が「モデルの能力」を測っているのか「システムの能力」を測っているのか**という区別が、ここでようやく数字で扱えるようになった。

ERNIE-Image はまた、美的評価そのものを主題化した点でも本 wiki に新しい（[[aesthetic-scoring]]）。**LAION-Aesthetic の人間ラベルとの順位相関が 0.29 しかない**という測定は、各社が「高品質データで学習した」と述べるときの足場に直接関わる。

### 批判される側の言い分 — HunyuanImage 3.0（Tencent 2025）

上で見た Z-Image と ERNIE-Image は、どちらも冒頭で「大規模化は限界収穫の逓減に直面する」と論じ、その代表例として **Hunyuan-Image-3.0 の 80B** を名指ししていた。当事者の主張も記録しておくべきだろう（[[summaries/2025-hunyuanimage-3]]）。

Tencent 側の言い分は、実は「大きさ」ではない。**既存の MoE 大規模言語モデル（Hunyuan-A13B）をそのまま土台にした**結果が 80B（活性 13B）なのであって、画像モデルを大きく作ったわけではない、という構図である。狙いは LLM が既に持っている世界知識・推論能力・指示追従を**内側で**使うことにある（[[unified-multimodal-generation]]）。したがって「素朴なスケーリング」という批判が正確に的を射ているとは限らない——ただし**推論コストが一切報告されない**ので、Z-Image の 8.19GB VRAM・$628K と並べて比較することはできない。**効率論争は片方だけが数字を出している**状態にある。

結果は GSB（Good/Same/Bad）による人間評価で、HunyuanImage 2.1 に対し相対勝率 14.10%、Seedream 4.0 に +1.17%、Nano Banana に +2.64%、GPT-Image に +5.00%。ただし後ろの 3 つは 1,000 サンプルの評価としては**統計的に有意か怪しい僅差**であり、信頼区間も検定も示されない。「クローズドソースに匹敵する」は妥当だが「上回る」と読むのは慎重であるべきだろう。

#### SSAE — CLIP Score 批判の具体例

本ページの評価をめぐる議論に、この論文は鋭い一例を加える。既存ベンチマーク（T2I-CompBench・GenEval）への批判が 2 点あり、

1. プロンプトが定型的で短い（「a photo of a [object] with [attribute]」）ため、実世界の複雑な指示を負荷試験できない。
2. **CLIP Score が人間の判断の代理として貧弱**である。

2 番目に付く具体例が効いている——CLIP Score は「**『少年の下の蜂』と『蜂の下の少年』を取り違えた画像を高く評価しうる**」。CLIP は語の集合としての一致は測れても、**空間的な関係の向き**を区別できない。本ページで繰り返し見てきた「Position が弱い」という各モデルの傾向が、**そもそも指標の側でも捉えられていなかった**可能性を示唆する。

代替として提案される **SSAE** は、500 プロンプトから LLM で 3,500 のキーポイントを抽出し、12 の細粒度フィールド（主要／副次的な被写体の名詞・属性・動作、場面、カメラのショット、様式、構図）に分類する。別の LLM が幻覚ポイントを除去して欠落を補い、人間が修正する。以降のモデル比較ではこのポイント集合を**固定**し、MLLM が CoT で 0-1 照合する。ただし**これも自作ベンチマークであり、図で「すべてのフィールドで同等」と述べるのみで数値表がない**。

### 評価が「AI っぽさ」に報いる問題 — bakeyness

FLUX.1 Kontext（[[summaries/2025-flux-kontext]]）が提起した評価上の論点も記しておく価値がある。T2I ベンチマークが「**どちらの画像を好むか**」という単一の問いに頼ると、**過飽和の色・中心被写体への過度な集中・強いボケ・均質なスタイル**という特徴的な「AI 的美学」が有利になってしまう。著者らはこれを **bakeyness** と名づけ、単一軸の選好評価が**モデルをその方向へ最適化させてしまう**危険を指摘する。

対策として評価を **プロンプト追従・美的品質・写実性・タイポグラフィ・推論速度**の 5 次元に分解した。すると「Recraft は美的品質が高いがプロンプト遵守が弱い」「GPT-Image-1 はその逆」といった**カテゴリ間のトレードオフ**が見えるようになる。SDXL が「COCO zero-shot FID は悪化したが人間評価では勝った」と報告した問題（[[summaries/2023-sdxl]]）と同じ、**指標が実際の良さと乖離する**論点の続きであり、本ページ冒頭で触れた FID/IS の限界を補強する。

## 下流応用：personalization（被写体駆動生成）

汎用 text-to-image は「テキストで言えるもの」しか作れず、**特定個体（ユーザーの犬・バッグなど）の同一性を保ったまま**新文脈で再生成することは苦手である。これを補うのが **[[subject-driven-generation]]（被写体駆動生成 / personalization）** で、少数画像で T2I モデルを特定の被写体に適応させる。代表手法 **DreamBooth**（[[summaries/2023-dreambooth]]）は、Imagen / Stable Diffusion を 3〜5 枚の画像で fine-tune し、被写体を一意識別子「[V] [class noun]」に紐づけて再文脈化・視点変更・スタイル変換を可能にする。対極にあるのが **Textual Inversion**（[[summaries/2022-textual-inversion]]）で、こちらはモデルを**一切変更せず凍結**し、テキスト埋め込み空間に概念を表す擬似単語の埋め込みだけを学ぶ——同じ「personalization」を、モデルを fine-tune するか語彙を 1 単語拡張するかの対照的な両極として確立した二大原典である。テキスト条件付け（本ページ）が「何を描くか」を制御するのに対し、personalization は「誰／どの個体を描くか」を制御する補完的な軸である。

## 既存知識との接続

- [[latent-diffusion]]：cross-attention 条件付けと潜在空間拡散により text-to-image を実用解像度・計算量で実現した代表手法。
- [[denoising-diffusion]]：text-to-image 拡散モデルの生成エンジンとなる拡散の基礎。
- [[classifier-free-guidance]]：テキスト条件への忠実度を高める標準手法。LDM の高品質化も CFG に依存する。
- [[controllable-generation]]：テキストだけでは難しい空間構図（姿勢・形・レイアウト）の精密制御を、ControlNet が空間条件画像で補完する。テキスト＋空間条件の併用が実用の主流。
- [[subject-driven-generation]]：テキストでは指定しきれない「特定個体の同一性」を、少数画像の fine-tune（DreamBooth）で埋め込む下流 personalization。
- [[visual-text-rendering]]：画像の中に「読める文字」を描く部分問題。汎用の生成品質とは別軸で、VAE の再現限界・データのロングテール・位置符号化が効く。
- [[instruction-based-image-editing]]：生成した／与えられた画像を自然言語の指示で編集する下流タスク。Qwen-Image のようにマルチモーダル LLM を条件エンコーダにすると、T2I と編集が同じモデルで扱える。
- [[reinforcement-learning-for-diffusion]]：事前学習後に人間の選好や報酬モデルで仕上げる事後学習。プロンプト忠実度（位置・属性・個数）の底上げに効く。
- [[character-consistency]]：生成した被写体を複数画像・複数ターンにわたって「同じもの」として保つ軸。T2I 単体では見えないが、実運用（ブランド・ストーリーテリング）では決定的になる。
- [[diffusion-distillation]]：蒸留で NFE を 1〜4 まで落とし、対話的な創作ワークフローを可能にする。

## 参考文献（summaries）

- [[summaries/2022-latent-diffusion]] — Latent Diffusion Models（cross-attention によるテキスト条件付き拡散の確立、Stable Diffusion の基盤）
- [[summaries/2023-controlnet]] — Adding Conditional Control to Text-to-Image Diffusion Models（空間条件による text-to-image の制御）
- [[summaries/2023-dreambooth]] — DreamBooth（少数画像による被写体駆動 personalization、Imagen/SD を fine-tune）
- [[summaries/2022-textual-inversion]] — Textual Inversion（凍結モデルの埋め込み空間に擬似単語を学ぶ、personalization のもう一方の原典）
- [[summaries/2023-sdxl]] — SDXL（高解像 T2I の後継：2 テキストエンコーダ＋pooled embedding、micro-conditioning）
- [[summaries/2024-sd3]] — Stable Diffusion 3（MM-DiT＋3 テキストエンコーダ＋rectified flow、タイポグラフィ強化）
- [[summaries/2025-qwen-image]] — Qwen-Image（凍結 Qwen2.5-VL を条件エンコーダに据えた 20B MMDiT。中国語テキスト描画で大差の SOTA、DPO/GRPO 事後学習、編集まで統一）
- [[summaries/2026-qwen-image-2]] — Qwen-Image-2.0（Qwen3-VL＋f16 トークナイザで生成と編集を単一モデルに統一。Prompt Enhancer・5 報酬 RLHF・DMD 蒸留、LMArena ELO 1168）
- [[summaries/2025-flux-kontext]] — FLUX.1 Kontext（T2I と編集を単一の rectified flow に統一。bakeyness 批判と 5 次元評価、1024² を 3〜5 秒）
- [[summaries/2025-hidream-i1]] — HiDream-I1（17B・疎な MoE と 4 系統のハイブリッドテキスト符号化。HPSv2.1 全カテゴリ 1 位、Position と Global に穴）
- [[summaries/2025-wan]] — Wan（テキストエンコーダを直接アブレーション。umT5 の双方向注意が decoder-only LLM を上回る）
- [[summaries/2025-hunyuanimage-3]] — HunyuanImage 3.0（80B/活性 13B の MoE LLM を土台に。GSB でクローズドソースに匹敵、SSAE で CLIP Score の空間関係の盲点を指摘）
- [[summaries/2026-ernie-image]] — ERNIE-Image（8B。GenEval 0.89・Position 0.86、人間評価でオープンソース 1 位。PE ありなしを分離報告し、美的評価を主題化）
- [[summaries/2025-z-image]] — Z-Image（6B・$628K で Elo 世界 4 位。S3-DiT＋データ基盤＋PE＋Decoupled DMD の総合。合成データ蒸留を明示的に拒否）
- [[summaries/2026-hidream-o1-image]] — HiDream-O1-Image（8B で GenEval 0.90・Position 0.93。VAE も外部テキストエンコーダも持たない統一トークン空間）
- [[summaries/2024-sana]] — Sana（0.6B で FLUX-12B に匹敵。深圧縮 AE・線形注意・Flow-DPM-Solver の 3 方向同時削減、4096px で 104×、16GB ラップトップ GPU で動作。decoder-only LLM テキストエンコーダの先駆け）
