---
type: concept
aliases: [Text-to-Image, T2I, テキストからの画像生成, text2img]
tags: [text-to-image-generation, latent-diffusion, generative-models, conditional-generation]
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
updated: 2026-08-17
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

### Qwen-Image（Qwen Team 2025）— MLLM をテキストエンコーダに据える

**Qwen-Image**（[[summaries/2025-qwen-image]]）は、SD3 が確立した MM-DiT＋rectified flow の路線を引き継ぎつつ、**テキスト条件付けの部品自体を differentiate した**世代である。CLIP や T5 のような専用テキストエンコーダを並べる代わりに、**凍結したマルチモーダル LLM（Qwen2.5-VL, 7B）1 本**を条件エンコーダに据える。視覚と言語がすでに整合済みのモデルを使うことで T2I に適した表現が得られ、しかも**画像入力を受けられるため編集タスク（[[instruction-based-image-editing]]）へ地続きに拡張できる**のが設計上の狙いである。バックボーンは 20B の MMDiT で、テキストを画像の対角線上に配置する新しい位置符号化 **MSRoPE** を導入する（[[diffusion-model-architecture]]）。

Qwen-Image がとくに押し進めたのが **[[visual-text-rendering]]（画像内テキストの描画）** で、VAE デコーダのテキスト特化微調整・3 種のデータ合成・非テキスト→テキストのカリキュラム学習を積み上げ、**中国語のような表語文字で他モデルを大きく引き離した**（ChineseWord 総合 58.30 対 GPT Image 1 の 36.14）。さらに事前学習の後に **DPO と Flow-GRPO による強化学習**（[[reinforcement-learning-for-diffusion]]）を回し、GenEval を 0.87→0.91 へ押し上げている。オープンソースながら人手評価（AI Arena, Elo）でトップ 3 に入り、T2I 基盤モデルが「事前学習だけでなく事後学習まで含めた総合設計」の段階に入ったことを示す。

その後継 **Qwen-Image-2.0**（[[summaries/2026-qwen-image-2]]）は、この総合設計をさらに推し進めて **生成と編集を単一モデルに統一**した（[[instruction-based-image-editing]]）。条件エンコーダを Qwen3-VL に更新し、**16 倍圧縮のトークナイザ**（[[image-tokenizer]]）でネイティブ 2K 生成を可能にし、学習は「解像度と編集データ比率を同時に上げる」カリキュラム（256/512 → 512/1024/2048、T2I:TI2I を 9:1 → 7:3）で回す。仕上げに **5 種の報酬モデルによる RLHF**（[[reinforcement-learning-for-diffusion]]）と **DMD 蒸留で 4 NFE 化**（[[diffusion-distillation]]）を重ね、LMArena で ELO 1168（世界 9 位）に達した。

**プロンプトの前処理も設計対象になった**点も新しい。複雑なレイアウト（インフォグラフィック・ポスター）では生成品質がプロンプトの具体性に強く依存するため、Qwen-Image-2.0 は **Prompt Enhancer（PE）**——曖昧なユーザー指示を構造化された詳細プロンプトへ書き換える 9B のモジュール——を前段に挟む。学習データの作り方が巧妙で、**詳細なアノテーションをわざと「劣化」させて口語的な短い指示を作り、その逆操作を思考連鎖（CoT）として記録する**逆工学パイプラインを使う。さらに書き換え結果を凍結生成器に通し、視覚的一貫性・美的品質で GRPO 報酬を与えることで、**下流の画像品質そのものでプロンプト書き換えを最適化**する。

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
