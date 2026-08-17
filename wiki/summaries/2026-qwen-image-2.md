---
type: summary
source_path: raw/papers/Qwen-Image-2.0 Technical Report.md
source_kind: paper
title: "Qwen-Image-2.0 Technical Report"
authors: [Qwen Team (Alibaba), Bing Zhao, Chenfei Wu, Jiahao Li, Junyang Lin, Jingren Zhou]
year: 2026
venue: "arXiv:2605.10730（テクニカルレポート）"
ingested: 2026-06-25
tags: [text-to-image-generation, instruction-based-image-editing, visual-text-rendering, diffusion-distillation, reinforcement-learning-for-diffusion, image-tokenizer, diffusion-model-architecture]
translation: "[[translations/2026-qwen-image-2]]"
---

# Qwen-Image-2.0 — 生成と編集を 1 つのモデルに統一する

> 原典: [[translations/2026-qwen-image-2]] ・ `raw/papers/Qwen-Image-2.0 Technical Report.md`
> 著者・年・出典: Qwen Team（Alibaba）, arXiv:2605.10730, 2026年4月

## 一言まとめ

初代 Qwen-Image（[[summaries/2025-qwen-image]]）の後継で、**text-to-image 生成と画像編集をパイプライン切り替えなしに単一モデルで扱う「全能型（omni-capable）」基盤モデル**。条件エンコーダを Qwen3-VL に更新し、**16 倍圧縮の VAE**（[[image-tokenizer]]・[[summaries/2026-qwen-image-vae-2]]）でネイティブ 2K 生成を可能にし、**プロンプトエンハンサ**・**5 種の報酬モデルによる RLHF**・**DMD 蒸留（4 NFE）**（[[diffusion-distillation]]）を積み上げた。LMArena で ELO 1168・世界 9 位（中国モデル 1 位）。

## 背景と問題意識

初代 Qwen-Image が中国語テキスト描画で大きく先行した後も、実務のワークフローには 5 つの壁が残っていた——**超長文テキスト**（文字数が増えるほどグリフが歪みレイアウトが崩れる）、**多言語タイポグラフィ**（英中以外は字形も字間も読み順も崩れる）、**高解像度の写実性**（2K 以上でテクスチャが反復し照明が破綻する）、**複雑な指示追従**（複数実体・空間制約で概念が欠落する）、**推論コスト**。

さらに根本的な問題として、既存システムは**どれか 1 軸でしか秀でない**——写実的な絵か正確な文字か、生成か編集か——という点があった。別々のパイプラインに頼らず、あるいは品質を犠牲にせず、これらすべてを 1 つのモデルで提供できるものはほとんどない。本論文の主題はこの**統一**である。

## 提案手法 / 主張

### アーキテクチャ：Qwen3-VL ＋ f16 VAE ＋ MMDiT

初代からの変更点が要点である。

- **条件エンコーダを Qwen3-VL に更新**（凍結）。役割は初代と同じだが、注目すべき変更として、**視覚表現 $h_x$ は VAE 潜在 $\mathcal{E}_x$ で置き換えられる**（アーキテクチャ図では Qwen3-VL の視覚出力経路に ✗ が付く）。初代の「MLLM 意味特徴と VAE 特徴を*両方*流す二重符号化」から、**画像側は VAE 潜在に一本化**された形で、テキスト系列と連結して MMDiT へ入る。
- **16 倍圧縮 VAE（f16c64）**。従来の 8 倍から圧縮率を倍にし、DiT の系列長を 1/4 に削減してネイティブ 2K 生成を現実的にした。詳細は姉妹論文 [[summaries/2026-qwen-image-vae-2]]。
- **MMDiT の細部**：位置符号化は初代の MSRoPE を継承。加えて (1) **バイアスなし変調**——従来のアフィン変調 $h'=\alpha h+\beta$ からバイアス項を落として純粋な乗法 $h'=\alpha h$ に、(2) **SwiGLU**——テキスト・画像同時学習で活性値が過大になりニューロンが早期飽和する問題への対処。いずれも大規模同時学習の安定化が動機である。

### プロンプトエンハンサ（PE）— 逆工学でデータを作る

複雑なレイアウト（インフォグラフィック・ポスター・ストーリーボード）では、生成品質はモデルの能力だけでなく**プロンプトがレイアウトや視覚的階層をどれだけ具体的に指定しているか**に左右される。しかし実ユーザーのプロンプトは粒度がばらばらである。そこで **Qwen3.5-9B ベースの書き換えモジュール**を挟む。

学習データの作り方が巧い——**詳細なアノテーションを「劣化」させて口語的なユーザープロンプトを作り、その逆操作を CoT（Chain-of-Thought, 思考連鎖）として記録する**。劣化戦略（文体の単純化・口語化・照明やテクスチャの省略）を確率的にサンプルすることで難易度の異なる訓練例が自動生成でき、得られる三つ組 $(P_{\text{short}}, \text{CoT}, P_{\text{fine}})$ でモデルは「短い指示から詳細な意図を復元する過程」ごと学べる。学習は SFT → GRPO の 2 段で、RL では書き換えたプロンプトを凍結生成器に通して視覚的一貫性・美的品質・規則ベースのテキスト制約で報酬を与える（＝**下流の画像品質で書き換えを最適化する**）。

### 学習：3 段階 ＋ RLHF ＋ 蒸留

- **多段階学習**：事前学習 700K ステップ（256/512、T2I:TI2I = 9:1、lr 1e-4）→ 継続事前学習 250K（512/1024/2048、7:3、lr 2e-5）→ SFT 10K（lr 1e-5）。**解像度と編集データ比率を同時に上げていく**カリキュラム。
- **RLHF**（[[reinforcement-learning-for-diffusion]]）：**5 つのタスク特化型報酬モデル**——美的品質・画像テキスト整合・肖像（T2I 用）、指示追従・視覚的一貫性（TI2I 用）。単一次元への過剰最適化を避けるため重みを動的に調整する。最適化は GRPO だが、**CFG の扱いにハイブリッド戦略**を採る：ロールアウトのサンプリングでは CFG を使って高品質な候補を作り報酬評価を信頼できるものにしつつ、**無条件分岐は方策最適化の目的から除外**して計算を削る。
- **少ステップ蒸留**（[[diffusion-distillation]]）：**DMD（Distribution Matching Distillation）** を採用し、**4 NFE の生徒が 40 ステップの教師に匹敵**する品質を達成。著者らは既存の蒸留研究がほぼ ImageNet のクラス条件付きに限られており、T2I や編集での有効性は未探求だった点を指摘する。

### データフライホイール — 失敗を自動で振り分ける

運用面の貢献として、**誤り帰属（error attribution）駆動の閉ループ**を導入する。評価・バッドケースマイニング・ユーザーフィードバックで失敗を集め、原因に応じて 3 トラックへ自動で振り分ける：**RL トラック**（整合の問題→報酬方策の調整）、**事前学習トラック**（知識の欠如→ベクトル検索でデータを補充）、**プロンプトエンジニアリング・トラック**（能力はあるが指示理解の問題→PE で入力を洗練）。手動介入は必要なデータレビューのみに限定される。

## 実験結果と知見

- **LMArena**（実ユーザーのブラインド対戦・ELO）：**ELO 1168、世界 9 位、中国モデル中 1 位**。Nano Banana を上回る。初代 Qwen-Image シリーズから生成・編集の双方で大幅な改善。
- **VAE 再構成**（表1）：f16c64 で ImageNet PSNR 33.42 / SSIM 0.9225 と、**8 倍圧縮の従来 VAE と同等以上を 16 倍圧縮で達成**。ただしテキスト画像 PSNR は 32.81 で、初代 Qwen-Image の f8c16（36.63）には及ばない——圧縮率を倍にした代償が出ている。
- **定性評価**：中国語テキスト、肖像、複雑な中国語詩の編集、同一性保持で GPT-Image-2・NanoBanana Pro・Wan2.7 Pro・Seedream 5.0 Lite・Qwen-Image-2512 と比較し、**文字レベルの正確さと空間的な結びつきを同時に保つ唯一のモデル**と報告。とくに題画詩（縦書き右から左・余白への配置）のような文化的な組版規約まで守れる点を強調する。
- **蒸留**：4 NFE で 40 ステップ相当の視覚品質・意味整合・構成の一貫性を維持。

## 限界・批判的視点

- **定量ベンチマークがほとんどない**。初代が GenEval・DPG・ChineseWord 等で詳細な数値を出したのに対し、本レポートは **LMArena の ELO と VAE 再構成表以外はすべて定性比較**である。生成・編集の主要主張（超長文レンダリング、多言語、2K 写実性、指示追従）に対応する定量評価が提示されていない。
- **アブレーションが皆無**。バイアスなし変調・SwiGLU・PE・報酬重みの動的調整・ハイブリッド CFG など設計判断が多いが、個々の寄与を切り分けた比較はない。
- **テキスト再構成は初代 f8 VAE に劣る**（32.81 対 36.63 PSNR）。高圧縮の代償であり、[[visual-text-rendering]] の観点では一長一短。姉妹論文の f16c128 はこれを解決するが、本モデルが採用したのは f16c64 である。
- **多言語対応の具体性が乏しい**。「広範な言語」と述べるが、どの言語でどの程度かの内訳は示されない。
- 比較対象（GPT-Image-2、NanoBanana Pro、Wan2.7 Pro、Seedream 5.0 Lite）はいずれもクローズドで、再現検証が難しい。

## 既存 wiki との接続

Qwen-Image-2.0 は [[text-to-image-generation]] と [[instruction-based-image-editing]] を**同一モデル・同一学習パラダイムで統一**した点が最大の主張で、初代（[[summaries/2025-qwen-image]]）が二重符号化で編集の一貫性を確保したのに対し、本作は画像側を VAE 潜在に一本化しつつ、マルチタスク学習の比率制御（9:1→7:3）で生成と編集の統一を図る。アーキテクチャは SD3 由来の MM-DiT（[[diffusion-model-architecture]]）の系譜にあり、[[flow-matching]] ベースの学習も継承する。新たに wiki へ持ち込むのは **[[diffusion-distillation]]**（DMD による 4 NFE 化——[[diffusion-sampling]] の「ソルバーを速くする」とは別系統の高速化）と **[[image-tokenizer]]**（16 倍圧縮 VAE、詳細は [[summaries/2026-qwen-image-vae-2]]）である。RLHF は [[reinforcement-learning-for-diffusion]] の実例として、初代の 2 報酬から 5 報酬・ハイブリッド CFG へと精緻化された形になる。

## 用語と略称

- **omni-capable（全能型）**：生成・編集など複数タスクを 1 モデルで扱えること。
- **MMDiT** = Multimodal Diffusion Transformer（テキストと画像を共有バックボーンで同時にモデル化する拡散 Transformer）。
- **MSRoPE** = Multimodal Scalable RoPE（テキストを画像の対角線上に配置する位置符号化。初代 Qwen-Image が導入）。
- **bias-free modulation（バイアスなし変調）**：$h'=\alpha h+\beta$ の $\beta$ を落として $h'=\alpha h$ にする。
- **SwiGLU**：$\Phi_1(x)\otimes\sigma(\Phi_2(x))$ の形のゲート付き活性化（$\sigma$ は SiLU）。活性値の暴走を抑える。
- **f16c64**：空間圧縮率 16 倍・潜在チャネル 64 の VAE 設定（[[image-tokenizer]]）。
- **PE** = Prompt Enhancer（プロンプトエンハンサ。曖昧な指示を構造化された詳細プロンプトへ書き換えるモジュール）。
- **CoT** = Chain-of-Thought（思考連鎖。ここでは「劣化の逆操作」を推論トレースとして学習に使う）。
- **RLHF** = Reinforcement Learning from Human Feedback（人間フィードバックからの強化学習）。
- **GRPO** = Group Relative Policy Optimization（グループ内で報酬を標準化してアドバンテージにする方策最適化）。
- **CFG** = Classifier-Free Guidance（条件忠実度を高めるガイダンス。ここではロールアウトのみに使う）。
- **DMD** = Distribution Matching Distillation（教師と生徒の**分布**を一致させる蒸留。軌道を追うのではなくスコアの差で学ぶ）。
- **NFE** = Number of Function Evaluations（生成 1 枚あたりのネットワーク評価回数。少ないほど速い）。
- **ELO / LMArena**：実ユーザーのブラインド対比較から相対的強さを推定するレーティングとその運用基盤。
- **T2I / TI2I** = text-to-image / text-image-to-image（生成／編集）。

## 関連ページ

- [[concepts/text-to-image-generation]]
- [[concepts/instruction-based-image-editing]]
- [[concepts/diffusion-distillation]]
- [[concepts/image-tokenizer]]
- [[concepts/reinforcement-learning-for-diffusion]]
- [[concepts/visual-text-rendering]]
- [[concepts/diffusion-model-architecture]]
- [[summaries/2025-qwen-image]]
- [[summaries/2026-qwen-image-vae-2]]
- [[translations/2026-qwen-image-2]]
