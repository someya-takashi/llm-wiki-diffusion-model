# Log — Diffusion Model LLM Wiki

時系列の append-only ログ。`## [YYYY-MM-DD] ingest | <タイトル>` 形式で追記する（CLAUDE.md §5）。
スキーマ変更は `## [YYYY-MM-DD] schema-update | <要点>` で記録する。

## [2026-08-19] ingest | HunyuanImage 3.0 Technical Report

- 取り込み: `raw/papers/HunyuanImage 3.0 Technical Report.md`（ar5iv 由来 markdown・ケース A, arXiv:2509.23951, 2025年9月。Tencent Hunyuan Foundation Model Team）。**Z-Image と ERNIE-Image が「scale-at-all-costs の代表例」として名指しで批判していた 80B の当事者**を直接読む形になった。
- 作成: [[translations/2025-hunyuanimage-3]], [[summaries/2025-hunyuanimage-3]], [[concepts/unified-multimodal-generation]], [[concepts/position-embedding]]
- **新規概念ページ 2 件**（ユーザー回答「unified-multimodal-generation ＋ position-embedding」）:
  - [[concepts/unified-multimodal-generation]] — 統一マルチモーダル生成。**Janus-Pro・BAGEL・Emu3・Show-o・OmniGen・BLIP3-o が本 wiki のほぼすべてのベンチマーク表に比較対象として並びながら、それらが何かを説明する場所がなかった**という認識から新設。統一したい 3 つの動機、テキストと画像の性質の違い（離散/連続・順序・生成方式・注意）を表で整理し、**(A) 全離散 AR／(B) AR＋拡散ハイブリッド／(C) 拡散の単一トークン空間**の 3 分類を立てた。注意マスクの混ぜ方（HiDream-O1 と HunyuanImage 3.0 が独立に同型解へ到達、および多画像学習時の「穴」）、位置符号化における「LLM を壊さない」制約、統一が可能にすること（MoE の専門化・自動解像度・ネイティブ CoT・タスク境界の消失）。
  - [[concepts/position-embedding]] — 位置符号化。RoPE の仕組みから始め、画像を扱うと生じる 3 つの問題（画像は 2 次元／座標系の衝突／複数画像の区別）を立て、**5 通りの答え**を対比した：MSRoPE（Qwen）・仮想タイムステップ（FLUX.1 Kontext）・3D Unified RoPE（Z-Image）・**Generalized 2D RoPE**（本論文）・動画の 3D RoPE（Wan）。**共通構造**（空間座標は使い切っているので画像どうしの区別には別の軸が要る）と**分岐点**（テキストをどこに置くか）を明示した。
- 更新（本文）: [[concepts/mixture-of-experts-diffusion]]（**長年の空白だった「エキスパートが何を専門化したかの分析がない」を実測が埋めた**。動機が逆から来る場合＝MoE LLM の転用、モダリティ専門化の KL 増大、および「浅い層ほど分布が近い」が FLUX.1 系の直観と逆向きという食い違いの記録）, [[concepts/diffusion-model-architecture]]（LLM 側から拡散へ寄る系譜、目的関数の同居、Generalized Causal Attention、二重エンコーダ）, [[concepts/image-tokenizer]]（**同じ実効圧縮率への 2 つの到達路**＝f16 単体 対 f8＋パッチ化。統一モデルではトークナイザへの要求が変わる）, [[concepts/reinforcement-learning-for-diffusion]]（**5 段階の事後学習**。DPO は「壊れを直す」・オンライン RL は「良くする」という割り当て、MixGRPO / SRPO / ReDA）, [[concepts/prompt-enhancement]]（**5 つ目の設計＝書き換えの内部化**。実効パラメータ数が不透明という問題が消える利点と、分離できない／大きな LLM が要るという代償）, [[concepts/text-to-image-generation]]（**批判される側の言い分**と、SSAE の CLIP Score 批判＝「少年の下の蜂」と「蜂の下の少年」）, [[concepts/data-curation]]（**AIGC をソース単位で除去**、階層的キャプションスキーマと構成的合成、双方向検証ループ）, [[concepts/pixel-space-diffusion]]（**統一とピクセル空間化は独立した判断**であることの反例）, [[concepts/video-diffusion]]（時間軸の「発明」と「実在」、位置で分けるか注意で分けるか）, [[concepts/instruction-based-image-editing]], [[concepts/visual-text-rendering]]
- 更新: [[overview]]（「LLM 側から拡散へ寄る」の項）, [[index]]（Summaries / Translations / Concepts＋略称リダイレクト 7 行）
- 画像: ar5iv から **8 枚**を取得（プレースホルダなし、取得失敗なし、欠落した図もなし）。ar5iv の変換が完全だった珍しいケース。
- 翻訳: 本文 §1–6 を全訳。References と §7 貢献者一覧を除外。**付録は存在しない**。表 1（学習ステージ）を markdown 化。図 8 のキャプションは ar5iv で数式が壊れていたため LaTeX を復元して訳した。
- メモ: 本 ingest の要点は 4 つ。(1) **批判される側の言い分が分かった**——Tencent の主張は「大きさ」ではなく「**既存の MoE LLM をそのまま土台にした**」であり、80B は画像モデルを大きく作った結果ではない。ただし推論コストが一切報告されないため、Z-Image の 8.19GB VRAM・$628K と並べられず、**効率論争は片方だけが数字を出している**状態にある。(2) **目的関数レベルの異種混合**——テキストは AR・画像は拡散が同一の重みに同居する点で、HiDream-O1 や Z-Image の「統一」（学習目的は拡散ひとつ）とは質が違う。(3) **[[concepts/mixture-of-experts-diffusion]] の限界が 1 つ解消**——エキスパートのモダリティ専門化が初めて実測された。ただし「様式ごと・被写体ごと」という細粒度の専門化は依然として未検証。(4) **位置符号化が独立したページに値するだけ事例が溜まった**——5 通りの設計が揃い、とくに本論文の「1D RoPE への後方互換」という基準は、ゼロから学習するモデルには存在しない LLM 由来の制約である。**批判的視点として記録した主要な点**: (i) **アブレーションが 1 つもない**——f16 単体・二重エンコーダ・Generalized 2D RoPE のいずれも比較実験がなく、「実証する」と書きながら実証がない、(ii) 推論コスト・VRAM・学習コストがすべて非公開、(iii) GSB の勝率が Seedream 4.0 に +1.17% 等の**僅差**で信頼区間も検定もない、(iv) SSAE も自作ベンチマークで数値表がない、(v) 「ネイティブなマルチモーダルモデル」を謳いながら**公開されるのは画像生成モジュールのみ**（理解も image-to-image も非公開）、(vi) MixGRPO / SRPO / ReDA の詳細が外部論文に投げられている、(vii) **CoT ありなしの比較数値がない**（ERNIE-Image が PE ありなしを分離報告したのと比べると後退）。

## [2026-08-18] ingest | ERNIE-Image Technical Report

- 取り込み: `raw/papers/ERNIE-Image Technical Report.md`（ar5iv 由来 markdown・ケース A, arXiv:2605.25347, 2026年。ERNIE Team / Baidu）。**直前に取り込んだ Z-Image を名指しで引き継ぐ論文**で、8B の単一ストリーム DiT（FLUX.2 VAE ＋ Ministral-3 3B のテキストエンコーダ）。
- 作成: [[translations/2026-ernie-image]], [[summaries/2026-ernie-image]], [[concepts/aesthetic-scoring]]
- **新規概念ページ 1 件**（ユーザー回答「aesthetic-scoring を新設」）:
  - [[concepts/aesthetic-scoring]] — 美的スコアリング。**本 wiki が LAION の美的分類器を Wan・Z-Image・HiDream のデータパイプラインで繰り返し言及しながらページを持たなかった**領域。美的モデルが「データ選別」と「RLHF 報酬」の 2 箇所で働き、**モデルが何を見て育ち何を目指すかの両方を規定する**という位置づけから始め、Likert のスコアドリフト／Elo の比較回数／スイス式トーナメントという方法論の選択、既存予測器の名指しのバイアス（LAION-Aesthetic の SRCC 0.29、ArtiMuse・UniPercept の白黒偏重）、ベンチマーク自体の偏り（Flickr / DPChallenge 由来＝西洋の写真の伝統）、フィルタでなくサンプリング確率としての使い方、そして**自己循環的な評価**という限界までを整理した。
- 更新（本文）: [[concepts/diffusion-distillation]]（**MT-DMD** の節を新設。DMD → DMD2 → Decoupled DMD → DMDR の系譜表、**Capability Drift** の指摘、$\mathcal{O}\in\{CA,DM\}$ でも切り替えるゲーティング、**同一学習インスタンス内での非対称な勾配トポロジー**と軌道に沿った専門家の引き継ぎ。アブレーションがないという留保も明記）, [[concepts/reinforcement-learning-for-diffusion]]（**flow matching 上の DPO** を係数付きで。**L2 が非有界なので拒否サンプルの誤差を膨らませるだけで報酬を稼げる**という失敗モードと Anchor Loss、DMDR との「元の目的関数に錨を下ろす」という共通構図）, [[concepts/data-curation]]（**品質フィルタの足場**への注意喚起＝LAION-Aesthetic の SRCC 0.29 と AIGC フィルタとの逆方向の引き合い、美的スコアをサンプリング確率に使う設計、**事前学習ボトムアップ／SFT トップダウン**の段階別使い分け）, [[concepts/prompt-enhancement]]（**PE ありなしを分離して測る**節を新設。GenEval では PE で下がり OneIG の Reasoning では大きく上がるという非一様性、Diversity がむしろ上がる観察、3B PE と大型 LM PE の規模比較）, [[concepts/text-to-image-generation]]（8B でオープンソース 1 位、**Position 0.86** という空間関係の前進）, [[concepts/visual-text-rendering]]（OCR 前置キャプションの独立した再確認、LongText-Bench 0.973、文字描画と複雑な指示追従の不可分性）, [[concepts/diffusion-model-architecture]]
- 更新: [[overview]]（「『良い画像』の定義を問い直す」の項）, [[index]]（Summaries / Translations / Concepts＋略称リダイレクト 5 行）
- 画像: ar5iv の原典には 11 枚しか埋め込まれていなかったが、**図 5・6・8・11 が ar5iv 側で分解されていた**ため arXiv HTML から構成画像を回収し、計 **29 枚**とした（プレースホルダなし、取得失敗なし）。内訳：
  - **図 6（PE 比較の 3×3）は 9 枚中 1 枚しか残っていなかった**。残り 8 枚（aime / rpg / webpage × wo_pe / w_pe / w_llm_pe）を回収。本 ingest の主要な論点の 1 つを支える図なので価値が高い。
  - **図 5（美的予測器のバイアス比較）は 9 パネル**（標本画像＋各 4 モデルのバープロット）だが、**4 種のバープロットが全パネルで同一 SVG ファイルとして重複参照される形に変換されており、パネルごとの予測スコアとマーカー位置は HTML からは復元不能**。本文が「image 6」「images 4 and 8」等と番号で参照しているため、標本画像 9 枚のみ**文書順で `fig5_img1..9` に連番化**して回収した（連番が本文の参照と一致することを image 6＝アニメ、image 5＝白黒で目視確認済み）。バープロットは訳注で欠落を明示。
  - 図 8・図 11 は各 2 枚のうち 1 枚が欠落していたため回収。
- 翻訳: 本文 §1–5 を全訳（556 行）。References と §6 著者一覧を除外。**付録は存在しない**。表 1–7 を markdown 化。
- メモ: 本 ingest の要点は 3 つ。(1) **本 wiki が使ってきた道具（LAION-Aesthetic）への初めての正面からの検討**——SRCC 0.29 という数字は [[concepts/data-curation]] の「品質フィルタリング」の記述に直接跳ね返るため、あちらにも注意喚起を書き足した。(2) **Z-Image の Decoupled DMD が MT-DMD へ発展した**——CA と DM が別の役割を持つという分離が、教師の割り当てレベルまで貫かれる。副次的に、**Z-Image が外部論文に投げていた DMDR の中身がこの論文で読める**（Z-Image の要約で「本レポート単体では検証できない」と批判した点の一部が補われた）。(3) **PE ありなしの分離報告**——Z-Image の要約で「評価が PE 込みか否かが曖昧」と批判した論点に答える形になり、しかも方向が一様でないことが分かった。**批判的視点として記録した主要な点**: (i) MT-DMD にアブレーションがなく $K$ もゲーティングの学習法も単一教師との比較も示されない、(ii) 人間評価の看板指標「Total HP」の算出方法が説明されていない、(iii) テストセットが社内かつ非公開、(iv) **ERNIE-Image-Aes の評価が自己循環的**（同じチーム・同じプロトコル・同種のアノテータで作った ERIA-1K で評価し、既存データセットでの評価がない）、(v) アーキテクチャの記述が「8B の単一ストリーム DiT」以外ほぼ皆無で Z-Image の S3-DiT との異同が分からない、(vi) Ministral-3 を選んだ根拠となる実験が示されない（Wan の umT5 アブレーションとは水準が違う）、(vii) 学習コストが非公開で Z-Image の $628K と直接比較できない、(viii) 「一般大衆の選好」を掲げながらアノテータは中国の美術系機関の専門家であり、偏りを別の偏りで置き換えていないかは開かれた問い。

## [2026-08-18] ingest | Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer

- 取り込み: `raw/papers/Z-Image_ An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer.md`（ar5iv 由来 markdown・ケース A, arXiv:2511.22699, 2025年11月。Z-Image Team / Alibaba Tongyi-MAI）。**6B・$628K で人間選好 Elo 世界 4 位**を主張し、「scale-at-all-costs」パラダイムと「プロプライエタリモデルからの合成データ蒸留」の双方を明示的に拒否する。
- 作成: [[translations/2025-z-image]], [[summaries/2025-z-image]], [[concepts/data-curation]], [[concepts/prompt-enhancement]]
- **新規概念ページ 2 件**（ユーザー回答「data-curation ＋ prompt-enhancement」。本論文の中核（S3-DiT・Decoupled DMD・DMDR）はいずれも既存ページに収まるため、**複数の原典で繰り返し現れながらページがなかった領域**を選んだ）:
  - [[concepts/data-curation]] — データキュレーション。**「2025 年以降の基盤モデルのレポートは例外なくデータに 1 章を割いているのに受け皿がなかった」**という認識から新設。データが設計対象になった 3 つの理由（規模の収穫逓減／AI 生成画像による汚染＝10% 未満でも劣化／長尾が能力の上限を決める）、5 工程（重複除去・品質フィルタ・再キャプション・構造化と概念均衡・能動的精錬）、タスク特化のデータ構築（編集ペアのグラフ表現・動画フレーム・可制御レンダリング）。Wan / Qwen-Image / HiDream 系 / Z-Image を横断して整理した。
  - [[concepts/prompt-enhancement]] — プロンプト拡張。**同じ問題への 4 通りの答え**を対比する表を中心に据えた（Wan＝凍結して原則指示のみ、Qwen-Image-2.0＝PE を RL、HiDream-O1＝推論品質を報酬化、Z-Image＝**VLM を凍結して拡散側を SFT で合わせる**）。分布のずれという第一の動機と、小さいモデルの認知的ギャップを外付けで埋めるという第二の動機、そして「書き換えから推論へ」という流れ。評価が PE 込みか否かが曖昧という批判も記録。
- 更新（本文）: [[concepts/diffusion-model-architecture]]（**S3-DiT が二重→単一の系譜を畳み切る**。3D Unified RoPE でテキストを時間次元に増分、Sandwich-Norm とゼロ初期化ゲート、低ランク条件射影。**アブレーションがない**という留保も明記）, [[concepts/diffusion-distillation]]（**Decoupled DMD が DMD の理解を書き換える**——駆動役は CFG-Augmentation で分布マッチングは正則化子。**分布マッチング系でも教師を超えうる**事例、および DMDR）, [[concepts/reinforcement-learning-for-diffusion]]（**正則化子を蒸留側から調達する** DMDR、**DPO/GRPO を「VLM で検証可能か」で分担する**設計、要素分解によるクリック式アノテーション）, [[concepts/text-to-image-generation]]（規模を増やさずに勝つ。学習コストの公開、合成データ蒸留の拒否、Diversity 0.194/0.139 という弱点）, [[concepts/visual-text-rendering]]（**キャプションに OCR を先に走らせる**CoT 方式と、OCR 結果を翻訳しない工夫）, [[concepts/instruction-based-image-editing]]（編集データの作り方 3 種と「正確さと現実性は別物」）, [[concepts/lora-merging]]（**基盤モデル自身のモデルマージ**を隣接系統として追加。目的が「概念の共存」でなく「バイアスの中和」である点、素朴な線形和が破綻しない理由）, [[concepts/video-diffusion]]（data-curation / prompt-enhancement への導線）
- 更新: [[overview]]（「規模至上主義への反論」の項）, [[index]]（Summaries / Translations / Concepts＋略称リダイレクト 7 行）
- 画像: ar5iv から **28 枚**を取得（プレースホルダなし、取得失敗なし）。加えて、**ar5iv 側で図 8（Z-Captioner のパイプライン）の複合構造が壊れていた**（本文には `dabenzhong.png` と「Single Image」という断片的ラベルのみが残存）ため、同一図の構成画像 4 枚（image11 / image618 / world_knowledge / ocr_show）を arXiv HTML から回収し、訳注付きで併記した（計 32 枚）。**図 7・11 は ar5iv・arXiv HTML のどちらにも画像が存在しない**（図 7 は原典 markdown に LaTeX の `\begin{overpic}[figures/i2i_data.pdf]` がそのまま残っている）ため訳注で明示。
- **版の相違について**: arXiv HTML は v5 で図 35 枚・Arena 順位の記述も異なる（「8 位」）が、raw/ に置かれた原典は ar5iv 由来の旧版（図 30 枚・「4 位」）である。**原典ファイルに忠実に旧版の内容で翻訳・要約した**。この相違は summaries の批判的視点にも記載した。
- 翻訳: 本文 §1–6 を全訳（869 行）。References と §7 著者一覧を除外。**付録 A（Prompts Used in the Report）はユーザー判断で除外**——図 1–3 を再現するための生成プロンプト集（中国語原文＋英訳、378 行・82 KB）であり、**これらは「モデルへの入力そのもの」なので日本語に訳すと再現性を壊す**ため。スキーマの「付録はデフォルトで含める」の例外にあたる。表 1–15 を markdown 化（TIIF の 24 列と PRISM の 8 トラック×3 値は、数値を落とさず列群で 2 表に分割して再構成）。
- メモ: 本 ingest の要点は 3 つ。(1) **アーキテクチャの系譜が一周した**——二重ストリーム（SD3）→ 二重から単一の二段（FLUX.1・HiDream-I1）→ 完全な単一（Z-Image）。ただし HiDream-O1-Image が VAE ごと捨てたのとは違い Flux VAE を流用しており、単一ストリーム化とピクセル空間化が独立した判断であることが確認できた。(2) **学習コストが初めて金額で公開された**——314K H800 GPU 時間・$628K、しかも段階別内訳付き。直前の Wan の要約で「学習コストが報告されない」を批判点に挙げたところに答える形になった。(3) **Decoupled DMD が既存ページの記述を書き換えた**——[[concepts/diffusion-distillation]] の DMD の説明（分布マッチングが主役）に対し、「実際に駆動しているのは CFG-Augmentation で、分布マッチングは正則化子」という主張を並置した。**批判的視点として記録した主要な点**: (i) **アブレーションが 1 つもない**——中心的主張である S3-DiT の優位が未検証、(ii) Decoupled DMD と DMDR の技術詳細が外部論文に投げられており本レポート単体では再現も検証もできない、(iii) 主要指標の Alibaba AI Arena は著者らと同じ Alibaba が運営、(iv) **Diversity が 0.194（Turbo 0.139）と低く**、蒸留がさらに悪化させている点が説明されない、(v) 合成データ蒸留を批判しながら自身も VLM 依存の閉ループを持つ、(vi) 評価が PE 込みか否かが不明で「6B」の実効規模が不透明。

## [2026-08-18] ingest | Wan: Open and Advanced Large-Scale Video Generative Models

- 取り込み: `raw/papers/Wan_ Open and Advanced Large-Scale Video Generative Models.md`（ar5iv 由来 markdown・ケース A, arXiv:2503.20314, 2025年3月。Wan Team / Alibaba Group）。**本 wiki 初の動画生成の原典**。CLAUDE.md §1 が例示していた `video-diffusion` のスラグをここで開設した。
- 作成: [[translations/2025-wan]], [[summaries/2025-wan]], [[concepts/video-diffusion]], [[concepts/inference-caching]], [[concepts/large-scale-training-infrastructure]]
- **新規概念ページ 3 件**（ユーザー回答「video-diffusion ＋ 推論キャッシュ ＋ インフラ」）:
  - [[concepts/video-diffusion]] — 動画拡散の総論。**「定式化は何も変わらず、変わるのはデータの形と計算の規模だけ」**を軸に、時間軸が持ち込む 4 つの困難（系列長爆発／時間的因果性／動きの品質という新評価軸／時間的一貫性）、3D causal VAE、cross-attention 型 DiT を選ぶ理由、マスクによるタスク統一（I2V・動画継続・first-last frame・補間が同一モデル）、Streamer の滑動窓ノイズ除去キュー、VBench と Wan-Bench。
  - [[concepts/inference-caching]] — **サンプラー改良でも蒸留でもない第 3 の高速化軸**。何を変えるか／再学習の要否／出力が変わるかの 3 軸で既存 2 系統と対比する表を置いた。注意の類似性と CFG の類似性という 2 つの冗長性、後者が [[concepts/classifier-free-guidance]] の「2 倍コスト」に学習なしで効く点、加速率が 1.6 倍程度にとどまるという留保。
  - [[concepts/large-scale-training-infrastructure]] — 本 wiki の完全な空白地帯。**計算は $s^2$・メモリは $s$ でスケールする非対称性**を中心に据え、そこから「長い系列ほど活性化オフロードが有利」が導かれる構造を明示。$b$/重み/$s$/$h$ のどの軸で切るかによる分散戦略の分類、2D Context Parallel（外 Ring・内 Ulysses）、モジュールごとの戦略切り替え、FP8 GEMM と 8-bit FlashAttention の数値的工夫。
- 更新（本文）: [[concepts/diffusion-model-architecture]]（**動画では MM-DiT が自明な最適解ではない**という差し戻し、umT5 の双方向注意、adaLN 共有のアブレーション表＝「adaLN に割くより層を深くする方が良い」）, [[concepts/image-tokenizer]]（「時間軸を持つトークナイザ」節。GroupNorm→RMSNorm と因果性、特徴キャッシュ）, [[concepts/latent-diffusion]]（圧縮が「効率化」から**学習成立の前提**へ変わる。同年の [[concepts/pixel-space-diffusion]] と対照）, [[concepts/diffusion-sampling]]（第 3 の軸への導線）, [[concepts/classifier-free-guidance]]（「2 倍コスト」への 2 つの答え＝ガイダンス蒸留と CFG キャッシュ。**CFG の効きが段階で変わる**という含意）, [[concepts/controllable-generation]]（マスクでタスクが切り替わる構造、Plücker 座標のカメラ制御）, [[concepts/image-inpainting]]（VACE の概念分離＝変える画素と保つ画素を別系列に）, [[concepts/subject-driven-generation]]（**特徴抽出器を使わない第 3 の答え**＝顔画像を VAE 潜在に置いて inpainting として解く）, [[concepts/character-consistency]]（フレーム方向の一貫性、Wan-Bench の ID 一貫性次元）, [[concepts/instruction-based-image-editing]]（VACE の VCU と概念分離）, [[concepts/visual-text-rendering]]（動画内テキスト。**測る手段自体がない**）, [[concepts/super-resolution]]（カスケードから単一モデルの解像度漸進へ）, [[concepts/diffusion-distillation]]（LCM/VideoLCM、蒸留・キャッシュ・量子化の掛け算）, [[concepts/text-to-image-generation]]（**テキストエンコーダ論争に測って答えた**事例）, [[concepts/flow-matching]], [[concepts/noise-schedule]]
- 更新: [[overview]]（「動画へ：時間軸が持ち込むもの」の項）, [[index]]（Summaries / Translations / Concepts＋略称リダイレクト 10 行）
- 画像: ar5iv から **29 枚**を取得（プレースホルダ混入なし、取得失敗なし）。ただし**図 4・11・13・14 の 4 枚は ar5iv・arXiv HTML のどちらにも画像が存在しない**（LaTeX ネイティブのプロットで両変換系とも失敗したとみられる）。翻訳側では該当箇所を `> 図N: ...（訳注: ar5iv・arXiv HTML のいずれでも本図の画像が生成されておらず、取得できなかった）` の引用ブロックで明示した。ファイル名は元名保持（figures/ 配下のものは basename のみ）。
- 翻訳: 本文 §1–6 を全訳（ユーザー確認済み、816 行）。References と §7 貢献者一覧は除外。表 1–8 を markdown テーブル化（表 1 のプロンプト書き換え例は英訳版を和訳）。
- メモ: **本 ingest の主眼は「拡散モデルの定式化は動画でも一切変わらない」ことを確認した上で、では何が変わるのかを切り分けること**。rectified flow ＋ logit-normal も潜在拡散も DiT もそのまま持ち上がる。変わるのは (a) 系列長 100 万・活性化 8 TB という計算規模、(b) 時間的因果性という新しい制約、(c) 動きの品質という新しい評価軸、の 3 点に集約される。**批判的視点として記録した主要な点**: (1) Wan-Bench が自作でありその重み付けも自作——個別指標では Sora や CN-TopA に負けている項目が複数あり、総合首位は重みの決め方に支えられている（[[concepts/text-to-image-generation]] の bakeyness 批判の動画版）、(2) 競合が CN-TopA/B/C/D と匿名化されており追試不能、(3) **様式化能力 0.328 は自ら報告した最下位**なのに本文で一切説明されず「多様な芸術様式を巧みに扱う」という定性的主張と矛盾する、(4) VAE の設計判断（RMSNorm 置換・特徴キャッシュ・最初のフレームの特別扱い）にアブレーションがない、(5) Streamer の無限長生成と real-time が定性評価のみで、長時間での一貫性劣化を測る指標が提示されない、(6) 学習コスト（GPU 時間・総額）が非公開でデータ量も $\mathcal{O}(\cdot)$ 記法で伏せられている。

## [2026-08-17] ingest | HiDream-I1 / HiDream-O1-Image（姉妹論文 2 件）

- 取り込み: `raw/papers/HiDream-I1_ ....md`（ar5iv 由来 markdown・ケース A, arXiv:2505.22705, 2025年5月）と `raw/papers/HiDream-O1-Image_ ....md`（同, arXiv:2605.11061, 2026年）。いずれも HiDream.ai（責任著者 Ting Yao / Tao Mei）で、著者も大きく重複する**同チームの前作・後作**。
- 作成: [[translations/2025-hidream-i1]], [[summaries/2025-hidream-i1]], [[translations/2026-hidream-o1-image]], [[summaries/2026-hidream-o1-image]], [[concepts/pixel-space-diffusion]], [[concepts/mixture-of-experts-diffusion]]
- **新規概念ページ 2 件**（ユーザー回答「pixel-space + MoE の 2 枚」。既存 27 concept に記述ゼロだった領域）:
  - [[concepts/pixel-space-diffusion]] — VAE を経由せず生画素で拡散する系統。**「初期の DDPM への回帰ではない」**という歴史的整理（LDM が主流化した理由＝スケーリングの破綻、いま再挑戦できる理由＝Transformer 化・モデル規模・LLM 資産の再利用）、図5 に基づく (a) latent DiT / (b) pixel-space DiT / (c) natively unified の 3 分類、UiT・ハイブリッド注意（テキストは因果マスク、生成トークンは完全注意）、ピクセル空間ゆえの LPIPS/DINO 損失、未解決（計算コストが一切報告されない・因果の交絡・記憶リスク）。
  - [[concepts/mixture-of-experts-diffusion]] — 拡散 Transformer の FFN を疎な MoE に置き換える系統。ルーター・共有エキスパート・活性化パラメータ・負荷分散損失といった基礎、HiDream-I1 の配置（図3 実測）、**「MoE のおかげで SOTA が出た」とは言えない**という強い留保（アブレーション皆無・ハイパラ非公開・FLOPs 対比なし・後継が MoE を継承していない）。
- 更新（本文）: [[concepts/diffusion-model-architecture]]（HiDream-I1 節＝FFN 内部への介入とテキスト符号化の足し算 vs 引き算、HiDream-O1 節＝**adaLN を捨ててタイムステップをトークン化**する LLM バックボーン）, [[concepts/latent-diffusion]]（**「前提そのものへの異議」節を新設**）, [[concepts/image-tokenizer]]（「反対側の答え：トークナイザを作らない」節。両者が DINO 特徴という同じ道具に行き着く指摘）, [[concepts/visual-text-rendering]]（「VAE を外す」という第 3 の答え。LongText-Bench-ZH 0.024 → 0.978 の比較表）, [[concepts/instruction-based-image-editing]]（HiDream-E1 の**空間連結**を加えて二重符号化／系列連結／空間連結の 3 実装比較表を新設）, [[concepts/diffusion-distillation]]（DMD＋敵対の組み合わせ、O1 の第 3 項 $\mathcal{L}_\text{diff}$）, [[concepts/noise-schedule]]（**事前学習 logit-normal → SFT 一様サンプリング**の切り替え）, [[concepts/reinforcement-learning-for-diffusion]]（「推論品質」報酬とエージェント自体の RL）, [[concepts/text-to-image-generation]]（HiDream 2 本の節）, [[concepts/subject-driven-generation]], [[concepts/multi-concept-customization]]（UniSubject の参照数スケーラビリティ表）, [[concepts/character-consistency]]（VLM 採点による顔以外の一貫性評価）, [[concepts/flow-matching]]
- 更新: [[overview]]（「効率をアーキテクチャで買う／統一の 2 つの向き」の項）, [[index]]（Summaries / Translations / Concepts＋略称リダイレクト 6 行）
- 画像: 計 15 枚を取得（I1 5 枚 / O1 10 枚）。**ar5iv の画像取得が両論文とも部分的に失敗**——O1 は clip 内の URL（`assets/x1.png` 等）が ar5iv 上に存在せず、全 9 枚が例の 325x400「NO IMAGE AVAILABLE」プレースホルダ（MD5 `ded85833...`、FLUX.1 Kontext の時と同一）だった。`arxiv.org/html/2605.11061v1/` から取得し直して解決。さらに **ar5iv/arxiv HTML のファイル名が図番号とズレていた**（`fig2.png` が Figure 3、`fig3.png` が Figure 2 等）ため、取り違え防止に**実際の図番号で `figN.png` に連番化**した。Figure 6（データキュレーション概観）は両 HTML レンダリングとも `<figure>` が壊れており、x 系列にのみ存在した画像を採用。I1 は ar5iv の元名を保持（図番号と一致していたため）。取得失敗なし。
- 翻訳: 両論文とも本文 §1–9 を全訳（ユーザー確認済み）。References と Appendix A（貢献者一覧・謝辞）は除外。表は I1 が 1–4、O1 が 1–8。原典で HTML `<table>` だった複雑な表（I1 表4、O1 表4・表8）は markdown テーブルに再構成した（O1 表8 は 3 群 × 4 指標の入れ子ヘッダを平坦化）。
- メモ: **同チームが 1 年で正反対の設計に振れている**のが本 ingest の主眼。I1＝潜在空間＋外部エンコーダ 4 系統＋MoE で「断片化を受け入れて効率を稼ぐ」、O1＝VAE も外部エンコーダも捨てて「断片化そのものを消す」。この対立は [[concepts/latent-diffusion]] と [[concepts/image-tokenizer]] の前提に直接触るため、両ページに異議として明記した。**批判的視点として記録した主要な点**: (1) I1 の §3.2 本文「両ストリームとも MoE」と図3(b)（テキスト側 dense SwiGLU / 画像側 MoE）の食い違い、(2) 両論文ともアブレーションが皆無で、O1 の「VAE を外したからテキストが描ける」という中心的主張が Qwen3-VL 初期化と交絡していること、(3) O1 がピクセル空間の代償（系列長・レイテンシ・FLOPs）を一切報告しないこと、(4) UniSubject が自作ベンチマークであること、(5) I1 の Fast 版ステップ数が本文内で 14 と 16 に食い違うこと。

## [2026-08-17] ingest | FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space

- 取り込み: `raw/papers/FLUX.1 Kontext_ Flow Matching for In-Context Image Generation and Editing in Latent Space.md`（ar5iv 由来 markdown・ケース A, arXiv:2506.15742, 2025年6月。Black Forest Labs）。SD3 と同系譜（BFL）から出た、**生成と編集を単一 rectified flow に統一**する基盤モデル。
- 作成: [[translations/2025-flux-kontext]], [[summaries/2025-flux-kontext]], [[concepts/character-consistency]]
- **新規概念ページ 1 件**（ユーザー回答「＋多ターン一貫性を新設」）:
  - [[concepts/character-consistency]] — キャラクタ／被写体の一貫性。多ターン編集で被写体が少しずつ別人になる **visual drift** を主題に、**学習型（DreamBooth/LoRA）vs 文脈型（FLUX.1 Kontext）** の対比表、AuraFace 埋め込み類似度による定量化（Kontext は 6 ターン後も高い類似度を保つが競合は急落）、限界（多段の劣化蓄積・指示無視・世界知識違反）。
- 更新（本文）: [[concepts/diffusion-distillation]]（**「(3) 敵対的蒸留（ADD/LADD）」節を新設**——ページ自身が「未取り込み」と明記していた穴を埋める。蒸留が品質を「落とす」のでなく「上げる」ことがある、CFG 由来のアーティファクト低減も動機、ガイダンス蒸留。あわせて「2 系統」→「3 系統」等の整合修正）, [[concepts/instruction-based-image-editing]]（「別解：FLUX.1 Kontext の『連結するだけ』」節を新設。Qwen 系の二重符号化との対比）, [[concepts/noise-schedule]]（「タイムステップのシフトと logit-normal の等価性」節を新設。付録 A.2 の μ=log α 導出）, [[concepts/diffusion-model-architecture]]（FLUX.1 の double stream → 38 single stream・fused feed-forward・因子分解 3D RoPE と**仮想タイムステップ**。MSRoPE の frame 次元と同型の解に独立到達した点）, [[concepts/text-to-image-generation]]（**bakeyness** ——単一の選好比較が過飽和・中心構図・強いボケという「AI 的美学」に報いてしまう問題と 5 次元評価。SDXL の FID 乖離論点の続き）, [[concepts/flow-matching]], [[concepts/image-tokenizer]]（「比較対象としての Flux-VAE」節）, [[concepts/subject-driven-generation]]（学習型 vs 文脈型の対比）, [[concepts/image-composition]], [[concepts/multi-concept-customization]]
- 更新（frontmatter のみ）: 上記に加え計 10 概念ページに `[[character-consistency]]` の相互リンクと `[[summaries/2025-flux-kontext]]` を追加
- 更新: [[overview]]（「文脈内での生成と編集の統一（2025）」の項を追加）, [[index]]（Summaries / Translations / Concepts＋略称リダイレクト 5 行）
- 画像: ar5iv 画像 15 枚を `raw/assets/2025-flux-kontext/` に取得。うち **3 枚（x1.png・x3.png・x9.png）は ar5iv の変換失敗によるプレースホルダ**（3 枚とも同一 MD5・325x400 のグレースケール「NO IMAGE AVAILABLE」）だったため削除し、**有効 12 枚**を採用。翻訳側の該当箇所には `> 図N: ...（訳注: ar5iv 側の変換失敗により画像が取得できなかった）` の訳注を残した。
- 翻訳: 本文 §1–5 ＋ Appendix A（rectified flow の導入・A.2 の解像度シフト α と logit-normal 平均 μ=log α の等価性導出）・Appendix B を全訳。References・謝辞は除外。
- メモ: 本論文の技術的な要点は「**アーキテクチャに何も足さない**」こと——コンテキスト画像を VAE 符号化して対象トークン列にただ連結し、3D RoPE の時間軸に定数オフセット（仮想タイムステップ）を与えて区別する。ControlNet 的なアダプタ枝も参照専用エンコーダも不要。既存の [[concepts/controllable-generation]]（アダプタ型）・[[concepts/instruction-based-image-editing]]（二重符号化型）と並ぶ第三の解として位置づけた。評価面では **KontextBench**（1026 対・5 タスク）と bakeyness 批判が、本 wiki の「指標が実際の良さと乖離する」論点（SDXL の zero-shot FID）を延長する。

## [2026-06-25] ingest | Qwen-Image Technical Report

- 取り込み: `raw/papers/Qwen-Image Technical Report.pdf`（PDF・ケース B, arXiv:2508.02324, 2025年8月。Qwen Team / Alibaba。46 ページ、本文 §1–7＋References）。SD3 以降の最新世代 T2I 基盤モデルで、本 wiki に**未収録だった 3 領域**（画像内テキスト描画・指示ベース編集・拡散の強化学習）を一挙に持ち込む原典。
- 作成: [[translations/2025-qwen-image]], [[summaries/2025-qwen-image]]
- **新規概念ページ 3 件**（ユーザー回答「3 ページすべて作成」。いずれも既存 21 concept に記述ゼロだった領域）:
  - [[concepts/visual-text-rendering]] — 画像内テキストレンダリング。なぜ文字だけ特別に難しいか（グリフの精密さ・表語文字のロングテール・レイアウト・**VAE が上限を決める**）、Qwen-Image の 4 層の対策（デコーダのみ微調整／3 種のデータ合成／非テキスト→テキストのカリキュラム／MSRoPE）、評価（CVTG-2K・ChineseWord・LongText-Bench）、限界（Level-3 漢字 6.48%）。
  - [[concepts/instruction-based-image-editing]] — 指示ベース編集（TI2I）。視覚的一貫性↔意味的一貫性の綱引きと、**二重符号化（MLLM=意味／VAE=画素）でそれを分業に変える**設計。隣接タスク（inpainting・composition・personalization・空間条件付け）との棲み分け表、GEdit/ImgEdit、新視点合成・深度推定も編集として統一する「生成的理解」。
  - [[concepts/reinforcement-learning-for-diffusion]] — 拡散の事後学習。なぜ必要か（分布再現≠良さ）、拡散特有の難しさ（多ステップの信用割り当て・尤度の扱い・**flow の ODE は決定論的で探索できない**）、DPO（速度予測誤差の差を選好に）と GRPO（Flow-GRPO は SDE 再定式化で探索性を得る）、報酬ハッキング等の限界。
- 更新: [[concepts/text-to-image-generation]]（Qwen-Image を「MLLM を条件エンコーダに据える世代」として節追加）, [[concepts/diffusion-model-architecture]]（MSRoPE と条件エンコーダ置換の節を MM-DiT の後に追加）, [[concepts/flow-matching]]（rectified flow の後続＋**flow が事後学習の土台にもなった**点）, [[concepts/latent-diffusion]]（限界節に「VAE がテキスト再現の上限を決める」を追記）, [[concepts/image-inpainting]]・[[concepts/image-composition]]・[[concepts/controllable-generation]]（指示編集との棲み分けをクロスリンク）, [[overview]], [[index]]。加えて 11 概念ページの frontmatter に相互リンク（related/summaries）を追加。
- 画像: **取り込まない**（PDF＝ケース B）。図 1–28 はキャプションのテキスト訳のみを `> 図N:` 形式で保持。
- 翻訳: **本文 §1–6 全訳**（ユーザー回答）。§7 著者一覧・References は除外。表 1–14 を markdown 化（Table 7 の TIIF は多層表のため Overall/Text を抜粋）。rectified flow・DPO・GRPO・SDE 再定式化の数式は LaTeX 保持。
- メモ: 核心＝(1) 凍結 Qwen2.5-VL（7B）＋Wan-2.1-VAE（単一エンコーダ・二重デコーダ、**画像デコーダのみ微調整**）＋20B MMDiT、(2) MSRoPE でテキストを画像対角線上に配置、(3) rectified flow＋logit-normal で事前学習、(4) SFT→DPO→Flow-GRPO の事後学習（GenEval 0.87→0.91）、(5) 編集は MLLM 意味特徴＋VAE 再構成特徴の二重符号化＋MSRoPE の frame 次元。成績＝ChineseWord 58.30（GPT Image 1 は 36.14）、GEdit/ImgEdit 首位、AI Arena でオープンソース唯一のトップ3。限界＝Level-3 漢字 6.48%、英語テキストは GPT Image 1 に及ばず、ablation が限定的、GEdit/ImgEdit は GPT-4.1 審判依存。未取り込み注記: TextDiffuser-2, AnyText, TextCrafter（テキスト描画の先行研究）, InstructPix2Pix, FLUX.1 Kontext（指示編集）, Diffusion-DPO 原典。

## [2026-06-25] schema-update | ingest / query / lint の各 skill に git add & commit 手順を追加

- 変更: `.claude/skills/ingest/SKILL.md`（標準フローに **手順 10「git add & commit」** を追加）, `.claude/skills/query/SKILL.md`（**手順 5** を追加）, `.claude/skills/lint/SKILL.md`（**手順 4** を追加）。
- 要点: **ファイルを作成・更新したら、その作業の最後に必ず `git add` して `git commit` する**というルールを 3 skill に明文化した。従来はコミットしないまま作業が積み上がっていたため、1 オペレーション＝1 コミットで履歴を追えるようにする。
- 各 skill の差分:
  - **ingest**: 最終検証の後にコミットする。`wiki/` 配下だけでなく **`raw/papers/`・`raw/articles/` の原典と `raw/assets/<source-slug>/` の画像も必ず add**（未追跡で残さない）。メッセージは `ingest: <原典タイトル>`。複数件の同時 ingest は**原典 1 件 1 コミット**が基本。
  - **query**: `questions/` ページを作った場合のみコミット（`index.md`・`log.md` と一緒に）。メッセージは `query: <質問の要点>`。**回答しただけでファイルを作らなかった場合はコミット不要**。
  - **lint**: 本来は検出と提示のみなのでコミット不要。ただし**ユーザーの指示で修正まで行った場合はコミットする**。メッセージは `lint: <修正内容の要点>`。
- メモ: CLAUDE.md 本体は変更していない（3 skill 共通の運用ルールだが、手順の所在は各 SKILL.md に置く方針を維持）。

## [2026-06-25] ingest | Qwen-Image-2.0 / Qwen-Image-VAE-2.0 Technical Report（2 件同時）

- 取り込み: `raw/papers/Qwen-Image-2.0 Technical Report.md`（ar5iv 由来 markdown・ケース A, arXiv:2605.10730, 2026年4月22日）と `raw/papers/Qwen-Image-VAE-2.0 Technical Report.md`（同, arXiv:2605.13565）。いずれも Qwen Team（Alibaba）。**前ステップで取り込んだ [[summaries/2025-qwen-image]] の直接の後継 2 本**で、VAE-2.0 は Qwen-Image-2.0 の中で実際に使われているトークナイザの技術報告という関係にある。
- 作成: [[translations/2026-qwen-image-2]], [[summaries/2026-qwen-image-2]], [[translations/2026-qwen-image-vae-2]], [[summaries/2026-qwen-image-vae-2]]
- **新規概念ページ 2 件**（ユーザー回答「蒸留＋tokenizer」）:
  - [[concepts/image-tokenizer]] — 画像トークナイザ／潜在空間を作るオートエンコーダの設計論。**圧縮率 $f$・チャネル $C$・総情報ボトルネック $N(z)=CHW/f^2$**、**三者間トレードオフ（圧縮率↔再構成忠実度↔拡散可能性 diffusability）**、GSC・attention-free・非対称エンコーダデコーダ、KL/GAN 除去、DINOv2 中間層への意味的整合、OmniDoc-TokenBench と OCR ベース NED。VAE-2.0 論文が丸ごとこのページの中身になる。
  - [[concepts/diffusion-distillation]] — 蒸留による少ステップ生成。**「ソルバーを変える」[[diffusion-sampling]] と「モデル自体を変える」蒸留の対比表**、軌道ベース（progressive・consistency）と分布マッチング（DMD）の 2 系統、DMD の勾配（生徒スコア−教師スコア）の直感、限界（教師を超えられない・多様性低下・細部の劣化）。
- 更新: [[concepts/diffusion-model-architecture]]（QI-2.0 節＝バイアスなし変調・SwiGLU・f16 トークナイザの影響）, [[concepts/diffusion-sampling]]（蒸留との対比節を新設）, [[concepts/visual-text-rendering]]（VAE-2.0 が**圧縮率を上げながら**文字再現の上限を上げた話、背景込み合成の知見）, [[concepts/text-to-image-generation]]（QI-2.0 と Prompt Enhancer の逆工学データ生成）, [[concepts/instruction-based-image-editing]]（生成と編集の統一＝入力表現の一本化＋学習比率制御）, [[concepts/reinforcement-learning-for-diffusion]]（5 種のタスク特化型報酬の表・CFG ハイブリッド戦略）, [[concepts/latent-diffusion]]（トークナイザ設計を [[image-tokenizer]] へ委譲）, [[overview]], [[index]]。加えて 11 概念ページの frontmatter に相互リンク（related/summaries）を追加。
- 画像: **全 20 枚取得**（ケース A）。`raw/assets/2026-qwen-image-2/`（16 枚: x1–x15＋arena0422.png）と `raw/assets/2026-qwen-image-vae-2/`（4 枚: x1,x2,x3,x5）。`file` で全件 PNG 妥当確認。取得失敗なし。
- 翻訳: **両方とも本文全訳**（ユーザー回答）。Abstract〜Conclusion を対象、Authors 章は除外（References 章は原典クリップに無く、`\cite` キーがインライン展開されている形式）。QI-2.0 は表 1（VAE 比較）・表 2（学習構成）、VAE-2.0 は表 1（モデル構成）・表 2（ベースライン比較）・表 3（OmniDoc-TokenBench）を markdown 化。原典のクリップに画像が無い図 17–19 は訳注付きで引用ブロックにした。
- メモ: **QI-2.0 の要点** = (1) Qwen3-VL へ更新し、**視覚表現を VAE 潜在で置き換える**（初代の二重符号化から画像側を一本化。図8 で Qwen3-VL の視覚出力経路に ✗）、(2) f16c64 でネイティブ 2K、(3) バイアスなし変調＋SwiGLU で joint training を安定化、(4) Prompt Enhancer（詳細アノテーションを劣化させて短い指示を作り逆操作を CoT 化する逆工学＋GRPO）、(5) 5 報酬 RLHF＋CFG ハイブリッド、(6) DMD 蒸留 4 NFE、(7) 誤り帰属駆動のデータフライホイール。LMArena ELO 1168・世界 9 位。**VAE-2.0 の要点** = 三者間トレードオフを $C$ で補償＋GSC＋DINOv2 中間層整合（段階的にマージンを緩める）で解き、**f16c128 が NED 0.9617 で全 f8 VAE を上回る**（f16 で f8 超えは初）。KL は意味的整合と競合するため除去、GAN は予算十分なら不要。**批判的所見**: QI-2.0 は LMArena と VAE 表以外**定量ベンチマークもアブレーションも無い**（初代が GenEval/DPG/ChineseWord で詳細だったのと対照的）。またテキスト画像 PSNR は f16c64 の 32.81 < 初代 f8c16 の 36.63 で、高圧縮の代償が出ている（f16c128 なら解決するが採用は c64）。未取り込み注記: DMD 原典（Yin ら 2024）、Consistency Models、Progressive Distillation、VA-VAE、DC-AE。

## [2026-06-25] query | 拡散モデル理論の直感ガイド（DDPM → Flow Matching）

- 質問: 拡散モデルの理論（DDPM〜flow matching）を数式なし・例え話で初心者向けに解説する記事を作成（まず理論項目を整理）。
- 作成: [[questions/diffusion-theory-ddpm-to-flow-matching]]（長文解説記事。スコープ＝コア理論＋応用編、形式＝長文 markdown、いずれもユーザー回答）。
- 構成: 導入＋第0章 生成とは＋第1章 DDPM＋第2章 スコア＋第3章 SDE/ODE＋第4章 サンプラー＋第5章 ノイズスケジュール＋第6章 Flow Matching＋第7章 Stochastic Interpolants＋全体地図（流れ図＋比較表）＋応用編（A. guidance / B. latent diffusion）＋用語ミニ辞典。
- 参照: [[denoising-diffusion]]・[[score-based-generative-models]]・[[probability-flow-ode]]・[[diffusion-sampling]]・[[noise-schedule]]・[[flow-matching]]・[[stochastic-interpolants]]・[[classifier-free-guidance]]・[[latent-diffusion]]・[[overview]]、summaries（2020-ddpm/2021-score-sde/2021-ddim/2022-edm/2023-flow-matching/2024-stochastic-interpolants/2025-flow-matching-diffusion-intro/2022-classifier-free-guidance/2022-latent-diffusion/2024-sd3）。
- 更新: [[index]]（Questions セクション新規追記）。
- メモ: 全章を「ノイズ↔データを結ぶ道の描き方・たどり方の違い」という 1 本のメタファーで統一。数式不使用（ユーザー指示）、各章末に既存ページへの深掘りリンク。

## [2026-06-25] ingest | An Introduction to Flow Matching and Diffusion Models（MIT 6.S184 講義ノート）

- 取り込み: `raw/papers/An Introduction to Flow Matching and Diffusion Models.md`（ar5iv 由来 markdown・ケース A, arXiv:2506.02070, MIT 6.S184 2025。Peter Holderrieth & Ezra Erives）。特定手法でなく flow matching と拡散を ODE/SDE 統一の枠組みで導く**教科書的入門**。2800 行・約 18,000 語。
- 作成: [[translations/2025-flow-matching-diffusion-intro]], [[summaries/2025-flow-matching-diffusion-intro]]
- 更新（**新規概念ページなし**・reference として既存を強化）: [[concepts/flow-matching]]（主軸。周辺化トリック・CFM の導出を本講義ノートの定式化として補強、frontmatter summaries＋本文＋参考文献）, [[concepts/score-based-generative-models]]・[[concepts/probability-flow-ode]]・[[concepts/classifier-free-guidance]]（frontmatter summaries＋参考文献追加）, [[concepts/stochastic-interpolants]]・[[concepts/denoising-diffusion]]・[[concepts/diffusion-model-architecture]]（参考文献クロスリンク）, [[overview]]（理論系を束ねる入門リファレンスとして追記）, [[index]]（新設「article / 講義ノート」節）
- 概念ページ: **新規作成なし**（ユーザー回答「既存を強化」。schema どおり手法でなく統一的入門は reference summary 扱い）。
- 画像: **全 16 枚取得**（ケース A・すべて解説的）。`raw/assets/2025-flow-matching-diffusion-intro/`。flow 軌道・Brownian motion・OU 過程・noised MNIST・conditional vs marginal・conditional ODE/SDE・Langevin・conditional-marginal path・score field・Salimans CFG・guidance・U-Net・DiT・MM-DiT・joints。`?as=webp` なしの素 PNG、`file` で全件妥当確認。
- 翻訳: **全訳（§1–5 ＋ Appendix A 確率論・B Fokker-Planck 証明）**（ユーザー回答）。§6 謝辞・References 除外。ar5iv が分割した数式ブロックは 1 つにまとめて表記、ODE/SDE・連続の方程式・Fokker-Planck・CFM/SM 目的・CFG の数式は LaTeX 保持。Key Idea/Theorem/Summary 等の囲み見出しは見出し階層を保って訳出。アルゴリズム 1–5 はコードブロック。wiki 最大級の翻訳ファイル。
- メモ: 核心＝(1) flow＝ODE・diffusion＝SDE のシミュレーション（Euler / Euler-Maruyama）、(2) conditional probability path を周辺化して marginal path を作る周辺化トリック（連続の方程式で証明）＋SDE 拡張（Fokker-Planck）、(3) 扱えない marginal target への回帰は手で書ける conditional target への回帰と同勾配（定理 18・20）→ CFM/denoising score matching、(4) ガウスパスで flow↔score 変換可能（確率フロー ODE）、(5) §4.3 文献ガイド（離散/連続時間・順過程・時間反転 vs FPE・FM/SI と拡散の関係）、(6) §5 CFG・U-Net/DiT/MM-DiT・latent diffusion・SD3/Movie Gen。拡散はガウス初期/ガウスパス限定だが FM/SI は任意 p_init→p_data を許す点を強調。

## [2026-06-25] ingest | Custom Diffusion: Multi-Concept Customization of Text-to-Image Diffusion

- 取り込み: `raw/papers/Multi-Concept Customization of Text-to-Image Diffusion.md`（ar5iv 由来 markdown・ケース A, arXiv:2212.04488, CVPR 2023。Kumari・Zhang・Zhang・Shechtman・Zhu, CMU & Tsinghua & Adobe Research。通称 **Custom Diffusion**）。既存 [[multi-concept-customization]] 概念ページの founding paper（従来は line 60 に一行言及のみ）。personalization 三大原典（DreamBooth・Textual Inversion・Custom Diffusion）が揃う。
- 作成: [[translations/2023-custom-diffusion]], [[summaries/2023-custom-diffusion]]
- 更新: [[concepts/multi-concept-customization]]（**源流ランドマークとして本格記述**：冒頭に「タスクと 2 解法の源流」を追加、(a) 重みマージ節に閉形式マージを起点として接続、関連手法の一行言及を実リンク節へ昇格、frontmatter summaries 先頭＋参考文献）, [[concepts/subject-driven-generation]]（一行言及を W^k,W^v 限定 fine-tune＋V*＋実画像正則化の personalization 手法へ加筆、トレードオフ表に Custom Diffusion 行追加、summaries／参考文献／実リンク化）, [[concepts/lora-merging]]（「(0) 源流：Custom Diffusion の閉形式マージ」節を新設、summaries／参考文献）, [[concepts/low-rank-adaptation]]（差分行列 SVD 低ランク圧縮 75MB→15MB と「fine-tune 中の低ランク制約は suboptimal」をクロスリンク）, [[overview]], [[index]]
- 概念ページ: **新規作成なし**（schema どおり landmark 手法は既存 [[multi-concept-customization]] の源流ランドマーク＋関連 personalization 概念のクロスリンクとして記述）。
- 画像: **全 24 枚取得**（x1–x25、x15 欠番。ケース A・ユーザー回答）。`raw/assets/2023-custom-diffusion/`。大半は生成ギャラリーで、解説的なのは Fig2 手法図(x2)・Fig3 重み変化(x3)・Fig4 cross-attn(x4)・Fig5 正則化(x5)・Fig8 alignment散布(x8)。`?as=webp` なしの素 PNG、`file` で全件妥当確認。
- 翻訳: 本文 §1–5 ＋ **Appendix A–F 全訳**（A=CustomConcept101、B=最適化マージ全導出 Lagrange 乗数法、C=実験/モデル圧縮、D=評価、E=実装詳細、F=社会的影響）。Appendix G（changelog）・References・Acknowledgement 除外。Table 1–8 を markdown 化（`<math>±</math>` 等は ± へ正規化）。cross-attn・閉形式マージ・拡散損失の数式は LaTeX 保持。
- メモ: 核心＝(1) fine-tune 時の層別重み変化率 Δ_l を測ると cross-attention（全体の 5%）が突出 → テキスト→画像写像が入る W^k,W^v だけ更新（75MB・約 6 分、DreamBooth の 1 時間より 2–4× 速）。(2) modifier token V*（稀少トークン初期化）。(3) language drift・過学習を LAION-400M の CLIP 類似 >0.85 実画像 200 枚の正則化で抑制。(4) 多概念は joint training か closed-form constrained-optimization merge（W^k,W^v を制約付き最小二乗→Lagrange 乗数法で閉形式、約 2 秒）。限界＝似カテゴリ（cat+dog）・3 概念以上は困難（attention map 重複）。未取り込み注記: Cones/Cones2, Prompt-to-Prompt（編集に利用）。

## [2026-06-24] ingest | Textual Inversion: An Image is Worth One Word

- 取り込み: `raw/papers/An Image is Worth One Word_ Personalizing Text-to-Image Generation using Textual Inversion.md`（ar5iv 由来 markdown・ケース A, arXiv:2208.01618, ICLR 2023。Gal・Alaluf・Atzmon・Patashnik・Bermano・Chechik・Cohen-Or, Tel-Aviv University & NVIDIA。通称 **Textual Inversion**）。lint/「次に読むべき論文」で [[subject-driven-generation]] が「未取り込み」プレースホルダを抱える personalization の基盤ギャップとして特定。DreamBooth（[[summaries/2023-dreambooth]]）と並ぶ二大原典の片方。
- 作成: [[translations/2022-textual-inversion]], [[summaries/2022-textual-inversion]]
- 更新: [[concepts/subject-driven-generation]]（「代表手法 2: Textual Inversion」節を**原典で本格記述**し「まだ原典を取り込んでいない」注記を削除、summary を実リンク化、frontmatter summaries／参考文献に追加）, [[concepts/text-to-image-generation]]（personalization 段落に「凍結＋擬似単語のみ」の対極として追記、summaries／参考文献に追加）, [[concepts/latent-diffusion]]（凍結 LDM 上の personalization としてクロスリンク）, [[concepts/low-rank-adaptation]]（DreamBooth↔TI の中間＝LoRA の記述を実リンク化）, [[overview]], [[index]]
- 概念ページ: **新規作成なし**（schema どおり landmark 手法は既存 [[subject-driven-generation]] の代表手法として記述）。
- 画像: **全ユニーク 13 枚取得**（ケース A・ユーザー回答）。`raw/assets/2022-textual-inversion/`。ar5iv は各図 1 アセットのみ抽出（大半は学習画像 1 枚＝多パネル図の代用）。解説的なのは Fig2 手法図 `x1.png`・Fig10 評価プロット `quant_eval.jpg`・Fig12 `num_images.jpg`。重複（headless_statue 3×・qinni 2×）は 1 ファイル保存し複数箇所から参照。`?as=webp` なしの素 PNG/JPEG、`file` で全件妥当確認。
- 翻訳: 本文 §1–8 ＋ **Appendix A–D 全訳**（A=Bipartite DDIM-inversion・Pivotal Tuning、B=学習集合サイズ、C=追加結果、D=学習プロンプトテンプレ 27 個を箇条書き保持）。References・Acknowledgements 除外。LDM 損失・v\* 最適化の数式は LaTeX 保持。
- メモ: 核心＝凍結 text-to-image（LDM 1.4B・LAION-400M・BERT）の埋め込み空間に擬似単語 S\* の埋め込み v\* を 1 つだけ再構成損失で最適化（3〜5 枚・約 2 時間・粗い記述語で初期化）。GAN inversion 由来の多語化／正則化／画像ごとトークンより**単一語が最良**、distortion-editability トレードオフを学習率で移動。応用＝画風擬似単語化・概念合成・バイアス低減・Blended Latent Diffusion。限界＝精密形状が苦手・最適化が遅い・凍結ゆえ忠実度は DreamBooth に劣る。

## [2026-06-24] ingest | EDM: 拡散ベース生成モデルの設計空間の解明

- 取り込み: `raw/papers/Elucidating the Design Space of Diffusion-Based Generative Models.pdf`（PDF, arXiv:2206.00364, NeurIPS 2022。Karras・Aittala・Aila・Laine, NVIDIA。通称 **EDM**）。lint/「次に読むべき論文」で SD3・SDXL・Stochastic Interpolants の 3 本が揃って参照する最大ギャップとして特定された基盤論文。
- 作成: [[translations/2022-edm]], [[summaries/2022-edm]], [[concepts/noise-schedule]]（**新規概念ページ**。学習時ノイズ分布＋推論時の時間離散化を扱い、DDPM β／cosine／VP・VE／SD3 サンプラー／SDXL shift／EDM σ(t)=t・ρ=7・対数正規を横断接続。ランドマーク＝EDM）
- 更新: [[concepts/diffusion-sampling]]（**EDM を Heun・ρ・churn の代表サンプラーとして本格記述**、「今後の ingest で拡充」節を更新）, [[concepts/score-based-generative-models]]（VP/VE を σ(t)/s(t)/preconditioning で統一）, [[concepts/diffusion-model-architecture]]（preconditioning＝ネット入出力設計軸）, [[concepts/probability-flow-ode]]（σ(t)=t で軌道直線化）, [[concepts/denoising-diffusion]]（損失重み・ノイズ分布）, [[concepts/flow-matching]]（SD3 サンプラーとの同型をクロスリンク）, [[overview]], [[index]]
- 画像: **取り込まない**（PDF・ユーザー指示でケース B）。図はキャプションのテキスト訳のみ、`<figure>`/`![]()` 画像記法なし。
- 翻訳: 本文 §1–6 ＋ **Appendix A–F 全訳**（B 式導出 incl. preconditioning 導出 B.6、C VP/VE/iDDPM 再構成、D ステップサイズ解析＋2 次 RK 一般族、E 確率サンプリング、F 学習/ネット/データセット詳細）。References・Acknowledgements 除外。Table 1–8 を markdown 化、Algorithm 1/2/3 をコードブロック保持。ODE・preconditioning・σ スケジュール・λ(σ) の数式は LaTeX 保持。
- メモ: 4 本柱＝(1) 共通枠組み Table 1（VP/VE/iDDPM/EDM 統一、σ(t)=t,s(t)=1 推奨）、(2) Heun 2 次サンプラー＋ρ=7、(3) churn 確率サンプラー（S_churn 等）、(4) preconditioning（c_skip/c_out/c_in/c_noise を第一原理導出）＋対数正規ノイズ分布（P_mean=−1.2,P_std=1.2）＋損失重み λ=1/c_out²＋non-leaky augmentation。CIFAR-10 1.79/1.97・ImageNet-64 1.36・35 NFE。未取り込み注記: DPM-Solver（高次ソルバ）, consistency models, EDM2。

## [2026-06-24] ingest | Stochastic Interpolants: flows と diffusions の統一枠組み（バッチ 2/2）

- 取り込み: `raw/papers/Stochastic Interpolants_ ...md`（ar5iv 由来 markdown, arXiv:2303.08797, 2023。Albergo・Boffi・Vanden-Eijnden, NYU）。SD3→SI の 2 件バッチの 2 件目。
- 作成: [[translations/2024-stochastic-interpolants]]（**本文§1–8＋Appendix A–C 全訳、全証明逐次**。3081 行の高密度理論論文）, [[summaries/2024-stochastic-interpolants]], [[concepts/stochastic-interpolants]]（**新規概念ページ**。flows と diffusions を統一する枠組み。flow-matching・score-based-generative-models・probability-flow-ode の上位概念）
- 更新: [[concepts/flow-matching]]（rectified flow / SI を一般化として）, [[concepts/score-based-generative-models]]（SBDM を片側補間として内包）, [[concepts/probability-flow-ode]]（ODE/SDE の統一）, [[overview]], [[index]]
- 画像: ar5iv 画像 15 枚（x1–x15）を `raw/assets/2024-stochastic-interpolants/` に保存。全 PNG・全取得成功。図1（パラダイム）・図2（設計柔軟性）・図3（アルゴリズム）を翻訳に配置。
- 翻訳メモ: ユーザー指定で **Appendix B の全証明（~1000 行）も逐次全訳**。確率的補間 $x_t=I(t,x_0,x_1)+\gamma(t)z$、輸送方程式→ODE、Fokker-Planck→SDE、速度 $b$・スコア $s$・ノイズ除去器 $\eta_z$ の二乗回帰。数式は LaTeX 保持。Theorem/Lemma の主張＋証明を一文ずつ。アルゴリズム 1–5 はコードブロック保持。References（脚注 [^N]）・Acknowledgements 除外。**翻訳完了：本文§1–8 ＋ Appendix A（ガウス混合）・B（証明 B.1–B.8 全訳）・C（実験仕様・表18）。画像 15 枚（x1–x15）全参照。`<figure>` 15/15・ar5iv 残骸 0。**（2026-06-24 の lint 指摘 🔴 を解消）
- メモ: 未取り込み注記: Schrödinger bridge / 最適輸送の専論、stochastic localization、EDM（連続時間拡散）。SD3 の rectified flow は本枠組みの線形インスタンス。

## [2026-06-24] ingest | Stable Diffusion 3: Rectified Flow Transformer のスケーリング（バッチ 1/2）

- 取り込み: `raw/papers/Scaling Rectified Flow Transformers ...md`（ar5iv 由来 markdown, arXiv:2403.03206, ICML 2024。Esser ら, Stability AI。通称 **Stable Diffusion 3 / SD3**）。
- 作成: [[translations/2024-sd3]]（本文§1–6＋Appendix A–E 全訳）, [[summaries/2024-sd3]]（概念は既存ページに分散収録）
- 更新: [[concepts/flow-matching]]（rectified flow＋改良サンプラーを「代表手法：Rectified Flow と大規模化」節に）, [[concepts/diffusion-model-architecture]]（MM-DiT＝DiT のマルチモーダル拡張・QK-normalization）, [[concepts/latent-diffusion]]（LDM→SDXL→SD3 の系譜）, [[concepts/text-to-image-generation]]（MM-DiT＋3 テキストエンコーダ）, [[overview]], [[index]]
- 画像: ar5iv 画像 19 枚（teaser・x1/x2/x3・サンプル/プロット類、サブパスは安定名に平坦化）＋ **Figure 2（MM-DiT アーキ）はインライン SVG 2 枚を抽出保存**（fig2a/fig2b）。raster は全 PNG/JPEG 妥当・全取得成功。SVG は foreignObject によるラベルを含むため `<figcaption>` でアーキを文章化。
- 翻訳メモ: HTML 表（Table 1/2/5/6）を markdown 化（`<math><semantics>` 残骸・512² の上付き等を除去）。Table 3/4/7 は元から markdown。Alg.1/2（重複除去・記憶検出の擬似コード）はコードブロックで保持。rectified flow $z_t=(1-t)x_0+t\epsilon$、logit-normal/mode/CosMap サンプラー、解像度依存時刻シフト、QK-norm の数式は LaTeX 保持。References・Acknowledgements 除外。
- メモ: 4 本柱＝(1) RF＋中間時刻に重みを置くサンプラー（rf/lognorm(0,1) 最良）、(2) MM-DiT（テキスト/画像別重み＋attention 結合）、(3) 8B スケーリング（検証損失が人間評価と相関・飽和なし）、(4) 改良 VAE 16ch・合成キャプション・DPO。GenEval で DALL-E 3 超え。未取り込み注記: EDM, DPO, SDEdit, T2I-CompBench/GenEval。

## [2026-06-24] ingest | SDXL: 高解像度画像合成のための潜在拡散モデルの改良

- 取り込み: `raw/papers/SDXL_ ...md`（ar5iv 由来 markdown, arXiv:2307.01952, 2023。Podell ら, Stability AI）。直前の ZipLoRA が base model として全面依存していた SDXL を取り込み、被参照ページを原典で裏付け。
- 作成: [[translations/2023-sdxl]], [[summaries/2023-sdxl]]（schema 規約どおり**専用概念ページは作らず** [[latent-diffusion]] の代表的後継モデルとして収録。アーキ詳細は [[diffusion-model-architecture]]。DiT と同じ扱い）
- 更新: [[concepts/latent-diffusion]]（「代表的後継モデル：SDXL」節を新設）, [[concepts/diffusion-model-architecture]]（ADM→SDXL→DiT の流れに UNet スケーリングを追記、SDXL は transformer 化を当時見送り）, [[concepts/controllable-generation]]（micro-conditioning＝学習時メタデータ条件付け）, [[concepts/text-to-image-generation]]（2 テキストエンコーダ＋pooled emb の高解像 T2I）, [[concepts/subject-driven-generation]]・[[concepts/low-rank-adaptation]]（既存 SDXL 言及を実リンク化）, [[overview]], [[index]]
- 画像: ar5iv 画像 16 枚（teaser＋Fig1–15。直 PNG x1/x2/x3 ＋ JPEG/JPEG 多数）を `raw/assets/2023-sdxl/` に保存。サブパス（`img/...`・`comp_old_model/sd1-5/`・`refiner_magic/` 等）を安定名に平坦化（`comp_catpoleon_row` 等、別名なので衝突なし）。全画像妥当・全取得成功。`sd-xl-vs.jpg`(Fig1) は実体がユーザー選好棒グラフ（右側の 2 段パイプライン図は別アセットでなく原図注釈）であるためキャプションで補記。説明図（Fig1/6）を要約に再掲。
- 翻訳: 本文 §1–3 ＋ **Appendix B–J 全訳**（A 謝辞・References 除外）。Tab.1/2/3 と App I の 40 行アスペクト比表は markdown 保持、**Alg.1（size/crop 条件付け擬似コード）と App J の Python コードはコードブロックで保持**（base64 data URI リンクは除外）。probability flow ODE/SDE・DSM・CFG（App C, EDM 流定式化）の数式は LaTeX 保持。図 16 枚を `<figure>`。
- メモ: 4 本柱＝(1) 3× UNet（2.6B）＋transformer block 不均一配分 `[0,2,10]`＋2 テキストエンコーダ＋pooled emb、(2) micro-conditioning（size/crop/aspect-ratio を Fourier 埋め込みで timestep emb に加算）、(3) multi-aspect training（bucket）、(4) 改良 VAE＋base/refiner の 2 段（SDEdit）。知見：人間評価で SOTA 級だが COCO zero-shot FID は悪化（指標の限界、付録F）。限界：手・concept bleeding・長文テキスト。未取り込み注記: EDM（Karras 2022, 連続時間）, SDEdit, StyleDrop, simple diffusion, offset-noise。

## [2026-06-24] ingest | ZipLoRA: 任意の被写体を任意のスタイルで（バッチ 2/2）

- 取り込み: `raw/papers/ZipLoRA_ ...md`（ar5iv 由来 markdown, arXiv:2311.13600, ECCV 2024。Shah ら, Google Research・UIUC）。Mix-of-Show→ZipLoRA バッチの 2 件目（完了）。
- 作成: [[translations/2024-ziplora]], [[summaries/2024-ziplora]]（概念は 1 件目で作った [[lora-merging]] に「(3) 学習係数マージ」として収録済み）
- 更新: [[concepts/lora-merging]]（ZipLoRA を学習係数マージのランドマークとして記述済み）, [[concepts/subject-driven-generation]]（content+style 分離学習→再結合）, [[concepts/latent-diffusion]]（SDXL 上で動作）, [[concepts/low-rank-adaptation]]（LoRA の疎性、1 件目で追記済み）, [[overview]], [[index]]
- 画像: ar5iv 画像 9 枚（PNG 5: x1–x5 ＋ JPEG 4: `figs/{fig3_final1,compare_main_new,recontext3,moe}.jpg`）を `raw/assets/2024-ziplora/` に保存。`figs/` を `figs_` 連結で平坦化。全 PNG/JPEG・全取得成功。説明図（x2 疎性・x4 手法概観）を Read で確認しキャプション作成。
- 翻訳: 本文 §1–5 全訳（**appendix なし**）。References・Acknowledgements 除外。HTML 表 1 個（Table 1 ユーザー選好）を markdown 化（`<math><semantics>…%` 残骸除去）。Table 2 は既に markdown。merger 係数・$\mathcal L_{merge}$（3 項）の数式保持。$\Delta W_m=m_c\otimes\Delta W_c+m_s\otimes\Delta W_s$（原典の式は $m_s\otimes W_s$ と表記ゆれ → 文脈上 $\Delta W_s$ として訳出）。
- メモ: 2 観察＝(1) LoRA の $\Delta W$ は疎（90% を 0 にしても品質維持）、(2) 列の cosine 類似度が高いと直和が破綻（signal interference）。解＝列ごとの学習係数 $m_c,m_s$ で個別 LoRA 挙動を保ちつつ列を直交化。被写体×画風の 2 LoRA 特化（複数前景は範囲外、LoRA-Composer と守備範囲が異なる）。lora-merging 概念で Mix-of-Show（gradient fusion）と対比整理。**バッチ 2 件完了**。未取り込み注記: StyleDrop, SDXL, DINO（評価特徴）, LoRAHub/MoLE（MoE 系 LoRA 合成）。

## [2026-06-24] ingest | Mix-of-Show: 分散型多概念カスタマイズ（バッチ 1/2）

- 取り込み: `raw/papers/Mix-of-Show_ ...md`（ar5iv 由来 markdown, arXiv:2305.18292, NeurIPS 2023。Gu ら, NUS Show Lab・Tencent ARC Lab）。Mix-of-Show→ZipLoRA の 2 件バッチの 1 件目。
- 作成: [[translations/2023-mix-of-show]], [[summaries/2023-mix-of-show]], [[concepts/lora-merging]]（**新規概念ページ**。複数 LoRA の重みマージ／融合を専門に扱う。multi-concept-customization の (a) 系統を細粒度化。ランドマーク＝Mix-of-Show・ZipLoRA）
- 更新: [[concepts/multi-concept-customization]]（(a) 重みマージを [[lora-merging]] へ委譲）, [[concepts/low-rank-adaptation]]（ED-LoRA・LoRA の疎性・[[lora-merging]] リンク）, [[concepts/controllable-generation]]（regionally controllable sampling を LoRA-Composer 領域注入の源流として）, [[overview]], [[index]]
- 画像: ar5iv 画像 11 枚（x1–x10 ＋ `imgs/mturk.png`）を `raw/assets/2023-mix-of-show/` に保存（`imgs/` を `imgs_mturk.png` に平坦化）。全 PNG・全取得成功。説明図（x4 パイプライン）を要約に再掲。
- 翻訳: 本文 §1–5 ＋ **Appendix §6 全訳**。References・Acknowledgements 除外。**HTML 表 8 個を markdown 化**（`<math><semantics>…` 残骸＝→ 等を除去、single→fused の矢印は「→」で表現、Table 3 は単一概念/融合 × 物体/キャラ/シーンの 6 サブ表に整理）。gradient fusion 目的・region-aware cross-attention の数式は LaTeX 保持。図 11 枚を `<figure>`。
- メモ: 課題＝concept conflict（embedding と LoRA 重みの役割未分離）と identity loss（重み平均が $\frac1n$ に薄める）。解＝ED-LoRA（$V=V_{rand}^+V_{class}^+$ で embedding に in-domain essence を残す）＋gradient fusion（$\arg\min_W\sum_i\|(W_0+\Delta W_i)X_i-WX_i\|_F^2$）。実写は Chilloutmix・アニメは Anything-v4 ベース。次は ZipLoRA を取り込み lora-merging に (c) 学習係数マージとして追記。

## [2026-06-24] ingest | Multi-LoRA Composition for Image Generation（バッチ 3/3）

- 取り込み: `raw/papers/Multi-LoRA Composition for Image Generation.md`（ar5iv 由来 markdown, arXiv:2402.16843, ICML 2024。Zhong ら, Microsoft）。3 件バッチの 3 件目（完了）。
- 作成: [[translations/2024-multi-lora-composition]], [[summaries/2024-multi-lora-composition]]（概念は 2 件目で作った [[multi-concept-customization]] に decoding-centric 系統として収録済み）
- 更新: [[concepts/multi-concept-customization]]（LoRA Switch/Composite を (c) 系統として記述済み）, [[concepts/classifier-free-guidance]]（LoRA Composite が CFG の多 LoRA 拡張）, [[overview]], [[index]]
- 画像: ar5iv 画像 9 枚（x1–x8 ＋ `Figure/merge_case.png`）を `raw/assets/2024-multi-lora-composition/` に保存。サブパス `Figure/` を平坦化。Table 1（画像 merge_case.png）は `<figure>` で引用。全 PNG・全取得成功。
- 翻訳: 本文 §1–5 ＋ **Appendix A 全訳**。References・Impact Statements 末尾の定型は本文扱いで訳出、References 一覧は除外。HTML 表 2 個（Table 2 人手評価・Table 3 ComposLoRA LoRA 一覧）を markdown 化（civitai リンク列は省略）。Table 4/5 は画像（merge_case.png）で原典が同一ファイルを指すため 訳注 で説明。LoRA Switch/Composite 式保持。**broken cross-ref（`LABEL:fig:result`/`fig:switch_step`/`fig:switch_order`）は訳注「図（結果）」で明示**（該当画像は markdown に含まれず）。
- メモ: バッチ 3 件完了。LoRA→LoRA-Composer→Multi-LoRA Composition の系譜を low-rank-adaptation／multi-concept-customization の 2 概念に整理。multi-concept は (a) 重みマージ、(b) 注意制御（LoRA-Composer）、(c) decoding-centric（本論文）の 3 系統。未取り込み注記: LoRAHub, ZipLoRA（重みベース合成）, DPM-Solver++（サンプラー）。

## [2026-06-24] ingest | LoRA-Composer: 訓練不要の多概念カスタマイズ（バッチ 2/3）

- 取り込み: `raw/papers/LoRA-Composer_ ...md`（ar5iv 由来 markdown, arXiv:2403.11627, 2024。Yang ら）。3 件バッチの 2 件目。
- 作成: [[translations/2024-lora-composer]], [[summaries/2024-lora-composer]], [[concepts/multi-concept-customization]]（新規概念ページ。3 件目の Multi-LoRA Composition もここに収める）
- 更新: [[concepts/low-rank-adaptation]]（multi-concept への発展、既に相互リンク済み）, [[concepts/image-composition]]（AnyDoor 系 ↔ LoRA 系の対比）, [[concepts/controllable-generation]]（訓練不要の推論時合成）, [[index]]
- 画像: ar5iv 画像 11 枚（x1–x11）を `raw/assets/2024-lora-composer/` にローカル保存。全 PNG・全取得成功。
- 翻訳: 本文 §1–5 ＋ **Appendix 0.A–0.D 全訳**。References・Acknowledgements 除外。Table 1/2/3 は markdown（元から HTML table なし）。数式 LaTeX 保持（Region-Aware Injection・$\mathcal{L}_{ce}$/$\mathcal{L}_{fill}$/$\mathcal{L}_{region}$）。ar5iv 残骸除去。
- メモ: 新規 `multi-concept-customization` を作成し、(a) 重みマージ（LoRA Merge/Mix-of-Show）、(b) 訓練不要の注意制御（LoRA-Composer）、(c) decoding-centric（Multi-LoRA Composition）の 3 系統に整理。Mix-of-Show・Custom Diffusion・Cones は比較対象として言及（未取り込み）。

## [2026-06-24] ingest | LoRA: Low-Rank Adaptation of Large Language Models（バッチ 1/3）

- 取り込み: `raw/papers/LoRA_ Low-Rank Adaptation of Large Language Models.md`（ar5iv 由来 markdown, arXiv:2106.09685, ICLR 2022。Hu ら, Microsoft）。LoRA→LoRA-Composer→Multi-LoRA Composition の 3 件バッチの 1 件目。
- 作成: [[translations/2022-lora]], [[summaries/2022-lora]], [[concepts/low-rank-adaptation]]（新規概念ページ）
- 更新: [[concepts/subject-driven-generation]]（「LoRA 未取り込み」を解消、トレードオフ表に LoRA 追加）, [[concepts/latent-diffusion]]・[[concepts/diffusion-model-architecture]]（軽量 personalization としてクロスリンク）, [[index]]
- 画像: ar5iv 画像 8 枚（x1–x8）を `raw/assets/2022-lora/` にローカル保存。全 PNG・全取得成功。x1（再パラメータ化図）は Read で確認。
- 翻訳: 本文 §1–8 ＋ **Appendix A–H 全訳**（ユーザー確認）。References・Acknowledgements 除外。**HTML 表 14 個すべてを markdown 化**（ar5iv の `<math><semantics>…` 数式マークアップを除去し数値表に再構成。大型ハイパラ表 9/15 等は共通設定を見出しに畳んで整形）。数式 LaTeX 保持。
- メモ: LLM 論文だが拡散の軽量 personalization の基礎として取り込み。新規 `low-rank-adaptation` を作成し、DreamBooth（全層）↔ Textual Inversion（埋め込み）の中間に位置づけ。[[multi-concept-customization]] は 2 件目で作成（本エントリ時点では forward link）。未取り込み注記: HyperDreamBooth, Custom Diffusion, compacter（PEFT 後続）。

## [2026-06-24] ingest | DiT: Scalable Diffusion Models with Transformers

- 取り込み: `raw/papers/Scalable Diffusion Models with Transformers.md`（ar5iv 由来 markdown, arXiv:2212.09748, ICCV 2023。Peebles & Xie, UC Berkeley）
- 作成: [[translations/2023-dit]], [[summaries/2023-dit]]（新規概念ページは作らず）
- 更新: [[concepts/diffusion-model-architecture]]（**大幅拡充**：DiT を第 2 のランドマークとして追加、「改良 U-Net（ADM）→ Transformer 化（DiT）」の二系譜に再構成。patchify・adaLN-Zero・Gflops スケーリング・SOTA を記述、Fig3 引用）, [[concepts/latent-diffusion]]（DiT は LDM 潜在空間で backbone のみ Transformer 化）, [[concepts/denoising-diffusion]]（数学はそのまま backbone 差し替え）, [[concepts/classifier-free-guidance]]（DiT も CFG 使用・部分チャネル CFG）, [[concepts/text-to-image-generation]]（DiT backbone の将来展望）, [[overview]], [[index]]
- 画像: ar5iv 画像 33 枚（ユニーク）を `raw/assets/2023-dit/` にローカル保存。x1〜x13（説明・結果図 13 枚）＋ superimages 20 枚（無選別サンプル）。**`superimage-cfg-4.0-class-88.jpg` が 256/512 両ディレクトリに存在**するため、サブパスを `_` 連結で平坦化して衝突回避。全取得成功。説明図 x2/x3/x5/x8 は Read で内容確認のうえキャプション作成。
- 翻訳: 本文 §1–6 ＋ **Appendix A–D 全訳**（ユーザー確認済み。B のギャラリー Fig14–33 は図キャプションのみ）。References・Acknowledgements 除外。HTML `<table>` 4 個（Table 2/3/5/6）を markdown 化、Table 1/4 は markdown 保持。数式 LaTeX 保持（CFG 式・$L_\text{simple}$）。ar5iv math 残骸除去。
- メモ: **CLAUDE.md 明記「DiT は対応するアーキテクチャ概念ページの中で扱う」に従い、新規概念ページは作らず [[diffusion-model-architecture]] に追記**（ユーザー確認済み）。DiT は LDM 枠組み＋Stable Diffusion の VAE 潜在空間＋ADM の拡散ハイパラを流用し backbone だけ Transformer 化。dangling は `<slug>`（index テンプレ例）のみ想定。未取り込み注記: Stable Diffusion 3 / PixArt-α / Sora（DiT 後続）, ViT（基盤）, StyleGAN-XL（比較対象 GAN）。

## [2026-06-24] ingest | AnyDoor: Zero-shot Object-level Image Customization

- 取り込み: `raw/papers/AnyDoor_ Zero-shot Object-level Image Customization.md`（ar5iv 由来 markdown, arXiv:2307.09481, ICCV 2023。Chen ら, HKU / Alibaba / Ant）
- 作成: [[translations/2023-anydoor]], [[summaries/2023-anydoor]], [[concepts/image-composition]]（新規概念ページ）
- 更新: [[concepts/subject-driven-generation]]（zero-shot/参照ベースの対比軸として AnyDoor を追記）, [[concepts/controllable-generation]]（detail extractor の ControlNet スタイル・参照画像条件付け）, [[concepts/image-inpainting]]（ボックス領域を特定物体で再生成する対比）, [[concepts/latent-diffusion]]（SD を base にエンコーダ凍結・デコーダ学習）, [[overview]], [[index]]
- 画像: ar5iv 画像 11 枚（ユニーク x1〜x11）を `raw/assets/2023-anydoor/` にローカル保存。全 PNG・全取得成功。説明図 x2/x3/x4（パイプライン・注目領域・動画データ準備）は Read で内容確認のうえキャプション作成。
- 翻訳: 本文 §1–5（Abstract〜Conclusion）。**appendix なし**（§5 Conclusion 後は References）。References 除外。Table 1–5 は markdown 表を保持・整形（元から HTML table なし）。HF-map 式・injection 損失など数式 LaTeX 保持。ar5iv math 残骸なし（0）。
- メモ: ユーザー確認どおり新規概念ページ `image-composition`（画像コンポジション / object teleportation）を作成し、AnyDoor をランドマーク、Paint-by-Example・ObjectStitch を参照ベース同系として整理。古典的 image harmonization・[[subject-driven-generation]]（DreamBooth, tuning 型）・[[image-inpainting]] との違いを明示。AnyDoor の base は Stable Diffusion（[[summaries/2022-latent-diffusion]]）、detail extractor は ControlNet スタイル（[[summaries/2023-controlnet]]）、評価指標 DINO/CLIP-Score は DreamBooth（[[summaries/2023-dreambooth]]）由来。dangling は `<slug>`（index テンプレ例）のみ想定。未取り込み注記: Paint-by-Example / ObjectStitch / Graphit（参照ベース）, IP-Adapter（image-prompt アダプタ）, DINO-V2・SAM（基盤モデル）。

## [2026-06-24] ingest | DreamBooth: Subject-Driven Generation

- 取り込み: `raw/papers/DreamBooth_ Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation.md`（ar5iv 由来 markdown, arXiv:2208.12242, CVPR 2023。Ruiz ら, Google）
- 作成: [[translations/2023-dreambooth]], [[summaries/2023-dreambooth]], [[concepts/subject-driven-generation]]（新規概念ページ）
- 更新: [[concepts/text-to-image-generation]]（下流応用 personalization を追記、Imagen の言及も追加）, [[concepts/super-resolution]]（cascaded SR と DreamBooth の SR fine-tune＋低ノイズ増強）, [[concepts/latent-diffusion]]（SD を personalize する応用）, [[concepts/denoising-diffusion]]（拡散損失＋prior 保存項の fine-tune）, [[overview]], [[index]]
- 画像: ar5iv 画像 23 枚（ユニーク）を `raw/assets/2023-dreambooth/` にローカル保存（`x1`〜`x18.png` はそのまま、`figures/<name>.png` は `figures_<name>.png` に平坦化。basename 衝突なし）。全 PNG・全取得成功。説明図 x2/x3/x5/dino_metric は Read で内容確認のうえキャプション作成。
- 翻訳: 本文 §1–6 ＋ **Supplementary Material 全訳**（ユーザー確認済み。Background / Dataset / Subject Fidelity Metrics / User Study / Additional Applications / Additional Experiments / Societal Impact）。References・Acknowledgement（§6）除外。Table 1–6 は markdown 表を保持・整形（元から HTML table なし）。数式 LaTeX 保持（式(1)・PPL 損失）、ar5iv 残骸（図キャプションの `\sim` 等）除去。
- メモ: ユーザー確認どおり新規概念ページ `subject-driven-generation`（被写体駆動生成 / personalization）を作成し、DreamBooth（全層 fine-tune＋PPL＋rare-token id）と Textual Inversion（凍結モデルの token 埋め込み学習）を二大ランドマークとして対比。DreamBooth が prior に使うのは Imagen（T5-XXL＋cascaded diffusion）と Stable Diffusion（[[summaries/2022-latent-diffusion]]）。dangling は `<slug>`（index テンプレ例）のみ想定。未取り込み注記: Textual Inversion（Gal ら）, LoRA, Custom Diffusion（personalization 後続）, Imagen（専用原典）。

## [2026-06-24] ingest | RePaint: Inpainting using DDPM

- 取り込み: `raw/papers/RePaint_ Inpainting using Denoising Diffusion Probabilistic Models.md`（ar5iv 由来 markdown, arXiv:2201.09865, CVPR 2022。Lugmayr ら, ETH Zürich）
- 作成: [[translations/2022-repaint]], [[summaries/2022-repaint]], [[concepts/training-free-conditioning]]（新規概念ページ）
- 更新: [[concepts/image-inpainting]]（RePaint をランドマークとして追記、学習型 LDM ↔ 学習不要 RePaint の 2 系統に整理）, [[concepts/controllable-generation]]（推論時条件付けの (1b) 置き換え／射影ベースを追記）, [[concepts/diffusion-sampling]]（resampling＝調和のためのサンプリング時戦略、slowing down との違い）, [[concepts/denoising-diffusion]]（凍結 DDPM の推論時応用としてクロスリンク）, [[overview]], [[index]]
- 画像: ar5iv 画像 21 枚（ユニーク）を `raw/assets/2022-repaint/` にローカル保存。`figures/` と `supplement/figures/` に同名 basename（paper_c256_test_thick 等）があるため、サブパスを `_` 連結でファイル名に平坦化して衝突回避。全取得成功。説明図 x1/x2/x3（図1/2/9）は Read で内容確認のうえキャプション作成。
- 翻訳: 本文§1–8 ＋ **Appendix A–I を全訳**（ユーザー確認済み）。References・Acknowledgements 除外。HTML `<table>` 6 個（Table1 ×2・3・5・6 ほか）を markdown 化。Algorithm 1 を擬似コードブロックで保持。**Figure 8/10 の文字化け Python 擬似コード（ar5iv で `¿`/`¡` に化け）を正しいコードブロックに再構成**。数式 LaTeX 保持、ar5iv math 残骸除去。
- メモ: ユーザー確認どおり新規概念ページ `training-free-conditioning`（学習不要・推論時の条件付け）を作成し、RePaint をランドマークに。controllable-generation の「(1a) スコア勾配 guidance」「(2) アダプタ」に対し RePaint は「(1b) 置き換え／射影ベース」と整理。RePaint が prior に使う事前学習モデルは [[summaries/2021-adm]] の guided-diffusion。dangling は `<slug>`（index テンプレ例）のみ想定。未取り込み注記: ILVR / SDEdit / DDRM（関連する推論時条件付け）, Palette・GLIDE（学習型の画像条件付き拡散）。

## [2026-06-23] ingest | Diffusion Models Beat GANs on Image Synthesis (ADM)

- 取り込み: `raw/papers/Diffusion Models Beat GANs on Image Synthesis.md`（ar5iv 由来 markdown, arXiv:2105.05233, NeurIPS 2021。Dhariwal & Nichol, OpenAI）
- 作成: [[translations/2021-adm]], [[summaries/2021-adm]], [[concepts/diffusion-model-architecture]]（新規概念ページ）
- 更新: [[concepts/classifier-guidance]]（**主要原典として本格拡充・stub note 除去**：DDPM/DDIM 2 導出・勾配スケール s>1・ADM-G 成果・限界）, [[concepts/denoising-diffusion]]（ADM/IDDPM のアーキ改良・AdaGN）, [[concepts/diffusion-sampling]]（classifier-guided DDIM Algorithm 2・25 ステップ）, [[concepts/classifier-free-guidance]]（先行手法 ADM-G との対比）, [[concepts/controllable-generation]]（class-conditional のランドマーク＝ADM）, [[concepts/score-based-generative-models]]（ε＝スコア視点が DDIM guidance の根拠）, [[overview]], [[index]]
- 画像: ar5iv 画像 28 枚（ユニーク）を `raw/assets/2021-adm/` にローカル保存（`samples/` 配下はファイル名を `samples_<name>` に平坦化）。24 JPEG + 4 PNG、全取得成功。説明図 x1/x3/x6/x8（図2/4/5/12）は Read で内容確認のうえキャプション作成。
- 翻訳: 本文§1–8 ＋ **Appendix A–M を全訳**（ユーザー確認済み。K/L/M はサンプルギャラリーのため図キャプションのみ）。References・Acknowledgements 除外。HTML `<table>` 7 個（表1〜7,10,11〜16）を markdown テーブルに再構成、Algorithm 1/2 を擬似コードブロックで保持、数式 LaTeX 保持（H/B の導出も訳出）、ar5iv math 残骸除去。
- メモ: ユーザー確認どおり新規概念ページ `diffusion-model-architecture` を作成（U-Net 改良の系譜、ADM をランドマーク）。ADM 自体は methods/ ページを作らず diffusion-model-architecture＋classifier-guidance に分配。classifier-guidance の「未取り込み」注記を除去（本 ingest で解消）。dangling は `<slug>`（index テンプレ例）のみ。未取り込み注記: IDDPM（本文で言及・diffusion-model-architecture 内に記載）, BigGAN-deep（比較対象, GAN）。

## [2026-06-23] ingest | Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)

- 取り込み: `raw/papers/Adding Conditional Control to Text-to-Image Diffusion Models.md`（ar5iv 由来 markdown, arXiv:2302.05543, ICCV 2023 Marr Prize/Best Paper）
- 作成: [[translations/2023-controlnet]], [[summaries/2023-controlnet]]
- 更新: [[concepts/controllable-generation]]（ControlNet をアダプタ型アプローチとして大幅追記、「スコア操作の逆問題」と「アダプタ／FT」の 2 系統に再構成）, [[concepts/latent-diffusion]]（SD への空間条件追加）, [[concepts/text-to-image-generation]]（空間制御の補完）, [[concepts/classifier-free-guidance]]（CFG-RW）, [[overview]], [[index]]
- 画像: ar5iv 画像 12 枚（ユニーク）を `raw/assets/2023-controlnet/` にローカル保存。全取得成功。
- 翻訳: 本文§1–5 を全訳（**appendix なし**。本文が参照する supplementary materials は原典 markdown に含まれず）。References 除外。Table 1/2/3 を markdown テーブルに再構成。数式 LaTeX 保持＋主要式 (1)–(5)＋CFG 式を保持。ar5iv アーティファクト除去。
- メモ: ユーザー確認どおり ControlNet は新規ページを作らず controllable-generation 内にランドマーク手法として記述。dangling は `<slug>`（index テンプレ例）のみ。未取り込み注記: T2I-Adapter / LoRA / IP-Adapter（アダプタ系）, ADM（classifier-guidance 主要原典）, rectified flow 等（FM 後続）。

## [2026-06-23] ingest | Flow Matching for Generative Modeling

- 取り込み: `raw/papers/Flow Matching for Generative Modeling.md`（ar5iv 由来 markdown, arXiv:2210.02747, ICLR 2023）
- 作成: [[translations/2023-flow-matching]], [[summaries/2023-flow-matching]], [[concepts/flow-matching]]
- 更新: [[concepts/probability-flow-ode]]（CNF/FM の一般化視点・拡散 VF 一致）, [[concepts/score-based-generative-models]]（スコア回帰の VF 版・拡散パス内包）, [[concepts/denoising-diffusion]]（VP/VE パスは FM ガウス族の特別な場合）, [[concepts/diffusion-sampling]]（OT パスの少 NFE サンプリング）, [[concepts/super-resolution]]（FM-OT による超解像、SR3 比較）, [[overview]], [[index]]
- 画像: ar5iv 画像 16 枚（ユニーク）を `raw/assets/2023-flow-matching/` にローカル保存。`2d_vf_reference.png` は本文（図2）と付録 D で再利用＝1 ファイルを 2 箇所から参照。全取得成功。
- 翻訳: 本文§1–7＋**Appendix A–F を全訳**（ユーザー確認済み）。References・Acknowledgements は除外。Table 1（CIFAR-10/ImageNet）・2・3・4 を markdown テーブルに再構成。数式 LaTeX 保持＋主要式 (1)–(26) に \tag、付録の定理証明・CNF 尤度計算 ODE・拡散条件付き VF 導出も訳出。ar5iv アーティファクト除去。
- メモ: 概念ページは新規 flow-matching の 1 枚（ユーザー確認済み、CNF は flow-matching 内に前提として記述）。CLAUDE.md スコープに明記の「flow matching」を概念ページ化。dangling は `<slug>`（index テンプレ例）のみ。未取り込み注記: rectified flow / stochastic interpolants / mini-batch OT（FM 後続）, ADM（classifier-guidance 主要原典）。

## [2026-06-23] ingest | Score-Based Generative Modeling through SDEs (Score-SDE)

- 取り込み: `raw/papers/Score-Based Generative Modeling through Stochastic Differential Equations.md`（ar5iv 由来 markdown, arXiv:2011.13456, ICLR 2021 Outstanding Paper）
- 作成: [[translations/2021-score-sde]], [[summaries/2021-score-sde]], [[concepts/controllable-generation]]
- 更新: [[concepts/score-based-generative-models]]（SDE 統一枠組み＝VE/VP/sub-VP・逆時間 SDE を大幅拡充）, [[concepts/probability-flow-ode]]（主要原典として大幅拡充・「未取り込み」注記を除去）, [[concepts/diffusion-sampling]]（predictor-corrector・逆拡散・ODE サンプラー追記）, [[concepts/denoising-diffusion]]（DDPM=VP-SDE）, [[concepts/classifier-guidance]]（条件付き逆時間 SDE による一般化）, [[overview]], [[index]]
- 画像: ar5iv 画像 18 枚を `raw/assets/2021-score-sde/` にローカル保存（`figures/` 配下はフラット名）。全取得成功。Table 1/4/5 のテーブル埋め込み微小 SVG はテーブル再構成で吸収（独立図ではないため抽出せず）。
- 翻訳: 本文§1–6＋**Appendix A–I を全訳**（ユーザー確認済み、既存最大）。References・Acknowledgements は除外。Table 1/2/3/4/5 を markdown テーブルに再構成。数式 LaTeX 保持＋主要式 (1)–(17) に \tag、付録の Fokker-Planck 導出・PC アルゴリズム擬似コード・可制御生成アルゴリズムも訳出。ar5iv アーティファクト除去。
- メモ: 概念ページは新規 controllable-generation の 1 枚（ユーザー確認済み、SDE 枠組みは score-based-generative-models に統合）。**未取り込み注記のあった 2 本のうち Score-SDE を解消**。残る未取り込み主要原典は ADM（Dhariwal & Nichol, classifier-guidance の主要原典）。

## [2026-06-23] ingest | Classifier-Free Diffusion Guidance (CFG)

- 取り込み: `raw/papers/Classifier-Free Diffusion Guidance.md`（ar5iv 由来 markdown, arXiv:2207.12598, NeurIPS 2021 Workshop on DGMs）
- 作成: [[translations/2022-classifier-free-guidance]], [[summaries/2022-classifier-free-guidance]], [[concepts/classifier-free-guidance]], [[concepts/classifier-guidance]]
- 更新: [[concepts/text-to-image-generation]]（CFG リンク 2 箇所実在化）, [[concepts/latent-diffusion]]（同）, [[summaries/2022-latent-diffusion]]（「未作成」除去）, [[overview]], [[index]]
- 画像: ar5iv 画像 5 枚＋本文埋め込みインライン SVG 2 枚（Fig4・Fig5 の IS/FID 曲線）を `raw/assets/2022-classifier-free-guidance/` にローカル保存（計 7 ファイル）。全取得成功。
- 翻訳: 本文§1–6＋Appendix A を全訳（ユーザー確認済み）。References・Acknowledgements は除外。Table 1・2（HTML table）を markdown テーブルに再構成。数式 LaTeX 保持＋主要式（classifier guidance 式1・2、CFG 式6）に \tag。図1 は ar5iv に画像リンクが無くキャプションのみ＋訳注。
- メモ: 概念ページは classifier-free-guidance＋classifier-guidance の 2 枚（ユーザー確認済み）。**最後の dangling link `[[classifier-free-guidance]]` を解消**。classifier-guidance の主要原典（Dhariwal & Nichol, ADM 2021）は未取り込みで、ページ内に注記。これで本文中の主要 dangling は解消済み（probability-flow-ode 内の Score-SDE 言及は文中注記であり dangling リンクではない）。

## [2026-06-23] ingest | Denoising Diffusion Implicit Models (DDIM)

- 取り込み: `raw/papers/Denoising Diffusion Implicit Models.md`（ar5iv 由来 markdown, arXiv:2010.02502, ICLR 2021）
- 作成: [[translations/2021-ddim]], [[summaries/2021-ddim]], [[concepts/diffusion-sampling]], [[concepts/probability-flow-ode]]
- 更新: [[concepts/denoising-diffusion]]（diffusion-sampling リンク実在化）, [[concepts/score-based-generative-models]]（確率フロー ODE 追記）, [[summaries/2020-ddpm]]・[[summaries/2022-latent-diffusion]]（diffusion-sampling の「未作成」除去）, [[overview]], [[index]]
- 画像: ar5iv 画像 13 枚を `raw/assets/2021-ddim/` にローカル保存。`figures/` 配下はパス由来フラット名（figures_celeba-interp-line.png 等）。全取得成功。
- 翻訳: 本文§1–7＋Appendix A–D を全訳（ユーザー確認済み）。References・Acknowledgements は除外。Table 1（HTML table）と Table 3 を markdown テーブルに再構成。数式 LaTeX 保持＋主要式 (1)–(14) に \tag、Appendix の証明・式変形は省略せず全訳。ar5iv 見出しアーティファクト除去。
- メモ: 概念ページは diffusion-sampling＋probability-flow-ode の 2 枚（ユーザー確認済み）。**dangling link `[[diffusion-sampling]]` を解消**。残存 dangling（後続 ingest 予定）: [[classifier-free-guidance]]（Ho & Salimans）。probability-flow-ode の主要原典（Song ら Score-SDE）も未取り込みで、ページ内に注記。

## [2026-06-23] ingest | High-Resolution Image Synthesis with Latent Diffusion Models (LDM / Stable Diffusion)

- 取り込み: `raw/papers/High-Resolution Image Synthesis with Latent Diffusion Models.md`（ar5iv 由来 markdown, arXiv:2112.10752, CVPR 2022）
- 作成: [[translations/2022-latent-diffusion]], [[summaries/2022-latent-diffusion]], [[concepts/latent-diffusion]], [[concepts/text-to-image-generation]], [[concepts/image-inpainting]], [[concepts/super-resolution]]
- 更新: [[concepts/denoising-diffusion]]（latent-diffusion リンクを実在化）, [[overview]], [[index]]
- 画像: ar5iv 画像 33 枚を `raw/assets/2022-latent-diffusion/` にローカル保存。サブディレクトリ込みのパスを `_` 区切りのフラット名に変換し generic 名（sample_grid-0.jpg 等）の衝突を回避。全取得成功（取得失敗なし）。なお図18 は原典では画像比較表だが ar5iv に画像リンクが無く、翻訳では表＋キャプションのみ（画像なし）。
- 翻訳: 本文§1–6＋Appendix A–H を全訳（ユーザー確認済み）。References は除外。run-on 化していた表（Tab 1,2,3,5,7,8,9,10,11,18 等）は読みやすい markdown テーブルに再構成（数値は原典のまま、大規模ハイパラ表 12–17 は要点を散文＋抜粋整形）。ar5iv 見出しアーティファクト除去、数式 LaTeX 保持＋主要式 (1)–(3) に \tag。
- メモ: 概念ページは latent-diffusion＋text-to-image-generation＋image-inpainting＋super-resolution の 4 枚（ユーザー確認済み）。dangling link `[[latent-diffusion]]` を解消。残存 dangling（後続 ingest 予定）: [[diffusion-sampling]]（DDIM）, [[classifier-free-guidance]]（Ho & Salimans）。GLIDE/Imagen/DALL·E 2 は text-to-image-generation 内に枠だけ用意。

## [2026-06-23] ingest | Denoising Diffusion Probabilistic Models (DDPM)

- 取り込み: `raw/papers/Denoising Diffusion Probabilistic Models.md`（ar5iv 由来 markdown, arXiv:2006.11239, NeurIPS 2020）
- 作成: [[translations/2020-ddpm]], [[summaries/2020-ddpm]], [[concepts/denoising-diffusion]], [[concepts/score-based-generative-models]]
- 更新: [[overview]], [[index]]
- 画像: ar5iv 画像 16 枚＋本文埋め込みインライン SVG 2 枚（図5 レート歪み, 図10 漸進的品質）を `raw/assets/2020-ddpm/` にローカル保存（計 18 ファイル）。`?as=webp` 無し、元名保持。全て取得成功（取得失敗なし）。
- 翻訳: 本文＋Appendix A–D を全訳。References・Acknowledgments は除外。ar5iv 見出しの subscript アーティファクトはクリーンアップ。数式は LaTeX 保持＋式番号 (1)–(16) を \tag で復元。
- メモ: 概念ページは denoising-diffusion ＋ score-based-generative-models の 2 枚（ユーザー確認済み）。landmark 手法 DDPM は denoising-diffusion 内に詳述。未作成リンク（dangling）: [[diffusion-sampling]], [[latent-diffusion]], [[classifier-free-guidance]], [[text-to-image-generation]]（後続 ingest で作成予定）。

## [2026-06-23] schema-update | 3D Vision wiki を Diffusion Model wiki に再テーマ化

- 更新: `CLAUDE.md`, `.claude/skills/{ingest,query,lint}/SKILL.md`
- 領域定義を Diffusion Model（画像生成中心・広め）に差し替え
- `sources/` → `summaries/` に改称（`type`・クロスリファレンスフィールドも一貫変更）
- `methods/` `datasets/` を廃止し、landmark 手法・ベンチマークは概念ページ内に内包する方針に
- ingest: appendix をデフォルトで翻訳対象に含めるよう変更（除外指示がある場合のみ外す）
- 作成: `raw/{papers,articles,images,assets}/`, `wiki/{summaries,translations,concepts,questions}/`, `wiki/{index,log,overview}.md`

## [2026-08-19] ingest | Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers

- 取り込み: `raw/papers/Sana_ Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers.md`（arXiv:2410.10629・NVIDIA / MIT / 清華大学・ICLR 2025）
- 作成: [[translations/2024-sana]], [[summaries/2024-sana]], [[concepts/efficient-attention]]
- 更新: [[concepts/image-tokenizer]], [[concepts/position-embedding]], [[concepts/diffusion-model-architecture]], [[concepts/diffusion-sampling]], [[concepts/noise-schedule]], [[concepts/flow-matching]], [[concepts/data-curation]], [[concepts/prompt-enhancement]], [[concepts/text-to-image-generation]], [[concepts/large-scale-training-infrastructure]], [[concepts/latent-diffusion]], [[concepts/inference-caching]], [[concepts/video-diffusion]], [[overview]], [[index]]
- 翻訳範囲: 本文 §1–7 ＋ Appendix A/B/C の全訳（ユーザー選択：付録も全訳＝スキーマ既定）。表 1–14 を markdown 化。References・謝辞は除外。
- 画像メモ: **ar5iv 由来の画像 16 枚がすべてプレースホルダ**（325×400 のグレースケール、MD5 `ded85833de1226c2b391743296455b30`）だったため、`arxiv.org/html/2410.10629v3` から全 16 枚を再取得し `raw/assets/2024-sana/fig1.png`〜`fig16.png` に保存（図番号と 1:1）。取得失敗ゼロ。
- 原典の欠落と回収: ar5iv 版は **§6 Related Work と 表 2 / 表 3 / 表 4** を落としていたため arXiv HTML から回収。また ar5iv が **図 6 の画像に 表 2 のキャプションを誤って結合**していたので分離した。
- 新概念の設置理由: [[concepts/efficient-attention]] を新設（ユーザー選択）。Wan の 8-bit FlashAttention（[[concepts/large-scale-training-infrastructure]]）と Sana の線形注意は「注意の $O(N^2)$ への応答」という同じ問題への別解だが、これまで受け皿がなかった。近似・疎化・実装最適化・トークン数削減の 4 方向で整理し、[[concepts/inference-caching]]（回数を減らす）と [[concepts/image-tokenizer]]（$N$ を減らす）への橋を張った。
- 既存ページへの補完で特筆すべき点:
  - [[concepts/image-tokenizer]] が抱えていた宿題——HunyuanImage 3.0 の「$f16$ 単体 > $f8$＋パッチ化」という**アブレーションなしの主張**——に、Sana の表 1（F8C16P4 / F16C32P2 / F32C32P1 を同一トークン数で比較）が 2 年前から答えを出していた。あわせて $C=64$ で再構成は改善するのに生成 FID は悪化する、という三者間トレードオフのチャネル軸での実証も追記。
  - [[concepts/position-embedding]] に **(6) NoPE** を追加。Mix-FFN の 3×3 depthwise conv が局所性を供給するため PE を削れるが、**2K/4K の微調整では PE を再導入する**（付録 B）という限界も明記した。本文だけ読むと誤解しやすい箇所。
  - [[concepts/diffusion-sampling]] に Flow-DPM-Solver と、Tweedie の公式に基づく「$t\approx T$ でノイズ予測は $x_t$ の線形関数に退化し、データ予測は定数に近づく」という分析を追加。サンプラー設計の軸が刻み方・次数だけでないことの記録。
  - [[concepts/prompt-enhancement]] に **6 つ目の設計（CHI）** を追加。テキストを書き換えず、LLM テキストエンコーダに指示文を前置して**埋め込みの質だけ**を変える。既存 5 通りの「限界と注意点」の多く（意図の書き換え・評価が PE 込みか曖昧・実効パラメータ数の不透明さ）が構造的に発生しない代わり、能動的な推論はできない。
  - [[concepts/flow-matching]] に、SD3 の 61 定式化比較を補う**同一条件（120K ステップ）での DDPM 対 flow matching の直接比較**（FID 19.5→16.9）を追記。
- メモ: 本 wiki 直近 11 件のうち 10 件がテックレポートだったのに対し、本件は査読を経た学術論文（ICLR 2025）で、アブレーションの統制が明確に良い。既存ページの「アブレーションが示されない」という批判の多くに、後から答えを与える形になった。一方で**線形注意そのものは主流にならなかった**（2025–2026 の大規模モデルはフル注意＋FlashAttention＋トークナイザ圧縮）ため、Sana の遺産は深圧縮 AE と decoder-only LLM テキストエンコーダの 2 点にあると要約ページで評価した。

## [2026-08-19] ingest | Orthogonal Adaptation for Modular Customization of Diffusion Models

- 取り込み: `raw/papers/Orthogonal Adaptation for Modular Customization of Diffusion Models.md`（arXiv:2312.02432・Po ら / Stanford・Snap Research・CVPR 2024）
- 作成: [[translations/2024-orthogonal-adaptation]], [[summaries/2024-orthogonal-adaptation]]
- 更新: [[concepts/lora-merging]]（系統 (5) を新設）, [[concepts/multi-concept-customization]], [[concepts/low-rank-adaptation]], [[overview]], [[index]]
- 翻訳範囲: 本文 §1–6 ＋ 補足 §8–11 の全訳（ユーザー選択：付録も全訳）。表 1–2 を markdown 化（表 2 は HTML テーブルから変換）。References・謝辞は除外。
- 画像メモ: ar5iv 由来の 12 枚をすべて取得し `raw/assets/2024-orthogonal-adaptation/fig1.png`〜`fig12.png` に保存（図番号と 1:1）。プレースホルダ・取得失敗ともにゼロ。
- 位置づけ: [[concepts/lora-merging]] の系統 (0)〜(4) が**すべて事後型**（独立に学習された LoRA を後から混ぜる）だったところに、**干渉を学習時に構造的に排除する**系統 (5) が加わった。$\Delta\theta=AB^\top$ の $B$ を凍結し共有直交基底からランダムに $k$ 列を配ることで $B_i^\top B_j \approx 0$ を保証する。
- 記録すべき指摘:
  - **crosstalk $\|\Delta\theta_j X_i\|$ の定式化**。本 wiki が identity loss と signal interference と別々に呼んできた 2 つの破綻が 1 つの測れる量にまとまった。
  - **DB-LoRA ＋ 素朴な線形和の同一性整合が .683 → .098** と崩壊する数値は、[[concepts/lora-merging]] の「素朴な線形和は使い物にならない」という主張の最も明快な裏づけ。
  - **Mix-of-Show の ED-LoRA だけでも FedAvg の崩壊はかなり防げる**（-.022）。gradient fusion の 15 分はそこから -.011 へ改善するために払われており、費用対効果には厳しい見方ができる。
- 批判として要約に記録した点:
  - **既存 LoRA には適用できない**（原典も明記）。事後型の手法が依然として必要な理由。
  - **直交できる概念数の上限 $\lfloor n/r \rfloor$（SD v1.5・$r$=20 なら 16）を原典が論じていない**。ランダム抽出のため誕生日問題で遥か手前から列が重複するが、概念数増加に伴う劣化が定量化されていない。「無数の概念」というスケーラビリティ主張の中で最も検証が薄い。
  - **表 2 に内部矛盾**（本手法のテキスト整合が .624 → .644 なのに Δ が -.010）。
  - 評価が人物の顔にほぼ限定、12 概念、データセット非公開、ベースが ChilloutMix。

## [2026-08-19] ingest | Implicit Style-Content Separation using B-LoRA

- 取り込み: `raw/papers/Implicit Style-Content Separation using B-LoRA.md`（arXiv:2403.14572・Frenkel ら / テルアビブ大・ライヒマン大・ECCV 2024）
- 作成: [[translations/2024-b-lora]], [[summaries/2024-b-lora]], [[concepts/style-content-disentanglement]]
- 更新: [[concepts/lora-merging]], [[concepts/low-rank-adaptation]], [[concepts/subject-driven-generation]], [[concepts/diffusion-model-architecture]], [[concepts/character-consistency]], [[concepts/instruction-based-image-editing]], [[concepts/multi-concept-customization]], [[overview]], [[index]]
- 翻訳範囲: 本文 §1–6 ＋ Appendix A–E の全訳（ユーザー選択：付録も全訳）。表 1–2 を markdown 化。References・謝辞は除外。
- 画像メモ: **ar5iv 版は図の取り込みが大きく壊れていた**（複合図の 1 枚目だけを抜き出す、図 7・15 が欠落）ため、`arxiv.org/html/2403.14572v2` から回収した。回収の内訳:
  - 単一画像の図 8 枚（図 2・3・4・5・6・9・10・22）はそのまま保存。
  - **図 8（代替手法との比較、5 行 × 7 列 = 35 枚）と図 19（ブロック全組合せのアブレーション、上三角 8×8 = 36 枚）はグリッド構造がファイル名から復元できたため、markdown の表として再構成**した。図 17（5 枚）も同様。計 84 枚を `raw/assets/2024-b-lora/` に保存。
  - **図 1・7・25–30 は ar5iv・arXiv のいずれでも画像が生成されない**。原因は原典が LaTeX の `nicematrix` パッケージ（`NiceTabular`）で組んだ図表であるためで、HTML 中に `nicematrix-placeholder` が残っていることから特定した。訳注で明示。
  - 図 11–16・18・20・21・23・24・31 は 21〜49 枚の大きな比較グリッドで、構成画像に分解されて配置が失われている。構成画像は回収せず訳注でキャプションのみ訳出した。
- 新概念の設置理由: [[concepts/style-content-disentanglement]] を新設（ユーザー選択）。ZipLoRA が [[concepts/lora-merging]] の系統 (3) として既に入っていたが、**「スタイルとコンテンツを分離する」という問題設定そのもの**の受け皿がなかった。(a) 別々に学習してマージ / (b) 内在する分離を見つける / (c) 注意特徴の共有 / (d) エンコーダ注入、の 4 系統で整理。
- 記録すべき指摘:
  - **「共同で学習する」が本質**。$\{\Delta W^4, \Delta W^5\}$ を独立に学習しても分離しない。付録 C の上三角 8×8 アブレーションで $(2,5)$ が「再構成は良いが分離は弱い」ことが示され、**再構成品質と分離品質は別物**という指針が得られた。
  - **同時期の InstantStyle も独立に同じ第 5 ブロックをスタイル用に選んでいる**。現象の実在の強い傍証。
  - **図 3 から、$W_0^4$ と $W_0^5$ はどちらも Up Block 0（デコーダ側の隣接ブロック）**であることを確認した。分離は UNet の遠い場所にあるのではなく隣り合う 2 ブロックの間に走っている。
- 批判として要約に記録した点:
  - **「スタイル」の実体は「色」**。原典は CLIP の性質上、色をスタイルの代理として使ったと明記しており、そのツケが「物体固有の色までスタイル側に取られて同一性が壊れる」という限界として返っている（付録 B の $\alpha \in [0.4,0.5]$ は対症療法）。
  - **SDXL のブロック番号への強い依存**。[[concepts/diffusion-model-architecture]] の潮流は DiT / MM-DiT / 単一ストリームへ進んでおり、均質な Transformer 積層で同じ分離が現れる保証はない。本知見の賞味期限に関わる最大の論点として概念ページにも記録した。
  - **評価指標の交絡**——コンテンツ類似度が高いモデルは単に過学習しているだけかもしれない。原典は参照 1 枚に絞る追加実験でこれを検証しており（全手法でスタイル↓コンテンツ↑）、指標そのものを疑う姿勢は [[concepts/aesthetic-scoring]] と同型。この論点を新概念ページの中心に据えた。
  - ブロック対アブレーションが 2 物体のみ、比較実装の多くが非公式、本文の「単一画像」と付録 D（複数画像で personalization）の齟齬。
- メモ: この 2 本は「LoRA 合成で他に読むべき論文は？」というユーザーの問いに対して推薦した #1 と #3 にあたる。読み合わせると、2023 年末〜2024 年に**マージ機構そのものを不要にする**という共通の転換が起きていたことが見える——Orthogonal Adaptation は「どう学習するか」を、B-LoRA は「どこを学習するか」を変えた。[[concepts/lora-merging]] に「第三の道」の節を設けて両者を並置した。
- 付随修正: 既存の [[translations/2025-qwen-image]] に残っていたキリル文字の混入（「過小представされた」×3）を「過小表現された」に修正。
