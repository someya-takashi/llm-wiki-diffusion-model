---
type: summary
source_path: raw/papers/Qwen-Image Technical Report.pdf
source_kind: paper
title: "Qwen-Image Technical Report"
authors: [Qwen Team (Alibaba), Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Shuai Bai]
year: 2025
venue: "arXiv:2508.02324（テクニカルレポート）"
ingested: 2026-06-25
tags: [text-to-image-generation, visual-text-rendering, instruction-based-image-editing, reinforcement-learning-for-diffusion, diffusion-model-architecture, flow-matching]
translation: "[[translations/2025-qwen-image]]"
---

# Qwen-Image — 複雑なテキストレンダリングと精密な画像編集の基盤モデル

> 原典: [[translations/2025-qwen-image]] ・ `raw/papers/Qwen-Image Technical Report.pdf`
> 著者・年・出典: Qwen Team（Alibaba）, arXiv:2508.02324, 2025年8月
> 公開: https://github.com/QwenLM/Qwen-Image ・ https://huggingface.co/Qwen/Qwen-Image

## 一言まとめ

**20B の MMDiT（Multimodal Diffusion Transformer）に、凍結した Qwen2.5-VL をテキストエンコーダとして接続した text-to-image 基盤モデル**。最大の特徴は **画像内テキストのレンダリング（[[visual-text-rendering]]）**、とりわけ**中国語のような表語文字**で他を圧倒すること（ChineseWord 総合 58.30 対 GPT Image 1 の 36.14）。さらに **意味特徴＋再構成特徴の二重符号化**で一貫した [[instruction-based-image-editing]] を実現し、**DPO と Flow-GRPO による事後強化学習**（[[reinforcement-learning-for-diffusion]]）で GenEval を 0.87→0.91 に押し上げた。オープンソースで AI Arena 3 位。

## 背景と問題意識

拡散モデルによる text-to-image 生成（[[text-to-image-generation]]）は SDXL・SD3・FLUX で実用水準に達したが、著者らは残る 2 つの壁を指摘する。

**壁 1：複雑なテキストレンダリング。** 画像の中に「読める文字」を描くのは、猫や風景を描くのとは質的に異なる難題である。文字は形が少しでも崩れると別の字（あるいは無意味な模様）になってしまうため、生成モデルが得意とする「それらしさ」では通用しない。とくに**表語文字（logographic language, 中国語のように 1 文字が意味の単位をなす文字体系）**は、常用漢字だけで数千字あり、しかも実世界の画像データでの出現頻度が極端に偏る（ロングテール）。GPT Image 1 や Seedream 3.0 でさえ、複数行・多領域・非ラテン文字のレンダリングでは崩れる。

**壁 2：画像編集の一貫性。** 「髪の色だけ変えて」と指示したとき、(i) **視覚的一貫性**（顔の細部など指示していない部分を保つ）と (ii) **意味的一貫性**（ポーズを変えても本人だと分かる、シーンが破綻しない）を両立させる必要がある。従来手法はどちらかに寄ってしまう。

## 提案手法 / 主張

### アーキテクチャ：凍結 MLLM ＋ 動画 VAE ＋ 20B MMDiT

3 つの部品からなる（本文 図6）。

1. **条件エンコーダ = Qwen2.5-VL（7B, 凍結）**。CLIP や T5 のような専用テキストエンコーダではなく、**視覚と言語がすでに整合済みのマルチモーダル LLM** をそのまま使う。理由は、(a) T2I に適した視覚-言語整合が済んでいる、(b) 言語モデルとしての能力を保っている、(c) 画像入力を受けられるので編集タスクへ自然に拡張できる。最終層の隠れ状態を条件表現に使い、タスク別のシステムプロンプトで出力を誘導する。
2. **VAE = Wan-2.1-VAE ベースの「単一エンコーダ・二重デコーダ」**（[[latent-diffusion]]）。エンコーダは画像・動画で共有して凍結し、**画像デコーダだけを、PDF・スライド・ポスターなどテキストが密な社内コーパスで微調整**する。ここが効く：デコーダだけを鍛えることで小さな文字の再構成が劇的に改善し（テキスト画像 PSNR 36.63 と全 VAE 中最高、Wan2.1-VAE の 26.77 から大幅改善）、テキストレンダリング能力の「上限」が引き上げられる。学習は再構成損失＋知覚損失のみ（品質が上がると識別器が有効な勾配を出せなくなるため敵対的損失は外した）。
3. **バックボーン = MMDiT（20B, 60 層）**。SD3（[[summaries/2024-sd3]]）が導入したテキスト・画像の二流構造（[[diffusion-model-architecture]]）を踏襲する。

### MSRoPE — テキストを「画像の対角線」に置く位置符号化

MMDiT ではテキストと画像の位置符号化をどう共存させるかが課題になる。従来は平坦化した画像位置の後ろにテキストを連結し、Seedream 3.0 の Scaling RoPE は画像位置を中心にずらしてテキストを [1, L] の 2D トークンとして扱った。しかし後者では**テキストと画像 0 行目の位置符号化が同型**になり、モデルが両者を区別しづらい。

**MSRoPE（Multimodal Scalable RoPE）** は、テキスト入力を「縦横で同じ位置 ID を持つ 2D テンソル」として扱う——すなわち**テキストが画像の対角線上に並んでいるものとみなす**。これにより画像側は解像度スケーリングの恩恵を受けつつ、テキスト側は 1D-RoPE と機能的に等価なまま保たれ、「テキストをどこに連結するか」を悩む必要がなくなる。

### 学習：rectified flow ＋ カリキュラム学習

事前学習の目的関数は **flow matching**（[[flow-matching]]）である。**rectified flow** の直線パス $x_t=tx_0+(1-t)x_1$、目標速度 $v_t=x_0-x_1$ に対し速度を回帰する（タイムステップは **logit-normal 分布**からサンプル）。これは SD3 と同じ定式化で、[[noise-schedule]] でいう「中盤の難しい時刻を重点的に学ぶ」設計にあたる。

学習戦略は 5 軸の**カリキュラム学習**として設計される：
- **解像度**: 256² → 640² → 1328²
- **テキスト**: 非テキスト画像 → テキスト入り画像（先に一般的な視覚表現を獲得させてから文字を教える）
- **データ品質**: 大規模 → 厳選（7 段階フィルタ S1–S7）
- **データ分布**: 不均衡 → バランス
- **データ源**: 実世界 → 合成

**データ合成**（§3.4）が中国語の切り札。実画像だけでは稀な漢字に十分触れられないので、3 戦略で合成する：**Pure Rendering**（無地背景にコーパスから抽出した段落をレンダリング。1 文字でもフォント欠落等でレンダリング失敗したら段落ごと破棄という厳格な品質管理）、**Compositional Rendering**（紙・木板などに書かれた文字を実写背景に合成）、**Complex Rendering**（スライドや UI テンプレートをプログラムで書き換え、複雑レイアウトを作る）。

### 事後学習：SFT → DPO → Flow-GRPO

**[[reinforcement-learning-for-diffusion]]** を本格導入した点が新しい。SFT で写実性・詳細を底上げした後、
- **DPO（Direct Preference Optimization）**：同一プロンプトから複数シードで生成 → 人手で best/worst を選別 → **flow matching の速度予測誤差の差**を選好スコアとして最適化する（式3）。1 ステップで済み計算効率が良いので大規模に回す。
- **GRPO（Flow-GRPO）**：グループ内の報酬を標準化してアドバンテージを作り、方策比のクリップ付き目的を最適化（式4–5）。ただし flow matching の ODE サンプリングは決定論的で探索にならないため、**サンプリングを SDE に再定式化**してランダム性を注入する（式6–7、KL は閉形式・式8）。小規模で細粒度の仕上げに使う。

### 編集：二重符号化（dual-encoding）

TI2I（text-image-to-image）では入力画像を**2 通りに符号化して両方流す**：
- **Qwen2.5-VL の ViT 特徴 → テキストストリーム**（何が写っていて何をすべきかという**意味**）
- **VAE 潜在 → 画像ストリーム**（ノイズ潜在とシーケンス方向に連結。**画素レベルの見た目**）

さらに MSRoPE に **frame 次元**を足して「編集前／編集後」の画像を区別する。著者らの観察では、MLLM 側が指示追従を、VAE 側が構造的一貫性を担う——冒頭の「壁 2」に対する直接的な答えになっている。また学習には T2I・TI2I に加えて **I2I 再構成**も混ぜ、Qwen2.5-VL と MMDiT の潜在表現を整合させる。

## 実験結果と知見

- **一般生成**：DPG 88.32（1 位）、GenEval 0.87 →**RL 後 0.91**（リーダーボードで唯一 0.9 超え）、OneIG-Bench は英中とも総合 1 位（EN 0.539 / ZH 0.548）。TIIF は GPT Image 1 に次ぐ 2 位。
- **テキストレンダリング**：**ChineseWord 58.30**（GPT Image 1 は 36.14、Seedream 3.0 は 33.05）。特に Level-1（常用 3500 字）で **97.29%**。LongText-Bench は中国語 0.946 で 1 位、英語 0.943 で 2 位。英語の CVTG-2K は GPT Image 1 にわずかに及ばない（0.8288 対 0.8569）が CLIPScore は最高。
- **編集**：GEdit で英中とも 1 位（EN 総合 7.56 / CN 7.52）。ImgEdit 総合 4.27 で 1 位。**FLUX.1 Kontext [Pro] は GEdit-CN が 1.23 と壊滅**（中国語能力の欠如）で、多言語性の差が際立つ。
- **「編集」として解いた視覚タスク**：新視点合成（GSO）で PSNR 15.11 と専用 3D モデル CRM（15.93）に肉薄し、汎用生成モデルの GPT Image 1（12.07）を大きく上回る。深度推定も SFT のみで DepthAnything 級に迫る。
- **人間評価**：AI Arena（Elo, 200 名以上・各モデル 1 万回以上の対戦）で**オープンソース唯一のトップ 3**。Imagen 4 Ultra に約 30 Elo 差で及ばないが、GPT Image 1 [High] や FLUX.1 Kontext [Pro] には 30 Elo 以上の差をつける。
- **VAE**：デコーダのみの微調整で ImageNet PSNR 33.42・テキスト画像 PSNR 36.63 と全比較対象中最高。エンコーダ 19M／デコーダ 25M しか活性化しない効率。

## 限界・批判的視点

- **Level-3 の漢字は 6.48% しか描けない**。Level-1 が 97% なのと対照的で、ロングテールの稀少文字は依然として未解決（合成データでも救い切れていない）。
- **英語テキストの精度では GPT Image 1 に及ばない**（CVTG-2K 平均 0.8288 対 0.8569、LongText-Bench-EN も 2 位）。強みはあくまで中国語と長文・複雑レイアウト。
- **モデルが重い**：MMDiT 20B ＋ VLM 7B。学習には 4-way テンソル並列・Megatron・専用の Producer-Consumer データパイプラインを要し、追試のハードルが高い。活性化チェックポイントは反復時間 3.75× の代償が大きく不採用（メモリ削減は 11.3% のみ）という実務的知見も報告される。
- **テクニカルレポートであり ablation が限定的**。MSRoPE・二重符号化・カリキュラム各要素の寄与を切り分けた比較は示されていない（RL の効果のみ GenEval で SFT 版と比較）。
- 評価の多くが **GPT-4.1 を審判に使う**（GEdit・ImgEdit）ため、審判モデル依存のバイアスが残りうる。

## 既存 wiki との接続

Qwen-Image は [[text-to-image-generation]] の最新世代であり、アーキテクチャ的には SD3（[[summaries/2024-sd3]]）が確立した **MM-DiT ＋ rectified flow**（[[diffusion-model-architecture]]・[[flow-matching]]）の直系である。SD3 が 3 つのテキストエンコーダ（CLIP×2＋T5）を併用したのに対し、Qwen-Image は**マルチモーダル LLM 1 本**に置き換え、それにより編集タスクへ地続きに拡張した点が設計上の分岐になる。潜在空間で拡散する枠組みは [[latent-diffusion]] のままだが、**動画対応 VAE のデコーダのみを微調整**して文字再現を稼ぐ発想は本論文に特徴的である。本論文が本 wiki にもたらした 3 つの新概念——[[visual-text-rendering]]（文字描画）・[[instruction-based-image-editing]](指示編集)・[[reinforcement-learning-for-diffusion]]（DPO/GRPO）——は、いずれも「基盤モデルの品質を最後に詰める」段階の技術として、[[classifier-free-guidance]] や [[subject-driven-generation]] と並ぶ実用面の柱になる。

## 用語と略称

- **MMDiT** = Multimodal Diffusion Transformer（テキストと画像に別重みを与え attention で結合する拡散 Transformer。SD3 由来）。
- **MLLM** = Multimodal Large Language Model（マルチモーダル大規模言語モデル。ここでは Qwen2.5-VL）。
- **VAE** = Variational AutoEncoder（画像を潜在表現に圧縮・復元するトークナイザ）。
- **MSRoPE** = Multimodal Scalable RoPE（テキストを画像の対角線上に配置する位置符号化。RoPE = Rotary Position Embedding, 回転位置埋め込み）。
- **T2I / I2I / TI2I** = text-to-image / image-to-image / text-image-to-image（テキストのみ／画像のみ／テキスト＋画像を条件とする生成）。
- **rectified flow（整流フロー）** = ノイズとデータを直線で結ぶ flow matching のインスタンス。
- **logit-normal 分布** = ロジット変換した値が正規分布に従う分布。学習時刻 $t$ を中盤に偏らせるのに使う。
- **DPO** = Direct Preference Optimization（直接選好最適化。良い／悪いの対比較データから報酬モデルなしで直接学習）。
- **GRPO** = Group Relative Policy Optimization（グループ相対方策最適化。同一プロンプトの生成群内で報酬を標準化してアドバンテージにする）。**Flow-GRPO** は flow matching 用に SDE サンプリングで探索性を与えた版。
- **SFT** = Supervised Fine-Tuning（教師あり微調整）。
- **表語文字（logographic language）** = 1 文字が語や形態素を表す文字体系（中国語・漢字）。対して**表音文字（alphabetic language）** は音を表す（英語のアルファベット）。
- **PSNR / SSIM / LPIPS** = 再構成品質の指標（ピーク信号対雑音比／構造的類似性／知覚的類似度。LPIPS は小さいほど良い）。
- **NED** = Normalized Edit Distance（正規化編集距離。テキストレンダリングの文字列一致度）。
- **Elo レーティング** = 対戦成績から相対的強さを推定する仕組み（チェス由来）。AI Arena で採用。
- **ベンチマーク**：**GenEval**（物体中心の構成的生成）、**DPG**（密なプロンプト遵守）、**OneIG-Bench**（多次元の細粒度評価、英中）、**TIIF**（複雑指示の追従）、**CVTG-2K**（英語の多領域テキスト描画）、**ChineseWord**（本論文が新設した漢字レンダリング評価）、**LongText-Bench**（長文描画）、**GEdit / ImgEdit**（指示ベース編集）、**GSO**（新視点合成）。

## 関連ページ

- [[concepts/visual-text-rendering]]
- [[concepts/instruction-based-image-editing]]
- [[concepts/reinforcement-learning-for-diffusion]]
- [[concepts/text-to-image-generation]]
- [[concepts/diffusion-model-architecture]]
- [[concepts/flow-matching]]
- [[concepts/latent-diffusion]]
- [[summaries/2024-sd3]]
- [[translations/2025-qwen-image]]
