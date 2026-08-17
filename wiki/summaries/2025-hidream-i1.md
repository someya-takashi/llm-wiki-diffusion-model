---
type: summary
source_path: raw/papers/HiDream-I1_ A High-Efficient Image Generative Foundation Model with Sparse Diffusion Transformer.md
source_kind: paper
title: "HiDream-I1: A High-Efficient Image Generative Foundation Model with Sparse Diffusion Transformer"
authors: [HiDream.ai (Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Ting Yao, Zhaofan Qiu, Yiheng Zhang ほか)]
year: 2025
venue: arXiv:2505.22705（テクニカルレポート）
ingested: 2026-08-17
tags: [mixture-of-experts-diffusion, diffusion-model-architecture, text-to-image-generation, flow-matching, diffusion-distillation, instruction-based-image-editing, latent-diffusion]
translation: "[[translations/2025-hidream-i1]]"
---

# HiDream-I1: 疎な Diffusion Transformer による高効率な画像生成基盤モデル

> 原典: [[translations/2025-hidream-i1]] ・ `raw/papers/HiDream-I1_ A High-Efficient Image Generative Foundation Model with Sparse Diffusion Transformer.md`
> 著者・年: HiDream.ai（責任著者 Ting Yao, Tao Mei）・2025 年 5 月・arXiv:2505.22705

## 一言まとめ

**MM-DiT 系の二重ストリーム→単一ストリーム構成の FFN を、疎な Mixture-of-Experts（MoE）に置き換えた 17B のオープンソース T2I 基盤モデル**。同時に、4 種類のテキストエンコーダを混ぜる「ハイブリッド符号化」と、DMD に敵対的損失を足した「GAN を用いた蒸留」を提示し、HPSv2.1・GenEval・DPG-Bench で当時の SOTA を主張した。

## 背景と問題意識

本 wiki がここまで追ってきた 2024–2025 年の T2I 基盤モデルは、**品質を上げるほど計算量とレイテンシが増える**という一本道を歩いてきた。SD3（[[summaries/2024-sd3]]）の MM-DiT、FLUX.1（[[summaries/2025-flux-kontext]]）、Qwen-Image（[[summaries/2025-qwen-image]]）はいずれもパラメータを増やす方向で品質を稼いでいる。HiDream-I1 の出発点は「**その費用対効果をアーキテクチャで改善できないか**」という問いである。

ここで持ち込まれるのが **MoE（Mixture-of-Experts, 混合エキスパート）** ——大規模言語モデル側で確立した「**パラメータは増やすが、1 トークンあたりに実際に使う分は増やさない**」仕組みである。Transformer ブロックの FFN（feed-forward network, 各トークンを独立に変換する 2 層の全結合部）を、複数の並列な「エキスパート」FFN に分割し、**ルーター（router）** と呼ばれる小さなゲーティングネットワークがトークンごとに上位数個だけを選んで通す。総パラメータ（＝モデルの容量）は N 倍になるのに、1 トークンあたりの計算量（＝活性化パラメータ）はほぼ据え置きになる。これが「**疎（sparse）**」と呼ばれる所以で、拡散モデルへの本格的な適用例として本レポートは早い部類に入る。

もう 1 つの問題意識はテキスト条件付けにある。CLIP（対照学習で画像とテキストを同じ空間に埋める）は大域的な視覚接地に強いが長文に弱く、T5 は複雑な文構造の解析に強いが視覚と結びついていない、LLM は深い意味理解を持つ——という**相補性**をどう扱うか。Qwen-Image が「凍結 MLLM 1 本に集約する」という**引き算**の答えを出したのに対し、HiDream-I1 は **4 系統を全部混ぜる足し算**という正反対の答えを出した。

## 提案手法 / 主張

### 1. ハイブリッドなテキスト符号化 — 4 系統を混ぜる

- **Long-CLIP L/14 と G/14**：長文コンテキストに拡張された CLIP。プールされた 1 本のベクトル $h_\text{clip}$ として、**adaLN（adaptive layer normalization, 条件からスケールとシフトを作って正規化に流し込む条件付け）** による大域条件に使う。
- **T5-XXL**：系列埋め込み $h_\text{t5}$。
- **Llama 3.1 8B Instruct の複数の中間層**：$h_\text{llm}\in\mathbb{R}^{L\times M\times d}$。最終層ではなく**中間層を複数タップする**のがポイントで、最終層出力では薄まってしまう細粒度の意味的詳細を残す狙い。

T5 と Llama 中間層の系列を線形射影して**連結**したものが DiT へのテキスト条件系列になり、プール済み CLIP はタイムステップ埋め込みと**足し合わされて** adaLN の変調に回る（図3(a)）。

### 2. 疎な DiT バックボーン — どこに MoE を置くか

全体構成は FLUX.1 と同じく **dual-stream → single-stream**。前半は画像とテキストに別重みを与えて attention でのみ混ぜ、後半は両者を連結して 1 本で処理する。

MoE の配置が本論文の核だが、**本文の記述と図3 が食い違う**（後述）。図3(b)(c) が実際に描いているのは：

| ブロック | テキスト側 FFN | 画像側 FFN |
| --- | --- | --- |
| Dual-stream | **dense SwiGLU** | **MoE** |
| Single-stream | （連結後 1 本）**MoE** | |

MoE ブロックは図3(d) の通り、**ルーター＋複数エキスパート＋常時通る共有エキスパート（shared expert）** の構成で、各エキスパートの中身は **SwiGLU**（Swish-Gated Linear Unit, ゲート付き活性化。標準の FFN より表現力が高い）。共有エキスパートを置くのは「全トークンに共通して必要な変換」をルーティングの対象外に逃がし、専門エキスパートを本当に専門化させるための定石である。

安定化には SD3 由来の **QK-normalization**（attention の query/key を正規化して発散を防ぐ）を使う。

### 3. GAN を用いた拡散モデル蒸留

50 ステップの HiDream-I1-Full を教師に、**DMD（Distribution Matching Distillation, 生徒の生成分布を教師の分布に合わせる蒸留）** で 28 ステップ（Dev）と 14〜16 ステップ（Fast）の生徒を作る。ただし DMD だけでは細部の鮮鋭さが失われるため、**識別器を足して敵対的損失を併用**する：

$$\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{DMD}}+\lambda_{\text{adv}}\mathcal{L}_{\text{adv}}$$

識別器は独立したネットワークではなく、**凍結した教師バックボーンの多段階特徴**を使って本物／偽物を判定する。これは [[diffusion-distillation]] でいう「(3) 敵対的蒸留」と「(2) 分布マッチング」を**組み合わせた**形で、FLUX.1 Kontext の LADD が敵対単独だったのと対照的である。

### 4. HiDream-E1 — 潜在マップを「横に並べる」編集

ソース画像とターゲット画像をそれぞれ VAE で符号化し、**潜在マップを空間的に横並び（side-by-side）に連結**して、その全体を latent flow matching で生成させる。さらに $Z_T$ と $Z_S$ の差が大きい領域の損失を重く取る**空間重み付き損失** $\mathcal{L}_\text{Edit-Weighted}$ で「変えるべき所だけ変える」を促す。500 万件の（ソース, 指示, ターゲット）三つ組で微調整。

これは同時期の FLUX.1 Kontext が採った「**系列方向**に連結して 3D RoPE の時間軸でずらす」やり方と好対照で、[[instruction-based-image-editing]] における文脈型編集の 2 つの実装形態を並べて見られる。

## 実験結果と知見

**表3（HPSv2.1）** — Human Preference Score v2.1（人間の選好を予測するよう学習されたモデルによる採点）で平均 33.82、アニメーション／コンセプトアート／絵画／写真の**全カテゴリで 1 位**。Stable Cascade（32.95）、FLUX.1-dev（32.47）、SD3（31.53）を上回る。

**表2（GenEval）** — 総合 0.83 で 1 位。単一物体 1.00、2 物体 0.98 は最高。ただし **Position は 0.60 で Janus-Pro-7B の 0.79 に負けている**——「複数物体を正しい位置関係で置く」は苦手側に残った。

**表1（DPG-Bench）** — 総合 85.89 で 1 位。Relation 93.74・Other 91.83 が強い一方、**Global は 76.44 と全モデル中で最下位級**。これは「画像全体の雰囲気やスタイルを指定する記述」への追従で、細部の関係性は捉えるが大域的なトーン指定は取りこぼす、という偏りを示唆する。

**表4（EmuEdit / ReasonEdit）** — HiDream-E1 は EmuEdit 平均 6.40、ReasonEdit 7.54 で、クローズドソースの Gemini-2.0-Flash（5.99 / 6.95）を上回る。評価は GPT-4o に「(1) 指示の実行成功度」「(2) 過剰編集のなさ」を 0–10 で採点させ、**その最小値**を取る——片方だけ良くても点にならない設計で、[[instruction-based-image-editing]] の「視覚的一貫性↔意味的一貫性」の綱引きを 1 スカラーに落とし込む工夫として参考になる。

## 限界・批判的視点

- **本文と図の不一致**。§3.2 は「二重ストリームと単一ストリームの**両方**が密な FFN の代わりに MoE 構造を用いる」と書き、貢献リストは「異なるエキスパートが**様々な種類のテキスト入力**を扱うことを学習できる」と説明する。しかし図3(b) は明確に、**テキスト側が dense SwiGLU、画像側が MoE** と描いている。どちらが実装かは本文からは決められない。MoE が本論文の目玉である以上、これは軽くない曖昧さである。
- **MoE のハイパーパラメータが一切書かれていない**。エキスパート数、上位いくつを選ぶか（top-k）、負荷分散損失（load balancing loss, ルーティングが一部のエキスパートに偏るのを防ぐ正則化）の有無、**活性化パラメータ数**——MoE を名乗る以上もっとも知りたい「17B のうち実際に何 B が動くのか」が不明のままである。費用対効果を主張の柱に据えているのに、その定量的裏付け（レイテンシ・FLOPs の対比表）が存在しない。
- **アブレーションが皆無**。MoE を dense に戻したらどうなるか、テキストエンコーダ 4 系統のうちどれが効いているか、敵対的損失を外すと蒸留品質がどう落ちるか——いずれも検証されていない。テクニカルレポートという体裁上やむを得ない面はあるが、**設計判断の根拠は読者に委ねられている**。
- **§2.3「データフィルタリング」が見出しだけで中身がない**（ar5iv 変換の欠落ではなく、参考文献に LAION の美的スコアラ・NSFW 検出器・透かし検出器が挙がっているので、姉妹論文 HiDream-O1-Image の §2.3 に相当する内容が意図されていたと推測される）。
- **蒸留の「ステップ数」が本文内で食い違う**。Abstract と Introduction は HiDream-I1-Fast を 14 ステップと書き、§5 は 16 ステップと書いている。
- **評価がプロンプト追従と選好スコアに閉じている**。FLUX.1 Kontext が指摘した **bakeyness**（単一の選好比較が「AI っぽい」美学に報いてしまう問題・[[text-to-image-generation]]）の批判は、HPSv2.1 を主軸に据える本レポートにそのまま当てはまる。
- **テキストレンダリングの評価がない**。後年、姉妹論文 [[summaries/2026-hidream-o1-image]] の表5 が、HiDream-I1-Full の LongText-Bench スコアを **英語 0.543 / 中国語 0.024** と報告している。中国語はほぼ全滅であり、当時の [[visual-text-rendering]] の水準に照らしても弱い部分が、自レポートでは触れられていなかったことになる。

## 用語と略称

- **MoE** = Mixture-of-Experts（混合エキスパート）。複数の並列 FFN からルーターがトークンごとに一部だけ選んで通す仕組み。総容量を増やしつつ計算量を抑える。
- **DiT** = Diffusion Transformer。U-Net の代わりに Transformer を拡散モデルのバックボーンに使う設計。
- **MM-DiT** = Multimodal DiT。画像とテキストに別々の重みを与え、attention でのみ混ぜる SD3 の二重ストリーム設計。
- **SwiGLU** = Swish-Gated Linear Unit。ゲート機構付きの活性化を用いる FFN の変種。
- **adaLN** = adaptive Layer Normalization。条件ベクトルから正規化のスケールとシフトを作る条件付け方式。
- **QK-normalization**：attention の query と key を正規化し、大規模学習での発散を防ぐ手法（SD3 由来）。
- **DMD** = Distribution Matching Distillation。生徒の生成分布を教師の分布に合わせる蒸留。
- **FSDP** = Fully Sharded Data Parallel。モデル・勾配・オプティマイザ状態を GPU 間で分割して大規模学習を可能にする並列化。
- **VLM** = Vision-Language Model（視覚言語モデル）。ここでは学習画像のキャプション自動生成に MiniCPM-V 2.6 を使用。
- **SSCD**：画像コピー検出のための自己教師あり記述子。重複除去の特徴抽出に使用。
- **Faiss**：大規模な近傍探索ライブラリ。
- **HPSv2.1** = Human Preference Score v2.1。人間の選好を予測するモデルによる自動採点。
- **GenEval / DPG-Bench**：プロンプト追従性（物体数・属性・位置・関係）を測る T2I ベンチマーク。
- **EmuEdit / ReasonEdit**：指示ベース編集のベンチマーク。前者は 10 種 3,589 サンプル、後者は推論を要する 197 サンプル。
- **AIGC** = AI-Generated Content。

## 関連ページ

- [[concepts/mixture-of-experts-diffusion]] — 本論文が拡散モデルへ持ち込んだ疎な MoE
- [[concepts/diffusion-model-architecture]] — dual-stream → single-stream 構成の系譜における位置づけ
- [[concepts/diffusion-distillation]] — DMD ＋ 敵対的損失という組み合わせ型の蒸留
- [[concepts/instruction-based-image-editing]] — HiDream-E1 の「潜在マップを横に並べる」編集
- [[concepts/text-to-image-generation]] — T2I 基盤モデルとしての評価
- [[concepts/flow-matching]] — 学習定式化（latent flow matching・線形補間パス）
- [[summaries/2026-hidream-o1-image]] — 同チームの後継。VAE も外部テキストエンコーダも捨てる方向へ反転する
- [[summaries/2024-sd3]] — MM-DiT の原典。二重ストリームの源流
- [[summaries/2025-flux-kontext]] — 同時期の dual→single stream 実装と、系列連結型の編集
- [[summaries/2025-qwen-image]] — テキスト符号化について正反対の設計判断（凍結 MLLM 1 本）
