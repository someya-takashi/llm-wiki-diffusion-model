---
type: concept
aliases: [ai-toolkit, ostris/ai-toolkit, AI Toolkit, LoRA トレーナ, LoRA 学習ツール]
tags: [ai-toolkit, low-rank-adaptation, lora-merging, noise-schedule, large-scale-training-infrastructure, diffusion-distillation, training-tooling]
related:
  - "[[low-rank-adaptation]]"
  - "[[lora-merging]]"
  - "[[noise-schedule]]"
  - "[[large-scale-training-infrastructure]]"
  - "[[diffusion-distillation]]"
  - "[[flow-matching]]"
summaries:
  - "[[summaries/2026-ai-toolkit]]"
  - "[[summaries/2025-flux2]]"
updated: 2026-08-26
---

# ai-toolkit（LoRA 学習ツール）

> ⚠️ **本ページは CLAUDE.md §1 の「ツールに専用ページを作らない」という規約に対する、ユーザーの判断による明示的な例外である。** ai-toolkit は本 wiki が追ってきたモデル群を実際に学習する唯一の共通基盤であり、独立したノードとして参照できることに実務上の価値があるため。**リポジトリのコードに固有の事実（既定値・行番号・実装の癖）は [[summaries/2026-ai-toolkit]] に置き、本ページは「wiki の他の概念とどう繋がるか」に絞る。**

**ai-toolkit**（[ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)・MIT）は、拡散モデルの LoRA 学習・微調整を単一の設定ファイル形式で行うスイートである。本 wiki にとっての意味は、**理論として追ってきたモデルが実際に手元で学習できるかどうかを決める層**である点にある。論文が何を主張していても、トレーナが対応していなければ試せない。

## 本 wiki が追ってきたモデルとの対応

本 wiki は 2025–2026 の基盤モデルを 40 本近く取り込んできたが、**そのほとんどが ai-toolkit で学習可能な状態にある**。

| wiki の要約                                                            | `arch`                                        | 特記                                       |
| ------------------------------------------------------------------- | --------------------------------------------- | ---------------------------------------- |
| [[summaries/2025-flux2]]                                            | `flux2` / `flux2_klein_4b` / `flux2_klein_9b` | **klein は base のみ**。蒸留版の arch は無い        |
| [[summaries/2025-z-image]]                                          | `zimage` / `zimage_l2p`                       | **de-distillation アダプタあり**（Turbo を学習できる） |
| [[summaries/2025-qwen-image]] / [[summaries/2026-qwen-image-2]]     | `qwen_image` ほか 3 種                           | 編集版も別 arch                               |
| [[summaries/2026-ernie-image]]                                      | `ernie_image`                                 |                                          |
| [[summaries/2025-hidream-i1]] / [[summaries/2026-hidream-o1-image]] | `hidream` / `hidream_e1` / `hidream_o1`       |                                          |
| [[summaries/2025-wan]]                                              | `wan22_5b` / `wan22_14b` / `wan22_14b_i2v`    | de-distillation アダプタあり                   |
| [[summaries/2025-flux-kontext]]                                     | `flux_kontext`                                |                                          |

**この表は「読んだ理論を試せるか」の地図として使える。** 逆に言えば、arch が無いモデルは自前で実装しない限り触れない。

## ブロック名が 3 つの役割を兼ねる

ai-toolkit の設計で最も特徴的なのは、モデルごとに実装する `get_transformer_block_names()` という **1 つのメソッドが 3 つの境界を同時に決める**ことである。

```
get_transformer_block_names()  →  ["double_blocks", "single_blocks"]
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
  LoRA の標的選択            量子化の単位              学習対象の実質的な境界
  （このブロック内の         （ブロック単位で GPU へ    （外側は全精度のまま
   Linear だけ）              載せて量子化）             凍結される）
```

FLUX.2 では `["double_blocks", "single_blocks"]` が返るため、**モデル最上位にある共有変調（`double_stream_modulation_img.lin` など）・`img_in`・`txt_in`・`time_in`・`final_layer` はいずれも LoRA の対象外**になる。

これは [[questions/flux2-klein-lora-training-and-composition]] で「**複数 LoRA を合成する予定があるなら共有変調を標的から外せ**」と論じた推奨が、**既定で満たされている**ことを意味する。共有変調は局所性がゼロのまま全ブロックに効くため、2 つの LoRA が同時に触ると最悪の干渉源になる（[[summaries/2025-np-lora]] の命題 1）。ai-toolkit の既定はこの罠を構造的に回避している。

## 学習時の時刻分布は「モデルの事前学習と違ってよい」

[[concepts/noise-schedule]] は、学習時のノイズ分布が設計変数であること（SD3 の logit-normal、HiDream-O1 の SFT で一様へ切り替え、FLUX の解像度依存シフト）を追ってきた。ai-toolkit はこれを **`timestep_type` という 1 つの設定**に集約している。

| 値 | 対応する wiki の記述 |
| --- | --- |
| `sigmoid`（全体の既定） | 中央に厚い。SD3 の logit-normal と同じ動機 |
| `linear` | 一様。HiDream-O1 が SFT で採った方針 |
| `weighted` | **スケジュールは linear、損失側で重み付け** |
| `shift` / `flux_shift` | **推論の動的シフトに合わせる**（[[summaries/2025-flux2]] の `generalized_time_snr_shift` に相当） |
| `one_step` / `four_step` / `eight_step` | 少ステップ蒸留モデル向けに時刻を固定 |

**注目すべきは、FLUX.2 の既定が `weighted`（＝時刻は一様）であり、推論側の解像度依存シフトを学習では使わないこと**である。つまり**学習時の時刻分布はモデルの事前学習・推論の分布と一致していない**。しかも `weighted` が掛ける重みは **flex.1-alpha という別モデルで算出されたハードコード表**である。

理論側（[[concepts/noise-schedule]]）が「どの時刻に予算を配るかが品質を決める」と積み上げてきたのに対し、実務側は**モデルを跨いで使い回せる経験則**に落ち着いている、という対比になっている。どちらが正しいかを判定する材料は本 wiki にはない。

## 蒸留モデルを学習する — 「アダプタを −1 倍で当てる」

[[concepts/diffusion-distillation]] は、蒸留が多様性を狭めること・蒸留版の上に LoRA を積むと軌道と干渉しうることを記録してきた。ai-toolkit の実務的な答えは **assistant LoRA**（de-distillation adapter）である。

**蒸留の逆写像を学習した LoRA を、学習中に `multiplier = -1.0` で適用する。** これで蒸留の効果が打ち消され、勾配は「素のモデルのように振る舞うもの」に届く。理論的な枠組み（DMD など）とは独立した、純粋に運用上の回避策である。

用意されているのは **Z-Image（Turbo）・Krea2・Minimax-H3・Wan22・FLUX.1-schnell** で、**FLUX.2 には無い**。したがって FLUX.2 では base で学習するほかない——[[questions/flux2-klein-lora-training-and-composition]] の結論が、ツール側の事実としても裏づけられる形になっている。

## 複数 LoRA は「素朴な線形和」で合成される

[[concepts/lora-merging]] は、素朴な線形和が identity loss と signal interference で破綻すること、それを避けるために gradient fusion・学習係数マージ・零空間射影・信号ルーティングが積み上げられてきたことを追ってきた。

**ai-toolkit が提供するのは、そのうち最も素朴なものだけである。**

- 複数のネットワークを同時に適用すると、各モジュールが `forward` を patch して**連鎖**するので `out = base(x) + Δ₁(x) + Δ₂(x)` になる——**activation 空間での素朴な加算**。
- `merge_in` はベース重みに `ΔW = BA` を直接足し込む（順次適用すれば重み空間での加重和）。
- `extract` は学習済みモデルから **SVD で LoRA を抽出**する。
- **K-LoRA・NP-LoRA・SSR-Merge・ZipLoRA のような干渉を扱う手法は実装されていない。**

つまり**前景 LoRA と背景 LoRA を合成したい場合、ai-toolkit だけでは [[concepts/lora-merging]] が「破綻する」と記録している方法しか使えない**。干渉を扱う手法は別途実装するか、K-LoRA の公式 FLUX 実装のような外部の成果物を持ち込む必要がある。

## VRAM を削る手段と、その境界

[[concepts/large-scale-training-infrastructure]] は Wan の分散学習（2D Context Parallel・FP8・活性化オフロード）を「1 サンプルすら載らない規模」への応答として記録した。ai-toolkit が扱うのはその対極——**consumer GPU 1 枚で載せる**ための道具立てである。

| 手段 | 効き方 |
| --- | --- |
| 勾配チェックポイント（既定 **True**） | 活性化を捨てて再計算する |
| 量子化（quanto / torchao / Ostris 独自の 3 系統） | 重みのビット幅を落とす。`uint2`〜`uint8`・`nvfp4`・`convrot*` など |
| **ブロック単位の量子化** | 1 ブロックずつ GPU へ載せて量子化し CPU へ戻す。**ピーク VRAM が 1 ブロック分で済む** |
| レイヤオフロード | 層を CPU と GPU の間で出し入れする |
| パラメータスワップ | 一部のパラメータだけを活性にする（既定係数 0.1 ＝ 10%） |
| **ARA（Accuracy Recovery Adapter）** | 低ビット量子化の精度劣化を LoRA 状の補正で取り戻す |

**ARA が興味深い。** 量子化で失われた精度を「補正用の低ランクアダプタ」で埋め戻すという発想で、[[concepts/low-rank-adaptation]] の道具を**適応ではなく数値誤差の補償**に転用している。`qtype: "uint4|<adapter path>"` のように 1 つの設定値に量子化方式とアダプタを詰め込む。

なお **FLUX.2 用の既製 ARA は用意されていない**（wan22・qwen_image にはある）。

## 既存知識との接続

- [[concepts/low-rank-adaptation]]：LoRA を「どこに当てるか」という設計変数（B-LoRA が提起した論点）が、ツール側では `get_transformer_block_names()` と `only_if_contains` / `ignore_if_contains` として現れる。**transformer 系では `linear_alpha` が黙って無視される**という実装の癖もここに属する。
- [[concepts/lora-merging]]：ai-toolkit が実装するのは素朴な線形和と SVD 抽出のみ。理論側で積み上がった干渉対策との落差が大きい。
- [[concepts/noise-schedule]]：`timestep_type` が理論側の選択肢をほぼそのまま設定値にしている。ただし既定は**モデルの事前学習分布と一致しない**。
- [[concepts/diffusion-distillation]]：assistant LoRA（−1 倍）が、蒸留モデルを学習する実務的な回避策として存在する。
- [[concepts/large-scale-training-infrastructure]]：ブロック単位量子化・オフロード・パラメータスワップは、Wan の分散インフラとは逆方向（単一 consumer GPU）の最適化。
- [[concepts/flow-matching]]：`flowmatch` を `train.noise_scheduler` と `sample.sampler` の両方に指定し、**一致させる必要がある**（設定例に明記されている）。

## 参考文献（summaries）

- [[summaries/2026-ai-toolkit]] — ai-toolkit のコード読解（v0.12.26・コミット `8436c40`。34 arch、LoRA 標的選択の 3 段階、`timestep_type` の全選択肢、量子化バックエンド 3 系統、FLUX.2 の詳細と実装の癖）
- [[summaries/2025-flux2]] — FLUX.2 のリファレンス実装。ai-toolkit がベンダリングした dataclass が本家と一致し、構成値の**独立した裏付け**になっている
