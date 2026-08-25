---
type: question
asked: 2026-08-26
question: "LoRA 学習は FLUX.2 [klein] 4B と 4B base のどちらで行うべきか。また前景 LoRA と背景 LoRA を組み合わせる手法として FLUX.2 に適したものは何か"
summaries_used:
  - "[[summaries/2025-flux2]]"
  - "[[summaries/2025-k-lora]]"
  - "[[summaries/2025-np-lora]]"
  - "[[summaries/2026-ssr-merge]]"
  - "[[summaries/2024-b-lora]]"
  - "[[summaries/2025-z-image]]"
  - "[[summaries/2024-orthogonal-adaptation]]"
---

# FLUX.2 [klein] 4B：LoRA を base で学習すべき理由と、前景×背景の合成手法

> 質問: LoRA 学習は `flux.2-klein-4b`（蒸留）と `flux.2-klein-base-4b`（ベース）のどちらで行うべきか。また前景 LoRA と背景 LoRA を組み合わせる手法として FLUX.2 に適したものは何か。
> 回答日: 2026-08-26
> 出典: [[summaries/2025-flux2]]（コード由来）＋ Web 調査（BFL 公式ブログ・コミュニティ報告）

## 結論

| 問い | 答え |
| --- | --- |
| どちらで学習するか | **`flux.2-klein-base-4b`（ベース）**。蒸留版は学習中の評価すらできない |
| 蒸留版で推論してよいか | **よい**（BFL 自身が推奨）。ただし同一性・姿勢の忠実度が落ちる報告があるので**両方で目視評価する** |
| 合成手法 | **まず「混ぜない」を検討**（FLUX.2 のネイティブ多参照）。混ぜるなら**推論ステップ数が手法を決める**——蒸留版なら **NP-LoRA / SSR-Merge**、50 ステップなら **K-LoRA** も可 |

## Q1：ベースで学習すべき理由

### 前提：2 つはアーキテクチャが同一

`flux.2-klein-4b` と `flux.2-klein-base-4b` は**どちらも `Klein4BParams()`** を使う（`src/flux2/util.py` L16–27, L53–64）。違うのは重み・既定値・`guidance_distilled` フラグだけである。

| | 蒸留版 | ベース |
| --- | --- | --- |
| steps | **4（固定）** | 50 |
| guidance | **1.0（固定）** | 4.0・**真の CFG** |
| `fixed_params` | `{"guidance", "num_steps"}` | `{}` |
| `use_guidance_embed` | False | False |
| 総 NFE | 4 | **100**（50 × CFG 2 回） |
| ライセンス | Apache-2.0 | Apache-2.0 |

### 蒸留版を避ける 3 つの理由

**1. 学習中の評価経路が塞がれている。** `fixed_params` に guidance と num_steps の両方が入っており、**CLI が他の値を拒否して `sys.exit(1)` する**（`cli.py` L212–238）。「50 ステップならどう見えるか」を確認できない。

**2. guidance の摘みが構造的に存在しない。** 蒸留版は `use_guidance_embed=False` なので、`denoise` が `guidance_vec` を作って渡しても **`forward` が完全に無視する**。埋め込み層自体を持たない。CFG を効かせた比較ができない。

**3. 蒸留は多様性を狭め、LoRA はさらに狭める。** [[concepts/diffusion-distillation]] の既知の限界で、Z-Image では OneIG Diversity がベース 0.194 → Turbo 0.139 まで落ちる（[[summaries/2025-z-image]]）。狭まった分布の上に LoRA を積むと過学習の兆候が早く出る。

### 運用：ベースで学習 → 蒸留版で推論

BFL 公式ブログが明言している。

> ベースのチェックポイントに対して学習すれば、アダプタはその後も蒸留モデルにロードできる。速いし、我々のテストではむしろ良い結果になることが多い。

**なぜ機械的にロードできるのか**はコードから説明がつく——両者は同じ `Klein4BParams()` なので、モジュール名も形状も完全に一致する。

⚠️ **ただしコミュニティからは「蒸留版で走らせると品質がやや落ちる。とくに同一性（identity）と姿勢の忠実度」という報告がある。** 前景が工業製品のような**同一性が命の剛体**なら、ここは直撃する。**ベースと蒸留版の両方で必ず目視評価すること。**

## Q2：前景×背景の合成

### まず「混ぜない」を検討する

**FLUX.2 は複数の参照画像をネイティブに扱える。** 参照は VAE 符号化されて系列に連結され、**4 軸 RoPE の第 1 軸を `t = 10·(i+1)` でずらして**分離される（`sampling.py` L74–80）。参照は対象と解像度が一致する必要すらない。画素予算は参照 1 枚なら 2024²、複数なら 1024²。

前回の回答（[[questions/lora-foreground-background-composition]]）では「背景生成 → AnyDoor で物体を置く」という逐次パイプラインを推奨したが、**FLUX.2 ではその外部モデル（[[summaries/2023-anydoor]]）が要らない**。前景 LoRA だけ学習し、背景は参照画像として渡す構成が最も素直である。**多参照が第一級の機能である以上、LoRA 合成は「参照画像で表現できないもの」に限って使うのが合理的**——例えば「特定の画風」は参照 1 枚では伝わりにくいので LoRA 向き、「この特定の背景」は参照画像向き、という切り分けになる。

### 混ぜる場合：推論ステップ数が手法を決める

**これが FLUX.2 固有の判断軸で、既存のどの論文にも書かれていない。**

| | 性質 | 4 ステップ蒸留版 | 50 ステップ Base |
| --- | --- | --- | --- |
| **NP-LoRA** | **静的マージ**（零空間へ射影して足す） | ✅ **問題なし** | ✅ |
| **SSR-Merge** | **静的マージ**（ルータを上方射影へ吸収） | ✅ **問題なし** | ✅ |
| **K-LoRA** | **実行時スケジューラ**（層ごと・ステップごとに選択） | ⚠️ **機構が粗くなる** | ✅ |

**K-LoRA は時間依存のスケール $S=\alpha t/T+\beta$（$\alpha=1.5,\beta=0.5$）で「初期は被写体・後期は画風」を実装する**（[[summaries/2025-k-lora]]）。原典の図 5 は 50 ステップで青（物体）と緑（スタイル）が滑らかに相互浸透する様子を示しており、**この機構は刻みが細かいほど効く**。4 ステップでは選択の機会が 4 回しかなく、本来の粒度で働かない。

対して **NP-LoRA と SSR-Merge はマージ済みの重みを 1 つ作る**ので、ステップ数に一切依存しない。**蒸留版にデプロイするなら、この 2 つが構造的に安全である。**

> **副次的な観察**：K-LoRA の実務上の弱点だった「生成が Direct の 2.6 倍遅い」（[[summaries/2025-np-lora]] 表 III の実測）は、FLUX.2 klein では 25 ブロック × 4 ステップ＝100 回の比較にしかならないためほとんど問題にならない。**速度ではなくスケジュールの粒度に論点が移る**——同じ手法でもバックボーンが変わると弱点の所在が変わる例である。

### 移植性は FLUX 系が最良

[[summaries/2025-np-lora]] は「**既存のアプローチのうち K-LoRA のみが FLUX の公式実装を提供している**」と明記し、自手法も FLUX で検証している。[[summaries/2026-ssr-merge]] の主要実験は FLUX.1-dev。FLUX.2 klein はその直系（同じ DoubleStream → SingleStream の二相構成）なので、公開ハイパーパラメータがそのまま通る見込みが最も高い（[[questions/lora-merging-base-model-selection]]）。

## FLUX.2 固有の学習ルール 2 つ

### ① 共有変調の 4 行列に LoRA を当てない

FLUX.2 の変調は**モデル全体で 1 回だけ**作られ、同じタプルが全 25 ブロック（klein 4B なら 5 double ＋ 20 single）へ配られる。DiT 全体で変調に使われる重みは 4 つしかない。

- `double_stream_modulation_img.lin`
- `double_stream_modulation_txt.lin`
- `single_stream_modulation.lin`
- `final_layer.adaLN_modulation.1`

**ここに 2 つの LoRA が同時に触ると、局所性がゼロのまま全ブロックで衝突する。** [[summaries/2025-np-lora]] の命題 1（加重和では干渉が原理的に消せない）が最悪の形で効く場所である。**合成する予定があるなら標的から明示的に外すこと。**

これは Z-Image の共有下方射影と同型の問題で（[[questions/z-image-architecture-and-lora-placement]]）、**変調機構を共有するアーキテクチャに共通する落とし穴**として一般化できる。Wan の adaLN 共有も同じ性質を持つはずである。

### ② モダリティ選択は使えるが、範囲が狭い

FLUX.2 には `DoubleStreamBlock` があり、`img_attn.*` / `img_mlp.*` に限定すれば**テキスト処理に触れずに画像側だけ**を動かせる。Z-Image（完全単一ストリーム）では原理的に不可能だった選択肢である。

**ただし klein 4B の double-stream は 5 ブロックだけで、残り 20 は single-stream の共有重みである。** 画像側限定にすると**モデルの 8 割に触れない**ことになる。

| | double / single | 画像側限定で触れる割合 |
| --- | --- | --- |
| FLUX.1 dev | 19 / 38 | 33% |
| FLUX.2 dev | 8 / 48 | 14% |
| **FLUX.2 klein 4B** | **5 / 20** | **20%** |

FLUX.2 は FLUX.1 よりパラメータを single-stream 側へ寄せているので、**モダリティ分離の余地はむしろ狭まっている**。背景の画風 LoRA には有効かもしれないが、前景の同一性を教えるには容量不足の可能性が高い。

## 推奨レシピ

```
学習:  flux.2-klein-base-4b（Apache-2.0、50 ステップ・真の CFG）
標的:  DoubleStreamBlock — img_attn.qkv / img_attn.proj / img_mlp.0 / img_mlp.2 ＋ txt_* の対
       SingleStreamBlock — linear1 / linear2
除外:  共有変調の 4 行列（合成する予定があるなら必須）
rank:  8（NP-LoRA / K-LoRA の検証設定に合わせる。rank を変えると
       NP-LoRA の主方向 V_k の意味が変わる）
統一:  前景・背景とも同じトレーナ・同じ rank・同じ標的で学習する
       （マージ手法は「同じ流儀で学習されている」ことを暗黙に仮定する）
合成:  蒸留版で推論する  → NP-LoRA（μ=0.5 から。同一性が崩れたら下げる）
       50 ステップで推論 → K-LoRA も選択肢に入る
評価:  content 類似度と style 類似度を単独で見ない。
       NP-LoRA の調和平均 S_harm を使う（[[concepts/style-content-disentanglement]]）
```

## この回答の限界

- **FLUX.2 のリポジトリに学習コードは一切ない**（オプティマイザも損失も PEFT もなく `load_state_dict(..., strict=True)`）。上の標的名は実装から読める**モジュール名**であって、BFL が推奨する標的セットではない。
- **「K-LoRA が 4 ステップで劣化する」は機構からの演繹であって実測ではない。** K-LoRA の時間依存スケールが少ステップで粗くなるのは定義から従うが、実際にどの程度品質が落ちるかは検証されていない。**試す価値のある仮説**である。
- **「ベース学習 → 蒸留版推論で同一性がやや落ちる」はコミュニティ報告**であり、査読された測定ではない。
- 本 wiki には **FLUX.2 で LoRA merging を実際に行った報告が存在しない**（公開文献にも見当たらない）。上の適用可能性は各手法が要求する構造からの演繹である。

## 情報源（Web）

- [FLUX.2 [klein]: Towards Interactive Visual Intelligence（BFL blog）](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence)
- [Fine-tune FLUX.2 [klein] with a LoRA under 60 minutes（HF blog）](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora)
- [FLUX.2 [klein] Training（BFL docs）](https://docs.bfl.ml/flux_2/flux2_klein_training)
- [FLUX.2 Klein 4B & 9B LoRA Training with Ostris AI Toolkit（RunComfy）](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-klein-lora-training)
- [24yearsold/flux2-klein-character-transfer-portable-r128（Hugging Face）](https://huggingface.co/24yearsold/flux2-klein-character-transfer-portable-r128)

## 関連ページ

- [[summaries/2025-flux2]]
- [[concepts/lora-merging]]
- [[concepts/style-content-disentanglement]]
- [[concepts/diffusion-distillation]]
- [[concepts/low-rank-adaptation]]
- [[questions/lora-merging-base-model-selection]]
- [[questions/lora-foreground-background-composition]]
- [[questions/z-image-architecture-and-lora-placement]]
