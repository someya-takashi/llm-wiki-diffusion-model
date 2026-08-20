---
type: question
asked: 2026-08-21
question: "LoRA merging の実験用ベースモデルとして Z-Image / ERNIE-Image / FLUX.2 [klein] 4B のどれを選ぶべきか。LoRA 学習可能性・学習の容易性・lora-merging の各手法の適用可能性の 3 観点で評価"
summaries_used:
  - "[[summaries/2025-z-image]]"
  - "[[summaries/2026-ernie-image]]"
  - "[[summaries/2025-k-lora]]"
  - "[[summaries/2025-np-lora]]"
  - "[[summaries/2026-ssr-merge]]"
  - "[[summaries/2024-orthogonal-adaptation]]"
  - "[[summaries/2024-b-lora]]"
  - "[[summaries/2024-ziplora]]"
  - "[[summaries/2023-custom-diffusion]]"
  - "[[summaries/2023-mix-of-show]]"
---

# LoRA merging 実験のベースモデル選定：Z-Image / ERNIE-Image / FLUX.2 [klein] 4B

> 質問: LoRA merging の実験を行う拡散モデルの選定。モデルサイズとライセンスの観点から候補は Z-Image / ERNIE-Image / FLUX.2 [klein] 4B。(1) LoRA 学習可能か (2) LoRA 学習の容易性 (3) [[concepts/lora-merging]] の手法が適用可能か、で評価してほしい。
> 回答日: 2026-08-21
> 情報源: 本 wiki（[[summaries/2025-z-image]]・[[summaries/2026-ernie-image]] ほか）＋ Web 調査（各モデルカード・公式ドキュメント・エコシステム）

## 結論

**FLUX.2 [klein] 4B Base を第一推奨とする。** 理由は 1 つに集約できる——

> **[[concepts/lora-merging]] の主要手法（K-LoRA・NP-LoRA・SSR-Merge）はいずれも FLUX 系で検証済みであり、klein はその直系である。** 公開されたハイパーパラメータと実装がそのまま通る見込みが最も高い。

| 順位 | モデル | 一言 |
| --- | --- | --- |
| **1** | **FLUX.2 [klein] 4B Base** | 最小・公式の学習サポートあり・**主要手法の検証済み系列に最も近い** |
| **2** | **Z-Image（Base）** | アーキテクチャが完全公開・完全均質な 30 層は**統制実験の土俵として優れる** |
| **3** | ERNIE-Image | **アーキテクチャ非公開**が LoRA merging 研究では致命的に効く |

## 基礎データ

| | **Z-Image** | **ERNIE-Image** | **FLUX.2 [klein] 4B** |
| --- | --- | --- | --- |
| ライセンス | Apache 2.0 | Apache 2.0 | **Apache 2.0**（4B のみ。9B は非商用） |
| DiT パラメータ | 6.15B | 8B | **4B** |
| アーキテクチャ | **S3-DiT 完全単一ストリーム 30 層** | 単一ストリーム DiT（**詳細非公開**） | **8 DoubleStream ＋ 24 SingleStream** |
| 隠れ次元 | **3840**（ヘッド 32・FFN 中間 10240） | 非公開 | 非公開 |
| テキストエンコーダ | **Qwen3-4B** | Ministral-3（3B） | **Qwen3-4B**（層 9/18/27 を連結、7680 次元） |
| VAE | Flux VAE（流用） | FLUX.2 VAE（流用） | FLUX.2 VAE |
| 非蒸留ベース版 | ✅ `Tongyi-MAI/Z-Image` | ✅ ERNIE-Image（50 steps・CFG 4.0） | ✅ `FLUX.2-klein-base-4B` |
| 蒸留版 | Z-Image-Turbo（**model card に「fine-tune 不可」と明記**） | ERNIE-Image-Turbo（8 steps・CFG 1.0） | klein（step 蒸留版） |
| 公式 LoRA 学習ドキュメント | ✗ | ✗ | **✅ BFL 公式 docs ＋ HF 公式ブログ** |
| diffusers | ✅ `ZImagePipeline` | ✅ `ErnieImagePipeline` | ✅ |
| ai-toolkit | ✅（Turbo は de-distill アダプタ経由） | ✅ | ✅ `arch: "flux2_klein_4b"` |
| OneTrainer | — | ✅（**8GB VRAM**・int8） | — |
| LoRA 学習の目安 | 16GB〜 | 8GB〜（int8） | **<24GB / 4090 で 1 時間弱** |

> **偶然だが有用な一致**: Z-Image と FLUX.2 klein 4B は**同じ Qwen3-4B をテキストエンコーダに使う**（kohya/musubi-tuner の議論でも同じ `qwen_3_4b.safetensors` が参照されている）。2 モデルを比較する実験を組むとき、テキストエンコーダを統制変数にできる。

## 観点 1：LoRA 学習可能か

**3 モデルとも可能。ただし「どのチェックポイントで」が重要。**

| | 判定 | 補足 |
| --- | --- | --- |
| Z-Image | ✅ | **Base を使うこと。** model card は Z-Image を「LoRA 学習・ControlNet・意味的条件付けの良いベース」と明記する一方、**Turbo は fine-tune 不可**と比較表で示している。コミュニティは Turbo に de-distillation アダプタを噛ませて学習しているが、**merging の研究では余計な交絡になる** |
| ERNIE-Image | ✅ | ai-toolkit・OneTrainer・diffusers の DreamBooth LoRA が対応。**公式の学習スクリプトはなく**、README が外部ツールを参照する形 |
| FLUX.2 klein 4B | ✅✅ | **Base が「undistilled, 完全な学習信号を保持」と明記され、fine-tuning / LoRA 学習が推奨用途として挙げられている。** BFL 公式ドキュメントと HF 公式ブログの両方に手順がある |

**蒸留版を避けるべき理由**は本 wiki に根拠がある。[[concepts/diffusion-distillation]] の既知の限界として、Z-Image では OneIG の Diversity が**ベース 0.194 → Turbo 0.139** まで落ちる（[[summaries/2025-z-image]]、原典はこの悪化を議論していない）。蒸留で作られた少ステップの軌道の上に LoRA を積むと、**マージの結果が「干渉のせいか蒸留のせいか」を分離できなくなる**。

## 観点 2：LoRA 学習の容易性

### FLUX.2 klein 4B — ★★★

- **最小（4B）** → 1 本あたりの学習が最も速い。**これは merging 実験で決定的**（後述）
- **公式ドキュメントがある唯一のモデル**。ai-toolkit の設定は `arch: "flux2_klein_4b"` / `lr: 1e-4` / `flowmatch`、1800 ステップで 4090 で 1 時間弱
- ~13 GB（bf16）の重み、LoRA 学習は 24GB 未満に収まる
- Civitai・Replicate にもレシピがあり、**コミュニティ LoRA が集まりやすい**

### ERNIE-Image — ★★☆

- **OneTrainer の int8 で 8GB VRAM** という報告があり、絶対的な敷居は最も低い
- 一方で **8B と最大**なので、同じ VRAM 予算での学習速度は最も遅い
- 公式スクリプトがなく、設定の正解が community 頼み

### Z-Image — ★★☆

- 6.15B、Base なら 16GB クラスから
- **公式の学習スクリプトがない**。ai-toolkit のエコシステムは **Turbo ＋ de-distillation アダプタ**を中心に育っており、Base での学習は相対的に情報が薄い
- 一方 **アーキテクチャが原典の表 2 で完全公開**（30 層・3840・32 ヘッド・FFN 10240）されており、自前でトレーナを書く場合は最も見通しが良い

> **Z-Image 固有の落とし穴**（詳細は [[questions/z-image-architecture-and-lora-placement]]）: Z-Image は SFT 段階で**全入力プロンプトをプロンプトエンハンサに通し、拡散側を PE の出力分布に合わせている**（[[concepts/prompt-enhancement]]）。学習キャプションを短いユーザー風の文で書くと分布がずれる。また多様性が元から低い（OneIG Diversity 0.194、SD 1.5 は 0.429）ので**過学習が早い**。

## 観点 3：[[concepts/lora-merging]] の手法の適用可能性

**これが選定の本丸である。** 各手法が何を要求するかで、アーキテクチャ依存性が大きく分かれる。

| 手法 | 必要なもの | Z-Image | ERNIE-Image | FLUX.2 klein 4B |
| --- | --- | --- | --- | --- |
| **(0) Custom Diffusion 閉形式マージ** | cross-attention の $W^k, W^v$ | ✗ | ✗（推定） | **△** DoubleStream のテキスト側射影が最も近い対応物 |
| **(1) 素朴な線形和** | $\Delta W$ のみ | ✅ | ✅ | ✅ |
| **(2) Mix-of-Show gradient fusion** | $\Delta W$ ＋ 順伝播で $X_i$ を収集 | ✅ | ✅ | ✅ |
| ↳ **ED-LoRA** | 層ごとテキスト埋め込みの学習 | △ 要トレーナ改造 | △ | △ |
| **(3) ZipLoRA** | content/style LoRA 対、列ごと係数学習 | ✅ | ✅ | ✅ |
| **(4) LoRAHub** | 係数の勾配フリー最適化 | ✅ | ✅ | ✅ |
| **(5) Orthogonal Adaptation** | **学習プロセスの改造**（$B$ 凍結＋共有直交基底） | **△ 構造的に好都合**（後述） | △ 次元が不明で基底を作れない | △ 好都合 |
| **(6) NP-LoRA** | $\Delta W=BA$、$A_s^\top$ の QR 分解 | ✅ | ✅ | **✅✅ FLUX で検証済み** |
| **(7) SSR-Merge** | $A/B$ 因子、rank 連結、較正の順伝播 | ✅ | ✅ | **✅✅ FLUX.1-dev で検証済み** |
| **K-LoRA** | 注意層ごとの Top-K 比較（生成ループにフック） | ✅ | ✅ | **✅✅ FLUX の公式実装あり** |
| **B-LoRA** | SDXL の UNet ブロック 4/5 | ✗ | ✗ | ✗（ただし後述の利点あり） |
| **LoRA-Composer** | U-Net cross-attn への領域別注入 | ✗ | ✗ | △ |
| **Multi-LoRA Composition** | 復号時の切替／平均 | ✅ | ✅ | ✅ |

### 決定的な差 1：主要 3 手法が FLUX で検証済み

[[summaries/2025-np-lora]] は明記している——**「既存のアプローチのうち K-LoRA のみが FLUX の公式実装を提供しており、参照として含める」**、そして自手法についても「射影ベースの設計が異なる拡散バックボーンにわたってよく汎化する」と FLUX での検証結果を示す。[[summaries/2026-ssr-merge]] の主要実験は **FLUX.1-dev** である。

FLUX.2 klein は FLUX.1 の直系（同じ DoubleStream / SingleStream の二相構成）なので、**論文のハイパーパラメータ・実装・失敗モードがそのまま参照できる可能性が最も高い**。研究の立ち上げコストの差はここで最も大きく開く。

### 決定的な差 2：単一ストリームでは失われる選択肢

[[summaries/2023-custom-diffusion]] の「cross-attention の $K/V$ だけをマージする」という発想は、**cross-attention が存在して初めて成立する**。

- **Z-Image / ERNIE-Image**：完全な単一ストリーム。連結系列への self-attention だけなので、$W_k, W_v$ はテキストも画像も同じ重みで処理する。**テキスト側だけを扱う手法が原理的に使えない**
- **FLUX.2 klein**：8 個の DoubleStreamBlock でテキストと画像が**別重み**を持つ。(0) の対応物が存在し、さらに**「画像ストリームだけに LoRA を当てる」という選択肢**も残る

### 決定的な差 3：B-LoRA は 3 モデルとも使えないが、klein には探索の起点がある

[[concepts/style-content-disentanglement]] に記録したとおり、**B-LoRA の「ブロック 4 がコンテンツ、5 が色」は SDXL の UNet（非対称なダウン／ミドル／アップ構造）に固有の経験則**で、均質な Transformer 積層では未検証である。3 モデルとも DiT なので直接は使えない。

ただし **FLUX.2 klein には「8 DoubleStream ＋ 24 SingleStream」という既存の役割分化**があり、B-LoRA 流のプロンプト注入解析をかける際の**自然な仮説**になる。Z-Image は 30 層が完全に同型なので仮説がゼロから始まる。

**これは見方によっては Z-Image の利点でもある**——ブロック種別という交絡がないので、「層の位置そのものがマージ結果に影響するか」を**統制された形で測れる**唯一の候補になる。

### Orthogonal Adaptation について：DiT はむしろ好都合

[[summaries/2024-orthogonal-adaptation]] の要約で**最大の弱点として挙げた**のは、**厳密に直交できる概念数の上限 $\lfloor n/r \rfloor$** だった。SD v1.5 は最小の入力次元が 320 なので $r=20$ なら**わずか 16 概念**である。

**広い DiT ではこの制約が桁で緩む。**

| | 最小の重み次元 | $r=20$ での上限 $\lfloor n/r \rfloor$ | 共有直交行列の個数 |
| --- | --- | --- | --- |
| SD v1.5 | 320 | **16** | 4（320/640/768/1280） |
| **Z-Image** | **3840** | **192** | **2**（3840・10240） |
| FLUX.2 klein 4B | 非公開（2048〜3072 と推定） | 100〜150 | 少数 |

Z-Image は隠れ次元が全層で 3840 に統一されており、**共有直交基底として保存すべき正方行列が実質 2 つで済む**（SD1.5 は 4 つ）。実装は SD より単純になる。**要約で「スケーラビリティ主張の中で最も検証が薄い」と批判した点が、現代の DiT では実際にはあまり効かない**——これは本 wiki の記述を更新する価値のある発見である。

一方 **ERNIE-Image は次元が非公開なので、共有直交基底そのものを設計できない**（重みを読めば分かるが、論文からは分からない）。

## ERNIE-Image を 3 位にする理由

サイズとライセンスだけ見れば悪くない。しかし **LoRA merging の研究**という目的では、**アーキテクチャ非公開**が想像以上に効く。

[[summaries/2026-ernie-image]] の批判として記録したとおり——

> **アーキテクチャの記述が極端に薄い**。「8B の単一ストリーム DiT」以外に、層数・隠れ次元・注意機構・位置符号化・正規化の記述が**一切ない**。Z-Image が表 2 で構成を公開したのと対照的。

merging の研究では次が全部必要になる。

- **どの行列に LoRA を当てたか**を報告する（再現性）
- **NP-LoRA**：$V_k$ を作る主方向の数を rank と対応づける
- **SSR-Merge**：rank 連結後の部分空間次元 $Kr$ と、$N \gg Kr$ が成り立つかの見積もり
- **Orthogonal Adaptation**：$\lfloor n/r \rfloor$ の見積もりと共有基底の設計

重みファイルを読めば実測はできるが、**「公開された仕様に基づいて実験を設計する」ことができない**。加えて 8B と最大で反復が遅く、公式スクリプトもない。**三重に不利**である。

## 実験設計の推奨

### なぜ「1 本あたりの学習コスト」が効くのか

手法によって**必要な LoRA の本数が桁で違う**。

| 手法 | 必要な LoRA 数 |
| --- | --- |
| ZipLoRA・K-LoRA・NP-LoRA・B-LoRA | **2 本**（content × style） |
| Mix-of-Show・Orthogonal Adaptation | 3〜12 本 |
| **SSR-Merge** | **最大 21 本**（原典は 10 被写体プールから $K \in \{3,5,7,9\}$、付録 C で 21 まで） |

SSR-Merge の追試や、[[summaries/2024-orthogonal-adaptation]] の「概念数を増やしたときの劣化」を測る実験を視野に入れるなら、**4B で 1 本 1 時間**と **8B でそれ以上**の差は積み上がって効く。

### 推奨する進め方

```
Phase 1 — 再現（FLUX.2 klein 4B Base）
  ai-toolkit で 2 本（被写体 × 画風）を学習
  → K-LoRA / NP-LoRA / 素朴な線形和 を比較
  → 論文の数値傾向（content↔style の鏡像）が再現するか確認

Phase 2 — スケール（FLUX.2 klein 4B Base）
  DreamBooth データセット等から 10 本前後を学習
  → SSR-Merge / TIES / DARE / Task Arithmetic を K=3,5,7,9 で比較
  → 疎化系が成功率を落とす現象（0.69・0.62 対 0.76）が再現するか

Phase 3 — アーキテクチャ間の一般化（Z-Image Base を追加）
  同じ LoRA 学習設定・同じ被写体で Z-Image に移植
  → 「完全単一ストリームで結論が変わるか」を測る
  → テキストエンコーダが同じ Qwen3-4B なので統制しやすい
```

**Phase 3 が学術的に最も価値がある。** 本 wiki が繰り返し記録してきた未解決の問い——**マージ手法の知見はアーキテクチャをまたいで転移するのか**——に、統制された答えを出せる。[[summaries/2026-ssr-merge]] は FLUX.1 で 90〜98%、Qwen-Image で 97〜99% の回復率という**7 ポイント近い差を報告しながら「特徴空間の特性が違う」としか説明していない**（要約に批判として記録済み）。二相構成（klein）対 完全単一ストリーム（Z-Image）という対比は、この空白に切り込める。

## 落とし穴と注意点

**1. すべて Base で学習する。** Z-Image Turbo は model card が明示的に fine-tune 不可としている。ERNIE-Image-Turbo・klein の蒸留版も同様に避ける。

**2. トレーナを揃える。** マージ手法は「LoRA が同じ流儀で学習されている」ことを暗黙に仮定する。3 モデルとも ai-toolkit が使えるので、**ai-toolkit で統一**するのが最も統制が取りやすい。

**3. rank を揃える。** [[summaries/2025-np-lora]] は「K-LoRA の慣例に従い全 LoRA を rank 8 で学習」しており、**上位 8 個の特異ベクトルをすべて主方向とする**。rank を変えると $V_k$ の意味が変わる。再現を狙うなら rank 8 から始める。

**4. コミュニティ LoRA を混ぜるなら K-LoRA の $\gamma$ を思い出す。** 出所の違う LoRA は要素の絶対値の桁が違い、そのままでは Top-K 比較が機能しない。K-LoRA が全層の絶対値和の比 $\gamma$ で正規化するのはこのためである（[[summaries/2025-k-lora]]）。自前学習で揃えるなら不要。

**5. 評価指標を単独で見ない。** content 類似度と style 類似度は**トレードオフ曲線上の座標**であって、片方だけで順位を付ける意味はない。[[summaries/2025-np-lora]] の**調和平均 $S_\text{harm}$** を使う（詳細は [[concepts/style-content-disentanglement]] の「評価の落とし穴」）。

**6. リリース重みがモデルマージの産物である場合がある。** Z-Image は SFT の仕上げに能力次元ごとに偏らせた変種を線形補間している（[[concepts/model-merging]]）。$\Delta W$ の構造が単一の学習軌道の産物ではない点は、頭の片隅に置いておく価値がある。

## この回答の限界

- **本 wiki は FLUX.2 [klein] の原典を持っていない。** klein に関する記述はモデルカード・公式ブログ・コミュニティドキュメントに基づく Web 調査であり、査読された技術報告に基づくものではない。klein 4B の隠れ次元・注意ヘッド数は公開されていない。
- **3 モデルのいずれについても、LoRA merging を実際に行った実験報告は本 wiki にも公開文献にも見当たらない。** 上記の適用可能性は各手法が要求する構造（cross-attention の有無、$\Delta W=BA$ の形、較正順伝播の可否）からの**演繹**である。
- ERNIE-Image の「cross-attention なし」は「単一ストリーム DiT」という記述からの**推定**であり、原典に明示されていない。実装を読めば確定する。

## 情報源（Web）

- [black-forest-labs/FLUX.2-klein-base-4B（Hugging Face）](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
- [FLUX.2 [klein]: Towards Interactive Visual Intelligence（BFL blog）](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence)
- [Fine-tune FLUX.2 [klein] with a LoRA under 60 minutes（HF blog）](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora)
- [FLUX.2 [klein] Training（BFL docs）](https://docs.bfl.ml/flux_2/flux2_klein_training)
- [Text Encoders | black-forest-labs/flux2（DeepWiki）](https://deepwiki.com/black-forest-labs/flux2/3.2-text-encoders)
- [Diffusers welcomes FLUX-2](https://huggingface.co/blog/flux-2)
- [Tongyi-MAI/Z-Image（Hugging Face）](https://huggingface.co/Tongyi-MAI/Z-Image)
- [Tongyi-MAI/Z-Image（GitHub）](https://github.com/Tongyi-MAI/Z-Image)
- [baidu/ERNIE-Image（GitHub）](https://github.com/baidu/ernie-image)
- [ERNIE-Image supported by OneTrainer（Issue #8）](https://github.com/baidu/ERNIE-Image/issues/8)
- [FLUX 2 Dev, FLUX Klein and Z Image training text encoder question（musubi-tuner #886）](https://github.com/kohya-ss/musubi-tuner/issues/886)

## 関連ページ

- [[concepts/lora-merging]]
- [[concepts/model-merging]]
- [[concepts/style-content-disentanglement]]
- [[concepts/low-rank-adaptation]]
- [[concepts/diffusion-model-architecture]]
- [[concepts/diffusion-distillation]]
- [[questions/z-image-architecture-and-lora-placement]]
- [[questions/lora-foreground-background-composition]]
