---
type: summary
source_path: code_analysis/ai-toolkit/
source_kind: code
title: "ai-toolkit — ostris/ai-toolkit（拡散モデルの学習スイート）"
authors: [Ostris, LLC]
year: 2026
venue: "オープンソース実装（論文なし）・v0.12.26・コミット 8436c40"
tags: [ai-toolkit, low-rank-adaptation, lora-merging, noise-schedule, large-scale-training-infrastructure, diffusion-distillation, flux2, training-tooling]
ingested: 2026-08-26
---

# ai-toolkit：拡散モデルの学習スイートをコードから読む

> 原典: `code_analysis/ai-toolkit/`（v0.12.26・コミット `8436c40`）・[github.com/ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
> 作者: Ostris, LLC ・ ライセンス: **MIT**

> ⚠️ **本ページはコードを一次資料とする。** ai-toolkit に論文はなく、README は使い方の説明に留まる。以下の既定値・挙動はすべて実装から読み取り、ファイル名と行番号を添えてある。**個別の設定がなぜその値なのか（実験的根拠）はコードからは分からない。**

## 一言まとめ

**34 の arch を単一の設定ファイル形式で学習できる LoRA / 微調整のスイート。** 本 wiki が要約を持つモデル（FLUX.2・Z-Image・Qwen-Image・ERNIE-Image・HiDream・Wan・FLUX.1 Kontext）のほとんどが実際に学習可能な状態で登録されている。README の掲げる射程は「**consumer grade hardware で最新モデルを全部サポートする**」で、量子化・レイヤオフロード・パラメータスワップといった VRAM 削減の道具が一通り揃っている。

## 何ができるか

CLI（`python run.py config/foo.yml`）と Next.js の Web UI（`localhost:8675`）の両方から動く。設定は **YAML 駆動**で、Python SDK は実質ない（`toolkit/job.py` の `run_job` が薄い入口としてあるのみ）。

`job:` に指定できるのは 5 種類（`toolkit/job.py` L14–35）。

| `job:` | 中身 |
| --- | --- |
| `extension` | **実際の LoRA 学習はこれ**（`type: 'sd_trainer'`。UI は `ui_trainer`） |
| `train` | VAE・slider・ESRGAN・reference などの特殊学習 |
| `extract` | 学習済みモデルから LoRA / LoCon を **SVD 抽出** |
| `mod` | `rescale_lora`（LoRA の強度を書き換える） |
| `generate` | 画像生成のみ |

紛らわしい点として、**現代的な LoRA 学習の例はすべて `job: extension`** であり `job: train` ではない（`config/examples/train_lora_flux_24gb.yaml` L2, L6）。`MergeJob` は存在するが `get_job` から到達不能で `process_dict` が空、**事実上の死んだコード**である（`jobs/MergeJob.py` L6–7）。

## 対応モデル

`AI_TOOLKIT_MODELS`（`extensions_built_in/diffusion_models/__init__.py` L24–61）に 34 エントリ。解決は `ModelClass.arch == config.arch` の文字列一致（`toolkit/util/get_model.py` L46–47）。

**本 wiki が要約を持つモデルとの対応**（ここが本 wiki にとっての価値である）：

| wiki の要約 | ai-toolkit の `arch` | 備考 |
| --- | --- | --- |
| [[summaries/2025-flux2]] | `flux2` / `flux2_klein_4b` / `flux2_klein_9b` | **klein は base のみ登録**（後述） |
| [[summaries/2025-z-image]] | **`zimage`**（アンダースコアなし）/ `zimage_l2p` | UI に `zimage:turbo` があり **de-distillation アダプタ**付き |
| [[summaries/2025-qwen-image]] / [[summaries/2026-qwen-image-2]] | `qwen_image` / `qwen_image_edit` / `qwen_image_edit_plus` | |
| [[summaries/2026-ernie-image]] | `ernie_image` | |
| [[summaries/2025-hidream-i1]] / [[summaries/2026-hidream-o1-image]] | `hidream` / `hidream_e1` / **`hidream_o1`** | |
| [[summaries/2025-wan]] | `wan22_5b` / `wan22_14b` / `wan22_14b_i2v` | |
| [[summaries/2025-flux-kontext]] | `flux_kontext` | |

他に `chroma`・`ltx2`（`.3`/`.5`）・`omnigen2`・`krea2`・`minimax_h3`・`mageflow`・`ideogram4`・`anima`・`f-lite`・`prx_pixel`・`boogu_image`・`nucleus_image`・`zeta_chroma`、および音声の `ace_step_15`。

**注意すべき不整合**：`ModelConfig.arch` の型注釈 `ModelArch`（`toolkit/config_modules.py` L623）には上記の新しい arch が**含まれていない**。未知の文字列は `else: pass` で素通りする（L797–798）ので、型注釈は実質ヒントに過ぎない。

## LoRA 機構

### 何を標的にするか — 2 段階の絞り込み

**第 1 段階**はコンテナのクラス名で、モデル側が `self.target_lora_modules` で宣言する（`BaseSDTrainProcess.py` L1946–1947 で `target_lin_modules` として渡る）。FLUX.2 は `["Flux2"]`——**ルートのクラス名そのもの**なので、実質「モデル全体を走査せよ」を意味する。

**第 2 段階**は leaf のクラス名の許可リスト（`toolkit/lora_special.py` L29–40）。

```python
LINEAR_MODULES = ['Linear', 'LoRACompatibleLinear', 'QLinear', 'OstrisLinear']
CONV_MODULES  = ['Conv2d', 'LoRACompatibleConv', 'QConv2d']
```

**第 3 段階（実質的にこれが効く）** が `transformer_only` フィルタで、**既定は `True`**（`config_modules.py` L202）。モデルが `get_transformer_block_names()` を返すと、**名前にそのブロック名を含む leaf だけ**が残る（`lora_special.py` L531–541）。

### ユーザーが標的を絞る手段

`network.network_kwargs` 以下のキーで制御できる。

| キー | 効果 |
| --- | --- |
| `only_if_contains` | **許可リスト**。`clean_name`（ドット表記）**または** `lora_name` に部分一致 |
| `ignore_if_contains` | **除外**。**`clean_name` のみ**に部分一致 |
| `full_if_contains` | 該当層を LoRA でなく**フル微調整**（`FullModule`）にする |
| `all_layers` | Linear/Conv 以外の leaf も対象にする（既定 `False`） |

⚠️ **非対称が罠になる。** `ignore_if_contains` は `clean_name` しか見ないのに `only_if_contains` は両方見る（`lora_special.py` L524 対 L571–572）。**アンダースコア表記で除外パターンを書くと黙って何にも一致しない。** README は「`ignore_if_contains` が `only_if_contains` より優先される」とだけ述べる（L336–337）。

### network_type

| 設定値 | 実装 |
| --- | --- |
| `lora`（既定） | `LoRASpecialNetwork` / `LoRAModule` |
| `locon` / `lycoris` | `LycorisSpecialNetwork` |
| `lokr` | `LokrModule`。`lokr_full_rank` 既定 `True` で rank/alpha が 9999999999 に上書きされる |
| `lorm` | 蒸留的に元の層を置き換える（LoRA を足すのではない） |
| `dora` / `fullrank` | 実装はあるが `NetworkType` の Literal に**入っていない** |

**LoHa はない。**

### rank / alpha — FLUX.2 では alpha が効かない

既定は **rank 4・alpha 1.0**（`config_modules.py` L169–188）。スケールは `alpha / rank`。

**ところが transformer 系モデルでは `peft_format` が強制され、alpha が rank で上書きされる**（`lora_special.py` L423–433）。

```python
if self.peft_format:
    # no alpha for peft
    self.alpha = self.lora_dim
```

**FLUX.2・SD3・Lumina2・その他 `is_transformer` なモデルでは、設定した `linear_alpha` は黙って無視され、スケールは常に 1.0 になる。**

### 適用と保存

適用は **forward の monkey-patch**（`lora_special.py` L132–134）。元モジュールを Python の list に隠して state_dict から外す。順伝播は `out = org_forward(x) + multiplier * scale * up(down(x))` で、**ΔW は行列として実体化されない**。

保存形式は 2 系統。SD1/SDXL は kohya 形式（`lora_unet_*` / `.alpha`）、**transformer 系は diffusers/PEFT 形式**（`transformer.*.lora_A.weight`）。FLUX.2 はさらに保存時に `transformer.` → `diffusion_model.` へ書き換える（`flux2_model.py` L508–513）。

## 学習の設計

### 時刻分布 — 既定は sigmoid、FLUX.2 は weighted

`timestep_type` の既定は **`sigmoid`**（`config_modules.py` L556）で、`sigmoid(randn())` を `(1-t)*1000` に写して降順ソートする——**スケジュールの中央付近に厚く配る**。

受け付ける値と挙動（`toolkit/samplers/custom_flowmatch_sampler.py` L107–219）：

| 値 | 挙動 |
| --- | --- |
| `sigmoid`（既定） | 中央に厚い |
| `linear` | `linspace(1000, 1, N)` の一様 |
| **`weighted`** | **スケジュールは linear と同じ**。違いは**損失の重み付け**だけ |
| `shift` / `flux_shift` / `lumina2_shift` | **推論の動的シフトに合わせる**（解像度依存） |
| `lognorm_blend` | LogNormal 75% ＋ linear 25% |
| `one_step` / `two_step` / `four_step` / `eight_step` / `next_sample` | 少ステップ学習用に時刻を固定・限定 |

**FLUX.2 の UI 既定は `weighted`**（`options.tsx` L637 ほか）。つまり**時刻は 1000 個の線形グリッドから一様に引かれ**、解像度依存シフトは**かからない**。

### 損失

flow matching の速度回帰で、目標は `noise - latents`（`flux2_model.py` L497–500）。既定の損失は MSE（`loss_type` の既定 `'mse'`。他に `mae` / `wavelet` / `pseudo_huber` / `mean_flow` など）。

`weighted` のときに掛かる重みは、**1003 行のハードコード表**である（`toolkit/timestep_weighing/default_weighing_scheme.py`）。冒頭のコメントが出自を明かす。

> these weights were calculated using **flex.1-alpha**. A similar weighing scheme has been seen with other flowmatch models as well.

**別モデルで算出した重み表が、FLUX.2 を含む flow matching モデル全般に流用されている。**

## FLUX.2 の詳細

### 登録されているのは base だけ

| arch | 読むファイル | テキストエンコーダ |
| --- | --- | --- |
| `flux2` | `flux2-dev.safetensors` | Mistral-Small-3.1-24B |
| `flux2_klein_4b` | **`flux-2-klein-base-4b.safetensors`** | Qwen3-4B |
| `flux2_klein_9b` | **`flux-2-klein-base-9b.safetensors`** | Qwen3-8B |

**蒸留版 klein の arch は存在しない。** [[questions/flux2-klein-lora-training-and-composition]] で「base で学習すべき」と論じたが、**ai-toolkit ではそもそも選択の余地がない**というのが正確である。

### 実際の LoRA 標的は 80 モジュール（klein 4B）

`get_transformer_block_names()` が `["double_blocks", "single_blocks"]` を返すので（`flux2_model.py` L505–506）、標的は次だけになる。

| ブロック | 標的 | klein 4B での数 |
| --- | --- | --- |
| `DoubleStreamBlock` | `img_attn.qkv` / `img_attn.proj` / `img_mlp.0` / `img_mlp.2` ＋ `txt_*` の対 | 8 × **5** = 40 |
| `SingleStreamBlock` | `linear1` / `linear2` | 2 × **20** = 40 |

**共有変調の 4 行列は自動的に除外される。** FLUX.2 の `Modulation` はモデル最上位にあり（`double_stream_modulation_img.lin` など）、名前に `double_blocks`/`single_blocks` を含まないためフィルタで落ちる。同じ理由で `img_in` / `txt_in` / `time_in` / `guidance_in` / `final_layer` も対象外である。

これは [[questions/flux2-klein-lora-training-and-composition]] で「合成予定があるなら共有変調を標的から外せ」と述べた推奨が、**既定で満たされている**ことを意味する。

### guidance の扱い

学習時は**常に guidance が注入され、値は `train_config.cfg_scale`（既定 1.0）**（`SDTrainer.py` L1361–1363、`config_modules.py` L513）。

- **FLUX.2-dev** は `use_guidance_embed=True` なので、**guidance = 1.0 を焼き込んで学習する**（推論の既定は 4.0）。
- **klein** は `use_guidance_embed=False` なので、渡されても**モデル内部で無視される**。

### de-distillation アダプタは flux2 に無い

`assistant_lora`（アダプタを **multiplier = −1.0** で当てて蒸留を打ち消す仕組み）は **Z-Image・Krea2・Minimax-H3・Wan22・FLUX.1-schnell** には用意されているが、**flux2 には存在しない**。UI でも `zimage:turbo` だけが `assistant_lora_path` を露出する（`options.tsx` L674–677, L686）。

さらに実装自体が FLUX.1 専用で、`toolkit/assistant_lora.py` は冒頭で `if not sd.is_flux: raise ValueError(...)` と弾き、読むキーに `transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight` という **FLUX.1 の diffusers 命名を直書き**している。

### ベンダリングされたモデル定義は本家と一致する

ai-toolkit は FLUX.2 のモデル定義を自前で持つ（`flux2/src/model.py`）。**パラメータ dataclass は [[summaries/2025-flux2]] が本家から読み取った値と完全に一致する**（`Klein4BParams`: depth 5 / depth_single_blocks 20 / hidden 3072 / heads 24）。VAE も同一（`BatchNorm2d(affine=False)`・`ps=[2,2]`）。**独立した第 2 の実装による裏付け**になっている。

一方**削除されているものがある**——`forward_kv_extract` / `forward_kv_cached` / `causal_attn_fn` / `_blend_*_mods`。つまり **KV キャッシュ経路は学習側に存在しない**。参照トークンが自分自身にしか attend しない制約付き注意も、固定タイムステップ変調もかからず、**学習は素の全注意の下で行われる**。追加されているのは `enable_gradient_checkpointing` / `device` / `dtype` という学習用フックだけである。

## VRAM を削る手段

| 手段 | 設定キー | 既定 |
| --- | --- | --- |
| 勾配チェックポイント | `train.gradient_checkpointing` | **True** |
| 量子化（DiT） | `model.quantize` / `model.qtype` | False / `qfloat8` |
| 量子化（テキストエンコーダ） | `model.quantize_te` / `model.qtype_te` | `quantize` に追随 / `qfloat8` |
| レイヤオフロード | `model.layer_offloading` ＋ `*_percent` | False / 1.0 |
| 低 VRAM | `model.low_vram` | False |
| パラメータスワップ | `train.do_paramiter_swapping` ＋ `paramiter_swapping_factor` | False / 0.1 |
| テキスト埋め込みキャッシュ | `train.cache_text_embeddings` / `unload_text_encoder` | False |

量子化バックエンドは 3 系統が同居する——**optimum.quanto**（`qfloat8`・`qint8` など）、**torchao**（`int8`・`float8`）、**Ostris 独自**（`uint2`〜`uint8`・`orbit*`・`convrot*`・`nvfp4`）。

**量子化はブロック単位で行われる**（`quantize.py` L432–465）。`get_transformer_block_names()` が返すブロックを 1 つずつ GPU へ載せ、量子化して CPU へ戻すので**ピーク VRAM が 1 ブロック分で済む**。裏を返すと、**そのリストの外側（`img_in`・`time_in`・変調・`final_layer`）は量子化されず全精度のまま残る**——LoRA の標的選択とまったく同じ境界が使われている。

**Accuracy Recovery Adapter（ARA）** という仕組みもある。`qtype` に `|` を含めるとアダプタのパスとして解釈され（`config_modules.py` L741–743）、低ビット量子化で失われた精度を LoRA 状の補正で取り戻す。

```yaml
qtype: "uint4|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint4.safetensors"
```

**FLUX.2 用の既製 ARA は用意されていない**（wan22 や qwen_image にはある）。ARA と `assistant_lora_path` は排他である（L1489–1492）。

UI の FLUX.2 既定は `quantize: true` / `quantize_te: true` / `low_vram: true` / `qtype: 'qfloat8'`。

## 実装から読める設計判断

**(1) 「ブロック名」が 3 つの役割を兼ねている。** `get_transformer_block_names()` が返すリストは、**LoRA の標的選択**・**量子化の単位**・**（実質的な）学習対象の境界**の 3 つを同時に決めている。モデルごとにこの 1 メソッドを実装するだけで、トレーナ側の主要な挙動が揃う設計である。

**(2) 複数 LoRA は activation 空間で加算される。** 各モジュールが `org_module.forward` を patch して**連鎖**するので、2 つ目のネットワークは 1 つ目の forward を `org_forward` として掴む。結果は `out = base(x) + Δ₁(x) + Δ₂(x)`——**素朴な線形和そのもの**である（[[concepts/lora-merging]] が「破綻する」と記録している合成）。**NP-LoRA・K-LoRA・SSR-Merge のような干渉を扱う手法は実装されていない。**

**(3) 蒸留モデルへの対処が「アダプタを −1 倍で当てる」という単純な形に落ちている。** de-distillation は蒸留の逆写像を学習した LoRA を負の係数で適用するだけで、[[concepts/diffusion-distillation]] が扱ってきた DMD などの理論的な枠組みとは独立した、純粋に実務的な回避策である。

## コードからは分からないこと

- **既定値の根拠**。`timestep_type` がなぜモデルごとに違うのか、FLUX.2 でなぜ `weighted` なのか、`cfg_scale` がなぜ 1.0 なのか——実験も測定も残っていない。
- **flex.1-alpha 由来の重み表が他モデルで妥当かどうか。** コメントは「他の flowmatch モデルでも似た重み付けが見られる」と述べるだけで、検証はない。
- **学習の品質**。どの設定がどのモデルでどれだけ良いかは一切書かれていない。VRAM の目安（FAQ の「FLUX.1 は 24GB 最低」など）は散発的にあるが、品質の指標はない。

## 限界・批判的視点

- **`ignore_if_contains` の非対称**（`clean_name` のみ照合）は、除外を意図した設定が黙って無効化される罠である。エラーも警告も出ない。
- **transformer 系で `linear_alpha` が黙って無視される**（`peft_format` 強制）。設定ファイルに書けてしまうのに効果がないため、誤解を招く。
- **`ModelArch` の Literal が実際の arch を網羅していない**。型注釈が現実に追随しておらず、未知の文字列が素通りする。
- **`MergeJob` が到達不能な死んだコード**として残っている。
- **Mistral のテキストエンコーダ量子化が `qtype` を使う**（klein の Qwen3 は `qtype_te` を使う）。非対称で、意図的とは考えにくい。
- **`flux2_te_filename` という属性名が transformer のファイル名を指す**（"te" = text encoder のはずが）。無害だが紛らわしい。
- **テキストの attention mask が常に破棄される**（`pipeline.py` L140–146 ほか）。`padding="max_length"` で 512 に詰めるので、パディングトークンも注意の対象になる。本家 FLUX.2 も同じ実装なので**忠実な移植ではある**が、学習で意図した挙動かは不明。

## 用語と略称

- **LoRA** = Low-Rank Adaptation（[[concepts/low-rank-adaptation]]）。
- **LoKr / LoCon / LyCORIS** = LoRA の派生。LoKr は Kronecker 積、LoCon は畳み込み層にも当てる版。
- **DoRA** = Weight-Decomposed Low-Rank Adaptation。大きさと方向を分けて適応する。
- **ARA** = Accuracy Recovery Adapter。低ビット量子化の精度劣化を補正するアダプタ。
- **peft_format** = diffusers/PEFT 互換の重みキー命名（`lora_A` / `lora_B`）。
- **assistant LoRA** = 蒸留を打ち消すために負の係数で当てるアダプタ（de-distillation adapter）。
- **paramiter swapping**（原文ママ）= パラメータの一部だけを活性にして VRAM を削る仕組み。
- **flowmatch** = flow matching の学習・サンプリング（[[concepts/flow-matching]]）。`train.noise_scheduler` と `sample.sampler` の両方に指定し、一致させる必要がある。

## 関連ページ

- [[concepts/ai-toolkit]]
- [[concepts/low-rank-adaptation]]
- [[concepts/lora-merging]]
- [[concepts/noise-schedule]]
- [[concepts/diffusion-distillation]]
- [[concepts/large-scale-training-infrastructure]]
- [[summaries/2025-flux2]]
- [[summaries/2025-z-image]]
- [[questions/flux2-klein-lora-training-and-composition]]
