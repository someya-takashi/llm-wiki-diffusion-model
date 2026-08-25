---
type: concept
aliases: [CFG, Classifier-Free Guidance, 分類器なしガイダンス, guidance]
tags: [classifier-free-guidance, text-to-image-generation, denoising-diffusion, generative-models, flux2]
related:
  - "[[classifier-guidance]]"
  - "[[text-to-image-generation]]"
  - "[[denoising-diffusion]]"
  - "[[diffusion-sampling]]"
  - "[[controllable-generation]]"
  - "[[reinforcement-learning-for-diffusion]]"
  - "[[inference-caching]]"
summaries:
  - "[[summaries/2022-classifier-free-guidance]]"
  - "[[summaries/2023-controlnet]]"
  - "[[summaries/2021-adm]]"
  - "[[summaries/2023-dit]]"
  - "[[summaries/2024-multi-lora-composition]]"
  - "[[summaries/2025-flow-matching-diffusion-intro]]"
  - "[[summaries/2025-wan]]"
  - "[[summaries/2025-z-image]]"
  - "[[summaries/2025-flux2]]"
updated: 2026-08-26
---

# Classifier-Free Guidance（分類器なしガイダンス, CFG）

**Classifier-Free Guidance（分類器なしガイダンス, CFG）** とは、条件付き拡散モデルの生成時に、**条件への忠実度（fidelity）と多様性（diversity）をトレードオフする**ための手法である。追加の分類器を一切使わず、同じネットワークが出す**条件付きスコアと無条件スコアの線形結合**だけで「条件をどれだけ強く効かせるか」を制御する。実装が極めて単純（学習・生成とも各 1 行）でありながら効果が大きいため、Stable Diffusion をはじめ現代のほぼ全ての [[text-to-image-generation]] 拡散モデルが標準採用している。ランドマーク手法は **CFG（Ho & Salimans 2022）** で、先行手法の [[classifier-guidance]]（分類器ガイダンス）を分類器なしで置き換えたものである。

## guidance とは何か——品質と多様性のトレードオフ

GAN（BigGAN の truncation）や Glow（低温サンプリング）には「サンプルの多様性を下げる代わりに 1 枚ごとの品質・もっともらしさを上げる」ノブがあった。拡散モデルでは、スコアを単純にスケールしたりノイズを減らしたりしても**ぼやけた低品質画像**になるだけで、このノブが素朴には作れない。**guidance** は拡散モデルにこのノブを与える技術の総称で、条件付き分布を「鋭く」して条件に強く従わせる。

- **[[classifier-guidance]]（分類器ガイダンス）**：別途学習した分類器の勾配を足す先行手法（ADM, Dhariwal & Nichol 2021・[[summaries/2021-adm]]）。拡散モデルが GAN を超える原動力になったが、分類器の追加学習が必要で、敵対的攻撃に似るという難点があった。CFG はこの分類器を排しつつ同じノブを実現し、ImageNet 128 では $w=0.3$ で ADM-G を上回ると報告した。
- **classifier-free guidance（CFG）**：分類器を排し、生成モデル自身の無条件スコアを使う。本ページの主題。

## 代表手法：Classifier-Free Guidance（Ho & Salimans 2022）

### 同時学習（1 行の変更）

条件付きモデル $\epsilon_\theta(z_\lambda,c)$ と無条件モデル $\epsilon_\theta(z_\lambda)$ を**同一のニューラルネット**で表す。無条件は条件入力に **null トークン $\varnothing$** を与えたもの $\epsilon_\theta(z_\lambda)=\epsilon_\theta(z_\lambda,c{=}\varnothing)$。学習中に確率 $p_\text{uncond}$（例 0.1〜0.2）で条件 $c$ をランダムに $\varnothing$ へ落とすだけで、両モデルをパラメータ増加なしに同時学習できる。

### ガイド付きサンプリング（1 行の変更）

生成時、2 つのスコア推定を線形外挿して混ぜる：

$$
\tilde\epsilon_\theta(z_\lambda,c)=(1+w)\,\epsilon_\theta(z_\lambda,c)-w\,\epsilon_\theta(z_\lambda)
$$

- $w=0$：通常の条件付き生成。
- $w>0$：条件付き予測を増幅し無条件予測を差し引く＝「条件方向」に強く押す。$w$ を上げるほど **FID は悪化するが IS は向上**（品質↑・多様性↓）。
- この $\tilde\epsilon$ は [[diffusion-sampling]] の任意のサンプラー（DDIM 等）にそのまま差し込める。実装によっては $s=w+1$ を「guidance scale」と呼ぶ。

### なぜ効くのか（暗黙の分類器）

CFG は**暗黙の分類器** $p^i(c|z)\propto p(z|c)/p(z)$ の勾配 $-\frac1{\sigma_\lambda}[\epsilon^*(z,c)-\epsilon^*(z)]$ に着想を得ている。直感的には「**無条件尤度を負のスコア項で下げ、条件付き尤度を上げる**」。ただしスコアは制約のないニューラルネットの出力（非保存的ベクトル場）なので、厳密にはどんな分類器の勾配でもなく、それゆえ「分類器を騙す敵対的攻撃で指標を上げているだけ」という [[classifier-guidance]] への疑念を回避する。これが本論文の重要な主張：純粋な生成モデルだけで分類器ベース指標を最大化できる。

### 成果と限界

- ImageNet 64/128（クラス条件付き）で FID↔IS の明確なトレードオフ。128px は当時 SOTA、$w=0.3$ で ADM-G を、$w=4$ で BigGAN-deep を上回る（[[summaries/2022-classifier-free-guidance]]）。
- **限界**：多様性が下がる。サンプリングで条件付き・無条件の **2 回 forward** が必要で遅い。

詳細・実験・式は [[summaries/2022-classifier-free-guidance]] を参照。

## 「2 倍のコスト」への 2 つの答え

CFG は条件付きと無条件の予測を**両方**計算するので、推論コストが構造的に 2 倍になる。これは長らく「CFG を使う以上仕方ないもの」として扱われてきたが、近年 2 方向から削られている。

- **ガイダンス蒸留**（[[diffusion-distillation]]）：2 回の評価を 1 回に畳み込む生徒モデルを**学習する**。FLUX.1 Kontext [dev] がこれで作られている（[[summaries/2025-flux-kontext]]）。
- **CFG キャッシュ**（[[inference-caching]]）：**学習せずに**同じ冗長性を突く。[[summaries/2025-wan]] の観察は「**サンプリングの後期では条件付きと無条件の出力が似てくる**」——序盤は何を描くかが決まっておらず条件の有無で出力が大きく違うが、終盤は構図も被写体も既に決まっていて両者とも同じ絵の細部を詰めているだけになる。ならば後期に無条件側を毎回計算する意味は薄い。そこで無条件側を数ステップに 1 回だけ計算し、間は条件付きの結果を再利用する（細部の劣化を防ぐため残差補償を併用）。

**両方を 1 つのリポジトリで見比べられる実例**が FLUX.2 である（[[summaries/2025-flux2]]）。同じアーキテクチャの 3 系統が並んでいる。

| 変種 | guidance の扱い | 1 ステップの評価回数 | 総 NFE |
| --- | --- | --- | --- |
| `flux.2-dev` | **蒸留**（スカラーを $\sin/\cos$ 埋め込みして時刻ベクトルに加算） | 1 | 50 |
| `flux.2-klein`（蒸留版） | **蒸留（しかも摘みがない）**——引数は渡されるがモデルが**無視する** | 1 | **4** |
| `flux.2-klein-base` | **真の CFG**（バッチを 2 倍に複製して 1 回の順伝播で cond/uncond 同時） | **2** | **100** |

読み取れることが 2 つある。第一に、**蒸留 guidance の実装は驚くほど軽い**——時刻埋め込みと同じ関数に通して同じ `vec` に足すだけである。第二に、**klein では guidance のスケールが完全に固定されている**。`use_guidance_embed=False` なので埋め込み層すら持たず、CLI も `guidance` 以外の値を拒否する。「品質と多様性のトレードオフを 1 つのノブで制御する」という CFG の原点にあった性質が、**蒸留の徹底とともに失われている**ことになる。$100 \to 4$ NFE（25 倍）の代償がここに現れている。

後者は本ページにとって、コスト削減以上の含意を持つ。**CFG の効きがサンプリングの段階によって変わる**——序盤の条件付けは構図を決め、終盤の条件付けはほとんど何もしていない——という観察は、guidance が何をしているのかの理解そのものに関わる。冒頭で述べた「品質と多様性のトレードオフ」も、実は全ステップで一様に生じているわけではないことを示唆する。

## 実装が持つ 2 つのつまみ — Z-Image（2026-08-26 追記）

CFG の実務は「スケール $w$ を 1 つ選ぶ」で済まないことが知られている。Z-Image のリファレンス実装（[[summaries/2025-z-image]]）には**報告書に一切書かれていない 2 つのつまみ**があり、いずれも本ページが扱ってきた論点に対応している。

**(1) `cfg_truncation` — 終盤で CFG を切る。**

```python
# src/zimage/pipeline.py L227–229
if t_norm > cfg_truncation:
    current_guidance_scale = 0.0
```

`t_norm` は Z-Image の内部時刻（$0$ がノイズ、$1$ がクリーン。向きが一般の流儀と逆な点は [[concepts/flow-matching]]）なので、この分岐は「**クリーンに近づいた後半のステップでは guidance を切る**」を意味する。既定値は $1.0$ で、$t_\text{norm}<1$ のため実質常時 CFG が有効。$1$ より小さい値を与えると後半が無誘導になる。

これは本ページが述べてきた「CFG は忠実度を上げる代わりに多様性と自然さを削る」という性質への、**時間軸方向の処方**である。高ノイズ域では構図をプロンプトに引き寄せたいが、低ノイズ域で強い誘導を掛け続けると色や質感が飽和する——という観察に対応する。学術的には guidance interval（誘導を掛ける時刻区間を限る）として知られる系統で、**それが既定の推論パイプラインに実装されている**のが実務的な意味を持つ。

**(2) `cfg_normalization` — guided 予測のノルムに上限を掛ける。**

```python
# 同 L263–268
pred = pos + w * (pos - neg)
if cfg_normalization:
    max_new_norm = ||pos|| * cfg_normalization
    if ||pred|| > max_new_norm:
        pred = pred * (max_new_norm / ||pred||)
```

$w$ を上げると $\text{pos}-\text{neg}$ が大きく足され、予測ベクトルのノルムが条件付き予測より膨らむ。**膨らみすぎを条件付き予測のノルムを基準にクリップする**のがこれで、過飽和・白飛びに対する古典的な処方（CFG rescale）と同じ発想である。リポジトリの `README.md` は使い分けまで指定している——**写実的な生成では `True`、画風的な生成では `False`**。

**(3) そして罠が 1 つ。**

```python
# 同 L105
do_classifier_free_guidance = guidance_scale > 1.0
```

**`0 < \text{guidance\_scale} \le 1.0` は黙って無視される。** $w=1.0$ は数式上「条件付き予測そのもの」なので無誘導と等価であり実害はないが、$w=0.5$ のような**負の方向へ弱く引く**設定を意図しても何も起きない。エラーも警告も出ない。

**変種ごとの既定値**も本ページの「2 倍のコスト」の議論に直結する。

| | ステップ | CFG | NFE |
| --- | --- | --- | --- |
| Z-Image（非蒸留） | 28–50 | 3.0–5.0 | **56–100** |
| Z-Image-Turbo | 8 | **0.0（無効）** | **8** |

Turbo は Decoupled DMD の CFG-Augmentation により CFG そのものを不要にしている（[[concepts/diffusion-distillation]]）ので、**ステップ数の削減とバッチ 2 倍の解消が同時に効いて NFE が 1 桁違う**。本ページが挙げた「2 倍のコストへの答え」の中で、**蒸留は guidance を機構ごと消す**という最も徹底した形である。

## 既存知識との接続

- [[classifier-guidance]]：CFG が分類器なしで置き換えた先行手法。両者は同じ「条件分布を鋭くする」効果を持つ。
- [[text-to-image-generation]]：CFG はテキスト条件付き生成の品質を劇的に高める事実上の標準。Stable Diffusion（[[latent-diffusion]]）も text2img で CFG を用いる。
- [[denoising-diffusion]]：CFG はノイズ予測スコア $\epsilon_\theta$ の上で動く。
- [[diffusion-sampling]]：CFG はガイドされたスコア $\tilde\epsilon$ を作るだけで、サンプリング手続き（DDIM 等）と直交して組み合わせられる。
- [[controllable-generation]]：ControlNet は CFG と併用され、条件画像を $\epsilon_c/\epsilon_{uc}$ にどう加えるかでガイダンスが過不足になる問題を、ブロック解像度に応じた重み $w_i=64/h_i$ で調整する **CFG 解像度重み付け（CFG-RW）** を提案した。
- [[diffusion-model-architecture]]：DiT も CFG で高品質化し（cfg=1.5 で ImageNet SOTA）、潜在の一部チャネルだけにガイダンスを当てる「部分チャネル CFG」も有効だと示した。CFG はバックボーン（U-Net でも Transformer でも）と直交して効く。
- [[multi-concept-customization]]：Multi-LoRA Composition の **LoRA Composite** は、CFG のスコア（$e_\theta(\emptyset)+s(e_\theta(c)-e_\theta(\emptyset))$）を複数 LoRA について計算し平均する形で、CFG を多 LoRA 合成に拡張した例。

## 参考文献（summaries）

- [[summaries/2025-flux2]] — FLUX.2（蒸留 guidance と真の CFG が同一リポジトリに併存。klein は guidance 引数を受け取るが無視し、スケールを変える手段がない。base は 50 ステップ × 2 = 100 NFE、蒸留版は 4 NFE）

- [[summaries/2025-wan]] — Wan（CFG キャッシュ。サンプリング後期では条件付きと無条件の出力が似てくることを利用し、学習なしで無条件側の計算を間引く）

- [[summaries/2022-classifier-free-guidance]] — Classifier-Free Diffusion Guidance（Ho & Salimans, NeurIPS 2021 Workshop）
- [[summaries/2023-controlnet]] — Adding Conditional Control to Text-to-Image Diffusion Models（CFG 解像度重み付け CFG-RW を提案）
- [[summaries/2023-dit]] — Scalable Diffusion Models with Transformers（DiT も CFG 使用、部分チャネル CFG）
- [[summaries/2024-multi-lora-composition]] — Multi-LoRA Composition（LoRA Composite が CFG を多 LoRA に拡張）
- [[summaries/2021-adm]] — Diffusion Models Beat GANs on Image Synthesis（先行手法 classifier guidance の主要原典）
- [[summaries/2025-flow-matching-diffusion-intro]] — Flow Matching と拡散モデル入門（MIT 6.S184 講義ノート。CFG を ũ=(1-w)u(·|∅)+w·u(·|y) として Bayes 則から導き、flow/拡散の両方に適用）
