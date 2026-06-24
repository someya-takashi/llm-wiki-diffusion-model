---
type: concept
aliases: [CFG, Classifier-Free Guidance, 分類器なしガイダンス, guidance]
tags: [classifier-free-guidance, text-to-image-generation, denoising-diffusion, generative-models]
related:
  - "[[classifier-guidance]]"
  - "[[text-to-image-generation]]"
  - "[[denoising-diffusion]]"
  - "[[diffusion-sampling]]"
  - "[[controllable-generation]]"
summaries:
  - "[[summaries/2022-classifier-free-guidance]]"
  - "[[summaries/2023-controlnet]]"
  - "[[summaries/2021-adm]]"
  - "[[summaries/2023-dit]]"
  - "[[summaries/2024-multi-lora-composition]]"
  - "[[summaries/2025-flow-matching-diffusion-intro]]"
updated: 2026-06-25
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

## 既存知識との接続

- [[classifier-guidance]]：CFG が分類器なしで置き換えた先行手法。両者は同じ「条件分布を鋭くする」効果を持つ。
- [[text-to-image-generation]]：CFG はテキスト条件付き生成の品質を劇的に高める事実上の標準。Stable Diffusion（[[latent-diffusion]]）も text2img で CFG を用いる。
- [[denoising-diffusion]]：CFG はノイズ予測スコア $\epsilon_\theta$ の上で動く。
- [[diffusion-sampling]]：CFG はガイドされたスコア $\tilde\epsilon$ を作るだけで、サンプリング手続き（DDIM 等）と直交して組み合わせられる。
- [[controllable-generation]]：ControlNet は CFG と併用され、条件画像を $\epsilon_c/\epsilon_{uc}$ にどう加えるかでガイダンスが過不足になる問題を、ブロック解像度に応じた重み $w_i=64/h_i$ で調整する **CFG 解像度重み付け（CFG-RW）** を提案した。
- [[diffusion-model-architecture]]：DiT も CFG で高品質化し（cfg=1.5 で ImageNet SOTA）、潜在の一部チャネルだけにガイダンスを当てる「部分チャネル CFG」も有効だと示した。CFG はバックボーン（U-Net でも Transformer でも）と直交して効く。
- [[multi-concept-customization]]：Multi-LoRA Composition の **LoRA Composite** は、CFG のスコア（$e_\theta(\emptyset)+s(e_\theta(c)-e_\theta(\emptyset))$）を複数 LoRA について計算し平均する形で、CFG を多 LoRA 合成に拡張した例。

## 参考文献（summaries）

- [[summaries/2022-classifier-free-guidance]] — Classifier-Free Diffusion Guidance（Ho & Salimans, NeurIPS 2021 Workshop）
- [[summaries/2023-controlnet]] — Adding Conditional Control to Text-to-Image Diffusion Models（CFG 解像度重み付け CFG-RW を提案）
- [[summaries/2023-dit]] — Scalable Diffusion Models with Transformers（DiT も CFG 使用、部分チャネル CFG）
- [[summaries/2024-multi-lora-composition]] — Multi-LoRA Composition（LoRA Composite が CFG を多 LoRA に拡張）
- [[summaries/2021-adm]] — Diffusion Models Beat GANs on Image Synthesis（先行手法 classifier guidance の主要原典）
- [[summaries/2025-flow-matching-diffusion-intro]] — Flow Matching と拡散モデル入門（MIT 6.S184 講義ノート。CFG を ũ=(1-w)u(·|∅)+w·u(·|y) として Bayes 則から導き、flow/拡散の両方に適用）
