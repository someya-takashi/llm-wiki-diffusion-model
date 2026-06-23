---
type: summary
source_path: raw/papers/Classifier-Free Diffusion Guidance.md
source_kind: paper
title: "Classifier-Free Diffusion Guidance"
authors: [Jonathan Ho, Tim Salimans]
year: 2022
venue: NeurIPS 2021 Workshop on Deep Generative Models
ingested: 2026-06-23
tags: [classifier-free-guidance, classifier-guidance, text-to-image-generation, denoising-diffusion, generative-models, cfg]
translation: "[[translations/2022-classifier-free-guidance]]"
---

# Classifier-Free Diffusion Guidance（CFG）

> 原典: [[translations/2022-classifier-free-guidance]] ・ `raw/papers/Classifier-Free Diffusion Guidance.md`（arXiv:2207.12598）
> 著者・年・会議: Jonathan Ho, Tim Salimans（Google Research, Brain team）・2022・NeurIPS 2021 Workshop on Deep Generative Models

## 一言まとめ

条件付き拡散モデルの「条件への忠実度」を、**追加の分類器を一切使わずに**強められる手法。学習時に条件をランダムに落として**条件付きモデルと無条件モデルを 1 つのネットで同時に学習**し、生成時に両者のスコアを $\tilde\epsilon=(1+w)\epsilon_\theta(z,c)-w\epsilon_\theta(z)$ と混ぜるだけ。実装は各 1 行という極端な単純さで、品質↑・多様性↓のトレードオフを自在に制御できる。Stable Diffusion をはじめ現代のほぼ全ての text-to-image 拡散モデルが採用する標準技術（[[classifier-free-guidance]]）。

## 背景と問題意識

GAN（BigGAN の truncation）や Glow（低温サンプリング）には「多様性を下げて 1 枚ごとの品質を上げる」ノブがあるが、拡散モデルでスコアを単純にスケールしたりノイズを減らしたりしても**ぼやけた低品質画像**になるだけで効かない。

そこで Dhariwal & Nichol（ADM, 2021）は **分類器ガイダンス（classifier guidance）** を提案した（[[classifier-guidance]]）。これは拡散スコアに、別途学習した分類器の勾配 $\nabla_z\log p(c|z)$ を $w$ 倍して足す手法で、IS↑/FID のトレードオフを得る。しかし欠点がある：(1) **ノイズ画像で別途分類器を学習**せねばならず学習パイプラインが複雑（既製の分類器は使えない）、(2) スコアに分類器勾配を混ぜる操作は**分類器への敵対的攻撃**に似ており、「FID/IS が上がるのは単に分類器を騙しているからでは？」という疑念を生む。

本論文の問い：「分類器なしで guidance はできるか？」

## 提案手法 / 主張

### 同時学習（1 行）

条件付きモデル $\epsilon_\theta(z,c)$ と無条件モデル $\epsilon_\theta(z)$ を**同一ネットワーク**で表す。無条件モデルは条件 $c$ に **null トークン $\varnothing$** を入れたもの $\epsilon_\theta(z)=\epsilon_\theta(z,c{=}\varnothing)$。学習中、確率 $p_\text{uncond}$ で $c$ を $\varnothing$ に落とすだけで両者を同時に学習できる（パラメータ増加なし）。

### ガイド付きサンプリング（1 行）

生成時、2 つのスコアを線形外挿して混ぜる：

$$
\tilde\epsilon_\theta(z_\lambda,c)=(1+w)\,\epsilon_\theta(z_\lambda,c)-w\,\epsilon_\theta(z_\lambda)
$$

$w=0$ は通常の条件付き生成、$w$ を上げるほど条件方向に強く押し（条件付きを増幅し無条件を差し引く）、品質↑・多様性↓。この $\tilde\epsilon$（CFG スケール $w$、実装によっては $s=w{+}1$ と表記）は DDIM など任意のサンプラーに差し込める（[[diffusion-sampling]]）。

### なぜ効くのか

CFG は **暗黙の分類器（implicit classifier）** $p^i(c|z)\propto p(z|c)/p(z)$ の勾配 $-\frac1{\sigma}[\epsilon^*(z,c)-\epsilon^*(z)]$ に着想を得ている。直感的には「**無条件尤度を負のスコア項で下げ、条件付き尤度を上げる**」操作。ただしスコアは制約のないニューラルネットの出力で非保存的ベクトル場なので、これは厳密にはどんな分類器の勾配でもなく、**分類器への敵対的攻撃とは解釈できない**——これが本論文の重要な主張で、「純粋な生成モデルだけで分類器ベース指標を最大化できる」ことを示す。

## 実験結果と知見

- **ImageNet 64×64 / 128×128（クラス条件付き）** で $w$ を 0〜4 まで掃引。FID は $w$ とともに単調悪化、IS は単調向上する明確なトレードオフ（表1・2、図4・5）。
- **最良 FID は小さい $w$（0.1〜0.3）**、最良 IS は大きい $w$（≥4）。128px は当時の **SOTA** で、$w=0.3$ で分類器ガイドの ADM-G を FID で上回り、$w=4.0$ で BigGAN-deep を FID・IS 両方で上回る。
- **$p_\text{uncond}$**：$\{0.1,0.2\}$ がよく、$0.5$ は一貫して劣る。無条件タスクにはモデル容量のごく一部を割けば十分。
- 強くガイドすると**彩度が上がる**傾向（図3）。

## 限界・批判的視点

- **多様性の低下**：品質↑の代償。過小評価されたデータの表現が失われうる社会的懸念も指摘。
- **サンプリングが 2 倍遅い**：各ステップで条件付き・無条件の 2 回 forward が必要（分類器ガイダンスは小さい分類器 1 回で済むため、速度では不利な場合も）。後年、条件付けをネット後段に注入する等の緩和策が研究された。
- **概念実証の位置づけ**：ADM 用に調整したハイパラを流用しており、CFG 専用最適化ではない。
- ベイズ反転で良い分類器が得られる保証はない（モデル誤指定下では理論保証を失う）が、経験的には機能する。

## 用語と略称

- **CFG** = Classifier-Free Guidance（分類器なしガイダンス、本論文の手法）
- **classifier guidance** = 分類器ガイダンス（[[classifier-guidance]]、ADM の先行手法）
- **$w$ / guidance scale** = ガイダンス強度（大きいほど条件忠実度↑・多様性↓。実装によっては $s=w+1$）
- **$p_\text{uncond}$** = 学習中に条件を $\varnothing$ に落とす確率
- **null トークン $\varnothing$** = 無条件を表す特別な条件入力
- **implicit classifier** = 暗黙の分類器（生成モデルからベイズで導く $p(z|c)/p(z)$）
- **IS** = Inception Score（高いほど良い）・**FID** = Fréchet Inception Distance（低いほど良い）
- **truncation / low temperature sampling** = 切り詰め／低温サンプリング（GAN・Glow で品質↔多様性を制御するノブ）
- **log SNR (λ)** = 対数信号対雑音比（連続時間拡散の時間変数）

## 関連ページ

- [[concepts/classifier-free-guidance]] — CFG の位置づけと仕組み（本論文の中核）
- [[concepts/classifier-guidance]] — 先行手法の分類器ガイダンス（対比）
- [[concepts/text-to-image-generation]] — CFG が事実上の標準となった応用（Stable Diffusion 等）
- [[concepts/denoising-diffusion]] — CFG が乗る拡散モデルの基礎
- [[concepts/diffusion-sampling]] — CFG はサンプラー（DDIM 等）と組み合わせて使う
