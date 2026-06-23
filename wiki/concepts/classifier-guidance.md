---
type: concept
aliases: [Classifier Guidance, 分類器ガイダンス, ADM guidance]
tags: [classifier-guidance, classifier-free-guidance, denoising-diffusion, generative-models]
related:
  - "[[classifier-free-guidance]]"
  - "[[denoising-diffusion]]"
  - "[[diffusion-sampling]]"
summaries:
  - "[[summaries/2022-classifier-free-guidance]]"
updated: 2026-06-23
---

# Classifier Guidance（分類器ガイダンス）

**Classifier Guidance（分類器ガイダンス）** とは、条件付き拡散モデルの生成時に、**別途学習した分類器の勾配**を拡散スコアに加えることで、条件への忠実度を高め（多様性を下げ）る手法である。Dhariwal & Nichol の **ADM（"Diffusion Models Beat GANs", 2021）** が導入し、拡散モデルが GAN を画像生成で上回るのを後押しした。後続の [[classifier-free-guidance]]（分類器なしガイダンス）が分類器を不要にして置き換えたため、現在は主に「CFG の先行手法・対比対象」として理解される。

このページは分類器ガイダンスの俯瞰と仕組みを扱う軽めの概念ページである。主要原典である ADM 論文（Dhariwal & Nichol 2021）はまだ本 wiki に未取り込みのため、ここでは [[summaries/2022-classifier-free-guidance]]（CFG 論文 §3.1）が整理した内容を起点に記述する。

## 仕組み

拡散モデルのスコア $\epsilon_\theta(z_\lambda,c)\approx-\sigma_\lambda\nabla_{z_\lambda}\log p(z_\lambda|c)$ に、ノイズ画像 $z_\lambda$ 上で学習した分類器 $p_\phi(c|z_\lambda)$ の対数尤度勾配を強度 $w$ で加える：

$$
\tilde\epsilon_\theta(z_\lambda,c)=\epsilon_\theta(z_\lambda,c)-w\,\sigma_\lambda\nabla_{z_\lambda}\log p_\phi(c|z_\lambda)\approx-\sigma_\lambda\nabla_{z_\lambda}\big[\log p(z_\lambda|c)+w\log p_\phi(c|z_\lambda)\big]
$$

これは分布 $\tilde p(z_\lambda|c)\propto p(z_\lambda|c)\,p_\phi(c|z_\lambda)^w$ からのサンプリングに相当する。$w$ を上げると、分類器が正しいラベルを高確信で割り当てるデータの確率が重く見積もられ、**IS↑・FID トレードオフ**（多様性↓）が得られる。GAN の truncation に対応する「ノブ」を拡散モデルに与えた点が貢献。なお、すでにクラス条件付きのモデルにガイダンスを適用するのが最も良い結果になることが報告されている。

## 難点（CFG が解決した点）

- **分類器の追加学習が必要**：しかも**ノイズの乗った $z_\lambda$** で学習しなければならず、既製の事前学習済み分類器を差し込めない。学習パイプラインが複雑化する。
- **敵対的攻撃への類似**：スコアに分類器勾配を混ぜる操作は、勾配ベースで分類器を騙す敵対的攻撃に似る。このため「FID/IS が上がるのは単に分類器ベース指標を敵対的に騙しているからでは？」という疑念を招く。

[[classifier-free-guidance]] はこれらを、分類器を排して生成モデル自身の条件付き／無条件スコアの差 $\epsilon_\theta(z,c)-\epsilon_\theta(z)$ で代替することで解決した。理論上、無条件モデルに重み $w{+}1$ の分類器ガイダンスを当てるのは条件付きモデルに重み $w$ を当てるのと等価、という関係が CFG の出発点になっている。

## 既存知識との接続

- [[classifier-free-guidance]]：分類器を使わず同じ効果を達成し、分類器ガイダンスを置き換えた後継。両者の対比が CFG 論文の主眼。
- [[denoising-diffusion]]：分類器ガイダンスは拡散スコア $\epsilon_\theta$ を修正して使う。
- [[diffusion-sampling]]：修正スコア $\tilde\epsilon$ をサンプラーに差し込んで生成する。

## 参考文献（summaries）

- [[summaries/2022-classifier-free-guidance]] — Classifier-Free Diffusion Guidance（§3.1 で classifier guidance を整理）

> 注: 分類器ガイダンスの主要原典 Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis"（ADM, 2021）は未取り込み。取り込み時に本ページを拡充する。
