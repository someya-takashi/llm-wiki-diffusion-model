---
type: summary
source_path: raw/papers/Multi-Concept Customization of Text-to-Image Diffusion.md
source_kind: paper
title: "Multi-Concept Customization of Text-to-Image Diffusion"
authors: [Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, Jun-Yan Zhu]
year: 2023
venue: CVPR 2023
ingested: 2026-06-25
tags: [multi-concept-customization, subject-driven-generation, low-rank-adaptation, latent-diffusion, generative-models]
translation: "[[translations/2023-custom-diffusion]]"
---

# Custom Diffusion — テキスト画像拡散モデルの多概念カスタマイズ

> 原典: [[translations/2023-custom-diffusion]] ・ `raw/papers/Multi-Concept Customization of Text-to-Image Diffusion.md`
> 著者・年・会議: Nupur Kumari ら（CMU・Tsinghua・Adobe Research）, arXiv:2212.04488, CVPR 2023
> プロジェクト: https://www.cs.cmu.edu/~custom-diffusion/

## 一言まとめ

text-to-image 拡散モデル（Stable Diffusion）の **cross-attention 層の key/value 射影行列 $W^k,W^v$ だけ**（全体の約 5%・保存 75MB）を数枚の画像で fine-tune して新概念を素早く（約 6 分）埋め込み、さらに**複数の新概念を 1 シーンに合成**できるようにする手法。複数概念は同時学習（joint）または各概念モデルを**閉形式の制約付き最小二乗で結合（約 2 秒）**して扱える。**「compositional multi-concept fine-tuning」というタスクと重みレベルの概念マージを最初に確立した源流**であり、[[multi-concept-customization]] の founding paper。personalization 三大原典（[[summaries/2023-dreambooth]] DreamBooth・[[summaries/2022-textual-inversion]] Textual Inversion・本論文）の一角。

## 背景と問題意識

大規模 text-to-image モデル（[[text-to-image-generation]]）は「ありとあらゆるもの」を生成できるが、ユーザー固有の概念（自分の犬・特定のソファ・稀なカテゴリ "moongate"）は学習時に見ていないため出せない。少数画像でモデルを**カスタマイズ**したい。先行する 2 つの同時期研究は両極だった——DreamBooth（[[summaries/2023-dreambooth]]）は**全パラメータ fine-tune**で重く（モデル丸ごと 3GB・約 1 時間）、Textual Inversion（[[summaries/2022-textual-inversion]]）は**埋め込みのみ学習**で軽いが表現力に欠ける。

本論文はその中間を突くと同時に、両者がほとんど扱えなかった**より難しい問題＝compositional fine-tuning（複数の新概念を 1 枚に合成）**を主目的に据えた。素朴な少数ショット fine-tune には 2 つの壁がある：**catastrophic forgetting（破滅的忘却, 新概念を学ぶと既存知識を失う）／language drift（言語ドリフト, "moongate" を学ぶと "moon"・"gate" の意味が崩れる）**と、数枚への**過学習**である。

## 提案手法 / 主張

### どの重みを更新すべきか — cross-attention の K,V だけ

全層を fine-tune したときの層ごと相対変化率 $\Delta_l=\|\theta_l'-\theta_l\|/\|\theta_l\|$ を測ると、**cross-attention 層（全パラメータの 5%）の変化が突出**する（図3）。cross-attention はテキスト特徴 $\mathbf c$ を $K=W^k\mathbf c,\ V=W^v\mathbf c$ に射影して画像側の query と結ぶ層で、**テキスト→画像の写像はこの $W^k,W^v$ にのみ入る**。そこで Custom Diffusion は $W^k,W^v$ **だけ**を更新する（図4）。これで新概念の獲得に十分で、保存は 75MB（モデルの約 3%）・2×A100 で約 6 分と、DreamBooth より 2–4× 速い。

<figure>

![](../../raw/assets/2023-custom-diffusion/x2.png)

<figcaption>図2（再掲）: Custom Diffusion の全体像。対象画像＋類似キャプションで取得した実画像（正則化）で学習データを作り、modifier token V* と cross-attention の key・value 射影だけを学習対象（trainable）に、残りは凍結（frozen）する。</figcaption>
</figure>

### modifier token V* と実画像正則化

一般カテゴリの固有インスタンス（自分の犬）は、**稀少トークンで初期化した新しい modifier 埋め込み $V^*$** を「$V^*$ dog」のように使い、$W^k,W^v$ と同時最適化する。language drift と過学習を防ぐため、**LAION-400M から CLIP テキスト類似度 >0.85 の実画像 200 枚**を正則化集合に使う（生成画像を使う DreamBooth 流より KID が良い＝忘却が少ない、図5・表3）。

### 複数概念の合成 — joint training と閉形式マージ

- **joint training**：複数概念のデータを混ぜ、概念ごとに別の $V^*_i$ を使って同時に学習。
- **closed-form constrained optimization merge**：各概念で別々に fine-tune 済みの $W^k,W^v$ を後から結合する。「正則化キャプションでは元モデルの出力を保ちつつ（最小二乗）、対象概念の単語は fine-tune 済み value に一致させる（制約 $WC^\top=V$）」という制約付き最小二乗を **Lagrange 乗数法で閉形式に解く**（約 2 秒、$\hat W=W_0+\mathbf v^\top\mathbf d$）。これは [[lora-merging]]（ZipLoRA・Mix-of-Show の gradient fusion）に先立つ、重みレベルの概念マージの先駆である。

## 実験結果と知見

- **単一概念**：DreamBooth と同等・Textual Inversion を上回る（text-/image-alignment が右上、図8・表1）。学習時間・保存量は大幅に小さい（75MB vs 3GB、約 5× 高速）。
- **複数概念合成**：joint・optimization の両方が全ベースラインを上回る（Table+Chair を除く）。DreamBooth は片方の概念を欠落させがち（図7）。順次学習は最初の概念を忘れる。**cross-attention 限定 fine-tune が合成成功の鍵**と結論。
- **人間選好調査**（各 800 応答）：単一・複数概念とも本手法が選好され、全層 fine-tune 版より good。
- **ablation**（表3）：augmentation（ランダムリサイズ＋サイズ語付与）で image-alignment↑、正則化なし／生成画像正則化は KID 悪化。
- **モデル圧縮**（Appendix C）：$W^k,W^v$ 差分行列の特異値が急減 → SVD 低ランク近似で 75MB→15MB（上位 60% rank, 5× 圧縮）でも品質維持。ただし fine-tune 中に低ランク更新を強制するのは suboptimal だった。
- 評価指標：CLIP image-/text-alignment、KID（Kernel Inception Distance, 生成と実画像の分布距離。FID の小標本でも安定な代替）、MS-COCO FID。

## 限界・批判的視点

- **似カテゴリの合成が苦手**（猫＋犬、teddybear＋tortoise plushy）。"cat"・"dog" の attention map が重なるためで、事前学習済みモデル由来の限界を継承する（図11・19）。
- **3 概念以上の合成も困難**。
- text-alignment と image-alignment は本質的にトレードオフ（学習が進むほど image↑・text↓、図20）。
- 後続：複数 LoRA を扱う [[multi-concept-customization]]（LoRA-Composer の注意制御、Multi-LoRA Composition の復号合成）や重みマージ系（[[lora-merging]] の ZipLoRA・Mix-of-Show）が、Custom Diffusion の「学習型・重みマージ」路線をさらに発展・代替していく。

## 既存 wiki との接続

Custom Diffusion は [[multi-concept-customization]] の**源流**——「複数の新概念を 1 枚に合成する」タスクと、その 2 解法（joint training／閉形式の重みマージ）を最初に定式化した。単一概念の側では、全層を学ぶ DreamBooth（[[summaries/2023-dreambooth]]）と埋め込みのみの Textual Inversion（[[summaries/2022-textual-inversion]]）の中間に位置する **cross-attention 限定 fine-tune** という personalization 手法であり（[[subject-driven-generation]]）、後の [[low-rank-adaptation]]（LoRA）が主流化する「軽量 fine-tune」の流れに連なる（実際 Appendix C で $W^k,W^v$ 差分を SVD で低ランク圧縮する）。基盤は Stable Diffusion（[[latent-diffusion]]）。

## 用語と略称

- **Custom Diffusion**：cross-attention の key/value 射影 $W^k,W^v$ だけを fine-tune して新概念を埋め込み、複数概念を合成する手法。
- **cross-attention（交差注意）**：画像側の query $Q=W^q\mathbf f$ と、テキスト側の key $K=W^k\mathbf c$・value $V=W^v\mathbf c$ を結ぶ注意層。テキスト条件はここからモデルに入る。
- **modifier token $V^*$**：新概念を表すために語彙へ加える擬似トークン。稀少トークンで初期化し最適化する。
- **compositional fine-tuning（合成的 fine-tuning）**：複数の新概念を 1 シーンに同時生成できるよう学習すること。
- **language drift（言語ドリフト）**：新概念を学ぶことで既存の単語の意味（視覚的対応）が崩れる現象。
- **catastrophic forgetting（破滅的忘却）**：新タスク学習で既存知識を失う現象。
- **regularization set（正則化集合）**：過学習・忘却を防ぐため学習に混ぜる、対象と類似カテゴリの実画像群。
- **closed-form constrained optimization merge**：複数概念の $W^k,W^v$ を制約付き最小二乗（Lagrange 乗数法）で閉形式に結合する重みマージ。
- **LDM** = Latent Diffusion Model（潜在拡散モデル。Stable Diffusion の基盤）。
- **DDPM** = Denoising Diffusion Probabilistic Models（ノイズ除去を繰り返す拡散モデルの基礎）。
- **KID** = Kernel Inception Distance（生成画像と実画像の分布距離。小標本でも安定）。
- **FID** = Fréchet Inception Distance（生成画像の品質指標）。
- **CLIP** = Contrastive Language-Image Pre-training（画像とテキストを共通空間に埋め込むモデル。alignment 指標に使用）。
- **P2P** = Prompt-to-Prompt（cross-attention を制御するテキストベース画像編集手法）。

## 関連ページ

- [[concepts/multi-concept-customization]]
- [[concepts/subject-driven-generation]]
- [[concepts/lora-merging]]
- [[concepts/low-rank-adaptation]]
- [[concepts/latent-diffusion]]
- [[summaries/2023-dreambooth]]
- [[summaries/2022-textual-inversion]]
- [[translations/2023-custom-diffusion]]
