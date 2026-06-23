---
type: summary
source_path: raw/papers/DreamBooth_ Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation.md
source_kind: paper
title: "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation"
authors: [Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, Kfir Aberman]
year: 2023
venue: CVPR 2023
ingested: 2026-06-24
tags: [subject-driven-generation, text-to-image-generation, latent-diffusion, denoising-diffusion, generative-models]
translation: "[[translations/2023-dreambooth]]"
---

# DreamBooth: Subject-Driven Generation

> 原典: [[translations/2023-dreambooth]] ・ `raw/papers/DreamBooth_ Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation.md`
> 著者・年・会議: Ruiz, Li, Jampani, Pritch, Rubinstein, Aberman（Google）・2023・CVPR 2023（arXiv 2208.12242）

## 一言まとめ

**特定の被写体（自分の犬・バッグなど）の写真をわずか 3〜5 枚与え、事前学習済みの text-to-image 拡散モデル（Imagen / Stable Diffusion）全体を fine-tune して、その被写体を一意識別子「[V] [class noun]」に紐づける**手法。一度埋め込めば、テキストプロンプトで被写体を任意の文脈・姿勢・視点・スタイルに再生成できる（personalization / subject-driven generation）。鍵は **rare-token 識別子**と、language drift を防ぐ **class-specific prior preservation loss（PPL）**。

## 背景と問題意識

Imagen・DALL-E 2・Stable Diffusion のような大規模 **T2I（text-to-image, テキストからの画像生成, [[text-to-image-generation]]）** モデルは、テキストから高品質・多様な画像を生成できる。しかし「**この特定の被写体**を、見た目の同一性（identity）を保ったまま別の文脈で描く」ことができない。どんなに詳細にテキストで書いても（「白い文字盤に黄色い数字 3 のレトロな黄色い目覚まし時計」）、出力ドメインの表現力が足りず、毎回違う見た目になる（図2）。被写体の参照画像を「画像」として渡す DALL-E 2 の image-guided 生成でも、内容のバリエーションを作るだけで同一性は保てない。

DreamBooth はこれを「モデルの personalization（個人化）」として解く。すなわち、モデルの**言語・視覚辞書を拡張**し、新しい語（一意識別子）を特定の被写体に結びつける。同時期の **Textual Inversion**（Gal ら, 凍結モデルの token 埋め込みだけを学習）が比較対象で、両者は [[subject-driven-generation]] の二大ランドマーク。

## 提案手法 / 主張

### (1) rare-token 識別子「a [V] [class noun]」（§3.2）

入力画像すべてを「a [identifier] [class noun]」（例: 「a [V] dog」）でラベル付けする。

- **class noun（クラス名, 例: dog）**：モデルが既に持つそのクラスの強力な視覚的 prior に被写体を「繋ぐ」。誤ったクラス名（バッグに「can」）や class noun 無しは、対立・収束困難・性能低下を招く（表4）。
- **[V]（一意識別子）**：英単語（「unique」等）やランダム文字列は強い prior を持ち不適。語彙中の**希少トークン（rare token）**を逆引きした短い列（k=1〜3）を使い、言語・拡散モデルの両方で prior が弱い識別子にする。

### (2) class-specific prior preservation loss（PPL, §3.3）— 本手法の肝

全層を fine-tune すると 2 つの副作用が出る：
- **language drift（言語ドリフト）**：「dog」がこの特定の犬に崩れ、他の犬を生成できなくなる（言語モデルで知られる現象を拡散モデルで初めて指摘）。
- **多様性の低下**：少数ショットの姿勢・視点にスナップして overfitting する。

対策が PPL。**fine-tune 前の凍結モデル**に「a [class noun]」を与えて ancestral sampler でクラスサンプル $x_\text{pr}$ を約 1000 枚自己生成し、それを教師にする第 2 項を損失に足す（モデルが自分の prior を「自分で復習」する）：

$$
\mathbb{E}\big[w_t\|\hat{x}_\theta(\alpha_t x+\sigma_t\epsilon,\,c)-x\|^2_2 + \lambda\,w_{t'}\|\hat{x}_\theta(\alpha_{t'}x_\text{pr}+\sigma_{t'}\epsilon',\,c_\text{pr})-x_\text{pr}\|^2_2\big]
$$

第 1 項が被写体の再構成（「[V] dog」）、第 2 項が prior 保存（「a dog」の多様性維持）。λ=1、約 1000 iter、1 GPU で約 5 分。これで過学習せず長く学習でき、多様性と language drift を同時に解決する。

<figure>

![](../../raw/assets/2023-dreambooth/x3.png)

<figcaption>図3（再掲, [[translations/2023-dreambooth]] より）: fine-tune の構成。上の経路が「[V] dog」での再構成損失、下の経路が「a dog」での class-specific prior preservation loss（重みは共有）。</figcaption>
</figure>

### 新しい評価指標

被写体駆動生成の新タスク用に、データセット（30 被写体・25 プロンプト・3000 枚評価）と指標を提案。

- **DINO**（自己教師 ViT-S/16 特徴の類似度）：同クラスの異なる被写体の細部まで区別するので、被写体忠実度の主指標。人手選好との相関も CLIP-I より高い。
- **CLIP-I**（CLIP 画像埋め込み類似度）：被写体忠実度の従来指標だが、似た被写体を区別しにくい。
- **CLIP-T**（プロンプト・画像の CLIP 類似度）：プロンプト忠実度。

## 実験結果と知見

- **vs Textual Inversion**：DINO/CLIP-I/CLIP-T すべてで大差で上回る（表1）。ユーザー調査でも被写体忠実度 68% vs 22%、プロンプト忠実度 81% vs 12%（表2）。
- **DreamBooth(Imagen) > DreamBooth(SD) > Textual Inversion**：Imagen の表現力・出力品質が高く、実画像の被写体忠実度の上限に迫る。
- **PPL アブレーション**（表3）：PPL で PRES（prior 崩壊指標）が 0.664→0.493 に下がり、DIV（多様性）が上がる（被写体忠実度はわずかに低下）。
- **応用**：recontextualization（新文脈）、art renditions（画家風・新規ポーズ）、novel view synthesis（4 枚正面画像から背面等を外挿）、property modification（色・素材・種の掛け合わせ）、accessorization、expression manipulation、漫画生成。
- **入力枚数**（表5/6）：一般的な被写体（Corgi）は 1〜2 枚でも、稀な物体（バックパック）は 3 枚以上必要。最適は 4 枚 → 3〜5 枚に落ち着く。
- **超解像（[[super-resolution]]）**：Imagen の cascaded SR を fine-tune する際、noise augmentation を $10^{-3}\to10^{-5}$ に下げると被写体の高周波細部が保たれる（図21）。

## 限界・批判的視点

- **失敗モード**（図9）：稀な文脈の生成失敗、context-appearance entanglement（文脈が被写体の見た目に漏れる、例: 背景で時計の数字の色が変わる）、訓練画像への過学習。
- **被写体差**：犬・猫は学習しやすいが稀な被写体は変化を支えにくく、hallucination も起こる。
- **コスト**：被写体ごとに全層 fine-tune（数分・モデル丸ごと複製）。後続の **LoRA**（低ランク更新）や **Custom Diffusion**（cross-attention のみ更新）はこの重さを軽減する方向（本 wiki 未取り込み）。Textual Inversion は逆に超軽量だが表現力で劣る。
- 悪用（なりすまし）への社会的懸念。

## 用語と略称

- **PPL** = (class-specific) Prior Preservation Loss（クラス固有事前保存損失）。モデル自身が生成したクラスサンプルで prior を保つ正則化。
- **language drift（言語ドリフト）** = fine-tune でクラス名（「dog」）が特定インスタンスに崩れ、汎用語の意味を失う現象。
- **subject-driven generation / personalization** = 少数画像で特定の被写体にモデルを適応させ、その被写体を新文脈で再生成するタスク。
- **rare-token 識別子** = 弱い prior を持つ希少トークンで被写体を表す一意識別子「[V]」。
- **DINO** = 自己教師あり ViT（Caron ら）。本論文では被写体忠実度の特徴抽出器。**CLIP-I / CLIP-T** = CLIP 埋め込みによる画像・テキスト類似度。**LPIPS** = 知覚的距離（多様性 DIV 計算に使用）。
- **T2I** = text-to-image。**SR** = super-resolution（超解像）。**cascaded diffusion** = 低解像度生成→SR を段階接続する構成（Imagen）。
- **Textual Inversion** = 凍結 T2I モデルの埋め込み空間に新トークンを学習する比較手法（Gal ら）。

## 既存知識との接続

- [[subject-driven-generation]]：DreamBooth は本概念の代表。全層 fine-tune で被写体をモデル出力ドメインに埋め込む（Textual Inversion の token 埋め込みより表現力が高い）。
- [[text-to-image-generation]]：DreamBooth は Imagen / Stable Diffusion といった T2I 拡散モデルの下流 personalization。Imagen（T5-XXL＋cascaded diffusion）と SD（[[latent-diffusion]]）の両方で動く。
- [[latent-diffusion]]：Stable Diffusion を personalize する場合、U-Net（と任意でテキストエンコーダ）を学習しデコーダは固定。
- [[denoising-diffusion]]：PPL は標準の拡散二乗誤差損失（$\hat{x}$ 予測）の上に prior 保存項を足したもの。
- [[super-resolution]]：Imagen の cascaded SR を fine-tune＋低ノイズ増強で被写体細部を保つ。
- [[diffusion-sampling]]：PPL のクラスサンプル生成に ancestral sampler を使う。

## 関連ページ

- [[concepts/subject-driven-generation]] — 被写体駆動生成 / personalization（DreamBooth が代表）
- [[concepts/text-to-image-generation]] — DreamBooth が personalize する T2I 拡散モデル
- [[concepts/latent-diffusion]] / [[concepts/super-resolution]] / [[concepts/denoising-diffusion]]
- [[summaries/2022-latent-diffusion]] — Stable Diffusion（DreamBooth の適用先の 1 つ）
