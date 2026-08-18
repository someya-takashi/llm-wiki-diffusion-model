---
type: concept
aliases: [LoRA, Low-Rank Adaptation, 低ランク適応, PEFT, parameter-efficient fine-tuning]
tags: [low-rank-adaptation, subject-driven-generation, diffusion-model-architecture, generative-models, style-content-disentanglement]
related:
  - "[[subject-driven-generation]]"
  - "[[latent-diffusion]]"
  - "[[diffusion-model-architecture]]"
  - "[[multi-concept-customization]]"
  - "[[lora-merging]]"
  - "[[style-content-disentanglement]]"
summaries:
  - "[[summaries/2022-lora]]"
  - "[[summaries/2023-mix-of-show]]"
  - "[[summaries/2024-ziplora]]"
  - "[[summaries/2024-orthogonal-adaptation]]"
  - "[[summaries/2024-b-lora]]"
updated: 2026-08-19
---

# Low-Rank Adaptation（LoRA, 低ランク適応）

**Low-Rank Adaptation（LoRA, 低ランク適応）** とは、大規模な事前学習済みモデルの重みを**凍結したまま、各重み行列への更新を小さな低ランク行列の積 $\Delta W = BA$ として学習する** parameter-efficient fine-tuning（PEFT, 効率的微調整）手法である。更新部分だけを学習するため学習パラメータが桁違いに少なく（元モデルの 0.01% 程度）、推論時には $W=W_0+BA$ とマージできるので**追加のレイテンシがゼロ**。元は LLM のための手法（Hu ら 2022, [[summaries/2022-lora]]）だが、拡散モデルでは **Stable Diffusion / SDXL（[[summaries/2023-sdxl]]）の軽量 personalization の事実上の標準**となり、コミュニティが無数のキャラクター・画風 LoRA を共有する基盤になった。

## 仕組み

事前学習済み重み $W_0\in\mathbb R^{d\times k}$ を凍結し、適応による変化だけを低ランク分解で表す：

$$
h = W_0 x + \Delta W x = W_0 x + B A x,\qquad B\in\mathbb R^{d\times r},\ A\in\mathbb R^{r\times k},\ r\ll\min(d,k)
$$

- 学習するのは $A,B$ のみ（$W_0$ は固定）。$A$ はガウス初期化・$B$ はゼロ初期化なので、**開始時 $\Delta W=0$** で事前学習モデルから滑らかに出発する。$\Delta Wx$ は $\alpha/r$ でスケールする。
- **着想**：事前学習モデルは低い内在次元を持ち、適応時の重み変化 $\Delta W$ も低い「内在階数」を持つ。実際 $r=1$ でも有効なことがある（[[summaries/2022-lora]] §7）。$\Delta W$ は元の重み $W$ で強調されていない「タスク固有方向」を大きく増幅する。
- **プラグ&プレイ**：1 つの凍結ベースモデルを共有し、タスク／概念ごとに小さな LoRA（数 MB）を出し替えるだけで切り替えられる。これが拡散コミュニティでの爆発的普及につながった。

## 拡散モデルでの使われ方

LLM で生まれた LoRA は、拡散モデルでは **U-Net（とテキストエンコーダ）の注意・線形層**に当てて、特定の被写体・画風・キャラクターを少数画像で学習する用途で定着した。

- **personalization（[[subject-driven-generation]]）**：DreamBooth が**全層 fine-tune**でモデル全体を複製・更新するのに対し、LoRA は低ランク更新だけを学ぶので**軽量・可搬**。表現力は全層 fine-tune にやや劣るが、数 MB のファイルで共有でき、Stable Diffusion 派生エコシステムの中心になった。Textual Inversion（[[summaries/2022-textual-inversion]]、擬似単語の埋め込みのみ学習）と DreamBooth（全層）の中間の表現力／コストに位置する。
- **派生**：HyperDreamBooth（LoRA 重みを予測）、**ED-LoRA**（[[summaries/2023-mix-of-show]]、概念トークンを layer-wise＋multi-word に分解し identity を embedding 側に残す）など。
- **重みの疎性**：ZipLoRA（[[summaries/2024-ziplora]]）は LoRA の $\Delta W$ が**疎**で、要素の 90% を 0 にしても品質が保たれることを観察した。これが複数 LoRA を干渉なくマージできる根拠になる。
- **低ランク性の傍証**：Custom Diffusion（[[summaries/2023-custom-diffusion]]）は cross-attention の $W^k,W^v$ を全 fine-tune した差分行列の特異値が急減することを観察し、SVD 低ランク近似で 75MB→15MB（上位 60% rank, 5× 圧縮）に圧縮しても品質が保たれることを示した——personalization の重み変化が本質的に低ランクという、LoRA の前提を裏づける結果。ただし同論文は fine-tune **中**に低ランク更新を強制すると結果が suboptimal だったとも報告する。
- **どこに当てるか、が設計変数になる**：LoRA は伝統的に「注意・線形層すべてに当てる」を既定としてきたが、**B-LoRA**（[[summaries/2024-b-lora]]）は SDXL の 11 transformer ブロックのうち**第 4 ブロック（コンテンツを支配）と第 5 ブロック（色／スタイルを支配）の 2 つだけ**に当てると、style と content が勝手に分離した表現が得られることを示した。保存量は 70% 減り、学習容量が絞られるため**過学習もしない**（他手法が 400 ステップで止めるところ 1000 ステップ回せる）。層は等価ではない、という認識は $\mathcal{P}+$（層ごとのテキスト埋め込み）や ED-LoRA にも共通する。詳細は [[style-content-disentanglement]]。
- **何を学習しないか、も設計変数になる**：**Orthogonal Adaptation**（[[summaries/2024-orthogonal-adaptation]]）は $\Delta\theta = AB^\top$ の **$B$ を凍結して $A$ だけ学習する**。$B$ を共有直交基底からランダムに選んだ列に固定すれば、別々に学習された LoRA どうしが $B_i^\top B_j \approx 0$ を満たし、**素朴な足し算でマージしても干渉しない**。$B$ を凍結しても単一概念の忠実度が落ちないのは、text-to-image モデルが**過剰パラメータ化**されているためである——これは LoRA の前提（適応の内在階数は低い）をさらに一歩進めた観察と読める。
- **複数概念の合成（[[multi-concept-customization]]）と重みマージ（[[lora-merging]]）**：単一概念 LoRA を複数組み合わせて 1 枚に合成する研究が派生した。素朴な重みマージ（LoRA Merge $W'=W_0+\sum_i w_i B_iA_i$）は数が増えると identity loss・列干渉で不安定なため、推論挙動を整合させる gradient fusion（Mix-of-Show）や学習係数マージ（ZipLoRA）といった**重みマージ系（[[lora-merging]]）**、および重みを混ぜない注意制御（LoRA-Composer）・復号過程合成（Multi-LoRA Composition）が提案されている。

## 既存知識との接続

- [[subject-driven-generation]]：LoRA は personalization の軽量手段。DreamBooth（全層 fine-tune）・Textual Inversion（埋め込みのみ）と並ぶ第 3 の選択肢で、表現力↔コストのバランスが良い。
- [[multi-concept-customization]]：複数の単一概念 LoRA を 1 枚に合成するタスク。LoRA がプラグ&プレイで共有可能だからこそ成立する。
- [[style-content-disentanglement]]：LoRA を「どの層に当てるか」が分離能力を決めるという観察。LoRA の適用範囲そのものを設計対象にする視点。
- [[lora-merging]]：複数 LoRA を重みレベルで融合する系統（Mix-of-Show の gradient fusion・ZipLoRA の学習係数マージ）。LoRA の加算可能性と疎性を前提にする。
- [[diffusion-model-architecture]]：LoRA はバックボーン（U-Net でも Transformer/DiT でも）の重み行列に後付けで低ランク適応を施す。アーキテクチャ本体は変えない。
- [[latent-diffusion]]：拡散 LoRA は通常 Stable Diffusion（LDM）の U-Net に適用され、デコーダ／VAE はそのまま。

## 参考文献（summaries）

- [[summaries/2022-lora]] — LoRA: Low-Rank Adaptation of Large Language Models（Hu ら, ICLR 2022。$\Delta W=BA$・凍結・マージ・rank 分析）
- [[summaries/2023-mix-of-show]] — Mix-of-Show（ED-LoRA：分解埋め込み＋LoRA）
- [[summaries/2024-ziplora]] — ZipLoRA（LoRA の疎性に基づく content+style マージ）
- [[summaries/2024-b-lora]] — B-LoRA（SDXL の 2 ブロックのみに LoRA を当てる。当てる場所の選択が手法になる）
- [[summaries/2024-orthogonal-adaptation]] — Orthogonal Adaptation（$B$ を凍結し $A$ のみ学習。過剰パラメータ化ゆえ表現力は落ちない）
