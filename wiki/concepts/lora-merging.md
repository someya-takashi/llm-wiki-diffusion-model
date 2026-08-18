---
type: concept
aliases: [LoRA Merging, LoRA Fusion, LoRA Merge, Gradient Fusion, Weight Fusion, ED-LoRA, ZipLoRA, LoRAHub]
tags: [lora-merging, low-rank-adaptation, multi-concept-customization, subject-driven-generation, generative-models, style-content-disentanglement, orthogonal-adaptation]
related:
  - "[[low-rank-adaptation]]"
  - "[[multi-concept-customization]]"
  - "[[subject-driven-generation]]"
  - "[[controllable-generation]]"
  - "[[latent-diffusion]]"
  - "[[style-content-disentanglement]]"
summaries:
  - "[[summaries/2023-custom-diffusion]]"
  - "[[summaries/2023-mix-of-show]]"
  - "[[summaries/2024-ziplora]]"
  - "[[summaries/2025-z-image]]"
  - "[[summaries/2024-orthogonal-adaptation]]"
  - "[[summaries/2024-b-lora]]"
updated: 2026-08-19
---

# LoRA Merging / Fusion（複数 LoRA の重みマージ／融合）

**LoRA merging / fusion（LoRA のマージ／融合）** とは、**別々に学習された複数の [[low-rank-adaptation]] LoRA（低ランク適応, 重み変化を $\Delta W=BA$ で表す軽量 fine-tune）を、1 つの重みにまとめて 1 枚の画像に複数概念を同時生成する**ための手法群である。コミュニティに無数の単一概念 LoRA（特定キャラ・物体・画風など）が共有された結果、それらを「**重みのレベルで合成する**」ことが自然な課題になった。本ページは [[multi-concept-customization]]（多概念カスタマイズ）の中でも特に **(a) 重みマージ／融合系統**を細粒度に扱う。重みを混ぜずに推論時に合成する系統（注意制御・復号中心）は [[multi-concept-customization]] を参照。本 wiki のランドマークは **Mix-of-Show**（[[summaries/2023-mix-of-show]]）と **ZipLoRA**（[[summaries/2024-ziplora]]）。

## なぜ素朴なマージは破綻するのか

最も素朴な手法は **LoRA Merge（線形和）**：複数 LoRA の重み変化を加重平均してベースに差す。

$$
W'=W_0+\sum_i w_i\,\Delta W_i=W_0+\sum_i w_i B_i A_i,\qquad \sum_i w_i=1
$$

これには 2 つの典型的な破綻がある。

- **identity loss（同一性損失）**：$n$ 概念を平均すると各概念の寄与が $\frac1n$ に薄まり、個々の identity が失われる。Mix-of-Show が分析した課題。
- **signal interference（信号干渉）**：合成対象の LoRA の**列方向（出力次元）の cosine 類似度が高い**と、直和したときに互いの信号が干渉して破綻する。ZipLoRA が分析した課題（content と style を素朴に足すと両方劣化する）。

さらに LoRA 数が増えるほど不安定になり細部が崩れる（[[multi-concept-customization]] で言う concept vanishing / confusion）。マージ系統の研究は「**どう混ぜれば各 LoRA の挙動を保てるか**」を競う。

## 系統と代表手法

### (0) 源流：Custom Diffusion の閉形式マージ

LoRA を対象にする以前に、「**別々に学習した概念の重みを最小二乗で整合させて混ぜる**」発想を最初に示したのが **Custom Diffusion**（[[summaries/2023-custom-diffusion]]）である。各概念で別々に fine-tune した cross-attention の $W^k,W^v$ を、次の制約付き最小二乗で結合する：

$$
\hat W=\operatorname*{arg\,min}_{W}\|WC_{\text{reg}}^{\top}-W_0C_{\text{reg}}^{\top}\|_F\quad\text{s.t. }WC^{\top}=V
$$

「正則化キャプション $C_{\text{reg}}$ では元モデル $W_0$ の出力を保ち、対象概念の単語 $C$ は各概念の fine-tune 済み value $V$ に一致させる」という目的で、Lagrange 乗数法により**閉形式**で解ける（$\hat W=W_0+\mathbf v^\top\mathbf d$、約 2 秒）。LoRA の $BA$ ではなく cross-attention 重みそのものを対象にする点が後続と異なるが、「単独挙動を保ちつつ重みを 1 つに畳む」という目的関数は (2) Mix-of-Show の gradient fusion と本質的に同型で、重みマージ系統の出発点にあたる。

### (1) 素朴な線形和（LoRA Merge / weighted sum）

Ryu（kohya 系）由来の重み付き和 $W'=W_0+\sum_i w_i\Delta W_i$。実装は容易だが上記 identity loss / interference に直撃され、2〜3 個までが実用上の限界。後続手法のベースラインになる。

### (2) Gradient Fusion（推論挙動を整合させる融合）— Mix-of-Show

**Mix-of-Show**（[[summaries/2023-mix-of-show]]）は線形和をやめ、「**融合モデルが各概念を単独 LoRA と同じように推論する**」よう融合重みを最適化する。各概念をサンプリングして各層の入出力特徴 $X_i$ を集め、層ごとに次の最小二乗を解く（LBFGS）：

$$
W=\operatorname*{arg\,min}_{W}\sum_{i=1}^{n}\|(W_{0}+\Delta W_{i})X_{i}-WX_{i}\|^{2}_{F}
$$

すなわち各概念の単独出力 $(W_0+\Delta W_i)X_i$ を 1 枚の重み $W$ で再現する。重み平均が薄める identity を大幅に保ち（image-alignment の融合後劣化を −0.094→−0.025）、**理論上無制限**の概念融合を可能にする。

あわせて Mix-of-Show は **ED-LoRA（embedding-decomposed LoRA, 埋め込み分解 LoRA）** を提案する。通常の LoRA は概念の identity を LoRA 重みに過度に押し込むため、意味的に近い embedding が別概念に射影され **concept conflict（概念衝突）** を起こす。ED-LoRA は概念トークンを layer-wise embedding と multi-word 表現 $V=V_{rand}^{+}V_{class}^{+}$ に分解し、in-domain essence を embedding 側に残して衝突を防ぐ。さらに多概念サンプリングの属性結合を解く **regionally controllable sampling（領域制御可能サンプリング, region-aware cross-attention）** も導入し、これは [[summaries/2024-lora-composer]] の領域注入の源流になった。

### (3) 学習係数マージ（列の干渉を最小化）— ZipLoRA

**ZipLoRA**（[[summaries/2024-ziplora]]）は **content LoRA ＋ style LoRA** を安全にマージし「任意被写体×任意スタイル」を実現する（SDXL ベース）。2 つの観察に基づく：(1) LoRA の $\Delta W$ は**疎**（90% を 0 にしても品質維持）、(2) **列の cosine 類似度が高いと直和が破綻**する。そこで層・列ごとに学習可能な **merger 係数 $m_c,m_s$** を導入し、

$$
\Delta W_m=m_c\otimes\Delta W_c+m_s\otimes\Delta W_s
$$

ベースと個別 LoRA は凍結したまま係数のみを最適化する。損失は「個別 LoRA の挙動を保ちつつ、両 LoRA が同じ列を使わない（直交化する）」よう設計される：

$$
\mathcal L_{merge}=\|(D{\oplus}L_m)(x_c,p_c)-(D{\oplus}L_c)(x_c,p_c)\|_2+\|(D{\oplus}L_m)(x_s,p_s)-(D{\oplus}L_s)(x_s,p_s)\|_2+\lambda\sum_i|m_c^{(i)}\cdot m_s^{(i)}|
$$

hyperparameter-free・約 100 step・joint 学習比で 10× 高速。直和マージ・joint training・StyleDrop を subject/style fidelity で上回る。

### (5) 干渉を事前に防ぐ — Orthogonal Adaptation

ここまでの (0)〜(4) には共通点がある——**すべて「独立に学習された LoRA を、後からどう混ぜるか」を解いている**。学習は自由にやらせておき、干渉は事後に捌く。**Orthogonal Adaptation**（[[summaries/2024-orthogonal-adaptation]]）はこの前提を降り、**そもそも干渉しない LoRA を学習させる**。

まず著者らは破綻の条件を 3 行で定式化する。概念 $i$ の入力 $X_i$ に対し、マージ後の層出力が元と一致する条件は $\Delta\theta_{j}X_{i}=0$——**他人の重み残差が自分のデータをゼロに写す**ことである。この $\|\Delta\theta_j X_i\|$ を **crosstalk（クロストーク）** と名付ける。本ページ冒頭で別々に挙げた identity loss と signal interference を、**1 つの測れる量にまとめた**点が有用である。

手法は拍子抜けするほど単純で、$\Delta\theta_i = A_i B_i^\top$ のうち **$B_i$ を凍結して $A_i$ だけ学習する**。$B_i$ は**全ユーザーが共有する直交行列 $O$ からランダムに $k$ 列を取って**作る。$k \ll n$ なら $B_i^\top B_j \approx 0$ となり、$\Delta\theta_j$ の寄与が $B^\top_j \bar B_j = 0$ で消える。マージは (1) の**素朴な線形和のまま**でよい。

$B_i$ を凍結して表現力が落ちないのは、text-to-image モデルが**過剰パラメータ化**されているためで、単一概念の忠実度は制約なしの設定と同等である（マージ前の画像整合 .748 は比較対象中で最高）。実装上も、Stable Diffusion v1.5 の全層で一意な入力次元は 4 つ（320・640・768・1280）しかないので、共有すべき正方行列は 4 つだけ——固定シードでその場生成もできる。

| | マージ時間 | 同一性整合（単一 → マージ後） |
| --- | --- | --- |
| DB-LoRA ＋ 素朴な線形和 | < 1 秒 | .683 → **.098** |
| Mix-of-Show ＋ 線形和 | < 1 秒 | .728 → .706 |
| Mix-of-Show ＋ gradient fusion | **〜15 分** | .728 → .717 |
| **Orthogonal Adaptation ＋ 線形和** | **< 1 秒** | **.740 → .745** |

(2) の gradient fusion が 3 概念で 15 分を要するのに対し **1 秒未満**で、しかも**唯一劣化しない**。$n$ 概念の組合せが指数的に増える以上、組合せごとに融合を最適化する方式は原理的にスケールしない——著者らはこの設定を **modular customization（モジュラー・カスタマイゼーション）** と名付けて定式化している（[[multi-concept-customization]]）。

**代償は 2 つある。** 第一に、**既存の LoRA には適用できない**。学習プロセス自体を変えるので、コミュニティに共有された膨大な LoRA 資産を事後に直交化することはできない——(0)〜(4) の事後型が依然として必要な理由がここにある。第二に、**厳密に直交できる概念数は高々 $\lfloor n/r \rfloor$** である（SD v1.5 の最小次元 320・$r=20$ なら 16）。しかも実際はランダム抽出なので誕生日問題により遥か手前から列が重複し始める。原典は「無数の概念」へのスケーラビリティを謳いながら、**概念数の増加に伴う重複の蓄積を定量化していない**。

### (4) 係数学習による汎用融合 — LoRAHub ほか

LoRAHub は下流タスクに合わせて複数 LoRA の結合係数を（勾配フリー最適化で）学習する汎用的な重みベース融合。ZipLoRA の「係数を学習する」発想と同系統だが、画像の content×style 特化ではなくタスク適応寄り。

## 隣接：基盤モデル自身のマージ（model merging）

本ページは LoRA どうしのマージを扱ってきたが、**同じ発想が基盤モデルの全重みに対しても使われている**。**Z-Image**（[[summaries/2025-z-image]]）は SFT の最終段階で **モデルマージ** を採る。

やり方は本ページ (1) の素朴な線形和そのものである：同じバックボーンから初期化した複数の SFT 変種を、それぞれ**異なる能力次元へわずかに偏らせて**微調整し（例：厳密な指示追従に寄せたもの、美的レンダリングに寄せたもの）、重みを線形補間する。

$$\theta_{\text{final}}=\sum_{i}\alpha_{i}\theta_{i}$$

動機が本ページの文脈と少し違う点が興味深い。LoRA マージの目的は「**別々の概念を 1 枚に共存させる**」ことだったが、こちらの目的は「**個々のバイアスを中和して頑健にする**」ことである。SFT を特定の高品質データセットで行うと写実性 対 様式の柔軟性といったトレードオフやバイアスが入るので、複数の偏りを平均して**パレート最適に近づける**。著者らは「損失地形を実効的に滑らかにする」と表現する。

なぜ素朴な線形和で破綻しないのか——本ページ冒頭で見た通り、LoRA マージでは方向の干渉が deterioration を招いた。違いは**すべての変種が同じ初期点から短く離れただけ**である点にあるだろう。互いに独立に学習された LoRA と違い、共通の親から派生した近傍の点どうしなので、線形補間が意味を持つ領域に留まりやすい（LLM 側で model soup と呼ばれる現象と同じ構図）。**マージが成立する条件は「何を混ぜるか」より「どこから来たか」に依る**、という見方を補強する事例である。

## 第三の道：マージ機構が要らない LoRA を作る

本ページの (0)〜(4) は「どう混ぜるか」、下の (b)(c) は「混ぜずに推論時に合成する」を扱う。2023 年末から 2024 年にかけて、**そのどちらでもない第三の答え**が 2 つ独立に現れた——**学習の側を変えて、マージ機構そのものを不要にする**。

- **Orthogonal Adaptation**（上の (5)）は「**どう学習するか**」を変えた。$B$ を凍結して直交基底から配ることで、足し算が安全になる。
- **B-LoRA**（[[summaries/2024-b-lora]]）は「**どこを学習するか**」を変えた。SDXL の 11 ブロック中、コンテンツを支配する第 4 ブロックとスタイルを支配する第 5 ブロックだけを共同学習すると、**style と content が勝手に分離した 2 つの LoRA** が得られ、片方を別画像のものと差し替えるだけでスタイル転送が成立する。ZipLoRA が組合せごとに要求した再最適化が、まるごと消える。

両者に共通するのは、**成果物がマージ機構を必要としない LoRA である**点である。本ページが積み上げてきた閉形式マージ・gradient fusion・学習係数マージという工夫の系譜に対する、「そもそも工夫が要らない状態を作る」という応答にあたる。B-LoRA の詳細は [[style-content-disentanglement]] を参照。

## 「重みを混ぜない」系統との対比

LoRA を 1 枚に合成する手法は、重みマージ（本ページ）以外に 2 系統ある（詳細は [[multi-concept-customization]]）：

- **(b) 訓練不要の注意制御**：LoRA-Composer（[[summaries/2024-lora-composer]]）。重みを混ぜず、推論時に U-Net の注意を領域ごとに操作。
- **(c) 復号中心の合成**：Multi-LoRA Composition（[[summaries/2024-multi-lora-composition]]）の LoRA Switch / Composite。各ノイズ除去ステップで LoRA を切替・平均。

重みマージは「**1 度融合すれば追加推論コストがない**（推論は 1 モデル）」のが利点で、(b)(c) は「**重みを保つので元 LoRA を壊さず柔軟**」だが推論が重い、というトレードオフがある。

## 既存知識との接続

- [[style-content-disentanglement]]：本ページの ZipLoRA（content×style マージ）を、スタイル-コンテンツ分離という問題設定の側から扱う子ページ。B-LoRA の「分離を作らず見つける」という対極の答えを含む。
- [[low-rank-adaptation]]：マージ対象は単一概念の LoRA。$\Delta W=BA$ がプラグ&プレイで共有・加算可能だからこそマージが成立する。ED-LoRA は LoRA の派生。
- [[multi-concept-customization]]：本ページは多概念合成の「(a) 重みマージ／融合」系統の詳細版。注意制御・復号中心系は親ページに。
- [[subject-driven-generation]]：マージ対象の単一概念は DreamBooth/Textual Inversion 系の personalization で作られる。ZipLoRA は subject（content）と style を別々に学習して合成する。
- [[controllable-generation]]：Mix-of-Show の regionally controllable sampling は空間条件制御を多概念マージに組み合わせたもの。
- [[latent-diffusion]]：Mix-of-Show は Stable Diffusion 系（実写 Chilloutmix・アニメ Anything-v4）、ZipLoRA は SDXL 上で動く。

## 参考文献（summaries）

- [[summaries/2025-z-image]] — Z-Image（能力次元ごとに偏らせた複数の SFT 変種を線形補間するモデルマージ。目的は概念の共存ではなくバイアスの中和と頑健性）

- [[summaries/2023-custom-diffusion]] — Custom Diffusion（cross-attention K/V の閉形式制約付き最小二乗マージ。重みマージ系の源流）
- [[summaries/2023-mix-of-show]] — Mix-of-Show（ED-LoRA＋gradient fusion、分散型多概念カスタマイズ）
- [[summaries/2024-ziplora]] — ZipLoRA（content+style LoRA の学習係数マージ）
- [[summaries/2024-multi-lora-composition]] — Multi-LoRA Composition（重みマージのベースライン批判・decoding-centric）
- [[summaries/2024-lora-composer]] — LoRA-Composer（重みを混ぜない注意制御系）
- [[summaries/2024-orthogonal-adaptation]] — Orthogonal Adaptation（$B$ を凍結し共有直交基底からランダムに列を配る。crosstalk の定式化、素朴な線形和で 1 秒未満・唯一劣化しない。既存 LoRA には適用不可、直交可能な概念数に上限）
- [[summaries/2024-b-lora]] — B-LoRA（SDXL のブロック 4/5 だけを共同学習すると style/content が勝手に分離。マージ機構も組合せごとの再最適化も不要）
