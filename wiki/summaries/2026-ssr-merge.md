---
type: summary
source_path: raw/papers/SSR-Merge_ Subspace Signal Routing for Training-Free LoRA Merging in Diffusion Models.md
source_kind: paper
title: "SSR-Merge: Subspace Signal Routing for Training-Free LoRA Merging in Diffusion Models"
authors: [Zhengxuan Wei, Yi Dong, Zonghui Li, Xianhui Lin, Xing Liu, Hong Gu, Shaofeng Zhang, Wenbin Li, Qi Fan]
year: 2026
venue: "arXiv:2606.10617"
ingested: 2026-08-19
tags: [model-merging, lora-merging, low-rank-adaptation, multi-concept-customization, ssr-merge, task-arithmetic, ties-merging, dare, subspace-routing]
translation: "[[translations/2026-ssr-merge]]"
---

# SSR-Merge: パラメータを足すのをやめ、信号をルーティングする

> 原典: [[translations/2026-ssr-merge]] ・ `raw/papers/SSR-Merge_ Subspace Signal Routing for Training-Free LoRA Merging in Diffusion Models.md`
> 著者・年: Zhengxuan Wei, Qi Fan ら（南京大学 / vivo BlueImage Lab / ShanghaiTech 大学 / 東南大学 / 中国科学技術大学）, 2026（arXiv:2606.10617）

## 一言まとめ

$K$ 個の LoRA を**パラメータ空間で足すのをやめ**、rank 方向に連結して統一部分空間を作り、その中に**二次統計量から閉形式で導かれるルータ $R=\mathbf{Q}\mathbf{G}^{-1}$** を挿入して信号を各タスクへ振り分ける。$R$ が **OLS 推定量の射影に厳密に一致する**ことを証明し、$K=9$ でも単一 LoRA の性能の 90% 以上を保つ。**Task Arithmetic は $R=I$ の特別な場合**である。

## 背景と問題意識

ここまで本 wiki が [[lora-merging]] で扱ってきたのは、ほぼ **2 個の LoRA**（被写体 × 画風）の合成だった。SSR-Merge は問題設定が違う——**$K$ 個（最大 21）の任意のタスク LoRA を 1 つのモデルに詰め込み、それぞれの能力を保つ**。

この設定は LLM 側の**モデルマージ**（[[model-merging]]）の系譜そのものである。本論文はその語彙を拡散モデルへ持ち込む：Linear Average・Task Arithmetic・TIES-Merging・DARE・RegMean・RobustMerge・IterIS。本 wiki の [[lora-merging]] は「隣接：基盤モデル自身のマージ」節でこの系譜に触れながら原典の裏づけを持たなかったが、**ここで初めて埋まる**。

### 干渉を可視化する

論文冒頭のヒートマップが問題を端的に示す。10 個のタスク LoRA をマージし、**タスク $i$ の指示を入れたときにモジュール $k$ がどれだけ活性化するか**を測る。理想は対角線だけが光ることである。

<figure>

![](../../raw/assets/2026-ssr-merge/fig1.png)

<figcaption>図1（再掲）: 左が DARE ベースライン。指示が無関係なモジュールを強く誤活性化しており、非対角が広く光っている（crosstalk）。右が SSR で、明瞭な対角構造になっている。</figcaption>
</figure>

**DARE の非対角が広く光る**——「犬」の指示が「ティーポット」の LoRA を活性化してしまう。[[summaries/2024-orthogonal-adaptation]] が **crosstalk** と名付けた現象を、$K=10$ の規模で直接可視化した図である。

## 提案手法 / 主張

### rank 方向に連結して、間にルータを挟む

$K$ 個の LoRA の下方射影を縦に、上方射影を横に積む。

$$\mathbf{A}_{\text{comb}}=\begin{bmatrix}A_{1}\\ \vdots\\ A_{K}\end{bmatrix}\in\mathbb{R}^{Kr\times d},\qquad \mathbf{B}_{\text{comb}}=[B_{1}\ \dots\ B_{K}]\in\mathbb{R}^{d\times Kr}$$

**ここが本論文の最も明快な洞察**である——$K$ 個の LoRA を単純に足すことは、この連結形で $R=\mathbf{I}$ とすることに**厳密に等しい**。

$$\sum_{k}B_{k}A_{k}=\mathbf{B}_{\text{comb}}\,\mathbf{I}\,\mathbf{A}_{\text{comb}}$$

つまり **Task Arithmetic は「rank 方向に並べて、信号を一切制御せずそのまま通す」ことに他ならない**。破壊的な干渉が起きるのは当然で、**信号の方向を直交化せず盲目的に重ね合わせている**からである。この定式化は「rank 予算の公平性」も同時に保証する——Task Arithmetic・TIES・DARE・SSR はすべて同じ $K$ 個の低ランク更新に作用しており、SSR は単位ルーティングを統計量由来のルータに替えただけである。

### ルータの構成

較正データからの二次統計量で $R$ を作る。$Z_k = \mathbf{A}_\text{comb}X_k$（統一空間への射影）として、

$$\mathbf{G}=\sum_{k}Z_{k}Z_{k}^{\top}\quad(\text{相関行列}),\qquad \mathbf{Q}_k = (A_kX_k)Z_k^\top \quad(\text{方向ガイド})$$

$$R:=\mathbf{Q}\mathbf{G}^{-1}$$

役割分担が明快である。**$\mathbf{G}^{-1}$ が白色化フィルタとして混ざった信号の相関を落とし、$\mathbf{Q}$ が浄化された信号を各タスクの $B_k$ へ導く。**

### 最適性の証明

$\mathbf{B}_\text{comb}^\dagger B_k = \mathbf{E}_k$（ブロック選択行列）という擬似逆行列の性質を使うと、$\mathbf{Q}$ が因数分解でき、

$$R=\mathbf{B}_{\text{comb}}^{\dagger}\underbrace{\left(\sum_{k}Y_{k}Z_{k}^{\top}\right)\left(\sum_{k}Z_{k}Z_{k}^{\top}\right)^{-1}}_{\hat{\beta}_{\text{OLS}}}$$

——**ルータは通常最小二乗推定量の射影そのものである**。したがって $\mathcal{L}(R)=\sum_k\|\mathbf{B}_\text{comb}RZ_k - Y_k\|_F^2$ を最小化する。ヒューリスティックではなく解析解だ、という主張の根拠になっている。

### ワンショット較正

較正に**正解画像が要らない**。各タスクに「A [V] dog」のようなプロンプトを 1 つ割り当て、**単一のタイムステップだけ順伝播**して統計量を取る。トークン数で $N \approx 10^3$ となり $N \gg Kr$ なので $\mathbf{G}$ は良条件になる。

なぜ 1 ステップで足りるのかの説明が本質的である——**SSR は時間的な生成能力と信号の衝突解消を切り離している**。時間依存の力学は元の $A,B$ の中に既に入っているので、ルータは各タイムステップの操作を学ぶ必要がなく、**部分空間の幾何学的な向きを整えるだけ**でよい。幾何の向きは構造的に安定なので 1 ステップで十分、という論法である。付録 G の実測でも $T=1$ で 0.6713、$T=20$ で 0.6739 と、ほぼ差がない。

### 展開時のコストがゼロ

$\tilde{\mathbf{B}}_\text{comb} = \mathbf{B}_\text{comb}R$ と吸収すれば、結果は**標準的な LoRA と構造的に同一**になる。バックボーンへ完全にマージでき、**推論レイテンシは厳密にゼロ**。K-LoRA が生成のたびに選択を走らせて 2.6 倍遅くなる（[[summaries/2025-k-lora]]）のとは対照的である。

## 実験結果と知見

**RQ1: 単一タスクの保存**（FLUX.1-dev、DreamBooth の 10 被写体、rank 32）。目標タスクの LoRA に $K-1$ 個の「妨害」LoRA を混ぜ、目標被写体を生成できるかを測る。

| 手法 | K=3 DINO | K=5 DINO | K=7 DINO | K=9 DINO |
| --- | --- | --- | --- | --- |
| Task Arithmetic | 0.5814 | 0.4935 | 0.5165 | 0.5356 |
| TIES | 0.6264 | 0.5058 | 0.5095 | 0.4723 |
| DARE | 0.7171 | 0.6584 | 0.6087 | 0.5837 |
| IterIS | 0.7030 | 0.6720 | 0.6420 | 0.6240 |
| **SSR** | **0.7342** | **0.7059** | **0.6868** | **0.6713** |
| 回復率 | 98.6% | 94.8% | 92.3% | **90.2%** |
| Upper Bound（単独） | 0.7443 | — | — | — |

**読むべきは絶対値より劣化の傾き**である。TIES は $K=3\to9$ で 0.626 → 0.472（−25%）、SSR は 0.734 → 0.671（−9%）。スケールへの頑健さが本質的な差になっている。

**RQ2: 同時マルチタスク実行**——複数被写体を 1 枚に。Grounding DINO で検出し、**検出できなかった物体には類似度 0 を与える**（指示無視を厳しく罰する設計）。

| 手法 | DINO | CLIP | 成功率 |
| --- | --- | --- | --- |
| Task Arithmetic | 0.4052 | 0.6455 | 0.76 |
| TIES | 0.4475 | 0.6498 | 0.69 |
| DARE | 0.5050 | 0.6485 | 0.62 |
| **SSR** | **0.5704** | **0.7357** | **0.91** |

**TIES と DARE の成功率が Task Arithmetic より低い**（0.69・0.62 対 0.76）のが示唆的である。疎化は忠実度を保つ代わりに**タスクそのものを落とす**——これは [[multi-concept-customization]] でいう **concept vanishing** が、疎化ベースのマージで構造的に起きることを意味する。SSR は 0.91 で、DARE に対し +29 ポイント。

**RQ3: 画像編集への汎化**——FFHQ で口紅・チーク・アイシャドウの 3 LoRA を同時適用。逐次適用の結果を正解とする。ArcFace 0.9610、CLIP 0.9625 でいずれも最良。Task Arithmetic は「編集がほぼ入らない」、DARE は「口紅だけ過剰で他が消える」という属性の不均衡を示す。

**その他の注目点**：

- **スケーリング**（付録 C）：$K=21$ まで拡張し、回復率 90.2%（K=9）→ 77.0%（K=21）と**素直に劣化を報告している**。
- **効率**（表 2）：$K=9$ で 34.26 秒。TIES（88.93 秒）の 2.6 倍速く、DARE（20.95 秒）に対しては 13 秒のオーバーヘッド。
- **Qwen-Image でも検証**（付録 F）：回復率 **97〜99%** と、FLUX.1（90〜98%）より明確に良い。
- **GLUE への汎化**（付録 D）：拡散を離れた 8 タスクでも平均 80.9 で最良。DOGE TA（79.9）を上回る。
- **RegMean が壊れる理由**（付録 E）：$d\approx 12{,}288$ に対し $N\approx 4{,}096$ なので大域共分散行列が**特異**になる。正則化 $\lambda I$ を入れても、小さいと数値爆発、大きいとタスク固有の相関が洗い流される。**SSR が $Kr\approx 96$ の部分空間で統計を取るから可逆になる**という対比は、手法の設計動機を最もよく説明している。
- **有限サンプル誤差**（付録 H）：$\|\hat R - R^*\|_2 = \mathcal{O}(\sqrt{Kr/N})$。$Kr \ll d$ だから少数サンプルで足りる、という理論的裏づけ。

## 限界・批判的視点

- **定理と主要な結果が繋がっていない。** 定理 3.1 が最小化するのは $\sum_k\|\mathbf{B}_\text{comb}RZ_k - Y_k\|_F^2$、すなわち「**各タスクが単独で出したはずの出力を再現する**」ことである。これは RQ1（単一タスクの保存）の目的とは合うが、**RQ2（複合プロンプトで複数被写体を同時に出す）を全く保証しない**。ところが最大の改善幅が出たのは RQ2（成功率 +29 ポイント）である。著者らも限界で「局所的な線形の再構成目的であり大域的最適性を保証しない」と認めているが、**理論の射程と主張の射程のずれ**は要旨からは見えにくい。
- **「タスクの事前知識に依拠しない」という主張との齟齬。** 関連研究で SSR を「タスクの事前知識や複雑なアーキテクチャ設計に依拠せず」と位置づけるが、ワンショット較正は**各タスクのトリガー語を知っている必要がある**（「A [V] dog」）。正解画像が要らないのは事実だが、タスクの識別子は前提である。コミュニティ LoRA でトリガー語が不明・曖昧な場合の挙動は検証されていない。
- **アーキテクチャ間の差が説明されない。** Qwen-Image で 97〜99%、FLUX.1 で 90〜98%。$K=9$ では 97.0% 対 90.2% と 7 ポイント近い差がある。「Qwen-Image は FLUX.1 と異なる特徴空間の特性を示す」と述べるだけで、**なぜ部分空間ルーティングが片方でより効くのか**の分析はない。[[diffusion-model-architecture]] の観点からは重要な問いである。
- **ベースラインとの表現の非対称。** ベースラインは再構成した $\Delta W = BA$（$d\times d$）上でマージされる（因子のまま混ぜると劣化するため、と付録 B.2 で説明）。一方 SSR は因子のまま rank 方向に連結する。「rank 予算の公平性」の議論はこの差の一部を説明するが、**両者が同じ表現で比較されているわけではない**。
- **同時期の被写体×スタイル手法と比較されていない。** ZipLoRA・K-LoRA・NP-LoRA はいずれも「学習不要の LoRA 融合」でありながら比較対象に入らない。問題設定（$K$ 個の汎用タスク 対 2 個の content×style）が違うのは事実だが、$K=2$ に落とせば直接比較できるはずで、**この系譜との接続が空白のまま**である。
- **較正データが 1 プロンプト × 1 ステップ。** 付録 G はプロンプトのテンプレート 5 種で分散が小さいことを示すが、いずれも「A [V] [obj]」の変奏である。**タスクの性質が大きく異なる場合**（スタイル LoRA と被写体 LoRA を混ぜる等）に同じ安定性が成り立つかは未検証。著者ら自身、限界で「深刻な領域の衝突や高い意味的重複を持つタスク」では劣化しうると述べている。
- **図 2（手法概観図）が原典に存在しない。** ar5iv・arXiv のいずれの HTML 変換でも画像が生成されておらず、キャプションのみが残っている。手法の全体像を図で確認できない。

## 位置づけ

本論文は本 wiki に **[[model-merging]]** を持ち込んだ。Task Arithmetic のタスクベクトル、TIES の符号の選挙、DARE の Drop-and-Rescale、RegMean の閉形式——LLM 側で確立していた語彙が、初めて拡散モデルの文脈で整理された形で入ってくる。

とりわけ **Task Arithmetic = $R=\mathbf{I}$** という同定は、[[lora-merging]] の系譜全体を見直させる。素朴な線形和が壊れる理由を、本 wiki はこれまで「identity loss」「signal interference」「crosstalk」「非直交な部分空間」と複数の語彙で説明してきたが、SSR の定式化はそれらを**「rank 方向に並べたときに信号を制御していない」という 1 点**に還元する。

[[summaries/2025-np-lora]] との対比も明快である。NP-LoRA は**パラメータ空間で干渉を射影で消す**。SSR は**パラメータ空間で足すこと自体をやめ、信号の経路を作る**。同じ「部分空間の幾何」という道具立てから出発しながら、前者は重みを直交化し、後者は重みに触れず統計量でルーティングする。本 wiki で追ってきた干渉への応答は、こう並ぶ。

| | 干渉をどこで断つか | 代表 |
| --- | --- | --- |
| 学習時 | 基底を直交に配る | Orthogonal Adaptation |
| 学習時 | 学習する層を絞って分離を創発させる | B-LoRA |
| マージ時 | 主方向の零空間へ射影する | NP-LoRA |
| マージ時 | 統計量由来のルータで信号を振り分ける | **SSR-Merge** |
| 推論時 | 層・ステップごとに片方だけを使う | K-LoRA |

## 用語と略称

- **SSR** = Subspace Signal Routing, 部分空間信号ルーティング。
- **PEFT** = Parameter-Efficient Fine-Tuning, 効率的パラメータ微調整。LoRA はその代表。
- **Task Arithmetic** = タスクベクトル $\tau_k=\theta_{ft}-\theta_{pre}$ を足し引きしてモデルを編集する手法。本論文で $R=\mathbf{I}$ の特別な場合と同定された。
- **TIES-Merging** = Trim（刈り込み）・Elect Sign（符号の選挙）・Merge の 3 段階で符号の衝突を解消する手法。
- **DARE** = Drop And REscale。デルタをランダムに落として $1/p$ で再スケールする。
- **RegMean** = 活性化の $L_2$ 距離を最小化する閉形式マージ。$d \gg N$ で共分散行列が特異になり高次元 DiT では破綻する。
- **RobustMerge** = 低ランク部分空間で**方向の頑健性**に焦点を当てる PEFT 特化のマージ。
- **IterIS** = 層ごとの正則化最小二乗を反復して解く最適化ベースのマージ。
- **OLS** = Ordinary Least Squares, 通常最小二乗。$\|\beta Z - Y\|_F^2$ を最小化する一意の解。
- **十分統計量の加法性** = 共分散などの統計量がバッチごとに足し合わせられる性質。ストリーミング計算の根拠。
- **構造的な再パラメータ化** = ルータを上方射影へ吸収し、標準 LoRA と同じ形にすること。推論コストがゼロになる。
- **Grounding DINO** = テキストで指定した物体を検出するモデル。RQ2 の成功率の測定に使う。
- **FFHQ** = Flickr-Faces-HQ、高品質な顔画像データセット。
- **GLUE** = 自然言語理解の標準ベンチマーク群。付録 D で拡散外への汎化検証に使われる。
- **crosstalk（クロストーク）** = 無関係なタスクのモジュールが誤って活性化される干渉（[[summaries/2024-orthogonal-adaptation]] の命名）。

## 関連ページ

- [[concepts/model-merging]]
- [[concepts/lora-merging]]
- [[concepts/low-rank-adaptation]]
- [[concepts/multi-concept-customization]]
- [[concepts/instruction-based-image-editing]]
- [[summaries/2025-np-lora]]
- [[summaries/2025-k-lora]]
- [[summaries/2024-orthogonal-adaptation]]
- [[summaries/2024-b-lora]]
