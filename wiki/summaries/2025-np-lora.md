---
type: summary
source_path: raw/papers/NP-LoRA_ Null Space Projection Unifies Subject and Style in LoRA Fusion.md
source_kind: paper
title: "NP-LoRA: Null Space Projection Unifies Subject and Style in LoRA Fusion"
authors: [Chuheng Chen, Xiaofei Zhou, Geyuan Zhang, Yong Huang]
year: 2025
venue: "arXiv:2511.11051"
ingested: 2026-08-19
tags: [lora-merging, style-content-disentanglement, low-rank-adaptation, model-merging, np-lora, null-space-projection, subspace-geometry]
translation: "[[translations/2025-np-lora]]"
---

# NP-LoRA: 加重マージでは干渉が原理的に消えないことを証明する

> 原典: [[translations/2025-np-lora]] ・ `raw/papers/NP-LoRA_ Null Space Projection Unifies Subject and Style in LoRA Fusion.md`
> 著者・年: Chuheng Chen, Xiaofei Zhou, Geyuan Zhang, Yong Huang（中国科学院 情報工学研究所）, 2025（arXiv:2511.11051）
>
> **版に関する注意**: 本 wiki が取り込んだ原典は arXiv v1 に相当する。arXiv は現在より新しい版を配信しており、図が 14 枚から 13 枚に、表 I と表 II が 1 つに統合され、ランク別比較・コミュニティ LoRA 評価の表が新設されている。本ページの数値は取り込んだ版に基づく。

## 一言まとめ

「加重マージでは干渉を消せない」ことを**命題として証明**したうえで、スタイル LoRA を SVD して主方向 $V_k$ を保護対象と定め、**コンテンツ LoRA をその零空間へ射影してから足す**。硬い射影はコンテンツを削りすぎるので、Tikhonov 正則化から $P_\text{soft}=I-\frac{\mu}{1+\mu}V_kV_k^\top$ を導き、**$\mu$ ひとつで直和から硬射影までを連続的に動かせる**ようにした。

## 背景と問題意識

[[lora-merging]] のこれまでの手法は、干渉を**現象として**扱ってきた。素朴な直和は壊れる、列の cosine 類似度が高いと干渉する、crosstalk $\|\Delta\theta_j X_i\|$ が大きい——いずれも「壊れることを観測し、対処する」形である。

NP-LoRA はここを一段上げる。**干渉は構造的であって、加重マージという操作の枠内では原理的に消せない**と主張し、それを証明する。

### 命題 1：加重マージではスタイル部分空間を守れない

論法は短く、そして強い。スタイル LoRA の上位特異ベクトルを $V_k$、射影子を $P = V_kV_k^\top$ とする。加重マージ $\Delta W_m = a\Delta W_c + b\Delta W_s$ のスタイル部分空間への射影は

$$P\Delta W_m = a\,P\Delta W_c + b\,\Delta W_s$$

である。**元のスタイル成分をそのまま保つ**（$P\Delta W_m = \Delta W_s$）には

$$a\,P\Delta W_{c}=(1-b)\Delta W_{s}$$

が要求される。これが解を持つのは $P\Delta W_c$ と $\Delta W_s$ が**共線である場合に限られる**——独立に学習された LoRA では、高次元空間においてほぼ確実にそうならない。係数をベクトル化しても、$P\Delta W_c$ の各列が $\Delta W_s$ の張る空間に入る必要があり、統計的にありそうにない。

**したがって加重マージは必ずスタイル部分空間内に干渉を持ち込む。** ZipLoRA が係数を学習しても、K-LoRA が層ごとに切り替えても、この結論は変わらない——**係数の選び方の問題ではなく、加重和という操作の形の問題だから**である。

### なぜ主方向を守るのか

もう 1 つの前提として、著者らは **LoRA の生成的な振る舞いが少数の主方向に支配されている**ことを実験で示す。SVD の特異値スペクトルを可視化し、**支配的な成分を摂動するとスタイルの一貫性が急激に崩れ、微小な成分を摂動してもほとんど変わらない**（図 3）。ZipLoRA の「$\Delta W$ は疎で 90% を 0 にしても品質が保たれる」という観察と同じ方向の発見だが、**どの部分が重要かを特異値の順で特定している**点が進んでいる。

## 提案手法 / 主張

<figure>

![](../../raw/assets/2025-np-lora/fig2.png)

<figcaption>図2（再掲）: スタイル LoRA を SVD して零空間を構成し、そこへコンテンツ LoRA を射影してから足す。追加の学習もハイパーパラメータ調整も要らない。</figcaption>
</figure>

### 硬い射影

零空間への直交射影 $P_\text{null} = I - V_kV_k^\top$ を作り、

$$\Delta W_{m}=\Delta W_{s}+\Delta W_{c}(I-V_{k}V_{k}^{\top})$$

とする。命題 2 が保証するのは $P(\Delta W_c^\perp x) = 0$、すなわち**コンテンツの寄与がスタイルの主方向に一切入らない**ことである（$V_kV_k^\top(I - V_kV_k^\top) = 0$ から直ちに従う）。

### ソフト射影——ここが実用上の要

硬い射影は完璧にスタイルを守るが、**スタイル方向と部分的に重なるコンテンツの特徴まで削ってしまう**。そこで Tikhonov 正則化された目的

$$\min_{\Delta W_{c}^{\text{proj}}}\;\|\Delta W_{c}^{\text{proj}}-\Delta W_{c}\|_{2}^{2}+\mu\,(\Delta W_{c}^{\text{proj}})^{\top}P\,\Delta W_{c}^{\text{proj}}$$

を解く。第 1 項が「元のコンテンツから離れるな」、第 2 項が「スタイル部分空間にエネルギーを残すな」。Woodbury の恒等式から閉形式が出る。

$$P_{\text{soft}}=I-\tfrac{\mu}{1+\mu}V_{k}V_{k}^{\top}$$

**$\mu \to 0$ で直和、$\mu \to \infty$ で硬射影**。既存手法が離散的な選択肢だったところに、**連続なつまみ**が入った。既定値は $\mu = 0.5$。

### 射影の向きは対称ではない

見落としやすいが重要な点として、**コンテンツをスタイルの零空間へ射影するのであって、逆ではない**。著者らは両方向を実験し（図 7）、**スタイルをコンテンツの零空間へ入れるとスタイルの痕跡がほぼ完全に消える**ことを見出した。結論は「**コンテンツの表現は本質的に支配的で干渉に強く、スタイルは脆い**」。

これは [[style-content-disentanglement]] で B-LoRA について書いた「スタイル成分は上書きされやすい」という性質の、部分空間の言葉での再確認である。同時に**この手法は非対称**であり、どちらを保護するかを設計者が決めなければならないことも意味する。

### 実装：SVD ではなく QR

理論は SVD で書かれるが、実装は $A_s^\top$ の**薄い QR 分解**で済む。$\mathrm{span}(V) = \mathrm{span}(A_s^\top)$ が成り立つ（Cholesky 分解を経由した証明が付録 VIII にある）ので、$P = V_kV_k^\top = QQ^\top$ が厳密に成立する。計算量は $\mathcal{O}(mnr) \to \mathcal{O}(nr^2)$ で**1 桁削減**。

## 実験結果と知見

SDXL v1.0 と FLUX、32 のコンテンツ-スタイル対（動物 5・物体 5・スタイル 15）、各 10 枚。rank 8、$\mu=0.5$。

| 手法 | CLIP content | CLIP style | DINO content | DINO style | $S_\text{arith}$ | $S_\text{harm}$ ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| Direct | 0.75 | 0.55 | 0.54 | 0.23 | 0.51 | 0.42 |
| B-LoRA | 0.72 | 0.55 | 0.51 | 0.26 | 0.51 | 0.44 |
| ZipLoRA | 0.75 | 0.56 | 0.55 | 0.24 | 0.52 | 0.44 |
| K-LoRA | **0.76** | 0.55 | **0.57** | 0.22 | 0.52 | 0.42 |
| LoRA.rar | 0.70 | 0.53 | 0.50 | 0.30 | 0.51 | 0.46 |
| **NP-LoRA** | 0.73 | **0.59** | 0.52 | **0.33** | **0.55** | **0.50** |

**この表は K-LoRA の表 1 の鏡像である。** K-LoRA が content で 1 位・style で 3 位だったのに対し、NP-LoRA は **style で 1 位・content で 4 位**。しかも NP-LoRA の表でも K-LoRA は content 最良と記録されており、**両論文は事実認識を完全に共有している**。

著者らはこれに正面から応答する——**調和平均 $S_\text{harm}$ の導入**である。4 指標の調和平均は片方が低いと大きく下がるので、「片側だけ勝つ」戦略が報われない。NP-LoRA は $S_\text{harm}$ で 0.50、次点の LoRA.rar が 0.46、K-LoRA と Direct が 0.42 で最下位。**指標の交絡そのものへの手当て**であり、[[style-content-disentanglement]] の中心的な論点に対する具体的な提案になっている。

その他：

- **ユーザー選好 49.53%・GPT-5 選好 53.13%**（6 手法中）。次点の K-LoRA は 12.19% / 12.50%。
- **効率**（表 III）：NP-LoRA のマージは 13.4 秒で **Direct（13.5 秒）とほぼ同じ**。ZipLoRA は 10 分超、K-LoRA は生成が 60.4 秒（Direct の 2.6 倍）。**射影は一度きりなので推論に負担が残らない**。
- **$\mu$ のアブレーション**（$0, 0.1, 0.5, 1, 10, \infty$）：小さいと干渉、大きいとコンテンツが消える。硬射影（$\mu=\infty$）は表 IV で DINO style 0.40（最良）だが DINO content 0.39 まで落ちる——**ソフト化の必要性が数値で示されている**。
- **共同学習との比較**（表 IV）：Joint は $S_\text{harm}$ 0.43 で、しかも「時折うまくいくが失敗することが多い」と不安定性が指摘される。著者らは多目的最適化における**破滅的忘却**を原因に挙げる。
- **$U$ 空間ではなく $V$ 空間**（付録 IX）：$U$ は活性化空間、$V$ はパラメータ空間。マージは重みを操作するので $V$ が正しい。$U$ で構成するとコンテンツもスタイルも抑制されて表現力が落ちる。**細かいが本質的な区別**である。

## 限界・批判的視点

- **命題 1 が示すのは「厳密な保存の不可能性」であって「干渉が大きいこと」ではない。** 証明は $P\Delta W_m = \Delta W_s$ という**完全一致**を要求しており、それが共線性なしには成立しないことを示す。しかし実用上問題なのは干渉の**大きさ**であって、近似的な保存で十分かもしれない。理論的な鮮やかさが、実害の定量化を代替してはいない。
- **コンテンツ側の劣化が正面から扱われていない。** 表 I で content の CLIP は K-LoRA より 0.03、DINO は 0.05 低い。要旨は「被写体の忠実度とスタイルの一貫性を統一する」と述べるが、実態は**保護対象をスタイルに移した**結果であり、$S_\text{harm}$ という集約指標がその移動を有利に見せている面がある。集約指標の選択自体が主張の一部になっている点は割り引いて読む必要がある。
- **GPT-5 評価に位置バイアスの統制がない。** 付録 XV のプロンプトは候補の対応を **[Model_1]=Direct … [Model_6]=NP-LoRA と固定**しており、自手法が常に最後に置かれる。著者らは「各ケースを個別に判定して交差サンプルのバイアスを防いだ」と述べるが、これは別種のバイアスへの対処である。**候補順の無作為化がされていない**まま 53.13% という数字が出ている。
- **$\mu=0.5$ は経験的**（著者らも限界として明記）。しかも最適な $\mu$ がコンテンツ-スタイル対によって変わらない保証はなく、「ハイパーパラメータ調整が不要」という図 2 のキャプションの主張とは整合しない。
- **層をまたぐ独立性の仮定**（著者らが明記）。射影は層ごとに独立に行われる。ZipLoRA・K-LoRA・LoRA.rar と共有される仮定だが、[[summaries/2024-b-lora]] が**層ごとに役割が違う**ことを示した以上、層横断の依存を無視してよいかは自明でない。
- **rank 8 固定。** 「K-LoRA の慣例に従い」すべての LoRA を rank 8 で学習し、上位 8 個をすべて主方向とする。つまり**主方向の数 $k$ = rank** で、$k$ を絞るという選択肢が検討されていない。実際のコミュニティ LoRA は rank 16・32・64 と様々で、そこでの挙動は未検証である。
- **原典の版が流動的。** 本 wiki が取り込んだ版と現在 arXiv が配信する版で図表構成が異なる。査読前の原稿であり、数値も含めて確定的なものとして扱うべきではない。

## 位置づけ

本論文は [[lora-merging]] における**干渉の理解を「現象」から「幾何」へ移した**。

系譜として見ると綺麗に繋がる。**Orthogonal Adaptation**（[[summaries/2024-orthogonal-adaptation]]）は $B_i^\top B_j \approx 0$ を**学習時に**課して直交性を作った。NP-LoRA は同じ直交性を、**学習済みの LoRA に対して推論前の射影で**作る。前者は既存 LoRA に適用できないという限界を持ち、後者はそれを解いている——**「事前に防ぐ」の利点を「事後に直す」で再現した**格好である。

そして SSR-Merge（[[summaries/2026-ssr-merge]]）がさらに一歩進める。NP-LoRA が**パラメータ空間で射影する**のに対し、SSR は**そもそもパラメータ空間で足すのをやめて信号をルーティングする**。3 本を並べると「干渉はどこに住んでいるか」への答えが、学習時の基底 → マージ時のパラメータ幾何 → 推論時の信号経路、と移動していくのが見える。

## 用語と略称

- **NP-LoRA** = Null Space Projection LoRA。
- **零空間（null space）** = ある行列で写すとゼロになるベクトルの集合。ここでは $V_k$ の張る空間の直交補空間を指す。
- **SVD** = Singular Value Decomposition, 特異値分解。$\Delta W_s = U\Sigma V^\top$。
- **主方向（principal directions）** = 特異値の大きい上位の特異ベクトル。生成的な振る舞いを支配する。
- **$U$ 空間 / $V$ 空間** = それぞれ出力（活性化）領域 / 入力（パラメータ）領域を張る。マージは後者で行うので $V$ が正しい基底。
- **Tikhonov 正則化** = 二乗誤差に二次のペナルティを加える正則化。ここでは硬い制約を連続に緩める役割。
- **Woodbury の恒等式** = 低ランク更新された行列の逆行列を効率的に表す公式。
- **$S_\text{arith}$ / $S_\text{harm}$** = 4 つの類似度の算術平均 / 調和平均。後者は片方が低いと大きく下がるので**両立**を評価する。
- **LoRA.rar** = ハイパーネットワークで融合の重みを予測する手法（ICCV 2025、本 wiki 未取り込み）。
- **破滅的忘却（catastrophic forgetting）** = 新しい目的を学ぶ過程で以前に獲得した表現が失われる現象。

## 関連ページ

- [[concepts/lora-merging]]
- [[concepts/model-merging]]
- [[concepts/style-content-disentanglement]]
- [[concepts/low-rank-adaptation]]
- [[concepts/multi-concept-customization]]
- [[summaries/2025-k-lora]]
- [[summaries/2024-orthogonal-adaptation]]
- [[summaries/2024-ziplora]]
- [[summaries/2024-b-lora]]
- [[summaries/2026-ssr-merge]]
