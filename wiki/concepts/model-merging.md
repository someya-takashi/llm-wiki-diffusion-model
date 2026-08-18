---
type: concept
aliases: [Model Merging, モデルマージ, Weight Merging, 重みマージ, Task Arithmetic, タスクベクトル, Task Vector, TIES-Merging, DARE, RegMean, Model Soup, パラメータ干渉, Parameter Interference]
tags: [model-merging, lora-merging, low-rank-adaptation, multi-concept-customization, large-scale-training-infrastructure, generative-models]
related:
  - "[[lora-merging]]"
  - "[[low-rank-adaptation]]"
  - "[[multi-concept-customization]]"
  - "[[style-content-disentanglement]]"
  - "[[diffusion-model-architecture]]"
summaries:
  - "[[summaries/2026-ssr-merge]]"
  - "[[summaries/2025-np-lora]]"
  - "[[summaries/2025-z-image]]"
updated: 2026-08-19
---

# Model Merging（モデルマージ / 重みマージ）

**Model Merging（モデルマージ）** とは、**別々に学習された複数のモデル（あるいはアダプタ）を、再学習なしに重みのレベルで 1 つに統合する**手法群である。$K$ 個のタスクにそれぞれ特化したモデルがあるとき、それらを 1 つに畳めば、推論コストは 1 モデル分のまま複数の能力を持てる。

本ページは**汎用の重みマージ**を扱い、その LoRA に特化した子ページが [[lora-merging]] である。本 wiki のランドマークは **SSR-Merge**（[[summaries/2026-ssr-merge]]）で、この系譜を初めて拡散モデルの文脈へ体系的に持ち込んだ。

## 中心的な問題：パラメータ干渉

素朴なマージは、タスクベクトルを足し合わせることである。事前学習済みの重みを $\theta_{\text{pre}}$、タスク $k$ に微調整した重みを $\theta_k$ として、**タスクベクトル**を

$$\tau_k = \theta_k - \theta_{\text{pre}}$$

と定義し、$\theta_{\text{merged}} = \theta_{\text{pre}} + \lambda\sum_k \tau_k$ とする。ところが $K$ が増えるとこれが壊れる。**パラメータ干渉（parameter interference）**——別々のタスクの更新が同じパラメータを奪い合い、互いを打ち消したり増幅したりする。

現れ方には名前が付いている。

- **概念の希釈（concept dilution）**：平均を取ると各タスクの特徴の大きさが $1/K$ になり、概念をトリガーできなくなる。
- **crosstalk（クロストーク）**：あるタスクの指示が**無関係なモジュールを誤って活性化する**（[[summaries/2024-orthogonal-adaptation]] の命名）。SSR-Merge の図 1 は $K=10$ でこれを直接可視化しており、DARE ベースラインでは非対角が広く光る。
- **符号の衝突（sign conflict）**：同じパラメータについて、あるタスクは増やしたく、別のタスクは減らしたい。素朴な和では打ち消し合う。
- **意味的ドリフト**：$K$ が増えるにつれ、特定のコーギーが「一般的なハスキーのような犬」へ変質していく。

## 系譜

### (1) Model Soup / Linear Average — 単純平均

同じ事前学習済みモデルから派生した複数の微調整の**重みを平均する**。$\theta = \frac{1}{K}\sum_k \theta_k$。

これが破綻しにくいのは、**すべての変種が同じ初期点から短く離れただけ**だからである。互いに独立に学習された重みと違い、共通の親から派生した近傍の点どうしなので、線形補間が意味を持つ領域に留まりやすい。**Z-Image**（[[summaries/2025-z-image]]）が SFT の最終段階で採る「能力次元ごとに偏らせた変種を線形補間する」手法はこの系統で、目的も違う——概念の共存ではなく**個々のバイアスの中和**である。

**マージが成立する条件は「何を混ぜるか」より「どこから来たか」に依る**、という原則がここから出る。

### (2) Task Arithmetic — タスクベクトルの算術

$\tau_k$ をベクトルとして扱い、足す（能力の追加）・引く（能力の除去）といった**算術演算でモデルを編集する**。$\Delta W_{\text{TA}}=\lambda\sum_{k}\tau_{k}$。

**SSR-Merge がこれに明快な同定を与えた**——$K$ 個の LoRA を足すことは、rank 方向に連結した LoRA でルータを単位行列にすることに**厳密に等しい**。

$$\sum_{k=1}^{K}B_{k}A_{k}=\begin{bmatrix}B_{1}&\dots&B_{K}\end{bmatrix}\begin{bmatrix}A_{1}\\ \vdots\\ A_{K}\end{bmatrix}=\mathbf{B}_{\text{comb}}\,\mathbf{I}\,\mathbf{A}_{\text{comb}}$$

つまり Task Arithmetic は「**並べるが、信号を一切制御しない**」。破壊的な干渉が起きる理由が、これで 1 点に還元される。

### (3) TIES-Merging — 刈り込み・符号の選挙・マージ

干渉を**符号の衝突**として捉え、3 段階で処理する：**Trim**（大きさの小さいパラメータを刈る）→ **Elect Sign**（各パラメータについて多数派の符号を選ぶ）→ **Merge**（選ばれた符号に整合する成分だけを平均する）。

$$\Delta W_{\text{TIES}}=\text{Mean}\left(\text{SignSelect}\left(\text{Top-}k\left(\{\tau_{1},\dots,\tau_{K}\}\right)\right)\right)$$

### (4) DARE — 落として再スケールする

**Drop And REscale**。各タスクベクトルの要素を確率 $1-p$ でランダムに捨て、残りを $1/p$ で割って期待値を保つ。

$$\Delta W_{\text{DARE}}=\sum_{k=1}^{K}\frac{M_{k}\odot\tau_{k}}{p},\quad M_{k}\sim\text{Bernoulli}(p)$$

前提は「**デルタの重みは高度に冗長**で、ランダムに疎化しても中核的な機能は保たれ、他のタスクのための空間が空く」というもの。ZipLoRA が「$\Delta W$ は疎で 90% を 0 にしても品質が保たれる」と観察した性質（[[lora-merging]]）の、一般形にあたる。

**疎化系には構造的な副作用がある。** SSR-Merge の測定では、複数被写体を同時に出す設定で TIES と DARE の**成功率が素朴な Task Arithmetic より低い**（0.69・0.62 対 0.76）。忠実度を守る代わりに**タスクそのものを落としている**——[[multi-concept-customization]] の concept vanishing が、疎化ベースのマージで構造的に起きる。

### (5) RegMean — 活性化を合わせる閉形式解

重みではなく**活性化**の $L_2$ 距離を最小化する。

$$W_{M}=\left(\sum_{i}X_{i}^{T}X_{i}\right)^{-1}\sum_{i}X_{i}^{T}X_{i}W_{i}$$

発想は [[lora-merging]] の Custom Diffusion の閉形式マージや Mix-of-Show の gradient fusion と同型で、**「各モデルの入出力の振る舞いを 1 枚の重みで再現する最小二乗」**である。

**ところが高次元の DiT では破綻する。** SSR-Merge の分析（付録 E）が明快で、FLUX.1 では $d\approx 12{,}288$ に対し較正サンプルが $N\approx 4{,}096$ しかない。$N \ll d$ なので**大域的な共分散行列が階数不足で特異になり、逆行列が計算できない**。Tikhonov 正則化 $\lambda I$ を入れても逃げ場がなく、$\lambda$ が小さいと数値爆発、大きいとタスク固有の相関が洗い流されて同一性が消える。**モデルが大きくなるほど、大域空間で統計を取る手法は成立しなくなる。**

### (6) SSR-Merge — 部分空間で信号をルーティングする

**SSR-Merge**（[[summaries/2026-ssr-merge]]）はこの詰まりを、**統計を取る空間を変える**ことで抜ける。パラメータ空間で足すのをやめ、rank 方向に連結した統一部分空間（次元 $Kr$）の中にルータを挿す。

$$R:=\mathbf{Q}\mathbf{G}^{-1},\qquad \mathbf{G}=\sum_{k}Z_{k}Z_{k}^{\top},\quad \mathbf{Q}_k=(A_kX_k)Z_k^\top$$

$\mathbf{G}^{-1}$ が白色化フィルタとして混ざった信号の相関を落とし、$\mathbf{Q}$ が浄化された信号を各タスクの上方射影 $B_k$ へ導く。**$Kr\approx 96$ という小さな空間なので $N \gg Kr$ となり、$\mathbf{G}$ は自然に可逆**である——RegMean が破綻した理由がそのまま解決の理由になっている。

理論的な支えも強い。$R$ は **OLS（通常最小二乗）推定量の射影に厳密に一致する**ことが示され、有限サンプル誤差も $\mathcal{O}(\sqrt{Kr/N})$ で抑えられる。さらに展開時は $\tilde{\mathbf{B}}_\text{comb}=\mathbf{B}_\text{comb}R$ と吸収すれば標準 LoRA と同じ形になり、**推論コストの増加はゼロ**。

較正は**正解画像なしのワンショット**——タスクごとに 1 つのプロンプトで単一タイムステップだけ順伝播する。1 ステップで足りる理由の説明が本質的で、**時間依存の生成の力学は元の $A,B$ に既に入っているので、ルータは部分空間の幾何の向きを整えるだけでよい**。

$K=9$ で単独 LoRA の 90.2% を回復（最強ベースライン IterIS を DINO で +0.047）。$K=21$ でも 77.0%。GLUE でも最良（付録 D）で、**拡散に固有の手法ではない**ことが示されている。

### (7) その他：RobustMerge・IterIS

- **RobustMerge**：低ランク分解のもとでは $\tau_k$ の支配的な特異方向が干渉に敏感なので、**大きさより方向を守る**方が重要だとする。特異値に関するパラメータ間の関係で刈り込みと再スケールを行い、交差タスクの正規化で寄与を均衡させる。
- **IterIS**：層ごとの正則化最小二乗を**反復的に解く**。少数のラベルなし較正サンプル（従来の 1〜5%）で数回の反復で収束する。

## 較正データが要るか、要らないか

この系譜を横断する軸として、**マージにデータが必要かどうか**がある。

| | 較正データ | 代表 |
| --- | --- | --- |
| **データ不要** | なし | Linear Average、Task Arithmetic、TIES、DARE |
| **データ必要** | 入力活性化の統計 | RegMean、IterIS、**SSR-Merge** |

データ不要の系統は**タスクベクトルの形だけ**を見る。データを使う系統は**入力に対する応答**を見るので原理的に強いが、較正データの入手と計算のコストが乗る。SSR-Merge はこのコストを「正解画像なし・1 プロンプト・1 タイムステップ」まで削ることで、実質的にデータ不要系と同じ手軽さを実現している——マージ時間は $K=9$ で 34 秒、DARE（21 秒）と同じ桁である。

## 限界と未解決問題

- **スケールの上限がある。** SSR-Merge ですら $K=3$ の 98.6% から $K=21$ の 77.0% へ劣化する。「何個までなら実用か」を決める理論はなく、経験的に測るしかない。
- **アーキテクチャ依存が説明されていない。** 同じ SSR が Qwen-Image で 97〜99%、FLUX.1 で 90〜98% の回復率を示す。7 ポイント近い差の原因は「特徴空間の特性が違う」としか述べられない。[[diffusion-model-architecture]] の観点からは重要な問いである。
- **層をまたぐ依存が無視されている。** ほぼすべての手法が層ごとに独立にマージする。[[summaries/2024-b-lora]] が**層ごとに役割が違う**ことを示した以上、この仮定の妥当性は自明でない。
- **理論の射程と主張の射程がずれることがある。** SSR の最適性定理が保証するのは「各タスクが単独で出したはずの出力の再現」であって、複数タスクを**同時に**実行する場合ではない。ところが最大の改善はその同時実行で出ている。
- **意味的に近いタスクは依然として難しい。** SSR の限界節が明記するとおり、領域の衝突が激しい場合や意味的重複が高い場合はルーティングが困難になる。[[summaries/2023-mix-of-show]] が「意味的に類似した概念ほど重み残差が似て干渉が深刻になる」と述べたのと同じ壁である。
- **本 wiki は原典を 1 本しか持たない。** 本ページの記述の大半は SSR-Merge の関連研究節と付録 B.2 の定式化に依拠している。Task Arithmetic・TIES・DARE・RegMean の原典を直接取り込むまでは、**その手法の主張が第三者の再実装で評価された結果**を見ていることになる点に注意が必要である。

## 既存知識との接続

- [[lora-merging]]：本ページの LoRA 特化版。「被写体 × 画風」という 2 個の合成に特化した工夫（gradient fusion・学習係数マージ・零空間射影）は、汎用の $K$ 個マージとは別の設計圧力のもとにある。SSR-Merge が示した **Task Arithmetic $=R=\mathbf{I}$** という同定は、両ページの語彙を橋渡しする。
- [[low-rank-adaptation]]：LoRA の $\Delta W = BA$ が**加算可能でプラグ&プレイ**だからこそマージが成立する。rank 方向の連結という操作も低ランク構造に固有である。
- [[multi-concept-customization]]：$K$ 個のタスクを 1 モデルに詰める本ページと、$K$ 個の概念を 1 枚の画像に載せるあちらは別問題だが、SSR-Merge の RQ2 で交差する。疎化ベースのマージが concept vanishing を起こすという発見は両ページに効く。
- [[style-content-disentanglement]]：content×style の 2 個マージは本ページの最小ケース。ただしそこでは「両方を等しく保つ」のではなく**どちらを保護するか**という非対称な問題設定になる（[[summaries/2025-np-lora]]）。
- [[diffusion-model-architecture]]：$d\approx 12{,}288$ という DiT の幅が RegMean を破綻させ、部分空間で統計を取る設計を強いた。**アーキテクチャの規模がマージ手法の選択を決めている**例。
- [[instruction-based-image-editing]]：SSR-Merge の RQ3 は複数の編集 LoRA を同時適用する設定で、「逐次適用の結果を並列マージで再現できるか」という新しい評価軸を提示している。

## 参考文献（summaries）

- [[summaries/2026-ssr-merge]] — SSR-Merge（rank 方向に連結して統計量由来のルータを挿入。**Task Arithmetic $=R=\mathbf{I}$** の同定、OLS 最適性の証明、ワンショット較正、$K=21$ まで検証、GLUE への汎化、RegMean が高次元 DiT で破綻する分析）
- [[summaries/2025-np-lora]] — NP-LoRA（**加重マージでは干渉が原理的に消せない**ことを命題として証明。SVD の主方向とその零空間、Tikhonov 正則化による連続的なソフト射影）
- [[summaries/2025-z-image]] — Z-Image（能力次元ごとに偏らせた SFT 変種の線形補間。目的が概念の共存ではなく**バイアスの中和**である点で本ページの他手法と異なる）

> 未取り込みの主要原典：Model Soups（Wortsman ら 2022）、Task Arithmetic（Ilharco ら, ICLR 2023）、TIES-Merging（Yadav ら, NeurIPS 2023）、DARE（Yu ら, ICML 2024）、RegMean（Jin ら, ICLR 2023）、Fisher Merging（Matena & Raffel 2022）、Git Re-Basin（Ainsworth ら, ICLR 2023）、RobustMerge、IterIS（CVPR 2025）、LoRAHub（Huang ら, COLM 2024）。いずれも本ページの記述が依拠している基礎で、今後の ingest で追記する。
