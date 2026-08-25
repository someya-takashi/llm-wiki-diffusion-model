---
type: concept
aliases: [Diffusion Distillation, 拡散モデルの蒸留, 少ステップ生成, Few-step Generation, DMD, Distribution Matching Distillation, Consistency Models, Progressive Distillation, One-step Generation]
tags: [diffusion-distillation, diffusion-sampling, flow-matching, text-to-image-generation, generative-models, inference-caching, flux2, ai-toolkit]
related:
  - "[[diffusion-sampling]]"
  - "[[flow-matching]]"
  - "[[probability-flow-ode]]"
  - "[[score-based-generative-models]]"
  - "[[text-to-image-generation]]"
  - "[[reinforcement-learning-for-diffusion]]"
  - "[[mixture-of-experts-diffusion]]"
  - "[[inference-caching]]"
  - "[[video-diffusion]]"
  - "[[large-scale-training-infrastructure]]"
  - "[[aesthetic-scoring]]"
  - "[[ai-toolkit]]"
summaries:
  - "[[summaries/2026-qwen-image-2]]"
  - "[[summaries/2025-flux-kontext]]"
  - "[[summaries/2025-hidream-i1]]"
  - "[[summaries/2026-hidream-o1-image]]"
  - "[[summaries/2025-wan]]"
  - "[[summaries/2025-z-image]]"
  - "[[summaries/2026-ernie-image]]"
  - "[[summaries/2025-flux2]]"
  - "[[summaries/2026-ai-toolkit]]"
updated: 2026-08-26
---

# Diffusion Distillation（拡散モデルの蒸留 / 少ステップ生成）

**Diffusion Distillation（拡散モデルの蒸留）** とは、**多ステップの拡散モデル（教師）から、わずか数ステップ——ときに 1 ステップ——で同等の画像を生成できるモデル（生徒）を作る**技術の総称である。拡散モデルの最大の実用上の弱点は「1 枚生成するのにネットワークを何十回も呼ぶ」ことであり、その回数を **NFE（Number of Function Evaluations, 関数評価回数）** と呼ぶ。蒸留は NFE を 40 → 4、あるいは 1 まで落とす。本 wiki のランドマークは **Qwen-Image-2.0**（[[summaries/2026-qwen-image-2]]）が採用した **DMD（Distribution Matching Distillation）** で、4 NFE の生徒が 40 ステップの教師に匹敵する品質を達成した。

## サンプラー高速化との違い — 「道のたどり方」か「モデルそのもの」か

[[diffusion-sampling]] も同じく「速くする」技術だが、両者は**まったく別の層に手を入れている**。

| | 何を変えるか | 再学習 | 代表例 |
| --- | --- | --- | --- |
| **サンプラー／ソルバー**（[[diffusion-sampling]]） | **たどり方**（数値解法） | 不要（学習済みモデルをそのまま使う） | DDIM・Heun・EDM・DPM-Solver |
| **蒸留**（本ページ） | **モデルそのもの**（重み） | 必要（生徒を学習する） | DMD・consistency models・progressive distillation・ADD/LADD |

サンプラーは「同じ道を大きな歩幅で歩く」工夫なので、歩幅を大きくしすぎればいずれ道を外れ、10 ステップ程度が限界になりやすい。蒸留は「**そもそも一足飛びに着けるモデルを作る**」ので、1〜4 ステップという領域に到達できる。その代わり生徒の学習コストがかかり、教師の能力を完全には引き継げないリスクがある。両者は排他ではなく、蒸留した生徒を良いソルバーで回すこともできる。

## 主なアプローチ

蒸留の手法は大きく 3 系統に整理できる。

### (1) 軌道ベース（trajectory-based）

教師のノイズ除去**軌道**を生徒に真似させる。

- **Progressive Distillation**：教師の 2 ステップ分を生徒の 1 ステップに畳み込む、を再帰的に繰り返して 1024→512→…→4 ステップと半減させていく。
- **Consistency Models**：同じ軌道上のどの点から出発しても**同じ終点を返す**よう学習する（軌道に沿った出力の一貫性）。これにより任意の時刻から 1 ステップで終点へ飛べる。

直感的には「[[probability-flow-ode]] の曲がった道を、途中を飛ばして直接結ぶ近道を覚えさせる」ことにあたる。

### (2) 分布マッチング（distribution-level matching）

軌道を追うのをやめ、**生徒が作る画像の分布を教師の分布に一致させる**。個々の軌道が違っていても、出てくる絵の分布が同じならよいという発想である。

- **DMD（Distribution Matching Distillation）** が代表。Qwen-Image-2.0 が採用した理由は、異種アーキテクチャでの**経験的な安定性**と、多様な生成シナリオでの汎用性である。

**DMD の仕組み**（数式の直感）：生徒 $G_\theta$ がノイズ $\epsilon$ と条件 $c$ からクリーンなサンプル $x_\theta$ を作る。これに再びノイズを混ぜた $x_t$ について、

- $s_{\text{real}}$：**教師**が与える「本物らしさの方向」（スコア）
- $s_{\text{fake}}$：**生徒の分布**に対するスコア（生徒の生成物で学習した補助モデルが推定）

の**差** $(s_{\text{fake}} - s_{\text{real}})$ を勾配として生徒を更新する。直感的には「**生徒の分布が本物の分布からずれている方向を打ち消すように生徒を動かす**」。2 つのスコアが一致すれば分布が一致したことになる。補助的な偽スコアモデルは生徒のサンプルで [[flow-matching]] 目的により継続学習される。

なお、生徒・教師のスコアという言い方から分かる通り、この定式化は [[score-based-generative-models]] の「スコア＝データへ向かう矢印」という見方に直接乗っている。

### (3) 敵対的蒸留（adversarial distillation）

**識別器（discriminator）を持ち込み、「生徒の出力が本物か偽物か」を見分けさせながら生徒を鍛える**系統。GAN の学習を蒸留に接ぎ木したものと考えるとよい。代表が **ADD（Adversarial Diffusion Distillation）** と、それを潜在空間で行う **LADD（Latent Adversarial Diffusion Distillation）** である。

ここには他の 2 系統と決定的に違う主張がある——**蒸留が品質を「落とす」のでなく「上げる」ことがある**。通常、蒸留は教師の能力を圧縮する作業なので生徒は教師以下になる。しかし敵対的損失は「本物らしさ」を直接押し上げる圧力なので、ステップ数を削りながら同時にシャープさを増せる。

FLUX.1 Kontext（[[summaries/2025-flux-kontext]]）が LADD を採る動機は 2 つあり、そこがこの系統の性格をよく表している。

- **速度**：通常の flow matching サンプリングは 50〜250 回のガイダンス付き評価を要し、大規模なモデル提供が高価になり対話的応用を妨げる。
- **品質**：**ガイダンス（[[classifier-free-guidance]]）が過飽和などのアーティファクトを招く**ことがある。敵対的学習はこれを打ち消す方向に働く。

つまり「速くしたい」だけでなく「**CFG の副作用を消したい**」という動機が同居している。結果、FLUX.1 Kontext は 1024² を 3〜5 秒で生成し、競合を最大 1 桁上回る速度に達した。

**ガイダンス蒸留（guidance distillation）** も併記しておく価値がある。CFG は条件付きと無条件の 2 回評価を要するが、これを 1 回に畳み込む蒸留で、FLUX.1 Kontext [dev]（12B）がこれで作られている。ステップ数ではなく**1 ステップあたりのコスト**を半減させる別軸の高速化である。

### 組み合わせる — DMD ＋ 敵対的損失

上の 3 系統は排他ではない。**HiDream-I1**（[[summaries/2025-hidream-i1]]）は (2) と (3) を足し合わせる：

$$\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{DMD}}+\lambda_{\text{adv}}\mathcal{L}_{\text{adv}}$$

動機はそのまま両者の性格の裏返しで、「DMD だけでは細部の鮮鋭さが失われるので、敵対的損失で押し戻す」。識別器は独立したネットワークではなく、**凍結した教師バックボーンの多段階特徴**を使って判定する——教師をもう一度、今度は「目利き」として使い回す設計である。これで 50 ステップの教師から 28 ステップ（Dev）と 14〜16 ステップ（Fast）の生徒を作る。

後継の **HiDream-O1-Image**（[[summaries/2026-hidream-o1-image]]）はここに標準の拡散損失を第 3 項として足す：

$$\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{DMD}}+\lambda_{\text{diff}}\mathcal{L}_{\text{diff}}+\lambda_{\text{adv}}\mathcal{L}_{\text{adv}}$$

理由は「学習の安定化と最適化の振動の緩和」。教師の分布に合わせる項（DMD）と本物らしさを押し上げる項（敵対的）はどちらも間接的な信号なので、**元の学習目的そのもの**を錨として残す、という発想である。これが [[pixel-space-diffusion]]（ピクセル空間で拡散する設計）で導入されている点は示唆的で、潜在空間より高次元で不安定になりやすい環境では錨が要る、と読める。

### DMD の理解を書き換える — Decoupled DMD（2025）

上で (2) 分布マッチングを「生徒の分布を教師の分布に一致させる」と説明したが、**Z-Image**（[[summaries/2025-z-image]]）はその理解に正面から異議を唱える。

著者らが DMD を実際に使うと、**高周波の細部が失われ、色調がずれる**という持続的なアーティファクトに遭遇した（コミュニティでも記録されつつある現象）。原因を探った結論はこうである。

> DMD の有効性は一枚岩の現象ではなく、**2 つの独立して協働する機構の結果**である。
> - **CFG-Augmentation（CA）**：蒸留を実際に駆動する**主要なエンジン**。生徒の少ステップ生成能力を築き上げているのはこちら。**この要因は先行文献でほとんど見過ごされてきた**。
> - **Distribution Matching（DM）**：主として**正則化子**として機能し、学習を安定させアーティファクトを除去する。

つまり「分布マッチング蒸留」という名が示すものは主役ではなかった、という主張である。両者を切り離し、**CA 項と DM 項に別々の再ノイズ付与スケジュール**を適用したのが **Decoupled DMD** で、これで細部の喪失と色ずれが解消する。

結果として、蒸留された 8 ステップの生徒は **100 ステップの教師に匹敵するだけでなく、写実性と視覚的インパクトで教師を上回る**。本ページ下段の「限界」で「蒸留は教師を超えられない（ただし敵対的蒸留は例外）」と書いたが、**分布マッチング系でも同じことが起きる**事例がこれで加わった。

### DMDR — 正則化子を RL に転用する

上の洞察の自然な帰結である。生成モデルへの RL（[[reinforcement-learning-for-diffusion]]）は **報酬ハッキング**——報酬関数を悪用して高スコアだが視覚的に意味をなさない画像を作る——のリスクを抱え、通常は外から正則化を足す必要がある。ところが Decoupled DMD が示したのは「**DM 項はもともと正則化子である**」ことだった。ならば外付けは要らず、RL の目的関数とそのまま組める——これが **DMDR（Distribution Matching Distillation meets Reinforcement Learning）** である。

RL が人間の選好への整合を解き放ち、DM 項が報酬ハッキングを防ぐ制約として働く。**蒸留と RL が別々の工程ではなく、同じ損失の中で役割分担する**形になっており、本ページと [[reinforcement-learning-for-diffusion]] の境界を曖昧にする方向の発展である。

留保：Decoupled DMD と DMDR の技術的詳細は Z-Image のレポート本体には書かれておらず、それぞれ別の論文に委ねられている。**CA と DM をどう切り分けたか、再ノイズ付与スケジュールをどう変えたかは、本レポートからは分からない**。

### 教師が 1 人では足りない — MT-DMD（2026）

Decoupled DMD が「CA と DM は別の役割を持つ」と示したことには、もう 1 つ自然な帰結がある。**ERNIE-Image**（[[summaries/2026-ernie-image]]）はそれを **MT-DMD（Multi-Teacher DMD）** として展開する。

まず既存手法の整理が有用である。

| 手法 | 何を足したか |
| --- | --- |
| **DMD** | 分布マッチングによる少ステップ化 |
| **DMD2** | 敵対的に似た地形を安定化させるため、**生成器 1 更新につきスコアモデルを 5 更新**する分離された比率更新 |
| **Decoupled DMD** | 勾配の干渉を緩和するため CA と DM の 2 目的へ分岐 |
| **DMDR** | RL の統合に加え、教師スコアモデルへの**ステップを意識した LoRA スケーリング**。実経路のガイダンス強度をコサインで変調：$\alpha_{real}(t)=\frac{\alpha_{init}}{2}[1+\cos(\pi\min(t,T_{dynamic})/T_{dynamic})]$ |

> なお Z-Image は Decoupled DMD と DMDR の詳細を外部論文に投げていたが、**DMDR の中身はこの ERNIE-Image のレポートで読める**。

その上で ERNIE-Image が指摘する未解決問題が **Capability Drift（能力の漂流）** である——**単独の教師は、動的な LoRA ガイダンスと報酬信号で増強されていてさえ、異種混合のデータに対してすべての意味的領域にわたって一様に最適な教師信号を提供できない**。結果として、判読可能なタイポグラフィのレンダリング（[[visual-text-rendering]]）や様式化された美しさの維持といった**専門化された部分空間で収束が最適でなくなる**。

解は**専門家教師の委員会** $\mathcal{E}=\{E_1,\dots,E_K\}$（Text-Rendering Expert、Digital Art Expert、マクロ構図の専門家など）を動的ルーティングで束ねることである。

$$\hat{x}_{0}=\sum_{k=1}^{K}\mathcal{W}_{k}(x_{t},\sigma,c,\mathcal{O})\cdot E_{k}(x_{t},\sigma,c)$$

ゲーティング $\mathcal{W}_k$ の条件付けが要点で、ノイズ付き潜在 $x_t$・ノイズ尺度 $\sigma$・意味的条件 $c$ に加え、**最適化目的 $\mathcal{O}\in\{CA, DM\}$ でも切り替える**。ここから 2 つの帰結が出る。

- **非対称な勾配トポロジー**：**同一の学習インスタンスの内部で**、DM 項は Digital Art Expert に問い合わせて大域的な様式の一貫性を強制し、CA 項は Text-Rendering Expert を介して局所的な綴りを独立に最適化する。上の Decoupled DMD の分離が、**教師割り当てのレベルまで貫かれた**形である。
- **拡散軌道に沿った専門家の引き継ぎ**：高ノイズ域では Spatial Layout の専門家がマクロな構図を確立し、低ノイズ域では高周波のレンダリングの専門家（写実的な照明、材質のテクスチャ）へ移行する。[[diffusion-sampling]] でみた「タイムステップによって仕事の内容が違う」という理解が、教師の使い分けとして実装されている。

留保も大きい。**MT-DMD にアブレーションがない**——専門家の数 $K$、ゲーティングの学習方法、単一教師の DMDR との比較のいずれも数値が示されず、§2.2.3 全体が定性的な記述に終始する。問題設定は説得的だが、**解決されたという証拠は提示されていない**。

## 代表事例：Qwen-Image-2.0 の 4-NFE 蒸留

[[summaries/2026-qwen-image-2]] は 20B 級のマルチモーダル基盤モデルに DMD を適用した。著者らが強調する困難は、**既存の蒸留研究がほぼ ImageNet のクラス条件付き設定に限られていた**ことである。T2I 生成や画像編集のような開いた条件空間、しかも肖像・風景・**テキストレンダリング**（[[visual-text-rendering]]）まで含む多様なシナリオで、極端に少ない NFE のまま全能力を保てるかは未探求だった。

結果として、**Qwen-Image-2.0-RL（40 ステップ）を教師とする 4-NFE の生徒**が、肖像・風景・自然シーンにわたって視覚品質・意味的整合・構成の一貫性を維持したと報告している。対話的な創作ワークフローに耐える推論速度を、品質を犠牲にせず得たことになる。

## 動画では：無限長生成の前提条件になる

動画（[[video-diffusion]]）において蒸留は「あると速い」ではなく、**リアルタイム生成という応用そのものを成立させる条件**になる。[[summaries/2025-wan]] は **LCM（Latent Consistency Model）とその動画版 VideoLCM** で 4 ステップ化し、10–20 倍の加速を得る。これを滑動窓のストリーミング生成（Streamer）と組み合わせることで、**8 台の A100 で 15 分の動画を 8 FPS**、int8＋TensorRT 量子化で**単一 RTX 4090 で 20 FPS** に到達する。

ここで注目すべきは、蒸留が単独では足りず、**3 つの直交する軸を掛け算している**ことである——蒸留（ステップ数を減らす）、[[inference-caching]]（1 ステップの中身を削る）、量子化（1 演算を安くする・[[large-scale-training-infrastructure]]）。本ページ冒頭の表に「サンプラー vs 蒸留」を並べたが、実務ではこの 4 つ全部を積む。

## 2 種類の蒸留が同じモデルに重なる（FLUX.2）

本ページは**ステップ蒸留**（NFE を減らす）と**ガイダンス蒸留**（CFG の 2 回評価を 1 回に畳む）を別々に扱ってきたが、**FLUX.2 klein** はその両方を同時にかけた実例である（[[summaries/2025-flux2]]）。

| | ベース（教師） | 蒸留版（生徒） |
| --- | --- | --- |
| ステップ数 | 50 | **4** |
| guidance | 真の CFG（×2 評価） | 蒸留（埋め込みすら持たない） |
| 総 NFE | **100** | **4** |

**25 倍**である。しかもアーキテクチャは完全に同一（同じ `Klein4BParams` / `Klein9BParams`）で、違うのは重みだけ。**ベース版が教師で蒸留版が生徒**という関係が、同じリポジトリの別チェックポイントとして公開されている。

実装上の帯留めが徹底しているのも特徴で、`fixed_params = {"guidance", "num_steps"}` が設定され、**CLI が 4 ステップ・guidance 1.0 以外を拒否して終了する**。蒸留モデルを教師の設定で回してしまう事故を、ツール側で構造的に防いでいる。

さらに、**スケジュール側からの補完**がある（[[concepts/noise-schedule]]）。時刻シフト係数がステップ数に依存し、**4 ステップでは 50 ステップより大きなシフト**（$s \approx 9.9$ 対 $7.6$）が自動的に適用される。高ノイズ域に時間予算を厚く配ることで、少ステップでの構図の破綻を抑えている。**蒸留は「モデルを作り替える」だけでは完結せず、刻み方の調整とセットで効いている。**

## 限界と注意点

- **教師を超えられない**（通常は）。蒸留は教師の能力を圧縮する枠組みであり、生徒の上限は教師である。**ただし敵対的蒸留（ADD/LADD）は例外**で、識別器が「本物らしさ」を直接押し上げるため、ステップ数を削りつつ鮮鋭さを増しうる。
- **多様性の低下**。少ステップ化、とくに分布マッチング系では、生成の多様性が落ちる傾向が報告される。
- **能力の偏った劣化**。全体の見た目は保てても、細部（小さな文字、細かいテクスチャ）が先に崩れることがある。Qwen-Image-2.0 が「テキストレンダリングを含む多様なシナリオで保つのが難しい」と明示するのはこの点である。
- **学習コスト**。生徒の学習に加え、DMD では偽スコアモデルの学習も要る。「学習済みモデルをそのまま速くする」サンプラー改良と比べると重い。
- **評価が難しい**。教師との定性比較に頼りがちで、何がどれだけ失われたかを定量化する標準がまだない（Qwen-Image-2.0 も定性比較のみ）。

## 蒸留モデルの上で学習する — 「アダプタを −1 倍で当てる」

本ページは蒸留を「モデルを作り替える」操作として扱ってきたが、**作り替えられたモデルの上でさらに LoRA を学習したい**という実務的な要求がある。蒸留版は速いので、そこにユーザーの被写体や画風を足したい——ところが蒸留で歪んだ軌道の上に LoRA を積むと干渉しうる。

[[concepts/ai-toolkit]] の答えは拍子抜けするほど単純である——**蒸留の逆写像を学習した LoRA を、学習中に `multiplier = -1.0` で適用する**（de-distillation adapter / assistant LoRA）。これで蒸留の効果が打ち消され、勾配は「素のモデルのように振る舞うもの」に届く。

$$\text{学習時のモデル} = \text{蒸留済みモデル} - \text{蒸留アダプタ} + \Delta W_{\text{学習中の LoRA}}$$

DMD や consistency のような理論的枠組みとは独立した、純粋に運用上の回避策である。**「蒸留は可逆な写像として近似できる」という前提**が暗黙に置かれているが、その妥当性は検証されていない。

用意されているのは **Z-Image（Turbo）・Krea2・Minimax-H3・Wan22・FLUX.1-schnell** で、**FLUX.2 には無い**（[[summaries/2026-ai-toolkit]]）。FLUX.2 では非蒸留の base 版が公開されているので、そちらで学習すればよい——**蒸留版とベース版を両方公開するという BFL の判断が、de-distillation アダプタを不要にしている**とも読める。

なお本ページが記録してきた「蒸留は多様性を狭める」（Z-Image で OneIG Diversity 0.194 → 0.139）という性質は、この回避策では戻らない。打ち消しているのは軌道の歪みであって、失われた分布の広がりではない。

## 既存知識との接続

- [[diffusion-sampling]]：同じ「高速化」でも層が違う——あちらは学習済みモデルの**たどり方**（DDIM・Heun・EDM）、こちらは**モデルそのもの**を作り替える。併用も可能。
- [[probability-flow-ode]]：軌道ベースの蒸留は、確率フロー ODE の軌道を「近道」で結び直す操作として理解できる。
- [[score-based-generative-models]]：DMD は教師スコアと生徒スコアの**差**で学習する。スコアという言語がそのまま蒸留の道具になる。
- [[flow-matching]]：DMD の偽スコアモデルは flow matching 目的で学習される。また rectified flow のように**もともと直線的な道**を学ぶ手法は、蒸留前から少ステップに強い（[[summaries/2023-flow-matching]]）。
- [[mixture-of-experts-diffusion]]：どちらも「拡散モデルは重い」への答えだが向きが直交する。蒸留は**呼ぶ回数（NFE）を減らし**、MoE は**1 回あたりのコストを据え置いたまま容量を増やす**。HiDream-I1 は両方を同時に使う。
- [[reinforcement-learning-for-diffusion]]：どちらも事前学習後の「仕上げ」工程。Qwen-Image-2.0 では **SFT → RLHF → 蒸留**の順で、RL 済みモデルを教師にして蒸留する。
- [[text-to-image-generation]]：蒸留の実用的な動機は、対話的な創作ワークフローでの応答速度にある。

## 参考文献（summaries）

- [[summaries/2025-flux2]] — FLUX.2 klein（ステップ蒸留 4 NFE ＋ ガイダンス蒸留。ベース版 100 NFE から 25 倍。同一アーキテクチャで重みのみ異なる教師/生徒が並んで公開され、CLI が蒸留設定を強制する）

- [[summaries/2026-qwen-image-2]] — Qwen-Image-2.0（20B の T2I・編集統合モデルに DMD を適用し、4 NFE の生徒が 40 ステップの教師に匹敵）
- [[summaries/2025-flux-kontext]] — FLUX.1 Kontext（LADD で 1024² を 3〜5 秒に。速度だけでなく CFG 由来のアーティファクト低減も動機。[dev] はガイダンス蒸留で作られる）

- [[summaries/2025-hidream-i1]] — HiDream-I1（DMD ＋ 敵対的損失。識別器は凍結教師バックボーンの多段階特徴で判定）
- [[summaries/2026-hidream-o1-image]] — HiDream-O1-Image（さらに標準の拡散損失を安定化項として追加。50 → 28 ステップ）

- [[summaries/2025-wan]] — Wan（LCM / VideoLCM で 4 ステップ化。滑動窓ストリーミングと組み合わせてリアルタイム動画生成を成立させる）

- [[summaries/2025-z-image]] — Z-Image（**Decoupled DMD**＝DMD を CFG-Augmentation と Distribution Matching に分解、**DMDR**＝DM 項を RL の正則化子に転用。8 NFE で 100 ステップの教師を上回る）

- [[summaries/2026-ernie-image]] — ERNIE-Image（**MT-DMD**＝多教師の動的ルーティング。Capability Drift の指摘、目的関数ごとの非対称な教師割当、軌道に沿った専門家の引き継ぎ。DMD2 / DMDR の整理も収録）

> 未取り込みの主要原典：Progressive Distillation（Salimans & Ho 2022）、Consistency Models（Song ら 2023）、DMD 原典（Yin ら 2024）、ADD / LADD 原典（Sauer ら 2023・2024）。今後の ingest で本ページへ追記する。（ADD/LADD の実適用例は [[summaries/2025-flux-kontext]] で取り込み済み。）
