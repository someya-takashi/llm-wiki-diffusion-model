---
type: concept
aliases: [Diffusion Distillation, 拡散モデルの蒸留, 少ステップ生成, Few-step Generation, DMD, Distribution Matching Distillation, Consistency Models, Progressive Distillation, One-step Generation]
tags: [diffusion-distillation, diffusion-sampling, flow-matching, text-to-image-generation, generative-models]
related:
  - "[[diffusion-sampling]]"
  - "[[flow-matching]]"
  - "[[probability-flow-ode]]"
  - "[[score-based-generative-models]]"
  - "[[text-to-image-generation]]"
  - "[[reinforcement-learning-for-diffusion]]"
summaries:
  - "[[summaries/2026-qwen-image-2]]"
  - "[[summaries/2025-flux-kontext]]"
updated: 2026-08-17
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

## 代表事例：Qwen-Image-2.0 の 4-NFE 蒸留

[[summaries/2026-qwen-image-2]] は 20B 級のマルチモーダル基盤モデルに DMD を適用した。著者らが強調する困難は、**既存の蒸留研究がほぼ ImageNet のクラス条件付き設定に限られていた**ことである。T2I 生成や画像編集のような開いた条件空間、しかも肖像・風景・**テキストレンダリング**（[[visual-text-rendering]]）まで含む多様なシナリオで、極端に少ない NFE のまま全能力を保てるかは未探求だった。

結果として、**Qwen-Image-2.0-RL（40 ステップ）を教師とする 4-NFE の生徒**が、肖像・風景・自然シーンにわたって視覚品質・意味的整合・構成の一貫性を維持したと報告している。対話的な創作ワークフローに耐える推論速度を、品質を犠牲にせず得たことになる。

## 限界と注意点

- **教師を超えられない**（通常は）。蒸留は教師の能力を圧縮する枠組みであり、生徒の上限は教師である。**ただし敵対的蒸留（ADD/LADD）は例外**で、識別器が「本物らしさ」を直接押し上げるため、ステップ数を削りつつ鮮鋭さを増しうる。
- **多様性の低下**。少ステップ化、とくに分布マッチング系では、生成の多様性が落ちる傾向が報告される。
- **能力の偏った劣化**。全体の見た目は保てても、細部（小さな文字、細かいテクスチャ）が先に崩れることがある。Qwen-Image-2.0 が「テキストレンダリングを含む多様なシナリオで保つのが難しい」と明示するのはこの点である。
- **学習コスト**。生徒の学習に加え、DMD では偽スコアモデルの学習も要る。「学習済みモデルをそのまま速くする」サンプラー改良と比べると重い。
- **評価が難しい**。教師との定性比較に頼りがちで、何がどれだけ失われたかを定量化する標準がまだない（Qwen-Image-2.0 も定性比較のみ）。

## 既存知識との接続

- [[diffusion-sampling]]：同じ「高速化」でも層が違う——あちらは学習済みモデルの**たどり方**（DDIM・Heun・EDM）、こちらは**モデルそのもの**を作り替える。併用も可能。
- [[probability-flow-ode]]：軌道ベースの蒸留は、確率フロー ODE の軌道を「近道」で結び直す操作として理解できる。
- [[score-based-generative-models]]：DMD は教師スコアと生徒スコアの**差**で学習する。スコアという言語がそのまま蒸留の道具になる。
- [[flow-matching]]：DMD の偽スコアモデルは flow matching 目的で学習される。また rectified flow のように**もともと直線的な道**を学ぶ手法は、蒸留前から少ステップに強い（[[summaries/2023-flow-matching]]）。
- [[reinforcement-learning-for-diffusion]]：どちらも事前学習後の「仕上げ」工程。Qwen-Image-2.0 では **SFT → RLHF → 蒸留**の順で、RL 済みモデルを教師にして蒸留する。
- [[text-to-image-generation]]：蒸留の実用的な動機は、対話的な創作ワークフローでの応答速度にある。

## 参考文献（summaries）

- [[summaries/2026-qwen-image-2]] — Qwen-Image-2.0（20B の T2I・編集統合モデルに DMD を適用し、4 NFE の生徒が 40 ステップの教師に匹敵）
- [[summaries/2025-flux-kontext]] — FLUX.1 Kontext（LADD で 1024² を 3〜5 秒に。速度だけでなく CFG 由来のアーティファクト低減も動機。[dev] はガイダンス蒸留で作られる）

> 未取り込みの主要原典：Progressive Distillation（Salimans & Ho 2022）、Consistency Models（Song ら 2023）、DMD 原典（Yin ら 2024）、ADD / LADD 原典（Sauer ら 2023・2024）。今後の ingest で本ページへ追記する。（ADD/LADD の実適用例は [[summaries/2025-flux-kontext]] で取り込み済み。）
