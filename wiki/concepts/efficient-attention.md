---
type: concept
aliases: [Efficient Attention, 効率的注意, Linear Attention, 線形注意, ReLU Linear Attention, Sparse Attention, 疎注意, Mix-FFN, NoPE-friendly Attention]
tags: [efficient-attention, diffusion-model-architecture, large-scale-training-infrastructure, image-tokenizer, inference-caching, generative-models]
related:
  - "[[diffusion-model-architecture]]"
  - "[[large-scale-training-infrastructure]]"
  - "[[image-tokenizer]]"
  - "[[inference-caching]]"
  - "[[position-embedding]]"
  - "[[video-diffusion]]"
summaries:
  - "[[summaries/2024-sana]]"
  - "[[summaries/2025-wan]]"
updated: 2026-08-19
---

# Efficient Attention（効率的注意）

**Efficient Attention（効率的注意）** とは、**Transformer の自己注意が系列長 $N$ に対して $O(N^2)$ の計算量を要するという構造的な制約を、近似・置換・実装最適化のいずれかで緩める**手法群である。拡散モデル（[[diffusion-model-architecture]]）にとってこれが切実な問題になるのは、**画像・動画の潜在トークン数がそのまま系列長になる**からだ。1024×1024 の画像を $f8$ の VAE ＋ 2×2 パッチ化で扱えば $N \approx 4096$、4096×4096 なら $N \approx 65{,}536$、動画に至っては [[video-diffusion]] で $N$ が 100 万に達し、**注意だけで学習時間の 95%** を占める（[[summaries/2025-wan]]）。

本 wiki のランドマークは **Sana**（[[summaries/2024-sana]]）の **ReLU 線形注意 ＋ Mix-FFN** で、注意を $O(N)$ に落として 4096×4096 の生成で FLUX-dev 比 **104 倍**のスループットを達成した。

## 三つのアプローチ

$O(N^2)$ に対する攻め方は、原理的に三つに分かれる。**どれを選ぶかで「何を諦めるか」が変わる**。

| アプローチ | やること | 計算量 | 諦めるもの |
| --- | --- | --- | --- |
| **近似（線形注意など）** | softmax を分解可能なカーネルに置換 | $O(Nd^2)$ | 注意分布の表現力 |
| **疎化（sparse / windowed）** | 一部のトークン対のみ計算 | $O(N\sqrt{N})$ 等 | 長距離の相互作用 |
| **実装最適化（FlashAttention 等）** | 計算量は $O(N^2)$ のまま定数を削る | $O(N^2)$ | 何も（数値精度のみ） |

そして**そもそも $N$ を減らす**という第四の道がある。これは注意の手法ではなくトークナイザ側の設計だが（[[image-tokenizer]]）、実効的には最も強力で、Sana は線形注意と深圧縮オートエンコーダを**同時に**使っている。$N$ が 1/4 になれば $O(N^2)$ の注意は 1/16 になるので、$N$ を減らす方が線形化より効く場面すらある。

## 線形注意の原理：結合則を使う

通常の自己注意は

$$O_i = \sum_j \frac{\exp(Q_i K_j^\top / \sqrt{d})}{\sum_k \exp(Q_i K_k^\top / \sqrt{d})} V_j$$

を計算する。ここで $\exp(Q_i K_j^\top)$ は $i$ と $j$ が**分離できない**ため、$N \times N$ の行列を実体化せざるを得ない。これが $O(N^2)$ の正体である。

線形注意の着想は単純で、**指数関数を「$Q$ 側の関数と $K$ 側の関数の積」に分解できる形へ置き換える**。$\text{Sim}(Q_i, K_j) = \phi(Q_i)\phi(K_j)^\top$ と書ければ、

$$O_i = \frac{\phi(Q_i)\left(\sum_j \phi(K_j)^\top V_j\right)}{\phi(Q_i)\left(\sum_j \phi(K_j)^\top\right)}$$

と括り出せる。**分子の括弧の中は $i$ に依存しない $d \times d$ 行列**なので、**一度だけ計算して全クエリで使い回せる**。これで計算量は $O(N^2 d) \to O(N d^2)$ となり、$N \gg d$ の高解像度領域では劇的に安くなる。行列積の結合則を使い順序を入れ替えているだけで、近似が入るのは $\exp$ を $\phi$ に置き換えた一点のみである。

### 代表手法：Sana の ReLU 線形注意（NVIDIA / MIT 2024）

Sana は $\phi = \text{ReLU}$ を選ぶ。

$$O_i = \frac{\sum_j \text{ReLU}(Q_i)\text{ReLU}(K_j)^\top V_j}{\sum_j \text{ReLU}(Q_i)\text{ReLU}(K_j)^\top}$$

ReLU は非負性を保証するので分母が正になり、注意重みが「正の重み付き平均」であるという softmax の性質は維持される。実装上も ReLU の適用を QKV 射影の末尾に融合でき、Triton によるカーネル融合と相性が良い（学習 1.37×・推論 1.65×）。

**ただし置き換えるだけでは品質が落ちる。** Sana のアブレーション（原典 表12、512px・52K ステップ）は正直にこれを示す。

| 構成 | FID ↓ | CLIP ↑ |
| --- | --- | --- |
| FullAttn ＋ FFN（ベースライン） | 18.7 | 24.9 |
| ＋ 線形注意 | **21.5** | **23.3** |
| ＋ Mix-FFN | 18.9 | 24.8 |
| ＋ カーネル融合 | 18.8 | 24.8 |

線形注意にした瞬間 FID が 2.8 悪化し、Mix-FFN でほぼ元に戻る。**線形注意は単独では成立せず、失った表現力をどこかで補う必要がある**——これが本概念の中心的な教訓である。

### 補償装置としての Mix-FFN

Sana の **Mix-FFN** は、FFN の中に **3×3 depthwise convolution** を挟み、GLU（Gated Linear Unit, 出力の一部をゲートとして掛け合わせる構成）と組み合わせたものである。構造は `1×1 Conv → 3×3 depthwise Conv → ReLU → gate → 1×1 Conv`。

なぜこれが効くのかは、線形注意が何を失ったかを考えると見える。softmax の指数関数は**鋭いピークを作れる**——「このクエリはこの数トークンだけを強く見る」という選択的で局所的な集中が表現できる。ReLU カーネルはこれが苦手で、注意が広く平坦になりやすい。**Mix-FFN の 3×3 畳み込みは、その失われた「局所性」を明示的な帰納バイアスとして注入し直している**と読める。

副産物として、畳み込みが空間的な隣接関係を暗黙に伝えるため、**位置符号化を完全に取り除いても（NoPE, No Positional Encoding）性能が落ちない**。これは [[position-embedding]] に対する独立した答えのひとつになっている——ただし Sana 自身が付録で「2K/4K への微調整では PE を再導入する」と述べており、**高解像度の外挿までは畳み込みだけでは支えきれない**。

## 実装最適化の道：計算量ではなく定数を削る

近似を一切入れない選択肢もある。**FlashAttention** 系は $O(N^2)$ の計算量をそのままに、HBM（GPU の主記憶）と SRAM の間のデータ移動を最小化するタイリングによって、実測時間と峰メモリを大幅に削る。近似がないので品質は完全に保たれ、実装を差し替えるだけで済む——**主流の大規模モデルがこちらを選んだ**理由である。

Wan（[[summaries/2025-wan]]）はさらに **8-bit FlashAttention** に踏み込むが、ここで重要な観察が出る：**FlashAttention3 のネイティブ FP8 実装は動画生成で有意に品質が落ちる**。画像では問題にならなかった精度が、系列長の桁が違うと足りなくなる。近似のない道でも、**規模が上がると数値精度という別の壁が現れる**（[[large-scale-training-infrastructure]]）。

## いつどれを使うか

Sana の測定（原典 表14）は、線形注意の効き方が**解像度に強く依存する**ことを示している。

| 解像度 | Sana-0.6B の対 FLUX-dev 高速化 |
| --- | --- |
| 512 × 512 | 44.5× |
| 1024 × 1024 | 43.0× |
| 2048 × 2048 | 53.8× |
| 4096 × 4096 | **104.0×** |

$N$ が小さいうちは $O(N^2)$ と $O(Nd^2)$ の差が出ず、モデルサイズやトークナイザの寄与が支配的である。**線形化の御利益は $N$ が $d$ を大きく上回ってから現れる**。裏返せば、1024px 以下が主戦場のモデルにとって線形注意の動機は弱い。

実務的な優先順位はおおむね次のようになる。

1. **まず $N$ を減らす**（[[image-tokenizer]] の圧縮率を上げる）。品質への影響が最も読みやすく、注意以外の全計算も同時に安くなる。
2. **次に実装最適化**（FlashAttention、カーネル融合、量子化）。品質を犠牲にしない。
3. **それでも足りなければ近似**（線形注意・疎注意）。補償装置とセットで設計する。
4. **直交する軸として**、ステップ数を削る（[[diffusion-distillation]]・[[diffusion-sampling]]）とステップ間の冗長性を突く（[[inference-caching]]）。

## 限界・批判的視点

- **「品質を犠牲にしない」という主張は成立していない。** Sana の abstract は線形注意を「品質を犠牲にすることなく」と述べるが、表12 は明確な劣化を示す。正確には「別の機構で補償すれば取り返せる」であり、その補償が任意のタスクで効く保証はない。
- **線形注意は主流にならなかった。** 2025–2026 の大規模モデル（FLUX.1、Qwen-Image、HunyuanImage 3.0、Z-Image）はいずれもフル注意を採り、効率は FlashAttention・FP8・トークナイザ側の圧縮で稼いでいる。線形注意が退けられた理由は原典群では明示されないが、(a) 高解像度以外で得が薄い、(b) 事前学習済み LLM バックボーンを流用する潮流（[[unified-multimodal-generation]]）と互換性がない、(c) 品質の上限が読みにくい、あたりが推測できる。
- **長距離の合成的推論への影響が未検証。** 線形注意が苦手とするのは「少数のトークンへの鋭い集中」であり、これは複数物体の属性の結びつけ（GenEval の color attribution など）に効くはずの機能である。実際 Sana の GenEval 色属性スコアは 0.39–0.47 で、FLUX-schnell の 0.54 に届かない。Mix-FFN の局所的な補償では埋まらない領域が残っている可能性がある。
- **疎注意の系譜が本 wiki に未取り込み。** 動画生成で有力な windowed / block-sparse な注意（Sparse VideoGen、STA 等）や、注意行列の動的スパース化は、線形注意とは別の妥協点を選ぶ手法群である。今後の ingest で本ページへ追記する。

## 既存知識との接続

- [[diffusion-model-architecture]]：DiT が U-Net を置き換えたことで、拡散モデルの計算量は**系列長に二次**という性質を引き受けた。本ページはその代償に対する応答をまとめている。
- [[image-tokenizer]]：$N$ そのものを減らす道。Sana は $f32$ の深圧縮 AE と線形注意を掛け合わせており、両者は競合ではなく乗算的に効く。
- [[large-scale-training-infrastructure]]：注意を分散させる（Ring Attention / Ulysses）方向は、単一 GPU での効率化とは別軸の答え。動画のように 1 サンプルすら載らない規模ではこちらが必須になる。
- [[inference-caching]]：注意出力のステップ間類似性を使って**計算そのものを飛ばす**。注意を安くするのではなく回数を減らす、第三の軸。
- [[position-embedding]]：Mix-FFN の畳み込みが暗黙の位置情報を与えるため NoPE が成立する。効率化の設計が位置符号化の要否まで変えてしまう例。
- [[latent-diffusion]]：そもそも潜在空間で拡散するという選択が、$N$ を画素数から潜在トークン数へ落とす最初の一手だった。本ページの問題設定はその延長線上にある。

## 参考文献（summaries）

- [[summaries/2024-sana]] — Sana（ReLU 線形注意 ＋ Mix-FFN。$O(N^2) \to O(N)$、Triton カーネル融合、NoPE、4096px で 104×。線形注意単体では FID 18.7→21.5 と劣化し Mix-FFN で回復するアブレーションを含む）
- [[summaries/2025-wan]] — Wan（フル注意を保ったまま 8-bit FlashAttention・FP8 GEMM で高速化。系列長 100 万で注意が学習時間の 95%。FA3 のネイティブ FP8 は動画で品質劣化するという反例も）

> 未取り込みの主要原典：Linear Transformers（Katharopoulos ら 2020）、Performer（Choromanski ら 2020）、FlashAttention 1/2/3（Dao ら 2022–2024）、EfficientViT（Cai ら 2023, Sana の線形注意の直接の祖先）、Mamba / SSM 系、Sparse VideoGen（2025）。今後の ingest で本ページへ追記する。
