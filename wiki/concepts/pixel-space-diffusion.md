---
type: concept
aliases: [Pixel-Space Diffusion, ピクセル空間拡散, 画素空間拡散, Pixel Diffusion Transformer, VAE-free Diffusion, Unified Transformer, UiT]
tags: [pixel-space-diffusion, latent-diffusion, image-tokenizer, diffusion-model-architecture, text-to-image-generation, generative-models]
related:
  - "[[latent-diffusion]]"
  - "[[image-tokenizer]]"
  - "[[diffusion-model-architecture]]"
  - "[[denoising-diffusion]]"
  - "[[visual-text-rendering]]"
  - "[[super-resolution]]"
  - "[[text-to-image-generation]]"
  - "[[mixture-of-experts-diffusion]]"
summaries:
  - "[[summaries/2026-hidream-o1-image]]"
  - "[[summaries/2026-qwen-image-vae-2]]"
  - "[[summaries/2020-ddpm]]"
  - "[[summaries/2021-adm]]"
updated: 2026-08-17
---

# Pixel-Space Diffusion（ピクセル空間拡散）

**Pixel-Space Diffusion（ピクセル空間拡散）** とは、**VAE（Variational Autoencoder, 画像を潜在表現へ圧縮・復元するオートエンコーダ）による潜在空間への圧縮を経由せず、生の画素の上で直接ノイズ除去を学習する拡散モデル**の総称である。[[latent-diffusion]]（LDM, 潜在空間で拡散を行う枠組み）が 2022 年以降ほぼ唯一の実用解とされてきた中で、**その前提そのものを問い直す**立場にあたる。本 wiki のランドマークは **HiDream-O1-Image**（[[summaries/2026-hidream-o1-image]]）で、VAE と外付けテキストエンコーダの両方を捨て、生ピクセル・テキスト・条件を単一の Transformer に流し込む設計を 200B+ までスケールさせた。

## 歴史的な順序に注意 — 「元に戻る」のではない

紛らわしいので最初に整理しておく。**拡散モデルはもともとピクセル空間で始まった**。DDPM（[[summaries/2020-ddpm]]）も ADM（[[summaries/2021-adm]]）も、32² や 256² の画素を U-Net で直接ノイズ除去していた。高解像度は cascaded super-resolution（[[super-resolution]]）で段階的に上げるのが定石で、Imagen はこの路線である。

Latent Diffusion が主流になったのは、**画素空間のままでは 512² 以上へのスケーリングが計算的に破綻した**からだった。1024² の画像は 100 万画素以上あり、パッチ化しても系列が長すぎる。VAE で 8 分の 1 に圧縮すれば面積は 64 分の 1 になり、Transformer の注意計算は系列長の二乗に効くので**数千分の 1**になる。これが Stable Diffusion を成立させた。

だから 2025–2026 年の pixel-space diffusion は「初期の手法への回帰」ではない。**LDM が支払った代償が無視できなくなったので、別の方法でコストを吸収し直そう**という試みである。何が変わったかというと：

- **Transformer が U-Net を置き換えた**（[[diffusion-model-architecture]]）。パッチサイズを大きく取れば、ピクセル空間でも系列長を制御できる。
- **モデルとハードウェアの規模が桁違いになった**。8B〜200B のモデルなら、圧縮しない代償を容量で吸収できる余地がある。
- **LLM の事前学習資産を使いたくなった**。潜在空間はモデル固有の抽象表現なので、言語モデルのトークン空間と接続しづらい。生ピクセルなら「ただのパッチ」として言語トークンと同じ系列に並べられる。

## LDM が払っている代償

pixel-space 派の主張は、[[image-tokenizer]] で整理した論点の裏返しになっている。

**(1) 情報ボトルネック。** VAE の圧縮は非可逆であり、**高周波成分——細い線、小さな文字、細かいテクスチャ——が最初に失われる**。拡散モデルがどれほど良く学習されても、最後に VAE デコーダを通す以上、**再構成品質が生成品質の上限を決める**。Qwen-Image（[[summaries/2025-qwen-image]]）が「画像内テキストの描画品質は VAE デコーダが天井を作る」と診断し、デコーダをテキスト特化で微調整して対処したのは、この上限の存在を認めた上での**対症療法**だった（[[visual-text-rendering]]）。

**(2) 意味的なずれ。** これは VAE ではなくテキスト側の話だが、pixel-space 派の「統一」論はここまで含む。CLIP や T5 は**生成モデルとは別に学習されたもの**を持ってくるため、視覚表現と言語表現が根本から共同最適化されていない。HiDream-I1（[[summaries/2025-hidream-i1]]）が 4 系統のエンコーダを混ぜ、Qwen-Image が凍結 MLLM 1 本に集約したのは、いずれもこのずれを外側から埋める工夫にあたる。

**(3) パイプラインの断片化。** VAE・テキストエンコーダ・DiT が別々に学習された部品として並ぶため、エンドツーエンドの最適化ができない。

## 3 つの立ち位置

HiDream-O1-Image の図5 が、この地形をきれいに整理している。

| | 画像側 | テキスト側 | 代表 |
| --- | --- | --- | --- |
| **(a) latent DiT** | VAE エンコーダ／デコーダ | 外付けエンコーダ | SD3・FLUX.1・Qwen-Image |
| **(b) pixel-space DiT** | **Patchify / Unpatchify のみ** | 外付けエンコーダ | simpler diffusion・JiT |
| **(c) natively unified** | **Patchify / Unpatchify のみ** | **バックボーンのネイティブ語彙** | HiDream-O1-Image |

(b) は「VAE だけ外す」段階で、多くはまだ T2I 単機能である。(c) は「テキストエンコーダも外し、生成も編集も個人化も同じトークン空間の文脈内推論として扱う」段階にあたる。

## 代表手法: HiDream-O1-Image（HiDream.ai 2026）

[[summaries/2026-hidream-o1-image]] は (c) の具体化である。3 種類のトークンを 1 本の系列に並べる：

- **生成トークン**：目標画像とノイズの線形補間 $x_t = t\,x + (1-t)\,\varepsilon$ を、そのままパッチ分割して学習可能なパッチ埋め込みへ。
- **テキストトークン**：バックボーン（LLM）のネイティブ語彙でトークン化。
- **条件トークン**：編集元や参照被写体を SigLIP-2 で符号化し、学習可能な射影で共有空間へ。

バックボーンは **decoder-only Transformer**（8B 版は Qwen3-VL-8B-Instruct から初期化）。DiT の定番だった adaLN 変調（[[diffusion-model-architecture]]）を使わず、**拡散のタイムステップを「特別なトークン 1 個」として系列に混ぜる**ので、Transformer の中核構造を一切改変せずに済む。これがスケーリングのしやすさ（200B+ への拡張）の根拠とされている。

### ハイブリッド注意 — 言語は因果的、画像は全結合

LLM をそのまま拡散バックボーンにする際の核心的な難所は、**注意のパラダイムが正反対**であることだ。自己回帰的な言語モデリングは因果マスク（各トークンは先行トークンのみを見る）を前提にし、拡散 Transformer は完全注意（全トークンが互いを見る）を前提にする。HiDream-O1-Image は 1 つの注意行列の中でマスクを使い分ける：

- **条件・テキストトークン → 因果マスク**。事前学習された自己回帰的な能力を壊さない。
- **生成トークン → 完全注意**。画像は 2 次元の大域的依存を持つので、片方向では組み立てられない。

### ピクセル空間ならではの損失

ピクセル空間の拡散は細部を捉える一方、**長距離の意味的一貫性のモデル化に弱い**——潜在空間なら VAE が意味的な圧縮を済ませてくれていた仕事を、自前でやらねばならないからである。そこで [[flow-matching]] 損失に加え、**LPIPS 損失**（学習済み CNN 特徴での知覚的距離）と**知覚的 DINO 損失**（自己教師あり視覚特徴での整合）を足す。

興味深いのは、Qwen-Image-VAE-2.0（[[summaries/2026-qwen-image-vae-2]]）が**潜在を DINOv2 中間層へ意味的に整合させた**のと同じ道具が、こちらでは**生成モデル本体の損失**として現れることである。「意味的な足場を外から与える」という処方箋自体は共通で、それを潜在空間に埋め込むか損失として課すかが分かれている。

### 何が示されたか

最も雄弁なのは**画像内テキスト描画**である。LongText-Bench で、同チームの前作 HiDream-I1-Full（VAE あり）が英語 0.543 / 中国語 0.024 だったのに対し、HiDream-O1-Image（8B, VAE なし）は **0.979 / 0.978**。CVTG-2K の平均単語正解率も 0.9128 で、27B の Qwen-Image（0.8288）を上回る。著者らはこれを **VAE 圧縮を経由しないこと**に直接帰している。[[visual-text-rendering]] が積み上げてきた「VAE がテキスト描画の天井を作る」という命題に対する、最も直接的な答えになっている。

プロンプト追従では GenEval 総合 0.90（200B+ で 0.92）。とくに従来モデルが軒並み苦手だった **Position が 0.93**（Qwen-Image 0.76、FLUX.2 0.73）と突出しており、共有トークン空間が空間的接地に効いているという主張の柱になっている。

## 未解決の問題

- **計算コストの空白**。ピクセル空間のトークン数は、圧縮率 $f$ の潜在空間に比べ最大 $f^2$ 倍（$f=16$ なら 256 倍）になりうる。パッチサイズを大きく取って抑えているはずだが、HiDream-O1-Image はパッチサイズも系列長も学習コストも推論レイテンシも報告していない。[[image-tokenizer]] の中心命題（DiT の計算量は潜在トークン数に二次的）に真っ向から関わる論点なのに、**代償の側が定量化されていない**。「VAE を外せる」ことと「外した方が安い」ことは別である。
- **因果が交絡している**。「VAE を外したからテキストが描ける」という主張は、同時に導入された Qwen3-VL 初期化・共有トークン空間・LPIPS/DINO 損失・LM+MMU 同時学習と分離されていない。アブレーションが存在しないため、pixel-space という選択の寄与だけを取り出せない。
- **記憶（memorization）のリスク**。潜在圧縮を挟まないモデルは、学習画像をそのまま覚え込みやすい可能性がある。この論点は現時点の原典では扱われていない。
- **超高解像度の扱い**。2,048² を扱うと明記されるが、それを超える領域で系列長がどう振る舞うかは未検証。ここは伝統的に [[super-resolution]] のカスケードが担ってきた領域である。

## 既存知識との接続

- [[latent-diffusion]]：本ページが問い直す対象。LDM の「圧縮してから拡散する」は 2022 年以降の暗黙の前提であり、pixel-space はその代償（情報ボトルネック）を計上し直す立場。両者は排他ではなく、**圧縮率をどこまで下げるか**という連続軸の両端と見ることもできる。
- [[image-tokenizer]]：正面から対立する。あちらは「潜在空間の形が下流の学習しやすさを決める」として VAE の設計を精緻化し（$f16/f32$・チャネル補償・拡散可能性）、こちらは「そもそも潜在空間を作らない」と答える。**同じ 2026 年に、同じ問題への正反対の答えが出ている**構図で、決着はついていない。
- [[diffusion-model-architecture]]：UiT は「LLM の decoder-only Transformer をそのまま拡散バックボーンにする」という新しい枝。adaLN 変調を捨ててタイムステップをトークン化する点が、DiT 系譜との明確な分岐点になる。
- [[visual-text-rendering]]：VAE 起因の上限という診断に対する最も直接的な処方箋。
- [[denoising-diffusion]]：DDPM の原型はそもそもピクセル空間だった。歴史的な連続性はあるが、動機と手段はまったく異なる。
- [[super-resolution]]：ピクセル空間で高解像度に到達する古典的な答えはカスケードだった。pixel-space DiT は単一モデルで直接到達しようとする。
- [[flow-matching]]：HiDream-O1-Image の生成トークンは線形補間パスで作られ、学習目的も flow matching。定式化そのものは潜在空間の場合と変わらない——**変わるのは「どの空間の上で回すか」だけ**である。

## 参考文献（summaries）

- [[summaries/2026-hidream-o1-image]] — HiDream-O1-Image（VAE も外部テキストエンコーダも捨てる Unified Transformer。LongText-Bench で前作の 0.024 → 0.978）
- [[summaries/2026-qwen-image-vae-2]] — Qwen-Image-VAE-2.0（対立する立場。潜在空間を精緻に設計し尽くす路線）
- [[summaries/2025-hidream-i1]] — HiDream-I1（同チームの前作。潜在空間＋外部エンコーダ 4 系統という正反対の設計）
- [[summaries/2020-ddpm]] — DDPM（ピクセル空間で始まった拡散モデルの原型）
- [[summaries/2021-adm]] — ADM（ピクセル空間 U-Net の到達点）

> 未取り込みの主要原典：simpler diffusion（Hoogeboom ら 2025）、JiT。いずれも図5 の (b)「VAE だけ外す」段階にあたる先行研究で、今後の ingest で本ページへ追記する。
