---
type: summary
source_path: raw/papers/FLUX.1 Kontext_ Flow Matching for In-Context Image Generation and Editing in Latent Space.md
source_kind: paper
title: "FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space"
authors: [Black Forest Labs, Patrick Esser, Robin Rombach, Axel Sauer, Andreas Blattmann, Dustin Podell]
year: 2025
venue: "arXiv:2506.15742（テクニカルレポート）"
ingested: 2026-08-17
tags: [instruction-based-image-editing, character-consistency, flow-matching, diffusion-distillation, diffusion-model-architecture, text-to-image-generation]
translation: "[[translations/2025-flux-kontext]]"
---

# FLUX.1 Kontext — 文脈画像を「連結するだけ」で生成と編集を統一する

> 原典: [[translations/2025-flux-kontext]] ・ `raw/papers/FLUX.1 Kontext_ ...md`
> 著者・年・出典: Black Forest Labs, arXiv:2506.15742, 2025年6月

## 一言まとめ

**参照画像のトークンを対象画像のトークンに「くっつける」だけ**という驚くほど素朴な方法で、text-to-image 生成と画像編集を単一の rectified flow モデルに統一した（[[instruction-based-image-editing]]）。売りは 2 つ——**反復編集でも顔や物体が崩れにくい**こと（[[character-consistency]]）と、**1024² を 3〜5 秒で生成する速さ**（[[diffusion-distillation]] の LADD による）。評価には実利用から集めた新ベンチマーク **KontextBench**（1,026 ペア）を提案する。

## 背景と問題意識

指示ベースの画像編集（[[instruction-based-image-editing]]）は InstructPix2Pix 以来、合成した「指示→編集後」のペアで拡散モデルを微調整する路線で進んできた。しかし著者らは実運用で 3 つの壁が残ると指摘する。

1. **合成ペアの限界**：学習データを作るパイプライン自体の弱点をモデルが継承し、実現できる編集の多様性と写実性が頭打ちになる。
2. **同一性が保てない**：複数回の編集をまたいでキャラクタや物体の外見を保つのは未解決。ストーリーテリングやブランド用途では致命的である。
3. **遅い**：マルチモーダル LLM に組み込まれた自己回帰型のエディタは、品質面でも拡散系に劣るうえ実行時間が長く、対話的な利用に耐えない。

3 番目は見落とされがちだが重要である。編集は本質的に**試行錯誤の作業**なので、1 回 1 分かかるツールと 3 秒で返るツールは、同じ品質でも実用上まったく別物になる。

## 提案手法 / 主張

### 定式化：$y=\varnothing$ なら生成、$y\neq\varnothing$ なら編集

目標は条件付き分布 $p(x\mid y,c)$ の学習である（$x$＝出力画像、$y$＝コンテキスト画像**または $\varnothing$**、$c$＝自然言語の指示）。この $\varnothing$ が効いている——**同じネットワークが、コンテキスト画像があれば編集を、なければ純粋な text-to-image をこなす**。学習は FLUX.1 の T2I チェックポイントから始め、数百万の関係ペアで微調整する。

### 中核：系列連結（sequence concatenation）

<figure>

![](../../raw/assets/2025-flux-kontext/kontext_v2.jpg)

<figcaption>図4（再掲）: FLUX.1 Kontext の全体像。テキストは Text Stream へ、画像は VAE エンコーダを経て Visual Stream へ。Visual Stream は位置埋め込み [T=0, h, w] のノイズ化潜在と [T=1, h, w] の符号化済み入力画像からなる。N/2 個の Double Stream Blocks の後、N 個の Fused DiT Block が結合ストリームを処理する。</figcaption>
</figure>

やっていることは拍子抜けするほど単純で、**凍結 VAE で符号化したコンテキスト画像のトークンを、対象画像のトークンの後ろに追加するだけ**である。これで、

- 入出力の**解像度・アスペクト比が違っていてもよい**（連結するだけなので長さが変わっても困らない）
- **複数の参照画像へそのまま拡張できる**（$y_1,\dots,y_N$ を並べるだけ）

なお **チャネル方向の連結も試したが性能が劣った**と明記されている。素朴な設計が勝ったという報告は再現を試みる側にとって価値がある。

**位置符号化の工夫**が要点である。FLUX.1 は各トークンを時空座標 $(t,h,w)$ で添字づける 3D RoPE を使う（単一画像なら $t\equiv0$）。Kontext はここで、コンテキストトークンに**定数オフセット**を与える——対象は $\mathbf{u}_x=(0,h,w)$、$i$ 番目のコンテキストは $\mathbf{u}_{y_i}=(i,h,w)$。著者らはこれを「**仮想タイムステップ（virtual time step）**」と呼ぶ。つまり *2 枚の画像を「時間方向に 1 コマずらして並べた」ものとして扱う*。空間構造 $(h,w)$ には触らないので、画像内部の位置関係は保たれたまま、文脈と対象だけが分離される。

### 学習と高速化

損失は rectified flow の速度回帰（[[flow-matching]]）で、$z_t=(1-t)x+t\varepsilon$ に対し $\|v_\theta(z_t,t,y,c)-(\varepsilon-x)\|^2$。時刻 $t$ は **logit-normal** からサンプルし、**学習データの解像度に応じてモード $\mu$ を変える**（[[noise-schedule]]）。$y=\varnothing$ のペアではコンテキストトークンを丸ごと省くことで T2I 能力を保持する。

高速化には **LADD（latent adversarial diffusion distillation, 潜在敵対的拡散蒸留）** を使う（[[diffusion-distillation]]）。通常の flow matching サンプリングは 50〜250 回のガイダンス付き評価を要し、遅いうえガイダンスが過飽和などのアーティファクトを招く。LADD は**敵対的学習でステップ数を削りつつ品質を上げる**——蒸留が品質を落とすのでなく、むしろ上げうるという主張である。派生モデルは 3 つ：

| 版 | 作り方 | 用途 |
| --- | --- | --- |
| [pro] | フロー目的 → LADD | 標準 |
| [max] | より多くの計算 | 最高品質 |
| [dev] | 12B DiT へ**ガイダンス蒸留**、image-to-image のみで学習 | オープン重み・編集特化 |

### KontextBench — 実利用から集めたベンチマーク

既存ベンチマークへの批判が具体的である：InstructPix2Pix は合成 SD サンプル＋GPT 生成指示でバイアスがある、MagicBrush は DALLE-2 の当時の能力に制約される、Emu-Edit は低解像度で非現実的な分布、GEdit-bench は現代のマルチモーダルモデルの全範囲を代表しない、等。

そこで**クラウドソースした実世界のユースケース**から 108 枚の基底画像 → **1,026 組の画像-プロンプトペア**を編纂した。内訳は局所編集 416・大域編集 262・キャラクタ参照 193・テキスト編集 92・スタイル参照 63。ベンチマークとベースラインのサンプルを併せて公開するとしている。

## 実験結果と知見

- **速度**：1024² で **3〜5 秒**。GPT-Image-1 など競合を**最大 1 桁**上回る。
- **編集（人間評価）**：**局所編集・テキスト編集・一般的な CREF（キャラクタ参照）で最良**。大域編集は gpt-image-1 に、SREF（スタイル参照）は Gen-4 References に次ぐ 2 位。
- **キャラクタ一貫性の定量化**：**AuraFace の顔埋め込みのコサイン類似度**を編集前後で比較。人間評価と整合して他を上回る。反復編集でも**ドリフトが遅い**（図12 で GPT-Image-high・Gen-4 と比較）。
- **VAE**：Flux-VAE は PDist 0.332 / SSIM 0.896 / PSNR 31.1 で、SD3-VAE・SDXL-VAE・SD-VAE を上回る（[[image-tokenizer]]）。16 潜在チャネル・敵対的目的でゼロから学習。
- **T2I 評価の再設計**：後述の "bakeyness" 批判に基づき、プロンプト追従・美的品質・写実性・タイポグラフィ・速度の 5 次元に分解して評価（Internal-T2I-Bench, 1,000 プロンプト＋GenAI bench）。Recraft は美的品質が高いがプロンプト遵守が弱く、GPT-Image-1 はその逆、という**カテゴリ間のトレードオフ**を可視化し、FLUX.1 Kontext はバランス型と位置づける。

### "bakeyness" — 評価が「AI っぽさ」に報いてしまう問題

本論文の副次的だが刺さる指摘。T2I ベンチマークが「どちらの画像が好き？」という単一の問いに頼ると、**過飽和の色・中心被写体への過度な集中・強いボケ・均質なスタイル**という「AI 的美学」が有利になる。著者らはこれを **bakeyness** と名づけ、評価軸を分解しないと**モデルがこの方向へ最適化されてしまう**と警告する。SDXL が「COCO zero-shot FID は悪化したが人間評価では勝った」と報告した問題（[[summaries/2023-sdxl]]）と同じ、**評価指標が実際の良さと乖離する**論点の続きである。

## 限界・批判的視点

- **多ターン編集はやはり劣化する**。「過度な複数ターン編集は視覚的アーティファクトを導入しうる」と明記されており、**ドリフトが遅いだけで解決はしていない**。著者ら自身が今後の最重要課題に挙げる。
- **蒸留がアーティファクトを生む**。速度と引き換えの代償を認めている。
- **指示追従の取りこぼし**がある（特定のプロンプト要求を無視することがある）。
- **単一コンテキスト画像に限定**。定式化は複数画像をカバーするが、実装は当面 1 枚のみ。
- **定量評価が人間評価中心**で、KontextBench 上の数値表が本文にない。図で示される人間評価と AuraFace スコアが主で、**再現可能な数値比較は限定的**である。
- 学習データの詳細（「数百万の関係ペア」の作り方）が開示されていない。合成ペアの限界を批判した以上、自らのデータ構築法が伏せられている点は物足りない。

## 既存 wiki との接続

FLUX.1 Kontext は、本 wiki で**編集ベンチマークの比較対象として繰り返し登場してきた**モデルの原典である——Qwen-Image（[[summaries/2025-qwen-image]]）の GEdit-CN で 1.23 と中国語編集が壊滅していた一方、ImgEdit 総合 4.00 で 3 位につけていた、あの [pro] である。この非対称は、**多言語能力が基盤モデルの言語側に強く依存する**という [[instruction-based-image-editing]] の論点を裏づける。

技術的には SD3（[[summaries/2024-sd3]]）と同じ Black Forest Labs 系譜で、rectified flow（[[flow-matching]]）と MM-DiT 的な二重ストリーム（[[diffusion-model-architecture]]）を共有する。ただし Kontext は **double stream の後に single stream（fused DiT）を積む**構成と、**3D RoPE の仮想タイムステップ**という独自解を採る。これは Qwen-Image の MSRoPE が「テキストを画像の対角線に置く」で解いた問題——複数のモダリティ／画像を 1 本の系列でどう区別するか——に対する、別の答えになっている。

編集の統一という主題では Qwen-Image-2.0（[[summaries/2026-qwen-image-2]]）と直接競合し、あちらが「画像側を VAE 潜在に一本化＋学習比率のカリキュラム」で統一したのに対し、Kontext は「**$y=\varnothing$ を許す単一の定式化**」で統一する。蒸留については、本論文の LADD が [[diffusion-distillation]] の**敵対的ブランチ**を埋める（同ページで「未取り込み」と注記していた Sauer らの ADD 系列）。

## 用語と略称

- **in-context（文脈内）生成・編集**：参照画像を「文脈」としてトークン列に与え、追加学習なしに条件付ける方式。
- **sequence concatenation（系列連結）**：コンテキスト画像のトークンを対象トークンに連結する、本論文の中核手法。
- **virtual time step（仮想タイムステップ）**：3D RoPE の時間軸 $t$ にオフセットを与えて文脈と対象を分離する工夫。対象は $t=0$、$i$ 番目の文脈は $t=i$。
- **3D RoPE** = 3-dimensional Rotary Position Embedding（各トークンを時空座標 $(t,h,w)$ で添字づける回転位置埋め込み）。
- **double stream / single stream ブロック**：前者は画像とテキストに別重みを与え attention でのみ混ぜる（SD3 の MM-DiT と同型）、後者は両者を連結して 1 本で処理する。FLUX.1 は前者の後に後者を 38 個積む。
- **fused feed-forward ブロック**：変調パラメータを半減し、attention の入出力線形層を MLP と融合して行列積を大きくする GPU 効率化。
- **LADD** = Latent Adversarial Diffusion Distillation（潜在敵対的拡散蒸留。敵対的学習でステップ数を削りつつ品質を上げる蒸留）。**ADD** はその原型。
- **guidance distillation（ガイダンス蒸留）**：CFG の 2 回評価を 1 回に畳み込む蒸留。[dev] 版で使用。
- **CREF / SREF** = Character Reference / Style Reference（キャラクタ参照／スタイル参照）。
- **AuraFace**：顔認識・同一性保持の埋め込みモデル。編集前後の顔類似度の測定に使用。
- **visual drift（視覚的ドリフト）**：反復編集のたびに同一性や特徴が少しずつ失われる現象（[[character-consistency]]）。
- **bakeyness**：過飽和・中心被写体偏重・強いボケ・均質スタイルという「AI っぽい」美学。一般選好の評価が報いてしまうと著者らが指摘した造語。
- **KontextBench**：本論文が提案した 1,026 ペアの実世界編集ベンチマーク（5 タスク）。
- **PDist** = Perceptual Distance（VGG 特徴空間での知覚的距離。小さいほど良い）。
- **rectified flow / logit-normal**：ノイズとデータを直線で結ぶ flow matching と、その学習時刻の分布（[[flow-matching]]・[[noise-schedule]]）。
- **NCII / CSAM**：非合意の性的画像／児童性的虐待コンテンツ。分類器フィルタと敵対的学習で生成を防ぐ安全対策の対象。

## 関連ページ

- [[concepts/instruction-based-image-editing]]
- [[concepts/character-consistency]]
- [[concepts/diffusion-distillation]]
- [[concepts/flow-matching]]
- [[concepts/noise-schedule]]
- [[concepts/diffusion-model-architecture]]
- [[concepts/image-tokenizer]]
- [[summaries/2024-sd3]]
- [[summaries/2025-qwen-image]] ・ [[summaries/2026-qwen-image-2]]
- [[translations/2025-flux-kontext]]
