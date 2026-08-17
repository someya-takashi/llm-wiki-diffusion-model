---
type: concept
aliases: [Instruction-Based Image Editing, 指示ベース画像編集, TI2I, Text-Image-to-Image, Instructional Editing, InstructPix2Pix]
tags: [instruction-based-image-editing, text-to-image-generation, controllable-generation, latent-diffusion, generative-models]
related:
  - "[[text-to-image-generation]]"
  - "[[image-inpainting]]"
  - "[[image-composition]]"
  - "[[controllable-generation]]"
  - "[[subject-driven-generation]]"
  - "[[visual-text-rendering]]"
summaries:
  - "[[summaries/2025-qwen-image]]"
  - "[[summaries/2026-qwen-image-2]]"
updated: 2026-06-25
---

# Instruction-Based Image Editing（指示ベース画像編集）

**Instruction-Based Image Editing（指示ベース画像編集）** とは、**元画像と自然言語の指示文を与えると、指示された箇所だけを変えた画像を返す**タスクである。「この猫にサングラスをかけて」「背景を雪山にして」「看板の "Hope" を "Qwen" に変えて」「この人を立たせて」——マスクを描くことも、被写体ごとに学習することもなく、**文章で editing を指定する**のが特徴。入力がテキストと画像の両方なので **TI2I（Text-Image-to-Image）** とも呼ばれる。本 wiki のランドマークは **Qwen-Image**（[[summaries/2025-qwen-image]]）で、GEdit・ImgEdit の両ベンチマークで首位に立った。

## 何が難しいのか — 2 つの一貫性

指示編集の核心は「変えるべき所だけを変える」ことにある。これは 2 種類の一貫性の両立として整理できる。

- **視覚的一貫性（visual consistency）**：指示していない部分の見た目を保つ。髪の色を変えるとき、顔立ち・肌の質感・背景はそのままであってほしい。
- **意味的一貫性（semantic coherence）**：構造が変わっても意味が壊れない。ポーズを変えても**同じ人物**に見え、シーンの整合（照明・遠近・服の重なり）が保たれてほしい。

素朴な作りだと、この 2 つは綱引きになる。画像を強く条件づけると元画像のコピーに近づいて指示に従わなくなり、指示を強く効かせると別人・別シーンになってしまう。**この綱引きをどう解くかが手法の分かれ目**である。

## 隣接タスクとの違い

本 wiki には編集系の概念がいくつかあるが、入力と学習方式で棲み分けられる。

| タスク | 何を与えるか | 学習の要否 | 本 wiki のページ |
| --- | --- | --- | --- |
| **指示ベース編集** | 元画像＋**指示文** | 事前学習済みモデルをそのまま使う | 本ページ |
| inpainting（欠損補完） | 元画像＋**マスク**（＋テキスト） | 学習不要な手法もある | [[image-inpainting]] |
| 画像コンポジション | 元画像＋**参照物体画像**＋位置 | zero-shot | [[image-composition]] |
| personalization | 被写体の数枚の画像 | **被写体ごとに fine-tune** | [[subject-driven-generation]] |
| 空間条件付き生成 | エッジ・深度・姿勢マップ | アダプタを学習 | [[controllable-generation]] |

つまり指示編集は「**マスクも参照画像も被写体ごとの学習も要らず、文章だけで指定する**」点で最もユーザーに優しく、その分モデル側の負担が大きい。

## 代表手法：Qwen-Image の二重符号化（dual-encoding）

Qwen-Image（[[summaries/2025-qwen-image]]）の答えは、**入力画像を 2 通りに符号化して両方モデルに流す**というものである。

- **意味的表現（semantic features）**：入力画像を **Qwen2.5-VL（マルチモーダル LLM）** の ViT に通し、その特徴を**テキストストリーム**へ流す。「何が写っていて、指示は何をせよと言っているか」という高レベルの理解を担う。
- **再構成的表現（reconstructive features）**：同じ入力画像を **VAE エンコーダ**に通し、得た潜在をノイズ潜在とシーケンス方向に連結して**画像ストリーム**へ流す（[[latent-diffusion]]）。画素レベルの見た目・構造を担う。

著者らの観察では、**MLLM 側の特徴が指示追従を、VAE 側の特徴が視覚的忠実度と構造の一貫性を**担当する。つまり冒頭の「2 つの一貫性」に、**2 つの符号化経路をそれぞれ割り当てた**わけで、綱引きを設計で分業に変えている。

加えて、
- 位置符号化 **MSRoPE** に **frame 次元**を追加し、「編集前の画像」と「編集後の画像」を区別できるようにする。
- 学習では T2I・TI2I に加えて **I2I（画像再構成）** も混ぜ、Qwen2.5-VL と MMDiT の潜在表現を整合させる。

## 後継：生成と編集を「同じモデル・同じ学習」に統一する（2026）

初代 Qwen-Image は編集を **T2I・I2I・TI2I のマルチタスク学習**として扱ったが、後継の **Qwen-Image-2.0**（[[summaries/2026-qwen-image-2]]）はこれを推し進め、「**パイプラインを切り替えずに単一モデルで生成も編集もこなす（omni-capable）**」ことを主目的に据えた。著者らの問題意識は明快で、既存システムは「写実的な絵か正確な文字か」「生成か編集か」のどれか 1 軸でしか秀でず、全部を同時に出せるものがない、という点にある。

統一の方法は 2 つ。**(1) 入力表現の一本化**——初代の「MLLM 意味特徴と VAE 特徴を両方流す二重符号化」から、**画像側は VAE 潜在に一本化**し（Qwen3-VL の視覚出力は VAE 潜在で置き換えられる）、テキスト系列と連結して MMDiT に入れる。交互配置された複数画像入力も自然に扱える。**(2) 学習比率の制御**——事前学習では T2I:TI2I = 9:1 と生成寄りで基礎を作り、継続事前学習・SFT で 7:3 へ編集を厚くする。解像度カリキュラム（256/512 → 512/1024/2048）と同時に進行させる点が要点である。

事後学習でも編集専用の報酬が用意される（[[reinforcement-learning-for-diffusion]]）：**指示追従報酬**（指示された修正が正しく実行されたか）と**視覚的一貫性報酬**（未修正領域の幾何・位相・意味を保っているか）——本ページ冒頭の「2 つの一貫性」がそのまま報酬設計に反映された形である。

定性評価では、複数画像からの合成（猫＋別画像の帽子）、連鎖編集、古典中国詩の縦書き組版などで同一性と指示追従を同時に保てると報告される。ただし**本レポートは GEdit/ImgEdit のような定量ベンチマークを提示せず、LMArena の ELO と定性比較にとどまる**点は割り引いて読む必要がある。

## 「編集」として解ける意外なタスク

指示編集の枠組みは、一見別ジャンルに見えるタスクも飲み込む。Qwen-Image は次を**すべて TI2I として**扱う：

- **新視点合成**（「Turn right 90 degrees」）→ GSO で PSNR 15.11 と、専用 3D モデル CRM（15.93）に肉薄
- **深度推定・セグメンテーション・検出・canny 推定・超解像**（[[super-resolution]]）→ 深度は SFT のみで DepthAnything 級に接近

著者らはこれを「**生成的理解（generative understanding）**」と呼ぶ。従来のエキスパートモデルが入力→出力を直接写像する識別的理解であるのに対し、生成モデルはまず視覚内容の分布を構成し、そこから深度などを自然に導く、という主張である。

## 評価

- **GEdit-Bench**：実世界のユーザー指示 11 カテゴリ。**Semantic Consistency（SQ）**・**Perceptual Quality（PQ）**・総合（O）を 0〜10 で採点（GPT-4.1 が審判）。英中の 2 版がある。
- **ImgEdit**：9 種の編集タスク・734 ケース。指示遵守・編集品質・詳細保持を 1〜5 で採点。
- 定性評価では、テキスト／材質編集・物体の追加削除置換・ポーズ操作・**連鎖編集（chained editing, 出力を次の編集の入力に使う）**・新視点合成が観点になる。

**結果**：Qwen-Image は GEdit で英中とも 1 位（EN 総合 7.56 / CN 7.52）、ImgEdit 総合 4.27 で 1 位（GPT Image 1 [High] が 4.20 で僅差 2 位）。注目すべきは **FLUX.1 Kontext [Pro] の GEdit-CN が 1.23** と壊滅している点で、指示編集の性能は**基盤モデルの言語能力**に強く依存することが露呈している。

## 限界

- **未編集領域の保持は手法により崩れる**。GPT Image 1 [High] は全体の一貫性維持にしばしば失敗し、多くのモデルで編集時に意図しないズームイン／ズームアウトが起きる。
- **連鎖編集で劣化が蓄積する**。何度も編集を重ねると構造的特徴が失われやすい（Qwen-Image は保てるが、FLUX.1 Kontext [Pro] は途中で指示を落とす）。
- **評価が審判 LLM 依存**（GEdit・ImgEdit とも GPT-4.1）で、審判モデルのバイアスが結果に混入しうる。
- 非写実的画像（イラスト・漫画）の編集はスタイル一貫性が崩れやすい。

## 既存知識との接続

- [[text-to-image-generation]]：指示編集は T2I 基盤モデルの下流応用。テキスト条件付けの仕組みをそのまま画像条件へ拡張する。
- [[latent-diffusion]]：VAE 潜在を条件として連結するのが視覚的一貫性の要。Qwen-Image の画像ストリームはここに乗る。
- [[image-inpainting]]：マスクで領域を明示する編集。指示編集は「どこを」もモデルに推論させる点で一段難しい。
- [[image-composition]]：参照物体を指定位置に合成する zero-shot 系（AnyDoor）。入力が画像である点で近いが、指示編集はテキストで操作を指定する。
- [[subject-driven-generation]]：被写体ごとに学習する personalization とは対照的に、指示編集は**追加学習なし**で任意の画像を編集する。
- [[visual-text-rendering]]：「看板の文字を書き換えて」は、編集の一貫性と文字描画能力の両方を要求する複合タスク。
- [[controllable-generation]]：ControlNet 的な空間条件付けと目的（制御性）を共有するが、条件の与え方が構造マップか自然言語かで異なる。

## 参考文献（summaries）

- [[summaries/2025-qwen-image]] — Qwen-Image（意味特徴＋再構成特徴の二重符号化、MSRoPE の frame 次元、GEdit/ImgEdit で首位。新視点合成・深度推定まで TI2I として統一）
- [[summaries/2026-qwen-image-2]] — Qwen-Image-2.0（生成と編集を単一モデルに統一。画像側を VAE 潜在へ一本化、T2I:TI2I 比率のカリキュラム、編集専用の 2 報酬で RLHF）
