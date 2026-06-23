# Diffusion Model LLM Wiki — スキーマ

このリポジトリは Andrej Karpathy が提唱した「LLM Wiki」パターンに基づく、**Diffusion Model（拡散モデル, データに徐々にノイズを加える順過程を学習し、それを逆向きにたどってノイズからデータを生成する生成モデルの一群）** 領域のパーソナル・ナレッジベースです。画像生成を中心に幅広く扱います。具体的には text-to-image 生成（テキスト条件からの画像生成）、画像編集・inpainting（欠損領域の補完）・super-resolution（超解像）、DDPM（Denoising Diffusion Probabilistic Models, ノイズ除去を繰り返してデータを生成する拡散モデルの基礎）や score-based generative models（スコアベース生成モデル）といった定式化、サンプラー／ソルバー（DDIM 等の高速サンプリング）、guidance（classifier-free guidance などの条件付け強化）、latent diffusion（潜在空間での拡散）、video / audio など他モダリティへの拡張、さらに SDE/ODE による連続時間定式化や flow matching といった理論的基盤までを含みます。LLM（あなた）が wiki を**読み・書き・更新する側**、ユーザーは**情報源のキュレーションと質問**を担当します。ユーザーは wiki を直接編集することはほぼありません。

このファイルは「あなた（LLM）」のための運用ルール書（スキーマ）です。ingest / query / lint の各オペレーションは skill として切り出してあり（§3）、それらの skill は本ファイルのスキーマ規約に従って実行してください。

---

## 1. ディレクトリ構成

```
Diffusion Model/
├── CLAUDE.md                ← このファイル（スキーマ）
├── .claude/skills/          ← オペレーション skill（ingest / query / lint）
├── raw/                     ← 原典（immutable, LLM は読むだけ）
│   ├── papers/              ← 論文 PDF（arXiv 等）
│   ├── articles/            ← ブログ記事・Webクリップ等の markdown
│   ├── images/              ← 記事に紐づかない単独画像
│   └── assets/              ← 記事内画像のローカル保存先
└── wiki/                    ← LLM が完全所有する markdown 群
    ├── index.md             ← カタログ（全ページの一覧）
    ├── log.md               ← 時系列ログ（append-only）
    ├── overview.md          ← Diffusion Model 全体の総括ページ（随時更新）
    ├── summaries/           ← 原典 1 件につき 1 ページの要約・解説
    ├── translations/        ← 原典の全文翻訳（要約せず一文ずつ正確に）
    ├── concepts/            ← 概念・カテゴリ（抽象レベル）の解説ページ
    └── questions/           ← query で得た成果物（比較表・分析等）を保存
```

### 命名規約

- ファイル名は `kebab-case.md`（例: `latent-diffusion.md`、`classifier-free-guidance.md`）
- 原典の wiki ページ（summaries/translations）は元ファイル名にあわせる（例: `raw/papers/2020-ddpm.pdf` → `wiki/summaries/2020-ddpm.md`, `wiki/translations/2020-ddpm.md`）
- **概念ページ（concepts/）は「抽象的な概念・カテゴリ」を単位にする**。スラグは概念の英語正式名称（略称ではなくフルネーム）。
  - 例: `denoising-diffusion`, `score-based-generative-models`, `latent-diffusion`, `text-to-image-generation`, `classifier-free-guidance`, `diffusion-sampling`, `noise-schedule`, `image-inpainting`, `video-diffusion`, `flow-matching`。
- **landmark な個別手法・モデルは専用ページを作らず、属する概念ページ内に「代表手法」として記述する**。
  - 例: DDPM・DDIM は `[[diffusion-sampling]]` や `[[denoising-diffusion]]`、Stable Diffusion / LDM は `[[latent-diffusion]]`、Imagen・GLIDE・DALL·E 2 は `[[text-to-image-generation]]`、DiT（Diffusion Transformer）は対応するアーキテクチャ概念ページの中で扱う。
  - 仕組み・実験・限界は概念ページ内の該当手法の項に、俯瞰と位置づけは概念ページ冒頭に書く。ある手法の言及が増えて 1 ページに収まらなくなったら、ユーザーに相談のうえ概念ページの分割や新設を検討する（§7）。
- **主要なベンチマーク・データセット（例: ImageNet, LAION-5B, COCO, CIFAR-10, FFHQ）や評価指標（FID, Inception Score, CLIP score 等）も専用ページは作らず**、それを評価に使う概念ページ、または該当する `summaries/` ページ内で説明・言及する。
- **個別の人物・組織の専用ページは作らない**。それらは関連する concept、または該当する `summaries/` ページ内で言及する。
- 略称（DDPM, DDIM, LDM, CFG, SDE, FID 等）は専用ページを作らず、対応する正式名称の概念ページ（denoising-diffusion, diffusion-sampling, latent-diffusion, classifier-free-guidance 等）にリダイレクトする位置づけで `index.md` に併記する。

### 言語ポリシー

- **wiki 内の解説文（summaries / concepts / questions / overview / index / log）は日本語**で書く
- **translations のみ、原典が英語であれば日本語訳**を作成する（原典が日本語であれば翻訳は作成しない）
- 固有名詞・術語は無理に和訳せず、初出時に「Denoising Diffusion Probabilistic Models（DDPM, データにノイズを段階的に加える順過程を逆転してノイズから画像を生成する拡散モデルの基礎手法）」のように原語＋略称＋短い注釈を添える

### リンク規約

- 内部リンクは Obsidian の `[[wikilink]]` 記法を使う（`[[latent-diffusion]]` のように slug を直接書く）
- まだ存在しないページへのリンク（dangling link）も許容する。`lint` 時に未作成ページとして検出する
- 原典への参照は `[[summaries/2020-ddpm]]` のようにディレクトリ込みで書く（同名スラグの衝突を避けるため）

---

## 2. Frontmatter 規約

すべての wiki ページに YAML frontmatter を付与する。Obsidian Dataview で集計できるようにする。

**重要（Obsidian 互換）**: frontmatter 内で `[[wikilink]]` を値に使うときは**必ず引用符で囲む**（`"[[...]]"`）。裸の `[[...]]` は YAML では入れ子配列と解釈され、Obsidian で「無効なプロパティ」になる。リンクが複数あるフィールド（`related` / `summaries` / `summaries_used`）は **YAML ブロックリスト**（各行 `  - "[[...]]"`）にする。単一リンク（`translation` / `source_page`）は `key: "[[...]]"` と引用符付きの 1 行で書く。本文（frontmatter 外）の `[[wikilink]]` は引用符不要（通常どおり）。

### summaries/*.md

```yaml
---
type: summary
source_path: raw/papers/2020-ddpm.pdf
source_kind: paper            # paper | article | blog | video | podcast
title: "Denoising Diffusion Probabilistic Models"
authors: [Jonathan Ho, Ajay Jain, Pieter Abbeel]
year: 2020
venue: NeurIPS 2020
ingested: 2026-06-03
tags: [denoising-diffusion, image-generation, generative-models, ddpm]
translation: "[[translations/2020-ddpm]]"
---
```

### translations/*.md

```yaml
---
type: translation
source_path: raw/papers/2020-ddpm.pdf
source_page: "[[summaries/2020-ddpm]]"
original_language: en
translated_to: ja
translated_at: 2026-06-03
---
```

### concepts/*.md

```yaml
---
type: concept
aliases: [LDM, Latent Diffusion]
tags: [latent-diffusion, text-to-image-generation, generative-models]
related:
  - "[[denoising-diffusion]]"
  - "[[text-to-image-generation]]"
summaries:                      # この概念の根拠となる原典（summaries/）
  - "[[summaries/2020-ddpm]]"
  - "[[summaries/2022-latent-diffusion]]"
updated: 2026-06-03
---
```

### questions/*.md

```yaml
---
type: question
asked: 2026-06-03
question: "DDPM と DDIM はサンプリング速度と生成品質でどう違うか？"
summaries_used:
  - "[[summaries/2020-ddpm]]"
  - "[[summaries/2021-ddim]]"
---
```

---

## 3. オペレーション（ingest / query / lint）

主要オペレーションは **skill として切り出してある**。実行時は対応する skill の手順に従うこと（手順の本体は各 SKILL.md にあり、本ファイルには重複させない）。

- **ingest（原典の取り込み）** — `.claude/skills/ingest/SKILL.md`
  raw/ に置かれた論文・記事を読み、翻訳ファイル（translations）・要約ページ（summaries）・関連する概念ページ（concepts）を作成／更新し、index/log を更新する。**翻訳・要約・画像の具体テンプレと書式もこの skill 内に置いてある**。
- **query（質問への回答）** — `.claude/skills/query/SKILL.md`
  index から関連ページを辿り、`[[wikilink]]` 引用付きで回答する。必要なら成果物を `questions/` として保存する。
- **lint（健康診断）** — `.claude/skills/lint/SKILL.md`
  矛盾・古い記述・孤立ページ・dangling link・欠落クロスリファレンス・概念⇔要約間のリンク不備・データギャップを点検し、一覧で提示する。

共通方針：

- ingest は基本 **1 件ずつ**、ユーザーと対話しながら進める（1 件の ingest で 3〜8 ページが触られるのが普通）。
- 下記 §4（「機械的なまとめ」にしないルール）は、`translations/` を除くすべてのページ生成、および query の回答に **常時適用**する。

---

## 4. 「機械的なまとめ」にしない（最重要ルール）

`wiki/translations/` 以外のすべてのページ（summaries / concepts / questions / overview）および query の回答で守ること：

1. **略称は必ず初出時に展開する**。`DDPM` → `DDPM（Denoising Diffusion Probabilistic Models, ノイズ除去を繰り返してデータを生成する拡散モデルの基礎）` のように、**展開＋短い意味付け**をセットで書く。拡散モデルで頻出する略称の例：DDPM, DDIM（Denoising Diffusion Implicit Models, 決定論的に少ステップでサンプリングする手法）, LDM（Latent Diffusion Model, 潜在空間で拡散を行うモデル）, CFG（Classifier-Free Guidance, 条件付き／無条件の予測を組み合わせて条件への忠実度を上げる手法）, SDE/ODE（Stochastic / Ordinary Differential Equation, 拡散過程を連続時間の確率／常微分方程式で記述する定式化）, ELBO/VLB（Evidence / Variational Lower Bound, 変分下界）, FID（Fréchet Inception Distance, 生成画像の品質を測る指標）, DiT（Diffusion Transformer, U-Net の代わりに Transformer を用いる拡散モデル）, VAE（Variational Autoencoder）, SNR（Signal-to-Noise Ratio, 信号対雑音比）。
2. **難概念は補足を入れる**。専門用語（例: noise schedule（ノイズスケジュール, 各ステップで加えるノイズ量の決め方）, 順過程／逆過程（forward / reverse process）, score function（スコア関数, データ対数密度の勾配 ∇ₓ log p(x)）, denoising（ノイズ除去）, variational lower bound（変分下界）, classifier-free guidance, latent space（潜在空間）, Langevin dynamics（ランジュバン動力学）, reparameterization（再パラメータ化）, ε-prediction / v-prediction（ノイズ／速度の予測パラメータ化））が出てきたら、その場で 1 文の補足説明を付ける。リンクで概念ページに飛べる場合でも、文脈で必要なら補足を残す。
3. **初学者の読者を想定する**。学部高学年〜修士 1 年程度の機械学習初学者が読んで「何のことを言っているか分からない」段落を作らない。逆に、自明な内容を冗長に説明するのも避ける。
4. **原典の章立てをそのままコピーしない**。要約ページの構造は ingest skill の要約テンプレートに従い、原典の構造を「再解釈」した形にする。
5. **「研究の意義」を自分の言葉で説明する**。原典の Abstract をなぞるだけにしない。「なぜこの結果が拡散モデル／生成モデル分野にとって重要なのか」を 1〜2 文加える。
6. **既存 wiki との接続を明示する**。「Stable Diffusion は [[latent-diffusion]] により [[denoising-diffusion]]（DDPM）が抱えていた高計算コストを潜在空間での拡散で解き、[[text-to-image-generation]] を実用的な解像度・速度に引き上げた手法である」のように、既存知識（概念・代表手法）と結びつける一文を必ず入れる。

逆に、`wiki/translations/` では上記ルールは **適用しない**。翻訳ファイルでは補足・解釈・接続づけは一切せず、原典に忠実に訳すことに専念する。

---

## 5. index.md と log.md の運用

### index.md

カテゴリ別の全ページカタログ。1 行 = 1 ページ。フォーマット：

```markdown
- [[<slug>]] — <一行の説明>
```

セクション：
- Overview
- Summaries（さらに paper / article 等に分けてよい）
- Translations
- Concepts
- Questions

略称リダイレクト（例: `DDPM → [[denoising-diffusion]]`、`LDM → [[latent-diffusion]]`、`CFG → [[classifier-free-guidance]]`）は対応するセクション内に併記してよい。ingest / query で新規ページを作るたびに必ず更新する。

### log.md

時系列の append-only ログ。

```markdown
## [YYYY-MM-DD] ingest | <タイトル>

- 取り込み: `raw/papers/2020-ddpm.pdf`
- 作成: [[summaries/2020-ddpm]], [[translations/2020-ddpm]], [[concepts/denoising-diffusion]]
- 更新: [[concepts/score-based-generative-models]], [[overview]], [[index]]
- メモ: ...
```

`grep "^## \[" log.md | tail -10` で直近の動きを追えるよう、必ずこのプレフィックス形式を守る。スキーマ変更は `## [YYYY-MM-DD] schema-update | <要点>` で記録する（§7）。

---

## 6. ツール

- **Obsidian**：wiki の閲覧・グラフビュー確認。ユーザーが裏側で開いている。
- **Obsidian Web Clipper**：記事を markdown 化して `raw/articles/` に保存。
- **Marp**：スライド出力が必要な質問への回答に使う（任意）。
- **Dataview**：frontmatter ベースの動的集計（任意）。

検索は規模が小さいうちは index.md ベースで十分。ページ数が増えてきたら `qmd` 等の導入を検討する。

---

## 7. このスキーマ自体について

このスキーマはユーザーと共進化する。運用していく中で「このカテゴリが必要」「このルールは緩めたい」となったら、ユーザーに提案してこの CLAUDE.md（およびオペレーション手順を持つ `.claude/skills/` 配下の SKILL.md）を更新する。スキーマ変更は log.md にも `## [YYYY-MM-DD] schema-update | <要点>` として記録する。
