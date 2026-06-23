---
name: ingest
description: Diffusion Model LLM Wiki への原典取り込み。ユーザーが raw/ 配下に論文 PDF や記事 markdown を置いて「ingest して」「取り込んで」と指示したときに使う。原典を読解し、全文翻訳（translations）・要約ページ（summaries）・抽象概念ページ（concepts）を作成／更新し、index.md・log.md を更新する。翻訳・要約・画像の具体テンプレと書式もこの skill 内にある。
tools: Read, Write, Edit, Bash, WebFetch, WebSearch
---

# ingest — 原典の取り込み

Diffusion Model wiki に新しい原典を取り込む。スキーマ全体は `CLAUDE.md` を、frontmatter 規約は CLAUDE.md §2 を、横断的な品質ルール（略称展開・難概念補足・既存 wiki との接続など）は **CLAUDE.md §4「機械的なまとめにしない」を必ず参照**すること。本 skill は手順と、翻訳・要約・画像の具体テンプレを持つ。

## 標準フロー

ユーザーが `raw/` 配下に新しいファイルを置き「これを ingest して」と指示したときの標準フロー：

1. **読解**：原典を読み、主要な主張・手法・結果・前提・限界を把握する。画像リンクが埋め込まれている場合は、画像 URL も保持し、必要に応じて画像も参照する（後述「画像の扱い」を参照）。
2. **対話**：ユーザーに重要な発見・疑問点・既存 wiki との関係（既知概念との接続、矛盾、新概念の発生）を簡潔に伝え、強調すべき観点をすり合わせる。
3. **翻訳ファイルの作成**（**論文・記事の場合は必須**）：`wiki/translations/<slug>.md` を作成し、原典の本文を**一文ずつ正確に翻訳**する。詳細は下記「翻訳ファイルの仕様」を参照。
4. **要約ページの作成**：`wiki/summaries/<slug>.md` を作成する。詳細は下記「要約ページの仕様」を参照。**ここは機械的な要約にしない**。略称の定義、難概念の補足、初学者でも読める平易な説明を心がける（CLAUDE.md §4 参照）。
5. **概念ページの追加・更新**：原典で重要な役割を果たす項目について、`wiki/concepts/` にページを作るか既存ページを更新する（CLAUDE.md §1 の命名規約）。
   - **概念（concepts/）**：抽象レベルの概念・カテゴリ（例: `denoising-diffusion`, `score-based-generative-models`, `latent-diffusion`, `text-to-image-generation`, `classifier-free-guidance`, `diffusion-sampling`）。スラグは英語フルネーム。
   - **landmark 手法・モデルは専用ページを作らず、属する概念ページ内に「代表手法」として記述する**（例: DDPM・DDIM は `[[diffusion-sampling]]` や `[[denoising-diffusion]]`、Stable Diffusion / LDM は `[[latent-diffusion]]`、Imagen・GLIDE・DALL·E 2 は `[[text-to-image-generation]]`）。仕組み・実験・限界は概念ページ内の該当手法の項に、俯瞰と位置づけは概念ページ冒頭に書く。
   - **主要なベンチマーク・データセット（ImageNet, LAION-5B, COCO 等）や評価指標（FID, Inception Score 等）も専用ページは作らず**、それを評価に使う概念ページ、または該当する要約ページ内で説明・言及する。
   - 人物・組織の専用ページは作らない（関連 concept / summary 内で言及）。
   - **どのページも初学者向けの丁寧な解説を心がける**（CLAUDE.md §4）。
6. **クロスリファレンスの更新**：関連する既存ページの「関連項目」「参考文献」セクション、frontmatter の `summaries` / `related` を更新する。
7. **overview.md の更新**：必要に応じて拡散モデル全体の総括に変化を反映する（新潮流・パラダイムシフト等）。
8. **index.md の更新**：新規作成したすべてのページをカテゴリ別に追記する（CLAUDE.md §5）。
9. **log.md への追記**：`## [YYYY-MM-DD] ingest | <タイトル>` のフォーマットで、何を取り込み・どのページを作成/更新したかを箇条書きで記録する（CLAUDE.md §5）。

1 件の ingest で 5〜15 ページが触られるのが普通。基本は **1 件ずつ**取り込み、ユーザーと対話しながら進める。

---

## 翻訳ファイル（wiki/translations）の仕様

### 何を翻訳するか

- 論文・記事の**本文**（abstract, introduction, related work, method, experiments, results, discussion, conclusion 等）
- **appendix（付録）はデフォルトで翻訳に含める**。本文と同じ方針で一文ずつ正確に訳す。
- **除外（デフォルト）**：references（参考文献一覧）、acknowledgments（謝辞）
- **Web 記事／ブログ特有の除外**：本文でない定型要素も除外する。具体的には、購読/フォロー誘導ウィジェット（例: 「Get X's stories in your inbox」「Subscribe」）、著者プロフィール（Author Bio）、引用方法（Citation / BibTeX）、関連記事・次稿への誘導リンク、SNS 共有・フッターなど。**本文中に混入していても訳出しない**。
- **例外**：ユーザーが明示的に「appendix は除外して」「付録は訳さなくていい」等と指示した場合は、**appendix（付録）を翻訳対象から外す**。その場合も references・acknowledgments は除外したままにする。appendix を除外したかどうかは log.md のメモに残す。

### 翻訳の方針

- **一文ずつ正確に翻訳する。要約・省略・意訳的圧縮はしない**。原典の論理構造をそのまま保持する。
- 原典の見出し階層をそのまま保持する（`## 1. Introduction` 等）
- 数式は LaTeX 記法のまま残す（`$...$` `$$...$$`）。数式中の変数定義は翻訳の地の文で日本語化してよい。
- 図表のキャプションも翻訳する。図表番号（Figure 1, Table 2 等）は保持する。
- 用語は初出時に「ノイズスケジュール（noise schedule）」のように **日本語訳（原語）** の形で書き、以降は文脈に応じて選択する。
- 原典に画像リンクが埋め込まれている場合、画像はそのまま該当位置に `![](パス)` で挿入する（後述「画像の扱い」参照）。
- 翻訳ファイル内では、解釈や補足を加えない。**純粋に翻訳のみ**。補足説明は `wiki/summaries/<slug>.md` の方に書く。

### 翻訳ファイルのテンプレート

```markdown
---
type: translation
source_path: raw/papers/2020-ddpm.pdf
source_page: "[[summaries/2020-ddpm]]"   # frontmatter 内の wikilink は必ず引用符で囲む（CLAUDE.md §2 参照）
original_language: en
translated_to: ja
translated_at: 2026-06-03
---

# <原題（日本語訳）>

> 原題: <Original Title>
> 著者: ...
> 出典: ...

## Abstract（要旨）

<一文ずつの翻訳>

## 1. Introduction（はじめに）

<一文ずつの翻訳>

...
```

---

## 要約ページ（wiki/summaries）の仕様

要約ページは、原典を「読まなくても本質が掴め、かつ深く読みたくなったら原典に戻れる」状態を目指す。以下のセクションを基本構成とする（必要に応じて増減）。**CLAUDE.md §4 の品質ルールを必ず守る**：

```markdown
---
<frontmatter — CLAUDE.md §2 の summaries/*.md を参照>
---

# <タイトル>

> 原典: [[translations/2020-ddpm]] ・ `raw/papers/2020-ddpm.pdf`
> 著者・年・会議: ...

## 一言まとめ

（1〜2 文で「何をした論文／記事か」）

## 背景と問題意識

（なぜこの研究／記事が出てきたのか。前提知識・先行研究の何が不足していたのか。**初学者でも理解できるよう、関連する概念を平易な言葉で補足する**）

## 提案手法 / 主張

（何をどう変えたのか。図解の引用があると望ましい）

## 実験結果と知見

（何が示されたのか。数値・比較対象も含める）

## 限界・批判的視点

（原典の限界、後の研究による反証、未解決問題など）

## 用語と略称

（本ページに出てきた略称をすべて展開・定義する。例: DDPM = Denoising Diffusion Probabilistic Models、DDIM = Denoising Diffusion Implicit Models、LDM = Latent Diffusion Model、CFG = Classifier-Free Guidance、FID = Fréchet Inception Distance）

## 関連ページ

- [[concepts/...]]
- [[summaries/...]]
```

---

## 画像の扱い

### 原典の取得経路（背景）

原典は主に次の経路で `raw/` に置かれる。原典の種類によって画像の扱いが変わるので、下記 3 ケースに分ける。

- **arXiv 論文**：arXiv ページを ar5iv で HTML 化 → Obsidian Web Clipper で markdown 保存（`raw/articles/`）。→ **ケース A**
- **PDF 論文**：ar5iv が HTML 変換に失敗する論文は PDF で保存（`raw/papers/`）。→ **ケース B**
- **Web 記事・ブログ**（Medium、The Gradient、個人ブログ等の非 arXiv）：Web Clipper で markdown 保存（`raw/articles/`）。→ **ケース C**

**重要な前提**：Obsidian Web Clipper は、特に非 arXiv の Web 記事で**画像をうまく取り込めないことが多い**（本文中の図が markdown から欠落する、画像でない埋め込み＝動画リンク等に化ける、装飾画像やサイト UI 画像が混入する、CDN の変換 URL になる）。クリップ済み markdown の画像を鵜呑みにせず、後述ケース C の方針で**柔軟に**補う。

### ケース A: arXiv（ar5iv）由来の markdown 原典（`raw/articles/`）— 画像リンクあり

- **原典に画像リンク（外部 URL の `![](http...)`、data URI、SVG など）があれば、必ずローカルにダウンロードして保存する**。保存先は `raw/assets/<source-slug>/`。これは任意ではなく必須の手順とする。
  - ファイル名は **元アセット名を保持**（例: ar5iv の `x12.png`）するか、原典が連番でない・名前が衝突する場合は `figN.png` の連番にする。図番号が非連続な原典では元名保持の方が取り違えを防げる。
  - 外部 URL 画像は `curl`（UA 付き: `-A "Mozilla/5.0"`）/`WebFetch` 等で取得してバイナリ保存する。**ar5iv は `?as=webp` を外して素の PNG を取得**する。SVG はそのまま `.svg` で保存してよい（必要なら png に変換）。data URI（`data:image/...;base64,...`）は base64 をデコードして `figN.png` 等に書き出す。
  - 保存後は、要約・翻訳ページから **ローカルパス（`../../raw/assets/<source-slug>/<name>`）** を参照する。元 URL を残さない。
  - `file` コマンド等で全画像が妥当な画像データか確認し、参照枚数と保存枚数が一致することを確かめる。
  - **どうしても取得できなかった画像のみ**、例外として元 URL をそのまま使い、その旨を log.md のメモに「取得失敗: <URL>」として残す。
  - macOS の bash 3.2 では連想配列（`declare -A`）が使えない。図名を一時ファイルに書き出して `while IFS= read -r ...` で回すと、URL のループ取得が安定する。

### ケース B: PDF 原典（`raw/papers/`）— 画像はプログラム的に取得不可

- PDF からは画像をプログラム的に取り出せないため、**LLM 側からの画像の取り込みは行わない**。要約・翻訳ページは画像なしで作成してよい（図に言及する場合はキャプションのテキスト訳のみ残す）。
- **例外: ユーザーが画像ファイルパスを指示した場合**。ユーザーが手動で図を切り出してローカル保存し、そのファイルパスを指示してきたら、それらを使用する。
  - 指示された画像ファイルは、markdown 原典のケースと同様に **`raw/assets/<source-slug>/` に適切なフォルダを作って移動**し（連番 `figN.*` にリネームしてよい）、ページからは `../../raw/assets/<source-slug>/figN.*` のローカルパスで参照する。
  - 移動元のパスがリポジトリ外（例: ダウンロードフォルダ）でも、コピーまたは移動して `raw/assets/` 配下に集約する。

### ケース C: Web 記事・ブログ（`raw/articles/`、非 arXiv）— Web Clipper が不安定

Medium・The Gradient・個人ブログ等を Obsidian Web Clipper で保存した markdown は、**画像の取り込みが不完全なことを前提に、柔軟に補う**。基本方針はケース A（本文中の図はローカル保存必須・元 URL を残さない・`<figure>` で引用）と同じだが、加えて次を行う。

1. **クリップ済み markdown の画像をまず点検する**。本文の図への言及数（「下図」「Figure N」「次のプロット」等）と、実際に埋め込まれている `![](...)` の数が食い違っていないか確認する。大きく欠落していれば下記 2 で原ページから補う。
2. **図が欠落／不足していれば、原ページ（frontmatter の `source:` URL）から再取得する**。`curl -A "Mozilla/5.0" <URL>` で HTML を取得し、本文画像の URL を **document 順**で抽出する。各画像の直前テキスト（数百字）も併せて抜き出すと、原文の文脈に正しくマッピングして配置できる。`source:` URL を残さない（必ずローカル保存に置換）。
3. **コンテンツ図とサイト UI（chrome）を区別し、chrome は除外する**。除外対象の例：サイトのロゴ／ファビコン、トラッキングピクセル（例: `ghostboard.io`）、関連記事のサムネイル、SNS 共有ボタンの画像（例: `pinterest.com/pin/...`）、購読/フォローウィジェットの画像、装飾的なカバー／フィーチャー写真（記事内容と無関係なヘッダー写真）。判断に迷う装飾画像は基本除外し、本文の理解に必要な図（プロット・数式・図解）のみ保存する。
4. **CDN の変換 URL を素のオリジナルに直して取得する**。例：Medium の `https://miro.medium.com/v2/resize:fit:NNN/format:webp/<id>` は `format:webp/` を外すと原 PNG が得られることが多い。GitHub の `user-images.githubusercontent.com/...` は直リンクでそのまま取得できる。取得後 `file` で形式を確認し、PNG/JPEG が取れなければ webp 等そのままの拡張子で保存する。
5. **画像でない埋め込みは画像として保存しない**。`![](https://www.youtube.com/watch?v=...)` のような動画リンクや外部サービス埋め込みは、図ではないので DL せず、要約（summaries）に**参照リンク**として記載する。
6. **数式が画像化されている場合**（Medium 等は LaTeX を画像でレンダリングする）、その式画像をローカル保存して該当位置に配置し、`<figcaption>` に「訳注: 式(N)。〜の式」のように**何の式かを補足**する。読み取れる範囲で本文側に LaTeX を起こしてもよいが、画像と食い違う式を創作しない。
7. ファイル名は元アセット名（CDN の ID 等）を保持してよい。`*` など FS 上扱いにくい文字は除いた安定名にする。配置位置は 2 で得た document 順＋直前テキストで決め、`<figure>`＋`<figcaption>` で引用する。
8. log.md のメモに、**Web Clipper が図を取りこぼしたため原ページから再取得した**旨、除外した chrome、取得失敗があればその URL を残す。

> 補足: 上記はケース C の標準手順。ユーザーが「今回は画像不要」「このパスの画像を使って」等を指示した場合はそれに従う（ケース B の例外と同様、ユーザー指示が優先）。

### 共通

- **要約ページ（summaries）と翻訳ページ（translations）の両方で**、画像がある場合は必要な箇所に下記「キャプション書式」に従って引用する。
- **読解時の挙動**：LLM は markdown 中のインライン画像を 1 パスで読めないため、まず本文を読み、保存した画像を別途 Read ツールで開いて文脈を補強する（PDF は Read ツールでページ指定して読む）。
- 概念ページ（concepts）でもアーキテクチャ図などを引用してよい。引用元（どの source の図N か）を必ずキャプションに書く。

### キャプション書式（Obsidian 対応）

Obsidian は Markdown 標準の alt テキストをキャプションとして表示しない。CSS スニペット不要で Reading view にそのままキャプションを出すために、**HTML の `<figure>` + `<figcaption>` で Markdown 画像記法を包む方式**を採用する。

```markdown
<figure>

![](../../raw/assets/<source-slug>/figN.png)

<figcaption>図N: ここにキャプション本文。原典のキャプションを翻訳した内容、または要約／概念ページでの文脈付き説明。</figcaption>
</figure>
```

ルール：
- `<figure>` の**直後と `</figure>` の直前に空行を入れる**こと。これがないと Obsidian は中の Markdown 画像記法を解釈せず HTML ブロックとして閉じてしまう。
- `<figcaption>` の**前にも空行を入れる**。
- 画像パスは Markdown 記法 `![](path)` を維持する（Obsidian のパス解決が効くため）。`<img src="">` は使わない。
- `<figcaption>` 内では **`$...$` などのインライン LaTeX を使わない**。確実に動かないので、Unicode 添字（`x₁`, `x²` 等）や簡易表記（`k-NN`, `224^2`）で代替する。
- キャプション本文は原典のキャプションに合わせ「図N: ...」または「表N: ...」で始める。要約／概念ページで再掲する場合は「図1（再掲）: ...」のように明示する。
- `alt` テキスト（`![](path)` の `[]` 内）は空でよい。内容は `<figcaption>` 側に書く。

表（Markdown テーブル）には `<figure>` を使わず、テーブル直上に `**表N**: ...` の太字キャプションを置くだけにする（原典スタイルの踏襲）。`<figure>` は図画像にのみ使う。
