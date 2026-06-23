---
name: query
description: Diffusion Model LLM Wiki への質問応答。ユーザーが wiki の内容について質問してきたとき（手法の比較、概念の説明、原典をまたいだ分析など）に使う。index から関連ページを辿り、[[wikilink]] 引用付きで初学者にも分かるように回答し、必要なら成果物を questions/ として保存する。
tools: Read, Write, Edit, Bash, WebFetch, WebSearch
---

# query — 質問への回答

Diffusion Model wiki に対する質問に答える。回答の品質ルール（略称展開・難概念補足・既存知識との接続）は **CLAUDE.md §4「機械的なまとめにしない」を必ず参照**すること。これは素の回答にも常時適用する。

## 標準フロー

1. **index.md を読む**：まず `wiki/index.md` で関連しそうなページ（concepts / summaries / translations / overview 等）を特定する。
2. **必要なページを読む**：drill-in し、複数ページを統合して回答する。原典に立ち戻る必要があれば `raw/` を直接読んでよい。wiki に情報が無い／薄い場合は WebSearch・WebFetch で補ってよいが、その旨を明示する。
3. **回答する**：ユーザーの質問に対して、引用元ページを `[[wikilink]]` 形式で明示しながら回答する。**初学者にも分かるように略称展開・概念補足を入れる**（CLAUDE.md §4）。
4. **成果物の wiki 化を提案**：回答が比較表・分析・新しい接続の発見を含む場合、`wiki/questions/<slug>.md` として保存することをユーザーに提案する。ユーザーが了承したら作成し（frontmatter は CLAUDE.md §2 の questions/*.md を参照）、`index.md` と `log.md` を更新する。

## 出力形式

質問に応じて選ぶ：markdown ページ、比較表、Marp スライド、matplotlib チャート、canvas 等。比較・トレードオフを問う質問（例: 「DDPM と DDIM のサンプリング速度・品質の長短」「Stable Diffusion と Imagen の比較」）では表形式が有効なことが多い。

## questions ページ化の流れ

ユーザーが了承した場合：

1. `wiki/questions/<slug>.md` を作成（frontmatter に `type: question` / `asked` / `question` / `summaries_used` を付与）。
2. `wiki/index.md` の Questions セクションに 1 行追記。
3. `wiki/log.md` に `## [YYYY-MM-DD] query | <タイトル>` 形式で追記（取り込んだ／参照したページ、作成したページを箇条書き）。
