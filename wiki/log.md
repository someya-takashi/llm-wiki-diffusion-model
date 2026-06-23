# Log — Diffusion Model LLM Wiki

時系列の append-only ログ。`## [YYYY-MM-DD] ingest | <タイトル>` 形式で追記する（CLAUDE.md §5）。
スキーマ変更は `## [YYYY-MM-DD] schema-update | <要点>` で記録する。

## [2026-06-23] ingest | Denoising Diffusion Implicit Models (DDIM)

- 取り込み: `raw/papers/Denoising Diffusion Implicit Models.md`（ar5iv 由来 markdown, arXiv:2010.02502, ICLR 2021）
- 作成: [[translations/2021-ddim]], [[summaries/2021-ddim]], [[concepts/diffusion-sampling]], [[concepts/probability-flow-ode]]
- 更新: [[concepts/denoising-diffusion]]（diffusion-sampling リンク実在化）, [[concepts/score-based-generative-models]]（確率フロー ODE 追記）, [[summaries/2020-ddpm]]・[[summaries/2022-latent-diffusion]]（diffusion-sampling の「未作成」除去）, [[overview]], [[index]]
- 画像: ar5iv 画像 13 枚を `raw/assets/2021-ddim/` にローカル保存。`figures/` 配下はパス由来フラット名（figures_celeba-interp-line.png 等）。全取得成功。
- 翻訳: 本文§1–7＋Appendix A–D を全訳（ユーザー確認済み）。References・Acknowledgements は除外。Table 1（HTML table）と Table 3 を markdown テーブルに再構成。数式 LaTeX 保持＋主要式 (1)–(14) に \tag、Appendix の証明・式変形は省略せず全訳。ar5iv 見出しアーティファクト除去。
- メモ: 概念ページは diffusion-sampling＋probability-flow-ode の 2 枚（ユーザー確認済み）。**dangling link `[[diffusion-sampling]]` を解消**。残存 dangling（後続 ingest 予定）: [[classifier-free-guidance]]（Ho & Salimans）。probability-flow-ode の主要原典（Song ら Score-SDE）も未取り込みで、ページ内に注記。

## [2026-06-23] ingest | High-Resolution Image Synthesis with Latent Diffusion Models (LDM / Stable Diffusion)

- 取り込み: `raw/papers/High-Resolution Image Synthesis with Latent Diffusion Models.md`（ar5iv 由来 markdown, arXiv:2112.10752, CVPR 2022）
- 作成: [[translations/2022-latent-diffusion]], [[summaries/2022-latent-diffusion]], [[concepts/latent-diffusion]], [[concepts/text-to-image-generation]], [[concepts/image-inpainting]], [[concepts/super-resolution]]
- 更新: [[concepts/denoising-diffusion]]（latent-diffusion リンクを実在化）, [[overview]], [[index]]
- 画像: ar5iv 画像 33 枚を `raw/assets/2022-latent-diffusion/` にローカル保存。サブディレクトリ込みのパスを `_` 区切りのフラット名に変換し generic 名（sample_grid-0.jpg 等）の衝突を回避。全取得成功（取得失敗なし）。なお図18 は原典では画像比較表だが ar5iv に画像リンクが無く、翻訳では表＋キャプションのみ（画像なし）。
- 翻訳: 本文§1–6＋Appendix A–H を全訳（ユーザー確認済み）。References は除外。run-on 化していた表（Tab 1,2,3,5,7,8,9,10,11,18 等）は読みやすい markdown テーブルに再構成（数値は原典のまま、大規模ハイパラ表 12–17 は要点を散文＋抜粋整形）。ar5iv 見出しアーティファクト除去、数式 LaTeX 保持＋主要式 (1)–(3) に \tag。
- メモ: 概念ページは latent-diffusion＋text-to-image-generation＋image-inpainting＋super-resolution の 4 枚（ユーザー確認済み）。dangling link `[[latent-diffusion]]` を解消。残存 dangling（後続 ingest 予定）: [[diffusion-sampling]]（DDIM）, [[classifier-free-guidance]]（Ho & Salimans）。GLIDE/Imagen/DALL·E 2 は text-to-image-generation 内に枠だけ用意。

## [2026-06-23] ingest | Denoising Diffusion Probabilistic Models (DDPM)

- 取り込み: `raw/papers/Denoising Diffusion Probabilistic Models.md`（ar5iv 由来 markdown, arXiv:2006.11239, NeurIPS 2020）
- 作成: [[translations/2020-ddpm]], [[summaries/2020-ddpm]], [[concepts/denoising-diffusion]], [[concepts/score-based-generative-models]]
- 更新: [[overview]], [[index]]
- 画像: ar5iv 画像 16 枚＋本文埋め込みインライン SVG 2 枚（図5 レート歪み, 図10 漸進的品質）を `raw/assets/2020-ddpm/` にローカル保存（計 18 ファイル）。`?as=webp` 無し、元名保持。全て取得成功（取得失敗なし）。
- 翻訳: 本文＋Appendix A–D を全訳。References・Acknowledgments は除外。ar5iv 見出しの subscript アーティファクトはクリーンアップ。数式は LaTeX 保持＋式番号 (1)–(16) を \tag で復元。
- メモ: 概念ページは denoising-diffusion ＋ score-based-generative-models の 2 枚（ユーザー確認済み）。landmark 手法 DDPM は denoising-diffusion 内に詳述。未作成リンク（dangling）: [[diffusion-sampling]], [[latent-diffusion]], [[classifier-free-guidance]], [[text-to-image-generation]]（後続 ingest で作成予定）。

## [2026-06-23] schema-update | 3D Vision wiki を Diffusion Model wiki に再テーマ化

- 更新: `CLAUDE.md`, `.claude/skills/{ingest,query,lint}/SKILL.md`
- 領域定義を Diffusion Model（画像生成中心・広め）に差し替え
- `sources/` → `summaries/` に改称（`type`・クロスリファレンスフィールドも一貫変更）
- `methods/` `datasets/` を廃止し、landmark 手法・ベンチマークは概念ページ内に内包する方針に
- ingest: appendix をデフォルトで翻訳対象に含めるよう変更（除外指示がある場合のみ外す）
- 作成: `raw/{papers,articles,images,assets}/`, `wiki/{summaries,translations,concepts,questions}/`, `wiki/{index,log,overview}.md`
