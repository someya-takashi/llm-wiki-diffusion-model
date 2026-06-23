# Log — Diffusion Model LLM Wiki

時系列の append-only ログ。`## [YYYY-MM-DD] ingest | <タイトル>` 形式で追記する（CLAUDE.md §5）。
スキーマ変更は `## [YYYY-MM-DD] schema-update | <要点>` で記録する。

## [2026-06-24] ingest | SDXL: 高解像度画像合成のための潜在拡散モデルの改良

- 取り込み: `raw/papers/SDXL_ ...md`（ar5iv 由来 markdown, arXiv:2307.01952, 2023。Podell ら, Stability AI）。直前の ZipLoRA が base model として全面依存していた SDXL を取り込み、被参照ページを原典で裏付け。
- 作成: [[translations/2023-sdxl]], [[summaries/2023-sdxl]]（schema 規約どおり**専用概念ページは作らず** [[latent-diffusion]] の代表的後継モデルとして収録。アーキ詳細は [[diffusion-model-architecture]]。DiT と同じ扱い）
- 更新: [[concepts/latent-diffusion]]（「代表的後継モデル：SDXL」節を新設）, [[concepts/diffusion-model-architecture]]（ADM→SDXL→DiT の流れに UNet スケーリングを追記、SDXL は transformer 化を当時見送り）, [[concepts/controllable-generation]]（micro-conditioning＝学習時メタデータ条件付け）, [[concepts/text-to-image-generation]]（2 テキストエンコーダ＋pooled emb の高解像 T2I）, [[concepts/subject-driven-generation]]・[[concepts/low-rank-adaptation]]（既存 SDXL 言及を実リンク化）, [[overview]], [[index]]
- 画像: ar5iv 画像 16 枚（teaser＋Fig1–15。直 PNG x1/x2/x3 ＋ JPEG/JPEG 多数）を `raw/assets/2023-sdxl/` に保存。サブパス（`img/...`・`comp_old_model/sd1-5/`・`refiner_magic/` 等）を安定名に平坦化（`comp_catpoleon_row` 等、別名なので衝突なし）。全画像妥当・全取得成功。`sd-xl-vs.jpg`(Fig1) は実体がユーザー選好棒グラフ（右側の 2 段パイプライン図は別アセットでなく原図注釈）であるためキャプションで補記。説明図（Fig1/6）を要約に再掲。
- 翻訳: 本文 §1–3 ＋ **Appendix B–J 全訳**（A 謝辞・References 除外）。Tab.1/2/3 と App I の 40 行アスペクト比表は markdown 保持、**Alg.1（size/crop 条件付け擬似コード）と App J の Python コードはコードブロックで保持**（base64 data URI リンクは除外）。probability flow ODE/SDE・DSM・CFG（App C, EDM 流定式化）の数式は LaTeX 保持。図 16 枚を `<figure>`。
- メモ: 4 本柱＝(1) 3× UNet（2.6B）＋transformer block 不均一配分 `[0,2,10]`＋2 テキストエンコーダ＋pooled emb、(2) micro-conditioning（size/crop/aspect-ratio を Fourier 埋め込みで timestep emb に加算）、(3) multi-aspect training（bucket）、(4) 改良 VAE＋base/refiner の 2 段（SDEdit）。知見：人間評価で SOTA 級だが COCO zero-shot FID は悪化（指標の限界、付録F）。限界：手・concept bleeding・長文テキスト。未取り込み注記: EDM（Karras 2022, 連続時間）, SDEdit, StyleDrop, simple diffusion, offset-noise。

## [2026-06-24] ingest | ZipLoRA: 任意の被写体を任意のスタイルで（バッチ 2/2）

- 取り込み: `raw/papers/ZipLoRA_ ...md`（ar5iv 由来 markdown, arXiv:2311.13600, ECCV 2024。Shah ら, Google Research・UIUC）。Mix-of-Show→ZipLoRA バッチの 2 件目（完了）。
- 作成: [[translations/2024-ziplora]], [[summaries/2024-ziplora]]（概念は 1 件目で作った [[lora-merging]] に「(3) 学習係数マージ」として収録済み）
- 更新: [[concepts/lora-merging]]（ZipLoRA を学習係数マージのランドマークとして記述済み）, [[concepts/subject-driven-generation]]（content+style 分離学習→再結合）, [[concepts/latent-diffusion]]（SDXL 上で動作）, [[concepts/low-rank-adaptation]]（LoRA の疎性、1 件目で追記済み）, [[overview]], [[index]]
- 画像: ar5iv 画像 9 枚（PNG 5: x1–x5 ＋ JPEG 4: `figs/{fig3_final1,compare_main_new,recontext3,moe}.jpg`）を `raw/assets/2024-ziplora/` に保存。`figs/` を `figs_` 連結で平坦化。全 PNG/JPEG・全取得成功。説明図（x2 疎性・x4 手法概観）を Read で確認しキャプション作成。
- 翻訳: 本文 §1–5 全訳（**appendix なし**）。References・Acknowledgements 除外。HTML 表 1 個（Table 1 ユーザー選好）を markdown 化（`<math><semantics>…%` 残骸除去）。Table 2 は既に markdown。merger 係数・$\mathcal L_{merge}$（3 項）の数式保持。$\Delta W_m=m_c\otimes\Delta W_c+m_s\otimes\Delta W_s$（原典の式は $m_s\otimes W_s$ と表記ゆれ → 文脈上 $\Delta W_s$ として訳出）。
- メモ: 2 観察＝(1) LoRA の $\Delta W$ は疎（90% を 0 にしても品質維持）、(2) 列の cosine 類似度が高いと直和が破綻（signal interference）。解＝列ごとの学習係数 $m_c,m_s$ で個別 LoRA 挙動を保ちつつ列を直交化。被写体×画風の 2 LoRA 特化（複数前景は範囲外、LoRA-Composer と守備範囲が異なる）。lora-merging 概念で Mix-of-Show（gradient fusion）と対比整理。**バッチ 2 件完了**。未取り込み注記: StyleDrop, SDXL, DINO（評価特徴）, LoRAHub/MoLE（MoE 系 LoRA 合成）。

## [2026-06-24] ingest | Mix-of-Show: 分散型多概念カスタマイズ（バッチ 1/2）

- 取り込み: `raw/papers/Mix-of-Show_ ...md`（ar5iv 由来 markdown, arXiv:2305.18292, NeurIPS 2023。Gu ら, NUS Show Lab・Tencent ARC Lab）。Mix-of-Show→ZipLoRA の 2 件バッチの 1 件目。
- 作成: [[translations/2023-mix-of-show]], [[summaries/2023-mix-of-show]], [[concepts/lora-merging]]（**新規概念ページ**。複数 LoRA の重みマージ／融合を専門に扱う。multi-concept-customization の (a) 系統を細粒度化。ランドマーク＝Mix-of-Show・ZipLoRA）
- 更新: [[concepts/multi-concept-customization]]（(a) 重みマージを [[lora-merging]] へ委譲）, [[concepts/low-rank-adaptation]]（ED-LoRA・LoRA の疎性・[[lora-merging]] リンク）, [[concepts/controllable-generation]]（regionally controllable sampling を LoRA-Composer 領域注入の源流として）, [[overview]], [[index]]
- 画像: ar5iv 画像 11 枚（x1–x10 ＋ `imgs/mturk.png`）を `raw/assets/2023-mix-of-show/` に保存（`imgs/` を `imgs_mturk.png` に平坦化）。全 PNG・全取得成功。説明図（x4 パイプライン）を要約に再掲。
- 翻訳: 本文 §1–5 ＋ **Appendix §6 全訳**。References・Acknowledgements 除外。**HTML 表 8 個を markdown 化**（`<math><semantics>…` 残骸＝→ 等を除去、single→fused の矢印は「→」で表現、Table 3 は単一概念/融合 × 物体/キャラ/シーンの 6 サブ表に整理）。gradient fusion 目的・region-aware cross-attention の数式は LaTeX 保持。図 11 枚を `<figure>`。
- メモ: 課題＝concept conflict（embedding と LoRA 重みの役割未分離）と identity loss（重み平均が $\frac1n$ に薄める）。解＝ED-LoRA（$V=V_{rand}^+V_{class}^+$ で embedding に in-domain essence を残す）＋gradient fusion（$\arg\min_W\sum_i\|(W_0+\Delta W_i)X_i-WX_i\|_F^2$）。実写は Chilloutmix・アニメは Anything-v4 ベース。次は ZipLoRA を取り込み lora-merging に (c) 学習係数マージとして追記。

## [2026-06-24] ingest | Multi-LoRA Composition for Image Generation（バッチ 3/3）

- 取り込み: `raw/papers/Multi-LoRA Composition for Image Generation.md`（ar5iv 由来 markdown, arXiv:2402.16843, ICML 2024。Zhong ら, Microsoft）。3 件バッチの 3 件目（完了）。
- 作成: [[translations/2024-multi-lora-composition]], [[summaries/2024-multi-lora-composition]]（概念は 2 件目で作った [[multi-concept-customization]] に decoding-centric 系統として収録済み）
- 更新: [[concepts/multi-concept-customization]]（LoRA Switch/Composite を (c) 系統として記述済み）, [[concepts/classifier-free-guidance]]（LoRA Composite が CFG の多 LoRA 拡張）, [[overview]], [[index]]
- 画像: ar5iv 画像 9 枚（x1–x8 ＋ `Figure/merge_case.png`）を `raw/assets/2024-multi-lora-composition/` に保存。サブパス `Figure/` を平坦化。Table 1（画像 merge_case.png）は `<figure>` で引用。全 PNG・全取得成功。
- 翻訳: 本文 §1–5 ＋ **Appendix A 全訳**。References・Impact Statements 末尾の定型は本文扱いで訳出、References 一覧は除外。HTML 表 2 個（Table 2 人手評価・Table 3 ComposLoRA LoRA 一覧）を markdown 化（civitai リンク列は省略）。Table 4/5 は画像（merge_case.png）で原典が同一ファイルを指すため 訳注 で説明。LoRA Switch/Composite 式保持。**broken cross-ref（`LABEL:fig:result`/`fig:switch_step`/`fig:switch_order`）は訳注「図（結果）」で明示**（該当画像は markdown に含まれず）。
- メモ: バッチ 3 件完了。LoRA→LoRA-Composer→Multi-LoRA Composition の系譜を low-rank-adaptation／multi-concept-customization の 2 概念に整理。multi-concept は (a) 重みマージ、(b) 注意制御（LoRA-Composer）、(c) decoding-centric（本論文）の 3 系統。未取り込み注記: LoRAHub, ZipLoRA（重みベース合成）, DPM-Solver++（サンプラー）。

## [2026-06-24] ingest | LoRA-Composer: 訓練不要の多概念カスタマイズ（バッチ 2/3）

- 取り込み: `raw/papers/LoRA-Composer_ ...md`（ar5iv 由来 markdown, arXiv:2403.11627, 2024。Yang ら）。3 件バッチの 2 件目。
- 作成: [[translations/2024-lora-composer]], [[summaries/2024-lora-composer]], [[concepts/multi-concept-customization]]（新規概念ページ。3 件目の Multi-LoRA Composition もここに収める）
- 更新: [[concepts/low-rank-adaptation]]（multi-concept への発展、既に相互リンク済み）, [[concepts/image-composition]]（AnyDoor 系 ↔ LoRA 系の対比）, [[concepts/controllable-generation]]（訓練不要の推論時合成）, [[index]]
- 画像: ar5iv 画像 11 枚（x1–x11）を `raw/assets/2024-lora-composer/` にローカル保存。全 PNG・全取得成功。
- 翻訳: 本文 §1–5 ＋ **Appendix 0.A–0.D 全訳**。References・Acknowledgements 除外。Table 1/2/3 は markdown（元から HTML table なし）。数式 LaTeX 保持（Region-Aware Injection・$\mathcal{L}_{ce}$/$\mathcal{L}_{fill}$/$\mathcal{L}_{region}$）。ar5iv 残骸除去。
- メモ: 新規 `multi-concept-customization` を作成し、(a) 重みマージ（LoRA Merge/Mix-of-Show）、(b) 訓練不要の注意制御（LoRA-Composer）、(c) decoding-centric（Multi-LoRA Composition）の 3 系統に整理。Mix-of-Show・Custom Diffusion・Cones は比較対象として言及（未取り込み）。

## [2026-06-24] ingest | LoRA: Low-Rank Adaptation of Large Language Models（バッチ 1/3）

- 取り込み: `raw/papers/LoRA_ Low-Rank Adaptation of Large Language Models.md`（ar5iv 由来 markdown, arXiv:2106.09685, ICLR 2022。Hu ら, Microsoft）。LoRA→LoRA-Composer→Multi-LoRA Composition の 3 件バッチの 1 件目。
- 作成: [[translations/2022-lora]], [[summaries/2022-lora]], [[concepts/low-rank-adaptation]]（新規概念ページ）
- 更新: [[concepts/subject-driven-generation]]（「LoRA 未取り込み」を解消、トレードオフ表に LoRA 追加）, [[concepts/latent-diffusion]]・[[concepts/diffusion-model-architecture]]（軽量 personalization としてクロスリンク）, [[index]]
- 画像: ar5iv 画像 8 枚（x1–x8）を `raw/assets/2022-lora/` にローカル保存。全 PNG・全取得成功。x1（再パラメータ化図）は Read で確認。
- 翻訳: 本文 §1–8 ＋ **Appendix A–H 全訳**（ユーザー確認）。References・Acknowledgements 除外。**HTML 表 14 個すべてを markdown 化**（ar5iv の `<math><semantics>…` 数式マークアップを除去し数値表に再構成。大型ハイパラ表 9/15 等は共通設定を見出しに畳んで整形）。数式 LaTeX 保持。
- メモ: LLM 論文だが拡散の軽量 personalization の基礎として取り込み。新規 `low-rank-adaptation` を作成し、DreamBooth（全層）↔ Textual Inversion（埋め込み）の中間に位置づけ。[[multi-concept-customization]] は 2 件目で作成（本エントリ時点では forward link）。未取り込み注記: HyperDreamBooth, Custom Diffusion, compacter（PEFT 後続）。

## [2026-06-24] ingest | DiT: Scalable Diffusion Models with Transformers

- 取り込み: `raw/papers/Scalable Diffusion Models with Transformers.md`（ar5iv 由来 markdown, arXiv:2212.09748, ICCV 2023。Peebles & Xie, UC Berkeley）
- 作成: [[translations/2023-dit]], [[summaries/2023-dit]]（新規概念ページは作らず）
- 更新: [[concepts/diffusion-model-architecture]]（**大幅拡充**：DiT を第 2 のランドマークとして追加、「改良 U-Net（ADM）→ Transformer 化（DiT）」の二系譜に再構成。patchify・adaLN-Zero・Gflops スケーリング・SOTA を記述、Fig3 引用）, [[concepts/latent-diffusion]]（DiT は LDM 潜在空間で backbone のみ Transformer 化）, [[concepts/denoising-diffusion]]（数学はそのまま backbone 差し替え）, [[concepts/classifier-free-guidance]]（DiT も CFG 使用・部分チャネル CFG）, [[concepts/text-to-image-generation]]（DiT backbone の将来展望）, [[overview]], [[index]]
- 画像: ar5iv 画像 33 枚（ユニーク）を `raw/assets/2023-dit/` にローカル保存。x1〜x13（説明・結果図 13 枚）＋ superimages 20 枚（無選別サンプル）。**`superimage-cfg-4.0-class-88.jpg` が 256/512 両ディレクトリに存在**するため、サブパスを `_` 連結で平坦化して衝突回避。全取得成功。説明図 x2/x3/x5/x8 は Read で内容確認のうえキャプション作成。
- 翻訳: 本文 §1–6 ＋ **Appendix A–D 全訳**（ユーザー確認済み。B のギャラリー Fig14–33 は図キャプションのみ）。References・Acknowledgements 除外。HTML `<table>` 4 個（Table 2/3/5/6）を markdown 化、Table 1/4 は markdown 保持。数式 LaTeX 保持（CFG 式・$L_\text{simple}$）。ar5iv math 残骸除去。
- メモ: **CLAUDE.md 明記「DiT は対応するアーキテクチャ概念ページの中で扱う」に従い、新規概念ページは作らず [[diffusion-model-architecture]] に追記**（ユーザー確認済み）。DiT は LDM 枠組み＋Stable Diffusion の VAE 潜在空間＋ADM の拡散ハイパラを流用し backbone だけ Transformer 化。dangling は `<slug>`（index テンプレ例）のみ想定。未取り込み注記: Stable Diffusion 3 / PixArt-α / Sora（DiT 後続）, ViT（基盤）, StyleGAN-XL（比較対象 GAN）。

## [2026-06-24] ingest | AnyDoor: Zero-shot Object-level Image Customization

- 取り込み: `raw/papers/AnyDoor_ Zero-shot Object-level Image Customization.md`（ar5iv 由来 markdown, arXiv:2307.09481, ICCV 2023。Chen ら, HKU / Alibaba / Ant）
- 作成: [[translations/2023-anydoor]], [[summaries/2023-anydoor]], [[concepts/image-composition]]（新規概念ページ）
- 更新: [[concepts/subject-driven-generation]]（zero-shot/参照ベースの対比軸として AnyDoor を追記）, [[concepts/controllable-generation]]（detail extractor の ControlNet スタイル・参照画像条件付け）, [[concepts/image-inpainting]]（ボックス領域を特定物体で再生成する対比）, [[concepts/latent-diffusion]]（SD を base にエンコーダ凍結・デコーダ学習）, [[overview]], [[index]]
- 画像: ar5iv 画像 11 枚（ユニーク x1〜x11）を `raw/assets/2023-anydoor/` にローカル保存。全 PNG・全取得成功。説明図 x2/x3/x4（パイプライン・注目領域・動画データ準備）は Read で内容確認のうえキャプション作成。
- 翻訳: 本文 §1–5（Abstract〜Conclusion）。**appendix なし**（§5 Conclusion 後は References）。References 除外。Table 1–5 は markdown 表を保持・整形（元から HTML table なし）。HF-map 式・injection 損失など数式 LaTeX 保持。ar5iv math 残骸なし（0）。
- メモ: ユーザー確認どおり新規概念ページ `image-composition`（画像コンポジション / object teleportation）を作成し、AnyDoor をランドマーク、Paint-by-Example・ObjectStitch を参照ベース同系として整理。古典的 image harmonization・[[subject-driven-generation]]（DreamBooth, tuning 型）・[[image-inpainting]] との違いを明示。AnyDoor の base は Stable Diffusion（[[summaries/2022-latent-diffusion]]）、detail extractor は ControlNet スタイル（[[summaries/2023-controlnet]]）、評価指標 DINO/CLIP-Score は DreamBooth（[[summaries/2023-dreambooth]]）由来。dangling は `<slug>`（index テンプレ例）のみ想定。未取り込み注記: Paint-by-Example / ObjectStitch / Graphit（参照ベース）, IP-Adapter（image-prompt アダプタ）, DINO-V2・SAM（基盤モデル）。

## [2026-06-24] ingest | DreamBooth: Subject-Driven Generation

- 取り込み: `raw/papers/DreamBooth_ Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation.md`（ar5iv 由来 markdown, arXiv:2208.12242, CVPR 2023。Ruiz ら, Google）
- 作成: [[translations/2023-dreambooth]], [[summaries/2023-dreambooth]], [[concepts/subject-driven-generation]]（新規概念ページ）
- 更新: [[concepts/text-to-image-generation]]（下流応用 personalization を追記、Imagen の言及も追加）, [[concepts/super-resolution]]（cascaded SR と DreamBooth の SR fine-tune＋低ノイズ増強）, [[concepts/latent-diffusion]]（SD を personalize する応用）, [[concepts/denoising-diffusion]]（拡散損失＋prior 保存項の fine-tune）, [[overview]], [[index]]
- 画像: ar5iv 画像 23 枚（ユニーク）を `raw/assets/2023-dreambooth/` にローカル保存（`x1`〜`x18.png` はそのまま、`figures/<name>.png` は `figures_<name>.png` に平坦化。basename 衝突なし）。全 PNG・全取得成功。説明図 x2/x3/x5/dino_metric は Read で内容確認のうえキャプション作成。
- 翻訳: 本文 §1–6 ＋ **Supplementary Material 全訳**（ユーザー確認済み。Background / Dataset / Subject Fidelity Metrics / User Study / Additional Applications / Additional Experiments / Societal Impact）。References・Acknowledgement（§6）除外。Table 1–6 は markdown 表を保持・整形（元から HTML table なし）。数式 LaTeX 保持（式(1)・PPL 損失）、ar5iv 残骸（図キャプションの `\sim` 等）除去。
- メモ: ユーザー確認どおり新規概念ページ `subject-driven-generation`（被写体駆動生成 / personalization）を作成し、DreamBooth（全層 fine-tune＋PPL＋rare-token id）と Textual Inversion（凍結モデルの token 埋め込み学習）を二大ランドマークとして対比。DreamBooth が prior に使うのは Imagen（T5-XXL＋cascaded diffusion）と Stable Diffusion（[[summaries/2022-latent-diffusion]]）。dangling は `<slug>`（index テンプレ例）のみ想定。未取り込み注記: Textual Inversion（Gal ら）, LoRA, Custom Diffusion（personalization 後続）, Imagen（専用原典）。

## [2026-06-24] ingest | RePaint: Inpainting using DDPM

- 取り込み: `raw/papers/RePaint_ Inpainting using Denoising Diffusion Probabilistic Models.md`（ar5iv 由来 markdown, arXiv:2201.09865, CVPR 2022。Lugmayr ら, ETH Zürich）
- 作成: [[translations/2022-repaint]], [[summaries/2022-repaint]], [[concepts/training-free-conditioning]]（新規概念ページ）
- 更新: [[concepts/image-inpainting]]（RePaint をランドマークとして追記、学習型 LDM ↔ 学習不要 RePaint の 2 系統に整理）, [[concepts/controllable-generation]]（推論時条件付けの (1b) 置き換え／射影ベースを追記）, [[concepts/diffusion-sampling]]（resampling＝調和のためのサンプリング時戦略、slowing down との違い）, [[concepts/denoising-diffusion]]（凍結 DDPM の推論時応用としてクロスリンク）, [[overview]], [[index]]
- 画像: ar5iv 画像 21 枚（ユニーク）を `raw/assets/2022-repaint/` にローカル保存。`figures/` と `supplement/figures/` に同名 basename（paper_c256_test_thick 等）があるため、サブパスを `_` 連結でファイル名に平坦化して衝突回避。全取得成功。説明図 x1/x2/x3（図1/2/9）は Read で内容確認のうえキャプション作成。
- 翻訳: 本文§1–8 ＋ **Appendix A–I を全訳**（ユーザー確認済み）。References・Acknowledgements 除外。HTML `<table>` 6 個（Table1 ×2・3・5・6 ほか）を markdown 化。Algorithm 1 を擬似コードブロックで保持。**Figure 8/10 の文字化け Python 擬似コード（ar5iv で `¿`/`¡` に化け）を正しいコードブロックに再構成**。数式 LaTeX 保持、ar5iv math 残骸除去。
- メモ: ユーザー確認どおり新規概念ページ `training-free-conditioning`（学習不要・推論時の条件付け）を作成し、RePaint をランドマークに。controllable-generation の「(1a) スコア勾配 guidance」「(2) アダプタ」に対し RePaint は「(1b) 置き換え／射影ベース」と整理。RePaint が prior に使う事前学習モデルは [[summaries/2021-adm]] の guided-diffusion。dangling は `<slug>`（index テンプレ例）のみ想定。未取り込み注記: ILVR / SDEdit / DDRM（関連する推論時条件付け）, Palette・GLIDE（学習型の画像条件付き拡散）。

## [2026-06-23] ingest | Diffusion Models Beat GANs on Image Synthesis (ADM)

- 取り込み: `raw/papers/Diffusion Models Beat GANs on Image Synthesis.md`（ar5iv 由来 markdown, arXiv:2105.05233, NeurIPS 2021。Dhariwal & Nichol, OpenAI）
- 作成: [[translations/2021-adm]], [[summaries/2021-adm]], [[concepts/diffusion-model-architecture]]（新規概念ページ）
- 更新: [[concepts/classifier-guidance]]（**主要原典として本格拡充・stub note 除去**：DDPM/DDIM 2 導出・勾配スケール s>1・ADM-G 成果・限界）, [[concepts/denoising-diffusion]]（ADM/IDDPM のアーキ改良・AdaGN）, [[concepts/diffusion-sampling]]（classifier-guided DDIM Algorithm 2・25 ステップ）, [[concepts/classifier-free-guidance]]（先行手法 ADM-G との対比）, [[concepts/controllable-generation]]（class-conditional のランドマーク＝ADM）, [[concepts/score-based-generative-models]]（ε＝スコア視点が DDIM guidance の根拠）, [[overview]], [[index]]
- 画像: ar5iv 画像 28 枚（ユニーク）を `raw/assets/2021-adm/` にローカル保存（`samples/` 配下はファイル名を `samples_<name>` に平坦化）。24 JPEG + 4 PNG、全取得成功。説明図 x1/x3/x6/x8（図2/4/5/12）は Read で内容確認のうえキャプション作成。
- 翻訳: 本文§1–8 ＋ **Appendix A–M を全訳**（ユーザー確認済み。K/L/M はサンプルギャラリーのため図キャプションのみ）。References・Acknowledgements 除外。HTML `<table>` 7 個（表1〜7,10,11〜16）を markdown テーブルに再構成、Algorithm 1/2 を擬似コードブロックで保持、数式 LaTeX 保持（H/B の導出も訳出）、ar5iv math 残骸除去。
- メモ: ユーザー確認どおり新規概念ページ `diffusion-model-architecture` を作成（U-Net 改良の系譜、ADM をランドマーク）。ADM 自体は methods/ ページを作らず diffusion-model-architecture＋classifier-guidance に分配。classifier-guidance の「未取り込み」注記を除去（本 ingest で解消）。dangling は `<slug>`（index テンプレ例）のみ。未取り込み注記: IDDPM（本文で言及・diffusion-model-architecture 内に記載）, BigGAN-deep（比較対象, GAN）。

## [2026-06-23] ingest | Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)

- 取り込み: `raw/papers/Adding Conditional Control to Text-to-Image Diffusion Models.md`（ar5iv 由来 markdown, arXiv:2302.05543, ICCV 2023 Marr Prize/Best Paper）
- 作成: [[translations/2023-controlnet]], [[summaries/2023-controlnet]]
- 更新: [[concepts/controllable-generation]]（ControlNet をアダプタ型アプローチとして大幅追記、「スコア操作の逆問題」と「アダプタ／FT」の 2 系統に再構成）, [[concepts/latent-diffusion]]（SD への空間条件追加）, [[concepts/text-to-image-generation]]（空間制御の補完）, [[concepts/classifier-free-guidance]]（CFG-RW）, [[overview]], [[index]]
- 画像: ar5iv 画像 12 枚（ユニーク）を `raw/assets/2023-controlnet/` にローカル保存。全取得成功。
- 翻訳: 本文§1–5 を全訳（**appendix なし**。本文が参照する supplementary materials は原典 markdown に含まれず）。References 除外。Table 1/2/3 を markdown テーブルに再構成。数式 LaTeX 保持＋主要式 (1)–(5)＋CFG 式を保持。ar5iv アーティファクト除去。
- メモ: ユーザー確認どおり ControlNet は新規ページを作らず controllable-generation 内にランドマーク手法として記述。dangling は `<slug>`（index テンプレ例）のみ。未取り込み注記: T2I-Adapter / LoRA / IP-Adapter（アダプタ系）, ADM（classifier-guidance 主要原典）, rectified flow 等（FM 後続）。

## [2026-06-23] ingest | Flow Matching for Generative Modeling

- 取り込み: `raw/papers/Flow Matching for Generative Modeling.md`（ar5iv 由来 markdown, arXiv:2210.02747, ICLR 2023）
- 作成: [[translations/2023-flow-matching]], [[summaries/2023-flow-matching]], [[concepts/flow-matching]]
- 更新: [[concepts/probability-flow-ode]]（CNF/FM の一般化視点・拡散 VF 一致）, [[concepts/score-based-generative-models]]（スコア回帰の VF 版・拡散パス内包）, [[concepts/denoising-diffusion]]（VP/VE パスは FM ガウス族の特別な場合）, [[concepts/diffusion-sampling]]（OT パスの少 NFE サンプリング）, [[concepts/super-resolution]]（FM-OT による超解像、SR3 比較）, [[overview]], [[index]]
- 画像: ar5iv 画像 16 枚（ユニーク）を `raw/assets/2023-flow-matching/` にローカル保存。`2d_vf_reference.png` は本文（図2）と付録 D で再利用＝1 ファイルを 2 箇所から参照。全取得成功。
- 翻訳: 本文§1–7＋**Appendix A–F を全訳**（ユーザー確認済み）。References・Acknowledgements は除外。Table 1（CIFAR-10/ImageNet）・2・3・4 を markdown テーブルに再構成。数式 LaTeX 保持＋主要式 (1)–(26) に \tag、付録の定理証明・CNF 尤度計算 ODE・拡散条件付き VF 導出も訳出。ar5iv アーティファクト除去。
- メモ: 概念ページは新規 flow-matching の 1 枚（ユーザー確認済み、CNF は flow-matching 内に前提として記述）。CLAUDE.md スコープに明記の「flow matching」を概念ページ化。dangling は `<slug>`（index テンプレ例）のみ。未取り込み注記: rectified flow / stochastic interpolants / mini-batch OT（FM 後続）, ADM（classifier-guidance 主要原典）。

## [2026-06-23] ingest | Score-Based Generative Modeling through SDEs (Score-SDE)

- 取り込み: `raw/papers/Score-Based Generative Modeling through Stochastic Differential Equations.md`（ar5iv 由来 markdown, arXiv:2011.13456, ICLR 2021 Outstanding Paper）
- 作成: [[translations/2021-score-sde]], [[summaries/2021-score-sde]], [[concepts/controllable-generation]]
- 更新: [[concepts/score-based-generative-models]]（SDE 統一枠組み＝VE/VP/sub-VP・逆時間 SDE を大幅拡充）, [[concepts/probability-flow-ode]]（主要原典として大幅拡充・「未取り込み」注記を除去）, [[concepts/diffusion-sampling]]（predictor-corrector・逆拡散・ODE サンプラー追記）, [[concepts/denoising-diffusion]]（DDPM=VP-SDE）, [[concepts/classifier-guidance]]（条件付き逆時間 SDE による一般化）, [[overview]], [[index]]
- 画像: ar5iv 画像 18 枚を `raw/assets/2021-score-sde/` にローカル保存（`figures/` 配下はフラット名）。全取得成功。Table 1/4/5 のテーブル埋め込み微小 SVG はテーブル再構成で吸収（独立図ではないため抽出せず）。
- 翻訳: 本文§1–6＋**Appendix A–I を全訳**（ユーザー確認済み、既存最大）。References・Acknowledgements は除外。Table 1/2/3/4/5 を markdown テーブルに再構成。数式 LaTeX 保持＋主要式 (1)–(17) に \tag、付録の Fokker-Planck 導出・PC アルゴリズム擬似コード・可制御生成アルゴリズムも訳出。ar5iv アーティファクト除去。
- メモ: 概念ページは新規 controllable-generation の 1 枚（ユーザー確認済み、SDE 枠組みは score-based-generative-models に統合）。**未取り込み注記のあった 2 本のうち Score-SDE を解消**。残る未取り込み主要原典は ADM（Dhariwal & Nichol, classifier-guidance の主要原典）。

## [2026-06-23] ingest | Classifier-Free Diffusion Guidance (CFG)

- 取り込み: `raw/papers/Classifier-Free Diffusion Guidance.md`（ar5iv 由来 markdown, arXiv:2207.12598, NeurIPS 2021 Workshop on DGMs）
- 作成: [[translations/2022-classifier-free-guidance]], [[summaries/2022-classifier-free-guidance]], [[concepts/classifier-free-guidance]], [[concepts/classifier-guidance]]
- 更新: [[concepts/text-to-image-generation]]（CFG リンク 2 箇所実在化）, [[concepts/latent-diffusion]]（同）, [[summaries/2022-latent-diffusion]]（「未作成」除去）, [[overview]], [[index]]
- 画像: ar5iv 画像 5 枚＋本文埋め込みインライン SVG 2 枚（Fig4・Fig5 の IS/FID 曲線）を `raw/assets/2022-classifier-free-guidance/` にローカル保存（計 7 ファイル）。全取得成功。
- 翻訳: 本文§1–6＋Appendix A を全訳（ユーザー確認済み）。References・Acknowledgements は除外。Table 1・2（HTML table）を markdown テーブルに再構成。数式 LaTeX 保持＋主要式（classifier guidance 式1・2、CFG 式6）に \tag。図1 は ar5iv に画像リンクが無くキャプションのみ＋訳注。
- メモ: 概念ページは classifier-free-guidance＋classifier-guidance の 2 枚（ユーザー確認済み）。**最後の dangling link `[[classifier-free-guidance]]` を解消**。classifier-guidance の主要原典（Dhariwal & Nichol, ADM 2021）は未取り込みで、ページ内に注記。これで本文中の主要 dangling は解消済み（probability-flow-ode 内の Score-SDE 言及は文中注記であり dangling リンクではない）。

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
