---
type: concept
aliases: [Character Consistency, キャラクタ一貫性, 同一性保持, Identity Preservation, Visual Drift, 視覚的ドリフト, 多ターン編集, Multi-turn Editing, CREF, Character Reference]
tags: [character-consistency, instruction-based-image-editing, subject-driven-generation, text-to-image-generation, generative-models]
related:
  - "[[instruction-based-image-editing]]"
  - "[[subject-driven-generation]]"
  - "[[image-composition]]"
  - "[[multi-concept-customization]]"
  - "[[text-to-image-generation]]"
summaries:
  - "[[summaries/2025-flux-kontext]]"
  - "[[summaries/2026-qwen-image-2]]"
  - "[[summaries/2023-dreambooth]]"
  - "[[summaries/2026-hidream-o1-image]]"
updated: 2026-08-17
---

# Character Consistency（キャラクタ一貫性 / 反復編集での同一性保持）

**Character Consistency（キャラクタ一貫性）** とは、**同じ人物・キャラクタ・物体を、複数の画像や複数回の編集をまたいで「同じもの」と分かる形で保ち続ける**能力、およびそれを測る評価軸を指す。1 枚を上手に作れることと、**その 1 枚を何度も編集しても本人のままでいられること**は別の難しさであり、後者が本ページの主題である。本 wiki のランドマークは **FLUX.1 Kontext**（[[summaries/2025-flux-kontext]]）で、反復編集での**視覚的ドリフト（visual drift）** を正面から問題として立て、顔埋め込みで定量化した。

## なぜ難しいのか — 誤差が蓄積する

単一の編集で同一性を保つのは、[[instruction-based-image-editing]] でいう「視覚的一貫性」の問題であり、指示された箇所以外を触らなければよい。しかし**反復編集（iterative / multi-turn editing）** では話が変わる。

編集を $n$ 回重ねると、各回のわずかなずれが**掛け算で積み上がる**。1 回あたり顔の特徴が 2% ずれるだけでも、10 回後には別人になっている。生成モデルは「もっともらしい画像」を作るよう学習されているので、**少しずつ「平均的な顔」へ引き寄せられる**傾向があり、このドリフトには方向性がある（個性が削れて一般的な顔立ちに近づく）。

実用上これは致命的になりうる。論文が挙げる例が具体的である：

- **マーケティング**：ブランドキャラクタが素材ごとに違う顔では使えない
- **メディア制作**：コマごとに主人公が変わる漫画やストーリーボードは成立しない
- **電子商取引**：商品の細部（ロゴ・縫い目・色味）が変わると別商品になる

## 2 つの系統 — 「学習して埋め込む」か「文脈として与える」か

同一性を保つアプローチは、大きく 2 つに分かれる。

| | やり方 | 追加学習 | 代表 |
| --- | --- | --- | --- |
| **学習型** | 被写体の数枚から重み／埋め込みを学ぶ | **必要**（被写体ごと） | DreamBooth・Textual Inversion・LoRA（[[subject-driven-generation]]） |
| **文脈型（in-context）** | 参照画像をそのまま入力に与える | **不要** | FLUX.1 Kontext の CREF・IP-Adapter |

[[subject-driven-generation]] が扱う学習型は、被写体ごとに fine-tune するぶん忠実度は高いが、**新しい人物ごとに数分〜数時間の学習が要る**。対して文脈型は参照画像を渡すだけで即座に効くので、対話的なワークフローに向く。FLUX.1 Kontext が採る **CREF（Character Reference, キャラクタ参照）** は後者で、参照画像のトークンを対象トークンに連結するだけ（[[instruction-based-image-editing]] の「系列連結」）で同一性を運ぶ。

## 代表手法：FLUX.1 Kontext（2025）

[[summaries/2025-flux-kontext]] はキャラクタ一貫性を**モデルの看板機能**に据えた。技術的な仕掛けは編集の枠組みそのもの（系列連結＋3D RoPE の仮想タイムステップ）で、一貫性のための特別な損失を足すわけではない。にもかかわらず一貫性が上がった、という報告である。

<figure>

![](../../raw/assets/2025-flux-kontext/evals_dustin_montage.jpg)

<figcaption>図12（引用, [[summaries/2025-flux-kontext]] より）: 同一の開始画像・同一のプロンプトで反復編集した比較（上: FLUX.1 Kontext、中: gpt-image-1、下: Runway Gen4）。下部は各ステップでの顔類似度スコア。最後の "Add sunglasses" は顔が隠れるため相対的な低下が予期される。</figcaption>
</figure>

**定量化の方法**が本ページにとって重要である。著者らは **AuraFace**（顔認識・同一性保持の埋め込みモデル）で編集前後の顔埋め込みを抽出し、**コサイン類似度**を測る。これを編集の各ステップで計算すれば、**ドリフトの速度**を曲線として描ける。人間評価とも整合したと報告されている。

「同一性が保たれているか」は本来かなり主観的だが、**顔埋め込みという既製の表現を借りて代理指標にする**というのがここでの解き方である。同じ発想は [[image-composition]] の AnyDoor が DINO-V2 の ID 特徴を使うのと通じる。

## 評価をどう設計するか

- **CREF タスク**：KontextBench（[[summaries/2025-flux-kontext]]）は 1,026 ペアのうち **193 例をキャラクタ参照**に割いている。
- **顔類似度の推移**：単一ターンのスコアではなく、**ターン数に対する曲線**で見るのが本質的。1 回目が良くても崩れ方が速ければ実用にならない。
- **編集専用の報酬**：Qwen-Image-2.0（[[summaries/2026-qwen-image-2]]）は事後学習で **視覚的一貫性報酬**（未修正領域の幾何・位相・意味を保っているか）を独立した報酬モデルとして立てている（[[reinforcement-learning-for-diffusion]]）。一貫性が「測って最適化する対象」になった例である。
- **顔以外**：物体・製品・スタイルの一貫性は顔ほど良い既製埋め込みがなく、評価が難しい領域として残る。ここを VLM 採点で埋めようとしたのが HiDream-O1-Image の **UniSubject**（[[summaries/2026-hidream-o1-image]]）で、Qwen-VL2.5-72B に「各参照画像と生成画像がペアごとにどれだけ同一被写体か」（Q-SC）を採点させる。顔埋め込みのような専用モデルに頼らず**任意の物体に適用できる**代わりに、採点者である VLM 自体の偏りを引き受けることになる。なお同ベンチマークは**参照数が増えたときの崩れ方**を測る点で、本ページの「ターン数に対する曲線」と同じ発想を空間方向に取ったものと言える（[[multi-concept-customization]]）。

## 限界と未解決問題

- **ドリフトは遅くなっただけで、消えていない**。FLUX.1 Kontext 自身が「**過度な複数ターン編集は視覚的アーティファクトを導入しうる**」と明記し、劣化の低減を今後の最重要課題に挙げる。「無限に流麗なコンテンツ制作」はまだ先である。
- **蒸留との衝突**：高速化のための蒸留（[[diffusion-distillation]]）自体がアーティファクトを生むため、速度と一貫性はトレードオフになりうる。
- **顔に偏った評価**：AuraFace 系の指標は人物には効くが、物体・キャラクタデザイン・ブランド要素には直接使えない。
- **意図した変化と劣化の区別**：「サングラスをかけて」のように顔が部分的に隠れる編集では類似度が下がるのが正しい。**指標の低下が失敗を意味するとは限らない**ため、解釈には注意が要る（論文もこの点を注記している）。

## 既存知識との接続

- [[instruction-based-image-editing]]：単一ターンの「視覚的一貫性」を、**複数ターンへ時間方向に拡張した**のが本ページ。編集の実用性を左右する軸。
- [[subject-driven-generation]]：同じ「特定対象の同一性保持」を目標とするが、あちらは**被写体ごとに学習**（DreamBooth・LoRA）、こちらは**参照画像を文脈として渡す**（CREF）。学習コストと即応性のトレードオフ。
- [[image-composition]]：AnyDoor が DINO-V2 の ID 特徴で物体の同一性を運ぶのは、CREF と同じ「既製の埋め込みを同一性の担い手にする」発想。
- [[multi-concept-customization]]：複数の被写体を 1 枚に入れるとき、各々の同一性を保つ必要がある点で隣接する。
- [[reinforcement-learning-for-diffusion]]：Qwen-Image-2.0 は視覚的一貫性を**報酬モデル**として立て、事後学習で直接最適化する。
- [[diffusion-distillation]]：高速化の代償としてのアーティファクトが一貫性を損ないうる。

## 参考文献（summaries）

- [[summaries/2025-flux-kontext]] — FLUX.1 Kontext（キャラクタ一貫性を看板機能に据え、AuraFace 埋め込みのコサイン類似度で反復編集のドリフトを定量化。CREF タスクを KontextBench に組み込む）
- [[summaries/2026-qwen-image-2]] — Qwen-Image-2.0（視覚的一貫性を独立した報酬モデルとして立て、RLHF で最適化する）
- [[summaries/2023-dreambooth]] — DreamBooth（被写体ごとの fine-tune で同一性を埋め込む学習型の代表。文脈型との対照）
- [[summaries/2026-hidream-o1-image]] — HiDream-O1-Image（UniSubject。VLM 採点で顔以外の被写体一貫性を測り、参照数に対する劣化曲線を見る）
