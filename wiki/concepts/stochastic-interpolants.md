---
type: concept
aliases: [Stochastic Interpolants, 確率的補間, interpolant, one-sided interpolant, mirror interpolant]
tags: [stochastic-interpolants, flow-matching, score-based-generative-models, probability-flow-ode, generative-models]
related:
  - "[[flow-matching]]"
  - "[[score-based-generative-models]]"
  - "[[probability-flow-ode]]"
  - "[[denoising-diffusion]]"
  - "[[diffusion-sampling]]"
summaries:
  - "[[summaries/2024-stochastic-interpolants]]"
updated: 2026-06-24
---

# Stochastic Interpolants（確率的補間）

**Stochastic Interpolants（確率的補間）** とは、**フローベース生成（決定論的 ODE）と拡散ベース生成（確率的 SDE）を 1 つの数学的枠組みに統一する**生成モデリングのパラダイムである。任意の 2 つの確率密度 $\rho_0$（基底）と $\rho_1$（目標）を**有限時間 $[0,1]$ で厳密に橋渡しする**連続時間確率過程

$$
x_t = I(t,x_0,x_1) + \gamma(t)z,\qquad t\in[0,1]
$$

を「補間（interpolant）」として明示的に作り（$I$ は $I(0,\cdot)=x_0,\ I(1,\cdot)=x_1$ の補間関数、$\gamma(0)=\gamma(1)=0$ で $\gamma(t)z$ が潜在ノイズ項、$z\sim\mathsf{N}(0,\mathrm{Id})$）、その時間依存密度 $\rho(t)$ を実現する ODE・SDE をデータから学んでサンプリングに使う。[[flow-matching]]（特に rectified flow）と [[score-based-generative-models]]（スコアベース拡散）の**双方を特別な場合として内包**する上位概念にあたる。本 wiki のランドマーク原典は **Stochastic Interpolants（Albergo–Boffi–Vanden-Eijnden, 2023）**（[[summaries/2024-stochastic-interpolants]]）。

## なぜ「統一枠組み」なのか

従来、生成モデルの 2 大系統は別々に論じられてきた。

- **拡散（確率的・SDE）**：[[score-based-generative-models]] / [[denoising-diffusion]]。データを OU 過程で標準ガウスへ劣化させ、逆時間 SDE で戻す。**片方がガウス必須・無限時間**（実際には truncate）という制約がある。
- **フロー（決定論的・ODE）**：[[flow-matching]] / 連続正規化フロー / [[probability-flow-ode]]。速度場を回帰し ODE で生成。厳密尤度・適応積分が使えるが、拡散と別物として扱われがちだった。

確率的補間は「**橋渡しする時間依存密度 $\rho(t)$ の設計**」と「**それを ODE でサンプルするか SDE でサンプルするか**」を**分離**する。鍵となる定理：同じ補間の密度 $\rho(t)$ は、

- **1 階輸送方程式** $\partial_t\rho+\nabla\cdot(b\rho)=0$ を満たす → 決定論的**確率フロー ODE** $\dot X_t=b(t,X_t)$ で実現。
- 拡散係数 $\epsilon(t)\geq0$ を**任意に選べる前向き・後ろ向き Fokker-Planck 方程式**を満たす → **SDE** $dX_t=(b+\epsilon s)\,dt+\sqrt{2\epsilon}\,dW_t$ で実現。

つまり ODE 生成と SDE 生成は「同じ密度の異なる実現」であり、**同じ速度 $b$ とスコア $s$ の推定**を共有する。$\epsilon=0$ ならフロー、$\epsilon>0$なら拡散——連続的に行き来できる。

## 仕組み：速度・スコアを二乗回帰で学ぶ

補間に入るドリフトはすべて条件付き期待値で、単純な二乗目的の唯一の最小化子として学習できる（[[flow-matching]] の CFM と同じ「シミュレーション不要」回帰）。

- **速度** $b(t,x)=\mathbb{E}[\dot x_t\mid x_t=x]$：目的 $\mathcal{L}_b[\hat b]=\int_0^1\mathbb{E}\big(\tfrac12|\hat b|^2-(\partial_t I+\dot\gamma z)\cdot\hat b\big)dt$。
- **スコア** $s(t,x)=\nabla\log\rho(t,x)=-\gamma^{-1}(t)\,\mathbb{E}(z\mid x_t=x)$：新しい二乗目的 $\mathcal{L}_s$ で学習（denoising score matching のベクトル場版）。
- **ノイズ除去器** $\eta_z=\mathbb{E}(z\mid x_t=x)$：$s=-\gamma^{-1}\eta_z$。端点で扱いやすい。

$(x_0,x_1)\sim\nu$（2 密度のカップリング）のサンプルから $x_t$ を任意時刻で生成でき、回帰だけで済む。

## ランドマーク：Stochastic Interpolants（Albergo–Boffi–Vanden-Eijnden 2023）

Albergo & Vanden-Eijnden（2022）の原型に、**潜在変数 $\gamma(t)z$** と **2 密度のカップリング $\nu$** を加えて一般化した（[[summaries/2024-stochastic-interpolants]]）。主な貢献：

- 補間の密度が Lebesgue 絶対連続で、TE と調整可能な FPE 族を満たすことを証明。
- 速度・スコア・ノイズ除去器を二乗回帰の最小化子として特徴づけ（スコアの新目的を含む）。
- **尤度制御の非対称性**：二乗目的の最小化は SDE モデルの尤度（KL）を制御するが、ODE モデルでは追加で Fisher ダイバージェンスの最小化が要る（より厳格）。→「近似が不完全なとき SDE の方が誤差に頑健」。
- **既存手法の内包**：スコアベース拡散（[[score-based-generative-models]]）は時間再パラメータ化で**片側補間**（$\rho_0$=ガウス）として書け、有限区間圧縮時の特異性を除去。rectified flow / flow matching（[[flow-matching]]）は特別な場合。補間 $I$ を最適化すると **Schrödinger bridge**（エントロピー正則化最適輸送）を復元する。

## 代表的なインスタンス

- **片側補間（one-sided interpolant）**：$\rho_0$ をガウスに取る従来設定。スコアベース拡散と接続。
- **ミラー補間（mirror interpolant）**：$\rho_0=\rho_1$。
- **rectified flow**：$x_t=(1-t)x_0+tx_1$（潜在変数なしの線形補間）。[[flow-matching]] の直線パスで、**SD3**（[[summaries/2024-sd3]]）が大規模 text-to-image で実用化した。

## 既存知識との接続

- [[flow-matching]]：flow matching・rectified flow は確率的補間の特別な場合（潜在変数なし・$\epsilon=0$ で ODE 生成）。確率的補間はそこに潜在変数・カップリング・調整可能拡散を加え、SDE 生成まで含めて一般化する。
- [[score-based-generative-models]]：SBDM は片側補間（ガウス基底）として再構成でき、無限時間・ガウス必須・特異性の制約を有限時間・任意基底で解消。スコアの二乗回帰目的も新規に与える。
- [[probability-flow-ode]]：輸送方程式から出る確率フロー ODE が決定論的生成。確率的補間は ODE（決定論）と SDE（確率）を同じ密度の異なる実現として統一する。
- [[denoising-diffusion]]：ノイズ除去器 $\eta_z$ とスコアの関係は拡散の denoising と整合。
- [[diffusion-sampling]]：$\epsilon(t)$ を選んで ODE/SDE のどちらでサンプルするかを後から決められる。少ステップ・誤差頑健性のトレードオフを与える。

## 参考文献（summaries）

- [[summaries/2024-stochastic-interpolants]] — Stochastic Interpolants: A Unifying Framework for Flows and Diffusions（Albergo, Boffi, Vanden-Eijnden, 2023）
- [[summaries/2024-sd3]] — SD3（rectified flow＝確率的補間の線形インスタンスを大規模実用化）
- [[summaries/2023-flow-matching]] — Flow Matching（特別な場合）
- [[summaries/2025-flow-matching-diffusion-intro]] — Flow Matching と拡散モデル入門（MIT 6.S184 講義ノート。ODE↔SDE 統一・conditional/marginal 構成・Langevin 拡張を入門的に解説）
