---
type: concept
aliases: [Classifier Guidance, 分類器ガイダンス, ADM guidance]
tags: [classifier-guidance, classifier-free-guidance, denoising-diffusion, generative-models]
related:
  - "[[classifier-free-guidance]]"
  - "[[controllable-generation]]"
  - "[[denoising-diffusion]]"
  - "[[diffusion-sampling]]"
  - "[[diffusion-model-architecture]]"
summaries:
  - "[[summaries/2021-adm]]"
  - "[[summaries/2022-classifier-free-guidance]]"
  - "[[summaries/2021-score-sde]]"
updated: 2026-06-23
---

# Classifier Guidance（分類器ガイダンス）

**Classifier Guidance（分類器ガイダンス）** とは、条件付き拡散モデルの生成時に、**別途学習した分類器の勾配**を拡散スコアに加えることで、条件への忠実度（fidelity）を高め多様性（diversity）を下げる手法である。**Dhariwal & Nichol の ADM 論文「Diffusion Models Beat GANs on Image Synthesis」（2021, NeurIPS）が導入した主要原典**で（[[summaries/2021-adm]]）、これにより拡散モデルは ImageNet で初めて GAN（BigGAN-deep）を画像品質で上回った。後続の [[classifier-free-guidance]]（分類器なしガイダンス, CFG）が分類器を不要にして置き換えたため、現在は「CFG の先行手法・対比対象」としても理解される。

## なぜ guidance が必要か——品質と多様性のトレードオフ

GAN には **truncation trick**（潜在ベクトルを切断正規分布から引き、多様性を下げる代わりに 1 枚ごとの品質を上げるノブ）があり、尤度ベースモデルには低温サンプリング（low-temperature sampling）があった。拡散モデルにはこれに相当する素直なノブがなく、スコアを単純にスケールしたり温度を下げたりしても**ぼやけた低品質画像**になるだけだった（ADM 付録 G が実際に温度法の失敗を示す）。classifier guidance は、このノブを拡散モデルに初めて与えた技術である。

## 仕組み——ADM による 2 つの導出

ノイズの乗った画像 $x_t$ 上で**専用の分類器 $p_\phi(y\mid x_t)$ を学習**する（ADM では U-Net のダウンサンプリング部＋8×8 アテンションプールを使い、拡散モデルと同じノイズ分布で学習）。この勾配 $\nabla_{x_t}\log p_\phi(y\mid x_t)$ で、サンプリングを任意のクラス $y$ に誘導する。導出はサンプラーによって 2 通り。

### 確率的サンプリング版（DDPM, Algorithm 1）

条件付き遷移は $p(x_t\mid x_{t+1},y)\propto p_\theta(x_t\mid x_{t+1})\,p_\phi(y\mid x_t)$ と書ける（厳密な証明は ADM 付録 H）。逆過程はガウス $p_\theta(x_t\mid x_{t+1})=\mathcal N(\mu,\Sigma)$ なので、$\log p_\phi(y\mid x_t)$ を $x_t=\mu$ まわりで 1 次 Taylor 展開すると、条件付き遷移は**平均を $\Sigma g$ だけずらしたガウス**で近似できる：

$$
p_\theta(x_t\mid x_{t+1})\,p_\phi(y\mid x_t)\approx \mathcal N\big(x_t;\ \mu+s\,\Sigma\, g,\ \Sigma\big),\qquad g=\nabla_{x_t}\log p_\phi(y\mid x_t)\big|_{x_t=\mu}
$$

つまり「いつものガウスから引く平均を、分類器勾配の方向に少しずらす」だけで条件付きサンプリングになる（$s$ は後述の勾配スケール）。この近似は「$\log p_\phi$ の曲率が $\Sigma^{-1}$ に比べ十分小さい」=「拡散ステップが十分多く $\|\Sigma\|\to0$」の極限で妥当。

### 決定論的（DDIM）版（Algorithm 2）

上の導出は確率的サンプリング専用で、決定論的な DDIM（[[diffusion-sampling]]）には使えない。そこで **ノイズ予測 $\epsilon_\theta$ がスコアのリスケールである**という関係（[[score-based-generative-models]]）

$$
\nabla_{x_t}\log p_\theta(x_t)=-\frac{1}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t)
$$

を使い、結合分布 $p(x_t)p_\phi(y\mid x_t)$ のスコアに対応する**修正ノイズ予測**

$$
\hat\epsilon(x_t)\coloneqq \epsilon_\theta(x_t)-\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p_\phi(y\mid x_t)
$$

を作り、これを通常の DDIM にそのまま差し込む。

## 勾配スケール $s$——ノブの正体

ADM は、実際にはスケール $s=1$ では「分類器の確信度は約 50% でも視覚的にはクラスに合っていない」ことを発見し、$s>1$ が必須だと示した。鍵は次の恒等式：

$$
s\cdot\nabla_x\log p(y\mid x)=\nabla_x\log\tfrac{1}{Z}p(y\mid x)^s
$$

$s>1$ は分類器分布 $p(y\mid x)^s$ を**鋭く**し、そのモードに集中させる。結果として **precision・IS↑／recall↓**（忠実度↑・多様性↓）という滑らかなトレードオフが得られる（FID・sFID は両者に依存するため中間の $s$ で最良）。これがまさに GAN の truncation に対応する「ノブ」である（コーギー例で scale 1.0→10.0 にすると FID 33→12）。ADM は guidance のフロンティアが、ある precision 閾値までは BigGAN-deep の truncation を上回ることも示した（[[summaries/2021-adm]] 図5）。すでにクラス条件付きのモデルにさらに guidance を当てる（ADM-G）と最良になる。

## 成果（ADM-G）

- ImageNet 128/256/512 で FID **2.97 / 4.59 / 7.72** を達成し、**BigGAN-deep（6.02 / 6.95 / 8.43）を上回る**。しかも recall（多様性）は GAN より高い。
- DDIM **25 ステップ**でも BigGAN 級の FID。
- アップサンプリング（ADM-U）と併用すると ImageNet 256→3.94、512→3.85（[[summaries/2021-adm]]）。

## 難点（CFG が解決した点）

- **分類器の追加学習が必要**：しかも**ノイズの乗った $x_t$** で学習せねばならず、既製の事前学習済み分類器を差し込めない。パイプラインが複雑化する。
- **敵対的攻撃への類似**：スコアに分類器勾配を混ぜる操作は、勾配で分類器を騙す敵対的攻撃に似る。このため「FID/IS が上がるのは分類器ベース指標を敵対的に騙しているだけでは？」という疑念を招く（ADM 著者は「スケールを 1 桁上げても敵対的サンプルにはならない」と反論）。

[[classifier-free-guidance]] はこれらを、分類器を排して生成モデル自身の条件付き／無条件スコアの差 $\epsilon_\theta(z,c)-\epsilon_\theta(z)$ で代替することで解決した。理論上、無条件モデルに重み $w{+}1$ の分類器ガイダンスを当てるのは条件付きモデルに重み $w$ を当てるのと等価、という関係が CFG の出発点になっている。

## 連続時間での一般化（Score-SDE）

分類器ガイダンスは、連続時間 SDE の枠組み（[[score-based-generative-models]]）における**条件付き逆時間 SDE の特別な場合**としても理解できる。Score-SDE（[[summaries/2021-score-sde]]）は、条件付きスコアが $\nabla_\mathbf{x}\log p_t(\mathbf{x}\mid\mathbf{y})=\nabla_\mathbf{x}\log p_t(\mathbf{x})+\nabla_\mathbf{x}\log p_t(\mathbf{y}\mid\mathbf{x})$ と分解できることを示した。$\mathbf{y}$ をクラスラベル、$p_t(\mathbf{y}\mid\mathbf{x})$ を time-dependent classifier とすれば、まさに分類器ガイダンス（尤度項の勾配を足す操作）になる。すなわち分類器ガイダンスは、Score-SDE が定式化した [[controllable-generation]]（可制御生成・逆問題）の一例である。実際 ADM の DDIM 版導出はこの Score-SDE のスコア視点をそのまま借用している。

## 既存知識との接続

- [[classifier-free-guidance]]：分類器を使わず同じ効果を達成し、分類器ガイダンスを置き換えた後継。両者の対比が CFG 論文の主眼。CFG は $w=0.3$ で ADM-G を上回ると報告。
- [[diffusion-model-architecture]]：ADM は guidance と同時に改良 U-Net（多解像度 attention・BigGAN resblock・AdaGN）を提案しており、両者あわせて「拡散が GAN を超えた」要因になった。
- [[controllable-generation]]：条件付き逆時間 SDE による生成。分類器ガイダンス（class-conditional）はその代表例で、Score-SDE がその一般形を与えた。
- [[denoising-diffusion]]：分類器ガイダンスは拡散スコア $\epsilon_\theta$ を修正して使う。
- [[diffusion-sampling]]：修正スコア $\hat\epsilon$／シフトした平均をサンプラー（DDPM・DDIM）に差し込んで生成する。

## 参考文献（summaries）

- [[summaries/2021-adm]] — Diffusion Models Beat GANs on Image Synthesis（ADM, 分類器ガイダンスの主要原典）
- [[summaries/2022-classifier-free-guidance]] — Classifier-Free Diffusion Guidance（§3.1 で classifier guidance を整理）
- [[summaries/2021-score-sde]] — Score-Based Generative Modeling through SDEs（条件付き逆時間 SDE による一般化）
