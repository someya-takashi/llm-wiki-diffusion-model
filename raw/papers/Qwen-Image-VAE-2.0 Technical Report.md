---
title: "Qwen-Image-VAE-2.0 Technical Report"
source: "https://ar5iv.labs.arxiv.org/html/2605.13565"
author:
published:
created: 2026-08-05
description: "We present Qwen-Image-VAE-2.0, a suite of high-compression Variational Autoencoders (VAEs) that achieve significant advances in both reconstruction fidelity and diffusability111Diffusability describes how easily a dist…"
tags:
  - "clippings"
---
Qwen Team

###### Abstract

We present Qwen-Image-VAE-2.0, a suite of high-compression Variational Autoencoders (VAEs) that achieve significant advances in both reconstruction fidelity and diffusability <sup>1</sup>. To address the reconstruction bottlenecks of high compression, we adopt an improved architecture featuring Global Skip Connections (GSC) and expanded latent channels. Moreover, we scale training to billions of images and incorporate a synthetic rendering engine to improve performance in text-rich scenarios. To tackle the convergence challenges of high-dimensional latent space, we implement an enhanced semantic alignment strategy to make the latent space highly amenable to diffusion modeling. To optimize computational efficiency, we leverage an asymmetric and attention-free encoder-decoder backbone to minimize encoding overhead. We present a comprehensive evaluation of Qwen-Image-VAE-2.0 on public reconstruction benchmarks. To evaluate performance in text-rich scenarios, we propose OmniDoc-TokenBench, a new benchmark comprising a diverse collection of real-world documents coupled with specialized OCR-based evaluation metrics. Qwen-Image-VAE-2.0 achieves state-of-the-art reconstruction performance, demonstrating exceptional capabilities in both general domains and text-rich scenarios at high compression ratio. Furthermore, downstream DiT experiments reveal our models possess superior diffusability, significantly accelerating convergence compared to existing high-compression baselines. These establish Qwen-Image-VAE-2.0 as a leading model with high compression, superior reconstruction, and exceptional diffusability.

## Introduction

Latent Diffusion Models (LDMs) have become the dominant paradigm in image synthesis (rombach2022high; flux2024; qwenimage; sd3; seedream3; qwen-image-2.0). These models typically employ a Variational Autoencoder (VAE) to project images into a compressed latent space for efficient diffusion modeling, where a widely adopted spatial compression ratio is 8 (denoted as $f8$). However, as the industry shifts toward native high-resolution synthesis, this standard ratio faces significant computational bottlenecks. Increasing the spatial compression ratio has thus become essential for computational efficiency, as the complexity of modern Diffusion Transformers (DiTs) (dit) scales quadratically with the number of latent tokens. Over the past few years, several advances have been achieved in this field, demonstrating the significant potential of high-compression VAEs (ltx; dcae; cosmos; dcae1.5).

Despite these advances, a critical challenge exists: the inevitable trade-off between high compression ratio, reconstruction fidelity, and diffusability (diffusability). Specifically, higher compression ratios often lead to severe degradation in reconstruction quality, particularly in text-rich scenarios where fine-grained detail is lost. While increasing the latent channel dimension can mitigate this information bottleneck, it frequently results in an over-complex and unstructured latent distribution, which significantly hinders the convergence and generative performance of downstream diffusion models (vavae; qiu2025image).

In this work, we introduce Qwen-Image-VAE-2.0, a series of high-compression image VAEs ($f16$ & $f32$), designed to overcome these challenges through improved architecture, comprehensive data engineering, and enhanced training strategy.

To address the challenge of reconstruction fidelity in high-compression regimes, we adopt an improved VAE architecture with Global Skip Connection (GSC), which establishes a global shortcut from pixels to latents, preserving fine-grained detail. Moreover, our design incorporates a higher latent dimensionality to alleviate the information bottleneck inherent in high-compression scenarios. On the data front, we scale our training corpus to billions of images and curate a specialized document collection (including academic papers, posters, slides, web pages, etc.) to enhance the reconstruction of text-rich images. Furthermore, we develop a synthetic pipeline that renders documents to provide dense supervisory signals for character-level reconstruction. Through these advancements, Qwen-Image-VAE-2.0 achieves state-of-the-art reconstruction performance, especially in text-rich scenarios, despite its high compression ratio. To address the challenge of diffusability brought by high compression ratio and expanded latent dimension, we demonstrate that semantic alignment with intermediate features of DINOv2 (dinov2) can effectively accelerate DiT convergence. Furthermore, we adopt a staged semantic alignment paradigm that transitions from strict semantic alignment to a balanced optimization of reconstruction and generation. Leveraging these techniques, Qwen-Image-VAE-2.0 achieves superior diffusability compared to existing high-compression VAEs, despite its large channel dimension.

To ensure computational efficiency, we leverage an asymmetric architecture that features a lightweight encoder to minimize encoding overhead during diffusion training. Additionally, we utilize an attention-free backbone to maintain high throughput even with ultra-high-resolution inputs.

We conduct a comprehensive evaluation to validate the performance of Qwen-Image-VAE-2.0, focusing on two key aspects: reconstruction fidelity and latent space diffusability. For reconstruction fidelity, we evaluate it across general reconstruction tasks and introduce OmniDoc-TokenBench, a benchmark specifically targeting challenging scenarios like real-world text-rich document reconstruction. For latent space diffusability, we also perform extensive downstream DiT experiments to empirically verify it. The results demonstrate that Qwen-Image-VAE-2.0 not only achieves superior reconstruction fidelity, especially in text-rich scenarios despite high compression ratios, but also exhibits excellent latent space compatibility, facilitating rapid DiT convergence even with large latent dimension.

The key contributions of Qwen-Image-VAE-2.0 can be summarized as follows:

- High-Compression VAE: We introduce a suite of $f16$ and $f32$ image VAEs, providing a robust solution for efficient and native high-resolution image generation.
- Superior Reconstruction Performance: Qwen-Image-VAE-2.0 achieves state-of-the-art reconstruction performance across multiple benchmarks. It maintains exceptional legibility in text-rich scenarios where traditional high-compression models typically fail.
- Enhanced Latent Diffusability: Through a refined semantic alignment strategy, we demonstrate that large-channel VAEs can achieve excellent diffusability. This provides a promising solution to the tripartite trade-off between compression ratio, reconstruction fidelity, and diffusability.

## Model

In this section, we present the detailed design and architectural innovations of Qwen-Image-VAE-2.0.

### 2.1 Design Principle: High Compression VAE with Large Channel

To optimize the training efficiency of downstream DiTs, we prioritize a higher spatial compression ratio. Given an input image $I\in\mathbb{R}^{H\times W\times 3}$, the VAE maps it to a latent representation $z\in\mathbb{R}^{\frac{H}{f}\times\frac{W}{f}\times C}$, where $f$ denotes the spatial compression ratio and $C$ represents the channel dimension. This results in a sequence length of $L=HW/f^{2}$ for the DiT. Since DiT’s computational complexity scales quadratically with sequence length (attention), the computation overhead of $\mathcal{O}(L^{2})=\mathcal{O}(H^{2}W^{2}/{f^{4}})$ presents a significant bottleneck in high-resolution image generation.

To alleviate this, we move beyond the conventional $f8$ paradigm (rombach2022high; flux2024; wan2025wan), adopting higher compression ratios of $f16$ and $f32$ to significantly reduce DiT training costs. While high spatial compression ratio $f$ promises training efficiency, it inevitably constrains the information capacity of the latent space, often resulting in the loss of fine-grained structural detail. To mitigate this, we rely on the principle that reconstruction fidelity is largely governed by the total information bottleneck $N(z)={CHW}/{f^{2}}$ (dcae). By increasing the channel dimension $C$, we compensate for the spatial information loss incurred by high compression ratio $f$. Notably, channel expansion does not compromise DiT training efficiency: during training, the DiT first projects latents into a fixed hidden dimension via a linear layer, ensuring that the computational complexity remains nearly invariant to channel dimension.

### 2.2 Model Architecture

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2605.13565/assets/x1.png)

Figure 1: Comparison of No Skip Connection (NSC), Local Skip Connection (LSC), and Global Skip Connection (GSC) on model architecture, reconstruction loss and PSNR performance. S2C is the abbreviation of Space to Channel module. The experiments are conducted on f 16 c 64 f16c64 model training from scratch.

##### Global Skip Connection (GSC).

A primary challenge in high-compression VAEs is the preservation of fine-grained detail during the aggressive downsampling process. The encoder, particularly its non-parametric downsampling layers, often struggles to retain high-frequency information from the original image, leading to optimization difficulties and blurry reconstructions (dcae). To alleviate this information loss, we introduce the Global Skip Connection (GSC).

The GSC establishes a direct residual path that bypasses the initial downsampling stage, feeding pixel-level information directly into the deeper latent space. As illustrated in Figure 1, we implement this by employing a space-to-channel operation followed by reshaping, which effectively ”folds” spatial information from the input image into the channel dimension.

To validate the efficacy of this design, we conducted an ablation study comparing three configurations: No Skip Connection (NSC), Local Skip Connection (LSC), and Global Skip Connection (GSC). Experiments on an f16c64 model trained from scratch demonstrate that the GSC significantly accelerates convergence by providing the network with high-frequency signal from the input. Based on these findings, we integrate the GSC across the entire Qwen-Image-VAE-2.0 series.

##### Attention-Free Backbone.

For an input of sequence length $N$, the computational complexity of self-attention is $\mathcal{O}(N^{2})$, whereas that of convolution with a kernel size $k$ is $\mathcal{O}(N\cdot k^{2})$. This quadratic scaling with pixels creates a significant throughput bottleneck for high-resolution image processing. Moreover, the activation memory required for self-attention also scales as $\mathcal{O}(N^{2})$, imposing a severe memory constraint during training. In addition, we observed no significant performance degradation when removing attention modules. Consequently, we adopt an attention-free backbone for the entire Qwen-Image-VAE-2.0 series to ensure both training efficiency and scalability.

##### Encoder-Decoder Asymmetry.

We adopt an asymmetric architecture to balance encoding speed with reconstruction quality. By employing a lightweight encoder, we streamline the latent extraction process, effectively reducing training latency for the downstream DiT. Meanwhile, the heavyweight decoder guarantees high-fidelity reconstruction and the preservation of intricate image detail.

### 2.3 Model Configurations

The detailed configurations of the Qwen-Image-VAE-2.0 series are summarized in Table 1.

Table 1: Configurations of Qwen-Image-VAE-2.0 suite. $d_{enc}$ and $d_{dec}$ denote the first projected hidden dimensions of the encoder and decoder. $n_{\text{layer}}$ denotes the number of layers.

| Model | $f$ | $C$ | $d_{enc}$ | $d_{dec}$ | $n_{\text{layer}}$ | Residual | #Params (Enc/Dec) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen-Image-VAE-2.0-f16c64 | 16 | 64 | 96 | 144 | 5 | GSC | 76M / 248M |
| Qwen-Image-VAE-2.0-f16c128 | 16 | 128 | 96 | 144 | 5 | GSC | 76M / 248M |
| Qwen-Image-VAE-2.0-f32c128 | 32 | 128 | 96 | 144 | 6 | GSC | 77M / 250M |
| Qwen-Image-VAE-2.0-f32c192 | 32 | 192 | 96 | 144 | 6 | GSC | 78M / 250M |

## Data

### 3.1 Data Collection

##### Scaling Data to Billion Scale.

To ensure robust generalization across diverse domains, we scale the VAE training corpus to encompass billions of images. This large-scale dataset covers a wide spectrum of visual content, spanning various categories, resolutions and aspect ratios. However, data at this scale inevitably contains noise like edge blur and compression artifacts, which impedes model’s ability to learn high-frequency detail. To mitigate this, we employ clarity and blur filters to prune low-quality samples, ensuring that the VAE is supervised by high-fidelity signals.

##### Text-Rich Image Collection.

To address the reconstruction bottleneck in text-rich scenarios, we adopt a two-fold strategy. First, we leverage an OCR filter to identify and prioritize samples with high character density from extensive real-world datasets. Second, we curate a specialized document corpus, which includes screenshots of academic papers, presentation slides, posters, and complex web pages. By training on these real-world text-rich images, our models learn to prioritize the preservation of sharp edges of characters and semantic structures, enabling legible text reconstruction that is challenging for high compression VAEs.

### 3.2 Data Synthesis

To further enhance character-level reconstruction, we develop a synthetic pipeline that renders text documents into images. Our pipeline supports both alphabetic (English) and logographic (Chinese) languages, accounting for their distinct stroke densities and complexities. Crucially, we observed models trained on background-free synthetic data (e.g., black text on white backgrounds) generalize poorly to real-world images where text is often overlaid on complex textures. To bridge this gap, we implement background-contained synthesis, where text is rendered onto backgrounds randomly sampled from general-domain images. Moreover, to adapt different compression settings, we construct synthetic datasets of varying difficulty by rendering characters ranging from 5 to 20 pixels. This multi-granularity supervision forces the VAE to capture fine detail, ensuring legibility even at $f32$ compression.

## Training

### 4.1 Training Loss

The training objective of our image VAE is designed to be simple yet effective. Unlike traditional VAE frameworks that introduce distributional priors and adversarial paradigms, our training process focuses on high-fidelity reconstruction and semantic alignment of the latent space.

The total training loss $\mathcal{L}_{total}$ is formulated as:

$$
\mathcal{L}_{total}=\mathcal{L}_{recon}+\lambda_{lpips}\mathcal{L}_{lpips}+\lambda_{align}\mathcal{L}_{align},
$$

where $\mathcal{L}_{recon}$ is the pixel-level $L_{1}$ reconstruction loss, and $\mathcal{L}_{lpips}$ denotes the perceptual loss (lpips) weighted by $\lambda_{lpips}$. To improve the “diffusability” of the latent space, we incorporate a semantic alignment loss $\mathcal{L}_{align}$ which aligns latents to semantic counterparts extracted from pretrained encoders (detailed in Sec 4.2). This ensures that the latent space captures more generation-friendly features.

Our empirical findings suggest that two common practices in VAE training (KL regularization and adversarial training) can be removed to achieve better performance and training stability.

##### Removing KL Loss for Enhanced Semantic Alignment.

We remove the Kullback-Leibler (KL) divergence loss as it inherently restricts latent capacity and compromises reconstruction fidelity. More importantly, we observe that the KL penalty acts as a competing constraint to our semantic alignment objective. Given that target semantic features are not necessarily Gaussian-distributed, forcing the model to satisfy both a normal prior and a semantic manifold leads to suboptimal alignment, which ultimately delays the convergence of the downstream DiT. By removing this constraint, we achieve a more flexible latent space that is better suited for generative tasks.

##### Removing GAN Loss for Training Stability and Efficiency.

While GAN loss (gan) are conventionally used to sharpen visual detail, we find them unnecessary when the training budget is sufficiently large. Given extensive data and iterations, the combination of $\mathcal{L}_{recon}$ and $\mathcal{L}_{lpips}$ is capable of producing high-quality and sharp reconstructions. Eliminating the discriminator not only simplifies the optimization landscape, but also improves training stability and accelerates the overall training process.

In summary, by breaking the conventional reliance on KL loss and GAN loss, we demonstrate the feasibility and effectiveness of a simplified training objective, providing insights for future VAEs.

### 4.2 Semantic Alignment

Inspired by vavae, we introduce a semantic alignment loss to strike a delicate balance between low-level detail preservation and high-level semantics, thereby making the latent space more generation-friendly.

##### Selection of Semantic Encoder.

Through extensive ablation studies comparing various pretrained vision encoders (including DINOv2 (dinov2), DINOv3 (dinov3), MAE (mae), and PE-Spatial (bolya2025perception)), we find that DINOv2 consistently outperforms other candidates in providing generation-friendly semantic priors. Consequently, we select DINOv2-L features as our default semantic guidance.

##### Selection of Aligned Layer.

We observe that the choice of encoder layer affects the alignment results. While conventional approaches often utilize the final layer, we find that middle layer of these encoders offer smoother spatial maps that are easier to align with, yielding more generation-friendly latent space. Furthermore, we find that naively combining features from different layers introduces unnecessary noise that corrupts the alignment signal. Consequently, we align the VAE latent with a single, optimally selected middle layer, rather than relying on the final output or multi-layer fusion.

Specifically, given a target image, we first extract the semantic feature map $f\in\mathbb{R}^{h\times w\times c}$ using the pretrained semantic encoder, where $h$ and $w$ denote the spatial resolution and $c$ is the feature dimension. Then, we project the VAE latent $z$ into the same dimensionality through a learnable linear transformation, $z^{\prime}=Wz$, where $W$ is a trainable projection matrix. Let $\mathcal{P}=\{(i,j)\mid 1\leq i\leq h,\;1\leq j\leq w\}$ denote the set of spatial positions, where $|\mathcal{P}|=N=hw$. For each position $p\in\mathcal{P}$, we denote by $f_{p}\in\mathbb{R}^{c}$ and $z^{\prime}_{p}\in\mathbb{R}^{c}$ the semantic feature and projected latent feature at position $p$, respectively.

The alignment objective consists of two complementary components: (1) a Marginal Cosine Similarity Loss $\mathcal{L}_{mcos}$ with margin $m_{cos}$ which aligns the direction of VAE latents with target semantics, and (2) a Marginal Distance Matrix Similarity Loss $\mathcal{L}_{mdms}$ with margin $m_{dist}$, which preserves the relative spatial layout. The core alignment objectives are formulated as:

$$
\displaystyle\mathcal{L}_{mcos}(z^{\prime},f)
$$
 
$$
\displaystyle=\frac{1}{N}\sum_{p\in\mathcal{P}}\mathrm{ReLU}\left(1-\cos(z^{\prime}_{p},f_{p})-m_{cos}\right),
$$
$$
\displaystyle\mathcal{L}_{mdms}(z^{\prime},f)
$$
 
$$
\displaystyle=\frac{1}{N^{2}}\sum_{p\in\mathcal{P}}\sum_{q\in\mathcal{P}}\mathrm{ReLU}\left(\left|\cos(z^{\prime}_{p},z^{\prime}_{q})-\cos(f_{p},f_{q})\right|-m_{dist}\right),
$$
$$
\displaystyle\mathcal{L}_{align}(z,f)
$$
 
$$
\displaystyle=\mathcal{L}_{mcos}(z^{\prime},f)+\mathcal{L}_{mdms}(z^{\prime},f),
$$

where $\mathcal{P}$ denotes the set of spatial positions and $N=hw$ is the total number of elements.

### 4.3 Training Strategy

We employ a multi-stage training paradigm designed to progressively improve spatial resolution, refine textual rendering, and ensure robust semantic alignment.

##### Enhancing Resolution: From Low Resolution to High Resolution.

To facilitate stable training, we adopt a curriculum-based resolution strategy, starting from low-resolution foundations and progressively scaling up to 2K. Throughout this progression, we incorporate a diverse spectrum of aspect ratios to enhance the model’s geometric robustness, ensuring the VAE maintains structural integrity across various image compositions without distortion. This progressive upscaling allows the model to first learn basic structures and then capture finer detail and textures.

##### Integrating Textual Rendering: From Non-text to Text.

To master high-fidelity text reconstruction, we employ a multi-stage data infusion strategy. We start with general-domain images to accelerate initial convergence. Subsequently, we progressively incorporate real-world text-rich samples to address the challenges of complex character recognition. In the final phase, we introduce synthetic text data across varying difficulty levels to refine character precision. As general textures and character detail require different reconstruction focuses, we maintain a balanced ratio between these two types of data to ensure high quality for both.

##### Calibrating Semantic Alignment: From Strict Alignment to Balanced Reconstruction.

At the beginning of training, we apply strict semantic alignment using a strict margin ($m_{cos}$ and $m_{dist}$). We found that strong alignment at the early stage significantly helps the diffusability of the latent space. As the training progresses, we gradually loose the alignment margins. This allows the model to strike a better balance between maintaining semantic consistency and achieving high-quality pixel-level reconstruction.

## OmniDoc-TokenBench

### 5.1 Motivation

Standard reconstruction benchmarks such as ImageNet (deng2009imagenet) and FFHQ (Karras2018ASG) consist predominantly of natural photographs with negligible textual content, making them ill-suited for evaluating text-rich image reconstruction. Conventional pixel-level metrics (PSNR, SSIM) are inherently insensitive to text legibility, as minor stroke distortions may lead to a negligible decrease in conventional evaluation metrics yet render characters unrecognizable. While TokBench (Wu2025TokBenchEY) introduces OCR-based reconstruction evaluation, its data is drawn from scene text datasets where text instances are sparse and character sizes are insufficiently small, making it inadequate for benchmarking reconstruction capability in text-rich scenarios.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2605.13565/assets/x2.png)

Figure 2: OmniDoc-TokenBench, a curated collection of ∼ {\\sim} 3K text-rich images.

To address these limitations, we propose OmniDoc-TokenBench (Figure 2), a curated benchmark of ${\sim}$ 3K text-rich document images spanning nine categories—book, slides, color textbook, exam paper, academic paper, magazine, financial report, newspaper, and note—covering both alphabetic (English) and logographic (Chinese) text. We perform full-page OCR on both the original and reconstructed images and compute Normalized Edit Distance (NED) (Liu2019ICDAR2R; Marzal1993ComputationON) between their OCR outputs, directly measuring page-level document readability without requiring word-level bounding box annotations. This annotation-free design also facilitates easy scaling to new document types.

### 5.2 Benchmark Construction

OmniDoc-TokenBench is derived from OmniDocBench (Ouyang2024OmniDocBenchBD), a document parsing dataset offering fine-grained layout annotations and text-level ground truth across diverse sources. We construct the benchmark through a four-stage pipeline:

##### Text block extraction and font normalization.

Specifically, we first crop a region from the top-left corner of each text block and then resize it to $256\!\times\!256$ pixels so that each character occupies approximately $f_{\text{ref}}\!\times\!f_{\text{ref}}$ pixels. We set $f_{\text{ref}}\!=\!16$ for Chinese and $f_{\text{ref}}\!=\!10$ for English. These reference sizes are chosen empirically to provide a meaningful evaluation regime: the resulting inputs remain challenging for VAE reconstruction, especially in preserving fine stroke details, while standard OCR models still maintain high recognition accuracy.

##### Content filtering.

We apply PP-OCRv5 (cui2025paddleocr30technicalreport) to each sample, and retain only samples whose total count of recognized characters fall within $[200,600]$ (Chinese) or $[300,600]$ (English), ensuring sufficient textual density for reliable metric computation while excluding overly sparse or dense samples.

##### Deduplication.

We compute character-level $n$ -gram overlap ($n\!=\!3$ for Chinese, $n\!=\!5$ for English) between samples from the same source page and across the same document category. Pairs exceeding overlap thresholds of $0.2$ (intra-page) or $0.3$ (intra-category) are considered duplicates, among each overlapping group, only the sample with the highest character count is retained.

##### Human inspection.

To ensure data quality, we manually prune samples that are blurred, visually redundant, or contain significant blank regions. The final benchmark maintains a roughly balanced distribution between Chinese and English text.

### 5.3 Evaluation Methodology

Beyond standard reconstruction metrics (PSNR (psnr), SSIM (ssim), LPIPS (lpips), FID (fid)), we employ NED as the primary text-fidelity metric. A key design choice is using the OCR output of the *original* image as reference rather than the ground-truth annotations. Since OCR models introduce systematic errors even on clean inputs (e.g., confusing visually similar characters such as “rn” vs. “m”), comparing against annotations would falsely attribute such errors to the VAEs. By applying the same OCR model to both images, these biases largely cancel in the edit distance computation, isolating degradation caused solely by reconstruction.

Concretely, for each image $I_{i}$ in the benchmark, we apply PP-OCRv5 independently to the original image and its VAE reconstruction, yielding text strings $s_{\mathrm{gt}}^{(i)}$ and $s_{\mathrm{recon}}^{(i)}$, respectively. The benchmark-level NED is defined as:

$$
\mathrm{NED}=\frac{1}{N}\sum_{i=1}^{N}\left(1-\frac{d_{\mathrm{edit}}\!\bigl(s_{\mathrm{gt}}^{(i)},\,s_{\mathrm{recon}}^{(i)}\bigr)}{\max\!\bigl(|s_{\mathrm{gt}}^{(i)}|,\,|s_{\mathrm{recon}}^{(i)}|\bigr)}\right),
$$

where $d_{\mathrm{edit}}(\cdot,\cdot)$ denotes the Levenshtein distance, $N$ is the total number of benchmark images, and $|\cdot|$ denotes the string length. Each term measures character-level agreement for a single image, averaging over the benchmark yields a robust global estimate of text reconstruction fidelity.

## Experiments

Table 2: Comparison of different baselines against our Qwen-Image-VAE-2.0 models across various compression settings. Our models are highlighted in purple. Underline means second best score.

<table><tbody><tr><td rowspan="2">Baseline</td><td rowspan="2">Setting</td><td colspan="2">Generation(w/o CFG)</td><td colspan="2">Recon@Imagenet</td><td colspan="2">Recon@FFHQ</td></tr><tr><td>IS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>gFID <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>PSNR <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>SSIM <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PSNR <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>SSIM <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td colspan="8">ViT-backone AutoEncoders</td></tr><tr><td>VTP-Large <cite>(vtp)</cite></td><td>f16c64</td><td>146.22</td><td>5.25</td><td>26.88</td><td>0.7718</td><td>16.52</td><td>0.3129</td></tr><tr><td>RAE-DINOv2-B <cite>(rae)</cite></td><td>f16c768</td><td>139.80</td><td>6.64</td><td>19.24</td><td>0.5025</td><td>–</td><td>–</td></tr><tr><td>RAE-SigLIP2-B <cite>(rae)</cite></td><td>f16c768</td><td>103.24</td><td>11.58</td><td>19.71</td><td>0.5279</td><td>–</td><td>–</td></tr><tr><td colspan="8">f8 Compression VAEs</td></tr><tr><td>FLUX.1-dev <cite>(flux2024)</cite></td><td>f8c16</td><td>54.64</td><td>25.41</td><td>32.84</td><td>0.9155</td><td>38.14</td><td>0.9574</td></tr><tr><td>HunyuanVideo <cite>(hunyuanvideo)</cite></td><td>f8c16</td><td>63.57</td><td>21.29</td><td>33.21</td><td>0.9143</td><td>39.85</td><td>0.9607</td></tr><tr><td>Qwen-Image <cite>(qwenimage)</cite></td><td>f8c16</td><td>73.52</td><td>17.68</td><td>33.42</td><td>0.9159</td><td>38.75</td><td>0.9512</td></tr><tr><td>Wan2.1 <cite>(wan2025wan)</cite></td><td>f8c16</td><td>78.60</td><td>16.25</td><td>31.29</td><td>0.8870</td><td>38.16</td><td>0.9456</td></tr><tr><td>Cosmos-0.1-CI8x8 <cite>(cosmos)</cite></td><td>f8c16</td><td>52.89</td><td>26.02</td><td>32.33</td><td>0.9024</td><td>39.16</td><td>0.9546</td></tr><tr><td colspan="8">f16 Compression VAEs</td></tr><tr><td>Cosmos-0.1-CI16x16 <cite>(cosmos)</cite></td><td>f16c16</td><td>85.14</td><td>15.21</td><td>25.13</td><td>0.7015</td><td>30.91</td><td>0.8285</td></tr><tr><td>HunyuanVideo-1.5 <cite>(hunyuanvideo1.5)</cite></td><td>f16c32</td><td>69.59</td><td>19.08</td><td>31.18</td><td>0.8710</td><td>37.30</td><td>0.9336</td></tr><tr><td>HunyuanImage-3.0 <cite>(hunyuanimage3.0)</cite></td><td>f16c32</td><td>73.84</td><td>17.87</td><td>31.08</td><td>0.8655</td><td>36.85</td><td>0.9260</td></tr><tr><td>VAVAE <cite>(vavae)</cite></td><td>f16c32</td><td>129.80</td><td>6.03</td><td>27.75</td><td>0.7986</td><td>32.84</td><td>0.8752</td></tr><tr><td>Wan2.2 <cite>(wan2025wan)</cite></td><td>f16c48</td><td>79.55</td><td>15.65</td><td>31.30</td><td>0.8784</td><td>37.75</td><td>0.9386</td></tr><tr><td>Stepvideo-T2V <cite>(stepvideo)</cite></td><td>f16c64</td><td>45.18</td><td>33.53</td><td>31.54</td><td>0.8973</td><td>37.46</td><td>0.9451</td></tr><tr><td>FLUX.2-dev <cite>(flux2)</cite></td><td>f16c128</td><td>91.53</td><td>10.61</td><td>34.34</td><td>0.9358</td><td>40.36</td><td>0.9676</td></tr><tr><td>Qwen-Image-VAE-2.0-f16c64</td><td>f16c64</td><td>102.76</td><td>9.52</td><td>32.72</td><td>0.9086</td><td>39.14</td><td>0.9541</td></tr><tr><td>Qwen-Image-VAE-2.0-f16c128</td><td>f16c128</td><td>92.42</td><td>10.29</td><td>35.90</td><td>0.9519</td><td>43.10</td><td>0.9795</td></tr><tr><td colspan="8">f32 Compression VAEs</td></tr><tr><td>DC-AE-sana <cite>(dcae)</cite></td><td>f32c32</td><td>75.73</td><td>16.88</td><td>24.82</td><td>0.6897</td><td>31.35</td><td>0.8303</td></tr><tr><td>HunyuanImage-2.1 <cite>(hunyuanimage2.1)</cite></td><td>f32c64</td><td>47.96</td><td>33.32</td><td>28.67</td><td>0.8199</td><td>35.30</td><td>0.9110</td></tr><tr><td>LTX-Video <cite>(ltx)</cite></td><td>f32c128</td><td>33.48</td><td>44.94</td><td>29.57</td><td>0.8329</td><td>35.56</td><td>0.9051</td></tr><tr><td>LTX-2 <cite>(ltx2)</cite></td><td>f32c128</td><td>42.57</td><td>38.19</td><td>26.06</td><td>0.7925</td><td>33.63</td><td>0.9058</td></tr><tr><td>Qwen-Image-VAE-2.0-f32c128</td><td>f32c128</td><td>81.23</td><td>15.05</td><td>29.69</td><td>0.8423</td><td>35.91</td><td>0.9177</td></tr><tr><td>Qwen-Image-VAE-2.0-f32c192</td><td>f32c192</td><td>72.31</td><td>18.33</td><td>31.13</td><td>0.8785</td><td>37.52</td><td>0.9381</td></tr></tbody></table>

### 6.1 Quantitative Results

#### 6.1.1 Performance of Reconstruction

We evaluate the reconstruction performance of Qwen-Image-VAE-2.0 on two standard benchmarks: ImageNet (deng2009imagenet) and FFHQ (Karras2018ASG). Specifically, we evaluate low-resolution (256p) general-domain performance on ImageNet and high-resolution (1K) performance on FFHQ, using the Peak Signal-to-Noise Ratio (PSNR) and the Structural Similarity Index Measure (SSIM) as our quality metrics. As demonstrated in Table 2, Qwen-Image-VAE-2.0 achieves state-of-the-art reconstruction fidelity within its corresponding compression tiers ($f16$ and $f32$). Notably, our $f32c192$ VAE performs comparably to established $f8$ VAEs (e.g., Wan2.1), despite operating at a $4\times$ compression factor. This superior performance in reconstruction is largely attributable to our refined VAE architecture, expanded channel dimensions, and extensive training regimen.

Table 3: Comprehensive evaluation on OmniDoc-TokenBench (${\sim}$ 3K text-rich images, $256\!\times\!256$). Models are grouped by spatial compression factor and sorted by NED within each group. Our models are highlighted in purple.

<table><tbody><tr><td>Model</td><td>Setting</td><td>SSIM  <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PSNR  <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LPIPS  <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>FID  <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>NED  <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td colspan="7">ViT-backbone AutoEncoders</td></tr><tr><td>RAE-DINOv2-B <cite>(rae)</cite></td><td>f16c768</td><td>0.3261</td><td>14.32</td><td>0.2290</td><td>18.21</td><td>0.0392</td></tr><tr><td>RAE-SigLIP2-B <cite>(rae)</cite></td><td>f16c768</td><td>0.3871</td><td>14.36</td><td>0.1972</td><td>11.49</td><td>0.0483</td></tr><tr><td>VTP-Large <cite>(vtp)</cite></td><td>f16c64</td><td>0.7185</td><td>18.11</td><td>0.1046</td><td>3.94</td><td>0.4170</td></tr><tr><td colspan="7">f8 Compression VAEs</td></tr><tr><td>Wan2.1 <cite>(wan2025wan)</cite></td><td>f8c16</td><td>0.8282</td><td>20.54</td><td>0.0679</td><td>4.57</td><td>0.8021</td></tr><tr><td>Cosmos-0.1-CI8x8 <cite>(cosmos)</cite></td><td>f8c16</td><td>0.9057</td><td>24.29</td><td>0.0464</td><td>2.89</td><td>0.9033</td></tr><tr><td>Qwen-Image <cite>(qwenimage)</cite></td><td>f8c16</td><td>0.8998</td><td>24.94</td><td>0.0519</td><td>4.48</td><td>0.9073</td></tr><tr><td>HunyuanVideo <cite>(hunyuanvideo)</cite></td><td>f8c16</td><td>0.9227</td><td>25.26</td><td>0.0434</td><td>2.03</td><td>0.9266</td></tr><tr><td>FLUX.1-dev <cite>(flux2024)</cite></td><td>f8c16</td><td>0.9364</td><td>26.24</td><td>0.0246</td><td>0.55</td><td>0.9546</td></tr><tr><td colspan="7">f16 Compression VAEs</td></tr><tr><td>Cosmos-0.1-CI16x16 <cite>(cosmos)</cite></td><td>f16c16</td><td>0.5460</td><td>15.55</td><td>0.1349</td><td>7.78</td><td>0.1547</td></tr><tr><td>VAVAE <cite>(vavae)</cite></td><td>f16c32</td><td>0.6905</td><td>17.50</td><td>0.0974</td><td>4.45</td><td>0.3488</td></tr><tr><td>HunyuanVideo-1.5 <cite>(hunyuanvideo1.5)</cite></td><td>f16c32</td><td>0.8422</td><td>21.49</td><td>0.0839</td><td>4.67</td><td>0.6938</td></tr><tr><td>HunyuanImage-3.0 <cite>(hunyuanimage3.0)</cite></td><td>f16c32</td><td>0.8672</td><td>22.66</td><td>0.0650</td><td>3.49</td><td>0.7753</td></tr><tr><td>Wan2.2 <cite>(wan2025wan)</cite></td><td>f16c48</td><td>0.8577</td><td>21.67</td><td>0.0525</td><td>3.05</td><td>0.8310</td></tr><tr><td>Stepvideo-T2V <cite>(stepvideo)</cite></td><td>f16c64</td><td>0.8970</td><td>23.69</td><td>0.0650</td><td>6.02</td><td>0.8838</td></tr><tr><td>Qwen-Image-VAE-2.0-f16c64</td><td>f16c64</td><td>0.9279</td><td>26.00</td><td>0.0382</td><td>1.94</td><td>0.9244</td></tr><tr><td>FLUX.2-dev <cite>(flux2)</cite></td><td>f16c128</td><td>0.9544</td><td>27.72</td><td>0.0216</td><td>0.73</td><td>0.9535</td></tr><tr><td>Qwen-Image-VAE-2.0-f16c128</td><td>f16c128</td><td>0.9706</td><td>30.45</td><td>0.0167</td><td>0.79</td><td>0.9617</td></tr><tr><td colspan="7">f32 Compression VAEs</td></tr><tr><td>DC-AE-sana <cite>(dcae)</cite></td><td>f32c32</td><td>0.5259</td><td>15.62</td><td>0.1441</td><td>7.26</td><td>0.0692</td></tr><tr><td>LTX-2 <cite>(ltx2)</cite></td><td>f32c128</td><td>0.7354</td><td>18.41</td><td>0.1192</td><td>9.94</td><td>0.3569</td></tr><tr><td>HunyuanImage-2.1 <cite>(hunyuanimage2.1)</cite></td><td>f32c64</td><td>0.7805</td><td>19.85</td><td>0.0957</td><td>5.19</td><td>0.4895</td></tr><tr><td>LTX-Video <cite>(ltx)</cite></td><td>f32c128</td><td>0.8055</td><td>20.92</td><td>0.1190</td><td>17.10</td><td>0.5651</td></tr><tr><td>Qwen-Image-VAE-2.0-f32c128</td><td>f32c128</td><td>0.8442</td><td>22.13</td><td>0.0642</td><td>3.36</td><td>0.7065</td></tr><tr><td>Qwen-Image-VAE-2.0-f32c192</td><td>f32c192</td><td>0.8908</td><td>23.84</td><td>0.0497</td><td>1.98</td><td>0.8555</td></tr></tbody></table>

#### 6.1.2 Performance of Text Rendering

To assess reconstruction fidelity in challenging text-rich scenarios, we evaluate all models on OmniDoc-TokenBench (§5), reporting both traditional pixel-based metrics and the proposed OCR-based NED metric. We compute NED on raw OCR outputs without text normalization, while minor spacing artifacts from OCR may inflate edit distance, the evaluation pipeline is applied consistently across all models ensuring fair comparison. Results are summarized in Table 3, with qualitative comparisons in Figure 3.

##### Traditional Reconstruction Metrics.

Our high-compression VAEs ($f16$ and $f32$ settings) achieve state-of-the-art pixel-level reconstruction, outperforming all competing methods at the same compression ratios. Notably, Qwen-Image-VAE-2.0-f16c128 attains an SSIM of 0.9706 and PSNR of 30.45 dB, surpassing the best $f8$ baseline (FLUX.1-dev at 0.9364 / 26.24 dB) despite $2\times$ higher spatial compression. Even our lighter f16c64 variant surpasses most $f8$ baselines (SSIM 0.9279 vs. HunyuanVideo’s 0.9227), demonstrating strong efficiency at moderate channel budgets. Under extreme $f32$ compression where existing methods degrade catastrophically (SSIM as low as 0.5259), our f32c192 achieves 0.8908 SSIM with FID 1.98—outperforming all $f32$ competitors by substantial margins.

##### NED for Text Fidelity.

Traditional pixel metrics are inherently insensitive to character-level legibility. A single-character reconstruction error such as “orange” $\to$ “orango” incurs negligible PSNR loss ($<$ 0.5 dB) yet reduces NED by 16.7%, exposing the semantic corruption that pixel metrics might miss. The NED metric directly measures text preservation by comparing recognized character sequences between original and reconstructed images.

In the $f16$ VAEs, baseline performance varies dramatically: Cosmos-0.1-CI16x16 collapses to NED 0.1547 (${\sim}$ 85% character loss), while others range 0.35–0.88, all below $f8$ state-of-the-art. Our Qwen-Image-VAE-2.0-f16c64 achieves 0.9244, competitive with leading $f8$ methods. Most remarkably, Qwen-Image-VAE-2.0-f16c128 reaches 0.9617, *surpassing all evaluated $f8$ VAEs* including FLUX.1-dev (0.9546). To the best of our knowledge, this is the first $f16$ autoencoder to achieve text fidelity exceeding $f8$ VAEs.

The $f32$ VAEs further validate our advantage. While competing models exhibit near-total text destruction (NED: 0.07–0.57), our Qwen-Image-VAE-2.0-f32c192 achieves 0.8555, surpassing multiple $f16$ baselines. This cross-compression superiority stems directly from our comprehensive data engineering incorporating diverse text-rich documents and synthetic rendering pipelines.

##### Correlation Analysis.

We observe an imperfect correlation between pixel metrics and text fidelity. In $f16$, Stepvideo-T2V achieves notably higher NED than HunyuanImage-3.0 (0.8838 vs. 0.7753) despite modest SSIM differences (0.8970 vs. 0.8672); in $f32$, LTX-Video outperforms HunyuanImage-2.1 in NED (0.5651 vs. 0.4895) despite worse FID (17.10 vs. 5.19). These discrepancies validate NED as a necessary complementary metric for text reconstruction evaluation.

#### 6.1.3 Performance of Diffusability

To empirically validate the diffusability of our learned latent space, we evaluate downstream generative performance by training SiT (sit) on the ImageNet 256 $\times$ 256 dataset. We strictly adhere to the codebase and default hyperparameter settings of repa-e, reporting generation quality at 80 epochs via the Inception Score (IS) (inception) and generative FID (gFID). Specifically, we employed the SiT-XL/2 architecture for the $f8$ compression setting, and the SiT-XL/1 architecture for the $f16$ and $f32$ settings. Since higher-dimensional latent space often require larger optimal Classifier-Free Guidance (CFG) (cfg) scales, and the evaluated VAEs possess varying dimensions, we report only the results without guidance to ensure a fair comparison. As shown in Table 2, Qwen-Image-VAE-2.0 demonstrates superior latent space diffusability, consistently outperforming existing high-compression baselines in overall generation quality. Despite their large latent dimensions, our models facilitate rapid DiT convergence. This exceptional diffusability effectively resolves the traditional tripartite trade-off and is primarily driven by our improved semantic alignment strategy and staged alignment paradigm.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2605.13565/assets/x3.png)

((a)) f 16 f16 Compression VAEs. Baselines exhibit character blurring and stroke merging, while ours VAEs preserves crisp boundaries.

### 6.2 Qualitative Results

#### 6.2.1 Text Rendering

Figure 3 visualizes qualitative reconstruction results across different compression ratios. At $f16$ VAEs (Figure 3(a)), the degradation gap widens dramatically. Weaker baselines show severe character blurring, stroke merging, and inter-character ghosting—Cosmos-0.1-CI16x16 (cosmos) exhibits near-complete text collapse. In contrast, our Qwen-Image-VAE-2.0-f16c128 preserves crisp character boundaries, accurate inter-character spacing, and fine stroke detail.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2605.13565/assets/x5.png)

Figure 4: Selected image samples generated by SiT on ImageNet with Qwen-Image-VAE-2.0.

For $f32$ VAEs (Figure 3(b)), competing models reduce text to fragmented noise patterns where individual characters become unrecognizable. Our Qwen-Image-VAE-2.0-f32c192 uniquely retains clearly distinguishable character forms and recognizable word boundaries, consistent with its substantially higher NED scores and validating the effectiveness of our architecture under extreme compression constraints.

#### 6.2.2 Diffusability

Figure 4 illustrates selected ImageNet samples generated by SiT-XL, serving as a visual validation of the latent space’s semantic coherence and fine-grained detail. To demonstrate cross-scale consistency, samples are generated at $256\times 256$ for $f16$ VAEs and $512\times 512$ for $f32$ VAEs using a further-trained SiT-XL with classifier-free guidance, following our quantitative training protocol. Across $f16$ and $f32$ compression ratios, the generations maintain high visual fidelity without structural degradation.

#### 6.2.3 Large-scale Text-to-Image Validation

The successful integration of Qwen-Image-VAE-2.0 <sup>2</sup> into the Qwen-Image-2.0 (qwen-image-2.0) further validates the diffusability of our latent space at a foundation-model scale. Operating within this large-scale text-to-image pipeline, our compressed representations readily support complex open-vocabulary conditioning and intricate compositional constraints. The resulting generations consistently demonstrate Qwen-Image-2.0’s core capabilities through precise text rendering and refined photorealistic textures across diverse semantic contexts. This large-scale deployment demonstrates that our latent space preserves the structural stability and information fidelity required for demanding synthesis tasks, thereby confirming the scalability and robustness of our alignment paradigm in advanced generative systems.

## Conclusion

In this paper, we introduce Qwen-Image-VAE-2.0, a suite of high-compression image VAEs ($f16$ and $f32$) designed to overcome the long-standing bottlenecks in native high-resolution image synthesis. Our work demonstrates a clear technical path to resolving the fundamental tripartite trade-off between high compression ratio, reconstruction fidelity, and downstream diffusability: by leveraging expanded latent channel dimensions to compensate for spatial information loss caused by high compression, while simultaneously applying advanced semantic alignment techniques to ensure the latent space remains suitable for diffusion modeling. To achieve state-of-the-art reconstruction fidelity in high-compression regimes, we adopt an improved VAE architecture featuring Global Skip Connections (GSC), establishing a direct path for fine-grained detail recovery. This is bolstered by a comprehensive data strategy where we scale the training corpus to billions of images and utilize a specialized synthetic document-rendering pipeline to ensure the legibility of dense text—a traditional failure point for high-compression models. To address the challenge of diffusability inherent in high-dimensional latent space, we introduce an improved semantic alignment strategy. By aligning latent representations with middle-layer feature of DINOv2, we significantly accelerate the convergence of downstream DiT. Finally, to ensure practical utility, we implement a light-encoder paradigm and an attention-free backbone, maintaining high throughput and minimal encoding overhead even at ultra-high resolutions. Extensive evaluations on public benchmarks and OmniDoc-TokenBench demonstrate that Qwen-Image-VAE-2.0 not only preserves exceptional structural and textual integrity but also facilitates efficient generative modeling. These models provide a robust foundation for the next generation of visual synthesis systems, marking a significant milestone in efficient image generation.

## Authors

Contributors: Zekai Zhang <sup>*</sup>, Deqing Li <sup>*</sup>, Kuan Cao <sup>*</sup>, Yujia Wu <sup>*</sup>, Chenfei Wu <sup>†</sup>, Yu Wu, Liang Peng, Hao Meng, Jiahao Li, Jie Zhang, Kaiyuan Gao, Kun Yan, Lihan Jiang, Ningyuan Tang, Shengming Yin, Tianhe Wu, Xiao Xu, Xiaoyue Chen, Yan Shu, Yanran Zhang, Yilei Chen, Yixian Xu, Yuxiang Chen, Zhendong Wang, Zihao Liu, Zikai Zhou, Yiliang Gu, Yi Wang, Xiaoxiao Xu, Lin Qu

<sup>†</sup>