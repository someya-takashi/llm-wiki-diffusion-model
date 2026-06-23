---
title: "RePaint: Inpainting using Denoising Diffusion Probabilistic Models"
source: "https://ar5iv.labs.arxiv.org/html/2201.09865"
author:
published:
created: 2026-06-23
description: "Free-form inpainting is the task of adding new content to an image in the regions specified by an arbitrary binary mask.Most existing approaches train for a certain distribution of masks, which limits their generaliza…"
tags:
  - "clippings"
---
Andreas Lugmayr Martin Danelljan Andres Romero Fisher Yu Radu Timofte Luc Van Gool  
Computer Vision Lab  
ETH Zürich, Switzerland

###### Abstract

Free-form inpainting is the task of adding new content to an image in the regions specified by an arbitrary binary mask. Most existing approaches train for a certain distribution of masks, which limits their generalization capabilities to unseen mask types. Furthermore, training with pixel-wise and perceptual losses often leads to simple textural extensions towards the missing areas instead of semantically meaningful generation. In this work, we propose RePaint: A Denoising Diffusion Probabilistic Model (DDPM) based inpainting approach that is applicable to even extreme masks. We employ a pretrained *unconditional* DDPM as the generative prior. To condition the generation process, we only alter the reverse diffusion iterations by sampling the unmasked regions using the given image information. Since this technique does not modify or condition the original DDPM network itself, the model produces high-quality and diverse output images for any inpainting form. We validate our method for both faces and general-purpose image inpainting using standard and extreme masks. RePaint outperforms state-of-the-art Autoregressive, and GAN approaches for at least five out of six mask distributions. Github Repository: [git.io/RePaint](https://www.git.io/RePaint)

## 1 Introduction

Image Inpainting, also known as Image Completion, aims at filling missing regions within an image. Such inpainted regions need to harmonize with the rest of the image and be semantically reasonable. Inpainting approaches thus require strong generative capabilities. To this end, current State-of-the-Art approaches [^48] [^20] [^51] [^40] rely on GANs [^8] or Autoregressive Modeling [^33] [^42] [^49]. Moreover, inpainting methods need to handle various forms of masks such as thin or thick brushes, squares, or even extreme masks where the vast majority of the image is missing. This is highly challenging since existing approaches train with a certain mask distribution, which can lead to poor generalization to novel mask types. In this work, we investigate an alternative generative approach for inpainting, aiming to design an approach that requires no mask-specific training.

Denoising Diffusion Probabilistic Models (DDPM) is an emerging alternative paradigm for generative modelling [^12] [^38]. Recently, Dhariwal and Nichol [^7] demonstrated that DDPM can even outperform the state-of-the-art GAN-based method [^4] for image synthesis. In essence, the DDPM is trained to iteratively denoise the image by reversing a diffusion process. Starting from randomly sampled noise, the DDPM is then iteratively applied for a certain number of steps, which yields the final image sample. While founded in principled probabilistic modeling, DDPMs have been shown to generate diverse and high-quality images [^12] [^28] [^7].

We propose RePaint: an inpainting method that solely leverages an off-the-shelf unconditionally trained DDPM. Specifically, instead of learning a mask-conditional generative model, we condition the generation process by sampling from the given pixels during the reverse diffusion iterations. Remarkably, our model is therefore not trained for the inpainting task itself. This has two important advantages. First, it allows our network to generalize to any mask during inference. Second, it enables our network to learn more semantic generation capabilities since it has a powerful DDPM image synthesis prior (Figure LABEL:fig:intro).

Although the standard DDPM sampling strategy produces matching textures, the inpainting is often semantically incorrect. Therefore, we introduce an improved denoising strategy that resamples (RePaint) iterations to better condition the image. Notably, instead of slowing down the diffusion process [^7], our approach goes forward and backward in diffusion time, producing remarkable semantically meaningful images. Our approach allows the network to effectively harmonize the generated image information during the entire inference process, leading to a more effective conditioning on the given image information.

We perform experiments on CelebA-HQ [^21] and ImageNet [^36], and compare with other State-of-the-Art inpainting approaches. Our approach generalizes better and has overall more semantically meaningful inpainted regions.

## 2 Related Work

Early attempts on Image Inpainting or Image Completion exploited low-level cues within the input image [^2] [^1] [^3], or within the neighbor of a large image dataset [^10] to fill the missing region.

Deterministic Image Inpainting: Since the introduction of GANs [^8], most of the existing methods follow a standard configuration, first proposed by Pathak *et al*. [^32], that is, using an encoder-decoder architecture as the main inpainting generator, adversarial training, and tailored losses that aim at photo-realism. Follow-up works have produced impressive results in recent years [^34] [^50] [^20] [^30] [^15].

As image inpainting requires a high-level semantic context, and to explicitly include it in the generation pipeline, there exist hand-crafted architectural designs such as Dilated Convolutions [^16] [^45] to increase the receptive field, Partial Convolutions [^19] and Gated Convolutions [^48] to guide the convolution kernel according to the inpainted mask, Contextual Attention [^46] to leverage on global information, Edges maps [^27] [^43] [^44] [^9] or Semantic Segmentation maps [^14] [^31] to further guide the generation, and Fourier Convolutions [^40] to include both global and local information efficiently. Although recent works produce photo-realistic results, GANs are well known for textural synthesis, so these methods shine on background completion or removing objects, which require repetitive structural synthesis, and struggle with semantic synthesis (Figure 4).

Diverse Image Inpainting: Most GAN-based Image Inpainting methods are prone to deterministic transformations due to the lack of control during the image synthesis. To overcome this issue, Zheng *et al*. [^55] and Zhao *et al*. [^53] propose a VAE-based network that trade-offs between diversity and reconstruction. Zhao *et al*. [^54], inspired by the StyleGAN2 [^18] modulated convolutions, introduces a co-modulation layer for the inpainting task in order to improve both diversity and reconstruction. A new family of auto-regressive methods [^49] [^33] [^42], which can handle irregular masks, has recently emerged as a powerful alternative for free-form image inpainting.

Usage of Image Prior: In a different direction closer to ours Richardson *et al*. [^35] exploits the StyleGAN [^17] prior to successfully inpaint missing regions. However, similar to super-resolution methods [^26] [^5] that leverage the StyleGAN latent space, it is to limited specific scenarios like faces. Noteworthy, a Ulyanov *et al*. [^41] showed that the structure of a non-trained generator network contains an inherent prior that can be used for inpaining and other applications. In contrast to these methods, we are leveraging on the high expressiveness of a pretrained Denoising Diffusion Probabilistic Model [^12] (DDPM) and therefore use it as a prior for generic image inpainting. Our method generates very detailed, high-quality images for both semantically meaningful generation and texture synthesis. Moreover, our method is not trained for the image inpainting task, and instead, we take full advantage of the prior DDPM, so each image is optimized independently.

Image Conditional Diffusion Models: The work by Sohl-Dickstein *et al*. [^38] applied early diffusion models to inpainting. More recently, Song *et al*. [^39] develop a score-based formulation using stochastic differential equations for unconditional image generation, with an additional application to inpainting. However, both these works only show qualitative results, and do not compare with other inpainting approaches. In contrast, we aim to advance the state-of-the-art in image inpainting, and provide comprehensive comparisons with the top competing methods in literature.

A different line of research is guided image synthesis with DDPM-based approaches [^6] [^25]. In the case of ILVR [^6], a trained diffusion model is guided using the low-frequency information from a conditional image. However, this conditioning strategy cannot be adopted for inpainting, since both high and low-frequency information is absent in the masked-out regions. Another approach for image-conditional synthesis is developed by [^25]. Guided generation is performed by initializing the reverse diffusion process from the guiding image at some intermediate diffusion time. An iterative strategy, repeating the reverse process several times, is further adopted to improve harmonization. Since a guiding image is required to start the reverse process at an intermediate time step, this approach is not applicable to inpainting, where new image content needs to be generated solely conditioned on the non-masked pixels. Furthermore, the resampling strategy proposed in this work differs from the concurrent [^25]. We proceed through the full reverse diffusion process, starting at the end time, at each step jumping back and forth a fixed number of time steps to progressively improve generation quality.

While we propose a method that conditions an unconditionally trained model, the concurrent work [^29] is based on classifier-free guidance [^13] for training an image-conditional diffusion model. Another direction for image manipulation is image-to-image translation using diffusion models as explored in the concurrent work [^37]. It trains an image-conditional DDPM, and shows an application to inpainting. Unlike both these concurrent works, we leverage an unconditional DDPM and only condition through the reverse diffusion process itself. It allows our approach to effortlessly generalize to any mask shape for free-form inpainting. Moreover, we propose a sampling schedule for the reverse process, which greatly improves image quality.

## 3 Preliminaries: Denoising Diffusion Probabilistic Models

In this paper, we use diffusion models [^38] as a generative method. As other generative models, the DDPM learns a distribution of images given a training set. The inference process works by sampling a random noise vector ${x_{T}}$ and gradually denoising it until it reaches a high-quality output image ${x_{0}}$. During training, DDPM methods define a diffusion process that transforms an image $x_{0}$ to white Gaussian noise $x_{T}\sim\mathcal{N}(0,1)$ in $T$ time steps. Each step in the forward direction is given by,

$$
\displaystyle q(x_{t}|x_{t-1})
$$
 
$$
\displaystyle=\mathcal{N}(x_{t};\sqrt{1-\beta_{t}}x_{t-1},\beta_{t}\mathbf{I})
$$

The sample $x_{t}$ is obtained by adding i.i.d. Gaussian noise with variance $\beta_{t}$ at timestep $t$ and scaling the previous sample $x_{t-1}$ with $\sqrt{1-\beta_{t}}$ according to a variance schedule.

The DDPM is trained to reverse the process in (1). The reverse process is modeled by a neural network that predicts the parameters $\mu_{\theta}(x_{t},t)$ and $\Sigma_{\theta}(x_{t},t)$ of a Gaussian distribution,

$$
p_{\theta}(x_{t-1}|x_{t})=\mathcal{N}(x_{t-1};\mu_{\theta}(x_{t},t),\Sigma_{\theta}(x_{t},t))
$$

The learning objective for the model (2) is derived by considering the variational lower bound,

$$
\displaystyle\mathbb{E}\left[-\log p_{\theta}(\mathbf{x}_{0})\right]\leq\mathbb{E}_{q}\bigg{[}
$$
 
$$
\displaystyle-\log\frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_{0})}\bigg{]}
$$
 
$$
\displaystyle=\mathbb{E}_{q}\bigg{[}-\log p(\mathbf{x}_{T})
$$
 
$$
\displaystyle-\sum_{t\geq 1}\log\frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})}\bigg{]}=L
$$

As extended by Ho *et al*. [^12], this loss can be further decomposed as,

$$
\displaystyle\mathbb{E}_{q}\bigg{[}\underbrace{D_{\mathrm{KL}}\!\left(q(\mathbf{x}_{T}|\mathbf{x}_{0})~{}\|~{}p(\mathbf{x}_{T})\right)}_{L_{T}}
$$
 
$$
\displaystyle+\sum_{t>1}\underbrace{D_{\mathrm{KL}}\!\left(q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})~{}\|~{}p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})\right)}_{L_{t-1}}\underbrace{-\log p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})}_{L_{0}}\bigg{]}
$$

Importantly the term $L_{t-1}$ trains the network (2) to perform one reverse diffusion step. Furthermore, it allows for a closed from expression of the objective since $q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})$ is also Gaussian [^12].

As reported by Ho *et al*. [^12], the best way to parametrize the model is to predict the cumulative noise $\epsilon_{0}$ that is added to the current intermediate image $x_{t}$. Thus, we obtain the following parametrization of the predicted mean $\mu_{\theta}(x_{t},t)$,

$$
\mu_{\theta}(x_{t},t)=\frac{1}{\sqrt{\alpha_{t}}}\left(x_{t}-\frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}}\epsilon_{\theta}(x_{t},t)\right)
$$

From $L_{t-1}$ in (4), the following simplified training objective is derived by Ho *et al*. [^12],

$$
L_{\text{simple}}=E_{t,x_{0},\epsilon}\left[||\epsilon-\epsilon_{\theta}(x_{t},t)||^{2}\right]
$$

As introduced by Nichol and Dhariwal [^28], learning the variance $\Sigma_{\theta}(x_{t},t)$ in (2) of the reverse process helps to reduce the number of sampling steps by an order of magnitude. They, therefore, add the variational lower bound loss. Specifically, we base our training and inference approach on the recent work [^7], which further reduced the inference time by factor four.

To train the DDPM, we need a sample $x_{t}$ and corresponding noise that is used to transform $x_{0}$ to $x_{t}$. By using the independence property of the noise added at each step (1), we can calculate the total noise variance as $\bar{\alpha_{t}}=\prod_{s=1}^{t}(1-\beta_{s})$. We can thus rewrite (1), as a single step,

$$
\displaystyle q(x_{t}|x_{0})
$$
 
$$
\displaystyle=\mathcal{N}(x_{t};\sqrt{\bar{\alpha}_{t}}x_{0},(1-\bar{\alpha}_{t})\mathbf{I})
$$

It allows us to efficiently sample pairs of training data to train a reverse transition step.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/x1.png)

Figure 1: Overview of our approach. RePaint modifies the standard denoising process in order to condition on the given image content. In each step, we sample the known region ( top ) from the input and the inpainted part from the DDPM output ( bottom ).

## 4 Method

In this section, we first present our approach for conditioning the reverse diffusion process of an unconditional DDPM for image inpainting in Section 4.1. Then, we introduce an approach to improve the reverse process itself for inpainting in Section 4.2.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/x2.jpg)

Figure 2: The effect of applying n 𝑛 sampling steps. The first example with = 1 n=1 is the DDPM baseline, the second with 2 n=2 is with one resample step. More resampling steps lead to more harmonized images. The benefit saturates at about 10 n=10 resamplings.

### 4.1 Conditioning on the known Region

The goal of inpainting is to predict missing pixels of an image using a mask region as a condition. In the remaining of the paper, we consider a trained unconditional denoising diffusion probabilistic model (2). We denote the ground truth image as $x$, the unknown pixels as $m\odot x$ and the known pixels as $(1-m)\odot x$.

Algorithm 1 Inpainting using our RePaint approach.

 $x_{T}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$

for $t=T,\dotsc,1$ do

for $u=1,\dotsc,U$ do

$\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ if $t>1$, else $\epsilon=\mathbf{0}$

 $x_{t-1}^{\text{known}}=\sqrt{\bar{\alpha}_{t}}x_{0}+(1-\bar{\alpha}_{t})\epsilon$

$z\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ if $t>1$, else $\mathbf{z}=\mathbf{0}$

 $x_{t-1}^{\text{unknown}}=\frac{1}{\sqrt{\alpha_{t}}}\left(x_{t}-\frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}}{\boldsymbol{\epsilon}}_{\theta}(x_{t},t)\right)+\sigma_{t}z$ $x_{t-1}=m\odot x_{t-1}^{\text{known}}+(1-m)\odot x_{t-1}^{\text{unknown}}$

if $u<U~{}\text{and}~{}t>1$ then

 $x_{t}\sim\mathcal{N}(\sqrt{1-\beta_{t-1}}x_{t-1},\beta_{t-1}\mathbf{I})$

end if

end for

end for

return $x_{0}$

Since every reverse step (2) from $x_{t}$ to $x_{t-1}$ depends solely on $x_{t}$, we can alter the known regions $(1-m)\odot x_{t}$ as long as we keep the correct properties of the corresponding distribution. Since the forward process is defined by a Markov Chain (1) of added Gaussian noise, we can sample the intermediate image $x_{t}$ at any point in time using (7). This allows us to sample the know regions $m\odot x_{t}$ at any time step $t$. Thus, using (2) for the unknown region and (7) for the known regions, we achieve the following expression for one reverse step in our approach,

$$
\displaystyle x_{t-1}^{\text{known}}
$$
 
$$
\displaystyle\sim\mathcal{N}(\sqrt{\bar{\alpha}_{t}}x_{0},(1-\bar{\alpha}_{t})\mathbf{I})
$$
 
$$
\displaystyle x_{t-1}^{\text{unknown}}
$$
 
$$
\displaystyle\sim\mathcal{N}(\mu_{\theta}(x_{t},t),\Sigma_{\theta}(x_{t},t))
$$
 
$$
\displaystyle x_{t-1}
$$
 
$$
\displaystyle=m\odot x_{t-1}^{\text{known}}+(1-m)\odot x_{t-1}^{\text{unknown}}
$$

Thus, $x_{t-1}^{\text{known}}$ is sampled using the known pixels in the given image $m\odot x_{0}$, while $x_{t-1}^{\text{unknown}}$ is sampled from the model, given the previous iteration $x_{t}$. These are then combined to the new sample $x_{t-1}$ using the mask. Our approach is illustrated in Figure 1.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/figures/paper_c256_test_thick_q97.jpg)

Figure 3: CelebA-HQ Qualitative Results. Comparison against the state-of-the-art methods for Face Inpainting over several mask settings. Zoom-in for better details.

### 4.2 Resampling

When directly applying the method described in Section 4.1, we observe that only the content type matches with the known regions. For example, in Figure 2 $n=1$, the inpainted area is a furry texture matching the hair of the dog. Although the inpainted region matches the texture of the neighboring region, it is semantically incorrect. Therefore, the DDPM is leveraging on the context of the known region, yet it is not harmonizing it well with the rest of the image. Next, we discuss possible reasons for this behavior.

From Figure 1, we analyze how the method is conditioning the known regions. As shown in (8), the model predicts $x_{t-1}$ using $x_{t}$, which comprises the output of the DDPM (2) and the sample from the known region. However, the sampling of the known pixels using (7) is performed without considering the generated parts of the image, which introduces disharmony. Although the model tries to harmonize the image again in every step, it can never fully converge because the same issue occurs in the next step. Moreover, in each reverse step, the maximum change to an image declines due to the variance schedule of $\beta_{t}$. Thus, the method cannot correct mistakes that lead to disharmonious boundaries in the subsequent steps due to restricted flexibility. As a consequence, the model needs more time to harmonize the conditional information $x_{t-1}^{\text{known}}$ with the generated information $x_{t-1}^{\text{unknown}}$ in one step before advancing to the next denoising step.

Since the DDPM is trained to generate an image that lies within a data distribution, it naturally aims at producing consistent structures. In our resampling approach, we use this DDPM property to harmonize the input of the model. Consequently, we diffuse the output $x_{t-1}$ back to $x_{t}$ by sampling from (1) as $x_{t}\sim\mathcal{N}(\sqrt{1-\beta_{t}}x_{t-1},\beta_{t}\mathbf{I})$. Although this operation scales back the output and adds noise, some information incorporated in the generated region $x_{t-1}^{\text{unknown}}$ is still preserved in $x_{t}^{\text{unknown}}$. It leads to a new $x_{t}^{\text{unknown}}$ which is both more harmonized with $x_{t}^{\mathtt{known}}$ and contains conditional information from it.

Since this operation can only harmonize one step, it might not be able to incorporate the semantic information over the entire denoising process. To overcome this problem, we denote the time horizon of this operation as jump length, which is $j=1$ for the previous case. Similar to the standard change in diffusion speed [^7] (a.k.a. slowing down), the resampling also increases the runtime of the reverse diffusion. Slowing down applies smaller but more resampling steps by reducing the added variance in each denoising step. However, that is a fundamentally different approach because slowing down the diffusion still has the problem of not harmonizing the image, as described in our resampling strategy. We empirically demonstrate this advantage of our approach in Sec. 5.6.

## 5 Experiments

We perform extensive experiments for face and generic inpainting, compare to the state-of-the-art solutions, and conduct an ablative analysis. In Section 5.3 and 5.4, we report a detailed discussion of mask robustness and diversity, respectively. We also report with additional results, analysis, and visuals in the appendix.

### 5.1 Implementation Details

We validate our solution over the CelebA-HQ [^21], and Imagenet [^36] datasets. As our method relies on a pretrained guided diffusion model [^7], we use the provided ImageNet model. For CelebA-HQ, we follow the same training hyper-parameters as for ImageNet. We use $256\times 256$ crops in three batches on 4 $\times$ V100 GPUs each. In contrast to the pretrained ImageNet model, the CelebA-HQ one is only trained for 250,000 iterations during roughly five days. Note that all our qualitative and quantitative results in the main paper are for 256 image size.

<table><thead><tr><th>CelebA-HQ</th><th colspan="2">Wide</th><th colspan="2">Narrow</th><th colspan="2">Super-Resolve <math><semantics><mrow><mn>2</mn> <mo>×</mo></mrow> <annotation>2\times</annotation></semantics></math></th><th colspan="2">Altern. Lines</th><th colspan="2">Half</th><th colspan="2">Expand</th></tr><tr><th>Methods</th><th>LPIPS <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></th><th>Votes [%]</th><th>LPIPS <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></th><th>Votes [%]</th><th>LPIPS <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></th><th>Votes [%]</th><th>LPIPS <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></th><th>Votes [%]</th><th>LPIPS <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></th><th>Votes [%]</th><th>LPIPS <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></th><th>Votes [%]</th></tr></thead><tbody><tr><td>AOT <sup><a href="#fn:51">51</a></sup></td><td>0.104</td><td><math><semantics><mrow><mn>11.6</mn> <mo>±</mo> <mn>2.0</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>11.6</cn> <cn>2.0</cn></apply> <annotation>11.6\pm 2.0</annotation></semantics></math></td><td>0.047</td><td><math><semantics><mrow><mn>12.8</mn> <mo>±</mo> <mn>2.1</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>12.8</cn> <cn>2.1</cn></apply> <annotation>12.8\pm 2.1</annotation></semantics></math></td><td>0.714</td><td><math><semantics><mrow><mn>1.1</mn> <mo>±</mo> <mn>0.6</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>1.1</cn> <cn>0.6</cn></apply> <annotation>1.1\pm 0.6</annotation></semantics></math></td><td>0.667</td><td><math><semantics><mrow><mn>2.4</mn> <mo>±</mo> <mn>1.0</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>2.4</cn> <cn>1.0</cn></apply> <annotation>2.4\pm 1.0</annotation></semantics></math></td><td>0.287</td><td><math><semantics><mrow><mn>9.0</mn> <mo>±</mo> <mn>1.8</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>9.0</cn> <cn>1.8</cn></apply> <annotation>9.0\pm 1.8</annotation></semantics></math></td><td>0.604</td><td><math><semantics><mrow><mn>8.3</mn> <mo>±</mo> <mn>1.7</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>8.3</cn> <cn>1.7</cn></apply> <annotation>8.3\pm 1.7</annotation></semantics></math></td></tr><tr><td>DSI <sup><a href="#fn:33">33</a></sup></td><td>0.067</td><td><math><semantics><mrow><mn>16.0</mn> <mo>±</mo> <mn>2.3</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>16.0</cn> <cn>2.3</cn></apply> <annotation>16.0\pm 2.3</annotation></semantics></math></td><td>0.038</td><td><math><semantics><mrow><mn>22.3</mn> <mo>±</mo> <mn>2.6</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>22.3</cn> <cn>2.6</cn></apply> <annotation>22.3\pm 2.6</annotation></semantics></math></td><td>0.128</td><td><math><semantics><mrow><mn>5.5</mn> <mo>±</mo> <mn>1.4</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>5.5</cn> <cn>1.4</cn></apply> <annotation>5.5\pm 1.4</annotation></semantics></math></td><td>0.049</td><td><math><semantics><mrow><mn>5.1</mn> <mo>±</mo> <mn>1.4</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>5.1</cn> <cn>1.4</cn></apply> <annotation>5.1\pm 1.4</annotation></semantics></math></td><td>0.211</td><td><math><semantics><mrow><mn>4.5</mn> <mo>±</mo> <mn>1.3</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>4.5</cn> <cn>1.3</cn></apply> <annotation>4.5\pm 1.3</annotation></semantics></math></td><td>0.487</td><td><math><semantics><mrow><mn>4.7</mn> <mo>±</mo> <mn>1.3</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>4.7</cn> <cn>1.3</cn></apply> <annotation>4.7\pm 1.3</annotation></semantics></math></td></tr><tr><td>ICT <sup><a href="#fn:42">42</a></sup></td><td>0.063</td><td><math><semantics><mrow><mn>27.6</mn> <mo>±</mo> <mn>2.8</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>27.6</cn> <cn>2.8</cn></apply> <annotation>27.6\pm 2.8</annotation></semantics></math></td><td>0.036</td><td><math><semantics><mrow><mn>30.9</mn> <mo>±</mo> <mn>2.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>30.9</cn> <cn>2.9</cn></apply> <annotation>30.9\pm 2.9</annotation></semantics></math></td><td>0.483</td><td><math><semantics><mrow><mn>4.2</mn> <mo>±</mo> <mn>1.2</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>4.2</cn> <cn>1.2</cn></apply> <annotation>4.2\pm 1.2</annotation></semantics></math></td><td>0.353</td><td><math><semantics><mrow><mn>0.7</mn> <mo>±</mo> <mn>0.5</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>0.7</cn> <cn>0.5</cn></apply> <annotation>0.7\pm 0.5</annotation></semantics></math></td><td>0.166</td><td><math><semantics><mrow><mn>12.7</mn> <mo>±</mo> <mn>2.1</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>12.7</cn> <cn>2.1</cn></apply> <annotation>12.7\pm 2.1</annotation></semantics></math></td><td>0.432</td><td><math><semantics><mrow><mn>8.8</mn> <mo>±</mo> <mn>1.8</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>8.8</cn> <cn>1.8</cn></apply> <annotation>8.8\pm 1.8</annotation></semantics></math></td></tr><tr><td>DeepFillv2 <sup><a href="#fn:47">47</a></sup></td><td>0.066</td><td><math><semantics><mrow><mn>23.9</mn> <mo>±</mo> <mn>2.6</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>23.9</cn> <cn>2.6</cn></apply> <annotation>23.9\pm 2.6</annotation></semantics></math></td><td>0.049</td><td><math><semantics><mrow><mn>21.0</mn> <mo>±</mo> <mn>2.5</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>21.0</cn> <cn>2.5</cn></apply> <annotation>21.0\pm 2.5</annotation></semantics></math></td><td>0.119</td><td><math><semantics><mrow><mn>9.8</mn> <mo>±</mo> <mn>1.8</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>9.8</cn> <cn>1.8</cn></apply> <annotation>9.8\pm 1.8</annotation></semantics></math></td><td>0.049</td><td><math><semantics><mrow><mn>10.6</mn> <mo>±</mo> <mn>1.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>10.6</cn> <cn>1.9</cn></apply> <annotation>10.6\pm 1.9</annotation></semantics></math></td><td>0.209</td><td><math><semantics><mrow><mn>4.1</mn> <mo>±</mo> <mn>1.2</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>4.1</cn> <cn>1.2</cn></apply> <annotation>4.1\pm 1.2</annotation></semantics></math></td><td>0.467</td><td><math><semantics><mrow><mn>13.1</mn> <mo>±</mo> <mn>2.1</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>13.1</cn> <cn>2.1</cn></apply> <annotation>13.1\pm 2.1</annotation></semantics></math></td></tr><tr><td>LaMa <sup><a href="#fn:40">40</a></sup></td><td>0.045</td><td><math><semantics><mrow><mn>41.8</mn> <mo>±</mo> <mn>3.1</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>41.8</cn> <cn>3.1</cn></apply> <annotation>41.8\pm 3.1</annotation></semantics></math></td><td>0.028</td><td><math><semantics><mrow><mn>33.8</mn> <mo>±</mo> <mn>3.0</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>33.8</cn> <cn>3.0</cn></apply> <annotation>33.8\pm 3.0</annotation></semantics></math></td><td>0.177</td><td><math><semantics><mrow><mn>5.5</mn> <mo>±</mo> <mn>1.4</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>5.5</cn> <cn>1.4</cn></apply> <annotation>5.5\pm 1.4</annotation></semantics></math></td><td>0.083</td><td><math><semantics><mrow><mn>20.6</mn> <mo>±</mo> <mn>2.5</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>20.6</cn> <cn>2.5</cn></apply> <annotation>20.6\pm 2.5</annotation></semantics></math></td><td>0.138</td><td><math><semantics><mrow><mn>35.6</mn> <mo>±</mo> <mn>3.0</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>35.6</cn> <cn>3.0</cn></apply> <annotation>35.6\pm 3.0</annotation></semantics></math></td><td>0.342</td><td><math><semantics><mrow><mn>24.7</mn> <mo>±</mo> <mn>2.7</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>24.7</cn> <cn>2.7</cn></apply> <annotation>24.7\pm 2.7</annotation></semantics></math></td></tr><tr><td>RePaint</td><td>0.059</td><td>Reference</td><td>0.028</td><td>Reference</td><td>0.029</td><td>Reference</td><td>0.009</td><td>Reference</td><td>0.165</td><td>Reference</td><td>0.435</td><td>Reference</td></tr></tbody></table>

<table><thead><tr><th>ImageNet</th><th colspan="2">Wide</th><th colspan="2">Narrow</th><th colspan="2">Super-Resolve <math><semantics><mrow><mn>2</mn> <mo>×</mo></mrow> <annotation>2\times</annotation></semantics></math></th><th colspan="2">Altern. Lines</th><th colspan="2">Half</th><th colspan="2">Expand</th></tr><tr><th>Methods</th><th>LPIPS <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></th><th>Votes [%]</th><th>LPIPS <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></th><th>Votes [%]</th><th>LPIPS <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></th><th>Votes [%]</th><th>LPIPS <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></th><th>Votes [%]</th><th>LPIPS <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></th><th>Votes [%]</th><th>LPIPS <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></th><th>Votes [%]</th></tr></thead><tbody><tr><th>DSI <sup><a href="#fn:33">33</a></sup></th><td>0.117</td><td><math><semantics><mrow><mn>31.7</mn> <mo>±</mo> <mn>2.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>31.7</cn> <cn>2.9</cn></apply> <annotation>31.7\pm 2.9</annotation></semantics></math></td><td>0.072</td><td><math><semantics><mrow><mn>28.6</mn> <mo>±</mo> <mn>2.8</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>28.6</cn> <cn>2.8</cn></apply> <annotation>28.6\pm 2.8</annotation></semantics></math></td><td>0.153</td><td><math><semantics><mrow><mn>26.9</mn> <mo>±</mo> <mn>2.8</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>26.9</cn> <cn>2.8</cn></apply> <annotation>26.9\pm 2.8</annotation></semantics></math></td><td>0.069</td><td><math><semantics><mrow><mn>23.6</mn> <mo>±</mo> <mn>2.6</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>23.6</cn> <cn>2.6</cn></apply> <annotation>23.6\pm 2.6</annotation></semantics></math></td><td>0.283</td><td><math><semantics><mrow><mn>31.4</mn> <mo>±</mo> <mn>2.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>31.4</cn> <cn>2.9</cn></apply> <annotation>31.4\pm 2.9</annotation></semantics></math></td><td>0.583</td><td><math><semantics><mrow><mn>9.2</mn> <mo>±</mo> <mn>1.8</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>9.2</cn> <cn>1.8</cn></apply> <annotation>9.2\pm 1.8</annotation></semantics></math></td></tr><tr><th>ICT <sup><a href="#fn:42">42</a></sup></th><td>0.107</td><td><math><semantics><mrow><mn>42.9</mn> <mo>±</mo> <mn>3.1</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>42.9</cn> <cn>3.1</cn></apply> <annotation>42.9\pm 3.1</annotation></semantics></math></td><td>0.073</td><td><math><semantics><mrow><mn>33.0</mn> <mo>±</mo> <mn>2.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>33.0</cn> <cn>2.9</cn></apply> <annotation>33.0\pm 2.9</annotation></semantics></math></td><td>0.708</td><td><math><semantics><mrow><mn>1.1</mn> <mo>±</mo> <mn>0.6</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>1.1</cn> <cn>0.6</cn></apply> <annotation>1.1\pm 0.6</annotation></semantics></math></td><td>0.620</td><td><math><semantics><mrow><mn>6.6</mn> <mo>±</mo> <mn>1.5</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>6.6</cn> <cn>1.5</cn></apply> <annotation>6.6\pm 1.5</annotation></semantics></math></td><td>0.255</td><td><math><semantics><mrow><mn>51.5</mn> <mo>±</mo> <mn>3.1</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>51.5</cn> <cn>3.1</cn></apply> <annotation>51.5\pm 3.1</annotation></semantics></math></td><td>0.544</td><td><math><semantics><mrow><mn>25.6</mn> <mo>±</mo> <mn>2.7</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>25.6</cn> <cn>2.7</cn></apply> <annotation>25.6\pm 2.7</annotation></semantics></math></td></tr><tr><th>LaMa <sup><a href="#fn:40">40</a></sup></th><td>0.105</td><td><math><semantics><mrow><mn>42.4</mn> <mo>±</mo> <mn>3.1</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>42.4</cn> <cn>3.1</cn></apply> <annotation>42.4\pm 3.1</annotation></semantics></math></td><td>0.061</td><td><math><semantics><mrow><mn>33.6</mn> <mo>±</mo> <mn>2.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>33.6</cn> <cn>2.9</cn></apply> <annotation>33.6\pm 2.9</annotation></semantics></math></td><td>0.272</td><td><math><semantics><mrow><mn>13.0</mn> <mo>±</mo> <mn>2.1</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>13.0</cn> <cn>2.1</cn></apply> <annotation>13.0\pm 2.1</annotation></semantics></math></td><td>0.121</td><td><math><semantics><mrow><mn>9.6</mn> <mo>±</mo> <mn>1.8</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>9.6</cn> <cn>1.8</cn></apply> <annotation>9.6\pm 1.8</annotation></semantics></math></td><td>0.254</td><td><math><semantics><mrow><mn>41.1</mn> <mo>±</mo> <mn>3.1</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>41.1</cn> <cn>3.1</cn></apply> <annotation>41.1\pm 3.1</annotation></semantics></math></td><td>0.534</td><td><math><semantics><mrow><mn>20.3</mn> <mo>±</mo> <mn>2.5</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>20.3</cn> <cn>2.5</cn></apply> <annotation>20.3\pm 2.5</annotation></semantics></math></td></tr><tr><th>RePaint</th><td>0.134</td><td>Reference</td><td>0.064</td><td>Reference</td><td>0.183</td><td>Reference</td><td>0.089</td><td>Reference</td><td>0.304</td><td>Reference</td><td>0.629</td><td>Reference</td></tr></tbody></table>

Table 1: CelebA-HQ (top) and ImageNet (bottom) Quantitative Results. Comparison against the state-of-the-art methods. We compute the LPIPS (lower is better) and Votes for six different mask settings. Votes refers to the ratio of votes with respect to ours.

For our final approach, we use $T=250$ timesteps, and applied $r=10$ times resampling with jumpy size $j=10$.

### 5.2 Metrics

We compare our RePaint with the baseline methods in a user study described as follows. The user is shown the input image with the blanked missing regions. Next to this image, we display two different inpainting solutions. The user is asked to select “Which image looks more realistic?”. The user thus evaluates the realism of our RePaint to the result of a baseline. To avoid biasing the user towards an approach, the methods were anonymized shown in a different random order for each image. Moreover, each user was asked every question twice and could only submit their answer if they were consistent with themselves in at least 75% of their answer. A self-consistency in 100% of the cases is often not possible since, for example, the LaMa method can have a very similar quality to RePaint on the mask settings they provide. Our user study evaluates all 100 test images of the test datasets CelebA-HQ and ImageNet for the following masks: Wide, Narrow, Every Second Line, Half Image, Expand, and Super-Resolve. We use the answers of five different humans for every image query, resulting in 1000 votes per method-to-method comparison in each dataset and mask setting, and show the 95% confidence interval next to the mean votes. In addition to the user study, we report the commonly reported perceptual metric LPIPS [^52], which is a learned distance metric based on the deep feature space of AlexNet. We compute the LPIPS over the same 100 test images used in the user study. The results are shown in Table 1. Furthermore, please refer to the appendix for additional quantitative results.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/figures/paper_inet256_test_thick_q97.jpg)

Figure 4: ImageNet Qualitative Results. Comparison against the state-of-the-art methods for pluralistic inpainting methods over different mask settings. Zoom-in for better details.

### 5.3 Comparison with State-of-the-Art

In this section, we first compare our approach against state-of-the-art on standard mask distributions, commonly employed for benchmarking. We then analyze the generalization capabilities of our method against other approaches. To this end, we evaluate their robustness under four challenging mask settings. Firstly, two different masks that probe if the methods can incorporate information from thin structures. Secondly, two masks that require to inpaint a large connected area of the image. All quantitative results are reported in Table 1 and visual results in Figure 3 and 4.

Methods: We compare our approach against several state-of-the-art autoregressive-based or GAN-based approaches. The autoregressive methods are DSI [^33] and ICT [^42], and the GAN methods are DeepFillv2 [^47], AOT [^51], and LaMa [^40]. We use their publicly available pretrained models. We used the existing FFHQ [^17] pretrained model of ICT for our CelebA-HQ testing. As LaMa does not provide ImageNet models, we trained their system for 300,000 iterations of batch size five using the original implementation.

Settings: We use 100 images of size 256 $\times$ 256 from CelebA-HQ [^21] and ImageNet test sets. The resulting LPIPS and the average votes of the user study are shown in Table 1. Additionally, refer to the appendix for qualitative and quantitative results over the Places2 [^56] dataset.

Wide and Narrow masks: To validate our method on the standard image inpainting scenario, we use the LaMa [^40] settings for Wide and Narrow masks. RePaint outperforms all other methods with a significance margin of 95% in both CelebA-HQ and ImageNet, for both Wide and Narrow settings. See qualitative results in Figure 3 and 4 and quantitative in Table 1. The best autoregressive method ICT seems to have less global consistency as observed in Figure 3 in the second row, where the eyes do not to match well. In general, the best GAN approach LaMa [^40] has better global consistency, yet it produces notorious checkerboard artifacts. Those observations might have influenced the users to vote for RePaint for the majority of images, in which our method generates more realistic images.

Thin Masks: Similar to a Nearest-Neighbor Super Resolution problem, the “Super-Resolution $2\times$ ” mask only leaves pixels with a stride of 2 in height and width dimension, and the “Alternating Lines” mask removes the pixels every second row of an image. As seen in Figure 3 and 4, AOT [^51] fails completely, while the others either produce blurry images or generate visible artifacts, or both. These observations are also confirmed by the user study, where RePaint achieves between 73.1% and 99.3% of the user votes.

Thick Masks: The “Expand” mask only leaves a center crop of $64\times 64$ from a $256\times 256$ image, and “Half” mask, which provides the left half of the image as input. As there is less contextual information, most of the methods struggle (see Figure 3 and 4). Qualitatively, LaMa comes closer to ours, yet our generated images are sharper and have overall more semantic hallucination. Noteworthy, LaMa outperforms RePaint in therms of LPIPS on “Expand” and “Half” for both CelebA and ImageNet (Tab. 1). We argue that this behavior is due to our method being more flexible and diverse in the generation. By generating a semantically different image than that in the Ground-Truth, it makes the LPIPS an unsuitable metric for this particular solution.

The artifacts produced by the baselines can be explained by strong overfitting to the training masks. In contrast, as our method does not involve mask training, our RePaint can handle any type of mask. In the case of large-area inpainting, RePaint produces a semantically meaningful filling, while others generate artifacts or copy texture. Finally, RePaint is preferred by the users with confidence 95% except for the inconclusive result of ICT with “Half” masks as shown in Table 1.

### 5.4 Analysis of Diversity

As shown in (2), every reverse diffusion step is inherently stochastic since it incorporates new noise from a Gaussian Distribution. Moreover, as we do not directly guide the inpainted area with any loss, the model is, therefore, free to inpaint anything that semantically aligns with the training set. Figure LABEL:fig:intro illustrates the diversity and flexibility of our model.

### 5.5 Class conditional Experiment

The pretrained ImageNet DDPM is capable of class-conditional generation sampling. In Figure 5 we show examples for the “Expand” mask for the “Granny Smith” class, as well as other classes.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/figures/cond_q97.jpg)

Figure 5: Visual results for class guided generation on ImageNet.

### 5.6 Ablation Study

Comparison to slowing down: To analyze if the increased computational budget causes the improved performance of resampling, we compare it with the commonly used technique of slowing down the diffusion process as described in Section 4.2. Therefore, in Figure 6 and Table 2, we show a comparison resampling and the slow down in diffusion using the same computational budget for each setting. We observe that the resampling uses the extra computational budget for harmonizing the image, whereas there is no visible improvement at slowing down the diffusion process.

|  | T | r | LPIPS | T | r | LPIPS | T | r | LPIPS | T | r | LPIPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Slowing down | 250 | 1 | 0.168 | 500 | 1 | 0.167 | 750 | 1 | 0.179 | 1000 | 1 | 0.161 |
| Resampling | 250 | 1 | 0.168 | 250 | 2 | 0.148 | 250 | 3 | 0.142 | 250 | 4 | 0.134 |

Table 2: Analysis of the use of computational budget. We compare slowing down the diffusion process and resampling. We use the ImageNet validation set with 32 images over the LaMa [^40] Wide mask setting. The number of diffusion steps is denoted by $T$, and the number of resamplings by $r$.

<table><thead><tr><th></th><th colspan="2"><math><semantics><mrow><mi>j</mi> <mo>=</mo> <mn>1</mn></mrow> <apply><ci>𝑗</ci> <cn>1</cn></apply> <annotation>j=1</annotation></semantics></math></th><th colspan="2"><math><semantics><mrow><mi>j</mi> <mo>=</mo> <mn>5</mn></mrow> <apply><ci>𝑗</ci> <cn>5</cn></apply> <annotation>j=5</annotation></semantics></math></th><th colspan="2"><math><semantics><mrow><mi>j</mi> <mo>=</mo> <mn>10</mn></mrow> <apply><ci>𝑗</ci> <cn>10</cn></apply> <annotation>j=10</annotation></semantics></math></th></tr><tr><th>r</th><th>LPIPS</th><th>Votes [%]</th><th>LPIPS</th><th>Votes [%]</th><th>LPIPS</th><th>Votes [%]</th></tr></thead><tbody><tr><th>5</th><td>0.075</td><td>42.50 <math><semantics><mo>±</mo> <csymbol>plus-or-minus</csymbol> <annotation>\pm</annotation></semantics></math> 7.7</td><td>0.072</td><td>46.88 <math><semantics><mo>±</mo> <csymbol>plus-or-minus</csymbol> <annotation>\pm</annotation></semantics></math> 7.8</td><td>0.073</td><td>53.12 <math><semantics><mo>±</mo> <csymbol>plus-or-minus</csymbol> <annotation>\pm</annotation></semantics></math> 7.8</td></tr><tr><th>10</th><td>0.088</td><td>42.50 <math><semantics><mo>±</mo> <csymbol>plus-or-minus</csymbol> <annotation>\pm</annotation></semantics></math> 7.7</td><td>0.073</td><td>45.62 <math><semantics><mo>±</mo> <csymbol>plus-or-minus</csymbol> <annotation>\pm</annotation></semantics></math> 7.8</td><td>0.068</td><td>56.25 <math><semantics><mo>±</mo> <csymbol>plus-or-minus</csymbol> <annotation>\pm</annotation></semantics></math> 7.8</td></tr><tr><th>15</th><td>0.065</td><td>46.25 <math><semantics><mo>±</mo> <csymbol>plus-or-minus</csymbol> <annotation>\pm</annotation></semantics></math> 7.8</td><td>0.063</td><td>53.12 <math><semantics><mo>±</mo> <csymbol>plus-or-minus</csymbol> <annotation>\pm</annotation></semantics></math> 5.5</td><td>0.065</td><td>53.75 <math><semantics><mo>±</mo> <csymbol>plus-or-minus</csymbol> <annotation>\pm</annotation></semantics></math> 7.8</td></tr></tbody></table>

Table 3: Ablation Study. Analysis of length of the jumps $j$ and number of resamplings $r$. We report LPIPS and the average user-study votes with respect to LaMa [^40]. We use 32 images from the CelebA validation set, and use the LaMa Wide mask setting.

Jumps Length: Moreover, to ablate the jump lengths $j$ and the number of resampling $r$, we study nine different settings in Table 3. We obtain better performance at applying the larger jump $j=10$ length than smaller step length steps. We observe that for jump length $j=1$, the DDPM is more likely to output a blurry image. Furthermore, this observation is stable across different numbers of resampling. Furthermore, the number of resamplings increases the performance.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/figures/paper_ts_vs_resampling_ts_q97.jpg)

Figure 6: Qualitative Analysis of the use of computational budget. RePaint produces higher visual quality with the same computational budget by resampling ( bottom ) compared to slowing down the diffusion process ( top ). The number of diffusion steps is denoted by T 𝑇 and resamplings by r 𝑟.

Comparison to alternative sampling strategy: To compare our resampling approach to SDEdit [^25], we first perform reverse diffusion from $t=T$ to $t=T/2$ to obtain the required initial inpainting at $t=T/2$. We then apply the resampling method from SDEdit, which repeats the reverse process from $t=T/2$ to $t=0$ several times. The results are shown in Table 4. Our approach achieves significantly better performance across all mask types except for one “Expand” case, where LPIPS $>0.6$ is outside a meaningful range for comparisons. In case of ‘super-resolution masks’, our approach reduces the LPIPS by over 53% on all datasets, clearly demonstrating the advantage of our resampling strategy.

| Dataset | Method | Wide | Narrow | Super-Res. | Alt. Lin. | Half | Expand |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ImageNet | SDEdit [^25] | 0.1532 | 0.0952 | 0.3902 | 0.1852 | 0.3272 | 0.6281 |
|  | RePaint (Ours) | 0.1341 | 0.0641 | 0.1831 | 0.0891 | 0.3041 | 0.6292 |
| Places2 | SDEdit [^25] | 0.1302 | 0.0622 | 0.2712 | 0.1302 | 0.3042 | 0.6202 |
|  | RePaint (Ours) | 0.1051 | 0.0441 | 0.0991 | 0.0511 | 0.2861 | 0.6151 |
| CelebA-HQ | SDEdit [^25] | 0.0762 | 0.0462 | 0.1132 | 0.0302 | 0.1892 | 0.4492 |
|  | RePaint (Ours) | 0.0591 | 0.0281 | 0.0291 | 0.0091 | 0.1651 | 0.4351 |

Table 4: Comparison with the resampling schedule proposed in [^25] in terms of LPIPS. The resampling method proposed in our RePaint (Sec. 4.2) achieves substantially better results, in particular for the Super-Resolution masks.

## 6 Limitations

Our method produces sharp, highly detailed, and semantically meaningful images. We believe that our work opens interesting research directions for addressing the current limitations of the method. Two directions are of particular interest. First, naturally, the per-image DDPM optimization process is significantly slower than the GAN-based and Autoregressive-based counterparts. That makes it currently difficult to apply it for real-time applications. Nonetheless, DDPM is gaining in popularity, and recent publications are working on improving the efficiency [^24] [^23]. Secondly, for the extreme mask cases, RePaint can produce realistic images completions that are very different from the Ground Truth image. That makes the quantitative evaluation challenging for those conditions. An alternative solution is to employ the FID score [^11] over a test set. However, a reliable FID for inpainting is usually computed with more than 1,000 images. For current DDPM, this would result in a runtime that is not feasible for most research institutes.

## 7 Potential Negative Societal Impact

On the one hand, RePaint is an inpainting method that relies on an unconditional pretrained DDPM. Therefore, the algorithm might be biased towards the dataset on which it was trained. Since the model aims to generate images of the same distribution as the training set, it might reflect the same biases, such as gender, age, and ethnicity. On the other hand, RePaint could be used for the anonymization of faces. For example, one could remove the information about the identity of people shown at public events and hallucinate artificial faces for data protection.

## 8 Conclusions

We presented a novel denoising diffusion probabilistic model solution for the image inpainting task. In detail, we developed a mask-agnostic approach that widely increases the degree of freedom of masks for the free-form inpainting. Since the novel conditioning approach of RePaint complies with the model assumptions of a DDPM, it produces a photo-realistic image regardless of the type of the mask.

Acknowledgements: This work was supported by the ETH Zürich Fund (OK), a Huawei Technologies Oy (Finland) project, and an Nvidia GPU grant.

## References

## Appendix

In this appendix, we provide additional details and analysis of our approach. We give more explanation on our user study in Section A. Further, we present additional details on how we implemented the diffusion time schedule for jumps in Section B. Visual results for our ablation for jump size and the number of resamplings are provided in Section C. The evaluation on the second part of the LaMa Benchmark on Places2 is presented in Section D. Furthermore, to compare the diversity of the inpaintings for RePaint compared with state-of-the-art, we provide a quantitative analysis in Section E. Details on failure cases and data bias on the ImageNet dataset are provided in Section F. For gaining a better intuitive understanding of the evolution of the latent space, we provide a video of the inference in Section G. And finally, we show additional visual examples in Section I.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/supplement/figures/user_study_full_q97.jpg)

Figure 7: User Study Interface. Example of the user-study interface. Based on the reference image on the Left, the user selects the image that looks more realistic.

t\_T = 250 jump\_len = 10 jump\_n\_sample = 10

jumps = for j in range(0, t\_T - jump\_len, jump\_len): jumps\[j\] = jump\_n\_sample - 1

t = t\_T ts = \[\]

while t ¿= 1: t = t-1 ts.append(t)

if jumps.get(t, 0) ¿ 0: jumps\[t\] = jumps\[t\] - 1 for \_in range(jump\_len): t = t + 1 ts.append(t)

ts.append(-1)

Figure 8: Diffusion Time Schedule. Pseudo code to generate diffusion time steps for jump length $j=10$ and resample $r=10$.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/x3.png)

Figure 9: Diffusion time during inference. The diffusion time t 𝑡 that a sample x subscript 𝑥 x\_{t} is transiting during the inference process with jump length j = 10 𝑗 j=10 and resampling r 𝑟 r=10.

## A User Study

As described in Section 5.2 in the main paper, we conduct a user study to determine which method is best perceived to the human eye. In Figure 7, we depict the user interface, where the user selects the most realistic solution from an input reference. To reduce bias, we show the two candidate images in random order. Additionally, to improve the consistency of the user decision and prevent answers with low effort, we show every example twice. The users that agree in less than 75% of their own votes are discarded.

times = get\_schedule() x = random\_noise()

for t\_last, t\_cur in zip(times\[:-1\], times\[1:\]): if t\_cur ¡ t\_last: # Apply Equation 8 (Main Paper) x = reverse\_diffusion(x, t, x\_known) else: # Apply Equation 1 (Main Paper) x = forward\_diffusion(x, t)

Figure 10: Inference Process. Pseudo code of RePaint inference process using a precalculated time schedule.

## B Algorithm for jump size larger than one

In addition to the resampling introduced in Algorithm 1 in the main paper, we use jumps in diffusion time as described in Section 4.2 in the main paper. Figure 8 shows a pseudo-code to further clarify the generation of state transitions. Note that each transition increases or decreases the diffusion time $t$ by one. For example, for a chosen jump length of $j=10$ shown in Figure 10, we apply ten forward transitions before applying ten reverse transitions. The diffusion time $t$ for the latent vector $x_{t}$ is plotted in Figure 9.

<table><thead><tr><th>Datasets</th><th colspan="2">Wide</th><th colspan="2">Narrow</th><th colspan="2">Super-Resolve <math><semantics><mrow><mn>2</mn> <mo>×</mo></mrow> <annotation>2\times</annotation></semantics></math></th><th colspan="2">Altern. Lines</th><th colspan="2">Half</th><th colspan="2">Expand</th></tr><tr><th>Methods</th><th>LPIPS</th><th>Votes [%]</th><th>LPIPS</th><th>Votes [%]</th><th>LPIPS</th><th>Votes [%]</th><th>LPIPS</th><th>Votes [%]</th><th>LPIPS</th><th>Votes [%]</th><th>LPIPS</th><th>Votes [%]</th></tr></thead><tbody><tr><td>AOT <sup><a href="#fn:51">51</a></sup></td><td>0.112</td><td><math><semantics><mrow><mn>35.4</mn> <mo>±</mo> <mn>3.0</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>35.4</cn> <cn>3.0</cn></apply> <annotation>35.4\pm 3.0</annotation></semantics></math></td><td>0.062</td><td><math><semantics><mrow><mn>36.0</mn> <mo>±</mo> <mn>3.0</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>36.0</cn> <cn>3.0</cn></apply> <annotation>36.0\pm 3.0</annotation></semantics></math></td><td>0.560</td><td><math><semantics><mrow><mn>2.2</mn> <mo>±</mo> <mn>0.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>2.2</cn> <cn>0.9</cn></apply> <annotation>2.2\pm 0.9</annotation></semantics></math></td><td>0.399</td><td><math><semantics><mrow><mn>0.8</mn> <mo>±</mo> <mn>0.6</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>0.8</cn> <cn>0.6</cn></apply> <annotation>0.8\pm 0.6</annotation></semantics></math></td><td>0.263</td><td><math><semantics><mrow><mn>34.0</mn> <mo>±</mo> <mn>2.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>34.0</cn> <cn>2.9</cn></apply> <annotation>34.0\pm 2.9</annotation></semantics></math></td><td>0.686</td><td><math><semantics><mrow><mn>0.7</mn> <mo>±</mo> <mn>0.5</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>0.7</cn> <cn>0.5</cn></apply> <annotation>0.7\pm 0.5</annotation></semantics></math></td></tr><tr><td>DSI <sup><a href="#fn:33">33</a></sup></td><td>0.101</td><td><math><semantics><mrow><mn>27.4</mn> <mo>±</mo> <mn>2.8</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>27.4</cn> <cn>2.8</cn></apply> <annotation>27.4\pm 2.8</annotation></semantics></math></td><td>0.054</td><td><math><semantics><mrow><mn>33.1</mn> <mo>±</mo> <mn>2.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>33.1</cn> <cn>2.9</cn></apply> <annotation>33.1\pm 2.9</annotation></semantics></math></td><td>0.157</td><td><math><semantics><mrow><mn>8.4</mn> <mo>±</mo> <mn>1.7</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>8.4</cn> <cn>1.7</cn></apply> <annotation>8.4\pm 1.7</annotation></semantics></math></td><td>0.083</td><td><math><semantics><mrow><mn>6.9</mn> <mo>±</mo> <mn>1.6</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>6.9</cn> <cn>1.6</cn></apply> <annotation>6.9\pm 1.6</annotation></semantics></math></td><td>0.265</td><td><math><semantics><mrow><mn>33.7</mn> <mo>±</mo> <mn>2.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>33.7</cn> <cn>2.9</cn></apply> <annotation>33.7\pm 2.9</annotation></semantics></math></td><td>0.565</td><td><math><semantics><mrow><mn>13.8</mn> <mo>±</mo> <mn>2.1</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>13.8</cn> <cn>2.1</cn></apply> <annotation>13.8\pm 2.1</annotation></semantics></math></td></tr><tr><td>ICT <sup><a href="#fn:42">42</a></sup></td><td>0.101</td><td><math><semantics><mrow><mn>35.7</mn> <mo>±</mo> <mn>3.0</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>35.7</cn> <cn>3.0</cn></apply> <annotation>35.7\pm 3.0</annotation></semantics></math></td><td>0.057</td><td><math><semantics><mrow><mn>33.7</mn> <mo>±</mo> <mn>2.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>33.7</cn> <cn>2.9</cn></apply> <annotation>33.7\pm 2.9</annotation></semantics></math></td><td>0.776</td><td><math><semantics><mrow><mn>0.9</mn> <mo>±</mo> <mn>0.6</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>0.9</cn> <cn>0.6</cn></apply> <annotation>0.9\pm 0.6</annotation></semantics></math></td><td>0.672</td><td><math><semantics><mrow><mn>1.3</mn> <mo>±</mo> <mn>0.7</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>1.3</cn> <cn>0.7</cn></apply> <annotation>1.3\pm 0.7</annotation></semantics></math></td><td>0.256</td><td><math><semantics><mrow><mn>26.0</mn> <mo>±</mo> <mn>2.7</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>26.0</cn> <cn>2.7</cn></apply> <annotation>26.0\pm 2.7</annotation></semantics></math></td><td>0.554</td><td><math><semantics><mrow><mn>26.6</mn> <mo>±</mo> <mn>2.7</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>26.6</cn> <cn>2.7</cn></apply> <annotation>26.6\pm 2.7</annotation></semantics></math></td></tr><tr><td>Deep Fill v2 <sup><a href="#fn:47">47</a></sup></td><td>0.097</td><td><math><semantics><mrow><mn>29.7</mn> <mo>±</mo> <mn>2.8</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>29.7</cn> <cn>2.8</cn></apply> <annotation>29.7\pm 2.8</annotation></semantics></math></td><td>0.051</td><td><math><semantics><mrow><mn>33.0</mn> <mo>±</mo> <mn>2.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>33.0</cn> <cn>2.9</cn></apply> <annotation>33.0\pm 2.9</annotation></semantics></math></td><td>0.120</td><td><math><semantics><mrow><mn>15.8</mn> <mo>±</mo> <mn>2.3</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>15.8</cn> <cn>2.3</cn></apply> <annotation>15.8\pm 2.3</annotation></semantics></math></td><td>0.070</td><td><math><semantics><mrow><mn>15.4</mn> <mo>±</mo> <mn>2.2</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>15.4</cn> <cn>2.2</cn></apply> <annotation>15.4\pm 2.2</annotation></semantics></math></td><td>0.254</td><td><math><semantics><mrow><mn>32.8</mn> <mo>±</mo> <mn>2.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>32.8</cn> <cn>2.9</cn></apply> <annotation>32.8\pm 2.9</annotation></semantics></math></td><td>0.550</td><td><math><semantics><mrow><mn>12.9</mn> <mo>±</mo> <mn>2.1</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>12.9</cn> <cn>2.1</cn></apply> <annotation>12.9\pm 2.1</annotation></semantics></math></td></tr><tr><td>LaMa <sup><a href="#fn:40">40</a></sup></td><td>0.078</td><td><math><semantics><mrow><mn>47.7</mn> <mo>±</mo> <mn>3.1</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>47.7</cn> <cn>3.1</cn></apply> <annotation>47.7\pm 3.1</annotation></semantics></math></td><td>0.039</td><td><math><semantics><mrow><mn>43.3</mn> <mo>±</mo> <mn>3.1</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>43.3</cn> <cn>3.1</cn></apply> <annotation>43.3\pm 3.1</annotation></semantics></math></td><td>0.369</td><td><math><semantics><mrow><mn>7.5</mn> <mo>±</mo> <mn>1.6</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>7.5</cn> <cn>1.6</cn></apply> <annotation>7.5\pm 1.6</annotation></semantics></math></td><td>0.138</td><td><math><semantics><mrow><mn>21.5</mn> <mo>±</mo> <mn>2.6</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>21.5</cn> <cn>2.6</cn></apply> <annotation>21.5\pm 2.6</annotation></semantics></math></td><td>0.233</td><td><math><semantics><mrow><mn>34.0</mn> <mo>±</mo> <mn>2.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>34.0</cn> <cn>2.9</cn></apply> <annotation>34.0\pm 2.9</annotation></semantics></math></td><td>0.512</td><td><math><semantics><mrow><mn>39.4</mn> <mo>±</mo> <mn>3.0</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>39.4</cn> <cn>3.0</cn></apply> <annotation>39.4\pm 3.0</annotation></semantics></math></td></tr><tr><td>RePaint</td><td>0.105</td><td><em>Reference</em></td><td>0.044</td><td><em>Reference</em></td><td>0.099</td><td><em>Reference</em></td><td>0.051</td><td><em>Reference</em></td><td>0.286</td><td><em>Reference</em></td><td>0.615</td><td><em>Reference</em></td></tr></tbody></table>

Table 5: Places2 Quantitative Results. We compute the LPIPS (lower is better) and votes for five different mask settings. Votes refers to the ratio of votes in favor our RePaint.

## C Ablation

In addition to the quantitative analysis in Table 3 in the main paper, this section shows visual examples for different jump lengths $j$ and number or resamplings $r$. As discussed in Section 5.5 in the main paper, smaller jump lengths $j$ tend to produce blurrier images as shown in Figure 11, and an increased number or resamplings $r$ improves the overall image consistency.

## D Evaluation on Places2

For a more comprehensive experimental framework, in this section, we provide the second part of the benchmark proposed in LaMa [^40], which is over the Places2 [^56] dataset. The experiments on Places2 were conducted using an unconditional model that we trained for 300k iterations with batch size four on four V100, taking about six days in total. All other training settings were kept as originally [^7] used for ImageNet. The model checkpoint will be published. We will clarify these aspects and add further details in the paper. We use the same mask generation procedure and settings described in the main paper. The results shown in Table 5 are in line with those on CelebA and ImageNet in Table 1 of the main paper. RePaint outperforms all other methods for all masks with significance 95% except for one inconclusive case. This case is when comparing RePaint to LaMa on Wide Masks, where the users vote in 52.4% for RePaint, but the significance interval overlaps with the 50% border. The visual comparison on the and Wide and Narrow mask is shown in Figure 21. Moreover, the visual results further confirm the robustness against sparse masks as shown in Figure 22. The mask pattern is clearly visible in all competing methods, while RePaint shows better harmonization. Regarding large masks, RePaint is able to inpaint semantically meaningful content such as the companion in the Bar in the same age, and overall lightning conditions as shown in the second row of Figure 23.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/supplement/figures/jump_Mask_idx31_q97.jpg)

Figure 11: Ablation Study. Analysis of length of the jumps j 𝑗 and number of resamplings r 𝑟 on ImageNet validation set with LaMa 40 Benchmark mask setting Wide.

<table><thead><tr><th>Mask</th><th colspan="2">Wide</th><th colspan="2">Narrow</th><th colspan="2">SR 2x</th><th colspan="2">Alter. Lines</th><th colspan="2">Half</th><th colspan="2">Expand</th></tr><tr><th>Measure</th><th>LPIPS</th><th>DS</th><th>LPIPS</th><th>DS</th><th>LPIPS</th><th>DS</th><th>LPIPS</th><th>DS</th><th>LPIPS</th><th>DS</th><th>LPIPS</th><th>DS</th></tr></thead><tbody><tr><th>DSI <sup><a href="#fn:33">33</a></sup></th><td>0.0639</td><td>16.68</td><td>0.0454</td><td>18.74</td><td>0.1404</td><td>12.38</td><td>0.0591</td><td>4.78</td><td>0.2348</td><td>15.30</td><td>0.5458</td><td>14.33</td></tr><tr><th>ICT <sup><a href="#fn:42">42</a></sup></th><td>0.0596</td><td>15.77</td><td>0.0402</td><td>18.65</td><td>0.5427</td><td>8.70</td><td>0.3916</td><td>8.16</td><td>0.1817</td><td>16.40</td><td>0.4779</td><td>17.25</td></tr><tr><th>RePaint</th><td>0.0552</td><td>16.40</td><td>0.0337</td><td>23.79</td><td>0.0327</td><td>19.84</td><td>0.0106</td><td>23.00</td><td>0.1839</td><td>17.31</td><td>0.4832</td><td>17.11</td></tr></tbody></table>

Table 6: Diversity Score. The Diversity Score (DS) and LPIPS calculated on CelebA-HQ on various masks for 32 images.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/supplement/figures/paper_p256_test_genhalf_failure_q97.jpg)

Figure 12: Failure Cases on ImageNet. When applying RePaint trained on ImageNet for inpainting it is more likely to inpaint dogs, due to the data bias. Zoom-in for better details.

## E Diversity

For our quantitative evaluation in the main paper, we sample a single image per input. However, since our method is stochastic, we can sample from it. To compare the diversity among the stochastic methods, we use the Diversity Score as described in [^22] (higher is better). In contrast to the standard diversity metric [^33] [^42] that only computes the mean LPIPS across pair of outputs, this score is designed to describe meaningful diversity yet also weighting the overall performance in LPIPS. It aims at measuring the diversity of the generations inside the manifold of plausible predictions. In detail, too extreme predictions or failures are therefore penalized. As shown in Table 6, for “Wide” and “Half”, there is no method with both best LPIPS and Diversity Score and for “Expand” ICT beats RePaint by $0.81\%$ in Diversity Score and $1.1\%$ in LPIPS. RePaint is superior by a large margin in both LPIPS and Diversity Score for the thin structured masks “Narrow”, “Super-Resolution $2\times$ ”, and “Alternating Lines” to both ICT [^42] and DSI [^33].

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/supplement/figures/video_q97.jpg)

Figure 13: Video of Diffusion Process. In the attachment we show the video of the denoising diffusion process on the CelebA-HQ validation set.

## F Failure Cases

As depicted in Figure 12, RePaint sometimes confuses the semantic context and mixes non-matching objects. Our model on ImageNet seems to be biased towards inpainting dogs more frequently than expected. Since ImageNet has many different breeds of dogs for classification tasks, dogs are over-represented in the training set, hence our model bias.

## G Attached Video

To inspect the latent space of the diffusion space, we provide a video in the attachment as shown in the screenshot in Figure 13. There we show the Ground Truth and the latent space $x_{t}$ after every transition in the diffusion process. Note that the diffusion time $t$, shown on top, jumps up and down according to the following schedule: The jump length is $j=5$, and the number of resamplings is $r=9$. To focus more on the visually interesting part of the diffusion process we set the number of diffusion steps to $T=100$ and start resampling below $t=50$.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/figures/inet512_q97.jpg)

Figure 14: Visual results on ImageNet 512 × 512\\times 512 for thin mask.

## H Experiment on larger resolution

As shown in Figure 14, our inpainting method also works on pretrained model from [^7] for $512\times 512$. However, we were not able to conduct our full analysis on that resolution due to limited computational resources.

## I Additional Visual Results

We also provide additional visual examples for CelebA-HQ and ImageNet, comparing our approach to the same state-of-the-art methods as in the main paper. We show the results for Wide and Narrow masks in Figures 15 and 18, respectively, for the sparse masks “Super-Resolve $2\times$ ” and “Alternating Lines” in Figures 16 and 19 and for “Half” and “Expand” in Figures 17 and 20.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/supplement/figures/paper_c256_test_thick_q97.jpg)

Figure 15: CelebA-HQ Qualitative Results. Comparison against the state-of-the-art methods for face inpainting. Zoom for better details.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/supplement/figures/paper_c256_test_nn2_q97.jpg)

Figure 16: CelebA-HQ Qualitative Results. Comparison against the state-of-the-art methods for face inpainting. Zoom for better details.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/supplement/figures/paper_c256_test_genhalf_q97.jpg)

Figure 17: CelebA-HQ Qualitative Results. Comparison against the state-of-the-art methods for face inpainting. Zoom for better details.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/supplement/figures/paper_inet256_test_thick_q97.jpg)

Figure 18: ImageNet Qualitative Results. Comparison against the state-of-the-art methods for diverse inpainting. Zoom for better details.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/supplement/figures/paper_inet256_test_nn2_q97.jpg)

Figure 19: ImageNet Qualitative Results. Comparison against the state-of-the-art methods for diverse inpainting. Zoom for better details.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/supplement/figures/paper_inet256_test_genhalf_q97.jpg)

Figure 20: ImageNet Qualitative Results. Comparison against the state-of-the-art methods for diverse inpainting. Zoom for better details.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/supplement/figures/paper_p256_test_thick_q97.jpg)

Figure 21: Places2 Qualitative Results. Comparison against the state-of-the-art methods for diverse inpainting. Zoom for better details.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/supplement/figures/paper_p256_test_nn2_q97.jpg)

Figure 22: Places2 Qualitative Results. Comparison against the state-of-the-art methods for diverse inpainting. Zoom for better details.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2201.09865/assets/x6.jpg)

Figure 23: Places2 Qualitative Results. Comparison against the state-of-the-art methods for diverse inpainting. Zoom for better details.

[^1]: Coloma Ballester, Marcelo Bertalmio, Vicent Caselles, Guillermo Sapiro, and Joan Verdera. Filling-in by joint interpolation of vector fields and gray levels. IEEE transactions on image processing, 10(8):1200–1211, 2001.

[^2]: Marcelo Bertalmio, Guillermo Sapiro, Vincent Caselles, and Coloma Ballester. Image inpainting. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques, pages 417–424, 2000.

[^3]: Marcelo Bertalmio, Luminita Vese, Guillermo Sapiro, and Stanley Osher. Simultaneous structure and texture image inpainting. IEEE transactions on image processing, 12(8):882–889, 2003.

[^4]: Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018.

[^5]: Kelvin CK Chan, Xintao Wang, Xiangyu Xu, Jinwei Gu, and Chen Change Loy. Glean: Generative latent bank for large-factor image super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14245–14254, 2021.

[^6]: Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. Ilvr: Conditioning method for denoising diffusion probabilistic models. arXiv preprint arXiv:2108.02938, 2021.

[^7]: Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis. arXiv preprint arXiv:2105.05233, 2021.

[^8]: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

[^9]: Xiefan Guo, Hongyu Yang, and Di Huang. Image inpainting via conditional texture and structure dual generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14134–14143, 2021.

[^10]: James Hays and Alexei A Efros. Scene completion using millions of photographs. ACM Transactions on Graphics (ToG), 26(3):4–es, 2007.

[^11]: Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in Neural Information Processing Systems 30 (NIPS 2017), 2017.

[^12]: Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020.

[^13]: Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021.

[^14]: Seunghoon Hong, Xinchen Yan, Thomas Huang, and Honglak Lee. Learning hierarchical semantic image manipulation through structured representations. arXiv preprint arXiv:1808.07535, 2018.

[^15]: Zheng Hui, Jie Li, Xiumei Wang, and Xinbo Gao. Image fine-grained inpainting. arXiv preprint arXiv:2002.02609, 2020.

[^16]: Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa. Globally and locally consistent image completion. ACM Transactions on Graphics (ToG), 36(4):1–14, 2017.

[^17]: Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4401–4410, 2019.

[^18]: Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8110–8119, 2020.

[^19]: Guilin Liu, Fitsum A Reda, Kevin J Shih, Ting-Chun Wang, Andrew Tao, and Bryan Catanzaro. Image inpainting for irregular holes using partial convolutions. In Proceedings of the European Conference on Computer Vision (ECCV), pages 85–100, 2018.

[^20]: Hongyu Liu, Bin Jiang, Yibing Song, Wei Huang, and Chao Yang. Rethinking image inpainting via a mutual encoder-decoder with feature equalizations. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 725–741. Springer, 2020.

[^21]: Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), December 2015.

[^22]: Andreas Lugmayr, Martin Danelljan, and Radu Timofte. Ntire 2021 learning the super-resolution space challenge. In CVPRW, 2021.

[^23]: Eric Luhman and Troy Luhman. Denoising synthesis: A module for fast image synthesis using denoising-based models. Software Impacts, 9:100076, 2021.

[^24]: Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388, 2021.

[^25]: Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.

[^26]: Sachit Menon, Alexandru Damian, Shijia Hu, Nikhil Ravi, and Cynthia Rudin. Pulse: Self-supervised photo upsampling via latent space exploration of generative models. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, pages 2437–2445, 2020.

[^27]: Kamyar Nazeri, Eric Ng, Tony Joseph, Faisal Qureshi, and Mehran Ebrahimi. Edgeconnect: Structure guided image inpainting using edge prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, pages 0–0, 2019.

[^28]: Alex Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. arXiv preprint arXiv:2102.09672, 2021.

[^29]: Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.

[^30]: Evangelos Ntavelis, Andrés Romero, Siavash Bigdeli, Radu Timofte, Zheng Hui, Xiumei Wang, Xinbo Gao, Chajin Shin, Taeoh Kim, Hanbin Son, et al. Aim 2020 challenge on image extreme inpainting. In European Conference on Computer Vision, pages 716–741. Springer, 2020.

[^31]: Evangelos Ntavelis, Andrés Romero, Iason Kastanis, Luc Van Gool, and Radu Timofte. Sesame: semantic editing of scenes by adding, manipulating or erasing objects. In European Conference on Computer Vision, pages 394–411. Springer, 2020.

[^32]: Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2536–2544, 2016.

[^33]: Jialun Peng, Dong Liu, Songcen Xu, and Houqiang Li. Generating diverse structure for image inpainting with hierarchical vq-vae. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10775–10784, 2021.

[^34]: Yurui Ren, Xiaoming Yu, Ruonan Zhang, Thomas H Li, Shan Liu, and Ge Li. Structureflow: Image inpainting via structure-aware appearance flow. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 181–190, 2019.

[^35]: Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro, and Daniel Cohen-Or. Encoding in style: a stylegan encoder for image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2287–2296, 2021.

[^36]: Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3):211–252, 2015.

[^37]: Chitwan Saharia, William Chan, Huiwen Chang, Chris A Lee, Jonathan Ho, Tim Salimans, David J Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. arXiv preprint arXiv:2111.05826, 2021.

[^38]: Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics, 2015.

[^39]: Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations, 2020.

[^40]: Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. arXiv preprint arXiv:2109.07161, 2021.

[^41]: Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Deep image prior. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 9446–9454, 2018.

[^42]: Ziyu Wan, Jingbo Zhang, Dongdong Chen, and Jing Liao. High-fidelity pluralistic image completion with transformers. arXiv preprint arXiv:2103.14031, 2021.

[^43]: Wei Xiong, Jiahui Yu, Zhe Lin, Jimei Yang, Xin Lu, Connelly Barnes, and Jiebo Luo. Foreground-aware image inpainting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5840–5848, 2019.

[^44]: Shunxin Xu, Dong Liu, and Zhiwei Xiong. E2i: Generative inpainting from edge to image. IEEE Transactions on Circuits and Systems for Video Technology, 31(4):1308–1322, 2020.

[^45]: Fisher Yu and Vladlen Koltun. Multi-scale context aggregation by dilated convolutions. arXiv preprint arXiv:1511.07122, 2015.

[^46]: Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. Generative image inpainting with contextual attention. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5505–5514, 2018.

[^47]: Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. Generative image inpainting with contextual attention. arXiv preprint arXiv:1801.07892, 2018.

[^48]: Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. Free-form image inpainting with gated convolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4471–4480, 2019.

[^49]: Yingchen Yu, Fangneng Zhan, Rongliang Wu, Jianxiong Pan, Kaiwen Cui, Shijian Lu, Feiying Ma, Xuansong Xie, and Chunyan Miao. Diverse image inpainting with bidirectional and autoregressive transformers. In Proceedings of the 29th ACM International Conference on Multimedia, 2021.

[^50]: Yanhong Zeng, Jianlong Fu, Hongyang Chao, and Baining Guo. Learning pyramid-context encoder network for high-quality image inpainting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1486–1494, 2019.

[^51]: Yanhong Zeng, Jianlong Fu, Hongyang Chao, and Baining Guo. Aggregated contextual transformations for high-resolution image inpainting. arXiv preprint arXiv:2104.01431, 2021.

[^52]: Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. CVPR, 2018.

[^53]: Lei Zhao, Qihang Mo, Sihuan Lin, Zhizhong Wang, Zhiwen Zuo, Haibo Chen, Wei Xing, and Dongming Lu. Uctgan: Diverse image inpainting based on unsupervised cross-space translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5741–5750, 2020.

[^54]: Shengyu Zhao, Jonathan Cui, Yilun Sheng, Yue Dong, Xiao Liang, Eric I Chang, and Yan Xu. Large scale image completion via co-modulated generative adversarial networks. arXiv preprint arXiv:2103.10428, 2021.

[^55]: Chuanxia Zheng, Tat-Jen Cham, and Jianfei Cai. Pluralistic image completion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1438–1447, 2019.

[^56]: Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017.