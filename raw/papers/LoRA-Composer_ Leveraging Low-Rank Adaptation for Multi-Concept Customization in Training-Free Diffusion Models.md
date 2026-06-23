---
title: "LoRA-Composer: Leveraging Low-Rank Adaptation for Multi-Concept Customization in Training-Free Diffusion Models"
source: "https://ar5iv.labs.arxiv.org/html/2403.11627"
author:
published:
created: 2026-06-24
description: "Customization generation techniques have significantly advanced the synthesis of specific concepts across varied contexts. Multi-concept customization emerges as the challenging task within this domain. Existing approa…"
tags:
  - "clippings"
---
<sup>1</sup> <sup>2</sup> <sup>3</sup> <sup>4</sup>

Yang Yang 11    Wen Wang 11    Liang Peng 22    Chaotian Song 33    Yao Chen 33    Hengjia Li 11    Xiaolong Yang 44    Qinglin Lu 44    Deng Cai 11    Boxi Wu 33    Wei Liu 44

###### Abstract

Customization generation techniques have significantly advanced the synthesis of specific concepts across varied contexts. Multi-concept customization emerges as the challenging task within this domain. Existing approaches often rely on training a Low-Rank Adaptations (LoRA) fusion matrix of multiple LoRA to merge various concepts into a single image. However, we identify this straightforward method faces two major challenges: 1) concept confusion, which occurs when the model cannot preserve distinct individual characteristics, and 2) concept vanishing, where the model fails to generate the intended subjects. To address these issues, we introduce LoRA-Composer, a training-free framework designed for seamlessly integrating multiple LoRAs, thereby enhancing the harmony among different concepts within generated images. LoRA-Composer addresses concept vanishing through Concept Injection Constraints, enhancing concept visibility via an expanded cross-attention mechanism. To combat concept confusion, Concept Isolation Constraints are introduced, refining the self-attention computation. Furthermore, Latent Re-initialization is proposed to effectively stimulate concept-specific latent within designated regions. Our extensive testing showcases a notable enhancement in LoRA-Composer’s performance compared to standard baselines, especially when eliminating the image-based conditions like canny edge or pose estimations. Code is released at [https://github.com/Young98CN/LoRA\_Composer](https://github.com/Young98CN/LoRA_Composer).

###### Keywords:

Multi-Concept Customization LoRA Integration Training-Free Controllable Generation

## 1 Introduction

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.11627/assets/x1.png)

Figure 1: Our method distinguishes itself from Mix-of-Show 7 by eliminating the image-based conditions (the sketch shown above) and the requirement to train a LoRA fusion matrix. Furthermore, we highlight the limitations of Mix-of-Show through the demonstration of failure cases. In the top row, we illustrate two key issues: concept vanishing, marked by the absence of intended concepts in the image, and concept confusion, where the model mistakenly merges and confuses distinct concepts.

Diffusion models [^26] have significantly advanced the field of image generation, particularly in creating images that adhere to user-specific concepts. The progress made in customization models [^7] [^13] [^30] [^27] [^40] [^4] play an important role in enriching the landscape of image synthesis. As technologies for single concept customization evolve, users are presented with various methods to personalize content, ranging from fine-tuning U-Net [^27] [^13], modifying text embeddings [^5] [^18], to leveraging Low-Rank Adaptations (LoRA) [^10]. LoRA, in particular, serves as a versatile, plug-and-play module that enables users to customize their models to generate diverse and lifelike personal images. Its adaptability and accuracy in image generation have established LoRA as a preferred method for customization tasks, significantly influencing how the community approaches the creation of tailored visual content.

While LoRA excels in single-concept customization, its application to emerging multi-concept customization tasks presents challenges. Recent developments have explored the integration of multiple-concept LoRAs to infuse images with diverse concepts via fusion tuning [^7] [^36]. However, as illustrated in Fig. 1, these integration strategies often necessitate a variety of conditions, including textual and image-based inputs [^42] (such as human pose estimations, and canny edge detection). This approach introduces constraints on variation and flexibility. Furthermore, in the process of combining multiple LoRAs, prior research [^7] [^36] [^32] focused on training a fusion ratio matrix. However, adjusting LoRA weights in this manner can exacerbate two problems: 1) concept vanishing, where the concept fails to be injected into the figure; and 2) concept confusion, where the model struggles to associate attributes with subjects or fails to capture distinct concept characteristics. Examples illustrating the issues mentioned are displayed in the top row of Fig. 1, showcasing outputs from the representative method Mix-of-Show [^7]. The left column shows a clear case of concept vanishing, where the model fails to generate one of the anime girls and one of the dogs (highlighted within the red box). The right column highlights issues with incorrect attribute binding, such as the dog’s color being mistaken, and the anime figure’s characteristics not being distinctly expressed (indicated within the blue box).

To overcome existing challenges, we introduce LoRA-Composer, a training-free framework that enables the synthesis of images with multiple integrated concepts, utilizing textual and layout cues. LoRA-Composer encompasses three principal components: Concept Injection Constraints, Concept Isolation Constraints and Latent Re-initialization. The Concept Injection Constraints introduce a novel cross-attention mechanism, consisting of: 1) Region-Aware LoRA Injection, which injects concept-specific LoRA features into designated regions through cross-attention, facilitating the seamless integration of multiple LoRAs within the diffusion model framework without the need for fusion fine-tune. 2) Concept Enhancement Constraints, which guide the refinement of latents to accentuate concepts in user-specified regions. This strategies enable the model to concentrate on areas designated for concept insertion, effectively mitigating the issue of concept vanishing. The Concept Isolation Constraints specifically address the issue of concept fusion by concentrating on self-attention. They implement restrictions to guarantee that each concept maintains its unique characteristics. This plays a crucial role in guiding the update process within the denoised latent space, maintaining alignment with each concept’s unique features and corresponding textual descriptions. Due to traditional single-concept LoRA training typically covers the entire image generation, which can be restrictive for localized area generation. We propose re-initializing the latent vector to establish a more accurate prior, directing the model’s focus on specific areas of the image.

We subjected our LoRA-Composer to rigorous testing across a broad spectrum of multi-concept customization scenarios, spanning categories from pets to characters and scenic backgrounds. Our approach displayed a strong performance compared to existing benchmarks through comprehensive qualitative and quantitative assessments.

In summary, our contributions are as follows:

- We propose a training-free model for integrating multiple LoRAs called LoRA-Composer. It requires only readily accessible conditions: layout and textual prompts. This approach simplifies the process of combining various concepts into a cohesive image, enhancing the convincing for users.
- To tackle concept vanishing and confusion, we implement Concept Injection Constraints and Concept Isolation Constraints. These strategies enhance the attention mechanism in U-Net, enabling the model to concentrate on the characteristics of individual concepts and prevent interference from the background or other concepts.
- We propose Latent Re-initialization to obtain a better prior enhancing the model’s capability to focus on specific image sections.
- Our extensive evaluations reveal that our method exceeds baseline performance, particularly in scenario that eliminate image-based conditions.

## 2 Related Work

### 2.1 Controllable Image Generation

Diffusion models [^9] [^33] trained on large-scale text-to-image datasets, like DALLE-2 [^25], Imagen [^29], and Stable Diffusion [^26], SDXL [^23] can produce text-aligned and diverse images in unprecedented high quality. To further support image generation from fine-grained spatial conditions, like sketches, human keypoints, semantic maps, etc., ControlNet [^42] finetunes a trainable copy of the pre-trained U-Net and connects the new layers and original U-Net weights with zero convolutions. A similar work, T2I-Adaptor [^20], finetunes lightweight adaptors for conditional generation from spatial conditions. Differently, GLIGEN [^15] considers controllable generation with sparse box layout conditions and injects a gated self-attention for fine-tuning. While these works rely on self-collected datasets of condition-image pairs for finetuning, recent works [^39] [^22] seek to explore test-time optimization for zero-shot controllable generation. For example, both BoxDiff [^39] and Attention Refocusing [^22] achieve zero-shot layout conditioned generation, by maximizing the attention weights between the features inside the box and its corresponding text description, while discouraging the latent features outside the box from attending to the text.

### 2.2 Single-Concept Customization

Concept customization aims at generating concepts specified by a few input images. Several represented methods [^27] [^5] [^34] [^31] [^37] [^8] [^21] focus on the customization of a single concept. For example, Dreambooth fine-tunes the pre-trained diffusion model on a few samples of the custom concept, and introduces a prior-preservation loss to alleviate overfitting. Differently, Textual Inversion [^5] fixes the pre-trained weights and only tunes a learnable text embedding to bind the novel concept. To further improve the capacity of the concept representation, P+ [^35] and NeTI [^1] learn a layer-wise text embedding or an implicit MLP for concept binding, respectively. Another line of work [^14] [^28] [^6] [^11] [^37] makes attempts to train a tailored model on large-scale image datasets for fast customization. For example, BLIP-Diffusion [^14] takes both the text prompt and subject image as inputs and learns on significant amount of data pairs to generate the customized images. Differently, HyperDreamBooth [^28] first collects Low-Rank Adaptation (LoRA) [^10] weights of massive custom concepts, then trains the model to predict the LoRA weights from input concept image.

### 2.3 Multi-Concept Customization

Different from the above work that focuses on single-concept customization, several recent works [^13] [^16] [^17] [^2] [^38] tackle the more challenging generic setting of multiple concepts. A pioneer work, Custom Diffusion [^13], jointly finetunes multiple concept images for customization. Cones [^16] finds concept-related neurons in pre-trained diffusion models for multi-concept customization. Cone 2 [^17] further incorporates residual token embedding and layout conditioning for better generation quality. To accelerate the customized generation, FastComposer [^38] finetunes diffusion model on massive data to take subject embedding as input and generate the composed image of multiple concepts. Similarly, Paint-by-Example [^40] and AnyDoor [^4] are trained on a significant amount of images and can achieve multi-concept generation through image inpainting. Considering the widespread utilization of LoRA for customization, several recent works [^7] [^36] [^30] [^44] seek to achieve multi-concept customization by combining multiple LoRA weights of individual concepts. For example, Mix-of-show [^7] proposes gradient fusion to train a composed LoRA weight that mimics the prediction of individual LoRAs. It further leverages T2I-Adaptor [^20] and sketches or keypoints for final generation. A concurrent work [^44] proposes two variants, namely LoRA Swich and LoRA Composite, to realize LoRA merge during decoding. The former uses multiple LoRA weights sequentially, while the latter averages the latent obtained from different LoRA weights. However, they focus on the combination of a single foreground and a single background LoRA weights. By contrast, we tackle the more challenging task of customizing multiple foreground characters, facing severe issues of concept confusion and vanishing.

Probably the most similar work to ours is Mix-of-Show, but we emphasize the following differences. Firstly, Mix-of-Show requires repeated gradient fusion training for each combination of multiple concepts, while we achieve this on the fly without retraining the LoRA weight. Secondly, Mix-of-Show requires additional image-based conditions, like sketches and keypoints, as input for high-quality image generation, which could be difficult to obtain.

## 3 Method

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.11627/assets/x2.png)

Figure 2: (a) LoRA-Composer utilizes textual, layout, and image-based conditions (optional) to integrate and customize multiple concepts through Latent Re-initialization for precise layout generation. (b) Modifications to the Stable Diffusion U-Net in LoRA-Composer Block include concept isolation in self-attention and Concept Injection in cross-attention, optimizing for accurate concept placement while preventing feature leakage across concepts.

In this section, we commence with a concise overview of diffusion models and ED-LoRA in Sec. 3.1. Following this, we introduce our innovative LoRA-Composer approach in Sec. 3.2, with the pipeline depicted in Fig. 2(a). The key point is augmenting the scalability of LoRA through the utilization of the LoRA-Composer Block, as illustrated in Fig. 2(b). We then delve into the specifics of the two primary components of the LoRA-Composer Block, outlined in Sec. 3.3 and Sec. 3.4, respectively. Finally, in Sec. 3.5 we discuss the implementation of Latent Re-initialization to achieve a more refined layout generation prior. We emphasize integrating LoRAs to enable multi-concept customization within a single image, aiming for a solution that is both more flexible and scalable.

### 3.1 Preliminary

Diffusion Models are famous for their capacity to generate high-quality images. Their framework operates in two primary phases: the forward phase, where Gaussian noise is progressively added to an image until it fully conforms to a Gaussian distribution, and the reverse phase, which aims to reconstruct the original image from its noised condition. The reverse phase typically employs a U-Net architecture enhanced with text conditioning, enabling the synthesis of images based on textual descriptions during inference. In this work, we employ Stable Diffusion [^26], which distinguishes itself by operating in the latent space rather than directly manipulating image pixels through these phases. This approach involves an autoencoder, with an encoder $\mathcal{E}$ and decoder $D$, trained to serve as a bridge between image pixel space $x$ and latent space $z$, i.e., $D(z)=D(\mathcal{E}(x))$. In each time step $t$, given a textual condition $\tau(P)$ and an image $x$, where $P$ represents the text prompt and $\tau$ denotes the pre-trained CLIP text encoder [^24]. The training objective for Stable Diffusion is to minimize the denoising objective by

$$
\mathcal{L}=\mathbb{E}_{z\sim\mathcal{E}(x),y,\epsilon\sim\mathcal{N}(0,1),t}\left[\|\epsilon-\epsilon_{\theta}(z_{t},t,\tau(y))\|_{2}^{2}\right],
$$

where $\epsilon_{\theta}$ is the denoising unet with learnable parameter $\theta$.

ED-LoRA [^7] aims to augment the expressiveness of the embedding by employing a decomposed structure. ED-LoRA implements a layer-wise embedding strategy, following the method described in P+ [^35], to forge a multi-faceted representation for the concept token ($V=V_{rand}V_{class}$). This involves adding a random variation ($V_{rand}$) and a class-specific component to the base embedding ($V_{class}$). Furthermore, it integrates a LoRA layer into the linear layers present in all attention modules of the text encoder and U-Net. This integration allows for a flexible adaptation of the model to specific concepts by modifying the linear layers in a low-rank manner, thereby enhancing the model’s ability to encode and synthesize images based on textual descriptions with high fidelity. We use it by default in the following description.

### 3.2 LoRA-Composer Pipeline Overview

LoRA-Composer utilizes a standard LoRA approach for subject registration, facilitating seamless integration of diverse subjects without requiring retraining for LoRA fusion. Our approach to the intricate task of multi-subject customization unfolds in two steps: initially, we create an accurate representation of each subject, followed by their coherent combination into an image. The process begins by acquiring a LoRA model, which can be achieved through training or downloading one shared by the community. Following this, LoRA-Composer efficiently combines these acquired LoRAs into a unified, coherent image. Additionally, to further refine the model’s capability in managing multiple conditions simultaneously, we provide the option to incorporate image-based conditions, using T2I-Adapter [^20] or ControlNet [^42].

Our primary contribution is the introduction of the LoRA-Composer Block, depicted in Fig. 2(b). In this innovation, we have re-designed the attention block within the U-Net architecture. Specifically, we preserved the residual block structure while adapting both the self-attention and cross-attention mechanisms. In the self-attention layer, we introduce Concept Isolation Constraints to effectively segregate different concepts, ensuring their distinctiveness. Concurrently, within the cross-attention mechanism, we implement Concept Injection Constraints designed to counteract concept vanishing. These strategies enable the refinement of the latent space into an image customized according to user preferences, utilizing both self-attention and cross-attention maps to direct this process effectively.

### 3.3 Concept Injection Constraints

Simply using text prompts to specify desired concepts may result in missing concepts in the basic Stable Diffusion [^3]. Although spatial attention guidance methods like BoxDiff [^39], Attend-and-Excite [^3], and Local control [^43] can mitigate the issue of missing objects in multi-concept generation, they fall short in precisely represent user-defined concepts. To tackle the challenge of accurately incorporating user-defined concepts, we introduce the concept injection constraint approach, which is comprised of two key components: Region-Aware LoRA Injection and Concept Enhancement Constraints.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.11627/assets/x3.png)

Figure 3: Modules of LoRA-Composer Block: (a) region-aware LoRA injection, (b)layout condition, (c) concept region mask.

Region-Aware LoRA Injection: Drawing inspiration from Regionally Controllable Sampling [^7], our method initiates with Region-Aware LoRA Injection. As illustrated in Fig. 3(a), upon receiving a layout condition,as shown in Fig. 3(b), we extract the queries, keys, and values.

$$
Q_{n}=M_{n}\odot W_{0}^{Q}(z),K_{n}=W_{n}^{K}(\tau_{n}(P^{n})),V_{n}=W_{n}^{V}(\tau_{n}(P^{n})),
$$

where $n\in[1,2,...]$ and $P^{n}$ denote the index of foreground concept LoRAs and the local prompt in a certain area respectively. Specifically, when $n=0$, $P^{n}$ indicates global prompt. The symbol $\odot$ denotes the Hadamard product, which identifies latent areas corresponding to the layout mask $M_{n}$. The symbols $W^{Q},W^{K}$, and $W^{V}$ stand for the projection matrices within the cross-attention modules of U-Net blocks which combined the concept LoRA, respectively. After replicating this process for each concept, we then update the hidden state through the cross-attention mechanism as follows:

$$
h_{n}=\text{softmax}\left(\frac{Q_{n}K_{n}}{\sqrt{d}}\right)V_{n},
$$

where the $d$ represents the dimension of queries and keys. This injection approach ensures a comprehensive integration of both background and foreground concepts, enhancing the model’s ability to accurately reflect user-specified concepts within the generated images.

Concept Enhancement Constraints: Building on the methodology presented by BoxDiff [^39], to ensure that synthesized objects accurately align with user-defined locations, a direct method involves confining high responses within the cross-attention mechanism to specified mask regions. Additionally, to address the issue of object generation tending towards the corners of the layout box, we incorporate a Gaussian weight into the cross-attention map sampling.

$$
\mathcal{L}_{ce}=\sum_{n=1}^{n}(1-\frac{1}{S}\sum^{S}_{S=0}\textbf{topk}(A^{c}_{n}\odot M_{n}\odot G,S)),
$$

where $\textbf{topk}(\cdot,S)$ indicates the selection of $S$ elements with the highest response in input. $A^{c}_{n}$ represents the position $c$ of the concept token in the cross-attention map $A$ for a specific concept. $G$ means standard Gaussian distribution. Moreover, we also propose the loss at the projection of the x-axis and y-axis, respectively, to ensure the concepts fill the entire box. First, we squeeze cross-attention maps on the w-axis and h-axis via the max operation as below:

$$
a^{c}_{n}(j)=\mathbf{Max}_{i=1,2,\dots,h}{A^{c}_{n}(i,j)},
$$
 
$$
a^{c}_{n}(i)=\mathbf{Max}_{j=1,2,\dots,w}{A^{c}_{n}(i,j)}.
$$

Then we compute the L1 loss in each axis as follows:

$$
\mathcal{L}_{fill}=\sum_{n=1}^{n}\frac{1}{K}(\sum(\mathbf{1}-\{M_{n}\odot a^{c}_{n}(j),M_{n}\odot a^{c}_{n}(i)\})),
$$

where $\{\cdot,\cdot\}$ denotes the concatenation operation and $\mathbf{1}$ represents a vector of ones. The term $K$ is defined as the length of $\mathbf{1}$.

### 3.4 Concept Isolation Constraints

While the Concept Injection Constraints effectively guarantee that objects will be placed within user-specified regions, it does not prevent the potential overlap or “infection” of customized concepts within these areas, as illustrated in Fig. 6(d). To preserve the integrity and distinctiveness of each concept within its designated region, we introduce Concept Isolation Constraints. This approach is divided into two main components: Concept Region Mask and Region Perceptual Restriction. Both elements are integrated within the self-attention mechanism of the U-Net block, ensuring that each concept remains isolated and unaffected by others, thereby maintaining the purity of concepts in their target regions.

Concept Region Mask: The self-attention mechanism creates connections among all query elements, essential for maintaining the distinct characteristics of each concept. To preserve the distinctiveness of each concept, we adopt the concept region mask strategy, guided by a given layout condition as depicted in Fig. 3(b). This methodology limits the interaction between queries within a specific concept region and those in other concept regions, as demonstrated in Fig. 3(c). Consequently, this ensures the preservation of each concept’s unique characteristics, free from the influence of neighboring concepts.

Region Perceptual Restriction: Due to down-sampling and residual convolution operations in the Stable Diffusion framework, concept features might spread into the elements designated for background areas, as highlighted by the yellow square in Fig. 3(b). To mitigate the risk of concept feature leakage into unintended regions, we employ Region Perceptual Restriction, aimed at minimizing interaction between queries of the foreground and background areas. This technique ensures that each concept remains distinct and unaffected by the features of the background feature, thereby preserving the uniqueness and integrity of each concept within the synthesized image. This formulated as

$$
\mathcal{L}_{region}=\sum_{n=1}^{n}(\frac{1}{P}\sum^{P}_{P=0}\textbf{topk}(\bar{A}_{[M_{n},\mathbf{1}-{M}_{n}]},P)),
$$

where the $\bar{A}_{[M_{n},\mathbf{1}-{M}_{n}]}$ refers to the self-attention map obtained through a matrix slicing operation across the channel dimension.

At each timestep, overall constraints loss are formulated as:

$$
\mathcal{L}=\mathcal{L}_{ce}+\alpha\mathcal{L}_{fill}+\beta\mathcal{L}_{region},
$$

where $\alpha$ and $\beta$ represent weighting coefficients. Using the constraints loss $\mathcal{L}$, the current latent $z_{t}$ can be updated with a step size of $\phi_{t}$ as follow:

$$
z^{\prime}_{t}\leftarrow z_{t}-\phi_{t}\cdot\nabla\mathcal{L}.
$$

Following Boxdiff [^39], the step size $\phi_{t}$ decays linearly with each timestep. By incorporating the previously mentioned constraints, $z^{\prime}_{t}$ is directed at each timestep to foster the generation of customized concepts within designated locations, while preventing the leakage of concept features into areas associated with other concepts. Subsequently, $z^{\prime}_{t}$ is utilized as the input for the U-Net for the ensuing inference step $z^{\prime}_{t}\xrightarrow{U-Net}z_{t-1}$. This strategic guidance ensures the precise synthesis of target objects within the user-specified layout regions.

### 3.5 Latent Re-initialization

We discovered that traditional LoRA is not ideally suited for generating specific local areas, because it is trained on the generation of entire images. This discrepancy between training and inference processes can result in misalignment. To address this issue, we propose re-initializing the latent space to better accommodate the integration of concept-specific LoRAs.

Before the denoising phase, we re-initialize the latent space to better integrate concept-specific LoRAs. This begins with sampling a random latent vector from a Gaussian distribution, then applying the LoRA-Composer process, which performs one-step update using Eq. 10. Afterward, the cross-attention map $A_{n}^{t}$ is generated, based on the latent query mapping and the textual embedding mapping for each local prompt. The subsequent step focuses on identifying the area with the highest scores to replace the predefined concept mask area $M_{n}$. Specifically, the indices of area $\hat{i_{n}},\hat{j_{n}}$ of certain concept is determined as follows:

$$
\hat{i}_{n},\hat{j}_{n}=\underset{i\in 0,1\dots w,j\in 0,1\dots h}{\mathrm{arg\,max}}\ \sum_{i}^{i+\mathbb{W}}\sum_{j}^{j+\mathbb{H}}\textbf{crop}(A^{c}_{n},[i,j],\mathbb{W},\mathbb{H}).
$$

In this context, $\mathbb{W}$ and $\mathbb{H}$ represent the width and height of the concept mask $M_{n}$, respectively. The variables $w$ and $h$ denote the width and height of the cross-attention map $A^{c}_{n}$. The function $\textbf{crop}(A,[i,j],\mathbb{W},\mathbb{H})$ refers to cropping the attention map to a shape of $\mathbb{W},\mathbb{H}$ with the top-left coordinate at $[i,j]$. Finally, the latent is normalized to a standard Gaussian distribution to prepare for subsequent processing steps.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.11627/assets/x4.png)

Figure 4: Three highlights of LoRA-Composer: a) full image customization in multi-style; b) manipulating interactions and attributes; c) multi-condition control generation.

## 4 Experiments

### 4.1 Experimental Setup

Datasets. For a thorough evaluation of LoRA-Composer, follow Mix-of-Show [^7], we compile a dataset featuring characters, animals, and backgrounds in both realistic and anime styles, encompassing a total of 16 customized subjects. Through comprehensive experimentation across diverse subject combinations, we demonstrate the superior performance of our approach.

Evaluation metrics. Our evaluation of the LoRA-Composer approach utilizes two metrics for customized generation, as proposed in prior studies [^18] [^7] [^13]. (1) Image similarity: This metric assesses the visual resemblance between the generated images and the target subjects in the CLIP [^24] image embedding. In our multi-concept generation task, we compute the image similarity for the generated images about each target concept individually and then determine the average value. (2) Textual similarity: This metric evaluates the mean similarity between generated images and their associated textual prompts, utilizing the CLIP model [^24] for quantification. For our multi-concept generation task, for the foreground concepts, we first isolate each concept within the generated images. Subsequently, we extract every concept token $V$ in local prompt $P^{n}_{L}$ and substitute it with the concept’s class name before computing the textual similarity. When evaluating background concepts, we input the entire image. The final score is the average of these similarity values.

Baseline. To assess the quality of our generated images, we compare our approach against four leading competitors in the field. Cones2 [^18] leverages text embeddings to support arbitrary combinations of concepts without necessitating model refinement. Mix-of-Show [^7] employs gradient fusion to integrate multiple concepts into a base model seamlessly. Both Anydoor and Paint by Example facilitate multi-concept generation by utilizing networks trained specifically for inpainting tasks. More implementation details are shown in the Appendix.

### 4.2 Visualization Results

Our methodology enables extensive customization of the entire image, addressing both foreground and background components and accommodating a wide range of styles, from anime to realistic. This broad customization capability is illustrated in Fig. 4(a). showcase our approach’s versatility in adapting to artistic expressions. Additionally, our method enables precise manipulation of interactions and attributes, such as shaking hands, wearing hats, and holding teddy bears in the picture, directly through textual prompts. This capability is showcased in Fig. 4(b). Moreover, our framework is designed for flexibility, capable of generating images under multiple conditions. It adeptly integrates specific constraints such as edge detection or pose estimation to guide the image synthesis process. This capacity for accommodating additional image-based conditions, as detailed in Fig. 4(c), highlights the adaptability of our approach in meeting varied and complex generation requirements.

| Method | Anime-I | Anime-T | Real-I | Real-T | Mean-I | Mean-T |
| --- | --- | --- | --- | --- | --- | --- |
| Cones2 [^18] | 0.5940 | 0.5691 | 0.5924 | 0.5912 | 0.5932 | 0.5801 |
| Mix-of-Show [^7] | 0.6296 | 0.5741 | 0.6743 | 0.5970 | 0.6519 | 0.5856 |
| Anydoor [^4] | \- | \- | 0.6801 | 0.6410 | 0.6801 | 0.6410 |
| Paint by Example [^40] | \- | \- | 0.6888 | 0.6356 | 0.6888 | 0.6356 |
| LoRA-Composer | 0.8219 | 0.5945 | 0.7363 | 0.6293 | 0.7791 | 0.6119 |
| Mix-of-Show\* [^7] | 0.8238 | 0.6067 | 0.7097 | 0.6203 | 0.7668 | 0.6135 |
| LoRA-Composer\* | 0.8320 | 0.5981 | 0.7367 | 0.6314 | 0.7843 | 0.6147 |

Table 1: LoRA-Composer’s performance in generating anime and realistic style concepts is benchmarked against baseline models. T refers to textual similarity, I refers to image similarity, with an asterisk \* indicating the use of image-based conditions. The highest scores in each column are marked in bold for clarity.

### 4.3 Quantitative Results

As detailed in Tab. 1, our LoRA-Composer surpasses prior techniques in image similarity, showcasing its effectiveness across both anime and realistic styles. Conversely, inpainting-based methods such as Anydoor and Paint by Example exhibit higher text similarity. This is because these methods specialize in inserting subjects into user-defined locations through reference images, focusing more on aligning with textual descriptions. Our method achieves significant improvements over Cones2, thanks to the efficacy of our LoRA-Composer block. These data illustrate that our method excel in precisely drawing out target concepts.

To ensure fairness in our comparison, we established two settings: one without using image-based conditions and another incorporating them (indicated by \*). Our method consistently outperforms in both scenarios. We observed that image-based conditions play a crucial role in Mix-of-Show. Without these conditions, it faces severe drops in both image and textual similarity. In contrast, LoRA-Composer exhibits enhanced robustness and gains further advantage from image-based conditions, offering increased convenience to users.

### 4.4 Qualitative Comparison

LoRA-Composer is evaluated without using image-based conditions across four benchmarks in multi-concept customization scenarios. The results are displayed in Fig. 5. It can be seen that Cones2 [^18] faces concept confusion, unable to clearly capture concept characteristics. In contrast, inpainting-based models like Anydoor [^4] and Paint by Example [^40] retain certain concept features, such as glasses and gray curly hair as illustrated in the first row, but struggle with detailed attributes, especially facial features. Mix-of-Show [^7] also suffers from concept confusion, as seen in the first row where glasses are incorrectly placed on a man. Furthermore, in the second and third rows, two people are completely vanishing. Differently, our method successfully synthesizes images that accurately incorporate all subjects with their correct characteristics, showcasing enhanced performance in multi-concept synthesis and attribute accuracy.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.11627/assets/x5.png)

Figure 5: Qualitative comparison with baselines. For each case, we use the same seeds across all of the methods.

| CE | CI | LR | Anime-I | Anime-T | Real-I | Real-T | Mean-I | Mean-T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ✓ | ✓ | ✓ | 0.8031 | 0.5948 | 0.7299 | 0.6341 | 0.7715 | 0.6144 |
| ✓ | ✓ |  | 0.8024 | 0.5923 | 0.7350 | 0.6314 | 0.7687 | 0.6119 |
| ✓ |  |  | 0.7957 | 0.5899 | 0.7271 | 0.6250 | 0.7614 | 0.6075 |
|  |  |  | 0.6597 | 0.5725 | 0.6683 | 0.6105 | 0.6640 | 0.5915 |

Table 2: Ablation studies on various components. "LR" stands for Latent Re-initialization, "CI" denotes Concept Isolation Constraints, and "CE" signifies Concept Enhancement Constraints within Concept Injection Constraints.

### 4.5 Ablation Study

To demonstrate the efficacy of each component in our method, we choose a challenging scenario with five and four concepts for this section. As illustrated in Fig. 6(c), the positions of the anime girl and the woman within the red box diverge from the layout conditions outlined in Fig. 6(a). This discrepancy serves as evidence that omitting Latent Re-Initialization (LR) hampers the precise placement of concepts, mainly because of the lack of spatial priors. Subsequently, as shown in Fig. 6(d), eliminating the Concept Isolation Constraints (CI) results in the blending of concept characteristics, such as the confusion observed in the anime girl’s haircut and the distortion of the man’s hair color. Without CI, concepts begin to overlap and influence each other, resulting in a disruption of harmony and coherence in the overall image composition. Finally, as shown in Fig. 6(e), eliminating the Concept Enhancement Constraints (CE) results in the disappearance of concepts. However, thanks to the presence of Region-Aware LoRA Injection, the system retains the capability to insert concepts, albeit with reduced precision in their placement and representation. This highlights each element’s critical role in achieving precise and harmonious concept integration.

To substantiate our findings as non-coincidental, we conducted a comprehensive quantitative evaluation. As detailed in Tab. 2, our analysis demonstrates the individual and collective impacts of Concept Enhancement Constraints (CE), Latent Re-Initialization (LR), and Concept Isolation Constraints (CI) on model performance. CE lead to significant improvements across all performance metrics, showcasing its effectiveness in activating concepts, evidenced by an increase of around 0.1 in mean image similarity. LR further contributed to these enhancements by refining region-specific priors. CI played a crucial role in preserving the distinctiveness of concept traits and enhancing model robustness.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.11627/assets/x6.png)

Figure 6: Visualized results from ablation study on individual components. “L” stands for Latent Re-initialization, “CI” denotes Concept Isolation Constraints, and “CE” signifies Concept Enhancement Constraints within Concept Injection Constraints.

## 5 Conclusion

In this paper, we introduce LoRA-Composer, a novel approach designed to seamlessly integrate multiple concepts within a single image. We explore two prevalent issues in multi-concept customization: concept vanishing and concept confusion. To this end, we employ Concept Injection Constraints to combat concept vanishing, while Concept Isolation Constraints and Latent Re-initialization to alleviate concept confusion. Our experiments highlight the capability of LoRA-Composer to customize entire images, including both background and foreground elements, and to manipulate the interactions and attributes of various concepts through textual prompts. Compared to traditional methods, LoRA-Composer offers enhanced flexibility and usability, allowing users to generate images with fewer conditions and readily accessible LoRA techniques. Furthermore, we demonstrate the method’s ability to achieve high-fidelity combinations of multiple concepts, underscoring its practical utility in complex image-generation tasks.

## References

Supplementary Material

Considering the space limitation of the main text, we provided more results and discussion in this supplementary material, which is organized as follows:

- Section 0.A: implementation details of our approach and the baseline models.
- Section 0.B: more detailed experiments analysis and discussion.
	- Section 0.A.1: our default setting in experiments and ablation study.
	- Section 0.B.1: ablation study on Concept Enhancement Constraints (in Eq. 7) and Concept Isolation Constraints (in Eq. 8).
	- Section 0.B.2: comparison with Mix-of-Show, under the image-based conditions.
	- Section 0.B.3: assess the human preference between our method and baseline approaches.
	- Section 0.B.4: more visual results of LoRA-Composer.
- Section 0.C: discussion of our method’s potential negative society impact.
- Section 0.D: failure cases and discussion.

## Appendix 0.A Implementation Detail

Pretrained Models. Following the approach used by Mix-of-Show [^7], we select Chilloutmix <sup>1</sup> as the pre-trained model for crafting real-world concept images. For anime concepts, Anything-v4 <sup>2</sup> serves as the chosen pre-trained model. To ensure equitable comparisons among different methods, all comparative analyses involving training-based methods, such as Cones2 [^18] and Mix-of-Show [^7], utilize these specified pre-trained models, guaranteeing uniformity in evaluation criteria. For inpainting-based models, specifically Anydoor [^4] (which refines Stable Diffusion v2.1) and Paint by Example [^40] (which refines Stable Diffusion 1.4), we adhere to their official models.

ED-LoRA Setting. We chose ED-LoRA due to its strong capability in maintaining concept fidelity. In alignment with the single-concept ED-LoRA tuning guidelines from [^7], we integrate the LoRA layer into the linear layers of all attention modules within both the text encoder and U-Net, setting a rank ($r$) of 4 for all experiments. For optimization, we employ the Adam optimizer [^12], utilizing learning rates of 1e-5 for the text encoder and 1e-4 for U-Net tuning.

Sample Details. For all experiments and evaluations in this paper, we use the DPM-Solver [^19], implementing adaptive sampling steps to enhance computational efficiency. Specifically, if the loss (as described in Eq. 9) ceases to decrease, we stop the process, thereby accelerating the overall procedure. For Eq. 9, the relative coefficients are set as $\alpha=0.25$ and $\beta=0.8$.

### 0.A.1 Evaluation Setting

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.11627/assets/x7.png)

Figure 7: The datasets utilized for our model encompass a diverse range of concepts, including real-world objects, anime characters, and background scenes, totaling 16 distinct concepts.

Baseline Implementation Detail. For Cones2 [^18], we utilize the official implementation provided at the repository <sup>3</sup>. The specified training configurations are a batch size of 4, a learning rate of 5e-6, and a total of 4000 training steps. To ensure consistency across experiments, we employ the same seed for image generation. Given that Paint by Example [^40] <sup>4</sup> and Anydoor [^4] <sup>5</sup> focus exclusively on real object inpainting, we ensure a fair comparison by limiting the comparison to real-world concepts. Specifically, our approach involves initially generating the background image using our model with the same prompt and seeds, while omitting the foreground prompts. Subsequently, their models are employed to introduce the foreground concepts. For Mix-of-Show [^7], we utilize the same LoRAs for both real-world and anime concepts. We apply their gradient fusion technique <sup>6</sup> separately to each concept type. Consistency across experiments is ensured by using the same seed for image generation, allowing for a direct comparison of outcomes.

## Appendix 0.B Additional Experiments

We collect a diverse dataset featuring characters, animals, and backgrounds in both realistic and anime styles, encompassing 16 unique subjects (As shown in Fig. 7). To assess our model, we randomly picked three varied settings in two styles, testing combinations of two to five subjects. We produced 50 images for each setting, culminating in $2\times 3\times 4\times 50=1200$ images for an extensive performance review.

For our ablation study, we selected three challenge settings within both anime and realistic styles, involving four and five concepts. This approach yielded 600 images, offering a substantial dataset to examine the effects of different model components and settings on our framework’s capability.

### 0.B.1 More Ablation Study

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.11627/assets/x8.png)

Figure 8: More ablation study on Concept Enhancement Constraints (in Eq. 7) and Concept Isolation Constraints (in Eq. 8). The upper portion displays outcomes utilizing our full methodology, while the lower portion illustrates results with specific modules omitted, highlighting the significance of each component within our approach. The red boxes and the yellow boxes are used to accentuate the distinctions between the anime style and the real-world style, respectively.

In the main ablation study Sec. 4.5, we explored the synergistic effects of combining modules with similar functionalities. Here, we delve deeper with an extensive ablation study on our Concept Enhancement Constraints, as outlined in Eq. 7 (which includes Gaussian sample strategy and $\mathcal{L}_{fill}$) and Concept Isolation Constraints (incorporating the concept region mask and $\mathcal{L}_{region}$ in Eq. 8). These examinations aim to illuminate their contributions to model performance, as illustrated in Fig. 8. Specifically, in the first column, the absence of Gaussian sampling leads to the concepts not being accurately centered within their designated boxes. This lack of precision can even cause anime concepts to appear outside their intended boundaries, resulting in a loss of their unique identity traits. In the second column, without $\mathcal{L}_{fill}$, both anime and realistic figures fail to occupy their designated boxes completely, pointing to a deficiency in fully utilizing the allocated space. In the third column, we observe the issue of concept confusion, such as the merging of anime haircuts with traits of realistic figures, which indicates a loss of distinctiveness. This highlights the role of the concept region mask in safeguarding each concept’s unique attributes. In the last column, concept features, such as the anime boy’s haircut being influenced by another character, and an unintended dog appearing, suggest leakage into unintended areas, due to down-sampling in U-Net. The inclusion of $\mathcal{L}_{region}$ addresses this issue by ensuring concept generation within specified regions only. These strategies validate the essential roles played by the Concept Enhancement and Concept Isolation Constraints in maintaining concept integrity and precision within the generated images, significantly bolstering the model’s capability to produce conceptually coherent and visually accurate outputs.

### 0.B.2 More Comparison

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.11627/assets/x9.png)

Figure 9: Comparison with Mix-of-Show, where image-based conditions are applied. The yellow box emphasizes the issue of concept confusion, while red boxes underscore instances of concept vanishing.

To facilitate a fair comparison with Mix-of-Show [^7], we follow their default setting, applying identical image-based conditions for customizing multi-concept images using the same random seed. The results are illustrated in Fig. 9, where we display four distinct concept combinations in both anime and real-world styles. Our observations indicate that Mix-of-Show encounters challenges in maintaining identity features (indicated by yellow boxes) and achieving the completeness of concepts (highlighted by red boxes). However, our method effectively addresses these issues, delivering high-fidelity and coherent images to users, enhancing the overall quality and satisfaction with the generated content.

### 0.B.3 User Study

To assess the efficacy of our multi-object customization outcomes more accurately, we implemented a user study to capture human preferences. Following the approach utilized in Mix-of-Show [^7], participants evaluated the generated images based on two key metrics: 1) Text-to-Image Alignment: This assesses how well the textual description matches the generated image. 2) Image-to-Image Alignment: This examines the resemblance between the character in the generated image and the provided character reference image.

Participants rated each aspect on a scale from 1 to 5, where higher scores denote superior quality. To thoroughly gauge the performance across various multi-object customization scenarios, we included setups involving 2, 3, 4, and 5 customization concepts. The sequence of all image-question pairs was randomized before being presented to 25 different users for evaluation. Each user was tasked with rating a total of 60 questions. The study results are shown in Tab. 3. Across all scenarios, LoRA-Composer emerged as the preferred choice, receiving the highest score of votes. Notably, our method demonstrated significant strengths, especially in scenarios that required eliminating image-based conditions. These outcomes demonstrate the effectiveness of LoRA-Composer in the generation of multi-concept customized images.

| Method | Text-to-Image | Image-to-Image |  |
| --- | --- | --- | --- |
| Cones2 [^18] | 1.99 | 1.25 |  |
| Mix-of-Show [^7] | 3.13 | 2.58 |  |
| Anydoor [^4] | 2.73 | 2.07 |  |
| Paint by Example [^40] | 2.19 | 1.53 |  |
| LoRA-Composer | 4.25 | 4.02 |  |
| Mix-of-Show\* [^7] | 3.84 | 3.28 |  |
| LoRA-Composer\* | 4.23 | 3.78 |  |

Table 3: User study. The scores reflect user preferences, with higher values indicating better quality. It shows that our approach is favored by users for multi-concept customization, excelling in both image and text alignment. An asterisk \* denotes using image-based conditions. The highest scores in each column are marked in bold.

### 0.B.4 More Visual Results

In Fig. 10, we showcase an extended collection of images produced using our method. This display illustrates the superior flexibility and usability of LoRA-Composer, enabling users to create images under few conditions and utilizing easily accessible LoRA techniques. Additionally, our method’s capability to seamlessly blend multiple concepts into high-fidelity images showcases its effective application in multi-concept generation tasks.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.11627/assets/x10.png)

Figure 10: More results of our method in three configurations.

## Appendix 0.C Potential Negative Society Impact

This project is dedicated to offering the community an advanced tool for multi-concept image customization, empowering users to merge various concepts seamlessly to craft complex visuals. Nonetheless, there’s a risk that such a powerful framework could be misused by malicious parties to create deceptive interactions with real-world figures, posing potential harm to the public. To counteract these risks, one potential solution is implementing protective measures akin to those proposed in DUAW [^41], which introduces a universal adversarial watermark. This watermark is designed to interfere with the variational autoencoder’s function, thereby hindering the model’s ability to be exploited for malicious customization.

## Appendix 0.D Limitation

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.11627/assets/x11.png)

Figure 11: Limitation of LoRA-Composer. (a) Concept boundaries disappear. (b) Beyond layout boundaries.

The first limitation is about disappearing concept boundaries (Fig. 11(a)): This issue arises when the space between concepts is too small, causing potential overlap due to down-sampling. Increasing the spacing between concepts can alleviate this problem.

The second limitation concerns concepts beyond layout boundaries (Fig. 11(b)): Sometimes, foreground pixels might exceed their intended layout boundaries, a result of the model’s (Stable Diffusion’s [^26]) design to yield results based on common sense. Opting for a more reasonable layout strategy could remedy this.

The final limitation concerns inference efficiency, where a notable delay occurs due to the requirement to load various LoRAs checkpoints and perform backward computations for updating latent representations.

In future work, we aim to enhance the attention mechanism to overcome existing limitations and optimize the IO process to improve inference efficiency.

[^1]: Alaluf, Y., Richardson, E., Metzer, G., Cohen-Or, D.: A neural space-time representation for text-to-image personalization. arXiv preprint arXiv:2305.15391 (2023)

[^2]: Avrahami, O., Aberman, K., Fried, O., Cohen-Or, D., Lischinski, D.: Break-a-scene: Extracting multiple concepts from a single image. arXiv preprint arXiv:2305.16311 (2023)

[^3]: Chefer, H., Alaluf, Y., Vinker, Y., Wolf, L., Cohen-Or, D.: Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG) 42(4), 1–10 (2023)

[^4]: Chen, X., Huang, L., Liu, Y., Shen, Y., Zhao, D., Zhao, H.: Anydoor: Zero-shot object-level image customization. arXiv preprint arXiv:2307.09481 (2023)

[^5]: Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A.H., Chechik, G., Cohen-Or, D.: An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022)

[^6]: Gal, R., Arar, M., Atzmon, Y., Bermano, A.H., Chechik, G., Cohen-Or, D.: Encoder-based domain tuning for fast personalization of text-to-image models. ACM Transactions on Graphics (TOG) 42(4), 1–13 (2023)

[^7]: Gu, Y., Wang, X., Wu, J.Z., Shi, Y., Chen, Y., Fan, Z., Xiao, W., Zhao, R., Chang, S., Wu, W., Ge, Y., Shan, Y., Shou, M.Z.: Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models. NEURIPS (2023)

[^8]: Han, L., Li, Y., Zhang, H., Milanfar, P., Metaxas, D., Yang, F.: Svdiff: Compact parameter space for diffusion fine-tuning. arXiv preprint arXiv:2303.11305 (2023)

[^9]: Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)

[^10]: Hu, J.E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Chen, W.: Lora: Low-rank adaptation of large language models. International Conference on Learning Representations (2021)

[^11]: Jia, X., Zhao, Y., Chan, K.C., Li, Y., Zhang, H., Gong, B., Hou, T., Wang, H., Su, Y.C.: Taming encoder for zero fine-tuning image customization with text-to-image diffusion models. arXiv preprint arXiv:2304.02642 (2023)

[^12]: Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. CoRR abs/1412.6980 (2014), [https://api.semanticscholar.org/CorpusID:6628106](https://api.semanticscholar.org/CorpusID:6628106)

[^13]: Kumari, N., Zhang, B., Zhang, R., Shechtman, E., Zhu, J.Y.: Multi-concept customization of text-to-image diffusion. Computer Vision and Pattern Recognition (2022). https://doi.org/10.1109/CVPR52729.2023.00192

[^14]: Li, D., Li, J., Hoi, S.C.: Blip-diffusion: Pre-trained subject representation for controllable text-to-image generation and editing. arXiv preprint arXiv:2305.14720 (2023)

[^15]: Li, Y., Liu, H., Wu, Q., Mu, F., Yang, J., Gao, J., Li, C., Lee, Y.J.: Gligen: Open-set grounded text-to-image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22511–22521 (2023)

[^16]: Liu, Z., Feng, R., Zhu, K., Zhang, Y., Zheng, K., Liu, Y., Zhao, D., Zhou, J., Cao, Y.: Cones: Concept neurons in diffusion models for customized generation. arXiv preprint arXiv:2303.05125 (2023)

[^17]: Liu, Z., Zhang, Y., Shen, Y., Zheng, K., Zhu, K., Feng, R., Liu, Y., Zhao, D., Zhou, J., Cao, Y.: Cones 2: Customizable image synthesis with multiple subjects. arXiv preprint arXiv:2305.19327 (2023)

[^18]: Liu, Z., Zhang, Y., Shen, Y., Zheng, K., Zhu, K., Feng, R., Liu, Y., Zhao, D., Zhou, J., Cao, Y.: Customizable image synthesis with multiple subjects. In: Thirty-seventh Conference on Neural Information Processing Systems (2023), [https://openreview.net/forum?id=h3QNH3qeC3](https://openreview.net/forum?id=h3QNH3qeC3)

[^19]: Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.: Dpm-solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In: Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., Oh, A. (eds.) Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022 (2022), [http://papers.nips.cc/paper\_files/paper/2022/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html)

[^20]: Mou, C., Wang, X., Xie, L., Wu, Y., Zhang, J., Qi, Z., Shan, Y., Qie, X.: T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453 (2023)

[^21]: Nichol, A.Q., Dhariwal, P.: Improved denoising diffusion probabilistic models. In: International Conference on Machine Learning. pp. 8162–8171. PMLR (2021)

[^22]: Phung, Q., Ge, S., Huang, J.B.: Grounded text-to-image synthesis with attention refocusing. arXiv preprint arXiv:2306.05427 (2023)

[^23]: Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023)

[^24]: Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PMLR (2021)

[^25]: Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 1(2), 3 (2022)

[^26]: Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. Computer Vision and Pattern Recognition (2021). https://doi.org/10.1109/CVPR52688.2022.01042

[^27]: Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. Computer Vision and Pattern Recognition (2022). https://doi.org/10.1109/CVPR52729.2023.02155

[^28]: Ruiz, N., Li, Y., Jampani, V., Wei, W., Hou, T., Pritch, Y., Wadhwa, N., Rubinstein, M., Aberman, K.: Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. arXiv preprint arXiv:2307.06949 (2023)

[^29]: Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems 35, 36479–36494 (2022)

[^30]: Shah, V., Ruiz, N., Cole, F., Lu, E., Lazebnik, S., Li, Y., Jampani, V.: Ziplora: Any subject in any style by effectively merging loras. arXiv preprint arXiv: 2311.13600 (2023)

[^31]: Shi, J., Xiong, W., Lin, Z., Jung, H.J.: Instantbooth: Personalized text-to-image generation without test-time finetuning. arXiv preprint arXiv:2304.03411 (2023)

[^32]: Smith, J.S., Hsu, Y.C., Zhang, L., Hua, T., Kira, Z., Shen, Y., Jin, H.: Continual diffusion: Continual customization of text-to-image diffusion with c-lora. arXiv preprint arXiv:2304.06027 (2023)

[^33]: Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S.: Deep unsupervised learning using nonequilibrium thermodynamics. In: International conference on machine learning. pp. 2256–2265. PMLR (2015)

[^34]: Tewel, Y., Gal, R., Chechik, G., Atzmon, Y.: Key-locked rank one editing for text-to-image personalization. In: ACM SIGGRAPH 2023 Conference Proceedings. pp. 1–11 (2023)

[^35]: Voynov, A., Chu, Q., Cohen-Or, D., Aberman, K.: $p+$: Extended textual conditioning in text-to-image generation. arXiv preprint arXiv:2303.09522 (2023)

[^36]: Wang, W., Zhao, C., Chen, H., Chen, Z., Zheng, K., Shen, C.: Autostory: Generating diverse storytelling images with minimal human effort. arXiv preprint arXiv:2311.11243 (2023)

[^37]: Wei, Y., Zhang, Y., Ji, Z., Bai, J., Zhang, L., Zuo, W.: Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. arXiv preprint arXiv:2302.13848 (2023)

[^38]: Xiao, G., Yin, T., Freeman, W.T., Durand, F., Han, S.: Fastcomposer: Tuning-free multi-subject image generation with localized attention. arXiv preprint arXiv:2305.10431 (2023)

[^39]: Xie, J., Li, Y., Huang, Y., Liu, H., Zhang, W., Zheng, Y., Shou, M.Z.: Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7452–7461 (2023)

[^40]: Yang, B., Gu, S., Zhang, B., Zhang, T., Chen, X., Sun, X., Chen, D., Wen, F.: Paint by example: Exemplar-based image editing with diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18381–18391 (2023)

[^41]: Ye, X., Huang, H., An, J., Wang, Y.: Duaw: Data-free universal adversarial watermark against stable diffusion customization. arXiv preprint arXiv: 2308.09889 (2023)

[^42]: Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3836–3847 (2023)

[^43]: Zhao, Y., Peng, L., Yang, Y., Luo, Z., Li, H., Chen, Y., Zhao, W., Liu, W., Wu, B., et al.: Local conditional controlling for text-to-image diffusion models. arXiv preprint arXiv:2312.08768 (2023)

[^44]: Zhong, M., Shen, Y., Wang, S., Lu, Y., Jiao, Y., Ouyang, S., Yu, D., Han, J., Chen, W.: Multi-lora composition for image generation. arXiv preprint arXiv:2402.16843 (2024)