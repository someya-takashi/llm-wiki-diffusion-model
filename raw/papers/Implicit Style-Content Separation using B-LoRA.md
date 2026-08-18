---
title: "Implicit Style-Content Separation using B-LoRA"
source: "https://ar5iv.labs.arxiv.org/html/2403.14572"
author:
published:
created: 2026-08-19
description: "Image stylization involves manipulating the visual appearance and texture (style) of an image while preserving its underlying objects, structures, and concepts (content). The separation of style and content is essentia…"
tags:
  - "clippings"
---
Yarden Frenkel      Yael Vinker      Ariel Shamir      Daniel Cohen-Or    \[10pt\] Tel Aviv University      Reichman University\[7pt\] [https://B-LoRA.github.io/B-LoRA/](https://b-lora.github.io/B-LoRA/) \[-20pt\]

###### Abstract

Image stylization involves manipulating the visual appearance and texture (style) of an image while preserving its underlying objects, structures, and concepts (content). The separation of style and content is essential for manipulating the image’s style independently from its content, ensuring a harmonious and visually pleasing result. Achieving this separation requires a deep understanding of both the visual and semantic characteristics of images, often necessitating the training of specialized models or employing heavy optimization. In this paper, we introduce B-LoRA, a method that leverages LoRA (Low-Rank Adaptation) to implicitly separate the style and content components of a single image, facilitating various image stylization tasks. By analyzing the architecture of SDXL combined with LoRA, we find that jointly learning the LoRA weights of two specific blocks (referred to as B-LoRAs) achieves style-content separation that cannot be achieved by training each B-LoRA independently. Consolidating the training into only two blocks and separating style and content allows for significantly improving style manipulation and overcoming overfitting issues often associated with model fine-tuning. Once trained, the two B-LoRAs can be used as independent components to allow various image stylization tasks, including image style transfer, text-based image stylization, consistent style generation, and style-content mixing.

![[Uncaptioned image]](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/teaser_style_content_blora.png)

Figure 1: By implicitly decomposing a single image into its style and content representation captured by B-LoRA, we can perform high quality style-content mixing and even swapping the style and content between two stylized images.©The painting on the left is by Judith Kondor Mochary.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/sheleg_results.png)

Figure 2: Examples of image stylization generated with our approach. The content image is shown on the left. We show here three results of image style transfer based on a reference style, one (on the right) based on a guiding text prompt. Note that our method requires only a single image, and preserves the image’s content and structure well while applying the desired style.

### 1 Introduction

Image stylization is a well-established task in computer vision, and has been actively researched for many years [^23] [^17]. This task involves changing the style of an image following some style reference, which can be text-based or image-based while preserving its content. Content refers to the semantic information and structure of the image, while style often refers to visual features and patterns such as colors and textures [^49]. Image style manipulation is a highly challenging task, since style and content are strongly connected, leading to an inherent trade-off between style transformation and content preservation. On the other hand, many style manipulation tasks require a clear separation between style and content within an image.

In this paper, we present B-LoRA, a method for style-content separation of any given image. Our method distills the style and content from a single image to support various style manipulation applications.

In the realm of recent advancements in large language-vision models, existing approaches utilize the strong visual-semantic priors embedded within these models to facilitate style manipulation tasks. Common techniques involve fine-tuning a pre-trained text-to-image model to account for a new style or content [^45] [^25] [^20] [^4]. However, fine-tuned models often suffer from the inherent trade-off between style transformation and content preservation as they are prone to over-fitting. Unlike these methods, we unify the learning of style and content components by separating them per image (see Figure 1). This separation is performed by fitting a light-weight adapter (B-LoRA) that is less prone to over-fitting issues, and enables task flexibility, allowing for both text-based and reference style image conditions.

Our method utilizes LoRA (Low Rank Adaptation) [^25], which has emerged as a popular approach due to its high-quality results and space-time efficiency. LoRA incorporates optimizing external low-rank weight matrices for the attention layers of the base model, while the pretrained model weights remain “frozen”. After training, these matrices define the adapted model that can be used for the desired task. LoRA is often utilized for image stylization by fine-tuning the base model with respect to a set of images that can either represent the desired style or the desired content.

Specifically, we use LoRA with Stable Diffusion XL (SDXL) [^41], a recently introduced text-to-image diffusion model renowned for its powerful style learning capabilities. Through detailed analysis of various layers within SDXL and their effect on the adaptation procedure, we made a surprising discovery: two specific transformer blocks can be used to separate the style and content of an input image, and to easily control them distinctly in generated images. For clarification, in this paper, we define a block as a sequence of 10 consecutive attention layers.

Therefore, when provided with a single input image, we jointly optimize the LoRA weights corresponding to these two distinct transformer blocks with the objective of reconstructing the given image based on a provided text prompt. Since we only optimize the LoRA weights of these two transformer blocks, we refer to them as “B-LoRAs”. The crucial aspect is that these B-LoRAs are trained on a single image only, yet they successfully disentangle its style and content, thereby circumventing the notorious overfitting problem associated with common LoRA techniques. Our technique benefits from the innate style-content disentanglement within the layers of the architecture. Another advantage of our method is that the B-LoRAs can be easily used as separate components, allowing various challenging style manipulation tasks without requiring any additional training or fine-tuning. In particular, we demonstrate style transfer, text-guided style manipulation and consistent style-conditioned image generation (see Figure 2).

We note that recent attempts have been made to combine trained LoRAs of style and content to a unified model [^47]. This approach requires a new optimization process for each style-content combination. This is both time-consuming and raises challenges in achieving an effective trade-off between style transformation and content preservation. In contrast, our trained B-LoRAs can be easily re-plugged into a pre-trained model combined with other learned blocks from other reference images, without any further training.

We provide extensive evaluation of our method showing its advantages compared to alternative approaches that are often designed to achieve one of these tasks. Our method provides a practical and simple way for image stylization that can be broadly used with existing models.

### 2 Related Work

##### Style Transfer

Image style transfer is a longstanding challenge in computer vision [^13] [^23], aimed at altering the style of an image based on a given reference. With the progress of deep learning research, Neural Style Transfer (NST) approaches rely on deep features extracted from pre-trained networks to merge content and style [^17] [^30] [^31]. Subsequent GAN-based [^18] techniques were proposed to transfer images across domains, using either paired [^29] or unpaired [^61] [^32] [^38] image sets, yet they require domain-specific datasets and training.

Recent advancements in language-vision models and diffusion models have revolutionized the field of image stylization. Leveraging the vast knowledge encoded in pre-trained language-vision models, modern approaches explore zero-shot image stylization and editing [^37] [^34] [^57] [^10] [^14] [^11] [^39] [^5], where images are manipulated without additional fine-tuning or data adaptation by intervening in the generation process. Prompt-to-Prompt [^21] proposes an approach to edit generated images by manipulating their cross-attention maps. In Plug-and-Play [^50] the appearance of a content image is manipulated with respect to a given text prompt by adjusting spatial features from the guidance image via the self-attention mechanism. Cross Image Attention (CIA) [^2] presents a method to modify the image appearance based on a reference image through alterations in cross-attention mechanisms. While these approaches effectively transform the appearance of the content image, they may encounter challenges in transferring appearance between subjects with differing semantics.

StyleAligned [^22] utilizes attention features sharing combined with the AdaIN mechanism [^26] to achieve style alignment between a sequence of generated images. However, the method is not explicitly designed to control the content of the generated image, potentially resulting in style image structure leakage. Similarly, the lack of style-content separation is also evident in encoder-based methods, such as IP-Adapter [^58]. InstantStyle [^54] is a concurrent work to ours, aiming to improve IP-Adapter for image stylization tasks by injecting the CLIP embedding of the style image into specific blocks within SDXL. In our work, we decompose the style and the content and learn a separate representation for each.

##### Text-to-Image Personalization

In another line of work [^15] [^45] [^20] [^3] [^53] [^4], optimization techniques are proposed to extend pre-trained Text-to-Image models to support the generation of novel visual concepts, including both style and content, based on a small set of input images with the same concept. This allows utilizing the rich semantic-visual prior of pre-trained models for customized tasks such as producing images of a desired style. Existing methods employ either token optimization techniques [^15] [^56] [^60] [^53] [^52] [^1], fine-tuning the model’s weights [^45], or a combination of both [^3] [^4] [^7] [^6]. Token optimization requires longer training times and often results in sub-optimal reconstruction. While model fine-tuning provides better reconstruction, it consumes substantial memory and tends to overfit. To address the memory inefficiency, and to facilitate more efficient model fine-tuning, Parameter Efficient Fine-Tuning (PEFT) approaches have been proposed [^24] [^25] [^33]. StyleDrop [^48] utilizes Muse [^9] as a base model, and adjusts its styles to align with a reference image. StyleDrop trains a lightweight adapter layer at the end of each attention block within the transformer model. However, similar to StyleAligned [^22], their approach is designed for style adaptation, but for content preservation, another optimization is required. Among existing PEFT methods, Low-Rank Adaptation (LoRA) [^25] is a popular fine-tuning technique, widely used by researchers and practitioners for its versatility and high-quality results.

##### LoRA for Image Stylization

LoRA is often used for image stylization by fine-tuning a model to produce images of a desired style. Commonly, a LoRA is trained on a set of images, and then it is combined with control methods such as stylistic Concept-Sliders [^16] or ControlNet [^59] [^43], along with a text prompt to condition the generated image content. While LoRA-based approaches have demonstrated significant abilities in capturing style and content, two separate LoRA models are required for this task, and there is no trivial way to combine them. A common naïve approach is to combine two LoRAs by directly interpolating their weights [^46], relying on a manual search for the desired coefficients. Alternative approaches [^19] [^40] propose an optimization-based strategy to find the optimal coefficients for such a combination. However, they focus on combining two objects and not on image stylization tasks.

Recently, Shah et al. introduced ZipLoRA [^47], proposing to merge two individual LoRAs trained for style and content into a new ’zipped’ LoRA by learning mixing coefficients for their columns. This work is closely related to ours, as we also mix LoRA weights trained on different images to facilitate image stylization. However, ZipLoRA requires an additional optimization stage for each new combination of content and style, thereby restricting the flexibility of reusing trained LoRA weights, which is LoRA’s primary advantage. In contrast, our approach allows for the direct reuse of learned styles and contents without additional training, enhancing efficiency and versatility. Moreover, we demonstrate that our implicit approach is more robust to challenging styles and contents.

### 3 Preliminaries

##### SDXL Architecture

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/style_injection.png)

Figure 3: Illustration of SDXL architecture and our text-based analysis. To examine the effect of the i’th transformer block on the generated image, we inject a different text prompt p ^ \\hat{p} to it, while is injected into all other blocks.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/vis_inject_generation2.png)

Figure 4: Prompt injection effect on the generated image. On the left, we demonstrate how blocks 2 and 4 affect the content in the generated image (turning into a tiger), whereas the rightmost image shows that injecting p ^ \\hat{p} to a block i ≠ 2, 4 i\\neq 2,4 has no effect on the generated image. On the right we show how the fifth block controls the generated image’s color.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/layers_res2.png)

Figure 5: Comparison of training B-LoRAs for the input images shown on the left for W 0 2, 5 W\_{0}^{2},W\_{0}^{5} (middle) and 4 W\_{0}^{4},W\_{0}^{5} (right). For each pair of trained LoRA weights, we show the results of applying both together (to reconstruct the input image) and applying the content layer separately (i.e. using only Δ \\Delta W^{2} and \\Delta W^{4} ). The results demonstrate that better captures the fine details of the input object.

In our work, we utilize the recently introduced publicly available text-to-image Stable Diffusion XL (SDXL) [^41], which is an upgraded version of the known Stable Diffusion [^44]. Both models are types of latent diffusion models (LDM), where the diffusion process is applied in the latent space of a pre-trained image autoencoder. The SDXL architecture leverages a three times larger UNet backbone compared to Stable Diffusion. The UNet consists of a total number of $70$ attention layers. Each layer consists of a cross and self-attention. These attention layers are often referred to as attention blocks. In this paper, for clarity, we refer to them as layers so they are not confused with the larger transformer blocks we optimize. These attention layers are divided into $11$ transformer blocks where the first two and last three blocks are comprised of four and six attention layers, respectively. The six inner blocks consist of 10 attention layers each, as illustrated in Figure 3).

Text condition generation is also extended in SDXL in the following way: given a text prompt $y$, it is encoded twice, with both OpenCLIP ViT-bigG [^28] and CLIP ViT-L [^42]. The resulting embeddings are then concatenated to define the conditioning encoding $c$. Then this text embedding is fed into the cross-attention layers of the network, following the attention mechanism [^51].

Specifically, in each layer, the deep spatial features $x$ are projected to a query matrix $Q=l_{Q}(x)$, and the textual embedding is projected to a key matrix $K=l_{K}(c)$ and a value matrix $V=l_{V}(c)$ via learned linear projections $l_{Q},l_{K},l_{V}$. The attention maps are then defined by:

$$
A_{t}=Softmax(\frac{QK^{T}}{\sqrt{d}})V,
$$

where $d$ is the projection dimension of the keys and queries.

##### LoRA

Low-Rank Adaptation [^25] is a method for efficiently fine-tuning large pre-trained models for specific tasks or domains. LoRA has emerged as a very popular approach for fine-tuning pre-trained text-to-image diffusion models [^46] due to its high-quality results and efficiency.

Let us denote the weights of a pre-trained text-to-image diffusion model with $W_{0}$, and the learned residuals after fine-tuning the model for a specific task with $\Delta W$. The key idea in LoRA is that $\Delta W\in\mathbb{R}^{m\times n}$ can be decomposed into two low intrinsic rank matrices $B\in\mathbb{R}^{m\times r}$ and $A\in\mathbb{R}^{r\times n}$, such that $\Delta W=BA$, and the rank $r<<\min(m,n)$. During training, the original model weights $W_{0}$ remain frozen, and only $A$ and $B$ are updated. Thus, by the end of the training, we can obtain the tuned model weights by using $W=W_{0}+\Delta W$.

LoRA is commonly used in text-to-image diffusion models only in the cross and self-attention layers. As discussed, the attention mechanism in each layer relies on four projection matrices: $l_{Q}$, $l_{K}$, $l_{V}$, and $l_{\text{out}}$. The LoRA weights $\Delta W_{Q}$, $\Delta W_{K}$, $\Delta W_{V}$, and $\Delta W_{\text{out}}$ are optimized for each of these pre-trained matrices. We denote by $\Delta W$ the LoRA weights of all four matrices.

### 4 Method

Our objective is to decouple the style and content aspects of an input image $I$ into separate components, enabling both text-based and image-based stylization applications. Our approach harnesses the capabilities of a pre-trained SDXL text-to-image generation model [^41], known for its robustness in capturing stylistic features [^47]. We conduct an analysis of the SDXL architecture to gain insight into the contributions of individual layers to either the style or the content of the generated image. Guided by our observations, we employ LoRA [^25] to train update matrices of only two specific transformer blocks within the SDXL model. These matrices capture the representation of the content and the style of the input image and they suffice to facilitate a number of image stylization tasks.

#### 4.1 SDXL Architecture Analysis

Similar to previous works [^53] [^1] we examine the effect of different layers within the base text-to-image model on the generated image. We adopt a similar approach to the one proposed in Voynov et al. [^53]. The key idea is to inject a different text prompt into the cross-attention layers of one of the transformer blocks within SDXL. Then examine the similarity between the different prompts and the resulting image. If we only change the input prompt corresponding to the i’th block, and the i’th block dominates a certain quality of the generated image, this will be apparent in the resulting image. Specifically, we examine six intermediate transformer blocks $\{W_{0}^{1},..W_{0}^{6}\}$ of SDXL, each containing $10$ attention layers (see Figure 3). These layers have been selected based on previous works [^53] [^1], which demonstrate that they are most likely to affect the important visual properties of the generated images.

We define two random sets of text prompts $P_{content}$ and $P_{style}$ describing different objects with different colors. The prompts in $P_{content}$ are defined by placing random objects in the template text “A photo of a \<object>”. For $P_{style}$ we use the template “A photo of a \<color> \<object>”. The random objects and colors are generated with ChatGPT. Note that color is used as a proxy for style since we use CLIP [^42] to evaluate results (as will be described next), and we found CLIP to be a better indicator for changes in color than changes in style. We sample a pair of prompts $(p,\hat{p})\in P_{content}$ and $(p,\hat{p})\in P_{style}$ such that $p\neq\hat{p}$. For each pair $(\hat{p},p)$, we generate an image $I_{\hat{p}\rightarrow i,p\rightarrow j\neq i}$ by injecting the embedding of $\hat{p}$ to $W_{0}^{i}$ while injecting the embedding of $p$ to all other layers $W_{0}^{j},j\neq i$ (depicted in Figure 3). This is performed for each of the six transformer blocks we target, yielding six images per pair.

Next, to measure the effect of injecting $\hat{p}$ into the i’th block on the generated image, we estimate the following similarity score:

$$
\mathcal{C}(I_{\hat{p}\rightarrow i,p\rightarrow j},\hat{p})=sim(C_{I}(I_{\hat{p}\rightarrow i,p\rightarrow j}),C_{T}(\hat{p})),
$$

where $C_{I}(I_{\hat{p}\rightarrow i,p\rightarrow j})$ and $C_{T}(\hat{p})$ are the CLIP image embedding of the generated image, and the CLIP text embedding of the prompt, respectively. $sim(x,y)=\frac{x\cdot y}{||x||\cdot||y||}$ indicates the cosine similarity between the clip embeddings.

In total, we examined 400 pairs of content and style prompts and averaged the scores of each layer. The three topmost layers that show similarity to one type of prompt are $W_{0}^{2}$ and $W_{0}^{4}$ which dominate the content of the generated image, and $W_{0}^{5}$ which dominates its color. We visually demonstrate these conclusions in Figure 4. On the left, we show the effect of blocks 2 and 4 on the generated content. Note that $I_{\hat{p}\rightarrow 2,p\rightarrow j}$ and $I_{\hat{p}\rightarrow 4,p\rightarrow j}$ demonstrate that when “A photo of a tiger” is injected to only one block (2 or 4), while “A photo of a bunny” is injected to the rest of the blocks, the generated images depict a tiger, while in all other options, the generated image will depict a bunny. Similarly, on the right we show the effect of block 5 on the generated image’s color.

#### 4.2 LoRA-Based Separation with B-LoRA

While the observations above apply to a generated image, our goal is to examine if the layers we locate could be useful in capturing the content and style of a given input image $I$. To fine-tune the model to generate variations of our given image we utilize the LoRA [^25] approach.

Let us denote the frozen weights of our base pre-trained SDXL model with $W_{0}$ and the learned residual matrices for each block with $\Delta W^{i}$. We follow the default settings of DreamBooth LoRA [^46] to finetune the model to reconstruct the given input image $I$.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/apps_method1.png)

Figure 6: B-LoRA for Image Stylization. (1) To stylize a given content image I c I\_{c} w.r.t an given style image reference s I\_{s}, we train our B-LoRAs for both images and then combine Δ W 4 \\Delta W\_{c}^{4} and 5 \\Delta W\_{s}^{5} to a single adapted model. (2) For text-based stylization we simply plug only the trained to adapt the model and then use the desired text prompt during inference. (3) The learned style weights \\Delta W\_{c}^{5} can be also used as is to adjust the backbone model to produce images with the style of.

However, instead of optimizing the LoRA weights of all eleven blocks (as usually done), we conduct two experiments, where in the first experiment we optimize the pair { $\Delta W^{2},\Delta W^{5}$ }, and in the second experiment we optimize $\{\Delta W^{4},\Delta W^{5}\}$ (as we found $W_{0}^{2}$ and $W_{0}^{4}$ to dominate the content, and $W_{0}^{5}$ to dominate the color). In addition, we use a general prompt “A \[v\]” during training to prevent the model from being explicitly guided to capture either the image’s style or content. This process and example results are depicted in Figure 5. As can be seen, we find that the best combination to optimize in terms of 1. Achieving a full reconstruction of the input concept, and 2. Capturing the input image’s content, are $\Delta W^{4},\Delta W^{5}$. Note that using the deeper layers of the UNet $\Delta W^{4}$, rather than $\Delta W^{2}$ during the LoRA training process, aligns with the goal of preserving finer details in the output image, as demonstrated in [^50]. We provide ablation and analysis of the effect of other layers and specific parts within them, as well as the effect of using different text prompts in the supplementary material. We call such a training scheme *B-LoRA*, as it only trains two transformer Blocks instead of the full weights. Hence, apart from the style-content separation abilities such a method also reduces storage requirements by $70\%$.

#### 4.3 B-LoRA for Image Stylization

Combining the insights from the above analyses, we now describe the B-LoRA training approach. Given an input image $I$, we only fine-tune the LoRA weights $\Delta W^{4},\Delta W^{5}$ with the objective of reconstructing the image, w.r.t a general text prompt “A \[v\]”. Besides increasing efficiency, we find that by training only these two layers, we can achieve an implicit style-content decomposition, where $\Delta W^{4}$ captures the content and $\Delta W^{5}$ captures the style.

Once we find these update matrices, we can easily use them by updating the corresponding block weights of the pre-trained SDXL model for style manipulation applications as described next and demonstrated in Figure 6.

##### Image stylization based on image style reference

Given two input images $I_{c},I_{s}$ depicting the desired content and style respectively, we use the process described above to learn their corresponding B-LoRA weights: $\Delta W_{c}^{4},\Delta W_{c}^{5}$ for $I_{c}$ and $\Delta W_{s}^{4},\Delta W_{s}^{5}$ for $I_{s}$. We then directly use $\Delta W_{c}^{4}$ and $\Delta W_{s}^{5}$ to update the transformer blocks $W_{0}^{4}$ and $W_{0}^{5}$ of the pre-trained network. For the inference process, we use the prompt “A \[c\] in \[s\] style”, as illustrated at the top of Figure 6.

##### Text-based image stylization

By omitting $\Delta W_{c}^{5}$ (capturing the style of $I_{c}$) and only using $\Delta W_{c}^{4}$ to update the weights of the pre-trained model, we get a personalized model that is adapted to only the content of $I_{c}$. To manipulate the style of $I_{c}$ with text-based guidance, we simply inject the desired text into the adapted layers during inference (see Figure 6 bottom-left). Note that because the style and content are separated and encoded in different blocks, our approach allows challenging style manipulations.

Figure 7: Results produced by our method for three image stylization tasks. Rows 1-3: image style transfer. Our method can operate on scene images and extract content from a stylized image. Fourth row: text-based image stylization applied to the content image reference on the left. Note how the pose and identity are preserved well. Last row: consistent style generation, where that style is extracted from the image on the left and used to generate new objects. In this row, we use $\alpha=1.1$ to enhance the style effect.<sup>†</sup>

##### Consistent style generation

Lastly, in a similar manner, one can adapt the model for a specific style provided in $I_{s}$ by excluding $\Delta W_{s}^{4}$ and using only $\Delta W_{s}^{5}$. This results in a model adapted to the desired style, and one can use text-based conditions to generate any content with the desired style (see Figure 6 bottom-right).

#### 4.4 Implementation details

We train the B-LoRA weights on SDXL v1.0 [^41] while keeping both the model weights and text encoders frozen during the fine-tuning process. All LoRA training was performed on a single image. We utilize the Adam optimizer with a learning rate of $5e-5$. For data augmentations, we only use center cropping during training. We set the LoRA weights rank to $r=64$ and use the prompt “A \[v\]” for 1000 optimization steps, requiring approximately 10 minutes per image on a single A100 GPU. Note that while other methods typically train LoRA for 400 steps to mitigate overfitting concerns, this was not an issue in our case.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/temp_figs/content_images/cat.jpeg)

Figure 8: Comparison with alternative approaches. The input style and content references are shown on the left, where multiple content images were used for alternative methods. In the last row, we applied other approaches to a single content image. ZipLoRA tends to overfit the content, and thus struggles with depicting the desired style. StyleDrop also struggles to preserve the content when trained on multiple images. In the case of a single content image (last row), both methods preserve the content but lose the style. StyleAligned preserves the style well; however, it tends to include semantic content originating in the style image, such as creating a couple in row 1. Additional comparisons to InstantStyle 54 are provided in the supplementary material.

### 5 Results

To produce the various results of our approach we optimized our B-LoRAs ($\Delta W^{4},\Delta W^{5}$) once for each image and then plugged either one of them or both of them (depending on the application) at inference time to receive image stylization without any further optimization or fine-tuning.

We present some qualitative results of the three applications discussed in Section 4.3 in Figure 7. In the first two rows of Figure 7, our method manages to transfer the style of the image references (top row) while preserving the content of the input image on the left. Notable, this can be done for challenging content inputs such as stylized images (first row) and images of whole scenes (second row). Our method is robust to many types of different styles and manages to preserve the essence of the content reference even in very abstract styles such as the one depicted in the third style column. In the third row, we show examples of text-based image stylization. As can be seen with our implicit style-content separation, the content of the input object is preserved well while the style is governed by the desired text prompt. In the last row, we demonstrate how our method can be used for consistent style generation where only the B-LoRA weights of the style are used. Observe that the object’s style is well preserved across all text-based generated images. Please refer to the supplementary material for many more examples.

Table 1: Quantitative comparison. We measure the average cosine similarity between the DINO features of the output image and the reference style and content. Our method performs best at adapting to the style without overfitting the content image.

|  | Input | StyleDrop | StyleAligned | ZipLoRA | DB-LoRA | Ours |
| --- | --- | --- | --- | --- | --- | --- |
| StyleTransfer | MultipleSingle | $0.826\pm{0.07}$ $0.790\pm{0.06}$ | $0.855\pm{0.05}$ $0.829\pm{0.05}$ | $0.796\pm{0.07}$ $0.782\pm{0.05}$ | $0.863\pm{0.06}$ | $\boldsymbol{0.881}\pm{0.05}$ |
| Content | MultipleSingle | $0.817\pm{0.06}$ $0.874\pm{0.08}$ | $0.779\pm{0.05}$ $0.792\pm{0.06}$ | $\boldsymbol{0.841}\pm{0.05}$ $\boldsymbol{0.933}\pm{0.05}$ | $0.769\pm{0.05}$ | $0.790\pm{0.05}$ |

#### 5.1 Comparisons

We next compare our method with alternative approaches, both qualitatively and quantitatively. Note that since we rely on SDXL as our backbone model, for a fair comparison we applied alternative approaches on SDXL as well. As a naïve baseline we employ DB-LoRA [^46] (fine-tuned for style) with a ControlNet [^59] for content conditioning. We additionally compare to three recent approaches for image stylization that rely on the prior of large pre-trained text-to-image models, namely, ZipLoRA [^47], StyleDrop [^48], and StyleAligned [^22]. StyleAligned is applied using the author’s official implementation. With the lack of official implementations for StyleDrop and ZipLoRA, we implemented StyleDrop on SDXL (as described in [^22]), and utilized a non-official implementation of ZipLoRA [^36].

Note that for content preservation, all three alternative methods require *multiple* content image examples, while our method can be applied to a *single* image. Thus, for a fair comparison, we collected a total set of 23 objects from existing personalization works [^15] [^52] [^45] [^33], where a small set of images is provided for each object. We collected 20 style image references from [^22] [^48], along with 5 additional style images of our own. From these sets, we randomly sampled 50 pairs of style and content images to compose our final evaluation set.

In terms of runtime, StyleAligned is zero-shot only for consistent style generation, while for content preservation it relies on LoRA to adapt the model to the desired concepts. Similarly, StyleDrop and ZipLoRA require LoRA training for content and style. Thus, our runtime is comparable to theirs. In contrast, ZipLoRA entails an additional training phase to merge the two LoRAs, which makes it more time-consuming than our approach.

##### Qualitative Evaluation

We show representative comparison results in Figure 8, where on the left we show the style and content reference images. On the first four rows, we show the results of alternative approaches when applied with multiple content images, whereas our method uses a single image. As can be seen, our method effectively preserves the subject from the content image while transferring the desired style. In contrast, other methods either overfit the content subject, thereby failing to alter its style (e.g., cat and sloth in ZipLoRA and StyleDrop), or they suffer from style image “leakage”. For instance, in the cat example of StyleAligned (first row), the model generates two cats, matching the number of people in the style reference image. We also include an example of alternative methods applied to a single content image, where StyleDrop and ZipLoRA exhibit increased overfitting.

##### Quantitative Evaluation

We measure content and style preservation by computing the cosine similarity between the embeddings of the input content and style references and the output image, utilizing the DINO ViT-B/8 embeddings [^8]. The average scores are presented in Table 1. Our method achieves the highest style alignment score, indicating its superior ability to adapt styles effectively. However, we observe lower object similarity scores, possibly due to content overfitting issues observed in alternative approaches.

To further support this observation, we conducted the same experiment using a single content image as a reference (scores shown in the “Single” row). The results indicate a decrease in style consistency scores across all methods, accompanied by an increase in content preservation scores, suggesting overfitting.

##### User Study

We conducted a user study to further validate the findings presented above. Using $30$ random images from our evaluation set, we compared our results with the three alternative approaches. The participants were presented with the reference style and content images along with two combined results, one produced by our method and the other by an alternative method (with the results presented in random order). Participants were asked to choose the result that “better transfers the style from the style image while preserving the content of the content image”. We collected responses from $34$ participants for the survey, which contained a total of $1020$ answers. The results demonstrate a strong preference for our method, with 94% of participants favoring our method over StyleAligned, 91% over ZipLoRA, and 88% over StyleDrop.

### 6 Conclusions, Limitations and Future work

We have presented a simple yet effective method to disentangle the style and content of a single input image. The style and content components are encoded separately with two B-LoRAs, providing high flexibility for independent use in various image stylization tasks. In contrast to existing methods that focus on style extraction, we employ a compound style-content learning approach that enables a better separation of style and content, enhancing stylization fidelity. While our work enables robust image stylization across various complex input images, it does have limitations. First, in our style-content separation procedure, the object’s color is often included in the style component. However, in some cases, color plays a crucial role in preserving identity. Therefore, when stylizing the content component, the results may not properly preserve the object’s identity, as illustrated in Figure 9(a). Second, since we use a single reference image, our learned style component may encompass background elements rather than focusing solely on the central object, as demonstrated in Figure 9(b). Lastly, while our method effectively stylizes scene images, it may encounter challenges with complex scenes containing numerous elements. Consequently, it may struggle to accurately capture the scene structure, potentially compromising content preservation, as depicted in Figure 9(c).

As for future research, one possible avenue is to further explore separation techniques within LoRA fine-tuning, to achieve more concrete separation into sub-components such as structure, shape, color, texture, etc. This could provide users with more control over the desired output. Another direction for future work is to leverage the robustness of our approach and extend it to combine LoRA weights from multiple distinct objects or combine a few styles.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/limitations_fig.png)

Figure 9: Method limitations. (a) Sub-optimal identity preservation due to color separation. (b) Style leakage from background objects. (c) Inability to adequately capture content in complex scenes.

### 7 Acknowledgements

We would like to thank Amir Hertz and Yuval Alaluf for their insightful feedback. Additionally, some of the artistic paintings presented in this paper were created by the artist Judith Kondor Mochary. We thank the artist’s family for granting us the privilege to use Judith’s drawings. This work was supported by the Israel Science Foundation under Grant No. 2492/20, 3441/21 and 1390/19, and Joint NSFC-ISF Research Grant Research Grant no. 3077/23.

### References

Supplementary Material

### Appendix A Comparisons

##### User Study

As described in the main paper, we conducted a user study to further validate our findings. We constructed an evaluation set comprising 50 unique pairs of style and content images, randomly sampled from a diverse pool of 23 objects and 25 style references. From this evaluation set, we selected 10 representative pairs for each of the competing methods: ZipLoRA, StyleDrop, and StyleAligned. For each pair, we generated images using both the respective method and our approach, presenting them alongside the original style and content references, as illustrated in Figure 10 The generated images were displayed in a randomized order to avoid bias. Participants were asked to choose the result that “better transfers the style from the style image while preserving the content of the content image.” In total, we gathered 1020 responses from 34 participants, ensuring a comprehensive evaluation of our method against alternative approaches.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/sup_figs/user_study.png)

Figure 10: Screenshot from the user study. Each of the two images, labeled A and B, represents a result obtained from a different method. Participants were tasked with selecting the image they believe is better in terms of both style adaptation and content preservation.

##### Qualitative Comparisons

In Section 5 of the main paper, we conducted a comparison of our B-LoRA method against four state-of-the-art baselines for image stylization incorporating personalization [^27] [^47] [^48] [^22]. In this section, we delve deeper into the implementation details and present additional qualitative results. To begin, we employed DreamBooth-LoRA [^46] fine-tuning to obtain both style and content LoRAs, utilizing the same parameter configuration as ZipLoRA [^47]. For content images, we conducted fine-tuning across a set of images of the same object, except for the experiment involving a single image. However, for style LoRAs, we conducted fine-tuning using a single style image. We utilized the prompts provided in DreamBooth [^45] and StyleDrop [^48], specifically “A \[v\] \<object>” or “A \<object> in \[s\] style” for content and style, respectively. Subsequently, for ControlNet combined with DreamBooth-LoRA, we leveraged the publicly available implementation of ControlNet on SDXL from huggingface [^27]. this approach involved utilizing the style LoRAs we trained for style transfer while employing CannyEdge with thresholds of 100 and 200 for content guidance in ControlNet. For StyleDrop [^48], we followed the methodology outlined in StyleAligned [^22] for fine-tuning the model over the style images, followed by fusing the content LoRAs with the SDXL weights. Similarly, for StyleAligned [^22], we utilized the authors’ implementation for subject-driven generation alongside our content LoRAs. Lastly, for ZipLoRA [^47], we use the unofficial implementation [^36] with default parameters. We provide additional comparisons of our B-LoRA method against the aforementioned approaches using the same evaluation set presented in Section 5 of the main paper. These additional comparisons are illustrated in Figure 11. Furthermore, we provide comparisons with challenging content inputs, such as stylized images, presented in Figure 12. We also showcase comparisons with challenging style inputs, such as object images, in Figure 13. These examples demonstrate the robustness of our method in handling diverse and complex content and style references.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/temp_figs/content_images/child.jpg)

Figure 11: Additional comparisons for image stylization based on reference image.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/temp_figs/style_images/drawing1.jpg)

Figure 12: Additional comparisons using challenging stylized images as content input. As can be seen, other methods encounter difficulties in disentangling the style and content from these images, consequently struggling to effectively transfer the style from one stylized image to another. ©The paintings in the first three rows are by Judith Kondor Mochary

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/temp_figs/content_images/dog2.jpg)

Figure 13: Additional comparisons using challenging subject images as style reference. As can be seen, other methods encounter difficulties in disentangling the style and content from these images, consequently struggling to effectively transfer the style from one object to another.

##### Comparisons to Baselines Beyond SDXL-Based Approaches

We provide additional comparisons of our method with three other image stylization techniques that do not rely on SDXL: StyTr2 [^12], AdaAttn [^35], and SWAG [^55]. We evaluated the results using the same quantitative metrics described in the main paper. Figure 14 presents a qualitative comparison of the same set shown in the main paper, and Table 2 contains the quantitative results.

Table 2: Quantitative comparison: We measure the average cosine similarity between the DINO features of the output image and the reference style and content. In this experiment, we use a single input image for evaluation.

|  | StyTr2 | AdaAttn | SWAG | Ours |
| --- | --- | --- | --- | --- |
| StyleTransfer | $0.83$ | $0.818$ | $0.883$ | $0.881$ |
| Content | $0.854$ | $0.828$ | $0.788$ | $0.790$ |

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/temp_figs/content_images/cat.jpeg)

Figure 14: Comparison with alternative approaches that do not rely on SDXL. The input style and content references are shown on the left.

##### Comparisons to InstantStyle

InstantStyle [^54] is a concurrent work to ours. Aimed at performing image stylization tasks based on a style image reference. InstantStyle achieves this by injecting the CLIP embedding of the style image into style-specific blocks within SDXL, similar to our method, where the fifth block is selected for the style condition. Notably, InstantStyle uses a trained IP-Adapter model and does not require fine-tuning, which is its main advantage over our method. Both approaches provide compelling results in consistent style generation, as presented in Figure 15. For content conditioning, InstantStyle utilizes ControlNet, while our method separates content from style and extracts both. This allows for better content preservation in scenarios where ControlNet may not capture the content well enough or may override the style, as shown in Figure 16. Additionally, InstantStyle requires the content component to be explicitly defined to subtract its CLIP embedding from the style embedding, whereas our approach learns the content and style implicitly. For a fair comparison, we trained our method using the style images from InstantStyle.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/rebuttal_files/instantStyle_compare/references/bunny.jpg)

Refer to caption

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/temp_figs/content_images/cat.jpeg)

Figure 16: Comparison of style and content mixing between our method and InstantStyle. The results illustrate cases where ControlNet, used by InstantStyle, may fail to adequately capture the content or may override the style. For example, in the fourth row, we can see that ControlNet failed to extract the shape of the dog, leading to unsatisfactory results, While our method demonstrates better content preservation. The images showcase the stylization applied to various content images, highlighting differences in how each approach handles content and style integration.

### Appendix B Limitations

In Section 6 of the main paper, we discussed the limitations of our method. Here, we expand upon this section and propose potential approaches to mitigate these limitations. The first limitation we aim to address is the sub-optimal identity preservation due to color separation. To overcome this issue, we propose applying a scaling factor of alpha between 0.4-0.5 to the style adapter $\Delta W^{5}$. This adjustment allows for preserving the original colors of the subject while minimizing interference with other style B-LoRA injections, as illustrated in Figure 17.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/temp_figs/content_images/dog6.jpg)

Figure 17: To mitigate the limitation of sub-optimal identity preservation due to color separation, we propose combining adapters { Δ W 4, 5 } \\{\\Delta W^{4},\\Delta W^{5}\\}, with \\Delta W^{5} assigned a coefficient α \\alpha within the range of \[0.4, 0.5\]. This method preserves the original colors of the subject while allowing stylizations using text prompts. The generated contents depicted in the figure are based on the prompt “Watercolor painting of \[c\]”.

To mitigate style leakage from background objects in the style reference image, we suggest preprocessing the training data by center cropping the desired style reference image. This approach increases precision by focusing on the central object during the B-LoRA training process.

Addressing the final limitation of adequately capturing content in complex scenes, we conducted an ablation study to explore the effect of injecting different prompts into different blocks of the network. Specifically, we conducted five experiments:

(1) Injecting our method’s prompt “A \[c\] in \[s\] style”, into all transformer blocks of the UNet. (2) Injecting “A \[c\]” into the content block $W^{4}$ while injecting “A \[s\]” into all other blocks. (3) The complement of (2), injecting “A \[s\]” into the style block $W^{5}$ and “A \[c\]” into all other blocks. (4) Similar to (2), but injecting “A \[c\]” into $W^{4}$ while other blocks receive “A \[c\] in \[s\] style”. (5) Similar to (3), but injecting “A \[s\]” into $W^{5}$ while other blocks receive our method’s prompt “A \[c\] in \[s\] style”.

We present the results of these experiments in Figure 18. Our findings indicate that injecting the prompt “A \[c\]” into $W^{4}$ while other blocks receive the prompt “A \[c\] in \[s\] style” often leads to improved generation results, particularly for complex scenes containing numerous elements.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/temp_figs/scene_images/lighthouse.jpg)

Figure 18: Qualitative results of an ablation study investigating the effect of injecting different prompts into different blocks of the network to address the limitation of capturing content in complex scenes. Five experiments were conducted presented in the five columns \[1-5\]: (1) Injecting our method prompt, denoted as p 1 p\_{1} = “A \[c\] in \[s\] style”, into the entire Unet. (2) Injecting “A \[c\]” into the content block W 4 W^{4} while all other blocks receive “A \[s\]”. (3) The complement, injecting “A \[s\]” into the style block 5 W^{5} and “A \[c\]” into all other blocks. (4) Similar to 2, but injecting “A \[c\]” into while other blocks receive “A \[c\] in \[s\] style”. (5) Similar to 3, but injecting “A \[s\]” into while other blocks receive our method’s prompt “A \[c\] in \[s\] style”. As can be seen the (4) columns contains the best results.

### Appendix C Analysis and Ablation

##### Layers Optimization

As detailed in Section 4 of the main paper, the SDXL UNet comprises 11 transformer blocks, with the high-resolution blocks containing 2 attention layers each and the middle 6 blocks containing 10 attention layers each (see Figure 3 in the main paper). To explore the impact of different block combinations on the resulting image, we divided the UNet into 8 blocks $\{W^{0}_{0}\ldots W^{7}_{0}\}$, where $\{W^{1}_{0}\ldots W^{6}_{0}\}$ represent the bottleneck blocks, as discussed in Section 4, and designated $W^{0}_{0}$ and $W^{7}_{0}$ for the high-resolution blocks at the edges. We aimed to assess the effects of optimizing various block combinations $\{\Delta W^{i},\Delta W^{j}\}$ by jointly training the LoRA weights of the corresponding blocks. Qualitative results are depicted in Figures 19 and 20, where each cell (i, j) represents the reconstruction image for the prompt “A \[v\]” after training the LoRAs solely for the i-th and j-th blocks of the SDXL Unet. The diagonal entries represent output generated by training a single block. Upon examination, we observed that optimizing $\{\Delta W^{4},\Delta W^{5}\}$ consistently produced the most satisfactory results for content and style, respectively, outperforming other combinations. Notably, the reconstruction in cell (4, 5) yielded the best results achievable among all combinations, supporting our findings in the main paper. Furthermore, we noted that the combination of blocks 2 and 5 also achieved satisfactory reconstruction. However, employing this combination may lead to less disentanglement of style from content, as $\Delta W^{5}$ needs to “cover” $\Delta W^{2}$ by learning content details instead of focusing primarily on style, as intended. This observation further solidifies our choice of optimizing $\{\Delta W^{4},\Delta W^{5}\}$ for effective style-content separation.

##### Prompt Selection

To validate our choice of the prompt “A \[v\]” during optimization, we conducted an ablation study regarding the prompt used during training. As described in the DreamBooth [^45] paper, the authors suggest that the most efficient way to conduct the fine-tuning process is by using the prompt “A \[v\] \<class-name>”, where \[v\] is the token dedicated for personalization, and \<class-name> is the class of the object depicted in the input image. We compare our method of optimizing $\Delta W^{4}$ and $\Delta W^{5}$ with the prompt “A \[v\]” against using the suggested “A \[v\] \<class-name>” prompt.

In Figure 21, we demonstrate the impact of different prompts on style transfer between objects by fusing $\Delta W^{4}_{c1}$ and $\Delta W^{5}_{c2}$ to transfer the style of object1 to object2. We use four different prompts: (1) “A \[c1\] in \[c2\] style” (our method), (2) “A \[c1\] \<obj1> in \[c2\] style”, (3) “A \[c1\] \<obj1> in \[c2\] \<obj2> style”, and (4) our method optimized without the class name.

As can be seen, the first column, using “A \[c1\] in \[c2\] style”, fails to reconstruct the object’s structure correctly. The second column, with “A \[c1\] \<obj1> in \[c2\] style”, successfully reconstructs the content but struggles to transfer the style. In the third column, using “A \[c1\] \<obj1> in \[c2\] \<obj2> style”, the structure of the resulting image is affected by the obj2 class name.

In contrast, our method in the fourth column, optimized without the class name, is able to preserve the content image’s structure and effectively transfer the style from the other object. This demonstrates the effectiveness of our approach using the prompt “A \[v\]” during optimization.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/sup_figs/blocks_opt_ablation/cat/opt_0_0.jpg)

Figure 19: Qualitative results of the ablation study showcasing the reconstruction images for prompt “A \[v\]” after training LoRAs for different block combinations of the SDXL Unet. Each cell (i, j) represents a specific block combination, with the diagonal representing output generated by training a single block. Notably, cells (4, 5) demonstrate the most consistent and optimal reconstruction for content and style, respectively

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/sup_figs/blocks_opt_ablation/scary_mug/opt_0_0.jpg)

Figure 20: Qualitative results of the ablation study showcasing the reconstruction images for prompt “A \[v\]” after training LoRAs for different block combinations of the SDXL Unet. Each cell (i, j) represents a specific block combination, with the diagonal representing output generated by training a single block. Notably, cells (4, 5) demonstrate the most consistent and optimal reconstruction for content and style, respectively.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/temp_figs/content_images/scary_mug.jpg)

Figure 21: Ablation study on the impact of different prompts for style transfer between objects. The first three columns use the prompts: (1) “A \[c1\] in \[c2\] style”, (2) “A \[c1\] in \[c2\] style”, (3) “A \[c1\] in \[c2\] style”, respectively. The fourth column shows our method using the prompt “A \[v\]” without class names during optimization of Δ W 4 \\Delta W^{4} and 5 \\Delta W^{5}. Our approach in the fourth column better preserves the content object’s structure while effectively transferring the style from the other object.

##### Alpha Effect

As mentioned in the main paper, by the end of the training, we can obtain the tuned model weights using $W=W_{0}+\Delta W$, where $\Delta W$ is our trained B-LoRA update. The strength of the fine-tuning merge equation can be adjusted and controlled by the alpha scalar: $W=W_{0}+\alpha\Delta W$. (in our experiments $\alpha=1$). We demonstrate alpha’s effect on style and content components in Figure 22. As can be seen, when the alpha value is small, both the style and the content may be lost.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/sup_figs/alpha_ctrl.jpg)

Figure 22: On the left is the style-content input pair. On the right is quantitative control over style and content by altering the α \\alpha parameter, shown in white.

### Appendix D B-LoRA for Personalization

Throughout the paper, our method has been implemented using a single image for decoupling style and content. However, by training our method using multiple images for content, we can recontextualize reference objects while preserving stylization quality. In Figures 23 and 24, we showcase the versatility of our method by combining various B-LoRAs for style and content with text prompts. Note that the style can be derived from the reference style or from other objects.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/temp_figs/content_images/dog2.jpg)

Figure 23: While maintaining the stylistic characteristics of the style, our method effectively re-contextualizes the content object. Note that our approach is capable of transferring the style from either a style or object reference image.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/temp_figs/content_images/dog2.jpg)

Figure 24: While maintaining the stylistic characteristics of the style, our method effectively re-contextualizes the content object. Note that our approach is capable of transferring the style from either a style or object reference image.

### Appendix E Additional Results

Our B-LoRA method focuses on three main applications: image stylization based on image style references, text-based image stylization, and consistent style generation. In Figures 25 and 26, we present additional results generated by our approach for image stylization based on image style references. The columns represent the style reference images, while the rows correspond to the content reference images. As discussed, our method demonstrates proficiency in extracting content from style images (Figure 27) and extracting style from objects for object mixing tasks (Figure 28). In Figures 29 and 30, we provide qualitative results showcasing our method’s performance on randomly selected objects and styles from our evaluation set. These examples further highlight the robustness of our approach to handling diverse content and style references. In Figure 31 we present additional qualitative results for text-based image stylization. As discussed in the paper, by utilizing only the learned B-LoRA weights capturing the content, our method enables text-guided style manipulation while effectively preserving the input object’s content and structure. These results demonstrate the flexibility of our approach in allowing challenging style manipulations through textual guidance.

Figure 25: Image stylization based on image style reference using B-LoRA, illustrating the performance on challenging content image references. ©The paintings in the first three columns are by Judith Kondor Mochary.<sup>†</sup>

Figure 26: Image stylization based on image style reference using B-LoRA, illustrating the performance on challenging content image references.<sup>†</sup>

Figure 27: Additional results generated using B-LoRA. Our method able to blend content and styles across different style images. Each object in the (i, j) cell is created by combining the $\Delta W^{4}$ of the i-th row with the $\Delta W^{5}$ of the j-th column, while the diagonal represents the reconstruction image. ©The paintings in the second and third columns (and rows) are by Judith Kondor Mochary.<sup>†</sup>

Figure 28: Additional results generated using B-LoRA. Our method able to blend content and styles across different objects. Each object in the (i, j) cell is created by combining the $\Delta W^{4}$ of the i-th row with the $\Delta W^{5}$ of the j-th column, while the diagonal represents the reconstruction image.<sup>†</sup>

Figure 29: Image stylization based on image style reference using B-LoRA for randomly selected objects and styles. ©The paintings in the last two columns are by Judith Kondor Mochary.<sup>†</sup>

Figure 30: Image stylization based on image style reference using B-LoRA for randomly selected objects and styles.<sup>†</sup>

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.14572/assets/temp_figs/content_images/teddybear.jpg)

Figure 31: Text-based Image stylization using B-LoRA, generated using the prompt “A \[v\] made of …”.

[^1]: Aishwarya Agarwal, Srikrishna Karanam, Tripti Shukla, and Balaji Vasan Srinivasan. An image is worth multiple words: Multi-attribute inversion for constrained text-to-image synthesis. *ArXiv*, abs/2311.11919, 2023.

[^2]: Yuval Alaluf, Daniel Garibi, Or Patashnik, Hadar Averbuch-Elor, and Daniel Cohen-Or. Cross-image attention for zero-shot appearance transfer. *ArXiv*, abs/2311.03335, 2023a.

[^3]: Yuval Alaluf, Elad Richardson, Gal Metzer, and Daniel Cohen-Or. A neural space-time representation for text-to-image personalization. *ACM Transactions on Graphics (TOG)*, 42(6):1–10, 2023b.

[^4]: Moab Arar, Andrey Voynov, Amir Hertz, Omri Avrahami, Shlomi Fruchter, Yael Pritch, Daniel Cohen-Or, and Ariel Shamir. Palp: Prompt aligned personalization of text-to-image models. *ArXiv*, abs/2401.06105, 2024.

[^5]: Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. *ACM Transactions on Graphics (TOG)*, 42:1 – 11, 2022.

[^6]: Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel Cohen-Or, and Dani Lischinski. Break-a-scene: Extracting multiple concepts from a single image. In *SIGGRAPH Asia 2023 Conference Papers*, New York, NY, USA, 2023a. Association for Computing Machinery.

[^7]: Omri Avrahami, Amir Hertz, Yael Vinker, Moab Arar, Shlomi Fruchter, Ohad Fried, Daniel Cohen-Or, and Dani Lischinski. The chosen one: Consistent characters in text-to-image diffusion models. *arXiv preprint arXiv:2311.10093*, 2023b.

[^8]: Mathilde Caron, Hugo Touvron, Ishan Misra, Herv’e J’egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. *2021 IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 9630–9640, 2021.

[^9]: Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, José Lezama, Lu Jiang, Ming Yang, Kevin P. Murphy, William T. Freeman, Michael Rubinstein, Yuanzhen Li, and Dilip Krishnan. Muse: Text-to-image generation via masked generative transformers. *ArXiv*, abs/2301.00704, 2023.

[^10]: Minghao Chen, Iro Laina, and Andrea Vedaldi. Training-free layout control with cross-attention guidance. *ArXiv*, abs/2304.03373, 2023.

[^11]: Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. *ArXiv*, abs/2210.11427, 2022.

[^12]: Yingying Deng, Fan Tang, Weiming Dong, Chongyang Ma, Xingjia Pan, Lei Wang, and Changsheng Xu. Stytr2: Image style transfer with transformers. *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 11316–11326, 2021.

[^13]: Alexei A. Efros and William T. Freeman. Image quilting for texture synthesis and transfer. *Proceedings of the 28th annual conference on Computer graphics and interactive techniques*, 2001.

[^14]: Dave Epstein, A. Jabri, Ben Poole, Alexei A. Efros, and Aleksander Holynski. Diffusion self-guidance for controllable image generation. *ArXiv*, abs/2306.00986, 2023.

[^15]: Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. *arXiv preprint arXiv:2208.01618*, 2022.

[^16]: Rohit Gandikota, Joanna Materzynska, Tingrui Zhou, Antonio Torralba, and David Bau. Concept sliders: Lora adaptors for precise control in diffusion models. *ArXiv*, abs/2311.12092, 2023.

[^17]: Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. Image style transfer using convolutional neural networks. *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 2414–2423, 2016.

[^18]: Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial networks. *Communications of the ACM*, 63:139 – 144, 2014.

[^19]: Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Wei Wu, Yixiao Ge, Ying Shan, and Mike Zheng Shou. Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models. *ArXiv*, abs/2305.18292, 2023.

[^20]: Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris N. Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. *2023 IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 7289–7300, 2023.

[^21]: Amir Hertz, Ron Mokady, Jay M. Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. *ArXiv*, abs/2208.01626, 2022.

[^22]: Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention. *arXiv preprint arXiv:2312.02133*, 2023.

[^23]: Aaron Hertzmann, Charles E. Jacobs, Nuria Oliver, Brian Curless, and David H. Salesin. Image analogies. page 327–340, 2001.

[^24]: Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. *ArXiv*, abs/1902.00751, 2019.

[^25]: J. Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. *ArXiv*, abs/2106.09685, 2021.

[^26]: Xun Huang and Serge J. Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. *2017 IEEE International Conference on Computer Vision (ICCV)*, pages 1510–1519, 2017.

[^27]: huggingface. Controlnet with stable diffusion xl.

[^28]: Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, et al. Openclip, 2021.

[^29]: Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with conditional adversarial networks. *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 5967–5976, 2016.

[^30]: Yongcheng Jing, Yezhou Yang, Zunlei Feng, Jingwen Ye, and Mingli Song. Neural style transfer: A review. *IEEE Transactions on Visualization and Computer Graphics*, 26:3365–3385, 2017.

[^31]: Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. *ArXiv*, abs/1603.08155, 2016.

[^32]: Oren Katzir, Dani Lischinski, and Daniel Cohen-Or. Cross-domain cascaded deep translation. In *European Conference on Computer Vision*, 2020.

[^33]: Nupur Kumari, Bin Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 1931–1941, 2022.

[^34]: Senmao Li, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang, and Jian Yang. Stylediffusion: Prompt-embedding inversion for text-based editing. *ArXiv*, abs/2303.15649, 2023.

[^35]: Songhua Liu, Tianwei Lin, Dongliang He, Fu Li, Meiling Wang, Xin Li, Zhengxing Sun, Qian Li, and Errui Ding. Adaattn: Revisit attention mechanism in arbitrary neural style transfer. *2021 IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 6629–6638, 2021.

[^36]: mkshing. Ziplora-pytorch. https://github.com/mkshing/ziplora-pytorch.

[^37]: Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023*, pages 6038–6047. IEEE, 2023.

[^38]: Taesung Park, Alexei A. Efros, Richard Zhang, and Jun-Yan Zhu. Contrastive learning for unpaired image-to-image translation. In *European Conference on Computer Vision*, 2020.

[^39]: Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. *ACM SIGGRAPH 2023 Conference Proceedings*, 2023.

[^40]: Ryan Po, Guandao Yang, Kfir Aberman, and Gordon Wetzstein. Orthogonal adaptation for modular customization of diffusion models. *ArXiv*, abs/2312.02432, 2023.

[^41]: Dustin Podell, Zion English, Kyle Lacey, A. Blattmann, Tim Dockhorn, Jonas Muller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. *ArXiv*, abs/2307.01952, 2023.

[^42]: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In *International Conference on Machine Learning*, 2021.

[^43]: Fermat Research. Cog sdxl canny controlnet with lora support. https://replicate.com/batouresearch/sdxl-controlnet-lora.

[^44]: Robin Rombach, A. Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 10674–10685, 2021.

[^45]: Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 22500–22510, 2023.

[^46]: Simo Ryu. Low-rank adaptation for fast text-to-image diffusion fine-tuning. https://github.com/cloneofsimo/lora.

[^47]: Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora: Any subject in any style by effectively merging loras. *arXiv preprint arXiv:2311.13600*, 2023.

[^48]: Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, Yuan Hao, Irfan Essa, Michael Rubinstein, and Dilip Krishnan. Styledrop: Text-to-image generation in any style, 2023.

[^49]: Joshua Tenenbaum and William Freeman. Separating style and content. In *Advances in Neural Information Processing Systems*. MIT Press, 1996.

[^50]: Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 1921–1930, 2022.

[^51]: Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Neural Information Processing Systems*, 2017.

[^52]: Yael Vinker, Andrey Voynov, Daniel Cohen-Or, and Ariel Shamir. Concept decomposition for visual exploration and inspiration. *ACM Trans. Graph.*, 42(6), 2023.

[^53]: Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. $p+$: Extended textual conditioning in text-to-image generation. *arXiv preprint arXiv:2303.09522*, 2023.

[^54]: Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards style-preserving in text-to-image generation. *ArXiv*, abs/2404.02733, 2024.

[^55]: Pei Wang, Yijun Li, and Nuno Vasconcelos. Rethinking and improving the robustness of image style transfer. *2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 124–133, 2021.

[^56]: Yu xin Zhang, Nisha Huang, Fan Tang, Haibin Huang, Chongyang Ma, Weiming Dong, and Changsheng Xu. Inversion-based style transfer with diffusion models. *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 10146–10156, 2022.

[^57]: Serin Yang, Hyun joo Hwang, and Jong-Chul Ye. Zero-shot contrastive loss for text-guided diffusion image style transfer. *2023 IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 22816–22825, 2023.

[^58]: Hu Ye, Jun Zhang, Siyi Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. *ArXiv*, abs/2308.06721, 2023.

[^59]: Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. *2023 IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 3813–3824, 2023a.

[^60]: Yuxin Zhang, Weiming Dong, Fan Tang, Nisha Huang, Haibin Huang, Chongyang Ma, Tong-Yee Lee, Oliver Deussen, and Changsheng Xu. Prospect: Prompt spectrum for attribute-aware personalization of diffusion models. *ACM Transactions on Graphics (TOG)*, 42(6):1–14, 2023b.

[^61]: Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. *2017 IEEE International Conference on Computer Vision (ICCV)*, pages 2242–2251, 2017.