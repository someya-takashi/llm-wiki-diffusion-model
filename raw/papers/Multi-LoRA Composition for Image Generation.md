---
title: "Multi-LoRA Composition for Image Generation"
source: "https://ar5iv.labs.arxiv.org/html/2402.16843"
author:
published:
created: 2026-06-24
description: "Low-Rank Adaptation (LoRA) is extensively utilized in text-to-image models for the accurate rendition of specific elements like distinct characters or unique styles in generated images. Nonetheless, existing methods fa…"
tags:
  - "clippings"
---
Ming Zhong    Yelong Shen    Shuohang Wang    Yadong Lu    Yizhu Jiao    Siru Ouyang    Donghan Yu    Jiawei Han    Weizhu Chen

###### Abstract

Low-Rank Adaptation (LoRA) is extensively utilized in text-to-image models for the accurate rendition of specific elements like distinct characters or unique styles in generated images. Nonetheless, existing methods face challenges in effectively composing multiple LoRAs, especially as the number of LoRAs to be integrated grows, thus hindering the creation of complex imagery. In this paper, we study multi-LoRA composition through a decoding-centric perspective. We present two training-free methods: LoRA Switch, which alternates between different LoRAs at each denoising step, and LoRA Composite, which simultaneously incorporates all LoRAs to guide more cohesive image synthesis. To evaluate the proposed approaches, we establish ComposLoRA, a new comprehensive testbed as part of this research. It features a diverse range of LoRA categories with 480 composition sets. Utilizing an evaluation framework based on GPT-4V, our findings demonstrate a clear improvement in performance with our methods over the prevalent baseline, particularly evident when increasing the number of LoRAs in a composition.

Machine Learning, ICML

[Project Website: Multi-LoRA-Composition](https://maszhongming.github.io/Multi-LoRA-Composition)

## 1 Introduction

In the dynamic realm of generative text-to-image models [^10] [^30] [^33] [^29] [^31] [^36], the integration of Low-Rank Adaptation (LoRA) [^11] stands out for its ability to fine-tune image synthesis with remarkable precision and minimal computational load. LoRA excels by specializing in one element — such as a specific character, a particular clothing, a unique style, or other distinct visual aspects — and being trained to produce diverse and accurate renditions of this element in generated images. For instance, users could customize their LoRA model to generate various images of themselves, achieving an array of personalized and realistic representations. The application of LoRA not only showcases its adaptability and precision in image generation but also opens new avenues in customized digital content creation, revolutionizing how users interact with and utilize generative text-to-image models for creating tailored visual content.

However, an image typically embodies a mosaic of various elements, making compositionality key to controllable image generation [^38] [^13]. In pursuit of this, the strategy of composing multiple LoRAs, each focused on a distinct element, emerges as a feasible approach for advanced customization. This technique enables the digitization of complex scenes, such as virtual try-ons, merging users with clothing in a realistic fashion, or urban landscapes where users interact with meticulously designed city elements. Prior investigations into multi-LoRA compositions have explored the context of pre-trained language models [^41] [^12] or stable diffusion models [^32] [^34]. These studies aim to merge multiple LoRA models to synthesize a new LoRA model by training coefficient matrices [^12] [^34] or through the direct addition or subtraction of LoRA weights [^32] [^41]. Nevertheless, these approaches centered on weight manipulation could destabilize the merging process as the number of LoRAs grows [^12] and also overlook the interaction between LoRA models and base models. This oversight becomes particularly critical in diffusion models, which depend on sequential denoising steps for image generation. Ignoring the interplay between LoRAs and these steps can result in misalignments in the generative process, as shown in Figure 1, where a merged LoRA model fails to preserve the full complexity of all desired elements, leading to distorted or unrealistic images.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2402.16843/assets/x1.png)

Figure 1: Multi-LoRA composition techniques effectively blend different elements such as characters, clothing, and objects into a cohesive image. Unlike the conventional LoRA Merge approach, which can lead to detail loss and image distortion as more LoRAs are added, our methods retain the accuracy of each element and the overall image quality.

In this paper, we delve into multi-LoRA composition from a decoding-centric perspective, keeping all LoRA weights intact. We present two learning-free approaches that utilize either one or all LoRAs at each decoding step to facilitate compositional image synthesis. Our first approach, LoRA Switch, operates by selectively activating a single LoRA during each denoising step, with a rotation among multiple LoRAs throughout the generation process. For instance, in a virtual try-on scenario, LoRA Switch alternates between a character LoRA and a clothing LoRA at successive denoising steps, thereby ensuring that each element is rendered with precision and clarity. In parallel, we propose LoRA Composite, a technique that draws inspiration from classifier-free guidance [^9]. It involves calculating unconditional and conditional score estimates derived from each respective LoRA at every denoising step. These scores are then averaged to provide balanced guidance for image generation, ensuring a comprehensive incorporation of all elements. Furthermore, by bypassing the manipulation on the weight matrix but directly influencing the diffusion process, both methods allow for the integration of any number of LoRAs and overcome the limitations of recent studies that typically merge only two LoRAs [^34].

Experimentally, we introduce ComposLoRA, the first testbed specifically designed for LoRA-based composable image generation. This testbed features an extensive array of six LoRA categories, spanning two distinct visual styles: reality and anime. Our evaluation includes 480 diverse composition sets, each incorporating a varying number of LoRAs to comprehensively evaluate the efficacy of each proposed method. Given the lack of standardized automatic metrics for this novel task, we propose to employ GPT-4V [^27] [^28] as an evaluator, assessing both the quality of the images and the effectiveness of the compositions. Our empirical findings consistently demonstrate that both LoRA Switch and LoRA Composite substantially outperform the prevalent LoRA merging approach, particularly noticeable as the number of LoRAs in a composition increases. To further validate our results, we also conduct human evaluations, which reinforce our conclusions and affirm the efficacy of our automated evaluation framework. In addition, we provide a detailed analysis of the applicable scenarios for each method, as well as discuss the potential bias of using GPT-4V as an evaluator.

To summarize, our key contributions are threefold:

(1) We introduce the first investigation of multi-LoRA composition from a decoding-centric perspective, proposing LoRA Switch and LoRA Composite. Our methods overcome existing constraints on the number of LoRAs that can be integrated, offering enhanced flexibility and improved quality in composable image generation.

(2) Our work establishes ComposLoRA, a comprehensive testbed tailored to this research area, featuring six varied categories of LoRAs and 480 composition sets. Addressing the absence of standardized metrics, we present an evaluator built upon GPT-4V, setting a new benchmark for assessing both image quality and compositional efficacy.

(3) Through extensive automatic and human evaluations, our findings reveal the superior performance of the proposed LoRA Switch and LoRA Composite compared to the prevalent LoRA merging approach. Additionally, we provide an in-depth analysis of different multi-composition methods and evaluation frameworks.

## 2 Method

In this section, we begin with an overview of essential concepts for understanding multi-LoRA composition, followed by detailed descriptions of our proposed methods.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2402.16843/assets/x2.png)

Figure 2: Overview of three multi-LoRA composition techniques, where each colored LoRA represents a distinct element. The prevalent approach, LoRA Merge, linearly merges multiple LoRAs into a single one. In contrast, our methods concentrate on the denoising process: LoRA Switch cycles through different LoRAs during the denoising, while LoRA Composite involves all LoRAs working together as the guidance throughout the generation process.

### 2.1 Preliminary

##### Diffusion Models.

Diffusion models [^35] [^10] [^2] [^37] [^25] represent a class of generative models adept at crafting data samples from Gaussian noise through a sequential denoising process. They build upon a sequence of denoising autoencoders that estimate the score of a data distribution [^15]. Given an image $x$, the encoder $\mathcal{E}$ is used to map $x$ into a latent space, thus yielding an encoded latent $z=\mathcal{E}(x)$. The diffusion process introduces noise to $z$, resulting in latent representation $z_{t}$ with different noise levels over timestep $t\in\mathcal{T}$. The diffusion model $\epsilon_{\theta}$ with learnable parameters $\theta$ is trained to predict the noise added to the noisy latent $z_{t}$ given text instruction conditioning $c_{T}$. Typically, a mean-squared error loss function is utilized as the denoising objective:

$$
L=\mathbb{E}_{\mathcal{E}(x),\epsilon\sim\mathcal{N}(0,1),t}\left[||\epsilon-\epsilon_{\theta}(z_{t},t,c_{T})||^{2}_{2}\right],
$$

where $\epsilon$ is the additive Gaussian noise. In this paper, we investigate multi-LoRA composition based on diffusion models, which is consistent with the settings of previous studies on LoRA merging [^32] [^34].

##### Classifier-Free Guidance.

In diffusion-based generative modeling, classifier-free guidance [^9] balances the trade-off between the diversity and quality of the generated images, particularly in scenarios where the model is conditioned on classes or textual descriptions. For the text-to-image task, it operates by directing the probability mass towards outcomes where the implicit classifier $p_{\theta}(c|\mathbf{z}_{t})$ predicts a high likelihood for the textual conditioning $c$. This necessitates the diffusion models to undergo a joint training paradigm for both conditional and unconditional denoising. Subsequently, during inference, the guidance scale $s\geq 1$ is used to adjust the score function $\tilde{e}_{\theta}(\mathbf{z}_{t},c)$ by moving it closer to the conditional estimation $e_{\theta}(\mathbf{z}_{t},c)$ and further from the unconditional estimation $e_{\theta}(\mathbf{z}_{t})$, enhancing the conditioning effect on the generated images, as formalized in the following expression:

$$
\tilde{e}_{\theta}(\mathbf{z}_{t},c)=e_{\theta}(\mathbf{z}_{t})+s\cdot(e_{\theta}(\mathbf{z}_{t},c)-e_{\theta}(\mathbf{z}_{t})).
$$

##### LoRA Merge.

Low-Rank Adaptation (LoRA) approach [^11] enhances parameter efficiency by freezing the pre-trained weight matrices and integrating additional trainable low-rank matrices within the neural network. This method is founded on the observation that pre-trained models exhibit low “intrinsic dimension” [^1]. Concretely, for a weight matrix $W\in\mathbb{R}^{n\times m}$ in the diffusion model $\epsilon_{\theta}$, the introduction of a LoRA module involves updating $W$ to $W^{\prime}$, defined as $W^{\prime}=W+BA$. Here, $B\in\mathbb{R}^{n\times r}$ and $A\in\mathbb{R}^{r\times m}$ are matrices of a low-rank factor $r$, satisfying $r\ll\min(n,m)$. The concept of LoRA Merge [^32] is realized by linearly combining multiple LoRAs to synthesize a unified LoRA, subsequently plugged into the diffusion model. Formally, when introducing $k$ distinct LoRAs, the consequent updated matrix $W^{\prime}$ in $\epsilon_{\theta}$ is given by:

$$
W^{\prime}=W+\sum_{i=1}^{k}w_{i}\times B_{i}A_{i},
$$

where $i$ denotes the index of the $i$ -th LoRA, and $w_{i}$ is a scalar weight, typically a hyperparameter determined through empirical tuning. LoRA Merge has emerged as a dominant approach for presenting multiple elements cohesively in an image, offering a straightforward baseline for various applications. However, merging too many LoRAs at once can destabilize the merging process [^12], and it completely overlooks the interaction with the diffusion model during the generative process, resulting in the deformation of the hamburger and fingers in Figure 2.

### 2.2 Multi-LoRA Composition through a Decoding-Centric Perspective

To address the above issues, we base our approach on the denoising process of diffusion models and investigate how to perform composition while maintaining the LoRA weights unchanged. This is specifically divided into two perspectives: in each denoising step, either activate only one LoRA or engage all LoRAs to guide the generation.

##### LoRA Switch (LoRA-s).

To explore activating a single LoRA in each denoising step, we present LoRA Switch. This method introduces a dynamic adaptation mechanism within diffusion models by sequentially activating individual LoRAs at designated intervals throughout the generation process. As illustrated in Figure 2, each LoRA is represented by a unique color corresponding to a specific element, with only one LoRA engaged per denoising step.

With a set of $k$ LoRAs, the methodology initiates with a prearranged sequence of permutations; in the example of the Figure, the sequence progresses from yellow to green to blue LoRAs. Starting from the first LoRA, the model transitions to the subsequent LoRA every $\tau$ step. This rotation persists, allowing each LoRA to be applied in turn after $k\tau$ steps, thereby endowing each element to contribute repeatedly to the image generation. The active LoRA at each denoising timestep $t$, ranging from 1 to the total number of steps required, is determined by the following equations:

$$
\displaystyle i
$$
 
$$
\displaystyle=\left\lfloor((t-1)\bmod(k\tau))/\tau\right\rfloor+1,
$$
$$
\displaystyle W^{\prime}_{t}
$$
 
$$
\displaystyle=W+w_{i}\times B_{i}A_{i}.
$$

In this formula, $i$ indicates the index of the currently active LoRA, iterating from 1 to $k$. The floor function $\left\lfloor\cdot\right\rfloor$ guarantees the integer value of $i$ is appropriately computed for $t$. The resulting weight matrix $W^{\prime}_{t}$ is updated to reflect the contribution from the active LoRA. By selectively enabling one LoRA at a time, LoRA Switch ensures focused attention to the details pertinent to the current element, thus preserving the integrity and quality of the generated image throughout the process.

##### LoRA Composite (LoRA-c).

To explore incorporating all LoRAs at each timestep without merging weight matrices, we propose LoRA Composite (LoRA-c), an approach grounded in the classifier-free guidance paradigm. This method involves calculating both unconditional and conditional score estimates for each LoRA individually at every denoising step. By aggregating these scores, the technique ensures balanced guidance throughout the image generation process, facilitating the cohesive integration of all elements represented by different LoRAs.

Formally, with $k$ LoRAs in place, let $\theta^{\prime}_{i}$ denote the parameters of the diffusion model ${e}_{\theta}$ after incorporating the $i$ -th LoRA. The collective guidance $\tilde{e}(\mathbf{z}_{t},c)$ based on textual condition $c$ is derived by aggregating the scores from each LoRA, as depicted in the equation below:

$$
\tilde{e}(\mathbf{z}_{t},c)=\frac{1}{k}\sum_{i=1}^{k}w_{i}\times\left[e_{\theta^{\prime}_{i}}(\mathbf{z}_{t})+s\cdot(e_{\theta^{\prime}_{i}}(\mathbf{z}_{t},c)-e_{\theta^{\prime}_{i}}(\mathbf{z}_{t}))\right].
$$

Here, $w_{i}$ is a scalar weight allocated to each LoRA, intended to adjust the influence of the $i$ -th LoRA. For our experimental setup, we set $w_{i}$ to 1, giving each LoRA equal importance. LoRA-c assures that every LoRA contributes effectively at each stage of the denoising process, addressing the potential issues of robustness and detail preservation that are commonly associated with merging LoRAs.

Overall, we are the first to adopt a decoding-centric perspective in multi-LoRA composition, steering clear of the instability inherent in weight manipulation on LoRAs. Our study introduces two training-free methods for activating either one or all LoRAs at each denoising step, with their comparative analysis presented in Sections $\S$ 3.2 and $\S$ 3.3.1.

## 3 Experiments

### 3.1 Experimental Setup

##### ComposLoRA Testbed.

Due to the absence of standardized benchmarks and automated evaluation metrics, existing studies involving evaluation for composable image generation lean heavily on quantitative analysis [^13] [^39] and human effort [^34], which also limits the advancements of multi-LoRA composition. To bridge this gap, we introduce a comprehensive testbed ComposLoRA designed to facilitate comparative analysis of various composition approaches. This testbed builds upon a collection of public LoRAs <sup>1</sup>, which are extensively shared and recognized as essential plug-in modules in this field. The selection of LoRAs for this testbed adheres to the following criteria:

(1) Each LoRA should be robustly trained, ensuring it can accurately replicate the specific elements it represents when integrated independently;

(2) The elements represented by the LoRAs should cover a diverse range of categories and demonstrate adaptability across different image styles;

(3) When composed, LoRAs from different categories should be compatible, preventing any conflicts in the resulting image composition.

Consequently, we curate two unique subsets of LoRAs representing realistic and anime styles. Each subset comprises a variety of elements: 3 characters, 2 types of clothing, 2 styles, 2 backgrounds, and 2 objects, culminating in a total of 22 LoRAs in ComposLoRA. In constructing composition sets, we strictly follow a crucial principle: each set must include one character LoRA and avoid duplication of element categories to prevent conflicts. Thus, the ComposLoRA evaluation incorporates a total of 480 distinct composition sets. This includes 48 sets comprising 2 LoRAs, 144 sets with 3 LoRAs, 192 sets featuring 4 LoRAs, and 96 sets containing 5 LoRAs. Key features for each LoRA are manually annotated and serve dual purposes: they act as input prompts for the text-to-image models to generate images, and also provide reference points for subsequent evaluations using GPT-4V. Detailed descriptions of each LoRA can be found in Table 3 in the Appendix.

![[Uncaptioned image]](https://ar5iv.labs.arxiv.org/html/2402.16843/assets/Figure/merge_case.png)

Table 1: Comparative evaluation with GPT-4V. The evaluation prompt and result are in a simplified version.

##### Comparative Evaluation with GPT-4V.

While existing metrics can calculate the alignment between text and images [^8] [^17], they fall short in assessing the intricacies of specific elements within an image and the quality of their composition. Recently, multimodal large language models like GPT-4V [^27] [^28] have significant progress and promise in various multimodal tasks, underscoring their potential in evaluating image generation task [^19] [^42] s. In our study, we leverage GPT-4V’s capabilities to serve as an evaluator for composable image generation.

Specifically, we employ a comparative evaluation method, utilizing GPT-4V to rate generated images across two dimensions: composition quality and image quality. We utilize a 0 to 10 scoring scale, with higher scores indicating superior quality. As outlined in Table 1, we provide GPT-4V with a prompt that includes the essential features of the elements to be composed, the criteria for scoring in the two dimensions, and the format for the expected output. GPT-4V, in its evaluation, identifies issues such as deformities in the burger and the hand in Image 1, as well as incorrect hair color, leading to appropriate point deductions. The complete evaluation prompts and results are available in Tables 4 and 5 in Appendix. This experimental setup allows us to compare the efficacy of each of the two proposed methods against the LoRA Merge approach. Additionally, we examine how GPT-4V-based scoring aligns with human judgment in Section $\S$ 3.2 and explore the potential biases of using it as an evaluator in Section $\S$ 3.3.3.

##### Implementation Details.

For our experiments, we employ stable-diffusion-v1.5 [^30] as the backbone model. We utilize two specific checkpoints for our experiments: “Realistic\_Vision\_V5.1” <sup>2</sup> for realistic images and “Counterfeit-V2.5” <sup>3</sup> for anime images, each fine-tuned to their respective styles. In the realistic style subset, we configure the model with 100 denoising steps, a guidance scale $s$ of 7, and set the image size to 1024x768, optimizing for superior image quality. For the anime style subset, the settings differ slightly with 200 denoising steps, a guidance scale $s$ of 10, and an image size of 512x512. The DPM-Solver++ [^23] [^24] is used as the scheduler in the generation process. The weight scale $w$ is consistently set at 0.8 for composing LoRAs within ComposLoRA. For the LoRA Switch approach, we apply a cycle with $\tau$ set to 5, meaning every 5 denoising steps activate the next LoRA in the sequence: character, clothing, style, background, then object. To ensure the reliability of our experimental results, we conduct image generation using three random seeds. All reported results in this paper represent the average evaluation scores across these three runs.

### 3.2 Results on ComposLoRA

##### GPT-4V-based Evaluation.

We first present the comparative evaluation results obtained using GPT-4V. This evaluation involves scoring the performance of LoRA-s versus LoRA Merge, and LoRA-c versus LoRA Merge across two dimensions, as well as determining the winner based on these scores. Specific scores and win rates are illustrated in Figure LABEL:fig:result, leading to several key observations:

(1) Our proposed method consistently outperforms LoRA Merge across all configurations and in both dimensions, with the margin of superiority increasing as the number of LoRAs grows. For instance, as shown in Figure LABEL:fig:switch\_scoring, the score advantage of LoRA Switch escalates from 0.04 with 2 LoRAs to 1.32 with 5 LoRAs. This trend aligns with the win rate observed in Figure LABEL:fig:switch\_win\_rate, where the win rate approaches 70% when composing 5 LoRAs.

(2) LoRA-S shows superior performance in composition quality, whereas LoRA-C excels in image quality. In scenarios involving 5 LoRAs and using LoRA Merge as a baseline, the win rate of LoRA-s in composition quality surpasses that of LoRA-c by 14% (69% vs. 55%). Conversely, for image quality, LoRA-c’s win rate is 10% higher than that of LoRA-s (56% vs. 46%).

(3) The task of compositional image generation remains highly challenging, especially as the number of elements to be composed increases. According to GPT-4V’s scoring, the average score for composing 2 LoRAs is above 8.5, but it sharply declines to around 6 for compositions involving 5 LoRAs. Hence, despite the considerable improvements our methods offer, there is still substantial room for further research in the field of compositional image generation.

##### Human Evaluation.

To complement our results, we conduct a human evaluation to assess the effectiveness of different methods and validate GPT-4V’s efficacy as an evaluator. Two graduate students rate 120 images on compositional and image quality using a 1-5 Likert scale: 1 signifies complete failure, 2-4 represent significant, moderate, and minor issues, respectively, while 5 denotes perfect execution. To ensure consistency, the annotators initially pilot-score 20 images to standardize their understanding of the criteria. The results, summarized in the upper section of Table 2, align with GPT-4V’s findings, confirming our methods outperform LoRA Merge — with LoRA Switch excelling in composition and LoRA Composite in image quality.

Furthermore, we analyze the Pearson correlations between human evaluations and scores derived from GPT-4V and CLIPScore [^8], with results presented in the lower section of Table 2. This comparison reveals that CLIPScore’s evaluations fall short in assessing specific compositional and quality aspects due to its inability to discern the nuanced features of each element. On the other hand, the GPT-4V-based evaluator we adopt shows substantially higher correlations with human judgments, affirming the validity of our evaluation framework.

Table 2: Human evaluation results and Pearson correlation between different metrics and human judgment.

<table><tbody><tr><td colspan="3">Human Evaluation</td></tr><tr><td></td><td>Composition</td><td>Image Quality</td></tr><tr><td>LoRA Merge</td><td>3.14</td><td>2.94</td></tr><tr><td>LoRA Switch</td><td>3.91</td><td>4.15</td></tr><tr><td>LoRA Composite</td><td>3.78</td><td>4.35</td></tr><tr><td colspan="3">Correlations with Human Judgments</td></tr><tr><td></td><td>Composition</td><td>Image Quality</td></tr><tr><td>CLIPScore</td><td>-0.006</td><td>0.083</td></tr><tr><td>Ours</td><td>0.454</td><td>0.457</td></tr></tbody></table>

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2402.16843/assets/x3.png)

Figure 6: Analysis on image styles. In general, LoRA-s is more adept at realistic styles, while LoRA-c has better performance in anime styles.

### 3.3 Analysis

To enhance our understanding of the proposed methods, we further investigate the following critical questions:

#### 3.3.1 Do Specific Image Styles Favor Different Methods?

To explore this, we separately evaluate the performance of methods on realistic and anime-style subsets within ComposLoRA. The win rate results, presented in Figure 6, reveal distinct tendencies for each method. Our observations reveal that, while LoRA-s may not excel in image quality compared to LoRA-c, it demonstrates comparable performance in this dimension within the realistic style subset, while maintaining a significant edge in composition quality. In contrast, in the anime-style subset, LoRA-c, shows a performance on par with LoRA-s in composition quality, while notably surpassing it in image quality. These findings suggest that LoRA-S is more adept at composing elements in realistic-style images, whereas LoRA-C shows a stronger performance in anime-style imagery.

#### 3.3.2 How Does the Step Size and Order of LoRA Activation Affect LoRA Switch?

To identify the optimal configuration for LoRA Switch, we examine the influence of two crucial hyperparameters: the sequence in which LoRAs are activated and the interval between each activation. Our findings, depicted in Figure LABEL:fig:switch\_step, show that overly frequent switching, such as changing LoRAs at every denoising step, leads to distortions in generated images and suboptimal performance. The efficiency of the LoRA Switch improves progressively with increased step size, reaching peak performance at $\tau$ = 5. Consequently, we adopt this step size in our experiments.

Moreover, our analysis underscores that the initial choice of LoRA in the activation sequence clearly influences overall performance, while alterations in the subsequent order have minimal impact. Activating the character LoRA first leads to the best performance, as demonstrated in Figure LABEL:fig:switch\_order. In contrast, starting with clothing, background, or object LoRAs yields results comparable to a completely randomized sequence. Notably, beginning with the style LoRA leads to a noticeable performance drop, even falling slightly below a random order. This observation underlines the critical role of prioritizing core image elements in the initial stage of the generation process to enhance both the image and compositional quality for LoRA Switch.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2402.16843/assets/x4.png)

Figure 8: Positional bias analysis for comparative evaluation uses GPT-4V. In each subfigure, the left side of the orange line compares LoRA-s with Merge, and the right side contrasts LoRA-c. “Merge First” indicates that the image produced by LoRA Merge is the first image input to GPT-4V during the comparative evaluation.

#### 3.3.3 Does GPT-4V Exhibit Bias as an Evaluator?

While GPT-4V has demonstrated utility in evaluating various image generation tasks [^19] [^42], our analysis uncovers a notable positional bias in its comparative evaluations. We investigate this potential bias by swapping the positions of images generated by different methods before inputting them to GPT-4V, and the results are illustrated in Figure 8. In the comparison of LoRA-s versus LoRA Merge, when the image generated by Merge is presented first (“Merge First”), the win rate for LoRA-s in composition quality stands at 60%. However, this win rate declines to 51% when LoRA-s’s image is the first input (“LoRA-S First”). Similarly, LoRA-c’s win rate decreases from 52% to 42%, suggesting that GPT-4V tends to favor the first image input in terms of composition quality. Intriguingly, the opposite trend is observed in image quality, where the second image tends to receive a higher score. These results indicate a significant positional bias in GPT-4V’s evaluation, varying with the dimension and the position of the images. To mitigate this bias in our study, the comparative evaluation results reported in this paper are averaged across both input orders.

## 4 Related Work

### 4.1 Composable Text-to-Image Generation

Composable image generation, a key aspect of digital content customization, involves creating images that adhere to a set of pre-defined specifications [^22]. Existing research in this domain primarily focuses on the following approaches: enhancing compositionality with scene graphs or layouts [^16] [^40] [^7], modifying the generative process of diffusion models to align with the underlying specifications [^6] [^14] [^13], or composing a series of independent models that enforce desired constraints [^4] [^20] [^26] [^21] [^18] [^5].

However, these methods typically operate at the concept level, where generative models excel in creating images based on broader categories or general concepts. For example, a model might be prompted to generate an image of “a woman wearing a dress”, and can adeptly accommodate variations in the textual description, such as changing the color of the dress. Yet, they struggle to accurately render specific, user-defined elements, like lesser-known characters or unique dress styles. Another line of work that can compose user-defined objects into images includes [^14] [^31]. However, these methods require extensive fine-tuning and do not perform well on multiple objects. Therefore, we introduce learning-free instance-level composition approaches utilizing LoRA in this paper, enabling the precise assembly of user-specified elements in image generation.

### 4.2 LoRA-based Manipulations

Leveraging large language models (LLMs) or diffusion models as the base model, recent research aims to manipulate LoRA weights to achieve a range of objectives: element composition in image generation [^32] [^34], enhancing or diminishing certain capabilities in LLMs [^41] [^12], incorporating world knowledge [^3], and transferring parametric knowledge from larger teacher models to smaller student models [^43]. Regarding LoRA composition techniques, both LoRAHub [^12] and ZipLoRA [^34] employ few-shot demonstrations to learn coefficient matrices for merging LoRAs, enabling the fusion of multiple LoRAs into a singular new LoRA. On the other hand, LoRA Merge [^32] [^41] introduces addition and negation operators to merge LoRA weights through arithmetic operations.

Nevertheless, these weight-based methods often lead to instability in the merging process as the number of LoRAs increases [^12]. They also fail to account for the interactive dynamics when applying the LoRA model in conjunction with the base model. To address these issues, our study explores a new perspective: instead of altering the weights of LoRAs, we maintain all LoRA weights intact and focus on the interactions between LoRAs and the underlying generative process.

## 5 Conclusion

In this paper, we present the first exploration of multi-LoRA composition from a decoding-centric perspective by introducing LoRA-s and LoRA-c that transcend the limitations of current weight manipulation techniques. Through establishing a dedicated testbed ComposLoRA, we introduce scalable automated evaluation metrics utilizing GPT-4V. Our study not only highlights the superior quality achieved by our methods but also provides a new standard for evaluating LoRA-based composable image generation.

## Impact Statements

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

## References

## Appendix A Appendix

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2402.16843/assets/x5.png)

Figure 9: Case study on composing 2 LoRAs in the realistic style.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2402.16843/assets/x6.png)

Figure 10: Case study on composing 2 LoRAs in the anime style.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2402.16843/assets/x7.png)

Figure 11: Case study on composing 3 LoRAs in the realistic style.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2402.16843/assets/x8.png)

Figure 12: Case study on composing 3 LoRAs in the anime style.

Table 3: Detailed descriptions of each LoRA in the ComposLoRA.

<table><tbody><tr><td>LoRA</td><td>Category</td><td>Trigger Words</td><td>Source</td></tr><tr><td colspan="4">Anime Style Subset</td></tr><tr><td>Kamado Nezuko</td><td>Character</td><td>kamado nezuko, black hair, pink eyes, forehead</td><td><a href="https://civitai.com/models/20346/nezuko-demon-slayer-kimetsu-no-yaiba-lora?modelVersionId=24191">Link</a></td></tr><tr><td>Texas the Omertosa in Arknights</td><td>Character</td><td>omertosa, 1girl, wolf ears, long hair</td><td><a href="https://civitai.com/models/6779/arknights-texas-the-omertosa?modelVersionId=7974">Link</a></td></tr><tr><td>Son Goku</td><td>Character</td><td>son goku, spiked hair, muscular male, wristband</td><td><a href="https://civitai.com/models/18279/son-goku-dragon-ball-all-series-lora?modelVersionId=21654">Link</a></td></tr><tr><td>Garreg Mach Monastery Uniform</td><td>Clothing</td><td>gmuniform, blue thighhighs, long sleeves</td><td><a href="https://civitai.com/models/151330/garreg-mach-monastery-uniform-fire-emblem-three-houses-lora-or-2-variants?modelVersionId=169217">Link</a></td></tr><tr><td>Zero Suit (Metroid)</td><td>Clothing</td><td>zero suit, blue gloves, high heels</td><td><a href="https://civitai.com/models/82304/zero-suit-metroid-outfit-lora?modelVersionId=87396">Link</a></td></tr><tr><td>Hand-drawn Style</td><td>Style</td><td>lineart, hand-drawn style</td><td><a href="https://civitai.com/models/143532/handdrawn">Link</a></td></tr><tr><td>Chinese Ink Wash Style</td><td>Style</td><td>shuimobysim, traditional chinese ink painting</td><td><a href="https://civitai.com/models/12597?modelVersionId=14856">Link</a></td></tr><tr><td>Bamboolight Background</td><td>Background</td><td>bamboolight, outdoors, bamboo</td><td><a href="https://civitai.com/models/113381/lora-bamboolight-concept-with-dropout-and-noise-version?modelVersionId=122477">Link</a></td></tr><tr><td>Auroral Background</td><td>Background</td><td>auroral, starry sky, outdoors</td><td><a href="https://civitai.com/models/85026?modelVersionId=156828">Link</a></td></tr><tr><td>Huge Two-Handed Burger</td><td>Object</td><td>two-handed burger, holding a huge burger with both hands</td><td><a href="https://civitai.com/models/27858/huge-two-handed-burger-lora?modelVersionId=34071">Link</a></td></tr><tr><td>Toast</td><td>Object</td><td>toast, toast in mouth</td><td><a href="https://civitai.com/models/26945/toast-in-mouth-lora?modelVersionId=32252">Link</a></td></tr><tr><td colspan="4">Realistic Style Subset</td></tr><tr><td>IU (Lee Ji Eun, Korean singer)</td><td>Character</td><td>iu1, long straight black hair, hazel eyes, diamond stud earrings</td><td><a href="https://civitai.com/models/11722/iu?modelVersionId=18576">Link</a></td></tr><tr><td>Scarlett Johansson</td><td>Character</td><td>scarlett, short red hair, blue eyes</td><td><a href="https://civitai.com/models/7468/scarlett-johanssonlora">Link</a></td></tr><tr><td>The Rock (Dwayne Johnson)</td><td>Character</td><td>th3r0ck with no hair, muscular male, serious look on his face</td><td><a href="https://civitai.com/models/22345/dwayne-the-rock-johnsonlora?modelVersionId=26680">Link</a></td></tr><tr><td>Thai University Uniform</td><td>Clothing</td><td>mahalaiuniform, white shirt short sleeves, black pencil skirt</td><td><a href="https://civitai.com/models/12657/thai-university-uniform?modelVersionId=58690">Link</a></td></tr><tr><td>School Dress</td><td>Clothing</td><td>school uniform, white shirt, red tie, blue pleated microskirt</td><td><a href="https://civitai.com/models/201305?modelVersionId=291486">Link</a></td></tr><tr><td>Japanese Film Color Style</td><td>Style</td><td>film overlay, film grain</td><td><a href="https://civitai.com/models/90393/japan-vibes-film-color?modelVersionId=107032">Link</a></td></tr><tr><td>Bright Style</td><td>Style</td><td>bright lighting</td><td><a href="https://civitai.com/models/70034/brightness-tweaker-lora-lora">Link</a></td></tr><tr><td>Library Bookshelf Background</td><td>Background</td><td>lib_bg, library bookshelf</td><td><a href="https://civitai.com/models/113488/library-bookshelf?modelVersionId=125699">Link</a></td></tr><tr><td>Forest Background</td><td>Background</td><td>slg, river, forest</td><td><a href="https://civitai.com/models/104292/the-forest-light?modelVersionId=127699">Link</a></td></tr><tr><td>Umbrella</td><td>Object</td><td>transparent umbrella</td><td><a href="https://civitai.com/models/54218/umbrellalora?modelVersionId=58578">Link</a></td></tr><tr><td>Bubble Gum</td><td>Object</td><td>blow bubble gum</td><td><a href="https://civitai.com/models/97550/bubble-gum-kaugummi-v20?modelVersionId=117038">Link</a></td></tr></tbody></table>

![[Uncaptioned image]](https://ar5iv.labs.arxiv.org/html/2402.16843/assets/Figure/merge_case.png)

Table 4: The full version of evaluation prompts for comparative evaluation with GPT-4V.

![[Uncaptioned image]](https://ar5iv.labs.arxiv.org/html/2402.16843/assets/Figure/merge_case.png)

Table 5: The full version of evaluation results from GPT-4V for comparative evaluation.

[^1]: Aghajanyan, A., Gupta, S., and Zettlemoyer, L. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. In Zong, C., Xia, F., Li, W., and Navigli, R. (eds.), *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021*, pp. 7319–7328. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.ACL-LONG.568. URL [https://doi.org/10.18653/v1/2021.acl-long.568](https://doi.org/10.18653/v1/2021.acl-long.568).

[^2]: Dhariwal, P. and Nichol, A. Q. Diffusion models beat gans on image synthesis. In Ranzato, M., Beygelzimer, A., Dauphin, Y. N., Liang, P., and Vaughan, J. W. (eds.), *Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual*, pp. 8780–8794, 2021. URL [https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html).

[^3]: Dou, S., Zhou, E., Liu, Y., Gao, S., Zhao, J., Shen, W., Zhou, Y., Xi, Z., Wang, X., Fan, X., Pu, S., Zhu, J., Zheng, R., Gui, T., Zhang, Q., and Huang, X. Loramoe: Revolutionizing mixture of experts for maintaining world knowledge in language model alignment. *CoRR*, abs/2312.09979, 2023. doi: 10.48550/ARXIV.2312.09979. URL [https://doi.org/10.48550/arXiv.2312.09979](https://doi.org/10.48550/arXiv.2312.09979).

[^4]: Du, Y., Li, S., and Mordatch, I. Compositional visual generation with energy based models. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020. URL [https://proceedings.neurips.cc/paper/2020/hash/49856ed476ad01fcff881d57e161d73f-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/49856ed476ad01fcff881d57e161d73f-Abstract.html).

[^5]: Du, Y., Durkan, C., Strudel, R., Tenenbaum, J. B., Dieleman, S., Fergus, R., Sohl-Dickstein, J., Doucet, A., and Grathwohl, W. S. Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and MCMC. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), *International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA*, volume 202 of *Proceedings of Machine Learning Research*, pp. 8489–8510. PMLR, 2023. URL [https://proceedings.mlr.press/v202/du23a.html](https://proceedings.mlr.press/v202/du23a.html).

[^6]: Feng, W., He, X., Fu, T., Jampani, V., Akula, A. R., Narayana, P., Basu, S., Wang, X. E., and Wang, W. Y. Training-free structured diffusion guidance for compositional text-to-image synthesis. In *The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net, 2023. URL [https://openreview.net/pdf?id=PUIqjT4rzq7](https://openreview.net/pdf?id=PUIqjT4rzq7).

[^7]: Gafni, O., Polyak, A., Ashual, O., Sheynin, S., Parikh, D., and Taigman, Y. Make-a-scene: Scene-based text-to-image generation with human priors. In Avidan, S., Brostow, G. J., Cissé, M., Farinella, G. M., and Hassner, T. (eds.), *Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XV*, volume 13675 of *Lecture Notes in Computer Science*, pp. 89–106. Springer, 2022. doi: 10.1007/978-3-031-19784-0“˙6. URL [https://doi.org/10.1007/978-3-031-19784-0\_6](https://doi.org/10.1007/978-3-031-19784-0_6).

[^8]: Hessel, J., Holtzman, A., Forbes, M., Bras, R. L., and Choi, Y. Clipscore: A reference-free evaluation metric for image captioning. In Moens, M., Huang, X., Specia, L., and Yih, S. W. (eds.), *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021*, pp. 7514–7528. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.EMNLP-MAIN.595. URL [https://doi.org/10.18653/v1/2021.emnlp-main.595](https://doi.org/10.18653/v1/2021.emnlp-main.595).

[^9]: Ho, J. and Salimans, T. Classifier-free diffusion guidance. *CoRR*, abs/2207.12598, 2022. doi: 10.48550/ARXIV.2207.12598. URL [https://doi.org/10.48550/arXiv.2207.12598](https://doi.org/10.48550/arXiv.2207.12598).

[^10]: Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020. URL [https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html).

[^11]: Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. Lora: Low-rank adaptation of large language models. In *The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022*. OpenReview.net, 2022. URL [https://openreview.net/forum?id=nZeVKeeFYf9](https://openreview.net/forum?id=nZeVKeeFYf9).

[^12]: Huang, C., Liu, Q., Lin, B. Y., Pang, T., Du, C., and Lin, M. Lorahub: Efficient cross-task generalization via dynamic lora composition. *CoRR*, abs/2307.13269, 2023a. doi: 10.48550/ARXIV.2307.13269. URL [https://doi.org/10.48550/arXiv.2307.13269](https://doi.org/10.48550/arXiv.2307.13269).

[^13]: Huang, L., Chen, D., Liu, Y., Shen, Y., Zhao, D., and Zhou, J. Composer: Creative and controllable image synthesis with composable conditions. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), *International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA*, volume 202 of *Proceedings of Machine Learning Research*, pp. 13753–13773. PMLR, 2023b. URL [https://proceedings.mlr.press/v202/huang23b.html](https://proceedings.mlr.press/v202/huang23b.html).

[^14]: Huang, Z., Chan, K. C. K., Jiang, Y., and Liu, Z. Collaborative diffusion for multi-modal face generation and editing. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023*, pp. 6080–6090. IEEE, 2023c. doi: 10.1109/CVPR52729.2023.00589. URL [https://doi.org/10.1109/CVPR52729.2023.00589](https://doi.org/10.1109/CVPR52729.2023.00589).

[^15]: Hyvärinen, A. Estimation of non-normalized statistical models by score matching. *J. Mach. Learn. Res.*, 6:695–709, 2005. URL [http://jmlr.org/papers/v6/hyvarinen05a.html](http://jmlr.org/papers/v6/hyvarinen05a.html).

[^16]: Johnson, J., Gupta, A., and Fei-Fei, L. Image generation from scene graphs. In *2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018*, pp. 1219–1228. Computer Vision Foundation / IEEE Computer Society, 2018. doi: 10.1109/CVPR.2018.00133. URL [http://openaccess.thecvf.com/content\_cvpr\_2018/html/Johnson\_Image\_Generation\_From\_CVPR\_2018\_paper.html](http://openaccess.thecvf.com/content_cvpr_2018/html/Johnson_Image_Generation_From_CVPR_2018_paper.html).

[^17]: Ku, M., Li, T., Zhang, K., Lu, Y., Fu, X., Zhuang, W., and Chen, W. Imagenhub: Standardizing the evaluation of conditional image generation models. *CoRR*, abs/2310.01596, 2023. doi: 10.48550/ARXIV.2310.01596. URL [https://doi.org/10.48550/arXiv.2310.01596](https://doi.org/10.48550/arXiv.2310.01596).

[^18]: Li, S., Du, Y., Tenenbaum, J. B., Torralba, A., and Mordatch, I. Composing ensembles of pre-trained models via iterative consensus. In *The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net, 2023. URL [https://openreview.net/pdf?id=gmwDKo-4cY](https://openreview.net/pdf?id=gmwDKo-4cY).

[^19]: Lin, K., Yang, Z., Li, L., Wang, J., and Wang, L. Designbench: Exploring and benchmarking DALL-E 3 for imagining visual design. *CoRR*, abs/2310.15144, 2023. doi: 10.48550/ARXIV.2310.15144. URL [https://doi.org/10.48550/arXiv.2310.15144](https://doi.org/10.48550/arXiv.2310.15144).

[^20]: Liu, N., Li, S., Du, Y., Tenenbaum, J., and Torralba, A. Learning to compose visual relations. In Ranzato, M., Beygelzimer, A., Dauphin, Y. N., Liang, P., and Vaughan, J. W. (eds.), *Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual*, pp. 23166–23178, 2021. URL [https://proceedings.neurips.cc/paper/2021/hash/c3008b2c6f5370b744850a98a95b73ad-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c3008b2c6f5370b744850a98a95b73ad-Abstract.html).

[^21]: Liu, N., Li, S., Du, Y., Torralba, A., and Tenenbaum, J. B. Compositional visual generation with composable diffusion models. In Avidan, S., Brostow, G. J., Cissé, M., Farinella, G. M., and Hassner, T. (eds.), *Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XVII*, volume 13677 of *Lecture Notes in Computer Science*, pp. 423–439. Springer, 2022. doi: 10.1007/978-3-031-19790-1“˙26. URL [https://doi.org/10.1007/978-3-031-19790-1\_26](https://doi.org/10.1007/978-3-031-19790-1_26).

[^22]: Liu, N., Du, Y., Li, S., Tenenbaum, J. B., and Torralba, A. Unsupervised compositional concepts discovery with text-to-image generative models. *CoRR*, abs/2306.05357, 2023. doi: 10.48550/ARXIV.2306.05357. URL [https://doi.org/10.48550/arXiv.2306.05357](https://doi.org/10.48550/arXiv.2306.05357).

[^23]: Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. Dpm-solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), *Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022*, 2022a. URL [http://papers.nips.cc/paper\_files/paper/2022/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html).

[^24]: Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. *CoRR*, abs/2211.01095, 2022b. doi: 10.48550/ARXIV.2211.01095. URL [https://doi.org/10.48550/arXiv.2211.01095](https://doi.org/10.48550/arXiv.2211.01095).

[^25]: Nichol, A. Q., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., and Chen, M. GLIDE: towards photorealistic image generation and editing with text-guided diffusion models. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvári, C., Niu, G., and Sabato, S. (eds.), *International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA*, volume 162 of *Proceedings of Machine Learning Research*, pp. 16784–16804. PMLR, 2022. URL [https://proceedings.mlr.press/v162/nichol22a.html](https://proceedings.mlr.press/v162/nichol22a.html).

[^26]: Nie, W., Vahdat, A., and Anandkumar, A. Controllable and compositional generation with latent-space energy-based models. In Ranzato, M., Beygelzimer, A., Dauphin, Y. N., Liang, P., and Vaughan, J. W. (eds.), *Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual*, pp. 13497–13510, 2021. URL [https://proceedings.neurips.cc/paper/2021/hash/701d804549a4a23d3cae801dac6c2c75-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/701d804549a4a23d3cae801dac6c2c75-Abstract.html).

[^27]: OpenAI. GPT-4: Contributions and System Card. https://cdn.openai.com/contributions/gpt-4v.pdf, 2023a.

[^28]: OpenAI. GPT-4v System Card. https://openai.com/research/gpt-4v-system-card, 2023b.

[^29]: Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen, M. Hierarchical text-conditional image generation with CLIP latents. *CoRR*, abs/2204.06125, 2022. doi: 10.48550/ARXIV.2204.06125. URL [https://doi.org/10.48550/arXiv.2204.06125](https://doi.org/10.48550/arXiv.2204.06125).

[^30]: Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022*, pp. 10674–10685. IEEE, 2022. doi: 10.1109/CVPR52688.2022.01042. URL [https://doi.org/10.1109/CVPR52688.2022.01042](https://doi.org/10.1109/CVPR52688.2022.01042).

[^31]: Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., and Aberman, K. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023*, pp. 22500–22510. IEEE, 2023. doi: 10.1109/CVPR52729.2023.02155. URL [https://doi.org/10.1109/CVPR52729.2023.02155](https://doi.org/10.1109/CVPR52729.2023.02155).

[^32]: Ryu, S. Merging loras. [https://github.com/cloneofsimo/lora](https://github.com/cloneofsimo/lora), 2023.

[^33]: Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, S. K. S., Lopes, R. G., Ayan, B. K., Salimans, T., Ho, J., Fleet, D. J., and Norouzi, M. Photorealistic text-to-image diffusion models with deep language understanding. In *NeurIPS*, 2022. URL [http://papers.nips.cc/paper\_files/paper/2022/hash/ec795aeadae0b7d230fa35cbaf04c041-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ec795aeadae0b7d230fa35cbaf04c041-Abstract-Conference.html).

[^34]: Shah, V., Ruiz, N., Cole, F., Lu, E., Lazebnik, S., Li, Y., and Jampani, V. Ziplora: Any subject in any style by effectively merging loras. *CoRR*, abs/2311.13600, 2023. doi: 10.48550/ARXIV.2311.13600. URL [https://doi.org/10.48550/arXiv.2311.13600](https://doi.org/10.48550/arXiv.2311.13600).

[^35]: Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In Bach, F. R. and Blei, D. M. (eds.), *Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015*, volume 37 of *JMLR Workshop and Conference Proceedings*, pp. 2256–2265. JMLR.org, 2015. URL [http://proceedings.mlr.press/v37/sohl-dickstein15.html](http://proceedings.mlr.press/v37/sohl-dickstein15.html).

[^36]: Sohn, K., Ruiz, N., Lee, K., Chin, D. C., Blok, I., Chang, H., Barber, J., Jiang, L., Entis, G., Li, Y., Hao, Y., Essa, I., Rubinstein, M., and Krishnan, D. Styledrop: Text-to-image generation in any style. *CoRR*, abs/2306.00983, 2023. doi: 10.48550/ARXIV.2306.00983. URL [https://doi.org/10.48550/arXiv.2306.00983](https://doi.org/10.48550/arXiv.2306.00983).

[^37]: Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In *9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021*. OpenReview.net, 2021. URL [https://openreview.net/forum?id=PxTIG12RRHS](https://openreview.net/forum?id=PxTIG12RRHS).

[^38]: Tenenbaum, J. Building machines that learn and think like people. In André, E., Koenig, S., Dastani, M., and Sukthankar, G. (eds.), *Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems, AAMAS 2018, Stockholm, Sweden, July 10-15, 2018*, pp. 5. International Foundation for Autonomous Agents and Multiagent Systems Richland, SC, USA / ACM, 2018. URL [http://dl.acm.org/citation.cfm?id=3237389](http://dl.acm.org/citation.cfm?id=3237389).

[^39]: Wang, Z., Jiang, Y., Lu, Y., Shen, Y., He, P., Chen, W., Wang, Z., and Zhou, M. In-context learning unlocked for diffusion models. *CoRR*, abs/2305.01115, 2023. doi: 10.48550/ARXIV.2305.01115. URL [https://doi.org/10.48550/arXiv.2305.01115](https://doi.org/10.48550/arXiv.2305.01115).

[^40]: Yang, Z., Liu, D., Wang, C., Yang, J., and Tao, D. Modeling image composition for complex scene generation. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022*, pp. 7754–7763. IEEE, 2022. doi: 10.1109/CVPR52688.2022.00761. URL [https://doi.org/10.1109/CVPR52688.2022.00761](https://doi.org/10.1109/CVPR52688.2022.00761).

[^41]: Zhang, J., Chen, S., Liu, J., and He, J. Composing parameter-efficient modules with arithmetic operations. *CoRR*, abs/2306.14870, 2023a. doi: 10.48550/ARXIV.2306.14870. URL [https://doi.org/10.48550/arXiv.2306.14870](https://doi.org/10.48550/arXiv.2306.14870).

[^42]: Zhang, X., Lu, Y., Wang, W., Yan, A., Yan, J., Qin, L., Wang, H., Yan, X., Wang, W. Y., and Petzold, L. R. Gpt-4v(ision) as a generalist evaluator for vision-language tasks. *CoRR*, abs/2311.01361, 2023b. doi: 10.48550/ARXIV.2311.01361. URL [https://doi.org/10.48550/arXiv.2311.01361](https://doi.org/10.48550/arXiv.2311.01361).

[^43]: Zhong, M., An, C., Chen, W., Han, J., and He, P. Seeking neural nuggets: Knowledge transfer in large language models from a parametric perspective. *CoRR*, abs/2310.11451, 2023. doi: 10.48550/ARXIV.2310.11451. URL [https://doi.org/10.48550/arXiv.2310.11451](https://doi.org/10.48550/arXiv.2310.11451).