---
title: "NP-LoRA: Null Space Projection Unifies Subject and Style in LoRA Fusion"
source: "https://ar5iv.labs.arxiv.org/html/2511.11051"
author:
published:
created: 2026-08-19
description: "Low-Rank Adaptation (LoRA) fusion has emerged as a key technique for reusing and composing learned subject and style representations for controllable generation without costly retraining.However, existing methods rely…"
tags:
  - "clippings"
---
Chuheng Chen    Xiaofei Zhou    Geyuan Zhang    and Yong Huang Thanks: All authors are with the Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China.

###### Abstract

Low-Rank Adaptation (LoRA) fusion has emerged as a key technique for reusing and composing learned subject and style representations for controllable generation without costly retraining. However, existing methods rely on weight-based merging, where one LoRA often dominates the other, leading to interference and degraded fidelity. This interference is structural: separately trained LoRAs occupy low-rank high-dimensional subspaces, leading to non-orthogonal and overlapping representations. In this work, we analyze the internal structure of LoRAs and find their generative behavior is dominated by a few principal directions in the low-rank subspace, which should remain free from interference during fusion. To achieve this, we propose Null Space Projection LoRA (NP-LoRA), a projection-based framework for LoRA fusion that enforces subspace separation to prevent structural interference among principal directions. Specifically, we first extract principal style directions via singular value decomposition (SVD) and then project the subject LoRA into its orthogonal null space. Furthermore, we introduce a soft projection mechanism that enables smooth control over the trade-off between subject fidelity and style consistency. Experiments show NP-LoRA consistently improves fusion quality over strong baselines (e.g., DINO and CLIP-based metrics, with human and LLM preference scores), and applies broadly across backbones and LoRA pairs without retraining.

## I introduction

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.11051/assets/introduction_5.png)

Fig. 1: Illustration of our motivation. (a) This task is to combine a content LoRA capturing subject identity with a style LoRA encoding artistic appearance, enabling the reuse and composition of learned generative knowledge. (b) Weighted addition of two LoRAs destroys style features in the merged results. Independently trained LoRAs, sharing the pretrained diffusion feature space, tend to occupy correlated low-rank high-dimensional subspaces (e.g., texture or color), which causes representational interference that degrades the fidelity of generated content. (c) The content LoRA is projected onto the null space of style lora, effectively avoiding overlap and preserving stylistic features.

Low-Rank Adaptation (LoRA) [^12] has emerged as an efficient customization method for diffusion models [^11] [^33] [^5] [^25]. By applying lightweight factorization to attention weights, LoRA learns subject or style-specific representations from a few examples [^27]. For flexible reuse of pretrained knowledge in LoRA, a key challenge arises in personalized concept composition: how to combine independently trained content and style LoRAs, which respectively capture subject identity and artistic appearance [^29] [^20] [^30]. Such scenarios commonly arise when users want to render a personalized subject in a desired artistic style, each learned from separate examples. For instance, one may aim to render a learned subject (e.g., a pet dog) in a separately trained artistic style (e.g., Van Gogh’s brushwork), as shown in Fig. 1(a). The goal is to fuse the content LoRA ($\Delta W_{c}$) and the style LoRA ($\Delta W_{s}$) into a single coherent module that retains both representations without destructive interference.

Existing merging techniques primarily rely on weight-based operations, which form a weighted combination of LoRAs in parameter space. Recent state-of-the-art approaches, such as ZipLoRA [^29], K-LoRA [^20], and LoRA.rar [^30], move beyond naive linear weight yet still adhere to the paradigm of weight-based fusion. Specifically, ZipLoRA [^29] learns an optimal fusion ratio through loss optimization, K-LoRA [^20] performs piecewise binary weight between the content and style LoRAs across diffusion timesteps, and LoRA.rar [^30] predicts fusion weights via a pretrained hypernetwork. However, such weight-based merging methods inherently cause interference between the two LoRAs. As shown in Fig. 1(b), weighted addition of $\Delta W_{c}$ and $\Delta W_{s}$ distorts the style LoRA and degrades stylistic fidelity in generated images. This interference arises because independently trained LoRAs are optimized within the same feature space of the pretrained diffusion model, which leads them to occupy correlated low-rank subspaces and adapt similar feature dimensions (e.g., texture or color patterns). As a result, their low-rank subspaces inevitably overlap and become non-orthogonal. Moreover, empirical analysis (Sec. IV-A) reveals that LoRAs exhibit a structured low-rank space, where a few principal directions dominate generative behavior. These directions are highly sensitive to perturbations, indicating the need to explicitly preserve them during fusion. To address this issue, we reformulate the merging objective: rather than simply performing a weighted merge of $\Delta W_{c}$ and $\Delta W_{s}$, the goal is to geometrically transform the content LoRA by projecting it into the null space of the style subspace.

To achieve this goal, we propose Null Space Projection LoRA (NP-LoRA), a projection-based framework that prevents interference between content and style LoRAs. Specifically, we perform singular value decomposition (SVD) on the style LoRA $\Delta W_{s}$ to extract its principal directions and define their orthogonal complement as a null space for interference-free fusion, as illustrated in Fig. 1(c). Based on this construction, we first formulate a strict projection scheme that projects the content LoRA $\Delta W_{c}$ fully onto the null space of the style LoRA to eliminate interference. This strict formulation completely removes style-interference effects, achieving a clear separation between content and style representations. Building on this foundation, we introduce a soft projection formulation that relaxes the hard projection into a continuous, tunable process. By modulating the projection strength with a single parameter, it adaptively suppresses interference while balancing subject fidelity and style consistency, effectively unifying both representations in LoRA fusion. The proposed NP-LoRA framework is fully training-free and applicable to a wide range of pretrained content-style LoRA pairs, offering a simple yet interpretable geometric view of LoRA fusion. Extensive experiments demonstrate that NP-LoRA consistently improves image fidelity and style coherence over strong baselines (e.g., DINO, CLIP-based, and human- and LLM-based preference metrics) across diverse backbones and LoRA combinations.

Our main contributions are summarized as follows:

- We propose Null Space Projection LoRA (NP-LoRA), a training-free projection-based geometric framework that unifies subject and style LoRAs through null space transformation.
- We provide a theoretical analysis showing that weight-based merging inherently introduces interference, and leverage SVD to identify the style subspace and construct its null space for interference-free fusion.
- We introduce a soft projection formulation that relaxes hard projection into a tunable process, adaptively suppressing interference and balancing subject fidelity and style consistency in LoRA fusion.
- We conduct extensive experiments demonstrating that NP-LoRA generalizes to diverse pretrained content-style LoRA pairs and consistently improves image fidelity and style coherence across strong baselines.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.11051/assets/method_8.png)

Fig. 2: Overview of the proposed method. NP-LoRA takes pretrained content and style LoRAs as inputs. The style LoRA is decomposed via singular value decomposition (SVD) to construct a null space, onto which the content LoRA is projected. This design enables effective fusion without extra training or hyperparameter tuning.

## II Related Work

Customization of diffusion models. Diffusion models [^11] [^33] [^5] [^25] have fueled extensive research on few-shot personalization. Approaches such as Textual Inversion [^1] [^36] [^45] and DreamBooth [^27] respectively learn token embeddings or fine-tune model parameters to capture subject identity. Subsequent methods, including Custom Diffusion [^17] and several training-free variants [^2] [^31] [^39] [^40], improve efficiency by reusing or selectively updating pretrained components. Despite these advances, most approaches remain computationally demanding and struggle to generalize. Low-Rank Adaptation (LoRA) [^12] [^10] [^16] [^23] [^44] [^47] [^48] [^26] offers an efficient alternative through rank-decomposed adapters that balance flexibility and efficiency. Among its extensions, B-LoRA [^7] learns disentangled style and content LoRAs for later fusion. In this work, we study how independently trained LoRAs can be effectively merged to achieve harmonious concept fusion.

Merging multiple LoRAs for image generation. Unlike joint training methods that learn content and style from scratch, we focus on fusing independently trained LoRAs. In image generation, LoRA merging generally falls into two categories: object composition [^6] [^9] [^15] [^19] [^41] [^46] [^42] and content-style fusion [^29] [^20] [^30]. We target the latter, aiming to combine content and style LoRAs while preserving both subject fidelity and stylistic consistency. Among recent methods, ZipLoRA [^29] and K-LoRA [^20] go beyond uniform addition but still rely on weighted combinations; i.e., ZipLoRA optimizes a fusion ratio to balance content and style, while K-LoRA adaptively switches between them during diffusion. LoRA.rar [^30] further employs a hypernetwork trained on content-style LoRA pairs to generalize to unseen combinations. Despite these advances, existing methods fuse LoRAs through weight-based operations, implicitly assuming that the two can be merged without interference, i.e., an assumption our analysis later disproves.

## III Background

LoRA Fine-tuning. To efficiently adapt large diffusion models [^11] [^33] [^5] [^25] to specific concepts, Low-Rank Adaptation (LoRA) [^12] has become a widely used fine-tuning technique. Instead of updating all pretrained parameters, LoRA inserts lightweight low-rank adapters into existing weight matrices. Given a pretrained weight $W_{0}\in\mathbb{R}^{m\times n}$, the LoRA update is

$$
\Delta W=BA,\quad B\in\mathbb{R}^{m\times r},;A\in\mathbb{R}^{r\times n},
$$

where $r\ll\min(m,n)$. Only $A$ and $B$ are optimized while $W_{0}$ remains frozen, greatly reducing memory and training cost. At inference, the adapted weight becomes

$$
W=W_{0}+\Delta W.
$$

Once trained, LoRA modules can be directly applied to a diffusion backbone, conditioning it on new subjects or styles.

Problem Setup. This work aims to merge two trained LoRAs, one for content and one for style, into a single LoRA that generates images faithful to both. We begin with a diffusion model containing pretrained weights $W^{(i)}_{0}$ and corresponding LoRA updates $\Delta W^{(i)}$. For simplicity, we omit the layer index $(i)$ in subsequent formulations, as the same operation applies to all LoRA matrices [^29]. We denote the content LoRA as $\Delta W_{c}$, learned from images of a specific subject, and the style LoRA as $\Delta W_{s}$, learned from images of a particular style. Our objective is to obtain a merged LoRA $\Delta W_{m}$ that integrates both content and style information:

$$
\Delta W_{m}=\text{Merge}(\Delta W_{c},\Delta W_{s}).
$$

A fundamental requirement is that the merged LoRA should preserve the distinctive characteristics of both the content and style LoRAs without mutual interference. To satisfy this requirement, it is essential to understand how LoRA encodes content and style information internally. Therefore, we delve into the internal structure of LoRA to identify the key information that must be preserved to achieve interference-free fusion.

## IV Methodology

In this work, we present NP-LoRA, a training-free framework for fusing independently trained LoRAs, as illustrated in Fig. 2. We first perform singular value decomposition (SVD) to identify the principal directions of the style LoRA. A hard projection then maps the content LoRA onto the orthogonal complement of the style subspace, fully eliminating overlapping directions. To provide controllable flexibility, we further introduce a soft projection controlled by a tunable parameter, enabling smooth and adaptive fusion between content and style.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.11051/assets/motivation_3_compressed.png)

Fig. 3: Singular value spectrum of a LoRA and perturbation effects. We first visualize the singular value spectrum, then perturb the principal and minor directions respectively. Perturbations on principal directions destroy style consistency, whereas those on minor directions have little impact.

### IV-A Principal Direction Extraction via SVD

We extract principal directions via singular value decomposition (SVD), decomposing the style LoRA weight as

$$
\Delta W_{s}=U\Sigma V^{\top},
$$

where the columns of $V\in\mathbb{R}^{d_{in}\times r_{s}}$ are orthonormal right singular vectors. Each vector $v\in V$ defines a distinct direction in the parameter space, and its corresponding singular value in $\Sigma$ quantifies the contribution strength of $\Delta W_{s}$ along that direction. Following the convention of K-LoRA [^20] [^27], all LoRAs are trained with rank 8; thus, we regard the top eight singular vectors (those with the largest singular values) as the principal directions. To investigate their impact, we visualize the generative effects of perturbing these directions. As shown in Fig. 3, perturbing dominant components causes a sharp drop in stylistic consistency, whereas perturbing minor ones yields minimal change, confirming that these principal directions play a critical role in generation fidelity.

Having identified the principal style directions that should be preserved, we next examine whether existing weight-based merging can protect them. A straightforward combination of content and style LoRAs takes the form

$$
\Delta W_{\mathrm{m}}=a\Delta W_{c}+b\Delta W_{s},\quad a,b\in\mathbb{R}.
$$

However, such linear fusion fails to prevent interference with the style-critical subspace introduced by the content LoRA, as shown in Fig. 4 (c).

###### Proposition 1.

Given content LoRA $\Delta W_{c}$ and style LoRA $\Delta W_{s}$, their weighted combination cannot, in general, isolate the content feature from the style-critical subspace. As a result, content-induced interference is unavoidable, and stylistic consistency cannot be guaranteed.

###### Proof.

Let the style subspace be spanned by the top singular vectors of $\Delta W_{s}$, denoted by $V_{k}$, and define the projection operator $P=V_{k}V_{k}^{\top}$. Decomposing the content LoRA yields

$$
\Delta W_{c}=(I-P)\Delta W_{c}+P\Delta W_{c},
$$

where $P\Delta W_{c}$ lies in the style subspace. For a weighted merge $\Delta W_{m}=a\Delta W_{c}+b\Delta W_{s}$, its projection onto the style subspace is

$$
P\Delta W_{m}=a\,P\Delta W_{c}+b\,\Delta W_{s}.
$$

To preserve the original style component, we require $P\Delta W_{m}=\Delta W_{s}$, implying

$$
a\,P\Delta W_{c}=(1-b)\Delta W_{s}.
$$

This condition admits a solution only if $P\Delta W_{c}$ and $\Delta W_{s}$ are colinear (i.e., $P\Delta W_{c}=\alpha\Delta W_{s}$), which is almost never the case for independently trained LoRAs. Even with vectorized weights $a,b$, each column of $P\Delta W_{c}$ must lie in the span of $\Delta W_{s}$, which is statistically improbable in high-dimensional space. Hence, weighted merging inevitably introduces interference within the style subspace, leading to style degradation. ∎

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.11051/assets/method_visual_compressed.png)

Fig. 4: Visualization of our method. (a) Content image. (b) Style image. (c) Results of direct weight-based merging, which causes style distortion and interference. (d) Results of hard projection (the base formulation of our method), which removes interference but suppresses content details. (e) Results of soft projection (ours), achieving a balanced fusion of subject fidelity and style consistency.

### IV-B Hard Projection for Subspace Separation

Having identified the principal style directions that must be preserved and shown that weight-based merging fails to protect them, we now introduce our projection-based formulation. We inject the content LoRA into the style LoRA, as style representations are inherently more fragile and easily overwritten than structural or semantic features [^8] [^13] [^18]. As demonstrated in Sec. V-D, this design preserves stylistic consistency while retaining content-specific information.

We define the orthogonal projection onto the null space of the style subspace:

$$
P_{\text{null}}=I-V_{k}V_{k}^{\top},
$$

where $I$ denotes the identity matrix. This operator removes any component aligned with the protected style directions, retaining only the part that is orthogonal to them. Applying this projection to the content LoRA $\Delta W_{c}$ yields the style-orthogonal component:

$$
\Delta W_{c}^{\perp}=\Delta W_{c}P_{\text{null}}.
$$

The merged LoRA is defined as

$$
\displaystyle\Delta W_{m}
$$
 
$$
\displaystyle=\Delta W_{s}+\Delta W_{c}^{\perp}
$$
 
$$
\displaystyle=\Delta W_{s}+\Delta W_{c}(I-V_{k}V_{k}^{\top}).
$$

This construction ensures that the content LoRA $\Delta W_{c}$ contributes only through directions orthogonal to the style subspace, leaving the principal style directions exclusively governed by $\Delta W_{s}$, as shown in Fig. 4 (d).

###### Proposition 2.

By applying Eq. 11, the proposed projection-based merging ensures that only style-orthogonal content information is injected. Consequently, no content-induced interference can occur within the protected style subspace.

###### Proof.

For any input $x$, the projected content LoRA is

$$
\Delta W_{c}^{\perp}x=\Delta W_{c}P_{\text{null}}x.
$$

To verify that it does not intrude into the style-critical subspace, we examine its projection onto it:

$$
\displaystyle P(\Delta W_{c}^{\perp}x)
$$
 
$$
\displaystyle=P\,\Delta W_{c}P_{\text{null}}x
$$
 
$$
\displaystyle=V_{k}V_{k}^{\top}\,\Delta W_{c}(I-V_{k}V_{k}^{\top})x.
$$

Since $V_{k}^{\top}V_{k}=I_{k}$, it follows that

$$
V_{k}V_{k}^{\top}(I-V_{k}V_{k}^{\top})=0,
$$

which immediately yields

$$
P(\Delta W_{c}^{\perp}x)=0.
$$

Thus, $\Delta W_{c}^{\perp}x$ has no component in $\mathrm{span}(V_{k})$ and is completely orthogonal to the style-critical subspace. ∎

This confirms that $\Delta W_{c}^{\perp}x$ lies entirely in the orthogonal complement of the style subspace, ensuring that the style-critical directions governed by $\Delta W_{s}$ remain unaffected. From an energy perspective, this means that the contribution of the content LoRA within the style subspace is suppressed. We quantify this residual interaction by the following quadratic form:

$$
(\Delta W_{c}^{\text{proj}})^{\top}P\Delta W_{c}^{\text{proj}},
$$

which measures the remaining energy of $\Delta W_{c}^{\text{proj}}$ along the style subspace. The hard projection in Eq. 10 minimizes this energy, effectively driving style interference to zero.

### IV-C Soft Projection for Balanced Fusion

While hard projection achieves perfect style preservation, it also removes content features that partially align with style directions, as shown in Fig. 4 (d). As a result, the projection becomes overly conservative, motivating the need for a more flexible alternative. We therefore introduce a relaxed formulation that balances style preservation and content retention. Specifically, we consider a Tikhonov-regularized [^34] variant of the previous objective:

$$
\min_{\Delta W_{c}^{\text{proj}}}\;\|\Delta W_{c}^{\text{proj}}-\Delta W_{c}\|_{2}^{2}+\mu\,(\Delta W_{c}^{\text{proj}})^{\top}P\,\Delta W_{c}^{\text{proj}},
$$

where the first term encourages closeness to the original content LoRA $\Delta W_{c}$, and the second penalizes the energy retained within the style subspace, with $\mu\geq 0$ controlling the suppression strength.

Taking the derivative and setting it to zero gives

$$
\Delta W_{c}^{\text{proj}}=(I+\mu P)^{-1}\Delta W_{c}.
$$

Since $P=V_{k}V_{k}^{\top}$ with $V_{k}^{\top}V_{k}=I_{k}$, the inverse admits a simple closed form derived from the Woodbury identity [^37] (see Supplementary Material for a detailed derivation):

$$
(I+\mu P)^{-1}=I-\tfrac{\mu}{1+\mu}V_{k}V_{k}^{\top}.
$$

This yields the soft projection operator and the projected content LoRA:

$$
P_{\text{soft}}=I-\tfrac{\mu}{1+\mu}V_{k}V_{k}^{\top},\qquad\Delta W_{c}^{\text{proj}}=P_{\text{soft}}\,\Delta W_{c}.
$$

Finally, the merged LoRA is

$$
\displaystyle\Delta W_{m}
$$
 
$$
\displaystyle=\Delta W_{s}+\Delta W_{c}^{\text{proj}}
$$
 
$$
\displaystyle=\Delta W_{s}+(I-\tfrac{\mu}{1+\mu}V_{k}V_{k}^{\top})\,\Delta W_{c}.
$$

This formulation explicitly separates the two components: the part of $\Delta W_{c}$ outside the style subspace is preserved, while the overlapping style-interference part is attenuated by a factor of $\tfrac{\mu}{1+\mu}$. By adjusting the parameter $\mu$, our method provides a continuous spectrum from linear merging $(\mu\to 0)$ to strict orthogonal projection $(\mu\to\infty)$. In this way, NP-LoRA achieves balanced fusion that maintains stylistic consistency without compromising content semantics, as shown in Fig. 4(e).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.11051/assets/ablation2_compressed.png)

Fig. 5: (a) and (e) are the content and style references, respectively. (c) shows the result of direct merging, which corresponds to μ = 0 \\mu=0 (i.e., the degenerate case of NP-LoRA). (d)-(g) show results for 0.1, 0.5 1 10 \\mu=0.1,0.5,1,10, and ∞ \\infty, where \\mu=\\infty represents the hard projection case of our method. Small \\mu values cause content-style interference, whereas large values preserve style at the expense of content fidelity.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.11051/assets/main_3_compressed.png)

Fig. 6: Qualitative comparison with Direct Weight Merge, B-LoRA, ZipLoRA, K-LoRA, LoRA.rar, and our proposed NP-LoRA, illustrating the trade-off between subject fidelity and style preservation.

## V Experiments

### V-A Experiment Setup

Datasets. Following common practice in customization [^29] [^20], we obtain LoRAs from publicly available images. For content, we adopt the DreamBooth [^28] dataset, where each subject is represented by 4–5 reference images. For style, we use the StyleDrop [^32] dataset, complemented with several representative artistic styles. For community-trained LoRAs, we use publicly available models from Hugging Face for evaluation.

Experimental details. All experiments are conducted on the SDXL v1.0 [^21] and FLUX [^3] models. For our method, we set the rank to 8 and preserve all principal directions, as described in Sec. IV-A. The key hyperparameter $\mu$ in the soft projection (Eq. 21) is fixed to 0.5, with a detailed analysis provided in Sec. V-B. Implementation details are provided in the supplemental material. Experiments based on SDXL are performed on an NVIDIA RTX 3090 GPU, while those on FLUX are conducted on an NVIDIA A800 GPU.

Baselines. We compare our method with representative approaches for combining or learning multiple LoRAs. Among them, most methods merge two independently trained LoRAs, i.e., the model takes two trained LoRAs as input and produces merged generations. Specifically, we evaluate (i) the direct weight merging baseline $\Delta W_{m}=\Delta W_{c}+\Delta W_{s}$, (ii) B-LoRA [^7] (ECCV2024), (iii) ZipLoRA [^29] (ECCV2024), (iv) K-LoRA [^20] (CVPR2025), and (v) LoRA.rar [^30] (ICCV2025).

### V-B Ablation on Projection Strength

We analyze the effect of the projection strength controlled by the parameter $\mu$, which balances subject fidelity and style consistency, as shown in Fig. 5. As $\mu$ increases from 0 to $\infty$, the projection strength grows: when $\mu\to 0$, NP-LoRA reduces to direct weight-based merging dominated by the content LoRA, yet leading to content-style interference, and as $\mu\to\infty$, it approaches a hard projection that removes content components overlapping with the style subspace. We evaluate NP-LoRA across $\mu\in\{0,0.1,0.5,1,10,\infty\}$ to study this transition. This ablation reveals a clear trade-off: small $\mu$ values lead to content-style interference, while large $\mu$ values preserve style but suppress content details. In particular, the hard projection ($\mu=\infty$) loses rich content features, confirming the necessity of the soft projection mechanism. Empirically, $\mu=0.5$ yields the best balance and is used as the default in all subsequent experiments. Additional qualitative and quantitative results are provided in the Supplementary Material.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.11051/assets/exp_projection_compressed.png)

Fig. 7: Comparison of LoRA Projection. (a) Content LoRA training images, (b) style LoRA projected into the content LoRA null space, (c) direct merge, (d) content LoRA projected into the style LoRA null space, and (e) style LoRA training images. (b) and (c) degrade style consistency, with (b) almost removing stylistic traits, while (d) better balances content and style.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.11051/assets/exp_flux3_compressed.png)

Fig. 8: Qualitative results on flux for generalization validation

TABLE I: Comparison of methods using CLIP and DINO similarities for content and style. The aggregated scores ($S_{\text{arith}}$, $S_{\text{harm}}$) summarize the overall content-style trade-off.

| Method | $S^{\text{CLIP}}_{\text{content}}$ | $S^{\text{CLIP}}_{\text{style}}$ | $S^{\text{DINO}}_{\text{content}}$ | $S^{\text{DINO}}_{\text{style}}$ | $S_{\text{arith}}$ | $S_{\text{harm}}$ $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| Direct | 0.75 | 0.55 | 0.54 | 0.23 | 0.51 | 0.42 |
| B-LoRA [^7] | 0.72 | 0.55 | 0.51 | 0.26 | 0.51 | 0.44 |
| ZipLoRA [^29] | 0.75 | 0.56 | 0.55 | 0.24 | 0.52 | 0.44 |
| K-LoRA [^20] | 0.76 | 0.55 | 0.57 | 0.22 | 0.52 | 0.42 |
| LoRA.rar [^30] | 0.70 | 0.53 | 0.50 | 0.30 | 0.51 | 0.46 |
| NP-LoRA | 0.73 | 0.59 | 0.52 | 0.33 | 0.55 | 0.50 |

TABLE II: User preference and GPT-5 evaluation comparison. Both human users and GPT-5 are asked to select the best result among the candidates. The reported score represents the proportion of cases where each method is selected as the best.

| Method | User Preference $\uparrow$ | GPT5 Feedback $\uparrow$ |
| --- | --- | --- |
| Direct | 7.97% | 6.25% |
| B-LoRA [^7] | 9.06% | 9.38% |
| ZipLoRA [^29] | 10.00% | 9.38% |
| K-LoRA [^20] | 12.19% | 12.50% |
| LoRA.rar [^30] | 11.25% | 9.38% |
| NP-LoRA | 49.53% | 53.13% |

TABLE III: Efficiency comparison of different LoRA merging strategies. Merge (s <sup>†</sup>) includes model loading time; Gen./Img. (s) is the average generation time per image.

| Method | Merge (s <sup>†</sup>)  $\downarrow$ | Gen./Img. (s)  $\downarrow$ |
| --- | --- | --- |
| Direct | 13.466 | 22.898 |
| ZipLoRA [^29] | $>10$  min | 29.931 |
| K-LoRA [^20] | 12.250 <sup>∗</sup> | 60.388 <sup>∗</sup> |
| NP-LoRA | 13.440 | 23.208 |

<sup>∗</sup> Only includes LoRA loading time, as K-LoRA performs on-the-fly merging during generation.

### V-C Results

#### Quantitative comparisons.

In order to ensure a fair evaluation, all experiments at this stage are conducted using SDXL. Following prior works including K-LoRA [^20] (18 pairs) and ZipLoRA [^29] (32 pairs), we evaluate our NP-LoRA on 32 randomly selected content-style LoRA pairs, covering 5 animal subjects, 5 object-level categories, and 15 distinct artistic styles. Each pair contains 10 generated images. For the evaluation, following [^29] [^20], we adopt CLIP [^22] and DINO [^43] similarities for jointly evaluating both subject preservation (content) and style alignment (style) [^14] [^29] [^20]. Specifically, for CLIP we compute $S^{\text{CLIP}}_{\text{content}}$ and $S^{\text{CLIP}}_{\text{style}}$, and for DINO we compute $S^{\text{DINO}}_{\text{content}}$ and $S^{\text{DINO}}_{\text{style}}$. To obtain unified content-style similarity measures, we report both the arithmetic mean and the harmonic mean following the common F1-style aggregation scheme [^35] [^4] [^24] [^38]. Given the four similarity scores $S_{i}\in\{S^{\text{CLIP}}_{\text{content}},S^{\text{CLIP}}_{\text{style}},S^{\text{DINO}}_{\text{content}},S^{\text{DINO}}_{\text{style}}\}$, we report both the arithmetic mean $S_{\text{arith}}=\tfrac{1}{4}\sum_{i}S_{i}$ and the harmonic mean $S_{\text{harm}}=\tfrac{4}{\sum_{i}(1/S_{i})}$, where $S_{\text{arith}}$ reflects the overall trend, while the harmonic mean $S_{\text{harm}}$ emphasizes joint fidelity to both content and style. As shown in Tab. I, our method achieves the highest $S_{\text{arith}}$ and $S_{\text{overall}}$, showing its superior ability to balance subject fidelity and style consistency.

To further assess perceptual quality, we conducted a user study with 20 participants following K-LoRA [^20]. Each case included outputs from SOTA baselines and our method, along with reference subject and style images. Participants selected the result best balancing subject fidelity and style consistency. As shown in Tab. II, our method was preferred in $49.53\%$ of cases. We further employed GPT-5-based automatic evaluation, which yielded a consistent $53.13\%$ preference for our method.

#### Qualitative comparisons.

Qualitative results are shown in Fig. 6. Direct merging and ZipLoRA [^29] achieve partial fusion but often fail on certain content-style pairs. B-LoRA [^7] captures style traits, but compromises on the content fidelity. K-LoRA [^20] preserves subject identity well yet struggles to capture the desired style, while LoRA.rar [^30] delivers only marginal improvements but still falls short. In contrast, NP-LoRA achieves the best results, achieving a superior balance between content fidelity and style preservation. A qualitative comparison with jointly training method in the Supplementary Material shows that NP-LoRA attains superior results.

#### Efficiency Comparison.

To evaluate efficiency, we compare both the merging and generation times in Tab. III. LoRA.rar [^30] is excluded because it depends on pretrained models for parameter fusion, and B-LoRA [^7] is also omitted since it requires training from scratch instead of merging pretrained LoRAs. ZipLoRA [^29] is the slowest due to additional fusion optimization, whereas K-LoRA [^20] is efficient during initialization but slows down because of dynamic fusion in the sampling stage. In contrast, NP-LoRA maintains merging and inference efficiency comparable to the Direct Merge baseline, yet consistently delivers superior results.

### V-D Projection Direction: Content into Style

As shown in Fig. 6 and Tab. I, direct merging overwhelmingly favors the content LoRA: structural information is preserved while stylistic traits are largely lost. The projection in Fig. 7 further confirms that embedding style into the content space almost fully erases style signals. This suggests that content representations are inherently more dominant and resistant to interference, whereas style features are fragile. Therefore, we reverse the direction of projection and map the content LoRA onto the null space of the style LoRA, injecting content only along interference-free directions. This preserves stylistic consistency while still retaining content-specific information.

### V-E Evaluation on Flux for Generalization

We further verify the generality of our method on the FLUX [^3] backbone, as illustrated in Fig. 8. Among existing approaches, only K-LoRA provides an official implementation for FLUX, and we include it as a reference. We use publicly available LoRAs to perform content-style fusion across several pairs. While all methods produce visually plausible results on FLUX, subtle differences remain: direct merging produces entangled outputs, K-LoRA favors content, and our NP-LoRA achieves a balanced fusion of content and style. These observations demonstrate that our projection-based design generalizes well across different diffusion backbones.

## VI Conclusions

In this paper, we propose NP-LoRA, a merging strategy designed to harmoniously unify style and content LoRAs while avoiding destructive interference. Our method first identifies the style-critical subspace via singular value decomposition (SVD) and projects the content LoRA onto its orthogonal complement through a hard projection. We then softly attenuate the overlapping components according to the desired level of flexibility. Furthermore, we theoretically analyze why conventional weight merging fails and show why null space projection effectively resolves this issue. Comprehensive experiments, including quantitative metrics, user studies, and GPT-5 evaluations, confirm the effectiveness and preference of our method. Notably, NP-LoRA delivers significant performance gains while maintaining comparable runtime efficiency to direct merging.

Limitations. Our method builds a single-layer projection mechanism under the assumption that LoRA representations are cross-layer independent, which is shared with prior merging approaches [^29] [^20] [^30]. This assumption may not fully capture complex non-linear or cross-layer dependencies. Moreover, the attenuation coefficient $\mu$ in NP-LoRA is empirically chosen; future work will explore adaptive or data-driven strategies for automatic selection.

## References

## Supplementary Material Overview

This supplementary material provides additional technical details, derivations, and extended analyses to support the findings presented in the main paper. Specifically, we first derive the formulation of the proposed soft projection operator in Sec. VII and present details for efficient implementation in Sec. VIII. Sec. IX investigates which null-space construction (U-space or V-space) better preserves stylistic fidelity. Sec. XI provides an additional comparison with joint training to further validate the effectiveness of NP-LoRA over training-based approaches. Sec. XII demonstrates the controllability of our method under various prompt conditions, while Sec. XIII evaluates its robustness across different random seeds. Sec. XIV extends our validation to the Flux model to assess cross-model generalization. Finally, Sec. XV details the GPT-5 evaluation protocol used for quantitative preference assessment.

Together, these sections offer deeper insights into NP-LoRA’s formulation, implementation, and empirical robustness beyond the main paper.

## VII Derivation of the Soft Projection Operator

We start from the relaxed objective introduced in the main text of our paper:

$$
\min_{\Delta W_{c}^{\text{proj}}}\;\|\Delta W_{c}^{\text{proj}}-\Delta W_{c}\|_{2}^{2}+\mu\,(\Delta W_{c}^{\text{proj}})^{\top}P\,\Delta W_{c}^{\text{proj}},
$$

where the first term preserves proximity to the original content adapter, and the second penalizes its activation within the style subspace defined by $P=V_{k}V_{k}^{\top}$.

#### Closed-form solution.

Taking the derivative of (22) with respect to $\Delta W_{c}^{\text{proj}}$ and setting it to zero gives

$$
(I+\mu P)\,\Delta W_{c}^{\text{proj}}=\Delta W_{c},
$$

hence

$$
\Delta W_{c}^{\text{proj}}=(I+\mu P)^{-1}\Delta W_{c}.
$$

#### Simplifying the inverse.

Since $P=V_{k}V_{k}^{\top}$ is a rank- $k$ projector with $V_{k}^{\top}V_{k}=I_{k}$, the inverse matrix admits the closed form

$$
(I+\mu P)^{-1}=I-\tfrac{\mu}{1+\mu}V_{k}V_{k}^{\top},
$$

which follows from either spectral decomposition or the Woodbury identity.

Substituting (25) into (24), we obtain the soft projection operator

$$
P_{\text{soft}}=I-\tfrac{\mu}{1+\mu}V_{k}V_{k}^{\top},\qquad\Delta W_{c}^{\text{proj}}=P_{\text{soft}}\,\Delta W_{c}.
$$

#### Final fusion rule.

The merged LoRA is then defined as

$$
\Delta W_{m}=\Delta W_{s}+\Delta W_{c}^{\text{proj}}=\Delta W_{s}+(I-\tfrac{\mu}{1+\mu}V_{k}V_{k}^{\top})\,\Delta W_{c}.
$$

This formulation continuously interpolates between direct merging ($\mu=0$) and hard projection ($\mu\rightarrow\infty$), offering a controllable balance between style preservation and content retention.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.11051/assets/supp_UorV_2_compressed.png)

Fig. 9: Comparison of output-Space (U) and parameter-space (V) projections for null-space construction. The U-space projection fails to remove style interference, leading to distorted stylistic features, whereas the V-space projection effectively eliminates such interference and preserves the intended style appearance.

## VIII Implementation Details

In our theoretical formulation, we employ the singular value decomposition (SVD) to obtain the right singular subspace of the LoRA update matrix $\Delta W_{s}=B_{s}A_{s}$, where $A_{s}\in\mathbb{R}^{r\times n}$ and $B_{s}\in\mathbb{R}^{m\times r}$. Specifically, the original projection of $\Delta W_{c}$ onto the orthogonal complement of $\Delta W_{s}$ is computed using the right singular vectors $V_{k}$ of $\Delta W_{s}$, i.e.,

$$
\Delta W_{c}^{\perp}=\Delta W_{c}(I-V_{k}V_{k}^{\top}).
$$

However, directly computing $V_{k}$ via SVD or PCA on the large matrix $\Delta W_{s}\in\mathbb{R}^{m\times n}$ is computationally expensive, as its complexity scales with $\mathcal{O}(mnr)$.

#### Efficient QR-based implementation.

To improve efficiency, we replace the SVD with a QR-based projection that is mathematically equivalent but far more efficient. Observe that $\Delta W_{s}$ has rank at most $r$ and its right singular subspace is equal to the column space of $A_{s}^{\top}$:

$$
\mathrm{span}(V)=\mathrm{span}(A_{s}^{\top}).
$$

Therefore, we compute a thin QR decomposition on $A_{s}^{\top}$:

$$
A_{s}^{\top}=QR,
$$

where $Q\in\mathbb{R}^{n\times r}$ has orthonormal columns spanning $\mathrm{span}(A_{s}^{\top})$, and $R\in\mathbb{R}^{r\times r}$ is an upper triangular matrix. We then replace $V$ with $Q$ and compute

$$
\Delta W_{c}^{\text{proj}}=\Delta W_{c}-\Delta W_{c}QQ^{\top},
$$

which requires only a QR decomposition of a small matrix of size $n\times r$ (where $r\ll n,m$), reducing complexity to $\mathcal{O}(nr^{2})$.

#### Proof of equivalence.

Let $\Delta W_{s}=B_{s}A_{s}$ denote the LoRA matrix. Since LoRA uses low-rank factors (A ${}_{s}\in\mathbb{R}^{r\times n}$) and ($B_{s}\in\mathbb{R}^{m\times r}$) with full column and row rank (i.e., $rank=r$), ($B_{s}^{\top}B_{s}$) is symmetric positive definite. Then

$$
\Delta W_{s}^{\top}\Delta W_{s}=A_{s}^{\top}(B_{s}^{\top}B_{s})A_{s}.
$$

Since $B_{s}^{\top}B_{s}$ is symmetric positive definite, it admits a Cholesky factorization $B_{s}^{\top}B_{s}=C^{\top}C$ with $C$ invertible. Substituting gives

$$
\Delta W_{s}^{\top}\Delta W_{s}=(CA_{s})^{\top}(CA_{s}).
$$

As $C$ represents a full-rank linear transform, it does not alter the column space of $A_{s}^{\top}$. Hence, the right singular subspace of $\Delta W_{s}$ coincides with $\mathrm{span}(A_{s}^{\top})$:

$$
\mathrm{Col}(\Delta W_{s}^{\top}\Delta W_{s})=\mathrm{span}(A_{s}^{\top}).
$$

Therefore, the orthogonal projectors constructed via SVD and QR are identical:

$$
P=V_{k}V_{k}^{\top}=QQ^{\top}.
$$

In practice, this means that replacing the expensive low-rank SVD with a simple QR decomposition of $A_{s}^{\top}$ yields a mathematically equivalent projection while reducing computation time by an order of magnitude.

## IX Which Null-Space Construction Preserves Style: UU-space or VV-space?

In the main paper, we construct the null space using the right singular vectors $V$ from the SVD decomposition $\Delta W_{s}=U\Sigma V^{\top}$, since interference in LoRA fusion occurs in the parameter space (i.e., the column space of $\Delta W_{s}$) rather than in the activation space represented by $U$. Specifically, $U$ spans the output (activation) domain of $\Delta W_{s}$, describing how style LoRA influences feature activations, while $V$ spans the input (parameter) domain, capturing how LoRA updates are distributed across the weight columns. Because LoRA fusion directly manipulates weight parameters, interference emerges along these column directions, making $V$ the appropriate basis for constructing the null space. Projecting in the $V$ -space thus directly eliminates content-induced perturbations along the style-critical directions within the weight domain. In contrast, constructing the null space based on $U$ operates in the output (activation) space, which suppresses both content and style responses and weakens overall expressiveness. As shown in Fig. 9, projections built from $V$ effectively preserve stylistic consistency, while those derived from $U$ fail to disentangle content and style, resulting in noticeable style degradation.

## X Additional Ablation on Soft Projection

We refer to the version without soft relaxation as the hard variant, denoted as NP-LoRA-H. The qualitative results are shown in Fig. 10, and the quantitative comparison is provided in Tab. IV. As illustrated, NP-LoRA-H effectively preserves the style but tends to lose content details. In contrast, our full NP-LoRA achieves a better balance between content and style, yielding the highest overall scores. This confirms the necessity of the proposed soft relaxation mechanism.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.11051/assets/supp_joint_compressed.png)

Fig. 10: Qualitative comparison with joint training. Joint training exhibits unstable performance and often fails to merge content and style effectively, while N-LoRA achieves consistent and well-balanced fusion without retraining.

## XI Additional Comparison with Joint Training

We further compare our method with joint training baseline, where a new LoRA is trained from scratch using mixed content and style images. The qualitative results are shown in Fig. 10, and the quantitative comparison is provided in Tab. IV. As observed, the joint-training approach exhibits high instability, i.e., while it occasionally produces reasonable fusion, it often fails to effectively combine content and style. This instability may arise from catastrophic forgetting during multi-objective optimization, where learning new style features interferes with previously acquired content representations. In contrast, our NP-LoRA achieves consistent and harmonious fusion without any retraining, effectively preserving both content fidelity and stylistic characteristics. Quantitatively, as shown in Table IV, NP-LoRA outperforms joint training across all metrics, with NP-LoRA achieving the highest overall score, confirming its superior ability to preserve content and style simultaneously.

TABLE IV: Comparison with joint training method using CLIP and DINO similarity scores for content preservation and style alignment. Only the aggregated results ($S_{\text{arith}}$ and $S_{\text{harm}}$) reflect the overall trade-off between content and style; thus, only these two columns are emphasized.

| Method | $S^{\text{CLIP}}_{\text{content}}$ | $S^{\text{CLIP}}_{\text{style}}$ | $S^{\text{DINO}}_{\text{content}}$ | $S^{\text{DINO}}_{\text{style}}$ | $S_{\text{arith}}$ | $S_{\text{harm}}$ $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| Joint Training | 0.67 | 0.61 | 0.49 | 0.25 | 0.50 | 0.43 |
| NP-LoRA-H | 0.68 | 0.60 | 0.39 | 0.40 | 0.52 | 0.48 |
| NP-LoRA | 0.73 | 0.59 | 0.52 | 0.33 | 0.55 | 0.50 |

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.11051/assets/suup_prompt_control_2_compressed.png)

Fig. 11: Our method effectively modifies the object’s actions and environment while maintaining the original style.

## XII Prompt Control

We conduct experiments to assess whether our method can alter the object’s actions, the surrounding environment, or introduce new elements by adjusting the prompts. As shown in 11, after modifying the prompts, our method effectively preserves the original object’s characteristics and stylistic attributes, while seamlessly integrating new elements or scene details.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.11051/assets/supp_robust_compressed.png)

Fig. 12: Results obtained with randomly selected seeds demonstrate the stability and robustness of our NP-LoRA.

## XIII Robustness Test

We evaluate the robustness of our method by running experiments with different random seeds to examine stability across stochastic conditions. As shown in Fig. 12, our approach produces consistently stable results under varying seeds, matching the qualitative trends observed in the main paper. Specifically, NP-LoRA achieves a balanced trade-off between content preservation and style consistency, yielding the most visually harmonious results.

## XIV Further Generalization Validation on FLUX

To further validate the generality of NP-LoRA, we apply our method to the FLUX backbone using publicly available LoRAs covering diverse subjects and styles. As shown in Fig. 13, 14, NP-LoRA consistently preserves the balance between subject identity and stylistic fidelity across various domains. These results demonstrate that our projection-based fusion mechanism remains effective even under different diffusion architectures, highlighting the robustness and general applicability of the proposed approach.

## XV Details for GPT-5 Evaluation Protocol

In the main paper, we employed the GPT-5 model to automatically evaluate and select the candidate image that best balances Subject Fidelity and Style Consistency. The exact prompt used for GPT-5 is listed below to ensure full reproducibility.

During evaluation, GPT-5 received the reference subject image (A.jpg), the reference style image (B.jpg), and six generated candidates from different merging strategies. It then followed the above prompt to independently select the best-balanced image. Each case was judged in isolation to prevent cross-sample bias. The final GPT-5 preference score reported represents the proportion of cases where each method was selected as the best overall result.

<svg id="S15.p3.pic1" height="759.39" overflow="visible" version="1.1" viewBox="0 0 477.38 759.39" width="477.38"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,759.39) matrix(1 0 0 -1 0 0)"><g style="--ltx-fill-color:#F5F5F5;" fill="#F5F5F5" fill-opacity="1.0"><path style="stroke:none" d="M 0 5.91 L 0 753.48 C 0 756.74 2.64 759.39 5.91 759.39 L 471.47 759.39 C 474.73 759.39 477.38 756.74 477.38 753.48 L 477.38 5.91 C 477.38 2.64 474.73 0 471.47 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z"></path></g><g style="--ltx-fill-color:#FEFEFE;" fill="#FEFEFE" fill-opacity="1.0"><path style="stroke:none" d="M 1.97 5.91 L 1.97 735.27 L 475.41 735.27 L 475.41 5.91 C 475.41 3.73 473.65 1.97 471.47 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 743.87)"><foreignObject style="--ltx-fo-width:31.37em;--ltx-fo-height:0.69em;--ltx-fo-depth:0.19em;font-size:10pt;" height="12.3" overflow="visible" transform="matrix(1 0 0 -1 0 9.61)" width="434.07"><span id="S15.p3.pic1.1" style="width:31.37em;"><span id="S15.p3.pic1.1.1"><span id="S15.p3.pic1.1.1.1" style="--ltx-fg-color:#FFFFFF;">Prompt used for GPT-5 evaluation</span></span> </span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 16.47)"><foreignObject style="--ltx-fo-width:31.37em;--ltx-fo-height:51.09em;--ltx-fo-depth:0.19em;font-size:10pt;" height="709.69" overflow="visible" transform="matrix(1 0 0 -1 0 707)" width="434.07"><span id="S15.p3.pic1.2" style="width:31.37em;"><span id="S15.p3.pic1.2.1"><span id="S15.p3.pic1.2.1.1" style="--ltx-fg-color:#000000;">Task: <span id="S15.p3.pic1.2.1.1.1">Evaluate and select the candidate image that best balances Subject Fidelity and Style Consistency.<br></span>Inputs:<br><span id="S15.p3.pic1.2.1.1.2">– First image = A.jpg (Subject Fidelity reference): core subject content to preserve.<br>– Second image = B.jpg (Style reference): artistic style, colors, lighting to apply.<br>– Next six images are candidates, named [Model_1]..[Model_6] in the given order.<br></span>Candidate mapping (fixed):<br><span id="S15.p3.pic1.2.1.1.3">[Model_1] <span id="S15.p3.pic1.2.1.1.3.1">= Direct<br></span>[Model_2] <span id="S15.p3.pic1.2.1.1.3.2">= Klora<br></span>[Model_3] <span id="S15.p3.pic1.2.1.1.3.3">= LoRArar<br></span>[Model_4] <span id="S15.p3.pic1.2.1.1.3.4">= B-LoRA<br></span>[Model_5] <span id="S15.p3.pic1.2.1.1.3.5">= ziplora<br></span>[Model_6] <span id="S15.p3.pic1.2.1.1.3.6">= NP-LoRA<br></span></span>Evaluation Criteria:<br><span id="S15.p3.pic1.2.1.1.4">1) Subject Fidelity: retain key characteristics/structure/recognizability from A.jpg.<br>2) Style Consistency: apply style/texture/colors/lighting present in B.jpg.<br></span>Crucial Requirement:<br><span id="S15.p3.pic1.2.1.1.5">Do not choose images that only retain the subject without the style, or only mimic the style with distorted subject. Select the one with the <em id="S15.p3.pic1.2.1.1.5.1">BEST BALANCE</em>.<br></span>Output Rules (IMPORTANT):<br><span id="S15.p3.pic1.2.1.1.6">You MUST respond with ONLY ONE of the following exact strings (case-sensitive):<br><span id="S15.p3.pic1.2.1.1.6.1">Direct<br>Klora<br>LoRArar<br>B-LoRA<br>ziplora<br>NP-LoRA<br></span></span></span></span><span id="S15.p3.pic1.2.2"><span id="S15.p3.pic1.2.2.1" style="--ltx-fg-color:#000000;">Do NOT output anything else. Do NOT include punctuation, brackets, markdown, or explanation.</span></span></span></foreignObject></g></g></svg>

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.11051/assets/supp_flux_soft_1_compressed.png)

Fig. 13: Qualitative results of NP-LoRA on the Flux backbone using diverse publicly available LoRAs. Each image corresponds to the combination of the content LoRA shown above and the style LoRA shown on the left, illustrating the results produced by our method.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.11051/assets/supp_flux_soft_2_compressed.png)

Fig. 14: Qualitative results of NP-LoRA on the Flux backbone using diverse publicly available LoRAs. Each image corresponds to the combination of the conten t LoRA shown above and the style LoRA shown on the left, illustrating the results produced by our method.

[^1]: Y. Alaluf, E. Richardson, G. Metzer, and D. Cohen-Or (2023) A neural space-time representation for text-to-image personalization. ACM Transactions on Graphics (TOG) 42 (6), pp. 1–10. Cited by: §II.

[^2]: O. Avrahami, K. Aberman, O. Fried, D. Cohen-Or, and D. Lischinski (2023) Break-a-scene: extracting multiple concepts from a single image. In SIGGRAPH Asia 2023 Conference Papers, pp. 1–12. Cited by: §II.

[^3]: S. Batifol, A. Blattmann, F. Boesel, S. Consul, C. Diagne, T. Dockhorn, J. English, Z. English, P. Esser, S. Kulal, et al. (2025) FLUX. 1 kontext: flow matching for in-context image generation and editing in latent space. arXiv e-prints, pp. arXiv–2506. Cited by: §V-A, §V-E.

[^4]: N. Chinchor (1992) MUC-4 evaluation metrics. In Proceedings of the 4th Conference on Message Understanding, MUC4 ’92, USA, pp. 22–29. External Links: ISBN 1558602739, [Link](https://doi.org/10.3115/1072064.1072067), [Document](https://dx.doi.org/10.3115/1072064.1072067) Cited by: §V-C.

[^5]: P. Dhariwal and A. Nichol (2021) Diffusion models beat gans on image synthesis. In Advances in Neural Information Processing Systems, M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. W. Vaughan (Eds.), Vol. 34, pp. 8780–8794. External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf) Cited by: §I, §II, §III.

[^6]: J. Dong, W. Liang, H. Li, D. Zhang, M. Cao, H. Ding, S. H. Khan, and F. Shahbaz Khan (2024) How to continually adapt text-to-image diffusion models for flexible customization?. Advances in Neural Information Processing Systems 37, pp. 130057–130083. Cited by: §II.

[^7]: Y. Frenkel, Y. Vinker, A. Shamir, and D. Cohen-Or (2024) Implicit style-content separation using b-lora. In European Conference on Computer Vision, pp. 181–198. Cited by: §II, §V-A, §V-C, §V-C, TABLE I, TABLE II.

[^8]: L. A. Gatys, A. S. Ecker, and M. Bethge (2016) Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2414–2423. Cited by: §IV-B.

[^9]: Y. Gu, X. Wang, J. Z. Wu, Y. Shi, Y. Chen, Z. Fan, W. Xiao, R. Zhao, S. Chang, W. Wu, et al. (2023) Mix-of-show: decentralized low-rank adaptation for multi-concept customization of diffusion models. Advances in Neural Information Processing Systems 36, pp. 15890–15902. Cited by: §II.

[^10]: S. Hayou, N. Ghosh, and B. Yu (2024) LoRA+ efficient low rank adaptation of large models. In Proceedings of the 41st International Conference on Machine Learning, pp. 17783–17806. Cited by: §II.

[^11]: J. Ho, A. Jain, and P. Abbeel (2020) Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems 33, pp. 6840–6851. Cited by: §I, §II, §III.

[^12]: E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al. (2022) Lora: low-rank adaptation of large language models.. ICLR 1 (2), pp. 3. Cited by: §I, §II, §III.

[^13]: X. Huang and S. Belongie (2017) Arbitrary style transfer in real-time with adaptive instance normalization. In Proceedings of the IEEE international conference on computer vision, pp. 1501–1510. Cited by: §IV-B.

[^14]: D. Jiang, Y. Liu, S. Liu, J. Zhao, H. Zhang, Z. Gao, X. Zhang, J. Li, and H. Xiong (2024) From clip to dino: visual encoders shout in multi-modal large language models. External Links: 2310.08825, [Link](https://arxiv.org/abs/2310.08825) Cited by: §V-C.

[^15]: J. Jiang, Y. Zhang, K. Feng, X. Wu, W. Li, R. Pei, F. Li, and W. Zuo (2025) MCˆ 2: multi-concept guidance for customized multi-concept generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 2802–2812. Cited by: §II.

[^16]: D. J. Kopiczko, T. Blankevoort, and Y. M. Asano (2024) VeRA: vector-based random matrix adaptation. In The Twelfth International Conference on Learning Representations, Cited by: §II.

[^17]: N. Kumari, B. Zhang, R. Zhang, E. Shechtman, and J. Zhu (2023) Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1931–1941. Cited by: §II.

[^18]: Y. Li, C. Fang, J. Yang, Z. Wang, X. Lu, and M. Yang (2017) Universal style transfer via feature transforms. Advances in neural information processing systems 30. Cited by: §IV-B.

[^19]: Z. Liu, R. Feng, K. Zhu, Y. Zhang, K. Zheng, Y. Liu, D. Zhao, J. Zhou, and Y. Cao (2023) Cones: concept neurons in diffusion models for customized generation. In Proceedings of the 40th International Conference on Machine Learning, pp. 21548–21566. Cited by: §II.

[^20]: Z. Ouyang, Z. Li, and Q. Hou (2025) K-lora: unlocking training-free fusion of any subject and style loras. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 13041–13050. Cited by: §I, §I, §II, §IV-A, §V-A, §V-A, §V-C, §V-C, §V-C, §V-C, TABLE I, TABLE II, TABLE III, §VI.

[^21]: D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach (2024) SDXL: improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, Cited by: §V-A.

[^22]: A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever (2021) Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, M. Meila and T. Zhang (Eds.), Proceedings of Machine Learning Research, Vol. 139, pp. 8748–8763. External Links: [Link](https://proceedings.mlr.press/v139/radford21a.html) Cited by: §V-C.

[^23]: P. Ren, C. Shi, S. Wu, M. Zhang, Z. Ren, M. Rijke, Z. Chen, and J. Pei (2024) MELoRA: mini-ensemble low-rank adapters for parameter-efficient fine-tuning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3052–3064. Cited by: §II.

[^24]: E. Ristani, F. Solera, R. Zou, R. Cucchiara, and C. Tomasi (2016) Performance measures and a data set for multi-target, multi-camera tracking. In European conference on computer vision, pp. 17–35. Cited by: §V-C.

[^25]: R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer (2022) High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10684–10695. Cited by: §I, §II, §III.

[^26]: A. Roy, S. Borse, S. Kadambi, D. Das, S. Mahajan, R. Garrepalli, H. Park, A. Nayak, R. Chellappa, M. Hayat, and F. Porikli (2025) DuoLoRA: cycle-consistent and rank-disentangled content-style personalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15395–15404. Cited by: §II.

[^27]: N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman (2023) Dreambooth: fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22500–22510. Cited by: §I, §II, §IV-A.

[^28]: N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman (2023) Dreambooth: fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22500–22510. Cited by: §V-A.

[^29]: V. Shah, N. Ruiz, F. Cole, E. Lu, S. Lazebnik, Y. Li, and V. Jampani (2024) Ziplora: any subject in any style by effectively merging loras. In European Conference on Computer Vision, pp. 422–438. Cited by: §I, §I, §II, §III, §V-A, §V-A, §V-C, §V-C, §V-C, TABLE I, TABLE II, TABLE III, §VI.

[^30]: D. Shenaj, O. Bohdal, M. Ozay, P. Zanuttigh, and U. Michieli (2025) LoRA.rar: learning to merge loras via hypernetworks for subject-style conditioned image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Cited by: §I, §I, §II, §V-A, §V-C, §V-C, TABLE I, TABLE II, §VI.

[^31]: J. Shi, W. Xiong, Z. Lin, and H. J. Jung (2024) Instantbooth: personalized text-to-image generation without test-time finetuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8543–8552. Cited by: §II.

[^32]: K. Sohn, N. Ruiz, K. Lee, D. C. Chin, I. Blok, H. Chang, J. Barber, L. Jiang, G. Entis, Y. Li, Y. Hao, I. Essa, M. Rubinstein, and D. Krishnan (2023) StyleDrop: text-to-image generation in any style. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA. Cited by: §V-A.

[^33]: J. Song, C. Meng, and S. Ermon (2021) Denoising diffusion implicit models. In International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=St1giarCHLP) Cited by: §I, §II, §III.

[^34]: A.N. Tikhonov and V.I.A. Arsenin (1977) Solutions of ill-posed problems. Halsted Press book, Winston. External Links: ISBN 9780470991244, LCCN 77003422, [Link](https://books.google.co.jp/books?id=ECrvAAAAMAAJ) Cited by: §IV-C.

[^35]: C.J. Van Rijsbergen (1979) Information retrieval. Butterworths. External Links: ISBN 9780408709293, LCCN 78040725, [Link](https://books.google.co.jp/books?id=t-pTAAAAMAAJ) Cited by: §V-C.

[^36]: A. Voynov, Q. Chu, D. Cohen-Or, and K. Aberman (2023) P+: extended textual conditioning in text-to-image generation. arXiv preprint arXiv:2303.09522. Cited by: §II.

[^37]: M.A. Woodbury and P. University. D. of Statistics (1950) Inverting modified matrices. Memorandum Report / Statistical Research Group, Princeton, Department of Statistics, Princeton University. External Links: [Link](https://books.google.co.jp/books?id=_zAnzgEACAAJ) Cited by: §IV-C.

[^38]: Y. Xian, B. Schiele, and Z. Akata (2017) Zero-shot learning-the good, the bad and the ugly. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4582–4591. Cited by: §V-C.

[^39]: G. Xiao, T. Yin, W. T. Freeman, F. Durand, and S. Han (2025) Fastcomposer: tuning-free multi-subject image generation with localized attention. International Journal of Computer Vision 133 (3), pp. 1175–1194. Cited by: §II.

[^40]: S. Xie, Z. Zhang, Z. Lin, T. Hinz, and K. Zhang (2023) Smartbrush: text and shape guided object inpainting with diffusion model. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22428–22437. Cited by: §II.

[^41]: Y. Yang, W. Wang, L. Peng, C. Song, Y. Chen, H. Li, X. Yang, Q. Lu, D. Cai, B. Wu, et al. (2024) Lora-composer: leveraging low-rank adaptation for multi-concept customization in training-free diffusion models. arXiv preprint arXiv:2403.11627. Cited by: §II.

[^42]: A. Zhang, X. Ding, H. Wang, S. McDonagh, and S. Kaski (2025) Rethinking inter-lora orthogonality in adapter merging: insights from orthogonal monte carlo dropout. arXiv preprint arXiv:2510.03262. Cited by: §II.

[^43]: H. Zhang, F. Li, S. Liu, L. Zhang, H. Su, J. Zhu, L. Ni, and H. Shum (2023) DINO: detr with improved denoising anchor boxes for end-to-end object detection. In The Eleventh International Conference on Learning Representations, Cited by: §V-C.

[^44]: L. Zhang, L. Zhang, S. Shi, X. Chu, and B. Li (2023) Lora-fa: memory-efficient low-rank adaptation for large language models fine-tuning. arXiv preprint arXiv:2308.03303. Cited by: §II.

[^45]: Y. Zhang, N. Huang, F. Tang, H. Huang, C. Ma, W. Dong, and C. Xu (2023) Inversion-based style transfer with diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10146–10156. Cited by: §II.

[^46]: M. Zhong, Y. Shen, S. Wang, Y. Lu, Y. Jiao, S. Ouyang, D. Yu, J. Han, and W. Chen (2024) Multi-lora composition for image generation. Transactions on Machine Learning Research 2024. Cited by: §II.

[^47]: H. Zhou, X. Lu, W. Xu, C. Zhu, T. Zhao, and M. Yang (2025) Lora-drop: efficient lora parameter pruning based on output evaluation. In Proceedings of the 31st International Conference on Computational Linguistics, pp. 5530–5543. Cited by: §II.

[^48]: B. Zi, X. Qi, L. Wang, J. Wang, K. Wong, and L. Zhang (2023) Delta-lora: fine-tuning high-rank parameters with the delta of low-rank matrices. arXiv preprint arXiv:2309.02411. Cited by: §II.