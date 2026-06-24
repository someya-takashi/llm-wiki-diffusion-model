---
title: "An Introduction to Flow Matching and Diffusion Models"
source: "https://ar5iv.labs.arxiv.org/html/2506.02070"
author:
published:
created: 2026-06-24
description:
tags:
  - "clippings"
---
(minitoc(hints)) Package minitoc(hints) Warning: W0099— The titlesec package is loaded. It is incompatible with the minitoc package

MIT Class 6.S184: Generative AI With Stochastic Differential Equations, 2025

An Introduction to Flow Matching and Diffusion Models

Peter Holderrieth and Ezra Erives  
Website: [https://diffusion.csail.mit.edu/](https://diffusion.csail.mit.edu/)

(minitoc) Package minitoc Warning: W0013No file main.toc PARTTOCS NOT PREPARED

toc

> Creating noise from data is easy; creating data from noise is generative modeling.
> 
> Song et al. \[[30](#bib.bibx30)\]
## 1 Introduction
### 1.1 Overview

In recent years, we all have witnessed a tremendous revolution in artificial intelligence (AI). Image generators like Stable Diffusion 3 can generate photorealistic and artistic images across a diverse range of styles, video models like Meta’s Movie Gen Video can generate highly realistic movie clips, and large language models like ChatGPT can generate seemingly human-level responses to text prompts. At the heart of this revolution lies a new ability of AI systems: the ability to generate objects. While previous generations of AI systems were mainly used for prediction, these new AI system are creative: they dream or come up with new objects based on user-specified input. Such generative AI systems are at the core of this recent AI revolution.

The goal of this class is to teach you two of the most widely used generative AI algorithms: denoising diffusion models \[[32](#bib.bibx32)\] and flow matching \[[14](#bib.bibx14), [16](#bib.bibx16), [1](#bib.bibx1), [15](#bib.bibx15)\]. These models are the backbone of the best image, audio, and video generation models (e.g., Stable Diffusion 3 and Movie Gen Video), and have most recently became the state-of-the-art in scientific applications such as protein structures (e.g., AlphaFold3 is a diffusion model). Without a doubt, understanding these models is truly an extremely useful skill to have.

All of these generative models generate objects by iteratively converting noise into data. This evolution from noise to data is facilitated by the simulation of ordinary or stochastic differential equations (ODEs/SDEs). Flow matching and denoising diffusion models are a family of techniques that allow us to construct, train, and simulate, such ODEs/SDEs at large scale with deep neural networks. While these models are rather simple to implement, the technical nature of SDEs can make these models difficult to understand. In this course, our goal is to provide a self-contained introduction to the necessary mathematical toolbox regarding differential equations to enable you to systematically understand these models. Beyond being widely applicable, we believe that the theory behind flow and diffusion models is elegant in its own right. Therefore, most importantly, we hope that this course will be a lot of fun to you.

###### Remark 1 (Additional Resources)

While these lecture notes are self-contained, there are two additional resources that we encourage you to use:

1. Lecture recordings: These guide you through each section in a lecture format.
2. Labs: These guide you in implementing your own diffusion model from scratch. We highly recommend that you “get your hands dirty” and code.

You can find these on our course website: [https://diffusion.csail.mit.edu/](https://diffusion.csail.mit.edu/).

### 1.2 Course Structure

We give a brief overview over of this document.

##### Required background.

Due to the technical nature of this subject, we recommend some base level of mathematical maturity, and in particular some familiarity with probability theory. For this reason, we included a brief reminder section on probability theory in appendix˜A. Don’t worry if some of the concepts there are unfamiliar to you.

### 1.3 Generative Modeling As Sampling

Let’s begin by thinking about various data types, or modalities, that we might encounter, and how we will go about representing them numerically:

1. Image: Consider images with $H\times W$ pixels where $H$ describes the height and $W$ the width of the image, each with three color channels (RGB). For every pixel and every color channel, we are given an intensity value in $\mathbb{R}$. Therefore, an image can be represented by an element $z\in\mathbb{R}^{H\times W\times 3}$.
2. Video: A video is simply a series of images in time. If we have $T$ time points or frames, a video would therefore be represented by an element $z\in\mathbb{R}^{T\times H\times W\times 3}$.
3. Molecular structure: A naive way would be to represent the structure of a molecule by a matrix  
	$z=(z^{1},\dots,z^{N})\in\mathbb{R}^{3\times N}$ where $N$ is the number of atoms in the molecule and each $z^{i}\in\mathbb{R}^{3}$ describes the location of that atom. Of course, there are other, more sophisticated ways of representing such a molecule.

In all of the above examples, the object that we want to generate can be mathematically represented as a vector (potentially after flattening). Therefore, throughout this document, we will have:

###### Key Idea 1 (Objects as Vectors)

We identify the objects being generated as vectors $z\in\mathbb{R}^{d}$.

A notable exception to the above is text data, which is typically modeled as a discrete object via autoregressive language models (such as *ChatGPT*). While flow and diffusion models for discrete data have been developed, this course focuses exclusively on applications to continuous data.

##### Generation as Sampling.

Let us define what it means to “generate” something. For example, let’s say we want to generate an image of a dog. Naturally, there are *many* possible images of dogs that we would be happy with. In particular, there is no one single “best” image of a dog. Rather, there is a spectrum of images that fit better or worse. In machine learning, it is common to think of this diversity of possible images as a *probability distribution*. We call it the data distribution and denote it as $p_{\rm{data}}$. In the example of dog images, this distribution would therefore give higher likelihood to images that look more like a dog. Therefore, how "good" an image/video/molecule fits - a rather subjective statement - is replaced by how "likely" it is under the data distribution $p_{\rm{data}}$. With this, we can mathematically express the task of generation as sampling from the (unknown) distribution $p_{\rm{data}}$:

###### Key Idea 2 (Generation as Sampling)

Generating an object $z$ is modeled as sampling from the data distribution $z\sim p_{\rm{data}}$.

A generative model is a machine learning model that allows us to generate samples from $p_{\rm{data}}$. In machine learning, we require data to train models. In generative modeling, we usually assume access to a finite number of examples sampled independently from $p_{\rm{data}}$, which together serve as a proxy for the true distribution.

###### Key Idea 3 (Dataset)

A dataset consists of a finite number of samples $z_{1},\dots,z_{N}\sim p_{\rm{data}}$.

For images, we might construct a dataset by compiling publicly available images from the internet. For videos, we might similarly use YouTube as a database. For protein structures, we can use experimental data bases from sources such as the Protein Data Bank (PDB) that collected scientific measurements over decades. As the size of our dataset grows very large, it becomes an increasingly better representation of the underlying distribution $p_{\rm{data}}$.

##### Conditional Generation.

In many cases, we want to generate an object conditioned on some data $y$. For example, we might want to generate an image conditioned on $y=$ “a dog running down a hill covered with snow with mountains in the background”. We can rephrase this as sampling from a conditional distribution:

###### Key Idea 4 (Conditional Generation)

Conditional generation involves sampling from $z\sim p_{\rm{data}}(\cdot|y)$, where $y$ is a conditioning variable.

We call $p_{\rm{data}}(\cdot|y)$ the conditional data distribution. The conditional generative modeling task typically involves learning to condition on an arbitrary, rather than fixed, choice of $y$. Using our previous example, we might alternatively want to condition on a different text prompt, such as $y=$ “a photorealistic image of a cat blowing out birthday candles”. We therefore seek a single model which may be conditioned on any such choice of $y$. It turns out that techniques for unconditional generation are readily generalized to the conditional case. Therefore, for the first 3 sections, we will focus almost exclusively on the unconditional case (keeping in mind that conditional generation is what we’re building towards).

##### From Noise to Data.

So far, we have discussed the what of generative modeling: generating samples from $p_{\rm{data}}$. Here, we will briefly discuss the how. For this, we assume that we have access to some initial distribution $p_{\rm{init}}$ that we can easily sample from, such as the Gaussian $p_{\rm{init}}=\mathcal{N}(0,I_{d})$. The goal of generative modeling is then to transform samples from $x\sim p_{\rm{init}}$ into samples from $p_{\rm{data}}$. We note that $p_{\rm{init}}$ does not have to be so simple as a Gaussian. As we shall see, there are interesting use cases for leveraging this flexibility. Despite this, in the majority of applications we take it to be a simple Gaussian and it is important to keep that in mind.

##### Summary

We summarize our discussion so far as follows.

###### Summary 2 (Generation as Sampling)

We summarize the findings of this section:

1. In this class, we consider the task of generating objects that are represented as vectors $z\in\mathbb{R}^{d}$ such as images, videos, or molecular structures.
2. Generation is the task of generating samples from a probability distribution $p_{\rm{data}}$ having access to a dataset of samples $z_{1},\dots,z_{N}\sim p_{\rm{data}}$ during training.
3. Conditional generation assumes that we condition the distribution on a label $y$ and we want to sample from $p_{\rm{data}}(\cdot|y)$ having access to data set of pairs $(z_{1},y)\dots,(z_{N},y)$ during training.
4. Our goal is to train a generative model to transform samples from a simple distribution $p_{\rm{init}}$ (e.g. a Gaussian) into samples from $p_{\rm{data}}$.

## 2 Flow and Diffusion Models

In the previous section, we formalized generative modeling as sampling from a data distribution $p_{\rm{data}}$. Further, we saw that sampling could be achieved via the transformation of samples from a simple distribution $p_{\rm{init}}$, such as the Gaussian $\mathcal{N}(0,I_{d})$, to samples from the target distribution $p_{\rm{data}}$. In this section, we describe how the desired transformation can be obtained as the simulation of a suitably constructed differential equation. For example, flow matching and diffusion models involve simulating ordinary differential equations (ODEs) and stochastic differential equations (SDEs), respectively. The goal of this section is therefore to define and construct these generative models as they will be used throughout the remainder of the notes. Specifically, we first define ODEs and SDEs, and discuss their simulation. Second, we describe how to parameterize an ODE/SDE using a deep neural network. This leads to the definition of a flow and diffusion model and the fundamental algorithms to sample from such models. In later sections, we then explore how to train these models.

### 2.1 Flow Models

We start by defining ordinary differential equations (ODEs). A solution to an ODE is defined by a trajectory, i.e. a function of the form

$$
\displaystyle X:[0,1]\to\mathbb{R}^{d},\quad t\mapsto X_{t},
$$

that maps from time $t$ to some location in space $\mathbb{R}^{d}$. Every ODE is defined by a vector field $u$, i.e. a function of the form

$$
\displaystyle u:\mathbb{R}^{d}\times[0,1]\to\mathbb{R}^{d},\quad(x,t)\mapsto u_{t}(x),
$$

i.e. for every time $t$ and location $x$ we get a vector $u_{t}(x)\in\mathbb{R}^{d}$ specifying a velocity in space (see fig.˜1). An ODE imposes a condition on a trajectory: we want a trajectory $X$ that “follows along the lines” of the vector field $u_{t}$, starting at the point $x_{0}$. We may formalize such a trajectory as being the solution to the equation:

$$
\displaystyle\frac{\mathrm{d}}{\mathrm{d}t}X_{t}
$$
 
$$
\displaystyle=u_{t}(X_{t})
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{ODE}
$$
 
$$
\displaystyle X_{0}
$$
 
$$
\displaystyle=x_{0}
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{initial conditions}
$$

Equation˜1a requires that the derivative of $X_{t}$ is specified by the direction given by $u_{t}$. Equation˜1b requires that we start at $x_{0}$ at time $t=0$. We may now ask: if we start at $X_{0}=x_{0}$ at $t=0$, where are we at time $t$ (what is $X_{t}$)? This question is answered by a function called the flow, which is a solution to the ODE

$$
\displaystyle\psi:\mathbb{R}^{d}\times[0,1]\mapsto
$$
 
$$
\displaystyle\mathbb{R}^{d},\quad(x_{0},t)\mapsto\psi_{t}(x_{0})
$$
 
$$
\displaystyle\frac{\mathrm{d}}{\mathrm{d}t}\psi_{t}(x_{0})
$$
 
$$
\displaystyle=u_{t}(\psi_{t}(x_{0}))
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{flow ODE}
$$
 
$$
\displaystyle\psi_{0}(x_{0})
$$
 
$$
\displaystyle=x_{0}
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{flow initial conditions}
$$

For a given initial condition $X_{0}=x_{0}$, a trajectory of the ODE is recovered via $X_{t}=\psi_{t}(X_{0})$. Therefore, vector fields, ODEs, and flows are, intuitively, three descriptions of the same object: vector fields define ODEs whose solutions are flows. As with every equation, we should ask ourselves about an ODE: Does a solution exist and if so, is it unique? A fundamental result in mathematics is "yes!" to both, as long we impose weak assumptions on $u_{t}$:

###### Theorem 3 (Flow existence and uniqueness)

If $u:\mathbb{R}^{d}\times[0,1]\to\mathbb{R}^{d}$ is continuously differentiable with a bounded derivative, then the ODE in 2 has a unique solution given by a flow $\psi_{t}$. In this case, $\psi_{t}$ is a diffeomorphism for all $t$, i.e. $\psi_{t}$ is continuously differentiable with a continuously differentiable inverse $\psi_{t}^{-1}$.

Note that the assumptions required for the existence and uniqueness of a flow are almost always fulfilled in machine learning, as we use neural networks to parameterize $u_{t}(x)$ and they always have bounded derivatives. Therefore, theorem˜3 should not be a concern for you but rather good news: flows exist and are unique solutions to ODEs in our cases of interest. A proof can be found in \[[20](#bib.bibx20), [4](#bib.bibx4)\].

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.02070/assets/fm_guide_assets/flow_1.png)

Figure 1: A flow ψ t ℝ d → \\psi\_{t}:\\mathbb{R}^{d}\\rightarrow\\mathbb{R}^{d} (red square grid) is defined by a velocity field u u\_{t}:\\mathbb{R}^{d}\\rightarrow\\mathbb{R}^{d} (visualized with blue arrows) that prescribes its instantaneous movements at all locations (here, = 2 d=2 ). We show three different times. As one can see, a flow is a diffeomorphism that "warps" space. Figure from \[ 15 \].

###### Example 4 (Linear Vector Fields)

Let us consider a simple example of a vector field $u_{t}(x)$ that is a simple linear function in $x$, i.e. $u_{t}(x)=-\theta x$ for $\theta>0$. Then the function

$$
\displaystyle\psi_{t}(x_{0})=\exp\left(-\theta t\right)x_{0}
$$

defines a flow $\psi$ solving the ODE in eq.˜2. You can check this yourself by checking that $\psi_{0}(x_{0})=x_{0}$ and computing

$$
\displaystyle\frac{\mathrm{d}}{\mathrm{d}t}\psi_{t}(x_{0})\overset{\lx@cref{refnum}{e:flow_linear_vf}}{=}\frac{\mathrm{d}}{\mathrm{d}t}\left(\exp\left(-\theta t\right)x_{0}\right)\overset{(i)}{=}-\theta\exp\left(-\theta t\right)x_{0}\overset{\lx@cref{refnum}{e:flow_linear_vf}}{=}-\theta\psi_{t}(x_{0})=u_{t}(\psi_{t}(x_{0})),
$$

where in (i) we used the chain rule. In fig.˜3, we visualize a flow of this form converging to $0$ exponentially.

##### Simulating an ODE.

In general, it is not possible to compute the flow $\psi_{t}$ explicitly if $u_{t}$ is not as simple as a linear function. In these cases, one uses numerical methods to simulate ODEs. Fortunately, this is a classical and well researched topic in numerical analysis, and a myriad of powerful methods exist \[[11](#bib.bibx11)\]. One of the simplest and most intuitive methods is the Euler method. In the Euler method, we initialize with $X_{0}=x_{0}$ and update via

$$
X_{t+h}=X_{t}+hu_{t}(X_{t})\quad(t=0,h,2h,3h,\dots,1-h)
$$

where $h=n^{-1}>0$ is a step size hyperparameter with $n\in\mathbb{N}$. For this class, the Euler method will be good enough. To give you a taste of a more complex method, let us consider Heun’s method defined via the update rule

$$
\displaystyle X_{t+h}^{\prime}
$$
 
$$
\displaystyle=X_{t}+hu_{t}(X_{t})\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{initial guess of new state}
$$
 
$$
\displaystyle X_{t+h}
$$
 
$$
\displaystyle=X_{t}+\frac{h}{2}(u_{t}(X_{t})+u_{t+h}(X_{t+h}^{\prime}))\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{update with average $u$ at current and guessed state}
$$

Intuitively, the Heun’s method is as follows: it takes a first guess $X_{t+h}^{\prime}$ of what the next step could be but corrects the direction initially taken via an updated guess.

##### Flow models.

We can now construct a generative model via an ODE. Remember that our goal was to convert a simple distribution $p_{\rm{init}}$ into a complex distribution $p_{\rm{data}}$. The simulation of an ODE is thus a natural choice for this transformation. A flow model is described by the ODE

$$
\displaystyle X_{0}
$$
 
$$
\displaystyle\sim p_{\rm{init}}
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{random initialization}
$$
 
$$
\displaystyle\frac{\mathrm{d}}{\mathrm{d}t}X_{t}
$$
 
$$
\displaystyle=u_{t}^{\theta}(X_{t})
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{ODE}
$$

where the vector field $u_{t}^{\theta}$ is a neural network $u_{t}^{\theta}$ with parameters $\theta$. For now, we will speak of $u_{t}^{\theta}$ as being a generic neural network; i.e. a continuous function $u_{t}^{\theta}:\mathbb{R}^{d}\times[0,1]\to\mathbb{R}^{d}$ with parameters $\theta$. Later, we will discuss particular choices of neural network architectures. Our goal is to make the endpoint $X_{1}$ of the trajectory have distribution $p_{\rm{data}}$, i.e.

$$
\displaystyle X_{1}\sim p_{\rm{data}}\quad\Leftrightarrow\quad\psi_{1}^{\theta}(X_{0})\sim p_{\rm{data}}
$$

where $\psi_{t}^{\theta}$ describes the flow induced by $u_{t}^{\theta}$. Note however: although it is called flow model, the neural network parameterizes the vector field, not the flow. In order to compute the flow, we need to simulate the ODE. In algorithm˜1, we summarize the procedure how to sample from a flow model.

Algorithm 1 Sampling from a Flow Model with Euler method

 Neural network vector field $u_{t}^{\theta}$, number of steps $n$

 Set $t=0$

 Set step size $h=\frac{1}{n}$

 Draw a sample $X_{0}\sim p_{\rm{init}}$

 for $i=1,\dots,n-1$ do

   $X_{t+h}=X_{t}+hu_{t}^{\theta}(X_{t})$

  Update $t\leftarrow t+h$

 end for

 return $X_{1}$

### 2.2 Diffusion Models

Stochastic differential equations (SDEs) extend the deterministic trajectories from ODEs with stochastic trajectories. A stochastic trajectory is commonly called a stochastic process $(X_{t})_{0\leq t\leq 1}$ and is given by

$$
\displaystyle X_{t}\text{ is a random variable for every }0\leq t\leq 1
$$
 
$$
\displaystyle X:[0,1]\to\mathbb{R}^{d},\quad t\mapsto X_{t}\text{ is a random trajectory for every draw of }X
$$

In particular, when we simulate the same stochastic process twice, we might get different outcomes because the dynamics are designed to be random.

##### Brownian Motion.

SDEs are constructed via a Brownian motion - a fundamental stochastic process that came out of the study physical diffusion processes. You can think of a Brownian motion as a continuous random walk.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.02070/assets/figures/brownian_motion_sample_paths.png)

Figure 2: Sample trajectories of a Brownian motion W t W\_{t} in dimension d = 1 d=1 simulated using eq. ˜ 5.

Let us define it: A Brownian motion $W=(W_{t})_{0\leq t\leq 1}$ is a stochastic process such that $W_{0}=0$, the trajectories $t\mapsto W_{t}$ are continuous, and the following two conditions hold:

1. Normal increments: $W_{t}-W_{s}\sim\mathcal{N}(0,(t-s)I_{d})$ for all $0\leq s<t$, i.e. increments have a Gaussian distribution with variance increasing linearly in time ($I_{d}$ is the identity matrix).
2. Independent increments: For any $0\leq t_{0}<t_{1}<\dots<t_{n}=1$, the increments $W_{t_{1}}-W_{t_{0}},\dots,W_{t_{n}}-W_{t_{n-1}}$ are independent random variables.

Brownian motion is also called a Wiener process, which is why we denote it with a " $W$ ".[^1] We can easily simulate a Brownian motion approximately with step size $h>0$ by setting $W_{0}=0$ and updating

$$
\displaystyle W_{t+h}=
$$
 
$$
\displaystyle W_{t}+\sqrt{h}\epsilon_{t},\quad\epsilon_{t}\sim\mathcal{N}(0,I_{d})\quad(t=0,h,2h,\dots,1-h)
$$

In fig.˜2, we plot a few example trajectories of a Brownian motion. Brownian motion is as central to the study of stochastic processes as the Gaussian distribution is to the study of probability distributions. From finance to statistical physics to epidemiology, the study of Brownian motion has far reaching applications beyond machine learning. In finance, for example, Brownian motion is used to model the price of complex financial instruments. Also just as a mathematical construction, Brownian motion is fascinating: For example, while the paths of a Brownian motion are continuous (so that you could draw it without ever lifting a pen), they are infinitely long (so that you would never stop drawing).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.02070/assets/figures/ou_process.png)

Figure 3: Illustration of Ornstein-Uhlenbeck processes ( eq. ˜ 8 ) in dimension d = 1 d=1 for θ 0.25 \\theta=0.25 and various choices of σ \\sigma (increasing from left to right). For 0 \\sigma=0, we recover a flow (smooth, deterministic trajectories) that converges to the origin as t → ∞ t\\to\\infty. For > \\sigma>0 we have random paths which converge towards the Gaussian 𝒩 (, 2 ) \\mathcal{N}(0,\\frac{\\sigma^{2}}{2\\theta}) as.

##### From ODEs to SDEs.

The idea of an SDE is to extend the deterministic dynamics of an ODE by adding stochastic dynamics driven by a Brownian motion. Because everything is stochastic, we may no longer take the derivative as in eq.˜1a. Hence, we need to find an equivalent formulation of ODEs that does not use derivatives. For this, let us therefore rewrite trajectories $(X_{t})_{0\leq t\leq 1}$ of an ODE as follows:

$$
\displaystyle\frac{\mathrm{d}}{\mathrm{d}t}X_{t}
$$
 
$$
\displaystyle=u_{t}(X_{t})\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{expression via derivatives}
$$
 
$$
\displaystyle\overset{(i)}{\Leftrightarrow}\quad\frac{1}{h}\left(X_{t+h}-X_{t}\right)
$$
 
$$
\displaystyle=u_{t}(X_{t})+R_{t}(h)
$$
 
$$
\displaystyle\Leftrightarrow\quad X_{t+h}
$$
 
$$
\displaystyle=X_{t}+hu_{t}(X_{t})+hR_{t}(h)\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{expression via infinitesimal updates}
$$

where $R_{t}(h)$ describes a negligible function for small $h$, i.e. such that $\lim\limits_{h\to 0}R_{t}(h)=0$, and in $(i)$ we simply use the definition of derivatives. The derivation above simply restates what we already know: A trajectory $(X_{t})_{0\leq t\leq 1}$ of an ODE takes, at every timestep, a small step in the direction $u_{t}(X_{t})$. We may now amend the last equation to make it stochastic: A trajectory $(X_{t})_{0\leq t\leq 1}$ of an SDE takes, at every timestep, a small step in the direction $u_{t}(X_{t})$ plus some contribution from a Brownian motion:

$$
\displaystyle X_{t+h}=X_{t}+\underbrace{hu_{t}(X_{t})}_{\text{deterministic}}+\sigma_{t}\underbrace{(W_{t+h}-W_{t})}_{\text{stochastic}}+\underbrace{hR_{t}(h)}_{\text{error term}}
$$

where $\sigma_{t}\geq 0$ describes the diffusion coefficient and $R_{t}(h)$ describes a stochastic error term such that the standard deviation $\mathbb{E}[\|R_{t}(h)\|^{2}]^{1/2}\to 0$ goes to zero for $h\to 0$. The above describes a stochastic differential equation (SDE). It is common to denote it in the following symbolic notation:

$$
\displaystyle\mathrm{d}X_{t}
$$
 
$$
\displaystyle=u_{t}(X_{t})\mathrm{d}t+\sigma_{t}\mathrm{d}W_{t}
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{SDE}
$$
 
$$
\displaystyle X_{0}
$$
 
$$
\displaystyle=x_{0}
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{initial condition}
$$

However, always keep in mind that the " $\mathrm{d}X_{t}$ "-notation above is a purely informal notation of eq.˜6. Unfortunately, SDEs do not have a flow map $\phi_{t}$ anymore. This is because the value $X_{t}$ is not fully determined by $X_{0}\sim p_{\rm{init}}$ anymore as the evolution itself is stochastic. Still, in the same way as for ODEs, we have:

###### Theorem 5 (SDE Solution Existence and Uniqueness)

If $u:\mathbb{R}^{d}\times[0,1]\to\mathbb{R}^{d}$ is continuously differentiable with a bounded derivative and $\sigma_{t}$ is continuous, then the SDE in 7 has a solution given by the unique stochastic process $(X_{t})_{0\leq t\leq 1}$ satisfying eq.˜6.

If this was a stochastic calculus class, we would spend several lectures proving this theorem and constructing SDEs with full mathematical rigor, i.e. constructing a Brownian motion from first principles and constructing the process $X_{t}$ via stochastic integration. As we focus on machine learning in this class, we refer to \[[18](#bib.bibx18)\] for a more technical treatment. Finally, note that every ODE is also an SDE - simply with a vanishing diffusion coefficient $\sigma_{t}=0$. Therefore, for the remainder of this class, when we speak about SDEs, we consider ODEs as a special case.

###### Example 6 (Ornstein-Uhlenbeck Process)

Let us consider a constant diffusion coefficient $\sigma_{t}=\sigma\geq 0$ and a constant linear drift $u_{t}(x)=-\theta x$ for $\theta>0$, yielding the SDE

$$
\displaystyle\mathrm{d}X_{t}=-\theta X_{t}\mathrm{d}t+\sigma dW_{t}.
$$

A solution $(X_{t})_{0\leq t\leq 1}$ to the above SDE is known as an Ornstein-Uhlenbeck (OU) process. We visualize it in fig.˜3. The vector field $-\theta x$ pushes the process back to its center $0$ (as I always go the inverse direction of where I am), while the diffusion coefficient $\sigma$ always adds more noise. This process converges towards a Gaussian distribution $\mathcal{N}(0,\sigma^{2})$ if we simulate it for $t\to\infty$. Note that for $\sigma=0$, we have a flow with linear vector field that we have studied in eq.˜3.

##### Simulating an SDE.

If you struggle with the abstract definition of an SDE so far, then don’t worry about it. A more intuitive way of thinking about SDEs is given by answering the question: How might we simulate an SDE? The simplest such scheme is known as the Euler-Maruyama method, and is essentially to SDEs what the Euler method is to ODEs. Using the Euler-Maruyama method, we initialize $X_{0}=x_{0}$ and update iteratively via

$$
\displaystyle X_{t+h}=X_{t}+hu_{t}(X_{t})+\sqrt{h}\sigma_{t}\epsilon_{t},\quad\quad\epsilon_{t}\sim\mathcal{N}(0,I_{d})
$$

where $h=n^{-1}>0$ is a step size hyperparameter for $n\in\mathbb{N}$. In other words, to simulate using the Euler-Maruyama method, we take a small step in the direction of $u_{t}(X_{t})$ as well as add a little bit of Gaussian noise scaled by $\sqrt{h}\sigma_{t}$. When simulating SDEs in this class (such as in the accompanying labs), we will usually stick to the Euler-Maruyama method.

Algorithm 2 Sampling from a Diffusion Model (Euler-Maruyama method)

 Neural network $u_{t}^{\theta}$, number of steps $n$, diffusion coefficient $\sigma_{t}$

 Set $t=0$

 Set step size $h=\frac{1}{n}$

 Draw a sample $X_{0}\sim p_{\rm{init}}$

 for $i=1,\dots,n-1$ do

  Draw a sample $\epsilon\sim\mathcal{N}(0,I_{d})$

   $X_{t+h}=X_{t}+hu_{t}^{\theta}(X_{t})+\sigma_{t}\sqrt{h}\epsilon$

  Update $t\leftarrow t+h$

 end for

 return $X_{1}$

##### Diffusion Models.

We can now construct a generative model via an SDE in the same way as we did for ODEs. Remember that our goal was to convert a simple distribution $p_{\rm{init}}$ into a complex distribution $p_{\rm{data}}$. Like for ODEs, the simulation of an SDE randomly initialized with $X_{0}\sim p_{\rm{init}}$ is a natural choice for this transformation. To parameterize this SDE, we can simply parameterize its central ingredient - the vector field $u_{t}$ - a neural network $u_{t}^{\theta}$. A diffusion model is thus given by

$$
\displaystyle\mathrm{d}X_{t}
$$
 
$$
\displaystyle=u_{t}^{\theta}(X_{t})\mathrm{d}t+\sigma_{t}\mathrm{d}W_{t}
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{SDE}
$$
 
$$
\displaystyle X_{0}
$$
 
$$
\displaystyle\sim p_{\rm{init}}
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{random initialization}
$$

In algorithm˜2, we describe the procedure by which to sample from a diffusion model with the Euler-Maruyama method. We summarize the results of this section as follows.

###### Summary 7 (SDE generative model)

Throughout this document, a diffusion model consists of a neural network $u_{t}^{\theta}$ with parameters $\theta$ that parameterize a vector field and a fixed diffusion coefficient $\sigma_{t}$:

$$
\displaystyle u^{\theta}:\mathbb{R}^{d}\times[0,1]\to\mathbb{R}^{d},\,\,(x,t)\mapsto u_{t}^{\theta}(x)\text{ with parameters }\theta
$$
 
$$
\displaystyle\sigma_{t}:[0,1]\to[0,\infty),\,\,t\mapsto\sigma_{t}
$$

To obtain samples from our SDE model (i.e. generate objects), the procedure is as follows:

$$
\displaystyle\textbf{Initialization:}\quad X_{0}
$$
 
$$
\displaystyle\sim p_{\rm{init}}\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Initialize with simple distribution, e.g. a Gaussian}
$$
 
$$
\displaystyle\textbf{Simulation:}\quad\mathrm{d}X_{t}
$$
 
$$
\displaystyle=u_{t}^{\theta}(X_{t})\mathrm{d}t+\sigma_{t}\mathrm{d}W_{t}\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Simulate SDE from 0 to 1}
$$
 
$$
\displaystyle\textbf{Goal:}\quad X_{1}
$$
 
$$
\displaystyle\sim p_{\rm{data}}\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Goal is to make $X_{1}$ have distribution $p_{\rm{data}}$}
$$

A diffusion model with $\sigma_{t}=0$ is a flow model.

## 3 Constructing the Training Target

In the previous section, we constructed flow and diffusion models where we obtain trajectories $(X_{t})_{0\leq t\leq 1}$ by simulating the ODE/SDE

$$
\displaystyle X_{0}\sim
$$
 
$$
\displaystyle p_{\rm{init}},\quad\mathrm{d}X_{t}=u_{t}^{\theta}(X_{t})\mathrm{d}t
$$
 
$$
\displaystyle X_{0}\sim
$$
 
$$
\displaystyle p_{\rm{init}},\quad\mathrm{d}X_{t}=u_{t}^{\theta}(X_{t})\mathrm{d}t+\sigma_{t}\mathrm{d}W_{t}
$$

where $u_{t}^{\theta}$ is a neural network and $\sigma_{t}$ is a fixed diffusion coefficient. Naturally, if we just randomly initialize the parameters $\theta$ of our neural network $u_{t}^{\theta}$, simulating the ODE/SDE will just produce nonsense. As always in machine learning, we need to train the neural network. We accomplish this by minimizing a loss function $\mathcal{L}(\theta)$, such as the mean-squared error

$$
\displaystyle\mathcal{L}(\theta)=\lVert u_{t}^{\theta}(x)-\underbrace{u^{\text{target}}_{t}(x)}_{\text{training target}}\rVert^{2},
$$

where $u^{\text{target}}_{t}(x)$ is the training target that we would like to approximate. To derive a training algorithm, we proceed in two steps: In this chapter, our goal is to find an equation for the training target $u^{\text{target}}_{t}$. In the next chapter, we will describe a training algorithm that approximates the training target $u^{\text{target}}_{t}$. Naturally, like the neural network $u_{t}^{\theta}$, the training target should itself be a vector field $u^{\text{target}}_{t}:\mathbb{R}^{d}\times[0,1]\to\mathbb{R}^{d}$. Further, $u^{\text{target}}_{t}$ should do what we want $u_{t}^{\theta}$ to do: convert noise into data. Therefore, the goal of this chapter is to derive a formula for the training target $u_{t}^{\text{ref}}$ such that the corresponding ODE/SDE converts $p_{\rm{init}}$ into $p_{\rm{data}}$. Along the way we will encounter two fundamental results from physics and stochastic calculus: the continuity equation and the Fokker-Planck equation. As before, we will first describe the key ideas for ODEs before generalizing them to SDEs.

###### Remark 8

There are a number of different approaches to deriving a training target for flow and diffusion models. The approach we present here is both the most general and arguably most simple and is in line with recent state-of-the-art models. However, it might well differ from other, older presentations of diffusion models you have seen. Later, we will discuss alternative formulations.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.02070/assets/figures/noised_mnist_reversed.png)

Figure 4: Gradual interpolation from noise to data via a Gaussian conditional probability path for a collection of images.

### 3.1 Conditional and Marginal Probability Path

The first step of constructing the training target $u^{\text{target}}_{t}$ is by specifying a probability path. Intuitively, a probability path specifies a gradual interpolation between noise $p_{\rm{init}}$ and data $p_{\rm{data}}$ (see fig.˜4). We explain the construction in this section. In the following, for a data point $z\in\mathbb{R}^{d}$, we denote with $\delta_{z}$ the Dirac delta “distribution”. This is the simplest distribution that one can imagine: sampling from $\delta_{z}$ always returns $z$ (i.e. it is deterministic). A conditional (interpolating) probability path is a set of distribution $p_{t}(x|z)$ over $\mathbb{R}^{d}$ such that:

$$
\displaystyle p_{0}(\cdot|z)=p_{\rm{init}},\quad p_{1}(\cdot|z)=\delta_{z}\quad\text{ for all }z\in\mathbb{R}^{d}.
$$

In other words, a conditional probability path gradually converts a single data point into the distribution $p_{\rm{init}}$ (see e.g. fig.˜4). You can think of a probability path as a trajectory in the space of distributions. Every conditional probability path $p_{t}(x|z)$ induces a marginal probability path $p_{t}(x)$ defined as the distribution that we obtain by first sampling a data point $z\sim p_{\rm{data}}$ from the data distribution and then sampling from $p_{t}(\cdot|z)$:

$$
\displaystyle z
$$
 
$$
\displaystyle\sim p_{\rm{data}},\quad x\sim p_{t}(\cdot|z)\quad\Rightarrow x\sim p_{t}
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{sampling from marginal path}
$$
 
$$
\displaystyle p_{t}(x)
$$
 
$$
\displaystyle=\int p_{t}(x|z)p_{\rm{data}}(z)\mathrm{d}z
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{density of marginal path}
$$

Note that we know how to sample from $p_{t}$ but we don’t know the density values $p_{t}(x)$ as the integral is intractable. Check for yourself that because of the conditions on $p_{t}(\cdot|z)$ in eq.˜12, the marginal probability path $p_{t}$ interpolates between $p_{\rm{init}}$ and $p_{\rm{data}}$:

$$
\displaystyle p_{0}=p_{\rm{init}}\quad\text{and}\quad p_{1}=p_{\rm{data}}.\quad\quad\quad\quad\blacktriangleright\,\,\text{noise-data interpolation}
$$
![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.02070/assets/figures/conditional_vs_marginal.png)

Figure 5: Illustration of a conditional (top) and marginal (bottom) probability path. Here, we plot a Gaussian probability path with α t =, β 1 − \\alpha\_{t}=t,\\beta\_{t}=1-t. The conditional probability path interpolates a Gaussian p init 𝒩 ( 0 I d ) p\_{\\rm{init}}=\\mathcal{N}(0,I\_{d}) and data δ z p\_{\\rm{data}}=\\delta\_{z} for single data point. The marginal probability path interpolates a Gaussian and a data distribution p\_{\\rm{data}} (Here, is a toy distribution in dimension 2 d=2 represented by a chess board pattern.)

###### Example 9 (Gaussian Conditional Probability Path)

One particularly popular probability path is the Gaussian probability path. This is the probability path used by denoising diffusion models. Let $\alpha_{t},\beta_{t}$ be noise schedulers: two continuously differentiable, monotonic functions with $\alpha_{0}=\beta_{1}=0$ and $\alpha_{1}=\beta_{0}=1$. We then define the conditional probability path

$$
\displaystyle p_{t}(\cdot|z)
$$
 
$$
\displaystyle=\mathcal{N}(\alpha_{t}z,\beta_{t}^{2}I_{d})
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Gaussian conditional path}
$$

which, by the conditions we imposed on $\alpha_{t}$ and $\beta_{t}$, fulfills

$$
\displaystyle p_{0}(\cdot|z)
$$
 
$$
\displaystyle=\mathcal{N}(\alpha_{0}z,\beta_{0}^{2}I_{d})=\mathcal{N}(0,I_{d}),\quad\text{and}\quad p_{1}(\cdot|z)=\mathcal{N}(\alpha_{1}z,\beta_{1}^{2}I_{d})=\delta_{z},
$$

where we have used the fact that a normal distribution with zero variance and mean $z$ is just $\delta_{z}$. Therefore, this choice of $p_{t}(x|z)$ fulfills eq.˜12 for $p_{\rm{init}}=\mathcal{N}(0,I_{d})$ and is therefore a valid conditional interpolating path. The Gaussian conditional probability path has several useful properties which makes it especially amenable to our goals, and because of this we will use it as our prototypical example of a conditional probability path for the rest of the section. In fig.˜4, we illustrate its application to an image. We can express sampling from the marginal path $p_{t}$ as:

$$
\displaystyle z\sim
$$
 
$$
\displaystyle p_{\rm{data}},\,\epsilon\sim p_{\rm{init}}=\mathcal{N}(0,I_{d})\,\Rightarrow\,x=\alpha_{t}z+\beta_{t}\epsilon\sim p_{t}\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{sampling from marginal Gaussian path}
$$

Intuitively, the above procedure adds more noise for lower $t$ until time $t=0$, at which point there is only noise. In fig.˜5, we plot an example of such an interpolating path between Gaussian noise and a simple data distribution.

### 3.2 Conditional and Marginal Vector Fields

We proceed now by constructing a training target $u^{\text{target}}_{t}$ for a flow model using the recently defined notion of a probability path $p_{t}$. The idea is to construct $u^{\text{target}}_{t}$ from simple components that we can derive analytically by hand.

###### Theorem 10 (Marginalization trick)

For every data point $z\in\mathbb{R}^{d}$, let $u^{\text{target}}_{t}(\cdot|z)$ denote a conditional vector field, defined so that the corresponding ODE yields the conditional probability path $p_{t}(\cdot|z)$, viz.,

$$
\displaystyle X_{0}
$$
 
$$
\displaystyle\sim p_{\rm{init}},\quad\frac{\mathrm{d}}{\mathrm{d}t}X_{t}=u^{\text{target}}_{t}(X_{t}|z)\quad\Rightarrow\quad X_{t}\sim p_{t}(\cdot|z)\quad(0\leq t\leq 1).
$$

Then the marginal vector field $u^{\text{target}}_{t}(x)$, defined by

$$
\displaystyle u^{\text{target}}_{t}(x)=\int u^{\text{target}}_{t}(x|z)\frac{p_{t}(x|z)p_{\rm{data}}(z)}{p_{t}(x)}\mathrm{d}z,
$$

follows the marginal probability path, i.e.

$$
\displaystyle X_{0}
$$
 
$$
\displaystyle\sim p_{\rm{init}},\quad\frac{\mathrm{d}}{\mathrm{d}t}X_{t}=u^{\text{target}}_{t}(X_{t})\quad\Rightarrow\quad X_{t}\sim p_{t}\quad(0\leq t\leq 1).
$$

In particular, $X_{1}\sim p_{\rm{data}}$ for this ODE, so that we might say " $u^{\text{target}}_{t}$ converts noise $p_{\rm{init}}$ into data $p_{\rm{data}}$ ".

See fig.˜6 for an illustration. Before we prove the marginalization trick, let us first explain why it is useful: The marginalization trick from theorem˜10 allows us to construct the marginal vector field from a conditional vector field. This simplifies the problem of finding a formula for a training target significantly as we can often find a conditional vector field $u^{\text{target}}_{t}(\cdot|z)$ satisfying eq.˜18 analytically by hand (i.e. by just doing some algebra ourselves). Let us illustrate this by deriving a conditional vector field $u_{t}(x|z)$ for our running example of a Gaussian probability path.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.02070/assets/figures/conditional_ode.png)

Figure 6: Illustration of theorem ˜ 10. Simulating a probability path with ODEs. Data distribution p data p\_{\\rm{data}} in blue background. Gaussian init p\_{\\rm{init}} in red background. Top row: Conditional probability path. Left: Ground truth samples from conditional path t ( ⋅ | z ) p\_{t}(\\cdot|z). Middle: ODE samples over time. Right: Trajectories by simulating ODE with u target x u^{\\text{target}}\_{t}(x|z) in eq. 21. Bottom row: Simulating a marginal probability path. Left: Ground truth samples from p\_{t}. Middle: ODE samples over time. Right: Trajectories by simulating ODE with marginal vector field flow u^{\\text{flow}}\_{t}(x). As one can see, the conditional vector field follows the conditional probability path and the marginal vector field follows the marginal probability path.

###### Example 11 (Target ODE for Gaussian probability paths)

As before, let $p_{t}(\cdot|z)=\mathcal{N}(\alpha_{t}z,\beta_{t}^{2}I_{d})$ for noise schedulers $\alpha_{t},\beta_{t}$ (see eq.˜16). Let $\dot{\alpha}_{t}=\partial_{t}\alpha_{t}$ and $\dot{\beta}_{t}=\partial_{t}\beta_{t}$ denote respective time derivatives of $\alpha_{t}$ and $\beta_{t}$. Here, we want to show that the conditional Gaussian vector field given by

$$
\displaystyle u^{\text{target}}_{t}(x|z)=\left(\dot{\alpha}_{t}-\frac{\dot{\beta}_{t}}{\beta_{t}}\alpha_{t}\right)z+\frac{\dot{\beta}_{t}}{\beta_{t}}x
$$

is a valid conditional vector field model in the sense of theorem˜10: its ODE trajectories $X_{t}$ satisfy $X_{t}\sim p_{t}(\cdot|z)=\mathcal{N}(\alpha_{t}z,\beta_{t}^{2}I_{d})$ if $X_{0}\sim\mathcal{N}(0,I_{d})$. In fig.˜6, we confirm this visually by comparing samples from the conditional probability path (ground truth) to samples from simulated ODE trajectories of this flow. As you can see, the distribution match. We will now prove this.

###### Proof.

Let us construct a conditional flow model $\psi^{\text{target}}_{t}(x|z)$ first by defining

$$
\displaystyle\psi^{\text{target}}_{t}(x|z)=\alpha_{t}z+\beta_{t}x.
$$

If $X_{t}$ is the ODE trajectory of $\psi^{\text{target}}_{t}(\cdot|z)$ with $X_{0}\sim p_{\rm{init}}=\mathcal{N}(0,I_{d})$, then by definition

$$
\displaystyle X_{t}=\psi^{\text{target}}_{t}(X_{0}|z)=\alpha_{t}z+\beta_{t}X_{0}\sim\mathcal{N}(\alpha_{t}z,\beta^{2}I_{d})=p_{t}(\cdot|z).
$$

We conclude that the trajectories are distributed like the conditional probability path (i.e, eq.˜18 is fulfilled). It remains to extract the vector field $u^{\text{target}}_{t}(x|z)$ from $\psi^{\text{target}}_{t}(x|z)$. By the definition of a flow (eq.˜2b), it holds

$$
\displaystyle\frac{\mathrm{d}}{\mathrm{d}t}\psi^{\text{target}}_{t}(x|z)
$$
 
$$
\displaystyle=u^{\text{target}}_{t}(\psi^{\text{target}}_{t}(x|z)|z)\quad\text{ for all }x,z\in\mathbb{R}^{d}
$$
 
$$
\displaystyle\overset{(i)}{\Leftrightarrow}\quad\dot{\alpha}_{t}z+\dot{\beta}_{t}x
$$
 
$$
\displaystyle=u^{\text{target}}_{t}(\alpha_{t}z+\beta_{t}x|z)\quad\text{ for all }x,z\in\mathbb{R}^{d}
$$
 
$$
\displaystyle\overset{(ii)}{\Leftrightarrow}\quad\dot{\alpha}_{t}z+\dot{\beta}_{t}\left(\frac{x-\alpha_{t}z}{\beta_{t}}\right)
$$
 
$$
\displaystyle=u^{\text{target}}_{t}(x|z)\quad\text{ for all }x,z\in\mathbb{R}^{d}
$$
 
$$
\displaystyle\overset{(iii)}{\Leftrightarrow}\quad\left(\dot{\alpha}_{t}-\frac{\dot{\beta}_{t}}{\beta_{t}}\alpha_{t}\right)z+\frac{\dot{\beta}_{t}}{\beta_{t}}x
$$
 
$$
\displaystyle=u^{\text{target}}_{t}(x|z)\quad\text{ for all }x,z\in\mathbb{R}^{d}
$$

where in $(i)$ we used the definition of $\psi^{\text{target}}_{t}(x|z)$ (eq.˜22), in $(ii)$ we reparameterized $x\rightarrow(x-\alpha_{t}z)/\beta_{t}$, and in $(iii)$ we just did some algebra. Note that the last equation is the conditional Gaussian vector field as we defined in eq.˜21. This proves the statement.[^2] ∎

The remainder of this section will now prove theorem˜10 via the continuity equation, a fundamental result in mathematics and physics. To explain it, we will require the use of the divergence operator $\mathrm{div}$, which we define as

$$
\displaystyle\mathrm{div}(v_{t})(x)=
$$
 
$$
\displaystyle\sum\limits_{i=1}^{d}\frac{\partial}{\partial x_{i}}v_{t}(x)
$$

###### Theorem 12 (Continuity Equation)

Let us consider an flow model with vector field $u^{\text{target}}_{t}$ with $X_{0}\sim p_{\rm{init}}$. Then $X_{t}\sim p_{t}$ for all $0\leq t\leq 1$ if and only if

$$
\displaystyle\partial_{t}p_{t}(x)=-\mathrm{div}(p_{t}u^{\text{target}}_{t})(x)\quad\text{ for all }x\in\mathbb{R}^{d},0\leq t\leq 1,
$$

where $\partial_{t}p_{t}(x)=\frac{\mathrm{d}}{\mathrm{d}t}p_{t}(x)$ denotes the time-derivative of $p_{t}(x)$. Equation 24 is known as the continuity equation.

For the mathematically-inclined reader, we present a self-contained proof of the Continuity Equation in appendix˜B. Before we move on, let us try and understand intuitively the continuity equation. The left-hand side $\partial_{t}p_{t}(x)$ describes how much the probability $p_{t}(x)$ at $x$ changes over time. Intuitively, the change should correspond to the net inflow of probability mass. For a flow model, a particle $X_{t}$ follows along the vector field $u^{\text{target}}_{t}$. As you might recall from physics, the divergence measures a sort of net outflow from the vector field. Therefore, the negative divergence measures the net inflow. Scaling this by the total probability mass currently residing at $x$, we get that the net $-\mathrm{div}(p_{t}u_{t})$ measures the total inflow of probability mass. Since probability mass is conserved, the left-hand and right-hand side of the equation should be the same! We now proceed with a proof of the marginalization trick from theorem˜10.

###### Proof.

By theorem˜12, we have to show that the marginal vector field $u^{\text{target}}_{t}$, as defined as in eq.˜19, satisfies the continuity equation. We can do this by direct calculation:

$$
\displaystyle\partial_{t}p_{t}(x)
$$
 
$$
\displaystyle\overset{(i)}{=}\partial_{t}\int p_{t}(x|z)p_{\rm{data}}(z)\mathrm{d}z
$$
 
$$
\displaystyle=\int\partial_{t}p_{t}(x|z)p_{\rm{data}}(z)\mathrm{d}z
$$
 
$$
\displaystyle\overset{(ii)}{=}\int-\mathrm{div}(p_{t}(\cdot|z)u^{\text{target}}_{t}(\cdot|z))(x)p_{\rm{data}}(z)\mathrm{d}z
$$
 
$$
\displaystyle\overset{(iii)}{=}-\mathrm{div}\left(\int p_{t}(x|z)u^{\text{target}}_{t}(x|z)p_{\rm{data}}(z)\mathrm{d}z\right)
$$
 
$$
\displaystyle\overset{(iv)}{=}-\mathrm{div}\left(p_{t}(x)\int u^{\text{target}}_{t}(x|z)\frac{p_{t}(x|z)p_{\rm{data}}(z)}{p_{t}(x)}\mathrm{d}z\right)(x)
$$
 
$$
\displaystyle\overset{(v)}{=}-\mathrm{div}\left(p_{t}u^{\text{target}}_{t}\right)(x),
$$

where in $(i)$ we used the definition of $p_{t}(x)$ in eq.˜13, in $(ii)$ we used the continuity equation for the conditional probability path $p_{t}(\cdot|z)$, in $(iii)$ we swapped the integral and divergence operator using eq.˜23, in $(iv)$ we multiplied and divided by $p_{t}(x)$, and in $(v)$ we used eq.˜19. The beginning and end of the above chain of equations show that the continuity equation is fulfilled for $u^{\text{target}}_{t}$. By theorem˜12, this is enough to imply eq.˜20, and we are done. ∎

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.02070/assets/figures/conditional_sde.png)

Figure 7: Illustration of theorem ˜ 13. Simulating a probability path with SDEs. This repeats the plots from fig. 6 with SDE sampling using eq. 25. Data distribution p data p\_{\\rm{data}} in blue background. Gaussian init p\_{\\rm{init}} in red background. Top row: Conditional path. Bottom row: Marginal probability path. As one can see, the SDE transports samples from into samples from δ z \\delta\_{z} (for the conditional path) and to (for the marginal path).

### 3.3 Conditional and Marginal Score Functions

We just successfully constructed a training target for a flow model. We now extend this reasoning to SDEs. To do so, let us define the marginal score function of $p_{t}$ as $\nabla\log p_{t}(x)$. We can use this to extend the ODE from the previous section to an SDE, as the following result demonstrates.

###### Theorem 13 (SDE extension trick)

Define the conditional and marginal vector fields $u^{\text{target}}_{t}(x|z)$ and $u^{\text{target}}_{t}(x)$ as before. Then, for diffusion coefficient $\sigma_{t}\geq 0$, we may construct an SDE which follows the same probability path:

$$
\displaystyle X_{0}\sim
$$
 
$$
\displaystyle p_{\rm{init}},\quad\mathrm{d}X_{t}=\left[u^{\text{target}}_{t}(X_{t})+\frac{\sigma_{t}^{2}}{2}\nabla\log p_{t}(X_{t})\right]\mathrm{d}t+\sigma_{t}\mathrm{d}W_{t}
$$
 
$$
\displaystyle\Rightarrow X_{t}\sim
$$
 
$$
\displaystyle p_{t}\quad(0\leq t\leq 1)
$$

In particular, $X_{1}\sim p_{\rm{data}}$ for this SDE. The same identity holds if we replace the marginal probability $p_{t}(x)$ and vector field $u^{\text{target}}_{t}(x)$ with the conditional probability path $p_{t}(x|z)$ and vector field $u^{\text{target}}_{t}(x|z)$.

We illustrate the theorem in fig.˜7. The formula in theorem˜13 is useful because, similar to before, we can express the marginal score function via the conditional score function $\nabla\log p_{t}(x|z)$

$$
\displaystyle\nabla\log p_{t}(x)=\frac{\nabla p_{t}(x)}{p_{t}(x)}=\frac{\nabla\int p_{t}(x|z)p_{\rm{data}}(z)\mathrm{d}z}{p_{t}(x)}=\frac{\int\nabla p_{t}(x|z)p_{\rm{data}}(z)\mathrm{d}z}{p_{t}(x)}=\int\nabla\log p_{t}(x|z)\frac{p_{t}(x|z)p_{\rm{data}}(z)}{p_{t}(x)}\mathrm{d}z
$$

and the conditional score function $\nabla\log p_{t}(x|z)$ is something we usually know analytically, as illustrated by the following example.

###### Example 14 (Score Function for Gaussian Probability Paths.)

For the Gaussian path $p_{t}(x|z)=\mathcal{N}(x;\alpha_{t}z,\beta_{t}^{2}I_{d})$, we can use the form of the Gaussian probability density (see eq.˜81) to get

$$
\displaystyle\nabla\log p_{t}(x|z)=\nabla\log\mathcal{N}(x;\alpha_{t}z,\beta_{t}^{2}I_{d})=-\frac{x-\alpha_{t}z}{\beta_{t}^{2}}.
$$

Note that the score is a linear function of $x$. This is a unique feature of Gaussian distributions.

In the remainder of this section, we will prove theorem˜13 via the Fokker-Planck equation, which extends the continuity equation from ODEs to SDEs. To do so, let us first define the Laplacian operator $\Delta$ via

$$
\displaystyle\Delta w_{t}(x)=
$$
 
$$
\displaystyle\sum\limits_{i=1}^{d}\frac{\partial^{2}}{\partial^{2}x_{i}}w_{t}(x)=\mathrm{div}(\nabla w_{t})(x).
$$

###### Theorem 15 (Fokker-Planck Equation)

Let $p_{t}$ be a probability path and let us consider the SDE

$$
\displaystyle X_{0}\sim p_{\rm{init}},\quad\mathrm{d}X_{t}=u_{t}(X_{t})\mathrm{d}t+\sigma_{t}\mathrm{d}W_{t}.
$$

Then $X_{t}$ has distribution $p_{t}$ for all $0\leq t\leq 1$ if and only if the Fokker-Planck equation holds:

$$
\displaystyle\partial_{t}p_{t}(x)=-\mathrm{div}(p_{t}u_{t})(x)+\frac{\sigma_{t}^{2}}{2}\Delta p_{t}(x)\quad\text{ for all }x\in\mathbb{R}^{d},0\leq t\leq 1.
$$

A self-contained proof of the Fokker-Planck equation can be found in appendix˜B. Note that the continuity equation is recovered from the Fokker-Planck equation when $\sigma_{t}=0$. The additional Laplacian term $\Delta p_{t}$ might be hard to rationalize at first. Those familiar with physics will note that the same term also appears in the heat equation (which is in fact a special case of the Fokker-Planck equation). Heat diffuses through a medium. We also add a diffusion process (not a physical but a mathematical one) and hence we add this additional Laplacian term. Let us now use the Fokker-Planck equation to help us prove theorem˜13.

###### Proof of.

By theorem˜15, we need to show that that the SDE defined in eq.˜25 satisfies the Fokker-Planck equation for $p_{t}$. We can do this by direction calculation:

$$
\displaystyle\partial_{t}p_{t}(x)\overset{(i)}{=}
$$
 
$$
\displaystyle-\mathrm{div}(p_{t}u^{\text{target}}_{t})(x)
$$
 
$$
\displaystyle\overset{(ii)}{=}
$$
 
$$
\displaystyle-\mathrm{div}(p_{t}u^{\text{target}}_{t})(x)-\frac{\sigma_{t}^{2}}{2}\Delta p_{t}(x)+\frac{\sigma_{t}^{2}}{2}\Delta p_{t}(x)
$$
 
$$
\displaystyle\overset{(iii)}{=}
$$
 
$$
\displaystyle-\mathrm{div}(p_{t}u^{\text{target}}_{t})(x)-\mathrm{div}(\frac{\sigma_{t}^{2}}{2}\nabla p_{t})(x)+\frac{\sigma_{t}^{2}}{2}\Delta p_{t}(x)
$$
 
$$
\displaystyle\overset{(iv)}{=}
$$
 
$$
\displaystyle-\mathrm{div}(p_{t}u^{\text{target}}_{t})(x)-\mathrm{div}(p_{t}\left[\frac{\sigma_{t}^{2}}{2}\nabla\log p_{t}\right])(x)+\frac{\sigma_{t}^{2}}{2}\Delta p_{t}(x)
$$
 
$$
\displaystyle\overset{(v)}{=}
$$
 
$$
\displaystyle-\mathrm{div}\left(p_{t}\left[u^{\text{target}}_{t}+\frac{\sigma_{t}^{2}}{2}\nabla\log p_{t}\right]\right)(x)+\frac{\sigma_{t}^{2}}{2}\Delta p_{t}(x),
$$

where in $(i)$ we used the Contuity Equation, in $(ii)$ we added and subtracted the same term, in $(iii)$ we used the definition of the Laplacian (eq.˜29), in $(iv)$ we used that $\nabla\log p_{t}=\frac{\nabla p_{t}}{p_{t}}$, and in $(v)$ we used the linearity of the divergence operator. The above derivation shows that the SDE defined in eq.˜25 satisfies the Fokker-Planck equation for $p_{t}$. By theorem˜15, this implies $X_{t}\sim p_{t}$ for $0\leq t\leq 1$, as desired. ∎

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.02070/assets/figures/langevin.png)

Figure 8: Top row: Particles evolving under the Langevin dynamics given by eq. ˜ 31, with p ( x ) p(x) taken to be a Gaussian mixture with 5 modes. Bottom row: A kernel density estimate of the same samples shown in the top row. As one can see, the distribution of samples converges to the equilibrium distribution (blue background colour).

###### Remark 16 (Langevin dynamics.)

The above construction has a famous special case when the probability path is static, i.e. $p_{t}=p$ for a fixed distribution $p$. In this case, we set $u^{\text{target}}_{t}=0$ and obtain the SDE

$$
\mathrm{d}X_{t}=\frac{\sigma_{t}^{2}}{2}\nabla\log p(X_{t})\mathrm{d}t+\sigma_{t}dW_{t},
$$

which is commonly known as Langevin dynamics. The fact that $p_{t}$ is static implies that $\partial_{t}p_{t}(x)=0$. It follows immediately from theorem˜13 that these dynamics satisfy the Fokker-Planck equation for the static path $p_{t}=p$ in theorem˜13. Therefore, we may conclude that $p$ is a stationary distribution of Langevin dynamics, so that

$$
\displaystyle X_{0}\sim p\quad\Rightarrow\quad X_{t}\sim p\quad(t\geq 0)
$$

As with many Markov chains, these dynamics converge to the stationary distribution $p$ under rather general conditions (see fig.˜8). That is, if we instead we take $X_{0}\sim p^{\prime}\neq p$, so that $X_{t}\sim p_{t}^{\prime}$, then under mild conditions $p_{t}\to p$. This fact makes Langevin dynamics extremely useful, and it accordingly serves as the basis for e.g., molecular dynamics simulations, and many other Markov chain Monte Carlo (MCMC) methods across Bayesian statistics and the natural sciences.

Let us summarize the results of this section.

###### Summary 17 (Derivation of the Training Target)

The flow training target is the marginal vector field $u^{\text{target}}_{t}$. To construct it, we choose a conditional probability path $p_{t}(x|z)$ that fulfils $p_{0}(\cdot|z)=p_{\rm{init}}$, $p_{1}(\cdot|z)=\delta_{z}$. Next, we find a conditional vector field $u^{\text{flow}}_{t}(x|z)$ such that its corresponding flow $\psi^{\text{target}}_{t}(x|z)$ fulfills

$$
\displaystyle X_{0}\sim p_{\rm{init}}\quad\Rightarrow\quad X_{t}=\psi^{\text{target}}_{t}(X_{0}|z)\sim p_{t}(\cdot|z),
$$

or, equivalently, that $u^{\text{target}}_{t}$ satisfies the continuity equation. Then the marginal vector field defined by

$$
\displaystyle u^{\text{target}}_{t}(x)=\int u^{\text{target}}_{t}(x|z)\frac{p_{t}(x|z)p_{\rm{data}}(z)}{p_{t}(x)}\mathrm{d}z,
$$

follows the marginal probability path, i.e.,

$$
\displaystyle X_{0}
$$
 
$$
\displaystyle\sim p_{\rm{init}},\quad\mathrm{d}X_{t}=u^{\text{target}}_{t}(X_{t})\mathrm{d}t\Rightarrow X_{t}\sim p_{t}\quad(0\leq t\leq 1).
$$

In particular, $X_{1}\sim p_{\rm{data}}$ for this ODE, so that $u^{\text{target}}_{t}$ "converts noise into data", as desired.

Extending to SDEs. For a time-dependent diffusion coefficient $\sigma_{t}\geq 0$, we can extend the above ODE to an SDE with the same marginal probability path:

$$
\displaystyle X_{0}\sim
$$
 
$$
\displaystyle p_{\rm{init}},\quad\mathrm{d}X_{t}=\left[u^{\text{target}}_{t}(X_{t})+\frac{\sigma_{t}^{2}}{2}\nabla\log p_{t}(X_{t})\right]\mathrm{d}t+\sigma_{t}\mathrm{d}W_{t}
$$
 
$$
\displaystyle\Rightarrow X_{t}\sim
$$
 
$$
\displaystyle p_{t}\quad(0\leq t\leq 1),
$$

where $\nabla\log p_{t}(x)$ is the marginal score function

$$
\displaystyle\nabla\log p_{t}(x)=
$$
 
$$
\displaystyle\int\nabla\log p_{t}(x|z)\frac{p_{t}(x|z)p_{\rm{data}}(z)}{p_{t}(x)}\mathrm{d}z.
$$

In particular, for the trajectories $X_{t}$ of the above SDE, it holds that $X_{1}\sim p_{\rm{data}}$, so that the SDE "converts noise into data", as desired. An important example is the Gaussian probability path, yielding the formulae:

$$
\displaystyle p_{t}(x|z)=
$$
 
$$
\displaystyle\mathcal{N}(x;\alpha_{t}z,\beta_{t}^{2}I_{d})
$$
 
$$
\displaystyle u^{\text{flow}}_{t}(x|z)=
$$
 
$$
\displaystyle\left(\dot{\alpha}_{t}-\frac{\dot{\beta}_{t}}{\beta_{t}}\alpha_{t}\right)z+\frac{\dot{\beta}_{t}}{\beta_{t}}x
$$
 
$$
\displaystyle\nabla\log p_{t}(x|z)=
$$
 
$$
\displaystyle-\frac{x-\alpha_{t}z}{\beta_{t}^{2}},
$$

for noise schedulers $\alpha_{t},\beta_{t}\in\mathbb{R}$: continuously differentiable, monotonic functions such that $\alpha_{0}=\beta_{1}=0$ $\alpha_{1}=\beta_{0}=1$.

## 4 Training the Generative Model

In the last two sections, we showed how to construct a generative model with a vector field $u_{t}^{\theta}$ given by a neural network, and we derived a formula for the training target $u^{\text{target}}_{t}$. In this section, we will describe how to train the neural network $u_{t}^{\theta}$ to approximate the training target $u^{\text{target}}_{t}$. First, we restrict ourselves to ODEs again, in doing so recovering flow matching. Second, we explain how to extend the approach to SDEs via score matching. Finally, we consider the special case of Gaussian probability paths, in doing so recovering denoising diffusion models. With these tools, we will at last have an end-to-end procedure to train and sample from a generative model with ODEs and SDEs.

### 4.1 Flow Matching

As before, let us consider a flow model given by

$$
\displaystyle X_{0}
$$
 
$$
\displaystyle\sim p_{\rm{init}},\quad\mathrm{d}X_{t}=u_{t}^{\theta}(X_{t})\,\mathrm{d}t.
$$
$$
\displaystyle\blacktriangleright\,\,\text{flow model}
$$

As we learned, we want the neural network $u_{t}^{\theta}$ to equal the marginal vector field $u^{\text{target}}_{t}$. In other words, we would like to find parameters $\theta$ so that $u_{t}^{\theta}\approx u^{\text{target}}_{t}$. In the following, we denote by $\text{Unif}=\text{Unif}_{[0,1]}$ the uniform distribution on the interval $[0,1]$, and by $\mathbb{E}$ the expected value of a random variable. An intuitive way of obtaining $u_{t}^{\theta}\approx u^{\text{target}}_{t}$ is to use a mean-squared error, i.e. to use the flow matching loss defined as

$$
\displaystyle\mathcal{L}_{\text{FM}}(\theta)
$$
 
$$
\displaystyle=\mathbb{E}_{t\sim\text{Unif},x\sim p_{t}}[\|u_{t}^{\theta}(x)-u^{\text{target}}_{t}(x)\|^{2}]
$$
 
$$
\displaystyle\overset{(i)}{=}\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},x\sim p_{t}(\cdot|z)}[\|u_{t}^{\theta}(x)-u^{\text{target}}_{t}(x)\|^{2}],
$$

where $p_{t}(x)=\int p_{t}(x|z)p_{\rm{data}}(z)\mathrm{d}z$ is the marginal probability path and in $(i)$ we used the sampling procedure given by eq.˜13. Intuitively, this loss says: First, draw a random time $t\in[0,1]$. Second, draw a random point $z$ from our data set, sample from $p_{t}(\cdot|z)$ (e.g., by adding some noise), and compute $u_{t}^{\theta}(x)$. Finally, compute the mean-squared error between the output of our neural network and the marginal vector field $u^{\text{target}}_{t}(x)$. Unfortunately, we are not done here. While we do know the formula for $u^{\text{target}}_{t}$ by theorem˜10

$$
\displaystyle u^{\text{target}}_{t}(x)=\int u^{\text{target}}_{t}(x|z)\frac{p_{t}(x|z)p_{\rm{data}}(z)}{p_{t}(x)}\mathrm{d}z,
$$

we cannot compute it efficiently because the above integral is intractable. Instead, we will exploit the fact that the conditional velocity field $u^{\text{target}}_{t}(x|z)$ is tractable. To do so, let us define the conditional flow matching loss

$$
\displaystyle\mathcal{L}_{\text{CFM}}(\theta)=\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},x\sim p_{t}(\cdot|z)}[\|u_{t}^{\theta}(x)-u^{\text{target}}_{t}(x|z)\|^{2}].
$$

Note the difference to eq.˜41: we use the conditional vector field $u^{\text{target}}_{t}(x|z)$ instead of the marginal vector $u^{\text{target}}_{t}(x)$. As we have an analytical formula for $u^{\text{target}}_{t}(x|z)$, we can minimize the above loss easily. But wait, what sense does it make to regress against the conditional vector field if it’s the marginal vector field we care about? As it turns out, by explicitly regressing against the tractable, conditional vector field, we are implicitly regressing against the intractable, marginal vector field. The next result makes this intuition precise.

###### Theorem 18

The marginal flow matching loss equals the conditional flow matching loss up to a constant. That is,

$$
\displaystyle\mathcal{L}_{\text{FM}}(\theta)=\mathcal{L}_{\text{CFM}}(\theta)+C,
$$

where $C$ is independent of $\theta$. Therefore, their gradients coincide:

$$
\displaystyle\nabla_{\theta}\mathcal{L}_{\text{FM}}(\theta)=\nabla_{\theta}\mathcal{L}_{\text{CFM}}(\theta).
$$

Hence, minimizing $\mathcal{L}_{\text{CFM}}(\theta)$ with e.g., stochastic gradient descent (SGD) is equivalent to minimizing $\mathcal{L}_{\text{FM}}(\theta)$ with in the same fashion. In particular, for the minimizer $\theta^{*}$ of $\mathcal{L}_{\text{CFM}}(\theta)$, it will hold that $u_{t}^{\theta^{*}}=u^{\text{target}}_{t}$ (assuming an infintely expressive parameterization).

###### Proof.

The proof works by expanding the mean-squared error into three components and removing constants:

$$
\displaystyle\mathcal{L}_{\text{FM}}(\theta)
$$
 
$$
\displaystyle\overset{(i)}{=}\mathbb{E}_{t\sim\text{Unif},x\sim p_{t}}[\|u_{t}^{\theta}(x)-u^{\text{target}}_{t}(x)\|^{2}]
$$
 
$$
\displaystyle\overset{(ii)}{=}\mathbb{E}_{t\sim\text{Unif},x\sim p_{t}}[\|u_{t}^{\theta}(x)\|^{2}-2u_{t}^{\theta}(x)^{T}u^{\text{target}}_{t}(x)+\|u^{\text{target}}_{t}(x)\|^{2}]
$$
 
$$
\displaystyle\overset{(iii)}{=}\mathbb{E}_{t\sim\text{Unif},x\sim p_{t}}\left[\|u_{t}^{\theta}(x)\|^{2}\right]-2\mathbb{E}_{t\sim\text{Unif},x\sim p_{t}}[u_{t}^{\theta}(x)^{T}u^{\text{target}}_{t}(x)]+\underbrace{\mathbb{E}_{t\sim\text{Unif}_{[0,1]},x\sim p_{t}}[\|u^{\text{target}}_{t}(x)\|^{2}]}_{=:C_{1}}
$$
 
$$
\displaystyle\overset{(iv)}{=}\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},x\sim p_{t}(\cdot|z)}[\|u_{t}^{\theta}(x)\|^{2}]-2\mathbb{E}_{t\sim\text{Unif},x\sim p_{t}}[u_{t}^{\theta}(x)^{T}u^{\text{target}}_{t}(x)]+C_{1}
$$

where $(i)$ holds by definition, in $(ii)$ we used the formula $\|a-b\|^{2}=\|a\|^{2}-2a^{T}b+\|b\|^{2}$, in $(iii)$ we define a constant $C_{1}$ and in $(iv)$ we used the sampling procedure of $p_{t}$ given by eq.˜13. Let us reexpress the second summand:

$$
\displaystyle\mathbb{E}_{t\sim\text{Unif},x\sim p_{t}}[u_{t}^{\theta}(x)^{T}u^{\text{target}}_{t}(x)]
$$
 
$$
\displaystyle\overset{(i)}{=}\int\limits_{0}^{1}\int p_{t}(x)u_{t}^{\theta}(x)^{T}u^{\text{target}}_{t}(x)\,\mathrm{d}x\,\mathrm{d}t
$$
 
$$
\displaystyle\overset{(ii)}{=}\int\limits_{0}^{1}\int p_{t}(x)u_{t}^{\theta}(x)^{T}\left[\int u^{\text{target}}_{t}(x|z)\frac{p_{t}(x|z)p_{\rm{data}}(z)}{p_{t}(x)}\mathrm{d}z\right]\mathrm{d}x\,\mathrm{d}t
$$
 
$$
\displaystyle\overset{(iii)}{=}\int\limits_{0}^{1}\int\int u_{t}^{\theta}(x)^{T}u^{\text{target}}_{t}(x|z)p_{t}(x|z)p_{\rm{data}}(z)\,\mathrm{d}z\,\mathrm{d}x\,\mathrm{d}t
$$
 
$$
\displaystyle\overset{(iv)}{=}\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},x\sim p_{t}(\cdot|z)}[u_{t}^{\theta}(x)^{T}u^{\text{target}}_{t}(x|z)]
$$

where in $(i)$ we expressed the expected value as an integral, in $(ii)$ we use eq.˜43, in $(iii)$ we use the fact that integrals are linear, in $(iv)$ we express the integral as an expected value. Note that this was really the crucial step of the proof: The beginning of the equality used the marginal vector field $u^{\text{target}}_{t}(x)$, while the end uses the conditional vector field $u^{\text{target}}_{t}(x|z)$. We plug is into the equation for $\mathcal{L}_{\text{FM}}$ to get:

$$
\displaystyle\mathcal{L}_{\text{FM}}(\theta)
$$
 
$$
\displaystyle\overset{(i)}{=}\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},x\sim p_{t}(\cdot|z)}[\|u_{t}^{\theta}(x)\|^{2}]-2\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},x\sim p_{t}(\cdot|z)}[u_{t}^{\theta}(x)^{T}u^{\text{target}}_{t}(x|z)]+C_{1}
$$
 
$$
\displaystyle\overset{(ii)}{=}\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},x\sim p_{t}(\cdot|z)}[\|u_{t}^{\theta}(x)\|^{2}-2u_{t}^{\theta}(x)^{T}u^{\text{target}}_{t}(x|z)+\|u^{\text{target}}_{t}(x|z)\|^{2}-\|u^{\text{target}}_{t}(x|z)\|^{2}]+C_{1}
$$
 
$$
\displaystyle\overset{(iii)}{=}\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},x\sim p_{t}(\cdot|z)}[\|u_{t}^{\theta}(x)-u^{\text{target}}_{t}(x|z)\|^{2}]+\underbrace{\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},x\sim p_{t}(\cdot|z)}[-\|u^{\text{target}}_{t}(x|z)\|^{2}]}_{C_{2}}+C_{1}
$$
 
$$
\displaystyle\overset{(iv)}{=}\mathcal{L}_{\text{CFM}}(\theta)+\underbrace{C_{2}+C_{1}}_{=:C}
$$

where in $(i)$ we plugged in the derived equation, in $(ii)$ we added and subtracted the same value, in $(iii)$ we used the formula $\|a-b\|^{2}=\|a\|^{2}-2a^{T}b+\|b\|^{2}$ again, and in $(iv)$ we defined a constant in $\theta$. This finishes the proof. ∎

Once $u_{t}^{\theta}$ has been trained, we may simulate the flow model

$$
\displaystyle\mathrm{d}X_{t}=u_{t}^{\theta}(X_{t})\,\mathrm{d}t,\quad\quad X_{0}\sim p_{\rm{init}}
$$

via e.g., algorithm˜1 to obtain samples $X_{1}\sim p_{\rm{data}}$. This whole pipeline is called flow matching in the literature \[[14](#bib.bibx14), [16](#bib.bibx16), [1](#bib.bibx1), [15](#bib.bibx15)\]. The training procedure is summarized in algorithm˜5 and visualized in fig.˜9. Let us now instantiate the conditional flow matching loss for the choice of Gaussian probability paths:

###### Example 19 (Flow Matching for Gaussian Conditional Probability Paths)

Let us return to the example of Gaussian probability paths $p_{t}(\cdot|z)=\mathcal{N}(\alpha_{t}z;\beta_{t}^{2}I_{d})$, where we may sample from the conditional path via

$$
\displaystyle\epsilon\sim\mathcal{N}(0,I_{d})\quad\Rightarrow\quad x_{t}=\alpha_{t}z+\beta_{t}\epsilon\sim\mathcal{N}(\alpha_{t}z,\beta_{t}^{2}I_{d})=p_{t}(\cdot|z).
$$

As we derived in eq.˜21, the conditional vector field $u^{\text{target}}_{t}(x|z)$ is given by

$$
\displaystyle u^{\text{target}}_{t}(x|z)=
$$
 
$$
\displaystyle\left(\dot{\alpha}_{t}-\frac{\dot{\beta}_{t}}{\beta_{t}}\alpha_{t}\right)z+\frac{\dot{\beta}_{t}}{\beta_{t}}x,
$$

where $\dot{\alpha}_{t}=\partial_{t}\alpha_{t}$ and $\dot{\beta}_{t}=\partial_{t}\beta_{t}$ are the respective time derivatives. Plugging in this formula, the conditional flow matching loss reads

$$
\displaystyle\mathcal{L}_{\text{CFM}}(\theta)
$$
 
$$
\displaystyle=\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},x\sim\mathcal{N}(\alpha_{t}z,\beta_{t}^{2}I_{d})}[\lVert u_{t}^{\theta}(x)-\left(\dot{\alpha}_{t}-\frac{\dot{\beta}_{t}}{\beta_{t}}\alpha_{t}\right)z-\frac{\dot{\beta}_{t}}{\beta_{t}}x\rVert^{2}]
$$
 
$$
\displaystyle\overset{(i)}{=}\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},\epsilon\sim\mathcal{N}(0,I_{d})}[\|u_{t}^{\theta}(\alpha_{t}z+\beta_{t}\epsilon)-(\dot{\alpha}_{t}z+\dot{\beta}_{t}\epsilon)\|^{2}]
$$

where in $(i)$ we plugged in eq.˜46 and replaced $x$ by $\alpha_{t}z+\beta_{t}\epsilon$. Note the simplicity of $\mathcal{L}_{\text{CFM}}$: We sample a data point $z$, sample some noise $\epsilon$ and then we take a mean squared error. Let us make this even more concrete for the special case of $\alpha_{t}=t$, and $\beta_{t}=1-t$. The corresponding probability $p_{t}(x|z)=\mathcal{N}(tz,(1-t)^{2})$ is sometimes referred to as the (Gaussian) CondOT probability path. Then we have $\dot{\alpha}_{t}=1,\dot{\beta}_{t}=-1$, so that

$$
\displaystyle\mathcal{L}_{\text{cfm}}(\theta)=
$$
 
$$
\displaystyle\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},\epsilon\sim\mathcal{N}(0,I_{d})}[\|u_{t}^{\theta}(tz+(1-t)\epsilon)-(z-\epsilon)\|^{2}]
$$

Many famous state-of-the-art models have been trained using this simple yet effective procedure, e.g. Stable Diffusion 3, Meta’s Movie Gen Video, and probably many more proprietary models. In fig.˜9, we visualize it in a simple example and in algorithm˜5 we summarize the training procedure.

Algorithm 3 Flow Matching Training Procedure (here for Gaussian CondOT path $p_{t}(x|z)=\mathcal{N}(tz,(1-t)^{2})$)

 A dataset of samples $z\sim p_{\rm{data}}$, neural network $u_{t}^{\theta}$

 for each mini-batch of data do

  Sample a data example $z$ from the dataset.

  Sample a random time $t\sim\text{Unif}_{[0,1]}$.

  Sample noise $\epsilon\sim\mathcal{N}(0,I_{d})$

  Set $x=tz+(1-t)\epsilon$ (General case: $x\sim p_{t}(\cdot|z)$)

  Compute loss 
$$
\displaystyle\mathcal{L}(\theta)=
$$
 
$$
\displaystyle\|u_{t}^{\theta}(x)-(z-\epsilon)\|^{2}\quad
$$
 
$$
\displaystyle(\text{General case: }=\|u_{t}^{\theta}(x)-u^{\text{target}}_{t}(x|z)\|^{2})
$$

  Update the model parameters $\theta$ via gradient descent on $\mathcal{L}(\theta)$.

 end for

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.02070/assets/figures/conditional_marginal_path.png)

Figure 9: Illustration of theorem ˜ 18 with a Gaussian CondOT probability path: simulating an ODE from a trained flow matching model. The data distribution is the chess board pattern (top right). Top row: Histogram from ground truth marginal probability path p t ( x ) p\_{t}(x). Bottom row: Histogram of samples from flow matching model. As one can see, the top row and bottom row match after training (up to training error). The model was trained using algorithm 5.

### 4.2 Score Matching

Let us extend the algorithm we just found from ODEs to SDEs. Remember we can extend the target ODE to an SDE with the same marginal distribution given by

$$
\displaystyle\mathrm{d}X_{t}
$$
 
$$
\displaystyle=\left[u^{\text{target}}_{t}(X_{t})+\frac{\sigma_{t}^{2}}{2}\nabla\log p_{t}(X_{t})\right]\mathrm{d}t+\sigma_{t}\mathrm{d}W_{t}
$$
 
$$
\displaystyle X_{0}
$$
 
$$
\displaystyle\sim p_{\rm{init}},\quad
$$
 
$$
\displaystyle\Rightarrow X_{t}
$$
 
$$
\displaystyle\sim p_{t}\quad(0\leq t\leq 1)
$$

where $u^{\text{target}}_{t}$ is the marginal vector field and $\nabla\log p_{t}(x)$ is the marginal score function represented via the formula

$$
\displaystyle\nabla\log p_{t}(x)=
$$
 
$$
\displaystyle\int\nabla\log p_{t}(x|z)\frac{p_{t}(x|z)p_{\rm{data}}(z)}{p_{t}(x)}\,\mathrm{d}z.
$$

To approximate the marginal score $\nabla\log p_{t}$, we can use a neural network that we call score network $s_{t}^{\theta}:\mathbb{R}^{d}\times[0,1]\to\mathbb{R}^{d}$. In the same way as before, we can design a score matching loss and a conditional score matching loss:

$$
\displaystyle\mathcal{L}_{\text{SM}}(\theta)
$$
 
$$
\displaystyle=\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},x\sim p_{t}(\cdot|z)}[\|s_{t}^{\theta}(x)-\nabla\log p_{t}(x)\|^{2}]
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{score matching loss}
$$
 
$$
\displaystyle\mathcal{L}_{\text{CSM}}(\theta)
$$
 
$$
\displaystyle=\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},x\sim p_{t}(\cdot|z)}[\|s_{t}^{\theta}(x)-\nabla\log p_{t}(x|z)\|^{2}]
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{conditional score matching loss}
$$

where again the difference is using the marginal score $\nabla\log p_{t}(x)$ vs. using the conditional score $\nabla\log p_{t}(x|z)$. As before, we ideally would want to minimize the score matching loss but can’t because we don’t know $\nabla\log p_{t}(x)$. But similarly as before, the conditional score matching loss is a tractable alternative:

###### Theorem 20

The score matching loss equals the conditional score matching loss up to a constant:

$$
\displaystyle\mathcal{L}_{\text{SM}}(\theta)=\mathcal{L}_{\text{CSM}}(\theta)+C,
$$

where $C$ is independent of parameters $\theta$. Therefore, their gradients coincide:

$$
\displaystyle\nabla_{\theta}\mathcal{L}_{\text{SM}}(\theta)=\nabla_{\theta}\mathcal{L}_{\text{CSM}}(\theta).
$$

In particular, for the minimizer $\theta^{*}$, it will hold that $s_{t}^{\theta^{*}}=\nabla\log p_{t}$.

###### Proof.

Note that the formula for $\nabla\log p_{t}$ (eq.˜51) looks the same as the formula for $u^{\text{target}}_{t}$ (eq.˜43). Therefore, the proof is identical to the proof of theorem˜18 replacing $u^{\text{target}}_{t}$ with $\nabla\log p_{t}$. ∎

The above procedure describes the vanilla procedure of training a diffusion model. After training, we can choose an arbitrary diffusion coefficient $\sigma_{t}\geq 0$ and then simulate the SDE

$$
\displaystyle X_{0}\sim
$$
 
$$
\displaystyle p_{\rm{init}},\quad\mathrm{d}X_{t}=\left[u_{t}^{\theta}(X_{t})+\frac{\sigma_{t}^{2}}{2}s_{t}^{\theta}(X_{t})\right]\mathrm{d}t+\sigma_{t}\mathrm{d}W_{t},
$$

to generate samples $X_{1}\sim p_{\rm{data}}$. In theory, every $\sigma_{t}$ should give samples $X_{1}\sim p_{\rm{data}}$ at perfect training. In practice, we encounter two types of errors: (1) numerical errors by simulating the SDE imperfectly and (2) training errors (i.e., the model $u_{t}^{\theta}$ is not exactly equal to $u^{\text{target}}_{t}$). Therefore, there is an optimal unknown noise level $\sigma_{t}$ - this can be determined empirically by just testing our different values of empirically (see e.g. \[[1](#bib.bibx1), [12](#bib.bibx12), [17](#bib.bibx17)\]). At first sight, it might seem to be a disadvantage that we have to learn both $s_{t}^{\theta}$ and $u_{t}^{\theta}$ if we wanted to use diffusion model now as opposed to a flow model. However, note we can often directly $s_{t}^{\theta}$ and $u_{t}^{\theta}$ in a single network with two outputs, so that the additional computational effort is usually minimal. Further, as we will see now for the special case of the Gaussian probability path, $s_{t}^{\theta}$ and $u_{t}^{\theta}$ may be converted into one another so that we don’t have to train them separately.

###### Remark 21 (Denoising Diffusion Models)

If you are familiar with diffusion models, you have probably encountered the term denoising diffusion model. This model has become so popular that most people nowadays drop the word "denoising" and simply use the term "diffusion model" to describe it. In the language of this document, these are simply diffusion models with Gaussian probability paths $p_{t}(\cdot|z)=\mathcal{N}(\alpha_{t}z;\beta_{t}^{2}I_{d})$. However, it is important to note that this might not be immediately obvious if you read some of the first diffusion model papers: they use a different time convention (time is inverted) - so you need apply an appropriate time re-scaling - and they construct their probability path via so-called forward processes (we will discuss this in section˜4.3).

###### Example 22 (Denoising Diffusion Models: Score Matching for Gaussian Probability Paths)

First, let us instantiate the denoising score matching loss for the case of $p_{t}(x|z)=\mathcal{N}(\alpha_{t}z,\beta_{t}^{2}I_{d})$. As we derived in eq.˜28, the conditional score $\nabla\log p_{t}(x|z)$ has the formula

$$
\displaystyle\nabla\log p_{t}(x|z)=-\frac{x-\alpha_{t}z}{\beta_{t}^{2}}.
$$

Plugging in this formula, the conditional score matching loss becomes:

$$
\displaystyle\mathcal{L}_{\text{CSM}}(\theta)
$$
 
$$
\displaystyle=\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},x\sim p_{t}(\cdot|z)}[\|s_{t}^{\theta}(x)+\frac{x-\alpha_{t}z}{\beta_{t}^{2}}\|^{2}]
$$
 
$$
\displaystyle\overset{(i)}{=}\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},\epsilon\sim\mathcal{N}(0,I_{d})}[\|s_{t}^{\theta}(\alpha_{t}z+\beta_{t}\epsilon)+\frac{\epsilon}{\beta_{t}}\|^{2}]
$$
 
$$
\displaystyle=\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},\epsilon\sim\mathcal{N}(0,I_{d})}[\frac{1}{\beta_{t}^{2}}\|\beta_{t}s_{t}^{\theta}(\alpha_{t}z+\beta_{t}\epsilon)+\epsilon\|^{2}]
$$

where in $(i)$ we plugged in eq.˜46 and replaced $x$ by $\alpha_{t}z+\beta_{t}\epsilon$. Note that the network $s_{t}^{\theta}$ essentially learns to predict the noise that was used to corrupt a data sample $z$. Therefore, the above training loss is also called denoising score matching and it was the one of the first procedures used to learn diffusion models. It was soon realized that the above loss is numerically unstable for $\beta_{t}\approx 0$ close to zero (i.e. denoising score matching only works if you add a sufficient amount of noise). In some of the first works on denoising diffusion models (see Denoising Diffusion Probabilitic Models, \[[9](#bib.bibx9)\]) it was therefore proprosed to drop the constant $\frac{1}{\beta_{t}^{2}}$ in the loss and reparameterize $s_{t}^{\theta}$ into a noise predictor network $\epsilon_{t}^{\theta}:\mathbb{R}^{d}\times[0,1]\to\mathbb{R}^{d}$ via:

$$
\displaystyle-\beta_{t}s_{t}^{\theta}(x)=\epsilon_{t}^{\theta}(x)\quad\Rightarrow\quad\mathcal{L}_{\text{DDPM}}(\theta)=
$$
 
$$
\displaystyle\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},\epsilon\sim\mathcal{N}(0,I_{d})}\left[\|\epsilon_{t}^{\theta}(\alpha_{t}z+\beta_{t}\epsilon)-\epsilon\|^{2}\right]
$$

As before, the network $\epsilon_{t}^{\theta}$ essentially learns to predict the noise that was used to corrupt a data sample $z$. In algorithm˜4, we summarize the training procedure.

Algorithm 4 Score Matching Training Procedure for Gaussian probability path

 A dataset of samples $z\sim p_{\rm{data}}$, score network $s_{t}^{\theta}$ or noise predictor $\epsilon_{t}^{\theta}$

 for each mini-batch of data do

  Sample a data example $z$ from the dataset.

  Sample a random time $t\sim\text{Unif}_{[0,1]}$.

  Sample noise $\epsilon\sim\mathcal{N}(0,I_{d})$

  Set $x_{t}=\alpha_{t}z+\beta_{t}\epsilon$ (General case: $x_{t}\sim p_{t}(\cdot|z)$)

  Compute loss 
$$
\displaystyle\mathcal{L}(\theta)=
$$
 
$$
\displaystyle\|s_{t}^{\theta}(x_{t})+\frac{\epsilon}{\beta_{t}}\|^{2}\quad
$$
 
$$
\displaystyle(\text{General case: }=\|s_{t}^{\theta}(x_{t})-\nabla\log p_{t}(x_{t}|z)\|^{2})
$$
 
$$
\displaystyle\text{Alternatively: }\mathcal{L}(\theta)=
$$
 
$$
\displaystyle\|\epsilon_{t}^{\theta}(x_{t})-\epsilon\|^{2}
$$

  Update the model parameters $\theta$ via gradient descent on $\mathcal{L}(\theta)$.

 end for

Beyond its simplicity, there is another useful property of the Gaussian probability path: By learning $s_{t}^{\theta}$ or $\epsilon_{t}^{\theta}$, we also learn $u_{t}^{\theta}$ automatically and the other way around:

###### Proposition 1 (Conversion formula for Gaussian probability path)

For the Gaussian probability path $p_{t}(x|z)=\mathcal{N}(\alpha_{t}z,\beta_{t}^{2}I_{d})$, it holds that that the conditional (resp. marginal) vector field can be converted into the conditional (resp. marginal) score:

$$
\displaystyle u^{\text{target}}_{t}(x|z)=\left(\beta_{t}^{2}\frac{\dot{\alpha}_{t}}{\alpha_{t}}-\dot{\beta}_{t}\beta_{t}\right)\nabla\log p_{t}(x|z)+\frac{\dot{\alpha}_{t}}{\alpha_{t}}x
$$
 
$$
\displaystyle u^{\text{target}}_{t}(x)=\left(\beta_{t}^{2}\frac{\dot{\alpha}_{t}}{\alpha_{t}}-\dot{\beta}_{t}\beta_{t}\right)\nabla\log p_{t}(x)+\frac{\dot{\alpha}_{t}}{\alpha_{t}}x
$$

where the formula for the above marginal vector field $u^{\text{target}}_{t}$ is called probability flow ODE in the literature (more correctly, the corresponding ODE).

###### Proof.

For the conditional vector field and conditional score, we can derive:

$$
\displaystyle u^{\text{target}}_{t}(x|z)=
$$
 
$$
\displaystyle\left(\dot{\alpha}_{t}-\frac{\dot{\beta}_{t}}{\beta_{t}}\alpha_{t}\right)z+\frac{\dot{\beta}_{t}}{\beta_{t}}x\overset{(i)}{=}\left(\beta_{t}^{2}\frac{\dot{\alpha}_{t}}{\alpha_{t}}-\dot{\beta}_{t}\beta_{t}\right)\left(\frac{\alpha_{t}z-x}{\beta_{t}^{2}}\right)+\frac{\dot{\alpha}_{t}}{\alpha_{t}}x=\left(\beta_{t}^{2}\frac{\dot{\alpha}_{t}}{\alpha_{t}}-\dot{\beta}_{t}\beta_{t}\right)\nabla\log p_{t}(x|z)+\frac{\dot{\alpha}_{t}}{\alpha_{t}}x
$$

where in $(i)$ we just did some algebra. By taking integrals, the same identity holds for the marginal flow vector field and the marginal score function:

$$
\displaystyle u^{\text{target}}(x)=\int u^{\text{target}}_{t}(x|z)\frac{p_{t}(x|z)p_{\rm{data}}(z)}{p_{t}(x)}\mathrm{d}z=
$$
 
$$
\displaystyle\int\left[\left(\beta_{t}^{2}\frac{\dot{\alpha}_{t}}{\alpha_{t}}-\dot{\beta}_{t}\beta_{t}\right)\nabla\log p_{t}(x|z)+\frac{\dot{\alpha}_{t}}{\alpha_{t}}x\right]\frac{p_{t}(x|z)p_{\rm{data}}(z)}{p_{t}(x)}\mathrm{d}z
$$
 
$$
\displaystyle\overset{(i)}{=}
$$
 
$$
\displaystyle\left(\beta_{t}^{2}\frac{\dot{\alpha}_{t}}{\alpha_{t}}-\dot{\beta}_{t}\beta_{t}\right)\nabla\log p_{t}(x)+\frac{\dot{\alpha}_{t}}{\alpha_{t}}x
$$

where in $(i)$ we used eq.˜51. ∎

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.02070/assets/figures/score_field_comparison.png)

Figure 10: A comparison of the score, as obtained in two different ways. Top: A visualization of the score field s t θ ( x ) s\_{t}^{\\theta}(x) learned independently with score matching (see algorithm ˜ 4 ). Bottom: A visualization of the score field ~ \\tilde{s}\_{t}^{\\theta}(x) parameterized using u u\_{t}^{\\theta}(x) as in eq. 55.

We can use the conversion formula to parameterize the score network $s_{t}^{\theta}$ and the vector field network $u_{t}^{\theta}$ into one another via

$$
\displaystyle u_{t}^{\theta}=\left(\beta_{t}^{2}\frac{\dot{\alpha}_{t}}{\alpha_{t}}-\dot{\beta}_{t}\beta_{t}\right)s_{t}^{\theta}(x)+\frac{\dot{\alpha}_{t}}{\alpha_{t}}x.
$$

Similarly, so long as $\beta_{t}^{2}\dot{\alpha}_{t}-\alpha_{t}\dot{\beta}_{t}\beta_{t}\neq 0$ (always true for $t\in[0,1)$), it follows that

$$
\displaystyle s_{t}^{\theta}(x)=\frac{\alpha_{t}u_{t}^{\theta}(x)-\dot{\alpha}_{t}x}{\beta_{t}^{2}\dot{\alpha}_{t}-\alpha_{t}\dot{\beta}_{t}\beta_{t}}.
$$

Using this parameterization, it can be shown that the denoising score matching and the conditional flow matching losses are the same up to a constant. We conclude that for Gaussian probability paths there is no need to separately train both the marginal score and the marginal vector field, as knowledge of one is sufficient to compute the other. In particular, we can choose whether we want to use flow matching or score matching to train it. In fig.˜10, we compare visually the score as approximated using score matching and the parameterized score using eq.˜55. If we have trained a score network $s_{t}^{\theta}$, we know by eq.˜52 that we can use arbitrary $\sigma_{t}\geq 0$ to sample from the SDE

$$
\displaystyle X_{0}\sim
$$
 
$$
\displaystyle p_{\rm{init}},\quad\mathrm{d}X_{t}=\left[\left(\beta_{t}^{2}\frac{\dot{\alpha}_{t}}{\alpha_{t}}-\dot{\beta}_{t}\beta_{t}+\frac{\sigma_{t}^{2}}{2}\right)s_{t}^{\theta}(x)+\frac{\dot{\alpha}_{t}}{\alpha_{t}}x\right]\mathrm{d}t+\sigma_{t}\mathrm{d}W_{t}
$$

to obtain samples $X_{1}\sim p_{\rm{data}}$ (up to training and simulation error). This corresponds to stochastic sampling from a denoising diffusion model.

### 4.3 A Guide to the Diffusion Model Literature

There is a whole family of models around diffusion models and flow matching in the literature. When you read these papers, you will likely find a different (but equivalent) way of presenting the material from this class. This makes it sometimes a little confusing to read these papers. For this reason, we want to give a brief overview over various frameworks and their differences and put them also in their historical context. This is not necessary to understand the remainder of this document but rather intended to be a support for you in case you read the literature.

##### Discrete time vs. continuous time.

The first denoising diffusion model papers \[[28](#bib.bibx28), [29](#bib.bibx29), [9](#bib.bibx9)\] did not use SDEs but constructed Markov chains in discrete time, i.e. with time steps $t=0,1,2,3,\dots$. To this date, you will find a lot of works in the literature working with this discrete-time formulation. While this construction is appealing due to its simplicity, the disadvantage of the time-discrete approach is that it forces you to choose a time discretization before training. Further, the loss function needs to be approximated via an evidence lower bound (ELBO) - which is, as the name suggests, only a lower bound to the loss we actually want to minimize. Later, \[[32](#bib.bibx32)\] showed that these constructions were essentially an approximation of a time-continuous SDEs. Further, the ELBO loss becomes tight (i.e. it is not a lower bound anymore) in the continuous time case (e.g. note that theorem˜18 and theorem˜20 are equalities and not lower bounds - this would be different in the discrete time case). This made the SDE construction popular because it was considered mathematically "cleaner" and that one could control the simulation error via ODE/SDE samplers post training. It is important to note however that both models employ the same loss and are not fundamentally different.

##### "Forward process" vs probability paths.

The first wave of denoising diffusion models \[[28](#bib.bibx28), [29](#bib.bibx29), [9](#bib.bibx9), [32](#bib.bibx32)\] did not use the term probability path but constructed a noising procedure of a data point $z\in\mathbb{R}^{d}$ via a so-called forward process. This is an SDE of the form

$$
\displaystyle\bar{X}_{0}=z,\quad\mathrm{d}\bar{X}_{t}=u^{\text{forw}}_{t}(\bar{X}_{t})\mathrm{d}t+\sigma^{\text{forw}}_{t}\mathrm{d}\bar{W}_{t}
$$

The idea is that after drawing a data point $z\sim p_{\rm{data}}$ one simulates the forward process and thereby corrupts or "noises" the data. The forward process is designed such that for $t\to\infty$ its distribution converges to a Gaussian $\mathcal{N}(0,I_{d})$. In other words, for $T\gg 0$ it holds that $\bar{X}_{T}\sim\mathcal{N}(0,I_{d})$ approximately. Note that this essentially corresponds to a probability path: the conditional distribution of $\bar{X}_{t}$ given $\bar{X}_{0}=z$ is a conditional probability path $\bar{p}_{t}(\cdot|z)$ and the distribution of $\bar{X}_{t}$ marginalized over $z\sim p_{\rm{data}}$ corresponds to a marginal probability path $\bar{p}_{t}$.[^3] However, note that with this construction, we need to know the distribution of $X_{t}|X_{0}=z$ in closed form in order to train our models to avoid simulating the SDE. This essentially restrict the vector field $u^{\text{forw}}_{t}$ to ones such that we know the distribution $\bar{X}_{t}|\bar{X}_{0}=z$ in closed form. Therefore, throughout the diffusion model literature, vector fields in forward processes are always of the affine form, i.e. $u^{\text{forw}}_{t}(x)=a_{t}x$ for some continuous function $a_{t}$. For this choice, we can use known formulas of the conditional distribution \[[27](#bib.bibx27), [31](#bib.bibx31), [12](#bib.bibx12)\]:

$$
\displaystyle\bar{X}_{t}|\bar{X}_{0}=z\sim\mathcal{N}\left(\alpha_{t}z,\beta_{t}^{2}I\right),\quad\alpha_{t}=\exp\left(\int\limits_{0}^{t}a_{r}\mathrm{d}r\right),\quad\beta_{t}^{2}=\alpha_{t}^{2}\int\limits_{0}^{t}\frac{(\sigma^{\text{forw}}_{r})^{2}}{\alpha^{2}_{r}}dr
$$

Note that these are simply Gaussian probability paths. Therefore, one can say that a forward process is a specific way of constructing a (Gaussian) probability path. The term probability path was introduced by flow matching \[[14](#bib.bibx14)\] to both simplify the construction and make it more general at the same time: First, the "forward process" of diffusion models is never actually simulated (only samples from $\bar{p}_{t}(\cdot|z)$ are drawn during training). Second, a forward process only converges for $t\to\infty$ (i.e. we will never arrive at $p_{\rm{init}}$ in finite time). Therefore, we choose to use probability paths in this document.

##### Time-Reversals vs Solving the Fokker-Planck equation.

The original description of diffusion models did not construct the training target $u^{\text{target}}_{t}$ or $\nabla\log p_{t}$ via the Fokker-Planck equation (or Continuity equation) but via a time-reversal of the forward process \[[2](#bib.bibx2)\]. A time-reversal $(X_{t})_{0\leq t\leq T}$ is an SDE with the same distribution over trajectories inverted in time, i.e.

$$
\displaystyle\mathbb{P}[\bar{X}_{t_{1}}\in A_{1},\dots,\bar{X}_{t_{n}}\in A_{n}]=\mathbb{P}[X_{T-t_{1}}\in A_{1},\dots,X_{T-t_{n}}\in A_{n}]
$$
 
$$
\displaystyle\text{ for all }0\leq t_{1},\dots,t_{n}\leq T,\text{ and }A_{1},\dots,A_{n}\subset S
$$

As shown in \[[2](#bib.bibx2)\], one can obtain a time-reversal satisfying the above condition by the SDE:

$$
\displaystyle\mathrm{d}X_{t}=
$$
 
$$
\displaystyle\left[-u_{t}(X_{t})+\sigma_{t}^{2}\nabla\log p_{t}(X_{t})\right]\mathrm{d}t+\sigma_{t}\mathrm{d}W_{t},\quad u_{t}(x)=u^{\text{forw}}_{T-t}(x),\sigma_{t}=\bar{\sigma}_{T-t}
$$

As $u_{t}(X_{t})=a_{t}X_{t}$, the above corresponds to a specific instance of training target we derived in proposition˜1 (this is not immediately trivial as different time conventions are used. See e.g. \[[15](#bib.bibx15)\] for a derivation). However, for the purposes of generative modeling, we often only use the final point $X_{1}$ of the Markov process (e.g., as a generated image) and discard earlier time points. Therefore, whether a Markov process is a “true” time-reversal or follows along a probability path does not matter for many applications. Therefore, using a time-reversal is not necessary and often leads to suboptimal results, e.g. the probability flow ODE is often better \[[12](#bib.bibx12), [17](#bib.bibx17)\]. All ways of sampling from a diffusion models that are different from the time-reversal rely again on using the Fokker-Planck equation. We hope that this illustrates why nowadays many people construct the training targets directly via the Fokker-Planck equation - as pioneered by \[[14](#bib.bibx14), [16](#bib.bibx16), [1](#bib.bibx1)\] and done in this class.

##### Flow Matching \[\] and Stochastic Interpolants \[\].

The framework that we present is most closely related to the frameworks of flow matching and stochastic interpolants (SIs). As we learnt, flow matching restricts itself to flows. In fact, one of the key innovations of flow matching was to show that one does not need a construction via a forward process and SDEs but flow models alone can be trained in a scalable manner. Due to this restriction, you should keep in mind that sampling from a flow matching model will be deterministic (only the initial $X_{0}\sim p_{\rm{init}}$ will be random). Stochastic interpolants included both the pure flow and the SDE extension via "Langevin dynamics" that we use here (see theorem˜13). Stochastic interpolants get their name from a interpolant function $I(t,x,z)$ intended to interpolate between two distributions. In the terminology we use here, this corresponds to a different yet (mainly) equivalent way of constructing a conditional and marginal probability path. The advantage of flow matching and stochastic interpolants over diffusion models is both their simplicity and their generality: their training framework is very simple but at the same time they allow you to go from an arbitrary distribution $p_{\rm{init}}$ to an arbitrary distribution $p_{\rm{data}}$ - while denoising diffusion models only work for Gaussian initial distributions and Gaussian probability path. This opens up new possibilities for generative modeling that we will touch upon briefly later in this class.

Let us summarize the results of this section:

###### Summary 23 (Training the Generative Model)

Flow matching consists of training a neural network $u_{t}^{\theta}$ via minimizing the conditional flow matching loss

$$
\displaystyle\mathcal{L}_{\text{CFM}}(\theta)=\mathbb{E}_{z\sim p_{\rm{data}},t\sim\text{Unif},\,x\sim p_{t}(\cdot|z)}[\|u_{t}^{\theta}(x)-u^{\text{target}}_{t}(x|z)\|^{2}]\quad
$$
 
$$
\displaystyle(\text{conditional flow matching loss})
$$

where $u^{\text{target}}_{t}(x|z)$ is the conditional vector field (see algorithm˜5). After training, one generates samples by simulating the corresponding ODE (see algorithm˜1). To extend this to a diffusion model, we can use a score network $s_{t}^{\theta}$ and train it via conditional score matching

$$
\displaystyle\mathcal{L}_{\text{CSM}}(\theta)
$$
 
$$
\displaystyle=\mathbb{E}_{z\sim p_{\rm{data}},\,t\sim\text{Unif},\,x\sim p_{t}(\cdot|z)}[\|s_{t}^{\theta}(x)-\nabla\log p_{t}(x|z)\|^{2}]\quad
$$
 
$$
\displaystyle(\text{denoising score matching loss})
$$

For every diffusion coefficient $\sigma_{t}\geq 0$, simulating the SDE (e.g. via algorithm˜2)

$$
\displaystyle X_{0}\sim
$$
 
$$
\displaystyle p_{\rm{init}},\quad\mathrm{d}X_{t}=\left[u_{t}^{\theta}(X_{t})+\frac{\sigma_{t}^{2}}{2}s_{t}^{\theta}(X_{t})\right]\mathrm{d}t+\sigma_{t}\mathrm{d}W_{t}
$$

will result in generating approximate samples from $p_{\rm{data}}$. One can empirically find the optimal $\sigma_{t}\geq 0$.

##### Gaussian probability paths.

For the special case of a Gaussian probability path $p_{t}(x|z)=\mathcal{N}(x;\alpha_{t}z,\beta_{t}^{2}I_{d})$, the conditional score matching is also called denoising score matching. This loss and conditional flow matching loss are then given by:

$$
\displaystyle\mathcal{L}_{\text{CFM}}(\theta)=
$$
 
$$
\displaystyle\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},\epsilon\sim\mathcal{N}(0,I_{d})}[\|u_{t}^{\theta}(\alpha_{t}z+\beta_{t}\epsilon)-(\dot{\alpha}_{t}z+\dot{\beta}_{t}\epsilon)\|^{2}]
$$
 
$$
\displaystyle\mathcal{L}_{\text{CSM}}(\theta)=
$$
 
$$
\displaystyle\mathbb{E}_{t\sim\text{Unif},z\sim p_{\rm{data}},\epsilon\sim\mathcal{N}(0,I_{d})}[\|s_{t}^{\theta}(\alpha_{t}z+\beta_{t}\epsilon)+\frac{\epsilon}{\beta_{t}}\|^{2}]
$$

In this case, there is no need to train $s_{t}^{\theta}$ and $u_{t}^{\theta}$ separately as we can convert them post training via the formula:

$$
\displaystyle u_{t}^{\theta}(x)=
$$
 
$$
\displaystyle\left(\beta_{t}^{2}\frac{\dot{\alpha}_{t}}{\alpha_{t}}-\dot{\beta}_{t}\beta_{t}\right)s_{t}^{\theta}(x)+\frac{\dot{\alpha}_{t}}{\alpha_{t}}x
$$

Also here, after training we can simulate the SDE in eq.˜62 via algorithm˜2 to obtain samples $X_{1}$.

##### Denoising diffusion models.

Denoising diffusion models are diffusion models with Gaussian probability paths. For this reason, it is sufficient for them to learn either $u_{t}^{\theta}$ or $s_{t}^{\theta}$ as they can be converted into one another. While flow matching only allows for a simulation procedure that is deterministic via ODE, they allow for a simulation that is deterministic (probability flow ODE) or stochastic (SDE sampling). However, unlike flow matching or stochastic interpolants that allow to convert arbitrary distributions $p_{\rm{init}}$ into arbitrary distributions $p_{\rm{data}}$ via arbitrary probability paths $p_{t}$, denoising diffusion models only works for Gaussian initial distributions $p_{\rm{init}}=\mathcal{N}(0,I_{d})$ and a Gaussian probability path.

##### Literature

Alternative formulations for diffusion models that are popular in the literature are:

1. Discrete-time: Approximations of SDEs via discrete-time Markov chains are often used.
2. Inverted time convention: It is popular to use an inverted time convention where $t=0$ corresponds to $p_{\rm{data}}$ (as opposed to here where $t=0$ corresponds to $p_{\rm{init}}$).
3. Forward process: Forward processes (or noising processes) are ways of constructing (Gaussian) probability paths.
4. Training target via time-reversal: A training target can also be constructed via the time-reversal of SDEs. This is a specific instance of the construction presented here (with an inverted time convention).

## 5 Building an Image Generator

In the previous sections, we learned how to train a flow matching or diffusion model to sample from a distribution $p_{\rm{data}}(x)$. This recipe is general and can be applied to a variety of different data types and applications. In this section, we learn how to apply this framework to build an image or video generator, such as e.g., Stable Diffusion 3 and Meta Movie Gen Video. To build such a model, there are two main ingredients that we are missing: First, we will need to formulate conditional generation (guidance), e.g. how do we generate an image that fits a specific text prompt, and how our existing objectives may be suitably adapted to this end. We will also learn about classifier-free guidance, a popular technique used to enhance the quality of conditional generation. Second, we will discuss common neural network architectures, again focusing on those designed for images and videos. Finally, we will examine in depth the two state-of-the-art image and video models mentioned above - Stable Diffusion and Meta MovieGen - to give you a taste of how things are done at scale.

### 5.1 Guidance

So far, the generative models we considered were unconditional, e.g. an image model would simply generate some image. However, the task is not merely to generate an arbitrary object, but to generate an object conditioned on some additional information. For example, one might imagine a generative model for images which takes in a text prompt $y$, and then generates an image $x$ conditioned on $y$. For fixed prompt $y$, we would thus like to sample from $p_{\rm{data}}(x|y)$, that is, the data distribution conditioned on $y$. Formally, we think of $y$ to live in a space $\mathcal{Y}$. When $y$ corresponds to a text-prompt, for example, $\mathcal{Y}$ would likely be some continuous space like $\mathbb{R}^{d_{y}}$. When $y$ corresponds to some discrete class label, $\mathcal{Y}$ would be discrete. In the lab, we will work with the MNIST dataset, in which case we will take $\mathcal{Y}=\{0,1,\dots,9\}$ to correspond to the identities of handwritten digits.

To avoid a notation and terminology clash with the use of the word "conditional" to refer to conditioning on $z\sim p_{\rm{data}}$ (conditional probability path/vector field), we will make use of the term guided to refer specifically to conditioning on $y$.

###### Remark 24 (Guided vs. Conditional Terminology)

In these notes, we opt to use the term guided in place of conditional to refer to the act of conditioning on $y$. Here, we will refer to e.g., a guided vector field $u^{\text{target}}_{t}(x|y)$ and a conditional vector field $u^{\text{target}}_{t}(x|z)$. This terminology is consistent with other works such as \[[15](#bib.bibx15)\].

The goal of guided generative modeling is thus to be able to sample from $p_{\rm{data}}(x|y)$ for any such $y$. In the language of flow and score matching, and in which our generative models correspond to the simulation of ordinary and stochastic differential equations, this can be phrased as follows.

###### Key Idea 5 (Guided Generative Model)

We define a guided diffusion model to consist of a guided vector field $u_{t}^{\theta}(\cdot|y)$, parameterized by some neural network, and a time-dependent diffusion coefficient $\sigma_{t}$, together given by

$$
\displaystyle\,u^{\theta}:\mathbb{R}^{d}\times\mathcal{Y}\times[0,1]\to\mathbb{R}^{d},\,\,(x,y,t)\mapsto u_{t}^{\theta}(x|y)
$$
 
$$
\displaystyle\,\sigma_{t}:[0,1]\to[0,\infty),\,\,t\mapsto\sigma_{t}
$$

Notice the difference from summary 7: we are additionally guiding $u_{t}^{\theta}$ with the input $y\in\mathcal{Y}$. For any such $y\in\mathbb{R}^{d_{y}}$, samples may then be generated from such a model as follows:

$$
\displaystyle\textbf{Initialization:}\quad X_{0}
$$
 
$$
\displaystyle\sim p_{\rm{init}}\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Initialize with simple distribution (such as a Gaussian)}
$$
 
$$
\displaystyle\textbf{Simulation:}\quad\mathrm{d}X_{t}
$$
 
$$
\displaystyle=u_{t}^{\theta}(X_{t}|y)\mathrm{d}t+\sigma_{t}\mathrm{d}W_{t}\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Simulate SDE from $t=0$ to $t=1$.}
$$
 
$$
\displaystyle\textbf{Goal:}\quad X_{1}
$$
 
$$
\displaystyle\sim p_{\rm{data}}(\cdot|y)\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Goal is for $X_{1}$ to be distributed like $p_{\rm{data}}(\cdot|y)$.}
$$

When $\sigma_{t}=0$, we say that such a model is a guided flow model.

#### 5.1.1 Guidance for Flow Models

If we imagine fixing our choice of $y$, and take our data distribution as $p_{\text{data}}(x|y)$, then we have recovered the unguided generative problem, and can accordingly construct a generative model using the conditional flow matching objective, viz.,

$$
\mathbb{E}_{z\sim p_{\text{data}}(\cdot|y),x\sim p_{t}(\cdot|z)}\lVert u_{t}^{\theta}(x|y)-u^{\text{target}}_{t}(x|z)\rVert^{2}.
$$

Note that the label $y$ does not affect the conditional probability path $p_{t}(\cdot|z)$ or the conditional vector field $u^{\text{target}}_{t}(x|z)$ (although in principle, we could make it dependent). Expanding the expectation over all such choices of $y$, and over all times $t\in\text{Unif}[0,1)$, we thus obtain a guided conditional flow matching objective

$$
\mathcal{L}_{\text{CFM}}^{\text{guided}}(\theta)=\mathbb{E}_{(z,y)\sim p_{\text{data}}(z,y),\,t\sim\text{Unif}[0,1),\,x\sim p_{t}(\cdot|z)}\lVert u_{t}^{\theta}(x|y)-u^{\text{target}}_{t}(x|z)\rVert^{2}.
$$

One of the main differences between the guided objective in eq.˜64 and the unguided objective from eq.˜44 is that here we are sampling $(z,y)\sim p_{\rm{data}}$ rather than just $z\sim p_{\rm{data}}$. The reason is that our data distribution is now, in principle, a joint distribution over e.g., both images $z$ and text prompts $y$. In practice, this means that a PyTorch implementation of eq.˜64 would involve a dataloader which returned batches of both $z$ and $y$. The above procedure leads to a faithful generation procedure of $p_{\rm{data}}(\cdot|y)$.

##### Classifier-Free Guidance.

While the above conditional training procedure is theoretically valid, it was soon empirically realized that images samples with this procedure did not fit well enough to the desired label $y$. It was discovered that perceptual quality is increased when the effect of the guidance variable $y$ is artificially reinforced. This insight was distilled into a technique known as classifier-free guidance that is widely used in the context of state-of-the-art diffusion models, and which we discuss next. For simplicity, we will focus here on the case of Gaussian probability paths. Recall from eq.˜16 that a Gaussian conditional probability path is given by

$$
\displaystyle p_{t}(\cdot|z)
$$
 
$$
\displaystyle=\mathcal{N}(\alpha_{t}z,\beta_{t}^{2}I_{d})
$$

where the noise schedulers $\alpha_{t}$ and $\beta_{t}$ are continuously differentiable, monotonic, and satisfy $\alpha_{0}=\beta_{1}=0$ and $\alpha_{1}=\beta_{0}=1$. To gain intuition for classifier-free guidance, we can use proposition˜1 to rewrite the guided vector field $u^{\text{target}}_{t}(x|y)$ in the following form using the guided score function $\nabla\log p_{t}(x|y)$

$$
u^{\text{target}}_{t}(x|y)=a_{t}x+b_{t}\nabla\log p_{t}(x|y),
$$

where

$$
(a_{t},b_{t})=\left(\frac{\dot{\alpha}_{t}}{\alpha_{t}},\frac{\dot{\alpha}_{t}\beta_{t}^{2}-\dot{\beta}_{t}\beta_{t}\alpha_{t}}{\alpha_{t}}\right).
$$

However, notice that by Bayes’ rule, we can rewrite the guided score as

$$
\nabla\log p_{t}(x|y)=\nabla\log\left(\frac{p_{t}(x)p_{t}(y|x)}{p_{t}(y)}\right)=\nabla\log p_{t}(x)+\nabla\log p_{t}(y|x),
$$

where we used that the gradient $\nabla$ is taken with respect to the variable $x$, so that $\nabla\log p_{t}(y)=0$. We may thus rewrite

$$
\displaystyle u^{\text{target}}_{t}(x|y)=a_{t}x+b_{t}(\nabla\log p_{t}(x)+\nabla\log p_{t}(y|x))=u^{\text{target}}_{t}(x)+b_{t}\nabla\log p_{t}(x|y).
$$

Notice the shape of the above equation: The guided vector field $u^{\text{target}}_{t}(x|y)$ is a sum of the unguided vector field *plus* a guided score $\nabla\log p_{t}(x|y)$. As people observed that their image $x$ did not fit their prompt $y$ well enough, it was a natural idea to scale up the contribution of the $\nabla\log p_{t}(y|x)$ term, yielding

$$
\displaystyle\tilde{u}_{t}(x|y)=u^{\text{target}}_{t}(x)+wb_{t}\nabla\log p_{t}(y|x),
$$

where $w>1$ is known as the guidance scale. Note that this is a heuristic: for $w\neq 1$, it holds that $\tilde{u}_{t}(x|y)\neq u^{\text{target}}_{t}(x|y)$, i.e. therefore not the true, guided vector field. However, empirical results have shown to yield preferable results (when $w>1$).

###### Remark 25 (Where is the classifier?)

The term $\log p_{t}(y|x)$ can be considered as a sort of classifier of noised data (i.e. it gives the likelihoods of $y$ given $x$). In fact, early works in diffusion trained actual classifiers and used them to the guide via the above procedure. This leads to classifier guidance \[[5](#bib.bibx5), [30](#bib.bibx30)\]. As it has been largely superseded by classifier-free guidance, we do not consider it here.

We may again apply the equality

$$
\nabla\log p_{t}(x|y)=\nabla\log p_{t}(x)+\nabla\log p_{t}(y|x)
$$

to obtain

$$
\displaystyle\tilde{u}_{t}(x|y)
$$
 
$$
\displaystyle=u^{\text{target}}_{t}(x)+wb_{t}\nabla\log p_{t}(y|x)
$$
 
$$
\displaystyle=u^{\text{target}}_{t}(x)+wb_{t}(\nabla\log p_{t}(x|y)-\nabla\log p_{t}(x))
$$
 
$$
\displaystyle=u^{\text{target}}_{t}(x)-(wa_{t}x+wb_{t}\nabla\log p_{t}(x))+(wa_{t}x+wb_{t}\nabla\log p_{t}(x|y))
$$
 
$$
\displaystyle=(1-w)u^{\text{target}}_{t}(x)+wu^{\text{target}}_{t}(x|y).
$$

We may therefore express the scaled guided vector field $\tilde{u}_{t}(x|y)$ as the linear combination of the unguided vector field $u^{\text{target}}_{t}(x)$ with the guided vector field $u^{\text{target}}_{t}(x|y)$. The idea might then to to train both an unguided $u^{\text{target}}_{t}(x)$ (using e.g., eq.˜44) as well as a guided $u^{\text{target}}_{t}(x|y)$ (using e.g., eq.˜64), and then combine them at inference time to obtain $\tilde{u}_{t}(x|y)$. "But wait!", you might ask, "wouldn’t we need to train two models then!?". It turns out we do can train both in model: we may thus augment our label set with a new, additional $\varnothing$ label that denotes the absence of conditioning. We can then treat $u^{\text{target}}_{t}(x)=u^{\text{target}}_{t}(x|\varnothing)$. With that, we do not need to train a separate model to reinforce the effect of a hypothetical classifier. This approach of training a conditional and unconditional model in one (and subsequently reinforcing the conditioning) is known as classifier-free guidance (CFG) \[[10](#bib.bibx10)\].

###### Remark 26 (Derivation for general probability paths)

Note that the construction

$$
\tilde{u}_{t}(x|y)=(1-w)u^{\text{target}}_{t}(x)+wu^{\text{target}}_{t}(x|y),
$$

is equally valid for any choice probability path, not just a Gaussian one. When $w=1$, it is straightforward to verify that $\tilde{u}_{t}(x|y)=u^{\text{target}}_{t}(x|y)$. Our derivation using Gaussian paths was simply to illustrate the intuition behind the construction, and in particular of amplifying the contribution of a “classifier” $\nabla\log p_{t}(y|x)$.

##### Training and Context-Free Guidance.

We must now amend the guided conditional flow matching objective from eq.˜64 to account for the possibility of $y=\varnothing$. The challenge is that when sampling $(z,y)\sim p_{\rm{data}}$, we will never obtain $y=\varnothing$. It follows that we must introduce the possibility of $y=\varnothing$ artificially. To do so, we will define some hyperparameter $\eta$ to be the probability that we discard the original label $y$, and replace it with $\varnothing$. We thus arrive at our CFG conditional flow matching training objective

$$
\displaystyle\mathcal{L}_{\text{CFM}}^{\text{CFG}}(\theta)
$$
 
$$
\displaystyle=\,\,\mathbb{E}_{\square}\lVert u_{t}^{\theta}(x|y)-u^{\text{target}}_{t}(x|z)\rVert^{2}
$$
 
$$
\displaystyle\square
$$
 
$$
\displaystyle=(z,y)\sim p_{\text{data}}(z,y),\,t\sim\text{Unif}[0,1),\,x\sim p_{t}(\cdot|z),\text{replace }y=\varnothing\text{ with prob. }\eta
$$

We summarize our findings below.

###### Summary 27 (Classifier-Free Guidance for Flow Models)

Given the unguided marginal vector field $u^{\text{target}}_{t}(x|\varnothing)$, the guided marginal vector field $u^{\text{target}}_{t}(x|y)$, and a guidance scale $w>1$, we define the classifier-free guided vector field $\tilde{u}_{t}(x|y)$ by

$$
\tilde{u}_{t}(x|y)=(1-w)u^{\text{target}}_{t}(x|\varnothing)+wu^{\text{target}}_{t}(x|y).
$$

By approximating $u^{\text{target}}_{t}(x|\varnothing)$ and $u^{\text{target}}_{t}(x|y)$ using the same neural network, we may leverage the following classifier-free guidance CFM (CFG-CFM) objective, given by

$$
\displaystyle\mathcal{L}_{\text{CFM}}^{\text{CFG}}(\theta)
$$
 
$$
\displaystyle=\,\,\mathbb{E}_{\square}\lVert u_{t}^{\theta}(x|y)-u^{\text{target}}_{t}(x|z)\rVert^{2}
$$
 
$$
\displaystyle\square
$$
 
$$
\displaystyle=(z,y)\sim p_{\text{data}}(z,y),\,t\sim\text{Unif}[0,1),\,x\sim p_{t}(\cdot|z),\text{replace }y=\varnothing\text{ with prob. }\eta
$$

In plain English, $\mathcal{L}_{\text{CFM}}^{\text{CFG}}$ might be approximated by

$$
\displaystyle(z,y)
$$
 
$$
\displaystyle\sim p_{\rm{data}}(z,y)\quad\quad\quad\quad
$$
 
$$
\displaystyle\blacktriangleright\quad\text{Sample $(z,y)$ from data distribution.}
$$
 
$$
\displaystyle t
$$
 
$$
\displaystyle\sim\text{Unif}[0,1)\quad\quad\quad\quad
$$
 
$$
\displaystyle\blacktriangleright\quad\text{Sample $t$ uniformly on $[0,1)$.}
$$
 
$$
\displaystyle x
$$
 
$$
\displaystyle\sim p_{t}(x|z)\quad\quad\quad\quad
$$
 
$$
\displaystyle\blacktriangleright\quad\text{Sample $x$ from the conditional probability path $p_{t}(x|z)$.}
$$
 
$$
\displaystyle\,\eta,\,y\leftarrow\varnothing\quad\quad\quad\quad
$$
 
$$
\displaystyle\blacktriangleright\quad\text{Replace $y$ with $\varnothing$ with probability $\eta$.}
$$
 
$$
\displaystyle\widehat{\mathcal{L}_{\text{CFM}}^{\text{CFG}}(\theta)}
$$
 
$$
\displaystyle=\lVert u_{t}^{\theta}(x|y)-u^{\text{target}}_{t}(x|z)\rVert^{2}\quad\quad\quad\quad
$$
 
$$
\displaystyle\blacktriangleright\quad\text{Regress model against conditional vector field.}
$$

Above, we made use multiple times of the fact that $u^{\text{target}}_{t}(x|z)=u^{\text{target}}_{t}(x|z,y)$. At inference time, for a fixed choice of $y$, we may sample via

$$
\displaystyle\textbf{Initialization:}\quad X_{0}
$$
 
$$
\displaystyle\sim p_{\rm{init}}(x)\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Initialize with simple distribution (such as a Gaussian)}
$$
 
$$
\displaystyle\textbf{Simulation:}\quad\mathrm{d}X_{t}
$$
 
$$
\displaystyle=\tilde{u}_{t}^{\theta}(X_{t}|y)\mathrm{d}t\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Simulate ODE from $t=0$ to $t=1$.}
$$
 
$$
\displaystyle\textbf{Samples:}\quad X_{1}
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Goal is for $X_{1}$ to adhere to the guiding variable $y$.}
$$

Note that the distribution of $X_{1}$ is not necessarily aligned with $X_{1}\sim p_{\rm{data}}(\cdot|y)$ anymore if we use a weight $w>1$. However, empirically, this shows better alignment with conditioning.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.02070/assets/figures/salimans_cfg.png)

Figure 11: The effect of classifier guidance. The prompt here is the "class" chosen to be "Corgi" (a specific type of dog). Left: samples generated with no guidance (i.e., w = 1 w=1 ). Right: samples generated with classifier guidance and 4 w=4. As shown, classifier-free guidance improves the similarity to the prompt. Figure taken from \[ 10 \].

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.02070/assets/figures/guidance.png)

Figure 12: The effect of classifier-free guidance applied at various guidance scales for the MNIST dataset of hand-written digits. Left: Guidance scale set to w = 1.0 w=1.0. Middle: Guidance scale set to 2.0 w=2.0. Right: Guidance scale set to 4.0 w=4.0. You will generate a similar image yourself in the lab three!

In fig.˜11, we illustrate class-based classifier-free guidance on 128x128 ImageNet, as in \[[10](#bib.bibx10)\]. Similarly, in fig.˜12, we visualize the affect of various guidance scales $w$ when applying classifier-free guidance to sampling from the MNIST dataset of handwritten digits.

Algorithm 5 Classifier-free guidance training for Gaussian probability path $p_{t}(x|z)=\mathcal{N}(x;\alpha_{t}z,\beta_{t}^{2}I_{d})$

 Paired dataset $(z,y)\sim p_{\rm{data}}$, neural network $u_{t}^{\theta}$

 for each mini-batch of data do

  Sample a data example $(z,y)$ from the dataset.

  Sample a random time $t\sim\text{Unif}_{[0,1]}$.

  Sample noise $\epsilon\sim\mathcal{N}(0,I_{d})$

  Set $x=\alpha_{t}z+\beta_{t}\epsilon$

  With probability $p$ drop label: $y\leftarrow\varnothing$

  Compute loss 
$$
\displaystyle\mathcal{L}(\theta)=
$$
 
$$
\displaystyle\|u_{t}^{\theta}(x|y)-(\dot{\alpha}_{t}\epsilon+\dot{\beta}_{t}z)\|^{2}
$$

  Update the model parameters $\theta$ via gradient descent on $\mathcal{L}(\theta)$.

 end for

#### 5.1.2 Guidance for Diffusion Models

In this section we extend the reasoning of the previous section to diffusion models. First, in the same way that we obtained eq.˜64, we may generalize the conditional score matching loss eq.˜61 to obtain the guided conditional score matching objective

$$
\displaystyle\mathcal{L}_{\text{CSM}}^{\text{guided}}(\theta)
$$
 
$$
\displaystyle=\mathbb{E}_{\square}[\|s_{t}^{\theta}(x|y)-\nabla\log p_{t}(x|z)\|^{2}]
$$
 
$$
\displaystyle\square
$$
 
$$
\displaystyle=(z,y)\sim p_{\rm{data}}(z,y),\,t\sim\text{Unif},\,x\sim p_{t}(\cdot|z).
$$

A guided score network $s_{t}^{\theta}(x|y)$ trained with eq.˜73 might then be combined with the guided vector field $u_{t}^{\theta}(x|y)$ to simulate the SDE

$$
X_{0}\sim p_{\rm{init}},\quad\mathrm{d}X_{t}=\left[u_{t}^{\theta}(X_{t}|y)+\frac{\sigma_{t}^{2}}{2}s_{t}^{\theta}(X_{t}|y)\right]\mathrm{d}t+\sigma_{t}\mathrm{d}W_{t}.
$$

##### Classifier-Free Guidance.

We now extend the classifier-free guidance construction to the diffusion setting. By Bayes’ rule (see eq.˜67),

$$
\nabla\log p_{t}(x|y)=\nabla\log p_{t}(x)+\nabla\log p_{t}(y|x),
$$

so that for guidance scale $w>1$ we may define

$$
\displaystyle\tilde{s}_{t}(x|y)
$$
 
$$
\displaystyle=\nabla\log p_{t}(x)+w\nabla\log p_{t}(y|x)
$$
 
$$
\displaystyle=\nabla\log p_{t}(x)+w(\nabla\log p_{t}(x|y)-\nabla\log p_{t}(x))
$$
 
$$
\displaystyle=(1-w)\nabla\log p_{t}(x)+w\nabla\log p_{t}(x|y)
$$
 
$$
\displaystyle=(1-w)\nabla\log p_{t}(x|\varnothing)+w\nabla\log p_{t}(x|y)
$$

We thus arrive at the CFG-compatible (that is, accounting for the possibility of $\varnothing$) objective

$$
\displaystyle\mathcal{L}_{\text{DSM}}^{\text{CFG}}(\theta)
$$
 
$$
\displaystyle=\,\,\mathbb{E}_{\square}\lVert s_{t}^{\theta}(x|y)-\nabla\log p_{t}(x|z)\rVert^{2}
$$
 
$$
\displaystyle\square
$$
 
$$
\displaystyle=(z,y)\sim p_{\text{data}}(z,y),\,t\sim\text{Unif}[0,1),\,x\sim p_{t}(\cdot|z),\,\text{replace }y=\varnothing\text{ with prob. }\eta,
$$

where $\eta$ is a hyperparameter (the probability of replacing $y$ with $\varnothing$). We will refer $\mathcal{L}_{\text{CSM}}^{\text{CFG}}(\theta)$ as the guided conditional score matching objective. We recap as follows

###### Summary 28 (Classifier-Free Guidance for Diffusions)

Given the unguided marginal score $\nabla\log p_{t}(x|\varnothing)$, the guided marginal score field $\nabla\log p_{t}(x|y)$, and a guidance scale $w>1$, we define the classifier-free guided score $\tilde{s}_{t}(x|y)$ by

$$
\tilde{s}_{t}(x|y)=(1-w)\nabla\log p_{t}(x|\varnothing)+w\nabla\log p_{t}(x|y).
$$

By approximating $\nabla\log p_{t}(x|\varnothing)$ and $\nabla\log p_{t}(x|y)$ using the same neural network $s_{t}^{\theta}(x|y)$, we may leverage the following classifier-free guidance CSM (CFG-CSM) objective, given by

$$
\displaystyle\mathcal{L}_{\text{CSM}}^{\text{CFG}}(\theta)
$$
 
$$
\displaystyle=\,\,\mathbb{E}_{\square}\lVert s_{t}^{\theta}(x|(1-\xi)y+\xi\varnothing)-\nabla\log p_{t}(x|z)\rVert^{2}
$$
 
$$
\displaystyle\square
$$
 
$$
\displaystyle=(z,y)\sim p_{\text{data}}(z,y),\,t\sim\text{Unif}[0,1),\,x\sim p_{t}(\cdot|z),\,\text{replace }y=\varnothing\text{ with prob. }\eta
$$

In plain English, $\mathcal{L}_{\text{DSM}}^{\text{CFG}}$ might be approximated by

$$
\displaystyle(z,y)
$$
 
$$
\displaystyle\sim p_{\rm{data}}(z,y)\quad\quad\quad\quad
$$
 
$$
\displaystyle\blacktriangleright\quad\text{Sample $(z,y)$ from data distribution.}
$$
 
$$
\displaystyle t
$$
 
$$
\displaystyle\sim\text{Unif}[0,1)\quad\quad\quad\quad
$$
 
$$
\displaystyle\blacktriangleright\quad\text{Sample $t$ uniformly on $[0,1)$.}
$$
 
$$
\displaystyle x
$$
 
$$
\displaystyle\sim p_{t}(x|z,y)\quad\quad\quad\quad
$$
 
$$
\displaystyle\blacktriangleright\quad\text{Sample $x$ from cond. path $p_{t}(x|z)$.}
$$
 
$$
\displaystyle\,\eta,\,y\leftarrow\varnothing\quad\quad\quad\quad
$$
 
$$
\displaystyle\blacktriangleright\quad\text{Replace $y$ with $\varnothing$ with probability $\eta$.}
$$
 
$$
\displaystyle\widehat{\mathcal{L}_{\text{DSM}}^{\text{CFG}}}(\theta)
$$
 
$$
\displaystyle=\lVert s_{t}^{\theta}(x|y)-\nabla\log p_{t}(x|z)\rVert^{2}\quad\quad\quad\quad
$$
 
$$
\displaystyle\blacktriangleright\quad\text{Regress model against conditional score.}
$$

At inference time, for a fixed choice of $w>1$, we may combine $s_{t}^{\theta}(x|y)$ with a guided vector field $u_{t}^{\theta}(x|y)$ and define

$$
\displaystyle\tilde{s}^{\theta}_{t}(x|y)
$$
 
$$
\displaystyle=(1-w)s_{t}^{\theta}(x|\varnothing)+ws_{t}^{\theta}(x|y),
$$
$$
\displaystyle\tilde{u}^{\theta}_{t}(x|y)
$$
 
$$
\displaystyle=(1-w)u_{t}^{\theta}(x|\varnothing)+wu_{t}^{\theta}(x|y).
$$

Then we may sample via

$$
\displaystyle\textbf{Initialization:}\quad X_{0}
$$
 
$$
\displaystyle\sim p_{\rm{init}}(x)\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Initialize with simple distribution (such as a Gaussian)}
$$
 
$$
\displaystyle\textbf{Simulation:}\quad\mathrm{d}X_{t}
$$
 
$$
\displaystyle=\left[\tilde{u}_{t}^{\theta}(X_{t}|y)+\frac{\sigma_{t}^{2}}{2}\tilde{s}_{t}^{\theta}(X_{t}|y)\right]\mathrm{d}t+\sigma_{t}\mathrm{d}W_{t}\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Simulate SDE from $t=0$ to $t=1$.}
$$
 
$$
\displaystyle\textbf{Samples:}\quad X_{1}
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Goal is for $X_{1}$ to adhere to the guiding variable $y$.}
$$

### 5.2 Neural network architectures

We next discuss the design of neural networks for flow and diffusion models. Specifically, we answer the question of how to construct a neural network architecture that represents the (guided) vector field $u_{t}^{\theta}(x|y)$ with parameters $\theta$. Note that the neural network must have 3 inputs - a vector $x\in\mathbb{R}^{d}$, a conditioning variable $y\in\mathcal{Y}$, and a time value $t\in[0,1]$ - and one output - a vector $u_{t}^{\theta}(x|y)\in\mathbb{R}^{d}$. For low-dimensional distributions (e.g. the toy distributions we have seen in previous sections), it is sufficient to parameterize $u_{t}^{\theta}(x|y)$ as a multi-layer perceptron (MLP), otherwise known as a fully connected neural network. That is, in this simple setting, a forward pass through $u_{t}^{\theta}(x|y)$ would involve concatenating our input $x$, $y$, and $t$, and passing them through an MLP. However, for complex, high-dimensional distributions, such as those over images, videos, and proteins, an MLP is rarely sufficient, and it is common to use special, application-specific architectures. For the remainder of this section, we will consider the case of images (and by extension, videos), and discuss two common architectures: the U-Net \[[25](#bib.bibx25)\], and the diffusion transformer (DiT).

#### 5.2.1 U-Nets and Diffusion Transformers

Before we dive into the specifics of these architectures, let us recall from the introduction that an image is simply a vector $x\in\mathbb{R}^{C_{\text{image}}\times H\times W}$. Here $C_{\text{image}}$ denotes the number of channels (an RGB image typically would have $C_{\text{input}}=3$ color channels), $H$ denotes the height of the image in pixels, and $W$ denotes the width of the image in pictures.

##### U-Nets.

The U-Net architecture \[[25](#bib.bibx25)\] is a specific type of convolutional neural network. Originally designed for image segmentation, its crucial feature is that both its input and its output have the shape of images (possibly with a different number of channels). This makes it ideal to parameterize a vector field $x\mapsto u_{t}^{\theta}(x|y)$ as for fixed $y,t$ its input has the shape of an image and its output does, too. Therefore, U-Net were widely used in the development of diffusion models. A U-Net consists of a series of encoders $\mathcal{E}_{i}$, and a corresponding sequence of decoders $\mathcal{D}_{i}$, along with a latent processing block in between, which we shall refer to as a midcoder (midcoder is a term is not used in the literature usually). For sake of example, let us walk through the path taken by an image $x_{t}\in\mathbb{R}^{3\times 256\times 256}$ (we have taken $(C_{\text{input}},H,W)=(3,256,256)$) as it is processed by the U-Net:

$$
\displaystyle x^{\text{input}}_{t}
$$
 
$$
\displaystyle\in\mathbb{R}^{3\times 256\times 256}\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Input to the U-Net.}
$$
 
$$
\displaystyle x^{\text{latent}}_{t}=\mathcal{E}(x^{\text{input}}_{t})
$$
 
$$
\displaystyle\in\mathbb{R}^{512\times 32\times 32}\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Pass through encoders to obtain latent.}
$$
 
$$
\displaystyle x^{\text{latent}}_{t}=\mathcal{M}(x^{\text{latent}}_{t})
$$
 
$$
\displaystyle\in\mathbb{R}^{512\times 32\times 32}\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Pass latent through midcoder.}
$$
 
$$
\displaystyle x^{\text{output}}_{t}=\mathcal{D}(x^{\text{latent}}_{t})
$$
 
$$
\displaystyle\in\mathbb{R}^{3\times 256\times 256}\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Pass through decoders to obtain output.}
$$

Notice that as the input passes through the encoders, the number of channels in its representation increases, while the height and width of the images are decreased. Both the encoder and the decoder usually consist of a series of convolutional layers (with activation functions, pooling operations, etc. in between). Not shown above are two points: First, the input $x^{\text{input}}_{t}\in\mathbb{R}^{3\times 256\times 256}$ is often fed into an initial pre-encoding block to increase the number of channels before being fed into the first encoder block. Second, the encoders and decoders are often connected by residual connections. The complete picture is shown in fig.˜13.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.02070/assets/figures/unet.png)

Figure 13: The simplified U-Net architecture used in lab three.

At a high level, most U-Nets involve some variant of what is described above. However, certain of the design choices described above may well differ from various implementations in practice. In particular, we opt above for a purely-convolutional architecture whereas it is common to include attention layers as well throughout the encoders and decoders. The U-Net derives its name from the “U”-like shape formed by its encoders and decoders (see fig.˜13).

##### Diffusion Transformers.

One alternative to U-Nets are diffusion transformers (DiTs), which dispense with convolutions and purely use attention \[[35](#bib.bibx35), [19](#bib.bibx19)\]. Diffusion transformers are based on vision transformers (ViTs), in which the big idea is essentially to divide up an image into patches, embed each of these patches, and then attend between the patches \[[6](#bib.bibx6)\]. Stable Diffusion 3, trained with conditional flow matching, parameterizes the velocity field $u_{t}^{\theta}(x)$ as a modified DiT, as we discuss later in section˜5.3 \[[7](#bib.bibx7)\].

###### Remark 29 (Working in Latent Space)

A common problem for large-scale applications is that the data is so high-dimensional that it consumes too much memory. For example, we might want to generate a high resolution image of $1000\times 10000$ pixels leading to $1$ million (!) dimensions. To reduce memory usage, a common design pattern is to work in a latent space that can be considered a compressed version of our data at lower resolution. Specifically, the usual approach is to combine a flow or diffusion model with a (variational) autoencoder \[[24](#bib.bibx24)\]. In this case, one first encodes the training dataset in the latent space via an autoencoder, and then training the flow or diffusion model in the latent space. Sampling is performed by first sampling in the latent space using the trained flow or diffusion model, and then decoding of the output via the decoder. Intuitively, a well-trained autoencoder can be thought of as filtering out semantically meaningless details, allowing the generative model to “focus” on important, perceptually relevant features \[[24](#bib.bibx24)\]. By now, nearly all state-of-the-art approaches to image and video generation involve training a flow or diffusion model in the latent space of an autoencoder - so called latent diffusion models \[[24](#bib.bibx24), [34](#bib.bibx34)\]. However, it is important to note: one also needs to train the autoencoder before training the diffusion models. Crucially, performance now depends also on how good the autoencoder compresses images into latent space and recovers aesthetically pleasing images.

#### 5.2.2 Encoding the Guiding Variable.

Up until this point, we have glossed over how exactly the guiding (conditioning) variable $y$ is fed into the neural network $u_{t}^{\theta}(x|y)$. Broadly, this process can be decomposed into two steps: embedding the raw input $y_{\text{raw}}$ (e.g., the text prompt “a cat playing a trumpet, photorealistic”) into some vector-valued input $y$, and feeding the resulting $y$ into the actual model. We now proceed to describe each step in greater detail.

##### Embedding Raw Input.

Here, we’ll consider two cases: (1) where $y_{\text{raw}}$ is a discrete class-label, and (2) where $y_{\text{raw}}$ is a text-prompt. When $y_{\text{raw}}\in\mathcal{Y}\triangleq\{0,\dots,N\}$ is just a class label, then it is often easiest to simply learn a separate embedding vector for each of the $N+1$ possible values of $y_{\text{raw}}$, and set $y$ to this embedding vector. One would consider the parameters of these embeddings to be included in the parameters of $u_{t}^{\theta}(x|y)$, and would therefore learn these during training. When $y_{\text{raw}}$ is a text-prompt, the situation is more complex, and approaches largely rely on frozen, pre-trained models. Such models are trained to embed a discrete text input into a continuous vector that captures the relevant information. One such model is known as CLIP (Contrastive Language-Image Pre-training). CLIP is trained to learn a shared embedding space for both images and text-prompts, using a training loss designed to encourage image embeddings to be close to their corresponding prompts, while being farther from the embeddings of other images and prompts \[[22](#bib.bibx22)\]. We might therefore take $y=\text{CLIP}(y_{\text{raw}})\in\mathbb{R}^{d_{\text{CLIP}}}$ to be the embedding produced by a frozen, pre-trained CLIP model. In certain cases, it may be undesirable to compress the entire sequence into a single representation. In this case, one might additionally consider embedding the prompt using a pre-trained transformer so as to obtain a sequence of embeddings. It is also common to combine multiple such pretrained embeddings when conditioning so as to simultaneously reap the benefits of each model \[[7](#bib.bibx7), [21](#bib.bibx21)\].

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.02070/assets/figures/dit.png)

Figure 14: Left: An overview of the diffusion transformer architecture, taken from \[ 19 \]. Right: A schematic of the contrastive CLIP loss, in which a shared image-text embedding space is learned, taken from 22.

##### Feeding in the Embedding.

Suppose now that we have obtained our embedding vector $y\in\mathbb{R}^{d_{y}}$. Now what? The answer varies, but usually it is some variant of the following: feed it individually into every sub-component of the architecture for images. Let us briefly describe how this is accomplished in the U-Net implementation used in lab three, as depicted in fig.˜13. At some intermediate point within the network, we would like to inject information from $y\in\mathbb{R}^{d_{y}}$ into the current activation $x^{\text{intermediate}}_{t}\in\mathbb{R}^{C\times H\times W}$. We might do so using the procedure below, given in PyTorch-esque pseudocode.

$$
\displaystyle y
$$
 
$$
\displaystyle=\text{MLP}(y)\in\mathbb{R}^{C}\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Map $y$ from $\mathbb{R}^{d_{y}}$ to $\mathbb{R}^{C}$.}
$$
 
$$
\displaystyle y
$$
 
$$
\displaystyle=\text{reshape}(y)\in\mathbb{R}^{C\times 1\times 1}\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Reshape $y$ to ``look'' like an image.}
$$
 
$$
\displaystyle x^{\text{intermediate}}_{t}
$$
 
$$
\displaystyle=\text{broadcast\_add}(x^{\text{intermediate}}_{t},y)\in\mathbb{R}^{C\times H\times W}\quad
$$
 
$$
\displaystyle\blacktriangleright\,\,\text{Add $y$ to $x^{\text{intermediate}}_{t}$ pointwise.}
$$

One exception to this simple-pointwise conditioning scheme is when we have a sequence of embeddings as produced by some pretrained language model. In this case, we might consider using some sort of cross-attention scheme between our image (suitably patchified) and the tokens of the embedded sequence. We will see multiple examples of this in section˜5.3.

### 5.3 A Survey of Large-Scale Image and Video Models

We conclude this section by briefly examining two large-scale generative models: Stable Diffusion 3 for image generation and Meta’s Movie Gen Video for video generation \[[7](#bib.bibx7), [21](#bib.bibx21)\]. As you will see, these models use the techniques we have described in this work along with additional architectural enhancements to both scale and accommodate richly structured conditioning modalities, such as text-based input.

#### 5.3.1 Stable Diffusion 3

Stable Diffusion is a series of state-of-the-art image generation models. These models were among the first to use large-scale latent diffusion models for image generation. If you have not done so, we highly recommend testing it for yourself online ([https://stability.ai/news/stable-diffusion-3](https://stability.ai/news/stable-diffusion-3)).

Stable Diffusion 3 uses the same conditional flow matching objective that we study in this work (see algorithm˜5).[^4] As outlined in their paper, they extensively tested various flow and diffusion alternatives and found flow matching to perform best. For training, it uses classifier-free guidance training (with dropping class labels) as outlined above. Further, Stable Diffusion 3 follows the approach outlined in section˜5.2 by training within the latent space of a pre-trained autoencoder. Training a good autoencoder was a big contribution of the first stable diffusion papers.

To enhance text conditioning, Stable Diffusion 3 makes use of both 3 different types of text embeddings (including CLIP embeddings as well as the sequential outputs produced by a pretrained instance of the encoder of Google’s T5-XXL \[[23](#bib.bibx23)\], and similar to approaches taken in \[[3](#bib.bibx3), [26](#bib.bibx26)\]). Whereas CLIP embeddings provide a coarse, overarching embedding of the input text, the T5 embeddings provide a more granular level of context, allowing for the possibility of the model attending to particular elements of the conditioning text. To accommodate these sequential context embeddings, the authors then propose to extend the diffusion transformer to attend not just to patches of the image, but to the text embeddings as well, thereby extending the conditioning capacity from the class-based scheme originally proposed for DiT to sequential context embeddings. This proposed modified DiT is referred to as a multi-modal DiT (MM-DiT), and is depicted in fig.˜15. Their final, largest model has 8 billion parameters. For sampling, they use $50$ steps (i.e. they have to evaluate the network $50$ times) using a Euler simulation scheme and a classifier-free guidance weight between $2.0$ - $5.0$.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.02070/assets/figures/mmdit.png)

Figure 15: The architecture of the multi-modal diffusion transformer (MM-DiT) proposed in \[ 7 \]. Figure also taken from.

#### 5.3.2 Meta Movie Gen Video

Next, we discuss Meta’s video generator, Movie Gen Video ([https://ai.meta.com/research/movie-gen/](https://ai.meta.com/research/movie-gen/)). As the data are not images but videos, the data $x$ lie in the space $\mathbb{R}^{T\times C\times H\times W}$ where $T$ represents the new temporal dimension (i.e. the number of frames). As we shall see, many of the design choices made in this video setting can be seen as adapting existing techniques (e.g., autoencoders, diffusion transformers, etc.) from the image setting to handle this extra temporal dimension.

Movie Gen Video utilizes the conditional flow matching objective with the same CondOT path (see algorithm˜5). Like Stable Diffusion 3, Movie Gen Video also operates in the latent space of frozen, pretrained autoencoder. Note that the autoencoder to reduce memory consumption is even more important for videos than for images - which is why most video generators right now are pretty limited in the length of the video they generate. Specifically, the authors propose to handle the added time dimension by introducing a temporal autoencoder (TAE) which maps a raw video $x_{t}^{\prime}\in\mathbb{R}^{T^{\prime}\times 3\times H\times W}$ to a latent $x_{t}\in\mathbb{R}^{T\times C\times H\times W}$, with $\tfrac{T^{\prime}}{T}=\tfrac{H^{\prime}}{H}=\tfrac{W^{\prime}}{W}=8$ \[[21](#bib.bibx21)\]. To accomodate long videos, a temporal tiling procedure is proposed by which the video is chopped up into pieces, each piece is encoder separately, and the latents are sticthed together \[[21](#bib.bibx21)\]. The model itself - that is, $u_{t}^{\theta}(x_{t})$ - is given by a DiT-like backbone in which $x_{t}$ is patchified along the time and space dimensions. The image patches are then passed through a transformer employing both self-attention among the image patches, and cross-attention with language model embeddings, similar to the MM-DiT employed by Stable Diffusion 3. For text conditioning, Movie Gen Video employs three types of text embeddings: UL2 embeddings, for granular, text-based reasoning \[[33](#bib.bibx33)\], ByT5 embeddings, for attending to character-level details (for e.g., prompts explicitly requesting specific text to be present) \[[36](#bib.bibx36)\], and MetaCLIP embeddings, trained in a shared text-image embedding space \[[13](#bib.bibx13), [21](#bib.bibx21)\]. Their final, largest model has 30 billion parameters. For a significantly more detailed and expansive treatment, we encourage the reader to check out the Movie Gen technical report itself \[[21](#bib.bibx21)\].

## 6 Acknowledgements

This course would not have been possible without the generous support of many others. We would like to thank Tommi Jaakkola for serving as the advisor and faculty sponsor for this course, and for thoughtful feedback throughout the process. We would like to thank Christian Fiedler, Tim Griesbach, Benedikt Geiger, and Albrecht Holderrieth for value feedback on the lecture notes. Further, we thank Elaine Mello from MIT Open Learning for support with lecture recordings and Ashay Athalye from Students for Open and Universal Learning for helping to cut and process the videos. We would additionally like to thank Cameron Diao, Tally Portnoi, Andi Qu, Roger Trullo, Ádám Burián, Zewen Yang, and many others for their invaluable contributions to the labs. We would also like to thank Lisa Bella, Ellen Reid, and everyone else at MIT EECS for their generous support. Finally, we would like to thank all participants in the original course offering (MIT 6.S184/6.S975, taught over IAP 2025), as well as readers like you for your interest in this class. Thanks!

## References

## Appendix A A Reminder on Probability Theory

We present a brief overview of basic concepts from probability theory. This section was partially taken from \[[15](#bib.bibx15)\].

### A.1 Random vectors

Consider data in the $d$ -dimensional Euclidean space $x=(x^{1},\ldots,x^{d})\in\mathbb{R}^{d}$ with the standard Euclidean inner product $\left\langle x,y\right\rangle=\sum_{i=1}^{d}x^{i}y^{i}$ and norm $\left\|x\right\|=\sqrt{\left\langle x,x\right\rangle}$. We will consider random variables (RVs) $X\in\mathbb{R}^{d}$ with continuous probability density function (PDF), defined as a *continuous* function $p_{X}:\mathbb{R}^{d}\rightarrow\mathbb{R}_{\geq 0}$ providing event $A$ with probability

$$
{\mathbb{P}}(X\in A)=\int_{A}p_{X}(x)\mathrm{d}x,
$$

where $\int p_{X}(x)\mathrm{d}x=1$. By convention, we omit the integration interval when integrating over the whole space ($\int\equiv\int_{\mathbb{R}^{d}}$). To keep notation concise, we will refer to the PDF $p_{X_{t}}$ of RV $X_{t}$ as simply $p_{t}$. We will use the notation $X\sim p$ or $X\sim p(X)$ to indicate that $X$ is distributed according to $p$. One common PDF in generative modeling is the $d$ -dimensional isotropic Gaussian:

$$
{\mathcal{N}}(x;\mu,\sigma^{2}I)=(2\pi\sigma^{2})^{-\frac{d}{2}}\exp\left(-\frac{\left\|x-\mu\right\|_{2}^{2}}{2\sigma^{2}}\right),
$$

where $\mu\in\mathbb{R}^{d}$ and $\sigma\in\mathbb{R}_{>0}$ stand for the mean and the standard deviation of the distribution, respectively.

The expectation of a RV is the constant vector closest to $X$ in the least-squares sense:

$$
\mathbb{E}\left[X\right]=\operatorname*{arg\,min}_{z\in\mathbb{R}^{d}}\int\left\|x-z\right\|^{2}p_{X}(x)\mathrm{d}x=\int xp_{X}(x)\mathrm{d}x.
$$

One useful tool to compute the expectation of *functions of RVs* is the *law of the unconscious statistician*:

$$
\mathbb{E}\left[f(X)\right]=\int f(x)p_{X}(x)\mathrm{d}x.
$$

When necessary, we will indicate the random variables under expectation as $\mathbb{E}_{X}f(X)$.

### A.2 Conditional densities and expectations

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.02070/assets/figures/joints.png)

Figure 16: Joint PDF p X, Y p\_{X,Y} (in shades) and its marginals p\_{X} and p\_{Y} (in black lines). Figure from \[ 15 \]

Given two random variables $X,Y\in\mathbb{R}^{d}$, their joint PDF $p_{X,Y}(x,y)$ has marginals

$$
\int p_{X,Y}(x,y)\mathrm{d}y=p_{X}(x)\text{ and }\int p_{X,Y}(x,y)\mathrm{d}x=p_{Y}(y).
$$

See fig.˜16 for an illustration of the joint PDF of two RVs in $\mathbb{R}$ ($d=1$). The *conditional* PDF $p_{X|Y}$ describes the PDF of the random variable $X$ when conditioned on an event $Y=y$ with density $p_{Y}(y)>0$:

$$
p_{X|Y}(x|y)\coloneqq\frac{p_{X,Y}(x,y)}{p_{Y}(y)},
$$

and similarly for the conditional PDF $p_{Y|X}$. Bayes’ rule expresses the conditional PDF $p_{Y|X}$ with $p_{X|Y}$ by

$$
p_{Y|X}(y|x)=\frac{p_{X|Y}(x|y)p_{Y}(y)}{p_{X}(x)},
$$

for $p_{X}(x)>0$.

The *conditional expectation* $\mathbb{E}\left[X|Y\right]$ is the best approximating *function* $g_{\star}(Y)$ to $X$ in the least-squares sense:

$$
\displaystyle g_{\star}
$$
 
$$
\displaystyle\coloneqq\operatorname*{arg\,min}_{g:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}}\mathbb{E}\left[\left\|X-g(Y)\right\|^{2}\right]=\operatorname*{arg\,min}_{g:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}}\int\left\|x-g(y)\right\|^{2}p_{X,Y}(x,y)\mathrm{d}x\mathrm{d}y
$$
 
$$
\displaystyle=\operatorname*{arg\,min}_{g:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}}\int\left[{\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}\int\left\|x-g(y)\right\|^{2}p_{X|Y}(x|y)\mathrm{d}x}\right]p_{Y}(y)\mathrm{d}y.
$$

For $y\in\mathbb{R}^{d}$ such that $p_{Y}(y)>0$ the conditional expectation function is therefore

$$
\mathbb{E}\left[X|Y=y\right]\coloneqq g_{\star}(y)=\int xp_{X|Y}(x|y)\mathrm{d}x,
$$

where the second equality follows from taking the minimizer of the inner brackets in eq.˜87 for $Y=y$, similarly to eq.˜82. Composing $g_{\star}$ with the random variable $Y$, we get

$$
\mathbb{E}\left[X|Y\right]\coloneqq g_{\star}(Y),
$$

which is a random variable in $\mathbb{R}^{d}$. Rather confusingly, both $\mathbb{E}\left[X|Y=y\right]$ and $\mathbb{E}\left[X|Y\right]$ are often called *conditional expectation*, but these are different objects. In particular, $\mathbb{E}\left[X|Y=y\right]$ is a function $\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}$, while $\mathbb{E}\left[X|Y\right]$ is a random variable assuming values in $\mathbb{R}^{d}$. To disambiguate these two terms, our discussions will employ the notations introduced here.

The *tower property* is an useful property that helps simplify derivations involving conditional expectations of two RVs $X$ and $Y$:

$$
\mathbb{E}\left[\mathbb{E}\left[X|Y\right]\right]=\mathbb{E}\left[X\right]
$$

Because $\mathbb{E}\left[X|Y\right]$ is a RV, itself a function of the RV $Y$, the outer expectation computes the expectation of $\mathbb{E}\left[X|Y\right]$. The tower property can be verified by using some of the definitions above:

$$
\displaystyle\mathbb{E}\left[\mathbb{E}\left[X|Y\right]\right]
$$
 
$$
\displaystyle=\int\left(\int xp_{X|Y}(x|y)\mathrm{d}x\right)p_{Y}(y)\mathrm{d}y
$$
 
$$
\displaystyle\overset{\lx@cref{refnum}{e:conditional}}{=}\int\int xp_{X,Y}(x,y)\mathrm{d}x\mathrm{d}y
$$
 
$$
\displaystyle\overset{\lx@cref{refnum}{e:marginals}}{=}\int xp_{X}(x)\mathrm{d}x
$$
 
$$
\displaystyle=\mathbb{E}\left[X\right].
$$

Finally, consider a helpful property involving two RVs $f(X,Y)$ and $Y$, where $X$ and $Y$ are two arbitrary RVs. Then, by using the Law of the Unconscious Statistician with 88, we obtain the identity

$$
\mathbb{E}\left[f(X,Y)|Y=y\right]=\int f(x,y)p_{X|Y}(x|y)\mathrm{d}x.
$$

## Appendix B A Proof of the Fokker-Planck equation

In this section, we give here a self-contained proof of the Fokker-Planck equation (theorem˜15) which includes the continuity equation as a special case (theorem˜12). We stress that this section is not necessary to understand the remainder of this document and is mathematically more advanced. If you desire to understand where the Fokker-Planck equation comes from, then this section is for you.

We start by showing that the Fokker-Planck is a necessary condition, i.e. if $X_{t}\sim p_{t}$, then the Fokker-Planck equation is fulfilled. The trick for the proof is to use test functions $f$, i.e. functions $f:\mathbb{R}^{d}\to\mathbb{R}$ that are infinitely differentiable ("smooth") and are only non-zero within a bounded domain (compact support). We use the fact that for arbitrary integrable functions $g_{1},g_{2}:\mathbb{R}^{d}\to\mathbb{R}$ it holds that

$$
\displaystyle g_{1}(x)=g_{2}(x)\text{ for all }x\in\mathbb{R}^{d}\quad\Leftrightarrow\quad\int f(x)g_{1}(x)\mathrm{d}x=\int f(x)g_{2}(x)\mathrm{d}x\text{ for all test functions }f
$$

In other words, we can express the pointwise equality as equality of taking integrals. The useful thing about test functions is that they are smooth, i.e. we can take gradients and higher-order derivatives. In particular, we can use integration by parts for arbitrary test functions $f_{1},f_{2}$:

$$
\displaystyle\int f_{1}(x)\frac{\partial}{\partial x_{i}}f_{2}(x)\mathrm{d}x=-\int f_{2}(x)\frac{\partial}{\partial x_{i}}f_{1}(x)\mathrm{d}x
$$

By using this together with the definition of the divergence and Laplacian (see eq.˜23), we get the identities:

$$
\displaystyle\int\nabla f_{1}^{T}(x)f_{2}(x)\mathrm{d}x=
$$
 
$$
\displaystyle-\int f_{1}(x)\mathrm{div}(f_{2})(x)\mathrm{d}x\quad(f_{1}:\mathbb{R}^{d}\to\mathbb{R},f_{2}:\mathbb{R}^{d}\to\mathbb{R}^{d})
$$
 
$$
\displaystyle\int f_{1}(x)\Delta f_{2}(x)\mathrm{d}x=
$$
 
$$
\displaystyle\int f_{2}(x)\Delta f_{1}(x)\mathrm{d}x\quad(f_{1}:\mathbb{R}^{d}\to\mathbb{R},f_{2}:\mathbb{R}^{d}\to\mathbb{R})
$$

Now let’s proceed to the proof. We use the stochastic update of SDE trajectories as in eq.˜6:

$$
\displaystyle X_{t+h}=
$$
 
$$
\displaystyle X_{t}+hu_{t}(X_{t})+\sigma_{t}(W_{t+h}-W_{t})+hR_{t}(h)
$$
 
$$
\displaystyle\approx
$$
 
$$
\displaystyle X_{t}+hu_{t}(X_{t})+\sigma_{t}(W_{t+h}-W_{t})
$$

where for now we simply ignore the error term $R_{t}(h)$ for readability as we will take $h\to 0$ anyway. We can then make the following calculation:

$$
\displaystyle f(X_{t+h})-f(X_{t})
$$
 
$$
\displaystyle\overset{\lx@cref{refnum}{eq:approximate_condition}}{=}
$$
 
$$
\displaystyle f(X_{t}+hu_{t}(X_{t})+\sigma_{t}(W_{t+h}-W_{t}))-f(X_{t})
$$
 
$$
\displaystyle\overset{(i)}{=}
$$
 
$$
\displaystyle\nabla f(X_{t})^{T}\left(hu_{t}(X_{t})+\sigma_{t}(W_{t+h}-W_{t}))\right)
$$
 
$$
\displaystyle+\frac{1}{2}\left(hu_{t}(X_{t})+\sigma_{t}(W_{t+h}-W_{t}))\right)^{T}\nabla^{2}f(X_{t})\left(hu_{t}(X_{t})+\sigma_{t}(W_{t+h}-W_{t}))\right)
$$
 
$$
\displaystyle\overset{(ii)}{=}
$$
 
$$
\displaystyle h\nabla f(X_{t})^{T}u_{t}(X_{t})+\sigma_{t}\nabla f(X_{t})^{T}(W_{t+h}-W_{t})
$$
 
$$
\displaystyle+\frac{1}{2}h^{2}u_{t}(X_{t})^{T}\nabla^{2}f(X_{t})u_{t}(X_{t})+h\sigma_{t}u_{t}(X_{t})^{T}\nabla^{2}f(X_{t})(W_{t+h}-W_{t})+\frac{1}{2}\sigma_{t}^{2}(W_{t+h}-W_{t})^{T}\nabla^{2}f(X_{t})(W_{t+h}-W_{t})
$$

where in (i) we used a 2nd Taylor approximation of $f$ around $X_{t}$ and in (ii) we used the fact that the Hessian $\nabla^{2}f$ is a symmetric matrix. Note that $\mathbb{E}[W_{t+h}-W_{t}|X_{t}]=0$ and $W_{t+h}-W_{t}|X_{t}\sim\mathcal{N}(0,hI_{d})$. Therefore

$$
\displaystyle\mathbb{E}[f(X_{t+h})-f(X_{t})|X_{t}]
$$
 
$$
\displaystyle=
$$
 
$$
\displaystyle h\nabla f(X_{t})^{T}u_{t}(X_{t})+\frac{1}{2}h^{2}u_{t}(X_{t})^{T}\nabla^{2}f(X_{t})u_{t}(X_{t})+\frac{h}{2}\sigma_{t}^{2}\mathbb{E}_{\epsilon_{t}\sim\mathcal{N}(0,I_{d})}[\epsilon_{t}^{T}\nabla^{2}f(X_{t})\epsilon_{t}]
$$
 
$$
\displaystyle\overset{(i)}{=}
$$
 
$$
\displaystyle h\nabla f(X_{t})^{T}u_{t}(X_{t})+\frac{1}{2}h^{2}u_{t}(X_{t})^{T}\nabla^{2}f(X_{t})u_{t}(X_{t})+\frac{h}{2}\sigma_{t}^{2}\text{trace}(\nabla^{2}f(X_{t}))
$$
 
$$
\displaystyle\overset{(ii)}{=}
$$
 
$$
\displaystyle h\nabla f(X_{t})^{T}u_{t}(X_{t})+\frac{1}{2}h^{2}u_{t}(X_{t})^{T}\nabla^{2}f(X_{t})u_{t}(X_{t})+\frac{h}{2}\sigma_{t}^{2}\Delta f(X_{t})
$$

where in $(i)$ we used the fact that $\mathbb{E}_{\epsilon_{t}\sim\mathcal{N}(0,I_{d})}[\epsilon_{t}^{T}A\epsilon_{t}]=\text{trace}(A)$ and in $(ii)$ we used the definition of the Laplacian and the Hessian matrix. With this, we get that

$$
\displaystyle\partial_{t}\mathbb{E}[f(X_{t})]
$$
 
$$
\displaystyle=
$$
 
$$
\displaystyle\lim\limits_{h\to 0}\frac{1}{h}\mathbb{E}[f(X_{t+h})-f(X_{t})]
$$
 
$$
\displaystyle=
$$
 
$$
\displaystyle\lim\limits_{h\to 0}\frac{1}{h}\mathbb{E}[\mathbb{E}[f(X_{t+h})-f(X_{t})|X_{t}]]
$$
 
$$
\displaystyle=
$$
 
$$
\displaystyle\mathbb{E}[\lim\limits_{h\to 0}\frac{1}{h}\left(h\nabla f(X_{t})^{T}u_{t}(X_{t})+\frac{1}{2}h^{2}u_{t}(X_{t})^{T}\nabla^{2}f(X_{t})u_{t}(X_{t})+\frac{h}{2}\sigma_{t}^{2}\Delta f(X_{t})\right)]
$$
 
$$
\displaystyle=
$$
 
$$
\displaystyle\mathbb{E}[\nabla f(X_{t})^{T}u_{t}(X_{t})+\frac{1}{2}\sigma_{t}^{2}\Delta f(X_{t})]
$$
 
$$
\displaystyle\overset{(i)}{=}
$$
 
$$
\displaystyle\int\nabla f(x)^{T}u_{t}(x)p_{t}(x)\mathrm{d}x+\int\frac{1}{2}\sigma_{t}^{2}\Delta f(x)p_{t}(x)\mathrm{d}x
$$
 
$$
\displaystyle\overset{(ii)}{=}
$$
 
$$
\displaystyle-\int f(x)\mathrm{div}(u_{t}p_{t})(x)\mathrm{d}x+\int\frac{1}{2}\sigma_{t}^{2}f(x)\Delta p_{t}(x)\mathrm{d}x
$$
 
$$
\displaystyle=
$$
 
$$
\displaystyle\int f(x)\left(-\mathrm{div}(u_{t}p_{t})(x)+\frac{1}{2}\sigma_{t}^{2}\Delta p_{t}(x)\right)\mathrm{d}x
$$

where in (i) we used the assumption that $p_{t}$ as the distribution of $X_{t}$ and in (ii) we used eq.˜94 and eq.˜95. Therefore, it holds that

$$
\displaystyle\partial_{t}\mathbb{E}[f(X_{t})]=
$$
 
$$
\displaystyle\int f(x)\left(-\mathrm{div}(p_{t}u_{t})(x)+\frac{\sigma_{t}^{2}}{2}\Delta p_{t}(x)\right)\mathrm{d}x\quad(\text{for all }f\text{ and }0\leq t\leq 1)
$$
 
$$
\displaystyle\overset{(i)}{\Leftrightarrow}\quad\partial_{t}\int f(x)p_{t}(x)\mathrm{d}x=
$$
 
$$
\displaystyle\int f(x)\left(-\mathrm{div}(p_{t}u_{t})(x)+\frac{\sigma_{t}^{2}}{2}\Delta p_{t}(x)\right)\mathrm{d}x\quad(\text{for all }f\text{ and }0\leq t\leq 1)
$$
 
$$
\displaystyle\overset{(ii)}{\Leftrightarrow}\quad\int f(x)\partial_{t}p_{t}(x)\mathrm{d}x=
$$
 
$$
\displaystyle\int f(x)\left(-\mathrm{div}(p_{t}u_{t})(x)+\frac{\sigma_{t}^{2}}{2}\Delta p_{t}(x)\right)\mathrm{d}x\quad(\text{for all }f\text{ and }0\leq t\leq 1)
$$
 
$$
\displaystyle\overset{(iii)}{\Leftrightarrow}\quad\partial_{t}p_{t}(x)=
$$
 
$$
\displaystyle-\mathrm{div}(p_{t}u_{t})(x)+\frac{\sigma_{t}^{2}}{2}\Delta p_{t}(x)\quad(\text{for all }x\in\mathbb{R}^{d},0\leq t\leq 1)
$$

where in (i) we used the assumption that $X_{t}\sim p_{t}$, in (ii) we swapped the derivative with the integral and (iii) we used eq.˜92. This completes the proof that the Fokker-Planck equation is a necessary condition.

Finally, we explain why it is also a sufficient condition. The Fokker-Planck equation is a partial differential equation (PDE). More specifically, it is a so-called *parabolic partial differential equation*. Similar to theorem˜3, such differential equations have a unique solution given fixed initial conditions (see e.g. \[[8](#bib.bibx8), Chapter 7\]). Now, if eq.˜30 holds for $p_{t}$, we just shown above that it must also hold for true distribution $q_{t}$ of $X_{t}$ (i.e. $X_{t}\sim q_{t}$) - in other words, both $p_{t}$ and $q_{t}$ are solutions to the parabolic PDE. Further, we know that the initial conditions are the same, i.e. $p_{0}=q_{0}=p_{\rm{init}}$ by construction of an interpolating probability path (see LABEL:eq:interpolating\_condition). Hence, by uniqueness of the solution of the differential equation, we know that $p_{t}=q_{t}$ for all $0\leq t\leq 1$ - which means $X_{t}\sim q_{t}=p_{t}$ and which is what we wanted to show.

[^1]: Michael S Albergo, Nicholas M Boffi and Eric Vanden-Eijnden “Stochastic interpolants: A unifying framework for flows and diffusions” In *arXiv preprint arXiv:2303.08797*, 2023

[^2]: Brian DO Anderson “Reverse-time diffusion equation models” In *Stochastic Processes and their Applications* 12.3 Elsevier, 1982, pp. 313–326

[^3]: Yogesh Balaji et al. “eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers”, 2023 arXiv: [https://arxiv.org/abs/2211.01324](https://arxiv.org/abs/2211.01324)

[^4]: Earl A Coddington, Norman Levinson and T Teichmann “Theory of ordinary differential equations” American Institute of Physics, 1956

[^5]: Prafulla Dhariwal and Alex Nichol “Diffusion Models Beat GANs on Image Synthesis”, 2021 arXiv: [https://arxiv.org/abs/2105.05233](https://arxiv.org/abs/2105.05233)

[^6]: Alexey Dosovitskiy et al. “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”, 2021 arXiv: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

[^7]: Patrick Esser et al. “Scaling Rectified Flow Transformers for High-Resolution Image Synthesis”, 2024 arXiv: [https://arxiv.org/abs/2403.03206](https://arxiv.org/abs/2403.03206)

[^8]: Lawrence C Evans “Partial differential equations” American Mathematical Society, 2022

[^9]: Jonathan Ho, Ajay Jain and Pieter Abbeel “Denoising diffusion probabilistic models” In *Advances in neural information processing systems* 33, 2020, pp. 6840–6851

[^10]: Jonathan Ho and Tim Salimans “Classifier-Free Diffusion Guidance”, 2022 arXiv: [https://arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598)

[^11]: Arieh Iserles “A first course in the numerical analysis of differential equations” Cambridge university press, 2009

[^12]: Tero Karras, Miika Aittala, Timo Aila and Samuli Laine “Elucidating the design space of diffusion-based generative models” In *Advances in Neural Information Processing Systems* 35, 2022, pp. 26565–26577

[^13]: Samuel Lavoie et al. “Modeling Caption Diversity in Contrastive Vision-Language Pretraining”, 2024 arXiv: [https://arxiv.org/abs/2405.00740](https://arxiv.org/abs/2405.00740)

[^14]: Yaron Lipman et al. “Flow matching for generative modeling” In *arXiv preprint arXiv:2210.02747*, 2022

[^15]: Yaron Lipman et al. “Flow Matching Guide and Code” In *arXiv preprint arXiv:2412.06264*, 2024

[^16]: Xingchao Liu, Chengyue Gong and Qiang Liu “Flow straight and fast: Learning to generate and transfer data with rectified flow” In *arXiv preprint arXiv:2209.03003*, 2022

[^17]: Nanye Ma et al. “Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers” In *arXiv preprint arXiv:2401.08740*, 2024

[^18]: Xuerong Mao “Stochastic differential equations and applications” Elsevier, 2007

[^19]: William Peebles and Saining Xie “Scalable Diffusion Models with Transformers”, 2023 arXiv: [https://arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)

[^20]: Lawrence Perko “Differential equations and dynamical systems” Springer Science & Business Media, 2013

[^21]: Adam Polyak et al. “Movie Gen: A Cast of Media Foundation Models”, 2024 arXiv: [https://arxiv.org/abs/2410.13720](https://arxiv.org/abs/2410.13720)

[^22]: Alec Radford et al. “Learning Transferable Visual Models From Natural Language Supervision”, 2021 arXiv: [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)

[^23]: Colin Raffel et al. “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer”, 2023 arXiv: [https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)

[^24]: Robin Rombach et al. “High-Resolution Image Synthesis with Latent Diffusion Models”, 2022 arXiv: [https://arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)

[^25]: Olaf Ronneberger, Philipp Fischer and Thomas Brox “U-net: Convolutional networks for biomedical image segmentation” In *Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18*, 2015, pp. 234–241 Springer

[^26]: Chitwan Saharia et al. “Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding”, 2022 arXiv: [https://arxiv.org/abs/2205.11487](https://arxiv.org/abs/2205.11487)

[^27]: Simo Särkkä and Arno Solin “Applied stochastic differential equations” Cambridge University Press, 2019

[^28]: Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan and Surya Ganguli “Deep unsupervised learning using nonequilibrium thermodynamics” In *International conference on machine learning*, 2015, pp. 2256–2265 PMLR

[^29]: Yang Song and Stefano Ermon “Generative modeling by estimating gradients of the data distribution” In *Advances in neural information processing systems* 32, 2019

[^30]: Yang Song et al. “Score-Based Generative Modeling through Stochastic Differential Equations”, 2021 arXiv: [https://arxiv.org/abs/2011.13456](https://arxiv.org/abs/2011.13456)

[^31]: Yang Song et al. “Score-Based Generative Modeling through Stochastic Differential Equations” In *International Conference on Learning Representations (ICLR)*, 2021

[^32]: Yang Song et al. “Score-based generative modeling through stochastic differential equations” In *arXiv preprint arXiv:2011.13456*, 2020

[^33]: Yi Tay et al. “UL2: Unifying Language Learning Paradigms”, 2023 arXiv: [https://arxiv.org/abs/2205.05131](https://arxiv.org/abs/2205.05131)

[^34]: Arash Vahdat, Karsten Kreis and Jan Kautz “Score-based generative modeling in latent space” In *Advances in neural information processing systems* 34, 2021, pp. 11287–11302

[^35]: Ashish Vaswani et al. “Attention Is All You Need”, 2023 arXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

[^36]: Linting Xue et al. “ByT5: Towards a token-free future with pre-trained byte-to-byte models”, 2022 arXiv: [https://arxiv.org/abs/2105.13626](https://arxiv.org/abs/2105.13626)