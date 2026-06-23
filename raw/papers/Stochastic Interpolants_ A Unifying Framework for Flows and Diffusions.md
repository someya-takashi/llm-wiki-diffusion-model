---
title: "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions"
source: "https://ar5iv.labs.arxiv.org/html/2303.08797"
author:
published:
created: 2026-06-23
description: "A class of generative models that unifies flow-based and diffusion-based methods is introduced. These models extend the framework proposed in [2], enabling the use of a broad class of continuous-time stochastic process…"
tags:
  - "clippings"
---
Michael S. Albergo Author ordering alphabetical; authors contributed equally. Center for Cosmology and Particle Physics, New York University Nicholas M. Boffi <sup>∗</sup> Courant Institute of Mathematical Sciences, New York University Eric Vanden-Eijnden Courant Institute of Mathematical Sciences, New York University

###### Abstract

A class of generative models that unifies flow-based and diffusion-based methods is introduced. These models extend the framework proposed in [^2], enabling the use of a broad class of continuous-time stochastic processes called ‘stochastic interpolants’ to bridge any two arbitrary probability density functions exactly in finite time. These interpolants are built by combining data from the two prescribed densities with an additional latent variable that shapes the bridge in a flexible way. The time-dependent probability density function of the stochastic interpolant is shown to satisfy a first-order transport equation as well as a family of forward and backward Fokker-Planck equations with tunable diffusion coefficient. Upon consideration of the time evolution of an individual sample, this viewpoint immediately leads to both deterministic and stochastic generative models based on probability flow equations or stochastic differential equations with an adjustable level of noise. The drift coefficients entering these models are time-dependent velocity fields characterized as the unique minimizers of simple quadratic objective functions, one of which is a new objective for the score of the interpolant density. We show that minimization of these quadratic objectives leads to control of the likelihood for generative models built upon stochastic dynamics, while likelihood control for deterministic dynamics is more stringent. We also construct estimators for the likelihood and the cross-entropy of interpolant-based generative models, and we discuss connections with other methods such as score-based diffusion models, stochastic localization processes, probabilistic denoising techniques, and rectifying flows. In addition, we demonstrate that stochastic interpolants recover the Schrödinger bridge between the two target densities when explicitly optimizing over the interpolant. Finally, algorithmic aspects are discussed and the approach is illustrated on numerical examples.

## 1 Introduction

### 1.1 Background and motivation

Dynamical approaches for deterministic and stochastic transport have become a central theme in contemporary generative modeling research. At the heart of progress is the idea to use ordinary or stochastic differential equations (ODEs/SDEs) to continuously transform samples from a base probability density function (PDF) $\rho_{0}$ into samples from a target density $\rho_{1}$ (or vice-versa), and the realization that inference over the velocity field in these equations can be formulated as an empirical risk minimization problem over a parametric class of functions [^24] [^58] [^25] [^60] [^5] [^2] [^41] [^39].

A major milestone was the introduction of score-based diffusion methods (SBDM) [^60], which map an arbitrary density into a standard Gaussian by passing samples through an Ornstein-Uhlenbeck (OU) process. The key insight of SBDM is that this process can be reversed by introducing a backwards SDE whose drift coefficient depends on the score of the time-dependent density of the process. By learning this score – which can be done by minimization of a quadratic objective function known as the denoising loss [^68] – the backwards SDE can be used as a generative model that maps Gaussian noise into data from the target. Though theoretically exact, the mapping takes infinite time in both directions, and hence must be truncated in practice.

While diffusion-based methods have become state-of-the-art for tasks such as image generation, there remains considerable interest in developing methods that bridge two arbitrary densities (rather than requiring one to be Gaussian), that accomplish the transport exactly, and that do so on a finite time interval. Moreover, while the highest quality results from score-based diffusion were originally obtained using SDEs [^60], this has been challenged by recent works that find equivalent or better performance with ODE-based methods if the score is learned sufficiently well [^32]. If made to match the performance of their stochastic counterparts, ODE-based methods exhibit a number of desirable characteristics, such as an exact, computationally tractable formula for the likelihood and the easy application of well-developed adaptive integration schemes for sampling. It is an open question of significant practical importance to understand if there exists a separation in sample quality between generative models based on deterministic dynamics and those based on stochastic dynamics.

In order to satisfy the desirable characteristics outlined in the previous paragraph, we develop a framework for generative modeling based on the method proposed in [^2], which is built on the notion of a stochastic interpolant $x_{t}$ used to bridge two arbitrary densities $\rho_{0}$ and $\rho_{1}$. We will consider more general designs below, but as one example the reader can keep in mind:

$$
x_{t}=(1-t)x_{0}+tx_{1}+\sqrt{2t(1-t)}z,\quad t\in[0,1],
$$

where $x_{0}$, $x_{1}$, and $z$ are random variables drawn independently from $\rho_{0}$, $\rho_{1}$, and the standard Gaussian density $\mathsf{N}(0,\text{\it Id})$, respectively. The stochastic interpolant $x_{t}$ defined in (1.1) is a continuous-time stochastic process that, by construction, satisfies $x_{t=0}=x_{0}\sim\rho_{0}$ and $x_{t=1}=x_{1}\sim\rho_{1}$. Its paths therefore exactly bridge between samples from $\rho_{0}$ at $t=0$ and from $\rho_{1}$ at $t=1$. A key observation is that:

> The law of the interpolant $x_{t}$ at any time $t\in[0,1]$ can be realized by many different processes, including an ODE and forward and backward SDEs whose drifts can be learned from data.

To see why this is the case, one must consider the probability distribution of the interpolant $x_{t}$. As shown below, for a large class of densities $\rho_{0}$ and $\rho_{1}$ supported on ${\mathbb{R}}^{d}$, this distribution is absolutely continuous with respect to the Lebesgue measure. Moreover, its time-dependent density $\rho(t)$ satisfies a first-order transport equation and a family of forward and backward Fokker-Planck equations in which the diffusion coefficient can be varied at will. Out of these equations, we can readily derive generative models that satisfy ODEs and SDEs, respectively, and whose densities at time $t$ are given by $\rho(t)$.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2303.08797/assets/x1.png)

Figure 1: The stochastic interpolant paradigm. Example generative models based on the proposed framework, which connects two densities ρ 0 subscript 𝜌 \\rho\_{0} and 1 \\rho\_{1} using samples from both. The design of the time-dependent probability density ( t ) 𝑡 \\rho(t) that bridges between is separated from the choice of how to sample it, which can be accomplished with deterministic or stochastic generative models. Left panel: Sampling with a deterministic (ODE) generative model known as the probability flow equation. Right panel: Sampling with a stochastic generative model given by an SDE with a tunable diffusion coefficient. The probability flow equation and the SDE have different paths, but their time-dependent density is the same. Moreover, the two equations rely on the same estimates for the velocity and the score.

Interestingly, the drift coefficients entering these ODEs/SDEs are the unique minimizers of quadratic objective functions that can be estimated empirically using data from $\rho_{0}$, $\rho_{1}$, and $\mathsf{N}(0,\text{\it Id})$. The resulting least-squares regression problem allows us to estimate the drift coefficients of the ODE/SDEs, which can then be used to push samples from $\rho_{0}$ onto new samples from $\rho_{1}$ and vice-versa.

### 1.2 Main contributions and organization

The approach introduced here is a versatile way to build generative models that unifies and extends many existing algorithms. In Sec. 2, we develop the framework in full generality, where we emphasize the following key contributions:

- We prove that the stochastic interpolant defined in Section 2.1 has a distribution that is absolutely continuous with respect to the Lebesgue measure on ${\mathbb{R}}^{d}$, and that its density $\rho(t)$ satisfies a first-order transport equation (TE) as well as a family of forward and backward Fokker-Planck equations (FPEs) with tunable diffusion coefficients.
- We show how the stochastic interpolant can be used to learn the drift coefficients that enter the TE and the FPEs. We characterize these coefficients as the minimizers of simple quadratic objective functions given in Section 2.2. We introduce a new objective for the score $\nabla\log\rho(t)$ of the interpolant density, as well as an objective function for learning a denoiser $\eta_{z}$, which we relate to the score.
- In Section 2.3, we derive ordinary and stochastic differential equations associated with the TE and FPEs that lead to deterministic and stochastic generative models. In Section 2.4, we show that regressing the drift for SDE-based models controls the likelihood, but that regressing the drift alone is not sufficient for ODE-based models, which must also minimize a Fisher divergence. We show how to optimally tune the diffusion coefficient to maximize the likelihood for SDEs.
- In Section 2.5, we develop a general formula to evaluate the likelihood of SDE-based generative models that serves as a natural counterpart to the continuous change-of-variables formula commonly used to compute the likelihood of ODE-based models. In addition, we give formulas to estimate the cross-entropy.

In Section 3, we discuss instantiations of the stochastic interpolant method. In Section 3.4 we first show that interpolants are equivalent to a class of stochastic bridges, but that they avoid the need for Doob’s $h$ -transform, which is generically unknown; we show that this simplifies the construction of a broad class of generative models. In Section 3.2, we define the one-sided interpolant, which corresponds to the conventional setting in which the base $\rho_{0}$ is taken to be a Gaussian. With a Gaussian base, several aspects of the interpolant simplify, and we detail the corresponding objective functions. In Section 3.3, we introduce a mirror interpolant in which the base $\rho_{0}$ and the target $\rho_{1}$ are identical. Finally, in Section 3.4, we show how the interpolant framework leads to a natural formulation of the Schrödinger bridge problem between two densities.

In Section 4, we discuss a special case in which the interpolant is spatially linear in $x_{0}$ and $x_{1}$. In this case, the velocity field can be factorized, which we show in Section 4.1 leads to a simpler learning problem. We detail specific choices of linear interpolants in Section 4.2, and in Section 4.3 we illustrate how these choices influence the performance of the resulting generative model, with a particular focus on the role of the latent variable and the diffusion coefficient. For exposition, we focus on Gaussian mixture densities, for which the drift coefficients can be computed analytically. We provide the resulting formula in Appendix A. Finally, in Section 4.4, we discuss the case of spatially linear one-sided interpolants.

In Section 5, we formalize the connection between stochastic interpolants and related classes of generative models. In Section 5.1, we show that score-based diffusion models can be re-written as one-sided interpolants after a reparameterization of time; we highlight how this approach eliminates singularities that appear when naively compressing score-based diffusion onto a finite-time interval. In Section 5.2, we show how interpolants can be used to derive the Bayes-optimal estimator for a denoiser, and we show how this approach can be iterated to create a generative model. In Section 5.3, we consider the possibility of rectifying the flow map of a learned generative model. We show that the rectification procedure does not change the underlying generative model, though it may change the time-dependent density of the interpolant.

In Section 6, we provide the details of practical algorithms associated with the mathematical results presented above. In Section 6.1, we describe how to numerically estimate the objectives given empirical datasets from the base and the target. In Section 6.2, we complement this discussion on learning with algorithms for sampling with the ODE or an SDE.

We provide numerical demonstrations in line with these recommendations in Section 7, and we conclude with some remarks in Section. 8.

### 1.3 Related work

#### Deterministic Transport and Normalizing Flows.

Transport-based sampling and density estimation has its contemporary roots in Gaussianizing data via maximum entropy methods [^23] [^12] [^64] [^63]. The change of measure under such transformation is the backbone of normalizing flow models. The first neural network realizations of these methods arose through imposing clever structure on the transformation to make the change of measure tractable in discrete, sequential steps [^52] [^16] [^50] [^28] [^19]. A continuous time version of this procedure was made possible by viewing the map $T=X_{t}(x)$ as the solution of an ODE [^11] [^24], whose parametric drift defining the transport is learned via maximum likelihood estimation. Training this way is intractable at scale, as it requires simulating the ODE. Various methods have introduced regularization on the path taken between the two densities to make the ODE solves more efficient [^22] [^48] [^65], but the fundamental difficulty remains. We also work in continuous time; however, our approach allows us to learn the drift without simulation of the dynamics, and can be formulated at sample generation time through either deterministic or stochastic transport.

#### Stochastic Transport and Score-Based Diffusions (SBDMs).

Complementary to approaches based on deterministic maps, recent works have realized that connecting a data distribution to a Gaussian density can be viewed as the evolution of an Ornstein-Ulhenbeck (OU) process which gradually degrades samples from the distribution of interest to Gaussian noise [^54] [^25] [^58] [^60]. The OU process specifies a path in the space of probability densities; this path is simple to traverse in the forward direction by addition of noise, and can be reversed if access to the score of the time-dependent density $\nabla\log\rho(t)$ is available. This score can be approximated through solution of a least-squares regression problem [^29] [^68], and the target can be sampled by reversing the path once the score has been learned. Interestingly, the resulting forward and backward stochastic processes have an equivalent formulation (at the distribution level) in terms of a deterministic probability flow equation, first noted by [^4] [^49] [^33] and then applied in [^44] [^57] [^34] [^7]. The probability flow formulation is useful for density estimation and cross-entropy calculations, but it is worth noting that the probability flow and the reverse-time SDE will have densities that differ when using an approximate score. The SBDM framework, as it has been originally presented, has a number of features which are not a priori well motivated, including the dependence on mapping to a normal density, the complicated tuning of the time parameterization and noise scheduling [^69] [^26], and the choice of the underlying stochastic dynamics [^17] [^32]. While there have been some efforts to remove dependency on the OU process using stochastic bridges [^51], the resulting procedures can be algorithmically complex, relying on inexact mixtures of diffusions with limited expressivity and no accessible probability flow formulation. Some of these difficulties have been lifted in follow-up works [^42]. As another step in this direction, we observe that the key idea behind SBDMs – the bridging of densities via a time-dependent density whose evolution equation is available – can be generalized to a much wider class of processes in a straightforward and computationally accessible manner.

#### Stochastic Interpolants, Rectified Flows, and Flow matching.

Variants of the stochastic interpolant method presented in [^2] were also presented in [^41] [^39]. In [^41], a linear interpolant was proposed with a focus on straight paths. This was employed as a step toward rectifying the transport paths [^40] through a procedure that improves sampling efficiency but introduces a bias. In Section 5.3, we present an alternative form of rectification that is bias-free. In [^39], the interpolant picture was assembled from the perspective of conditional probability paths connecting to a Gaussian, where a noise convolution was used to improve the learning at the cost of biasing the method. Extensions of [^39] were presented in [^66] that generalize the method beyond the Gaussian base density. In the method proposed here, we introduce an unbiased means to incorporate noise into the process, both via the introduction of a latent variable into the stochastic interpolant and the inclusion of a tunable diffusion coefficient in the asociated stochastic generative models. We provide theoretical and practical motivation for the presence of these noise terms.

#### Optimal Transport and Schrödinger Bridges.

There is both theoretical and practical interest in minimizing the transport cost of connecting $\rho_{0}$ and $\rho_{1}$, which. In the case of deterministic maps, this is characterized by the optimal transport problem, and in the case of diffusive maps, by the Schrödinger Bridge problem [^67] [^15]. Formally, these two problems can be related by viewing the Schrödinger Bridge as an entropy-regularized optimal transport. Optimal transport has primarily been employed as a means to regularize flow-based methods by imposing either a path length penalty [^71] [^48] [^22] [^65] or structure on the parameterization itself [^27] [^70]. A variety of recent works have formulated the Schrödinger problem in the context of a learnable diffusion [^8] [^62] [^14]. In the interpolant framework, [^2] [^41] [^39] [^66] all propose optimal transport extensions to the learning procedure. The method proposed in [^41] [^40] allows one to sequentially lower the transport cost through rectification, at the cost of introducing a bias unless the velocity field is perfectly learned. The method proposed in [^2] is an unbiased framework at the cost of solving an additional optimization problem over the interpolant function. The statement of optimal transport in [^39] only applies to Gaussians, but is shown to be practically useful in experimental demonstrations.

In the method proposed below, we provide two approaches for optimizing the transport under a stochastic dynamics. Our primary approach, based on the scheme introduced in [^2], is presented in Section 3.4. It offers an alternative route to solve the Schrödinger bridge problem under the Benamou-Brenier hydrodynamic formulation of transport by maximizing over the interpolant [^6]. However, we stress that this additional optimization step is not necessary in practice, as our approach leads to bias-free generative models for any fixed interpolant. In addition, Section 5.3 discusses an unbiased variant of the rectification scheme proposed in [^41].

#### Convergence bounds.

Inspired by the successes of score-based diffusion, significant recent research effort has been expended to understand the control that can be obtained on suitable distances between the distribution of the generative model and the target data distribution, such as $\mathsf{KL}$, $W_{2}$, or $\mathsf{TV}$. Perhaps the first line of work in this direction is [^57], which showed that standard score-based diffusion training techniques bound the likelihood of the resulting SDE model. Importantly, as we show here, the likelihood of the corresponding probability flow is not bounded in general by this technique, as first highlighted in the context of SBDM by [^43]. Control for SBDM-based techniques was later quantified more rigorously under the assumption of functional inequalities in a discretized setting by [^36], which were removed by [^37] and [^13] via Girsanov-based techniques. Most relevant to the PDE-based methods considered here is [^10], which applies similar techniques to our own in the SBDM context to obtain sharp guarantees with minimal assumptions.

### 1.4 Notation

Throughout, we denote probability density functions as $\rho_{0}(x)$, $\rho_{1}(x)$, and $\rho(t,x)$, with $t\in[0,1]$ and $x\in{\mathbb{R}}^{d}$, omitting the function arguments when clear from the context. We proceed similarly for other functions of time and space, such as $b(t,x)$ or $I(t,x_{0},x_{1})$. We use the subscript $t$ to denote the time-dependency of stochastic processes, such as the stochastic interpolant $x_{t}$ or the Wiener process $W_{t}$. To specify that the random variable $x_{0}$ is drawn from the probability distribution with density $\rho_{0}$, say, with a slight abuse of notations we use $x_{0}\sim\rho_{0}$. Similarly, we use ${\sf N}(0,\text{\it Id})$ to denote both the density and the distribution of the Gaussian random variable with mean zero and covariance identity. We denote expectation by ${\mathbb{E}}$, and usually specify the random variables this expectation is taken over. With a slight abuse of terminology, we say that the law of the process $x_{t}$ is $\rho(t)$ if $\rho(t)$ is the density of the probability distribution of $x_{t}$ at time $t$.

We use standard notation for function spaces: for example, $C^{1}([0,1])$ is the space of continuously differentiable functions from $[0,1]$ to ${\mathbb{R}}$, $(C^{2}({\mathbb{R}}^{d}))^{d}$ is the space of twice continuously differentiable functions from ${\mathbb{R}}^{d}$ to ${\mathbb{R}}^{d}$, and $C^{p}_{0}({\mathbb{R}}^{d})$ is the space of compactly supported functions from ${\mathbb{R}}^{d}$ to ${\mathbb{R}}$ that are continuously differentiable $p$ times. Given a function $b:[0,1]\times{\mathbb{R}}^{d}\to{\mathbb{R}}^{d}$ with value $b(t,x)$ at $(t,x)$, we use $b\in C^{1}([0,1];(C^{2}({\mathbb{R}}^{d}))^{d})$ to indicate that $b$ is continuously differentiable in $t$ for all $(t,x)\in[0,1]\times{\mathbb{R}}^{d}$ and that $b(t,\cdot)$ is an element of $(C^{2}({\mathbb{R}}^{d}))^{d}$ for all $t\in[0,1]$.

## 2 Stochastic interpolant framework

### 2.1 Definitions and assumptions

We begin by defining the stochastic processes that are central to our approach:

###### Definition 2.1 (Stochastic interpolant).

Given two probability density functions $\rho_{0},\rho_{1}:{{\mathbb{R}}^{d}}\rightarrow{\mathbb{R}}_{\geq 0}$, a stochastic interpolant between $\rho_{0}$ and $\rho_{1}$ is a stochastic process $x_{t}$ defined as

$$
x_{t}=I(t,x_{0},x_{1})+\gamma(t)z,\qquad t\in[0,1],
$$

where:

1. $I\in C^{2}([0,1],(C^{2}({\mathbb{R}}^{d}\times{\mathbb{R}}^{d})^{d})$ satisfies the boundary conditions $I(0,x_{0},x_{1})=x_{0}$ and $I(1,x_{0},x_{1})=x_{1}$, as well as
	$$
	\displaystyle\exists C_{1}<\infty\ :\
	$$
	 
	$$
	\displaystyle|\partial_{t}I(t,x_{0},x_{1})|\leq C_{1}|x_{0}-x_{1}|\quad
	$$
	 
	$$
	\displaystyle\forall(t,x_{0},x_{1})\in[0,1]\times{\mathbb{R}}^{d}\times{\mathbb{R}}^{d}.
	$$
2. $\gamma:[0,1]\to{\mathbb{R}}$ satisfies $\gamma(0)=\gamma(1)=0$, $\gamma(t)>0$ for all $t\in(0,1)$, and $\gamma^{2}\in C^{2}([0,1])$.
3. The pair $(x_{0},x_{1})$ is drawn from a probability measure $\nu$ that marginalizes on $\rho_{0}$ and $\rho_{1}$, i.e.
	$$
	\nu(dx_{0},{\mathbb{R}}^{d})=\rho_{0}(x_{0})dx_{0},\qquad\nu({\mathbb{R}}^{d},dx_{1})=\rho_{1}(x_{1})dx_{1}.
	$$
4. $z$ is a Gaussian random variable independent of $(x_{0},x_{1})$, i.e. $z\sim{\sf N}(0,\text{\it Id})$ and $z\perp(x_{0},x_{1})$.

Eq. (2.2) states that $I(t,x_{0},x_{1})$ does not move too fast along the way from $x_{0}$ at $t=0$ to $x_{1}$ at $t=1$, and as a result does not wander too far from either endpoint – this assumption is made for convenience but is not necessary for most arguments below. Later, we will find it useful to consider choices for $I$ that are spatially nonlinear, which we show can recover the solution to the Schrödinger bridge problem. Nevertheless, a simple example that serves as a valid $I$ in the sense of Definition 2.1 is given in (1.1). The measure $\nu$ allows for a coupling between the two densities $\rho_{0}$ and $\rho_{1}$, which affects the properties of the stochastic interpolant, but a simple choice is to take the product measure $\nu(dx_{0},dx_{1})=\rho_{0}(x_{0})\rho_{1}(x_{1})dx_{0}dx_{1}$, in which case $x_{0}$ and $x_{1}$ are independent. In Section 6 we discuss how to design the stochastic interpolant in (2.1) and state some properties of the corresponding process $x_{t}$. Examples of stochastic interpolants are also shown in Figure 2 for various choices of $I$ and $\gamma$.

###### Remark 2.2 (Comparison with ).

The main difference between the stochastic interpolant defined in (2.1) and the one originally introduced in [^2] is the inclusion of the latent variable $\gamma(t)z$. Many of the results below also hold when we set $\gamma(t)z=0$, but the objective of the present paper is to elucidate the advantages that this additional term provides when neither of the endpoints are Gaussian. We note that we could generalize the construction by making $\gamma(t)$ a tensor; here we focus on the scalar case for simplicity. Another difference is the possibility to couple $\rho_{0}$ and $\rho_{1}$ via $\nu$.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2303.08797/assets/x2.png)

Figure 2: Design flexibility. An illustration of how stochastic interpolants can be tailored to specific aims. All examples show one realization of x t subscript 𝑥 𝑡 x\_{t} with one 0 ∼ ρ similar-to 𝜌 x\_{0}\\sim\\rho\_{0}, one 1 x\_{1}\\sim\\rho\_{1} (the flowers at the left and right of the figures), and one z 𝖭 (, Id ) 𝑧 z\\sim{\\sf N}(0,\\text{\\it Id}). Top, Upper middle, and lower middle: various interpolants, ranging from direct interpolation with no latent variable (as in 2 ) to Gaussian encoding-decoding in which the data transitions to pure noise at the mid-point. Bottom: one-sided interpolant, which connects with score-based diffusion methods.

The stochastic interpolant $x_{t}$ in (2.1) is a continuous-time stochastic process whose realizations are samples from $\rho_{0}$ at time $t=0$ and from $\rho_{1}$ at time $t=1$ by construction. As a result, it offers a way to bridge $\rho_{0}$ and $\rho_{1}$ – we are interested in characterizing the law of $x_{t}$ over the full interval $[0,1]$, as it will allow us to design generative models. Mathematically, we want to characterize the properties of the time-dependent probability distribution $\mu(t,dx)$ such that

$$
\forall t\in[0,1]\quad:\quad\int_{{\mathbb{R}}^{d}}\phi(x)\mu(t,dx)={\mathbb{E}}\phi(x_{t})\quad\text{for any test function}\quad\phi\in C^{\infty}_{0}({\mathbb{R}}^{d}),
$$

where $x_{t}$ is defined in (2.1) and the expectation is taken independently over $(x_{0},x_{1})\sim\nu$, and $z\sim{\sf N}(0,\text{\it Id})$. To this end, we will need to use conditional expectations over $x_{t}$ <sup>1</sup>, as described in the following definition.

###### Definition 2.3.

Given any $f\in C^{\infty}_{0}([0,1]\times{\mathbb{R}}^{d}\times{\mathbb{R}}^{d}\times{\mathbb{R}}^{d})$, its conditional expectation ${\mathbb{E}}\left(f(t,x_{0},x_{1},z)|x_{t}=x\right)$ is the function of $x$ such that

$$
\int_{{\mathbb{R}}^{d}}{\mathbb{E}}\left(f(t,x_{0},x_{1},z)|x_{t}=x\right)\mu(t,dx)={\mathbb{E}}f(t,x_{0},x_{1},z),
$$

where $\mu(t,dx)$ is the time-dependent distribution of $x_{t}$ defined by (2.3), and the expectation on the right-hand side is taken independently over $(x_{0},x_{1})\sim\nu$, and $z\sim{\sf N}(0,\text{\it Id})$.

Vector-valued functions have conditional expectations that are defined analogously. Note that, with our definition, ${\mathbb{E}}\left(f(t,x_{0},x_{1},z)|x_{t}=x\right)$ is a deterministic function of $(t,x)\in[0,1]\times{\mathbb{R}}^{d}$, not to be confused with the random variable ${\mathbb{E}}\left(f(t,x_{0},x_{1},z)|x_{t}\right)$ that can be defined analogously.

###### Remark 2.4.

Another seemingly more general way to define the stochastic interpolant is via

$$
x^{\mathsf{d}}_{t}=I(t,x_{0},x_{1})+N_{t}
$$

where $N:[0,1]\to{\mathbb{R}}^{d}$ is a zero-mean Gaussian stochastic process constrained to satisfy $N_{t=0}=N_{t=1}=0$. As we will show below, our construction only depends on the single-time properties of $N_{t}$, which are completely specified by ${\mathbb{E}}[Z_{t}Z^{\mathsf{T}}_{t}]$. That is, if we take $\gamma(t)$ in (2.1) such that ${\mathbb{E}}[N_{t}N^{\mathsf{T}}_{t}]=\gamma^{2}(t)\text{\it Id}$, then the probability distribution of $x_{t}$ will coincide with that of $x^{\prime}_{t}$ defined in (2.6), $x_{t}\stackrel{{\scriptstyle\text{d}}}{{=}}x^{\mathsf{d}}_{t}$. For example, taking $\gamma(t)=\sqrt{t(1-t)}$ in (2.1) – a choice we will consider below in Sec. 3.4 – is equivalent to choosing $N_{t}$ to be a Brownian bridge in (2.6), i.e. the stochastic process realizable in terms of the Wiener process $W_{t}$ as $N_{t}=W_{t}-tW_{1}$. This observation will also help us draw an analogy between our approach and the construction used in score-based diffusion models. As we will show below, it is simpler for both analysis and practical implementation to work with the definition (2.1) for $x_{t}$.

To proceed, we will make the following assumption on the densities $\rho_{0}$, $\rho_{1}$, and the interplay between the measure $\nu$ to the function $I$:

###### Assumption 2.5.

The densities $\rho_{0}$ and $\rho_{1}$ are strictly positive elements of $C^{2}({\mathbb{R}}^{d})$ and are such that

$$
\int_{{\mathbb{R}}^{d}}|\nabla\log\rho_{0}(x)|^{2}\rho_{0}(x)dx<\infty\quad\text{and}\quad\int_{{\mathbb{R}}^{d}}|\nabla\log\rho_{1}(x)|^{2}\rho_{1}(x)dx<\infty.
$$

The measure $\nu$ and the function $I$ are such that

$$
\exists M_{1},M_{2}<\infty\ \ :\ \ {\mathbb{E}}\big{[}|\partial_{t}I(t,x_{0},x_{1})|^{4}\big{]}\leq M_{1};\quad{\mathbb{E}}\big{[}|\partial^{2}_{t}I(t,x_{0},x_{1})|^{2}\big{]}\leq M_{2},\quad\forall t\in[0,1],
$$

where the expectation is taken over $(x_{0},x_{1})\sim\nu$.

Note that for the interpolant (1.1), Assumption 2.5 holds if $\rho_{0}$ and $\rho_{1}$ both have finite fourth moments.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2303.08797/assets/x3.png)

Figure 3: Algorithmic implementation. A simple overview of suggested implementation strategies. For deterministic sampling, a single velocity field b 𝑏 can be learned by minimizing the empirical loss in the top row. For stochastic sampling, the velocity field, along with the denoiser η z subscript 𝜂 𝑧 \\eta\_{z}, can be learned by minimizing the two empirical losses specified in the bottom row. To sample deterministically, off-the-shelf ODE integrators can be used to integrate the probability flow equation. To sample stochastically, the listed SDE can be integrated using standard techniques such as the Euler-Maruyama method or the Heun sampler introduced in 32. The time-dependent diffusion coefficient ϵ ( t ) italic-ϵ 𝑡 \\epsilon(t) can be specified after learning to maximize sample quality.

### 2.2 Transport equations, score, and quadratic objectives

We now state a result that specifies some important properties of the probability distribution of the stochastic interpolant $x_{t}$:

\[Stochastic interpolant properties\]theoreminterpolation The probability distribution of the stochastic interpolant $x_{t}$ defined in (2.1) is absolutely continuous with respect to the Lebesgue measure at all times $t\in[0,1]$ and its time-dependent density $\rho(t)$ satisfies $\rho(0)=\rho_{0}$, $\rho(1)=\rho_{1}$, $\rho\in C^{1}([0,1];C^{p}({\mathbb{R}}^{d}))$ for any $p\in{\mathbb{N}}$, and $\rho(t,x)>0$ for all $(t,x)\in[0,1]\times{\mathbb{R}}^{d}$. In addition, $\rho$ solves the transport equation

$$
\partial_{t}\rho+\nabla\cdot\left(b\rho\right)=0,
$$

where we defined the velocity

$$
b(t,x)={\mathbb{E}}[\dot{x}_{t}|x_{t}=x]={\mathbb{E}}[\partial_{t}I(t,x_{0},x_{1})+\dot{\gamma}(t)z|x_{t}=x].
$$

This velocity is in $C^{0}([0,1];(C^{p}({\mathbb{R}}^{d}))^{d})$ for any $p\in{\mathbb{N}}$, and such that

$$
\forall t\in[0,1]\quad:\quad\int_{{\mathbb{R}}^{d}}|b(t,x)|^{2}\rho(t,x)dx<\infty.
$$

Note that this theorem means that we can write (2.4) as

$$
\forall t\in[0,1]\quad:\quad\int_{{\mathbb{R}}^{d}}\phi(x)\rho(t,x)dx={\mathbb{E}}\phi(x_{t})\quad\text{for any test function}\quad\phi\in C^{\infty}_{0}({\mathbb{R}}^{d}),
$$

The transport equation (2.9) can be solved either forward in time from the initial condition $\rho(0)=\rho_{0}$, in which case $\rho(1)=\rho_{1}$, or backward in time from the final condition $\rho(1)=\rho_{1}$, in which case $\rho(0)=\rho_{0}$.

The proof of Theorem 2.2 is given in Appendix B.1; it mostly relies on manipulations involving the characteristic function of the stochastic interpolant $x_{t}$. The transport equation (2.9) for $\rho$ lead to methods for generative modeling and density estimation, as explained in Secs. 2.3 and 2.5, provided that we can estimate the velocity $b$. This velocity is explicitly available only in special cases, for example when $\rho_{0}$ and $\rho_{1}$ are both Gaussian mixture densities: this case is treated in Appendix A. In general $b$ must be calculated numerically, which can be performed via empirical risk minimization of a quadratic objective function, as characterized by our next result:

\[Objective\]theoreminterpolatelosses

The velocity $b$ defined in (2.10) is the unique minimizer in $C^{0}([0,1];(C^{1}({\mathbb{R}}^{d}))^{d})$ of the quadratic objective

$$
\mathcal{L}_{b}[\hat{b}]=\int_{0}^{1}{\mathbb{E}}\left(\tfrac{1}{2}|\hat{b}(t,x_{t})|^{2}-\left(\partial_{t}I(t,x_{0},x_{1})+\dot{\gamma}(t)z\right)\cdot\hat{b}(t,x_{t})\right)dt
$$

where $x_{t}$ is defined in (2.1) and the expectation is taken independently over $(x_{0},x_{1})\sim\nu$ and $z\sim{\sf N}(0,\text{\it Id}).$

The proof of Theorem 2.2 is given in Appendix B.1: it relies on the definitions of $b$ in (2.10), as well as the definition of $\rho$ in (2.12) and some elementary properties of the conditional expectation. We discuss how to estimate the objective function (2.13) in practice in Section 6. Interestingly, we also have access to the score of the probability density, as shown by our next result:

\[Score\]theoremscore The score of the probability density $\rho$ specified in Theorem 2.2 is in $C^{1}([0,1];(C^{p}({\mathbb{R}}^{d}))^{d})$ for any $p\in{\mathbb{N}}$ and given by

$$
s(t,x)=\nabla\log\rho(t,x)=-\gamma^{-1}(t){\mathbb{E}}(z|x_{t}=x)\quad\forall(t,x)\in(0,1)\times{\mathbb{R}}^{d}
$$

In addition it satisfies

$$
\forall t\in[0,1]\quad:\quad\int_{{\mathbb{R}}^{d}}|s(t,x)|^{2}\rho(t,x)dx<\infty,
$$

and is the unique minimizer in $C^{1}([0,1];(C^{1}({\mathbb{R}}^{d}))^{d})$ of the quadratic objective

$$
\mathcal{L}_{s}[\hat{s}]=\int_{0}^{1}{\mathbb{E}}\left(\tfrac{1}{2}|\hat{s}(t,x_{t})|^{2}+\gamma^{-1}(t)z\cdot\hat{s}(t,x_{t})\right)dt
$$

where $x_{t}$ is defined in (2.1) and the expectation is taken independently over $(x_{0},x_{1})\sim\nu$ and $z\sim{\sf N}(0,\text{\it Id})$

The proof of Theorem 2.2 is given in Appendix B.1. We stress that the objective function is well defined despite the fact that $\gamma(0)=\gamma(1)=0$: see Section 6 for more details about how to evaluate this objective in practice.

###### Remark 2.6 (Denoiser).

The quantity

$$
\eta_{z}(t,x)={\mathbb{E}}(z|x_{t}=x),
$$

will be referred as the denoiser, for reasons that will be made clear in Section 5.2. By (2.14), this quantity gives access to the score on $t\in(0,1)$ (where $\gamma(t)>0$) since, from (2.14),

$$
s(t,x)=-\gamma^{-1}(t)\eta_{z}(t,x).
$$

This denoiser is the minimizer of an equivalent expression to (2.16),

$$
\mathcal{L}_{\eta_{z}}[\hat{\eta}_{z}]=\int_{0}^{1}{\mathbb{E}}\left(\tfrac{1}{2}|\hat{\eta}_{z}(t,x_{t})|^{2}-z\cdot\hat{\eta}_{z}(t,x_{t})\right)dt.
$$

The denoiser $\eta_{z}$ is useful for numerical realizations. In particular, the objective in (2.19) is easier to use than the one in (2.16) because it does not contain the factor $\gamma^{-1}(t)$, which needs careful handling as $t$ approaches 0 and 1.

Having access to the score immediately allows us to rewrite the TE (2.9) as forward and backward Fokker-Planck equations, which we state as:

\[Fokker-Planck equations\]corollaryinterpolationfpe For any $\epsilon\in C^{0}([0,1])$ with $\epsilon(t)\geq 0$ for all $t\in[0,1]$, the probability density $\rho$ specified in Theorem 2.2 satisfies:

1. The forward Fokker-Planck equation
	$$
	\partial_{t}\rho+\nabla\cdot\left(b_{\mathsf{F}}\rho\right)=\epsilon(t)\Delta\rho,\qquad\rho(0)=\rho_{0},
	$$
	where we defined the forward drift
	$$
	b_{\mathsf{F}}(t,x)=b(t,x)+\epsilon(t)s(t,x).
	$$
	Equation (2.20) is well-posed when solved forward in time from $t=0$ to $t=1$, and its solution for the initial condition $\rho(t=0)=\rho_{0}$ satisfies $\rho(t=1)=\rho_{1}$.
2. The backward Fokker-Planck equation
	$$
	\partial_{t}\rho+\nabla\cdot\left(b_{\mathsf{B}}\rho\right)=-\epsilon(t)\Delta\rho,\qquad\rho(1)=\rho_{1},
	$$
	where we defined the backward drift
	$$
	b_{\mathsf{B}}(t,x)=b(t,x)-\epsilon(t)s(t,x).
	$$
	Equation (2.22) is well-posed when solved backward in time from $t=1$ to $t=0$, and its solution for the final condition $\rho(1)=\rho_{1}$ satisfies $\rho(0)=\rho_{0}$.

In Section 2.3 we will use the results of this theorem to design generative models based on forward and backward stochastic differential equations. Note that we can replace the diffusion coefficient $\epsilon(t)$ by a positive semi-definite tensor; also note that if we define $\rho_{\mathsf{B}}(t_{\mathsf{B}},x)=\rho(1-t_{\mathsf{B}},x)$, the reversed FPE (2.22) can be written as

$$
\partial_{t_{\mathsf{B}}}\rho_{\mathsf{B}}({t_{\mathsf{B}}},x)-\nabla\cdot\left(b_{\mathsf{B}}(1-{t_{\mathsf{B}}},x)\rho_{\mathsf{B}}({t_{\mathsf{B}}},x)\right)=\epsilon(1-t_{\mathsf{B}})\Delta\rho_{\mathsf{B}}({t_{\mathsf{B}}},x),\qquad\rho_{\mathsf{B}}({t_{\mathsf{B}}}=0)=\rho_{1},
$$

which is now well-posed forward in (reversed) time $t_{\mathsf{B}}$. So as to have only one definition of time $t$, it is more convenient to work with (2.22).

Let us make a few remarks about the statements made so far:

###### Remark 2.7.

If we set $\gamma(t)=0$ in $x_{t}$ (i.e, if we remove the latent variable), the stochastic interpolant (2.1) reduces to the one originally considered in [^2]. In this setup, the results above formally stand except that we cannot guarantee the spatial regularity of $b(t,x)$ and $s(t,x)$, since it relies on the presence of the latent variable (as shown in the proof of Theorem 2.2). Hence, we expect the introduction of the latent variable $\gamma(t)z$ to help for generative modeling, where the solution to the corresponding ODEs/SDEs will be better behaved, and for statistical approximation, since the targets $b$ and $s$ will be more regular. We will see in Section 6 that it also gives us much greater flexibility in the way we can bridge $\rho_{0}$ and $\rho_{1}$, which will enable us to design generative models with appealing properties.

###### Remark 2.8.

We will see in Section 2.4 that the forward and backward FPE in (2.20) and (2.22) are more robust than the TE in (2.9) against approximation errors in the velocity $b$ and the score $s$, which has practical implications for generative models based on these equations.

###### Remark 2.9.

We could also obtain $b(t,\cdot)$ at any $t\in[0,1]$ by minimizing

$$
{\mathbb{E}}\left(\tfrac{1}{2}|\hat{b}(t,x_{t})|^{2}-\left(\partial_{t}I(t,x_{0},x_{1})+\dot{\gamma}(t)z\right)\cdot\hat{b}(t,x_{t})\right)\qquad t\in[0,1]
$$

and $s(t,\cdot)$ at any $t\in(0,1)$ by minimizing

$$
{\mathbb{E}}\left(\tfrac{1}{2}|\hat{s}(t,x_{t})|^{2}+\gamma^{-1}(t)z\cdot\hat{s}(t,x_{t})\right)\qquad t\in(0,1)
$$

Using the time-integrated versions of these objectives given in (2.13) and (2.16) is more convenient numerically as it allows one to parameterize $\hat{b}$ and $\hat{s}$ globally for $(t,x)\in[0,1]\times{\mathbb{R}}^{d}$.

###### Remark 2.10.

From (2.10) we can write

$$
b(t,x)=v(t,x)-\dot{\gamma}(t)\gamma(t)s(t,x),
$$

where $s$ is the score given in (2.14) and we defined the velocity field

$$
v(t,x)={\mathbb{E}}(\partial_{t}I(t,x_{0},x_{1})|x_{t}=x).
$$

The velocity field $v\in C^{0}([0,1];(C^{p}({\mathbb{R}}^{d}))^{d})$ for any $p\in{\mathbb{N}}$ and can be characterized as the unique minimizer of

$$
\mathcal{L}_{v}[\hat{v}]=\int_{0}^{1}{\mathbb{E}}\left(\tfrac{1}{2}|\hat{v}(t,x_{t})|^{2}-\partial_{t}I(t,x_{0},x_{1})\cdot\hat{v}(t,x_{t})\right)dt
$$

Learning this velocity and the score separately may be useful in practice.

###### Remark 2.11.

The objectives in (2.13) and (2.16) (as well as the ones in(2.19) and (2.29)) are amenable to empirical estimation if we have samples $(x_{0},x_{1})\sim\nu$, since in that case we can generate samples of $x_{t}=I(t,x_{0},x_{1})+\gamma(t)z$ at any time $t\in[0,1]$. We will use this feature in the numerical experiments presented below.

###### Remark 2.12.

Since $s$ is the score of $\rho$, an alternative objective to estimate it is [^29]

$$
\int_{0}^{1}{\mathbb{E}}\left(|\hat{s}(t,x_{t})|^{2}+2\nabla\cdot\hat{s}(t,x_{t})\right)dt.
$$

The derivation of (2.30) is standard: for the reader’s convenience we recall it at the end of Appendix B.1. The advantage of using (2.16) over (2.30) is that it does not require us to take the divergence of $\hat{s}$.

###### Remark 2.13 (Energy-based models).

By definition, the score $s(t,x)=\nabla\log\rho(t,x)$ is a gradient field. As a result, if we model $\hat{s}(t,x)=-\nabla\hat{E}(t,x)$, we can turn (2.16) into an objective function for $\hat{E}(t,x)$

$$
\mathcal{L}_{E}[\hat{E}]=\int_{0}^{1}{\mathbb{E}}\left(\tfrac{1}{2}|\nabla\hat{E}(t,x_{t})|^{2}+\gamma^{-1}(t)z\cdot\nabla\hat{E}(t,x_{t})\right)dt
$$

This objective is invariant to constant shifts in $\hat{E}$ and should therefore be minimized under some constraint, such as $\min_{x}\hat{E}(t,x)=0$ for all $t\in[0,1]$. The minimizer of (2.31) provides us with an energy-based model (EBM) [^35] [^59] that can in principle be used to sample the PDF of the stochastic interpolant, $\rho(t,x)$, at any fixed $t\in[0,1]$ using e.g. Langevin dynamics. We will not exploit this possibility here, and instead rely on generative models to sample $\rho(t,x)$, as discussed next in Sec. 2.3.

### 2.3 Generative models

Our next result is a direct consequence of Theorem 2.2, and it shows how to design generative models using the stochastic processes associated with the TE (2.9), the forward FPE (2.20), and the backward FPE (2.22):

\[Generative models\]corollarygenerative At any time $t\in[0,1]$, the law of the stochastic interpolant $x_{t}$ coincides with the law of the three processes $X_{t}$, $X^{\mathsf{F}}_{t}$, and $X^{\mathsf{B}}_{t}$, respectively defined as:

1. The solutions of the probability flow associated with the transport equation (2.9)
	$$
	\frac{d}{dt}X_{t}=b(t,X_{t}),
	$$
	solved either forward in time from the initial data $X_{t=0}\sim\rho_{0}$ or backward in time from the final data $X_{t=1}=x_{1}\sim\rho_{1}$.
2. The solutions of the forward SDE associated with the FPE (2.20)
	$$
	dX^{\mathsf{F}}_{t}=b_{\mathsf{F}}(t,X^{\mathsf{F}}_{t})dt+\sqrt{2\epsilon(t)}\,dW_{t},
	$$
	solved forward in time from the initial data $X^{\mathsf{F}}_{t=0}\sim\rho_{0}$ independent of $W$.
3. The solutions of the backward SDE associated with the backward FPE (2.22)
	$$
	dX^{\mathsf{B}}_{t}=b_{\mathsf{B}}(t,X^{\mathsf{B}}_{t})dt+\sqrt{2\epsilon(t)}\,dW^{\mathsf{B}}_{t},\quad W_{t}^{\mathsf{B}}=-W_{1-t},
	$$
	solved backward in time from the final data $X^{\mathsf{B}}_{t=1}\sim\rho_{1}$ independent of $W^{\mathsf{B}}$; the solution of (2.34) is by definition $X^{\mathsf{B}}_{t}=Z^{\mathsf{F}}_{1-t}$ where $Z^{\mathsf{F}}_{t}$ satisfies
	$$
	dZ^{\mathsf{F}}_{t}=-b_{\mathsf{B}}(1-t,Z^{\mathsf{F}}_{t})dt+\sqrt{2\epsilon(t)}\,dW_{t},
	$$
	solved forward in time from the initial data $Z^{\mathsf{F}}_{t=0}\sim\rho_{1}$ independent of $W$.

To avoid repeated applications of the transformation $t\mapsto 1-t$, it is convenient to work with (2.34) directly using the reversed Itô calculus rules stated in the following lemma, which follows from the results in [^3] and is proven in Appendix B.2:

\[Reverse Itô Calculus\]lemmareversed If $X^{\mathsf{B}}_{t}$ solves the backward SDE (2.34):

1. For any $f\in C^{1}([0,1];C_{0}^{2}({\mathbb{R}}_{d}))$ and $t\in[0,1]$, the backward Itô formula holds
	$$
	df(t,X^{\mathsf{B}}_{t})=\partial_{t}f(t,X^{\mathsf{B}}_{t})dt+\nabla f(X^{\mathsf{B}}_{t})\cdot dX^{\mathsf{B}}_{t}-\epsilon(t)\Delta f(t,X^{\mathsf{B}}_{t})dt.
	$$
2. For any $g\in C^{0}([0,1];(C_{0}({\mathbb{R}}_{d}))^{d})$ and $t\in[0,1]$, the backward Itô isometries hold:
	$$
	\displaystyle{\mathbb{E}}^{x}_{\mathsf{B}}\int_{t}^{1}g(t,X^{\mathsf{B}}_{t})\cdot dW^{\mathsf{B}}_{t}=0;\qquad{\mathbb{E}}^{x}_{\mathsf{B}}\left|\int_{t}^{1}g(t,X^{\mathsf{B}}_{t})\cdot dW^{\mathsf{B}}_{t}\right|^{2}
	$$
	 
	$$
	\displaystyle=\int_{t}^{1}{\mathbb{E}}^{x}_{\mathsf{B}}\left|g(t,X^{\mathsf{B}}_{t})\right|^{2}dt,
	$$
	where ${\mathbb{E}}^{x}_{\mathsf{B}}$ denotes expectation conditioned on the event $X_{t=1}^{\mathsf{B}}=x$.

The relevance of Corollary 2.3 for generative modeling is clear. Assuming, for example, that $\rho_{0}$ is a simple density that can be sampled easily (e.g. a Gaussian or a Gaussian mixture density), we can use the ODE (2.32) or the SDE (2.33) to push these samples forward in time and generate samples from a complex target density $\rho_{1}$. In Section 2.5, we will show how to use the ODE (2.32) or the reverse SDE (2.34) to estimate $\rho_{1}$ at any $x\in{\mathbb{R}}^{d}$ assuming that we can evaluate $\rho_{0}$ at any $x\in{\mathbb{R}}^{d}$. We will also show how similar ideas can be used to estimate the cross entropy between $\rho_{0}$ and $\rho_{1}$.

###### Remark 2.14.

We stress that the stochastic interpolant $x_{t}$, the solution $X_{t}$ to the ODE (2.32), and the solutions $X^{\mathsf{F}}_{t}$ and $X^{\mathsf{B}}_{t}$ of the forward and backward SDEs (2.33) and (2.34) are different stochastic processes, but their laws all coincide with $\rho(t)$ at any time $t\in[0,1]$. This is all that matters when applying these processes as generative models. However, the fact that these processes are different has implications for the accuracy of the numerical integration used to sample from them at any $t$ as well as for the propagation of statistical errors (see also the next remark).

Generative models based on solutions $X_{t}$ to the ODE (2.32), solutions $X^{\mathsf{F}}_{t}$ to the forward SDE (2.33), and solutions $X^{\mathsf{B}}_{t}$ to the backward SDE (2.34) will typically involve drifts $b$, $b_{\mathsf{F}}$, and $b_{\mathsf{B}}$ that are, in practice, imperfectly estimated via minimization of (2.13) and (2.16) over finite datasets. It is important to estimate how this statistical estimation error propagates to errors in sample quality, and how the propagation of error depends on the generative model used, which is the object of our next section.

### 2.4 Likelihood control

In this section, we demonstrate that jointly minimizing the objective functions (2.29) and (2.16) (or the losses (2.13) and (2.16)) controls the $\mathsf{KL}$ -divergence from the target density $\rho_{1}$ to the model density $\hat{\rho}_{1}$. We focus on bounds involving the score, but we note that analogous results hold for learning the denoiser $\eta_{z}(t,x)$ defined in (2.17) by the relation $\eta_{z}(t,x)=-s(t,x)/\gamma(t)$. The derivation is based on a simple and exact characterization of the $\mathsf{KL}$ -divergence between two transport equations or two Fokker-Planck equations with different drifts. Remarkably, we find that the presence of a diffusive term determines whether or not it is sufficient to learn the drift to control $\mathsf{KL}$. This can be seen as a generalization of the result for score-based diffusion models described in [^57] to arbitrary generative models described by ODEs or SDEs. The proofs of the statements in this section are provided in Appendix B.3.

We first characterize the $\mathsf{KL}$ divergence between two densities transported by two different continuity equations but initialized from the same initial condition:

lemmakltransport Let $\rho_{0}:{\mathbb{R}}^{d}\rightarrow{\mathbb{R}}_{\geq 0}$ denote a fixed base probability density function. Given two velocity fields $b,\hat{b}\in C^{0}([0,1],(C^{1}({\mathbb{R}}^{d}))^{d})$, let the time-dependent densities $\rho:[0,1]\times{\mathbb{R}}^{d}\to{\mathbb{R}}_{\geq 0}$ and $\hat{\rho}:[0,1]\times{\mathbb{R}}^{d}\to{\mathbb{R}}_{\geq 0}$ denote the solutions to the transport equations

$$
\displaystyle\partial_{t}\rho+\nabla\cdot(b\rho)=0,\qquad
$$
 
$$
\displaystyle\rho(0)=\rho_{0},
$$
$$
\displaystyle\partial_{t}\hat{\rho}+\nabla\cdot(\hat{b}\hat{\rho})=0,\qquad
$$
 
$$
\displaystyle\hat{\rho}(0)=\rho_{0}.
$$

Then, the Kullback-Leibler divergence of $\rho(1)$ from $\hat{\rho}(1)$ is given by

$$
\mathsf{KL}(\rho(1)\>\|\>\hat{\rho}(1))=\int_{0}^{1}\int_{{\mathbb{R}}^{d}}\left(\nabla\log\hat{\rho}(t,x)-\nabla\log\rho(t,x)\right)\cdot\big{(}\hat{b}(t,x)-b(t,x)\big{)}\rho(t,x)dxdt.
$$

Lemma 2.4 shows that it is insufficient in general to match $\hat{b}$ with $b$ to obtain control on the $\mathsf{KL}$ divergence. The essence of the problem is that a small error in $\hat{b}-b$ does not ensure control on the Fisher divergence $\mathsf{FI}(\rho(t)\>\|\>\hat{\rho}(t))=\int_{{\mathbb{R}}^{d}}\left|\nabla\log\rho(t,x)-\nabla\log\hat{\rho}(t,x)\right|^{2}\rho(t,x)dx$, which is necessary due to the presence of $\left(\nabla\log\hat{\rho}-\nabla\log\rho\right)$ in (2.39).

In the next lemma, we study the case for two Fokker-Planck equations, and highlight that the situation becomes quite different. lemmaklfpe Let $\rho_{0}:{\mathbb{R}}^{d}\rightarrow{\mathbb{R}}_{\geq 0}$ denote a fixed base probability density function. Given two velocity fields $b_{\mathsf{F}},\hat{b}_{\mathsf{F}}\in C^{0}([0,1],(C^{1}({\mathbb{R}}^{d}))^{d})$, let the time-dependent densities $\rho:[0,1]\times{\mathbb{R}}^{d}\to{\mathbb{R}}_{\geq 0}$ and $\hat{\rho}:[0,1]\times{\mathbb{R}}^{d}\to{\mathbb{R}}_{\geq 0}$ denote the solutions to the Fokker-Planck equations

$$
\displaystyle\partial_{t}\rho+\nabla\cdot(b_{\mathsf{F}}\rho)=\epsilon\Delta\rho,\qquad
$$
 
$$
\displaystyle\rho(0)=\rho_{0},
$$
$$
\displaystyle\partial_{t}\hat{\rho}+\nabla\cdot(\hat{b}_{\mathsf{F}}\hat{\rho})=\epsilon\Delta\hat{\rho},\qquad
$$
 
$$
\displaystyle\hat{\rho}(0)=\rho_{0}.
$$

where $\epsilon>0$. Then, the Kullback-Leibler divergence from $\rho(1)$ to $\hat{\rho}(1)$ is given by

$$
\displaystyle\mathsf{KL}(\rho(1)\>\|\>\hat{\rho}(1))
$$
 
$$
\displaystyle=\int_{0}^{1}\int_{{\mathbb{R}}^{d}}\left(\nabla\log\hat{\rho}(t,x)-\nabla\log\rho(t,x)\right)\cdot\left(\hat{b}_{\mathsf{F}}(t,x)-b_{\mathsf{F}}(t,x)\right)\rho(t,x)dxdt
$$
 
$$
\displaystyle\qquad-\epsilon\int_{0}^{1}\int_{{\mathbb{R}}^{d}}\left|\nabla\log\rho(t,x)-\nabla\log\hat{\rho}(t,x)\right|^{2}\rho(t,x)dxdt,
$$

and as a result

$$
\displaystyle\mathsf{KL}(\rho(1)\>\|\>\hat{\rho}(1))
$$
 
$$
\displaystyle\leq\frac{1}{4\epsilon}\int_{0}^{1}\int_{{\mathbb{R}}^{d}}\left|\hat{b}_{\mathsf{F}}(t,x)-b_{\mathsf{F}}(t,x)\right|^{2}\rho(t,x)dxdt.
$$

Lemma 2.4 shows that, unlike for transport equations, the $\mathsf{KL}$ -divergence between the solutions of two Fokker-Planck equations is controlled by the error in their drifts. The diffusive term in each Fokker-Planck equation provides an additional negative term in the $\mathsf{KL}$ -divergence, which eliminates the need for explicit control on the Fisher divergence.

Putting the above results together, we can state the following result, which demonstrates that the losses (2.13) and (2.16) control the likelihood for learned approximations to the FPE (2.20).

theoremlikelihoodbound Let $\rho$ denote the solution of the Fokker-Planck equation (2.20) with $\epsilon(t)=\epsilon>0$. Given two velocity fields $\hat{b},\hat{s}\in C^{0}([0,1],(C^{1}({\mathbb{R}}^{d}))^{d})$, define

$$
\hat{b}_{\mathsf{F}}(t,x)=\hat{b}(t,x)+\epsilon\hat{s}(t,x),\qquad\hat{v}(t,x)=\hat{b}(t,x)+\gamma(t)\dot{\gamma}(t)\hat{s}(t,x)
$$

where the function $\gamma$ satisfies the properties listed in Definition 2.1. Let $\hat{\rho}$ denote the solution to the Fokker-Planck equation

$$
\partial_{t}\hat{\rho}+\nabla\cdot(\hat{b}_{\mathsf{F}}\hat{\rho})=\epsilon\Delta\hat{\rho},\qquad\hat{\rho}(0)=\rho_{0}.
$$

Then,

$$
\mathsf{KL}(\rho_{1}\>\|\>\hat{\rho}(1))\leq\frac{1}{2\epsilon}\left(\mathcal{L}_{b}[\hat{b}]-\min_{\hat{b}}\mathcal{L}_{b}[\hat{b}]\right)+\frac{\epsilon}{2}\left(\mathcal{L}_{s}[\hat{s}]-\min_{\hat{s}}\mathcal{L}_{s}[\hat{s}]\right),
$$

where $\mathcal{L}_{b}[\hat{b}]$ and $\mathcal{L}_{s}[\hat{s}]$ are the objective functions defined in (2.13) and (2.16), and

$$
\mathsf{KL}(\rho_{1}\>\|\>\hat{\rho}(1))\leq\frac{1}{2\epsilon}\left(\mathcal{L}_{v}[\hat{v}]-\min_{\hat{v}}\mathcal{L}_{v}[\hat{v}]\right)+\frac{\sup_{t\in[0,1]}(\gamma(t)\dot{\gamma}(t)-\epsilon)^{2}}{2\epsilon}\left(\mathcal{L}_{s}[\hat{s}]-\min_{\hat{v}}\mathcal{L}_{s}[\hat{s}]\right).
$$

where $\mathcal{L}_{v}[\hat{v}]$ is the objective function defined in (2.29).

###### Remark 2.15 (Generative modeling).

The above results have practical ramifications for generative modeling. In particular, they show that minimizing either the losses (2.13) and (2.16) or (2.29) and (2.16) maximize the likelihood of the stochastic generative model

$$
d\hat{X}_{t}^{\mathsf{F}}=\left(\hat{b}(t,\hat{X}_{t}^{\mathsf{F}})+\epsilon\hat{s}(t,\hat{X}_{t}^{\mathsf{F}})\right)dt+\sqrt{2\epsilon}dW_{t},
$$

but that minimizing the objective (2.13) is insufficient in general to maximize the likelihood of the deterministic generative model

$$
\dot{\hat{X}}_{t}=\hat{b}(t,\hat{X}_{t}).
$$

Moreover, they show that, when learning $\hat{b}$ and $\hat{s}$, the choice of $\epsilon$ that minimizes the upper bound is given by

$$
\epsilon^{*}=\left(\frac{\mathcal{L}_{b}[\hat{b}]-\min_{\hat{b}}\mathcal{L}_{b}[\hat{b}]}{\mathcal{L}_{s}[\hat{s}]-\min_{\hat{s}}\mathcal{L}_{s}[\hat{s}]}\right)^{1/2},
$$

so that $\epsilon^{*}>1$ if the score is learned to higher accuracy than $\hat{b}$ and $\epsilon^{*}<1$ in the opposite situation. Note that (2.49) suggests to take $\epsilon=0$ if $\hat{b}$ is learned perfectly but $\hat{s}$ is not, and send $\epsilon\to\infty$ in the opposite situation. While taking $\epsilon=0$ is achievable in practice and leads to the ODE (2.32), taking $\epsilon\to\infty$ is not, as increasing $\epsilon$ increases the expense of the numerical integration in (2.33) and (2.34).

### 2.5 Density estimation and cross-entropy calculation

It is well-known that the solution of the TE (2.9) can be expressed in terms of the solution to the probability flow ODE (2.32); for completeness, we now recall this fact: lemmaTEs Given the velocity field $\hat{b}\in C^{0}([0,1],(C^{1}({\mathbb{R}}^{d}))^{d})$, let $\hat{\rho}$ satisfy the transport equation

$$
\partial_{t}\hat{\rho}+\nabla\cdot(\hat{b}\hat{\rho})=0,
$$

and let $X_{s,t}(x)$ solve the ODE

$$
\frac{d}{dt}X_{s,t}(x)=b(t,X_{s,t}(x)),\qquad X{{}_{s,s}(x)=x,\qquad t,s\in[0,1]}
$$

Then, given the PDFs $\rho_{0}$ and $\rho_{1}$:

1. The solution to (2.50) for the initial condition $\hat{\rho}(0)=\rho_{0}$ is given at any time $t\in[0,1]$ by
	$$
	\hat{\rho}(t,x)=\exp\left(-\int_{0}^{t}\nabla\cdot b(\tau,X_{t,\tau}(x))d\tau\right)\rho_{0}(X_{t,0}(x))
	$$
2. The solution to (2.50) for the final condition $\hat{\rho}(1)=\rho_{1}$ is given at any time $t\in[0,1]$ by
	$$
	\hat{\rho}(t,x)=\exp\left(\int_{t}^{1}\nabla\cdot b(\tau,X_{t,\tau}(x))d\tau\right)\rho_{1}(X_{t,1}(x))
	$$

The proof of Lemma 2.5 can be found in Appendix B.4. Interestingly, we can obtain a similar result for the solution of the forward and backward FPEs in (2.20) and (2.22). These results make use of auxiliary forward and backward SDEs in which the roles of the forward and backward drifts are switched:

theoremFK Given $\epsilon>0$ and two velocity fields $\hat{b},\hat{s}\in C^{0}([0,1],(C^{1}({\mathbb{R}}^{d}))^{d})$, define

$$
\hat{b}_{\mathsf{F}}(t,x)=\hat{b}(t,x)+\epsilon\hat{s}(t,x),\qquad\hat{b}_{\mathsf{B}}(t,x)=\hat{b}(t,x)-\epsilon\hat{s}(t,x),
$$

and let $Y^{\mathsf{F}}_{t}$ and $Y^{\mathsf{B}}_{t}$ denote solutions of the following forward and backward SDEs:

$$
dY^{\mathsf{F}}_{t}=b_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})dt+\sqrt{2\epsilon}dW_{t},
$$

to be solved forward in time from the initial condition $Y^{\mathsf{F}}_{t=0}=x$ independent of $W$; and

$$
dY^{\mathsf{B}}_{t}=b_{\mathsf{F}}(t,Y^{\mathsf{B}}_{t})dt+\sqrt{2\epsilon}dW^{\mathsf{B}}_{t},\quad W_{t}^{\mathsf{B}}=-W_{1-t},
$$

to be solved backwards in time from the final condition $Y^{\mathsf{B}}_{t=1}=x$ independent of $W^{\mathsf{B}}$. Then, given the densities $\rho_{0}$ and $\rho_{1}$:

1. The solution to the forward FPE
	$$
	\partial_{t}\hat{\rho}_{\mathsf{F}}+\nabla\cdot(\hat{b}_{\mathsf{F}}\hat{\rho}_{\mathsf{F}})=\epsilon\Delta\hat{\rho}_{\mathsf{F}},\qquad\hat{\rho}_{\mathsf{F}}(0)=\rho_{0},
	$$
	can be expressed at $t=1$ as
	$$
	\hat{\rho}_{\mathsf{F}}(1,x)={\mathbb{E}}_{\mathsf{B}}^{x}\left(\exp\left(-\int_{0}^{1}\nabla\cdot\hat{b}_{\mathsf{F}}(t,Y^{\mathsf{B}}_{t})dt\right)\rho_{0}(Y_{t=0}^{\mathsf{B}})\right),
	$$
	where ${\mathbb{E}}_{\mathsf{B}}^{x}$ denotes expectation on the path of $Y_{t}^{\mathsf{B}}$ conditional on the event $Y_{t=1}^{\mathsf{B}}=x$.
2. The solution to the backward FPE
	$$
	\partial_{t}\hat{\rho}_{\mathsf{B}}+\nabla\cdot(\hat{b}_{\mathsf{B}}\hat{\rho}_{\mathsf{B}})=-\epsilon\Delta\hat{\rho}_{\mathsf{B}},\qquad\hat{\rho}_{\mathsf{B}}(1)=\rho_{1},
	$$
	can be expressed at any $t=0$ as
	$$
	\hat{\rho}_{\mathsf{B}}(0,x)={\mathbb{E}}_{\mathsf{F}}^{x}\left(\exp\left(\int_{0}^{1}\nabla\cdot\hat{b}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})dt\right)\rho_{1}(Y^{\mathsf{F}}_{t=1})\right),
	$$
	where ${\mathbb{E}}_{\mathsf{F}}^{x}$ denotes expectation on the path of $Y^{\mathsf{F}}_{t}$ conditional on $Y^{\mathsf{F}}_{t=0}=x$.

The proof of Theorem 2.5 can be found in Appendix B.4. Note that to generate data from either $\hat{\rho}_{\mathsf{F}}(1)$ or $\hat{\rho}_{\mathsf{B}}(0)$ assuming that we can sample exactly the PDF at the other end, i.e. $\rho_{0}$ and $\rho_{1}$ respectively, we would still rely on the equivalent of the forward and backward SDE in (2.33) and (2.34), now used with the approximate drifts in (2.54), i.e.

$$
d\hat{X}^{\mathsf{F}}_{t}=\hat{b}_{\mathsf{F}}(t,\hat{X}^{\mathsf{F}}_{t})dt+\sqrt{2\epsilon}dW_{t},
$$

and

$$
d\hat{X}^{\mathsf{B}}_{t}=\hat{b}_{\mathsf{B}}(t,\hat{X}^{\mathsf{B}}_{t})dt+\sqrt{2\epsilon}dW^{\mathsf{B}}_{t},\quad W_{t}^{\mathsf{B}}=-W_{1-t},
$$

If we solve (2.61) forward in time from initial data $\hat{X}^{\mathsf{F}}_{t=0}\sim\rho_{0}$, we then have $\hat{X}^{\mathsf{F}}_{t=1}\sim\hat{\rho}_{\mathsf{F}}(1)$ where $\hat{\rho}_{\mathsf{F}}$ is the solution to the forward FPE (2.57). Similarly If we solve (2.62) backward in time from final data $\hat{X}^{\mathsf{B}}_{t=1}\sim\rho_{1}$, we then have $\hat{X}^{\mathsf{B}}_{t=0}\sim\hat{\rho}_{\mathsf{B}}(0)$ where $\hat{\rho}_{\mathsf{B}}$ is the solution to the backward FPE (2.59).

The results of Lemma 2.5 and Theorem 2.5 can be used to test the quality of samples generated by either the ODE (2.32) or the forward and backward SDEs (2.33) and (2.34). In particular, the following two results are direct consequences of Lemma 2.5 and Theorem 2.5, respectively:

corollarycrossentode Under the same conditions as Lemma 2.5, if $\hat{\rho}(0)=\rho_{0}$, the cross-entropy of $\hat{\rho}(1)$ relative to $\rho_{1}$ is given by

$$
\displaystyle\mathsf{H}(\rho_{1}\>\|\>\hat{\rho}(1))
$$
 
$$
\displaystyle=-\int_{{\mathbb{R}}^{d}}\log\hat{\rho}(1,x)\rho_{1}(x)dx
$$
 
$$
\displaystyle={\mathbb{E}}_{1}\int_{0}^{1}\nabla\cdot b(\tau,X_{1,\tau}(x_{1}))d\tau-{\mathbb{E}}_{1}\log\rho_{0}(X_{1,0}(x_{1}))
$$

where ${\mathbb{E}}_{1}$ denotes an expectation over $x_{1}\sim\rho_{1}$. Similarly, if $\hat{\rho}(1)=\rho_{1}$, the cross-entropy of $\hat{\rho}(0)$ relative to $\rho_{0}$ is given by

$$
\displaystyle\mathsf{H}(\rho_{0}\>\|\>\hat{\rho}(0))
$$
 
$$
\displaystyle=-\int_{{\mathbb{R}}^{d}}\log\hat{\rho}(0,x)\rho_{0}(x)dx
$$
 
$$
\displaystyle=-{\mathbb{E}}_{0}\int_{0}^{1}\nabla\cdot b(\tau,X_{0,\tau}(x_{0}))d\tau-{\mathbb{E}}_{0}\log\rho_{1}(X_{0,1}(x_{0}))
$$

where ${\mathbb{E}}_{0}$ denotes an expectation over $x_{0}\sim\rho_{0}$.

corollarycrossentsde Under the same conditions as Theorem 2.5, the cross-entropy of $\hat{\rho}_{\mathsf{F}}(1)$ relative to $\rho_{1}$ is given by

$$
\displaystyle\mathsf{H}(\rho_{1}\>\|\>\hat{\rho}_{\mathsf{F}}(1))
$$
 
$$
\displaystyle=-\int_{{\mathbb{R}}^{d}}\log\hat{\rho}_{\mathsf{F}}(1,x)\rho_{1}(x)dx
$$
 
$$
\displaystyle=-{\mathbb{E}}_{1}\log{\mathbb{E}}_{\mathsf{B}}^{x_{1}}\left(\exp\left(-\int_{0}^{1}\nabla\cdot b_{\mathsf{F}}(t,Y^{\mathsf{B}}_{t})dt\right)\rho_{0}(Y_{t=0}^{\mathsf{B}})\right),
$$

where ${\mathbb{E}}_{\mathsf{B}}^{x_{1}}$ denotes an expectation over $Y^{\mathsf{B}}_{t}$ conditioned on the event $Y^{\mathsf{B}}_{t=1}=x_{1}$, and ${\mathbb{E}}_{1}$ denotes an expectation over $x_{1}\sim\rho_{1}$. Similarly, the cross-entropy of $\hat{\rho}_{\mathsf{B}}(0)$ relative to $\rho_{0}$ is given by

$$
\displaystyle\mathsf{H}(\rho_{0}\>\|\>\hat{\rho}_{\mathsf{B}}(0))
$$
 
$$
\displaystyle=-\int_{{\mathbb{R}}^{d}}\log\hat{\rho}_{\mathsf{B}}(0,x)\rho_{0}(x)dx
$$
 
$$
\displaystyle=-{\mathbb{E}}_{0}\log{\mathbb{E}}_{\mathsf{F}}^{x_{0}}\left(\exp\left(\int_{0}^{1}\nabla\cdot b_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})dt\right)\rho_{1}(Y^{\mathsf{F}}_{t=1})\right),
$$

where ${\mathbb{E}}_{\mathsf{B}}^{x_{0}}$ denotes an expectation over $Y^{\mathsf{F}}_{t}$ conditioned on the event $Y^{\mathsf{F}}_{t=0}=x_{0}$, and ${\mathbb{E}}_{0}$ denotes an expectation over $x_{0}\sim\rho_{0}$.

If in (2.63), (2.64), (2.65), and (2.66) we approximate the expectations ${\mathbb{E}}_{0}$ and ${\mathbb{E}}_{1}$ over $\rho_{0}$ and $\rho_{1}$ by empirical expectations over the available data, these equations allow us to cross-validate different approximations of $\hat{b}$ and $\hat{s}$, as well as to compare the cross-entropies of densities evolved by the TE (2.50) with those of the forward and backward FPEs (2.57) and (2.59).

###### Remark 2.16.

When using (2.65) and (2.66) in practice, taking the $\log$ of the expectations ${\mathbb{E}}_{\mathsf{B}}^{x_{1}}$ and ${\mathbb{E}}_{\mathsf{F}}^{x_{0}}$ may create difficulties, such as when using Hutchinson’s trace estimator to compute the divergence of $b_{\mathsf{F}}$ or $b_{\mathsf{B}}$, which will introduce a bias. One way to remove this bias is to use Jensen’s inequality, which leads to the upper bounds

$$
\displaystyle\mathsf{H}(\rho_{1}\>\|\>\hat{\rho}_{\mathsf{F}}(1))\leq\int_{0}^{1}{\mathbb{E}}_{1}{\mathbb{E}}_{\mathsf{B}}^{x_{1}}\nabla\cdot b_{\mathsf{F}}(t,Y^{\mathsf{B}}_{t})dt-{\mathbb{E}}_{1}{\mathbb{E}}_{\mathsf{B}}^{x_{1}}\log\rho_{0}(Y_{t=0}^{\mathsf{B}}),
$$

and

$$
\displaystyle\mathsf{H}(\rho_{0}\>\|\>\hat{\rho}_{\mathsf{B}}(0))\leq-{\mathbb{E}}_{0}{\mathbb{E}}_{\mathsf{F}}^{x_{0}}\int_{0}^{1}\nabla\cdot b_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})dt-{\mathbb{E}}_{0}{\mathbb{E}}_{\mathsf{F}}^{x_{0}}\log\rho_{1}(Y^{\mathsf{F}}_{t=1}).
$$

However, these bounds are not sharp in general – in fact, using calculations similar to the one presented in the proof of Theorem 2.5, we can derive exact expressions that capture precisely what is lost when applying Jensen’s inequality:

$$
\displaystyle\mathsf{H}(\rho_{1}\>\|\>\hat{\rho}_{\mathsf{F}}(1))=\int_{0}^{1}{\mathbb{E}}_{1}{\mathbb{E}}_{\mathsf{B}}^{x_{1}}\left(\nabla\cdot b_{\mathsf{F}}(t,Y^{\mathsf{B}}_{t})-\epsilon|\nabla\log\hat{\rho}_{\mathsf{F}}(t,Y_{t}^{\mathsf{B}})|^{2}\right)dt-{\mathbb{E}}_{1}{\mathbb{E}}_{\mathsf{B}}^{x_{1}}\log\rho_{0}(Y_{t=0}^{\mathsf{B}}),
$$

and

$$
\displaystyle\mathsf{H}(\rho_{0}\>\|\>\hat{\rho}_{\mathsf{B}}(0))=-{\mathbb{E}}_{0}{\mathbb{E}}_{\mathsf{F}}^{x_{0}}\int_{0}^{1}\left(\nabla\cdot b_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})+\epsilon|\nabla\log\hat{\rho}_{\mathsf{B}}(t,Y_{t}^{\mathsf{F}})|^{2}\right)dt-{\mathbb{E}}_{0}{\mathbb{E}}_{\mathsf{F}}^{x_{0}}\log\rho_{1}(Y^{\mathsf{F}}_{t=1}).
$$

Unfortunately, since $\nabla\log\hat{\rho}_{\mathsf{F}}\not=\hat{s}$ and $\nabla\log\hat{\rho}_{\mathsf{B}}\not=\hat{s}$ in general due to approximation errors, we do not know how to estimate the extra terms on the right-hand side of (2.69) and (2.70). One possibility is to use $\hat{s}$ as a proxy for $\nabla\log\hat{\rho}_{\mathsf{F}}$ and $\nabla\log\hat{\rho}_{\mathsf{B}}$, which may be useful in practice, but this approximation is uncontrolled in general.

## 3 Instantiations and extensions

In this section, we instantiate the stochastic interpolant framework discussed in Section 2.

### 3.1 Diffusive interpolants

Recently, there has been a surge of interest in the construction of generative models through diffusive bridge processes [^8] [^51] [^42] [^55]. In this section, we connect these approaches with our own, highlighting that stochastic interpolants allow us to manipulate certain bridge processes in a simpler and more direct manner. We also show that this perspective leads to a generative process that samples any target density $\rho_{1}$ by pushing a point mass at any $x_{0}\in{\mathbb{R}}^{d}$ through an SDE. We begin by introducing a new kind of interpolant:

###### Definition 3.1 (Diffusive interpolant).

Given two probability density functions $\rho_{0},\rho_{1}:{{\mathbb{R}}^{d}}\rightarrow{\mathbb{R}}_{\geq 0}$, a diffusive interpolant between $\rho_{0}$ and $\rho_{1}$ is a stochastic process $x^{\mathsf{d}}_{t}$ defined as

$$
x^{\mathsf{d}}_{t}=I(t,x_{0},x_{1})+\sqrt{2a(t)}B_{t},\qquad t\in[0,1],
$$

where:

1. $I(t,x_{0},x_{1})$ is as in Definition 2.1;
2. $(x_{0},x_{1})\sim\nu$ with $\nu$ satisfying (2.3) in Definition 2.1;
3. $a(t)\in C^{2}([0,1])$ with $a(0)>0$ and $a(t)\geq 0$ for all $t\in(0,1]$, and;
4. $B_{t}$ is a standard Brownian bridge process, independent of $x_{0}$ and $x_{1}$.

Pathwise, (3.1) is different from the stochastic interpolant introduced in Definition 2.1: in particular, $x^{\mathsf{d}}_{t}$ is continuous but not differentiable in time. At the same time, since $B_{t}$ is a Gaussian process with mean zero and variance ${\mathbb{E}}B^{2}_{t}=t(1-t)$, (3.1) has the same single-time statistics and time-dependent density $\rho(t,x)$ as the stochastic interpolant (2.1) if we set $\gamma(t)=\sqrt{2a(t)t(1-t)}$, i.e.

$$
x_{t}=I(t,x_{0},x_{1})+\sqrt{2a(t)t(1-t)}z\quad\text{with}\quad(x_{0},x_{1})\sim\nu,\ z\sim{\sf N}(0,\text{\it Id}),\ (x_{0},x_{1})\perp z.
$$

As a result, (3.1) and (3.2) lead to the same generative models. Technically, it is easier to work with (3.2) than with (3.1), because it avoids the use of Itô calculus, and enables direct sampling of $x_{t}$ using samples from $\rho_{0}$, $\rho_{1}$, and $\mathsf{N}(0,\text{\it Id})$. However, (3.1) sheds light on some interesting properties of the generative models based on (3.2), i.e. stochastic interpolants with $\gamma(t)=\sqrt{2a(t)t(1-t)}$. To see why, we now re-derive the transport equation for the density $\rho(t,x)$ shared by (3.1) and (3.2) using the relation (3.1). For simplicity, we focus on the case where $a(t)$ is constant in time, i.e. we set $a(t)=a>0$ in (3.1).

To begin, recall that the Brownian Bridge $B_{t}$ can be expressed in terms of the Wiener process $W_{t}$ as $B_{t}=W_{t}-tW_{t=1}$. Moreover, it satisfies the SDE (obtained, for example, by conditioning on $B_{t=1}=0$ via Doob’s $h$ -transform [^18]):

$$
dB_{t}=-\frac{B_{t}}{1-t}dt+dW_{t},\qquad B_{t=0}=0.
$$

A direct application of Itô’s formula implies that

$$
de^{ik\cdot x_{t}^{\mathsf{d}}}=ik\cdot\Big{(}\partial_{t}I(t,x_{0},x_{1})-\frac{\sqrt{2a}B_{t}}{1-t}\Big{)}e^{ik\cdot x_{t}^{\mathsf{d}}}dt-a|k|^{2}e^{ik\cdot x_{t}^{\mathsf{d}}}dt+\sqrt{2a}ik\cdot dW_{t}e^{ik\cdot x_{t}^{\mathsf{d}}}.
$$

Taking the expectation of this expression and using the independence between $(x_{0},x_{1})$ and $B_{t}$, we deduce that

$$
\partial_{t}{\mathbb{E}}e^{ik\cdot x_{t}^{\mathsf{d}}}=ik\cdot{\mathbb{E}}\Big{(}\Big{(}\partial_{t}I(t,x_{0},x_{1})-\frac{\sqrt{2a}B_{t}}{1-t}\Big{)}e^{ik\cdot x_{t}^{\mathsf{d}}}\Big{)}-a|k|^{2}{\mathbb{E}}e^{ik\cdot x_{t}^{\mathsf{d}}}.
$$

Since for all fixed $t\in[0,1]$ we have $B_{t}\stackrel{{\scriptstyle d}}{{=}}\sqrt{t(1-t)}z$ and $x^{\mathsf{d}}_{t}\stackrel{{\scriptstyle d}}{{=}}x_{t}$ with $x_{t}$ defined in (3.2), the time derivative (3.5) can also be written as

$$
\partial_{t}{\mathbb{E}}e^{ik\cdot x_{t}}=ik\cdot{\mathbb{E}}\Big{(}\Big{(}\partial_{t}I(t,x_{0},x_{1})-\frac{\sqrt{2at}\,z}{\sqrt{1-t}}\Big{)}e^{ik\cdot x_{t}}\Big{)}-a|k|^{2}{\mathbb{E}}e^{ik\cdot x_{t}}.
$$

Moreover, since by definition of their probability density we have ${\mathbb{E}}e^{ik\cdot x_{t}^{\mathsf{d}}}={\mathbb{E}}e^{ik\cdot x_{t}}=\int_{{\mathbb{R}}^{d}}e^{ik\cdot x}\rho(t,x)dx$, we can deduce from (3.6) that $\rho(t)$ satisfies

$$
\partial_{t}\rho+\nabla\cdot(u\rho)=a\Delta\rho,
$$

where we defined

$$
u(t,x)={\mathbb{E}}\Big{(}\partial_{t}I(t,x_{0},x_{1})-\frac{\sqrt{2at}\,z}{\sqrt{1-t}}\Big{|}x_{t}=x\Big{)}.
$$

For the interpolant $x_{t}$ in (3.2), we have from the definitions of $b$ and $s$ in (2.10) and (2.14) that

$$
\displaystyle b(t,x)
$$
 
$$
\displaystyle={\mathbb{E}}\Big{(}\partial_{t}I(t,x_{0},x_{1})+\frac{a(1-2t)z}{\sqrt{2t(1-t)}}\Big{|}x_{t}=x\Big{)},
$$
$$
\displaystyle s(t,x)
$$
 
$$
\displaystyle=\nabla\log\rho(t,x)=-\frac{1}{\sqrt{2at(1-t)}}{\mathbb{E}}(z|x_{t}=x),
$$

As a result, $u-s=b$ and (3.7) can also be written as the TE (2.9) using $\Delta\rho=\nabla\cdot(s\rho)$.

Remarkably, the drift $u$ defined in (3.8) remains non-singular for all $t\in[0,1]$ (including $t=0$) even if $\rho_{0}$ is replaced by a point mass at $x_{0}$; by contrast, both $b$ and $s$ are singular at $t=0$ in this case. Hence, the SDE associated with the FPE (3.7) provides us with a generative model that samples $\rho_{1}$ from a base measure concentrated at a single $x_{0}$ (i.e. such that the density $\rho_{0}$ is replaced by a point mass measure at $x=x_{0}$). We formalize this result in the following theorem:

theoremdiffgen Assume that $I(t,x_{0},x_{1})=x_{0}$ for $t\in[0,\delta]$ with some $\delta\in(0,1]$. Given any $a>0$, let

$$
u^{\mathsf{d}}(t,x,x_{0})={\mathbb{E}}_{x_{1},z}\Big{(}\partial_{t}I(t,x_{0},x_{1})-\frac{\sqrt{2at}\,z}{\sqrt{1-t}}\Big{|}x_{t}=x\Big{)},
$$

where $x_{t}$ is given by (3.2) and where ${\mathbb{E}}_{x_{1},z}(\cdot|x_{t}=x)$ denotes an expectation over $x_{1}\sim\rho_{1}\perp z\sim\mathsf{N}(0,\text{\it Id})$ conditioned on $x_{t}=x$ with $x_{0}\in{\mathbb{R}}^{d}$ fixed. Then $u^{\mathsf{d}}(\cdot,\cdot,x_{0})\in C^{0}([0,1];(C^{p}({\mathbb{R}}^{d}))^{d})$ for any $p\in{\mathbb{N}}$ and $x_{0}\in{\mathbb{R}}^{d}$. Moreover, the solutions to the forward SDE

$$
dX_{t}^{\mathsf{d}}=u^{\mathsf{d}}(t,X_{t}^{\mathsf{d}},x_{0})dt+\sqrt{2a}\,dW_{t},\qquad X_{t=0}^{\mathsf{d}}=x_{0},
$$

are such that $X^{\mathsf{d}}_{t=1}\sim\rho_{1}$.

Note that the additional assumption we make on $I(t,x_{0},x_{1})$ is consistent with the requirements in Definition 2.1 and Assumption 2.5: this additional assumption is made for simplicity and can probably be relaxed to $\partial_{t}I(t=0,x_{0},x_{1})=0$.

The proof of Theorem 3.1 is given in Appendix B.5. It relies on the calculations that led to (3.8), along with the observation that at $t=0$ and $x=x_{0}$,

$$
u^{\mathsf{d}}(t=0,x_{0},x_{0})={\mathbb{E}}_{x_{1}}\left(\partial_{t}I(t=0,x_{0},x_{1})\right),
$$

whereas at $t=1$ and any $x\in{\mathbb{R}}^{d}$, we have

$$
u^{\mathsf{d}}(t=1,x,x_{0})=\partial_{t}I(t=1,x_{0},x)+2a\nabla\log\rho_{1}(x),
$$

which are both well-defined. To put the result in Theorem (3.1) in perspective, observe that no probability flow ODE with $b\in C^{0}([0,1];(C^{p}({\mathbb{R}}^{d}))^{d})$ can achieve the same feat as the diffusion in (3.11). This is because the solutions of such an ODE are unique, and therefore can only map $x_{0}$ onto a single point at time $t=1$. To the best of our knowledge, (3.11) is the first instance of an SDE that maps a point mass at $x_{0}$ into a density $\rho_{1}$ in finite time, and whose drift can be estimated by quadratic regression. Indeed, $u^{\mathsf{d}}(t,x,x_{0})$ is the unique minimizer of the objective function

$$
\mathcal{L}_{u^{\mathsf{d}}}[\hat{u}^{\mathsf{d}}]=\int_{0}^{1}{\mathbb{E}}_{x_{1},z}\left(|\hat{u}^{d}(t,x_{t},x_{0})|^{2}-2\Big{(}\partial_{t}I(t,x_{0},x_{1})-\frac{\sqrt{2at}z}{\sqrt{(1-t)}}\Big{)}\cdot\hat{u}^{d}(t,x_{t},x_{0})\right)dt.
$$

###### Remark 3.2 (Doob h-transform).

In principle, the approach above can be generalized to any stochastic bridge $B_{t}^{x_{0},x_{1}}$, which can be obtained from any SDE by conditioning its solution to satisfy $B_{t=0}^{x_{0},x_{1}}=x_{0}$ and $B_{t=1}^{x_{0},x_{1}}=x_{1}$ with the help of Doob’s $h$ -transform. In general, however, this construction cannot be made explicit, because the $h$ -transform is typically not available analytically. One approach would be to learn it, as proposed e.g. in [^51] [^8], but this adds an additional layer of difficulty that is avoided by the approach above.

### 3.2 One-sided interpolants for Gaussian ρ0subscript𝜌0\\rho\_{0}

A common choice of base density for generative modeling in the absence of prior information is to choose $\rho_{0}=\mathsf{N}(0,\text{\it Id})$. In this setting, we can group the effect of the latent variable $z$ with $x_{0}$. This leads to a simpler type of stochastic interpolant that, in particular, will enables us to instantiate score-based diffusion within our general framework (see Section 3.4).

###### Definition 3.3 (One-sided stochastic interpolant).

Given a probability density function $\rho_{1}:{{\mathbb{R}}^{d}}\rightarrow{\mathbb{R}}_{\geq 0}$, a one-sided stochastic interpolant between ${\sf N}(0,\text{\it Id})$ and $\rho_{1}$ is a stochastic process $x^{\mathsf{os}}_{t}$

$$
x^{\mathsf{os}}_{t}=\alpha(t)z+J(t,x_{1}),\qquad t\in[0,1]
$$

that fulfills the requirements:

1. $J\in C^{2}([0,1],C^{2}({\mathbb{R}}^{d})^{d})$ satisfies the boundary conditions $J(0,x_{1})=0$ and $J(1,x_{1})=x_{1}$.
2. $x_{1}$ and $z$ are independent random variables drawn from $\rho_{1}$ and ${\sf N}(0,\text{\it Id})$, respectively.
3. $\alpha:[0,1]\to{\mathbb{R}}$ satisfies $\alpha(0)=1$, $\alpha(1)=0$, $\alpha(t)>0$ for all $t\in[0,1)$, and $\alpha^{2}\in C^{2}([0,1])$.

By construction, $x^{\mathsf{os}}_{t=0}=z\sim{\sf N}(0,\text{\it Id})$ and $x^{\mathsf{os}}_{t=1}=x_{1}\sim\rho_{1}$, so that the distribution of the stochastic process $x^{\mathsf{os}}_{t}$ bridges ${\sf N}(0,\text{\it Id})$ and $\rho_{1}$. It is easy to see that the one-sided stochastic interpolant defined in (3.15) will have the same density as the stochastic interpolant defined in (2.1) if we set $I(t,x_{0},x_{1})=J_{t}(x_{1})+\delta(t)x_{0}$ and take $\delta^{2}(t)+\gamma^{2}(t)=\alpha^{2}(t)$. Restricting to this case, our earlier theoretical results apply where the velocity field $b$ defined in (2.10) becomes

$$
b(t,x)={\mathbb{E}}(\dot{\alpha}(t)z+\partial_{t}J(t,x_{1})|x^{\mathsf{os}}_{t}=x),
$$

and the quadratic objective in (2.13) becomes

$$
\mathcal{L}_{b}[\hat{b}]=\int_{0}^{1}{\mathbb{E}}\left(\tfrac{1}{2}|\hat{b}(t,x^{\mathsf{os}}_{t})|^{2}-\left(\dot{\alpha}(t)z+\partial_{t}J(t,x_{1})\right)\cdot\hat{b}(t,x^{\mathsf{os}}_{t})\right)dt.
$$

In the expression above, $x^{\mathsf{os}}_{t}$ is given by (3.15) and the expectation ${\mathbb{E}}$ is taken independently over $x_{1}\sim\rho_{1}$ and $z\sim{\sf N}(0,\text{\it Id})$. Similarly, the score is given by

$$
s(t,x)=-\alpha^{-1}(t)\eta_{z}(t,x),\qquad\eta_{z}(t,x)={\mathbb{E}}(z|x^{\mathsf{os}}_{t}=x),
$$

where $\eta_{z}(t,x)$ is the equivalent of the denoiser defined in (2.17). These functions are the unique minimizers of the objectives

$$
\mathcal{L}_{s}[\hat{s}]=\int_{0}^{1}{\mathbb{E}}\left(\tfrac{1}{2}|\hat{s}(t,x^{\mathsf{os}}_{t})|^{2}+\gamma^{-1}(t)z\cdot\hat{s}(t,x^{\mathsf{os}}_{t})\right)dt,
$$
 
$$
\mathcal{L}_{\eta_{z}}[\hat{\eta}_{z}]=\int_{0}^{1}{\mathbb{E}}\left(\tfrac{1}{2}|\hat{\eta}_{z}(t,x^{\mathsf{os}}_{t})|^{2}-z\cdot\hat{\eta}_{z}(t,x^{\mathsf{os}}_{t})\right)dt.
$$

Moreover, we can weaken Assumption 2.5 to the following requirement:

###### Assumption 3.4.

The density $\rho_{1}\in C^{2}({\mathbb{R}}^{d})$, satisfies $\rho_{1}(x)>0$ for all $x\in{\mathbb{R}}^{d}$, and:

$$
\int_{{\mathbb{R}}^{d}}|\nabla\log\rho_{1}(x)|^{2}\rho_{1}(x)dx<\infty.
$$

The function $J$ satisfies

$$
\displaystyle\exists C_{1}<\infty\ :\
$$
 
$$
\displaystyle|\partial_{t}J(t,x_{1})|\leq C_{1}|x_{1}|\quad
$$
 
$$
\displaystyle\text{for all}\quad(t,x_{1})\in[0,1]\times{\mathbb{R}}^{d},
$$

and

$$
\exists M_{1},M_{2}<\infty\ \ :\ \ {\mathbb{E}}\big{[}|\partial_{t}J(t,x_{1})|^{4}\big{]}\leq M_{1};\quad{\mathbb{E}}\big{[}|\partial^{2}_{t}J(t,x_{1})|^{2}\big{]}\leq M_{2},\quad\text{for all}\quad t\in[0,1],
$$

where the expectation is taken over $x_{1}\sim\rho_{1}$.

###### Remark 3.5.

The construction above can easily be generalized to the case where $\rho_{0}=\mathsf{N}(0,C_{0})$ with $C_{0}$ a positive-definite matrix. Without loss of generality, we can then assume that $C_{0}$ can be represented as $C_{0}=\sigma_{0}\sigma_{0}^{\mathsf{T}}$ where $\sigma_{0}$ is a lower-triangular matrix and replace (3.15)

$$
x^{\mathsf{os}}_{t}=\alpha(t)\sigma_{0}z+J(t,x_{1}),\qquad t\in[0,1],
$$

with $J$ and $\alpha$ satisfying the conditions listed in Definition 3.3 and where $z\sim{\sf N}(0,\text{\it Id})$.

### 3.3 Mirror interpolants

Another practically relevant setting is when the base and the target are the same density $\rho_{1}$. In this setting we can define a stochastic interpolant as:

###### Definition 3.6 (Mirror stochastic interpolant).

Given a probability density function $\rho_{1}:{{\mathbb{R}}^{d}}\rightarrow{\mathbb{R}}_{\geq 0}$, a mirror stochastic interpolant between $\rho_{0}$ and itself is a stochastic process $x^{\mathsf{mir}}_{t}$

$$
x^{\mathsf{mir}}_{t}=K(t,x_{1})+\gamma(t)z,\qquad t\in[0,1]
$$

that fulfills the requirements:

1. $K\in C^{2}([0,1],C^{2}({\mathbb{R}}^{d})^{d})$ satisfies the boundary conditions $K(0,x_{1})=x_{1}$ and $K(1,x_{1})=x_{1}$.
2. $x_{1}$ and $z$ are random variables drawn independently from $\rho_{1}$ and ${\sf N}(0,\text{\it Id})$, respectively.
3. $\gamma:[0,1]\to{\mathbb{R}}$ satisfies $\gamma(0)=\gamma(1)=0$, $\gamma(t)>0$ for all $t\in(0,1)$, and $\gamma^{2}\in C^{1}([0,1])$.

By construction, $x^{\mathsf{mir}}_{t=0}=x^{\mathsf{mir}}_{t=1}=x_{1}\sim\rho_{1}$, so that the distribution of the stochastic process $x^{\mathsf{mir}}_{t}$ bridges $\rho_{1}$ to itself. Note that a valid choice is $K(t,x_{1})=\alpha(t)x_{1}$ with $\alpha(0)=\alpha(1)=1$ (e.g. $\alpha(t)=1$): in this case, mirror interpolants are related to denoisers, as will be discussed in Section 5.2.

It is easy to see that our earlier theoretical results apply where the velocity field $b$ defined in (2.10) becomes

$$
b(t,x)={\mathbb{E}}(\partial_{t}K(t,x_{1})+\dot{\gamma}(t)z|x^{\mathsf{mir}}_{t}=x),
$$

and the quadratic objective in (2.13) becomes

$$
\mathcal{L}_{b}[\hat{b}]=\int_{0}^{1}{\mathbb{E}}\left(\tfrac{1}{2}|\hat{b}(t,x^{\mathsf{mir}}_{t})|^{2}-\left(\partial_{t}K(t,x_{1})+\dot{\gamma}(t)z\right)\cdot\hat{b}(t,x^{\mathsf{mir}}_{t})\right)dt.
$$

In the expression above, $x^{\mathsf{mir}}_{t}$ is given by (3.25) and the expectation ${\mathbb{E}}$ is taken independently over $x_{1}\sim\rho_{1}$ and $z\sim{\sf N}(0,\text{\it Id})$. Similarly, the score is given by

$$
s(t,x)=-\gamma^{-1}(t)\eta_{z}(t,x),\qquad\eta_{z}(t,x)={\mathbb{E}}(z|x^{\mathsf{mir}}_{t}=x),
$$

which are the unique minimizers of the objective functions

$$
\mathcal{L}_{s}[\hat{s}]=\int_{0}^{1}{\mathbb{E}}\left(\tfrac{1}{2}|\hat{s}(t,x^{\mathsf{mir}}_{t})|^{2}+\gamma^{-1}(t)z\cdot\hat{s}(t,x^{\mathsf{mir}}_{t})\right)dt.
$$
 
$$
\mathcal{L}_{\eta_{z}}[\hat{\eta}_{z}]=\int_{0}^{1}{\mathbb{E}}\left(\tfrac{1}{2}|\hat{\eta}_{z}(t,x^{\mathsf{mir}}_{t})|^{2}-z\cdot\hat{\eta}_{z}(t,x^{\mathsf{mir}}_{t})\right)dt.
$$

Moreover, we can weaken Assumption 2.5 to the following requirement:

###### Assumption 3.7.

The density $\rho_{1}\in C^{2}({\mathbb{R}}^{d})$ satisfies $\rho_{1}(x)>0$ for all $x\in{\mathbb{R}}^{d}$ and

$$
\int_{{\mathbb{R}}^{d}}|\nabla\log\rho_{1}(x)|^{2}\rho_{1}(x)dx<\infty.
$$

The function $K$ satisfies

$$
\displaystyle\exists C_{1}<\infty\ :\
$$
 
$$
\displaystyle|\partial_{t}K(t,x_{1})|\leq C_{1}|x_{1}|\quad
$$
 
$$
\displaystyle\text{for all}\quad(t,x_{1})\in[0,1]\times{\mathbb{R}}^{d},
$$

and

$$
\exists M_{1},M_{2}<\infty\ \ :\ \ {\mathbb{E}}\big{[}|\partial_{t}K(t,x_{1})|^{4}\big{]}\leq M_{1};\quad{\mathbb{E}}\big{[}|\partial^{2}_{t}K(t,x_{1})|^{2}\big{]}\leq M_{2},\quad\text{for all}\quad t\in[0,1],
$$

where the expectation is taken over $x_{1}\sim\rho_{1}$.

###### Remark 3.8.

Interestingly, if we take $K(t,x_{1})=x_{1}$, then $\partial_{t}K(t,x_{1})=0$, and the velocity field defined in (3.26) is completely defined by the denoiser $\eta_{z}$

$$
b(t,x)=\dot{\gamma}(t)\eta_{z}(t,x)
$$

Since the score $s$ also depends on $\eta_{z}$, this denoiser is the only quantity that needs to be learned.

###### Remark 3.9.

If $\rho_{1}$ is only accessible via empirical samples, mirror interpolants do not enable calculation of the functional form of $\rho_{1}$. A notable exception is if we set $K(t,x_{1})=0$ for $t\in[t_{1},t_{2}]$ with $0<t_{1}\leq t_{2}<1$: in that case, $x^{\mathsf{mir}}_{t}=\gamma(t)z\sim\gamma(t){\sf N}(0,\text{\it Id})$ for $t\in[t_{1},t_{2}]$, which gives us a reference density for comparison. In this setup, mirror interpolants essentially reduce to two one-sided interpolants glued together (with the second one time-reversed), or in fact a regular stochastic interpolant when $\rho_{0}=\rho_{1}$ and we set $I(t,x_{0},x_{1})=0$ for $t\in[t_{1},t_{2}]$.

### 3.4 Stochastic interpolants and Schrödinger bridges

The stochastic interpolant framework can also be used to solve the Schrödinger bridge problem. For background material on this problem, we refer the reader to [^38] and the references therein. Consistent with the overall viewpoint of this paper, we consider the hydrodynamic formulation of the Schrödinger bridge problem, in which the goal is to obtain a pair $(\rho,u)$, that solves the following optimization problem for a fixed $\epsilon>0$

$$
\displaystyle\min_{\hat{u},\hat{\rho}}\int_{0}^{1}\int_{{\mathbb{R}}^{d}}|\hat{u}(t,x)|^{2}\hat{\rho}(t,x)dxdt
$$
 
$$
\displaystyle\partial_{t}\hat{\rho}+\nabla\cdot\left(\hat{u}\hat{\rho}\right)=\epsilon\Delta\hat{\rho},\quad\hat{\rho}(0)=\rho_{0}\quad\hat{\rho}(1)=\rho_{1}
$$

Under our assumptions on $\rho_{0}$ and $\rho_{1}$ listed in Assumption 2.5, it is known (see e.g. Proposition 4.1 in [^38]) that (3.35) has a unique minimizer $(\rho,u=\nabla\lambda)$, with $(\rho,\lambda)$ classical solutions of the Euler-Lagrange equations:

$$
\displaystyle\partial_{t}\rho+\nabla\cdot\left(\nabla\lambda\rho\right)=\epsilon\Delta\rho,\quad\rho(0)=\rho_{0},\quad\rho(1)=\rho_{1},
$$
$$
\displaystyle\partial_{t}\lambda+\tfrac{1}{2}|\nabla\lambda|^{2}=-\epsilon\Delta\lambda.
$$

To proceed we will make the additional assumption that the solution $\rho$ to (3.36) can be reversibly mapped to a standard Gaussian:

###### Assumption 3.10.

There exists a reversible map $T:[0,1]\times{\mathbb{R}}^{d}\to{\mathbb{R}}^{d}$ with $T,T^{-1}\in C^{1}([0,1],(C^{d}({\mathbb{R}}^{d}))^{d})$ such that:

$$
\forall t\in[0,1]\quad:\quad z\sim{\sf N}(0,\text{\it Id})\ \Rightarrow\ T(t,z)\sim\rho(t);\quad x_{t}\sim\rho(t)\ \Rightarrow\ T^{-1}(t,x_{t})\sim{\sf N}(0,\text{\it Id}),
$$

where $\rho$ is the solution to (3.36).

We stress that the actual form of the map $T$ is not important for the arguments below. Assumption 3.10 can be used to show the existence of a stochastic interpolant whose density solves (3.36): lemmainterppdf If Assumption (3.10) holds, then the solution $\rho(t)$ to (3.36) is the density of the stochastic interpolant

$$
x_{t}=T(t,\alpha(t)T^{-1}(0,x_{0})+\beta(t)T^{-1}(1,x_{1}))+\gamma(t)z,
$$

as long as $\alpha^{2}(t)+\beta^{2}(t)+\gamma^{2}(t)=1$.

The proof is given in Appendix B.6: (3.38) corresponds to choosing $I(t,x_{0},x_{1})=T(t,\alpha(t)T^{-1}(t,x_{0})+\beta(t)T^{-1}(t,x_{1}))$ in (2.1). With the help of Lemma 3.10, we can establish the following result, which shows how to optimize over the function $I$ to solve the problem (3.35)

theoremsb Pick some $\gamma:[0,1]\to[0,1)$ such that $\gamma(0)=\gamma(1)=0$, $\gamma(t)>0$ for $t\in(0,1)$, $\gamma\in C^{2}((0,1))$ and $\gamma^{2}\in C^{1}([0,1])$, and let $\hat{x}_{t}=\hat{I}(t,x_{0},x_{1})+\gamma(t)z$, with $x_{0}\sim\rho_{0}$, $x_{1}\sim\rho_{1}$, and $z\sim{\sf N}(0,\text{\it Id})$ all independent. Consider the max-min problem over $\hat{I}\in C^{1}([0,1],(C^{1}({\mathbb{R}}^{d}\times{\mathbb{R}}^{d}))^{d})$ and $\hat{u}\in C^{0}([0,1],(C^{1}({\mathbb{R}}^{d}))^{d})$:

$$
\max_{\hat{I}}\min_{\hat{u}}\int_{0}^{1}{\mathbb{E}}\left(\tfrac{1}{2}|\hat{u}(t,\hat{x}_{t})|^{2}-\left(\partial_{t}\hat{I}(t,x_{0},x_{1})+(\dot{\gamma}(t)-\epsilon\gamma^{-1}(t))z\right)\cdot\hat{u}(t,\hat{x}_{t})\right)dt.
$$

If Assumption 3.10 holds, then all the optimizers $(I,u)$ of (3.39) are such that the density of the associated $x_{t}=I(t,x_{0},x_{1})+\gamma(t)z$ is the solution $\rho$ to (3.36). Moreover, $u=\nabla\lambda$, with $\lambda$ the solution to (3.36).

The proof is also given in Appendix B.6. Note that if we fix $\hat{I}$, the velocity $u$ minimizing this objective is the forward drift $b_{\mathsf{F}}$ defined in (2.21). Note also that if we set $\epsilon\to 0$, the minimizing velocity field is $b$ as defined in (2.10), and the max-min problem formally reduces to solving the optimal transport problem. In this case, Assumption 3.10 becomes more stringent, as we need to assume that that system (3.36) with $\epsilon=0$ (i.e. in the absence of the diffusive terms) has a classical solution. Theorem (3.10) gives a practical route towards solving the Schrödinger bridge problem with stochastic interpolants, and we leave the numerical investigation of this formulation to future work.

## 4 Spatially linear interpolants

In this section, we study the stochastic interpolants that are obtained when we specialize the function $I$ used in (2.1) to be linear in both $x_{0}$ and $x_{1}$, i.e. we consider

$$
x^{\mathsf{lin}}_{t}=\alpha(t)x_{0}+\beta(t)x_{1}+\gamma(t)z,
$$

where $(x_{0},x_{1})\sim\nu$ and $z\sim{\sf N}(0,\text{\it Id})$ with $(x_{0},x_{1})\perp z$, and $\alpha,\beta,\gamma^{2}\in C^{2}([0,1])$ satisfy the conditions

$$
\displaystyle\alpha(0)=\beta(1)=1;\quad\alpha(1)=\beta(0)=\gamma(0)=\gamma(1)=0;\quad\forall t\in(0,1)\ :\ \gamma(t)>0.
$$

Despite its simplicity, this setup offers significant design flexibility. The discussion highlights how the presence of the latent variable $\gamma(t)z$ can simplify the structure of the intermediate density $\rho(t)$. Since our ultimate aim is to investigate the properties of practical generative models built upon ODEs or SDEs, we will also study the effect of time-dependent diffusion coefficient $\epsilon(t)$, which controls the amplitude of the noise in a generative SDE. Throughout, to build intuition, we choose $\rho_{0}$ and $\rho_{1}$ to be Gaussian mixture densities, for which the drift coefficients can be computed analytically (see Appendix A). This enables us to visualize the effect of each choice on the resulting generative models.

### 4.1 Factorization of the velocity field

When the stochastic interpolant is of the form (4.1), the velocity $b$ and the score $s$ defined in (2.10) and (2.14) can both be expressed in terms of the following three conditional expectations (the third is the denoiser already defined in (2.17)):

$$
\eta_{0}(t,x)={\mathbb{E}}(x_{0}|x^{\mathsf{lin}}_{t}=x),\quad\eta_{1}(t,x)={\mathbb{E}}(x_{1}|x^{\mathsf{lin}}_{t}=x),\quad\eta_{z}(t,x)={\mathbb{E}}(z|x^{\mathsf{lin}}_{t}=x).
$$

Specifically, we have

$$
b(t,x)=\dot{\alpha}(t)\eta_{0}(t,x)+\dot{\beta}(t)\eta_{1}(t,x)+\dot{\gamma}(t)\eta_{z}(t,x),\qquad s(t,x)=-\gamma^{-1}(t)\eta_{z}(t,x).
$$

The second relation above holds for $t\in(0,1)$ (i.e. when $\gamma(t)\not=0$). Moreover, because ${\mathbb{E}}(x_{t}|x_{t}=x)=x$ by definition, the functions $\eta_{0}$, $\eta_{1}$, and $\eta_{z}$ satisfy

$$
\forall(t,x)\in[0,1]\times{\mathbb{R}}^{d}\quad:\quad\alpha(t)\eta_{0}(t,x)+\beta(t)\eta_{1}(t,x)+\gamma(t)\eta_{z}(t,x)=x.
$$

This enables us to reduce computational expense: given two of the $\eta$ ’s, the third can always be calculated via (4.5). Finally, it is easy to see that the functions $\eta_{0}$, $\eta_{1}$, and $\eta_{z}$ are the unique minimizers of the objectives

$$
\displaystyle\mathcal{L}_{\eta_{0}}(\hat{\eta}_{0})
$$
 
$$
\displaystyle=\int_{0}^{1}{\mathbb{E}}\left[\tfrac{1}{2}|\hat{\eta}_{0}(t,x^{\mathsf{lin}}_{t})|^{2}-x_{0}\cdot\hat{\eta}_{0}(t,x^{\mathsf{lin}}_{t})\right]dt,
$$
$$
\displaystyle\mathcal{L}_{\eta_{1}}(\hat{\eta}_{1})
$$
 
$$
\displaystyle=\int_{0}^{1}{\mathbb{E}}\left[\tfrac{1}{2}|\hat{\eta}_{1}(t,x^{\mathsf{lin}}_{t})|^{2}-x_{1}\cdot\hat{\eta}_{1}(t,x^{\mathsf{lin}}_{t})\right]dt,
$$
$$
\displaystyle\mathcal{L}_{\eta_{z}}(\hat{\eta}_{z})
$$
 
$$
\displaystyle=\int_{0}^{1}{\mathbb{E}}\left[\tfrac{1}{2}|\hat{\eta}_{z}(t,x^{\mathsf{lin}}_{t})|^{2}-z\cdot\hat{\eta}_{z}(t,x^{\mathsf{lin}}_{t})\right]dt,
$$

where $x^{\mathsf{lin}}_{t}$ is defined in (4.1) and the expectation is taken independently over $(x_{0},x_{1})\sim\nu$ and $z\sim{\sf N}(0,\text{\it Id}).$

<table><tbody><tr><td>Stochastic Interpolant</td><td></td><td><math><semantics><mrow><mi>α</mi> <mo></mo><mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></mrow> <apply><ci>𝛼</ci> <ci>𝑡</ci></apply> <annotation>\alpha(t)</annotation></semantics></math></td><td><math><semantics><mrow><mi>β</mi> <mo></mo><mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></mrow> <apply><ci>𝛽</ci> <ci>𝑡</ci></apply> <annotation>\beta(t)</annotation></semantics></math></td><td><math><semantics><mrow><mi>γ</mi> <mo></mo><mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></mrow> <apply><ci>𝛾</ci> <ci>𝑡</ci></apply> <annotation>\gamma(t)</annotation></semantics></math></td></tr><tr><td rowspan="3">Arbitrary <math><semantics><msub><mi>ρ</mi> <mn>0</mn></msub> <apply><csymbol>subscript</csymbol> <ci>𝜌</ci> <cn>0</cn></apply> <annotation>\rho_{0}</annotation></semantics></math> (two-sided)</td><td>linear</td><td><math><semantics><mrow><mn>1</mn> <mo>−</mo> <mi>t</mi></mrow> <apply><cn>1</cn> <ci>𝑡</ci></apply> <annotation>1-t</annotation></semantics></math></td><td><math><semantics><mi>t</mi> <ci>𝑡</ci> <annotation>t</annotation></semantics></math></td><td><math><semantics><msqrt><mrow><mi>a</mi> <mo></mo><mi>t</mi> <mo></mo><mrow><mo>(</mo><mrow><mn>1</mn> <mo>−</mo> <mi>t</mi></mrow><mo>)</mo></mrow></mrow></msqrt> <apply><apply><ci>𝑎</ci> <ci>𝑡</ci> <apply><cn>1</cn> <ci>𝑡</ci></apply></apply></apply> <annotation>\sqrt{at(1-t)}</annotation></semantics></math></td></tr><tr><td>trig</td><td><math><semantics><mrow><mi>cos</mi> <mo>⁡</mo> <mrow><mfrac><mi>π</mi> <mn>2</mn></mfrac> <mo></mo><mi>t</mi></mrow></mrow> <apply><apply><apply><ci>𝜋</ci> <cn>2</cn></apply> <ci>𝑡</ci></apply></apply> <annotation>\cos{\tfrac{\pi}{2}t}</annotation></semantics></math></td><td><math><semantics><mrow><mi>sin</mi> <mo>⁡</mo> <mrow><mfrac><mi>π</mi> <mn>2</mn></mfrac> <mo></mo><mi>t</mi></mrow></mrow> <apply><apply><apply><ci>𝜋</ci> <cn>2</cn></apply> <ci>𝑡</ci></apply></apply> <annotation>\sin{\tfrac{\pi}{2}t}</annotation></semantics></math></td><td><math><semantics><msqrt><mrow><mi>a</mi> <mo></mo><mi>t</mi> <mo></mo><mrow><mo>(</mo><mrow><mn>1</mn> <mo>−</mo> <mi>t</mi></mrow><mo>)</mo></mrow></mrow></msqrt> <apply><apply><ci>𝑎</ci> <ci>𝑡</ci> <apply><cn>1</cn> <ci>𝑡</ci></apply></apply></apply> <annotation>\sqrt{at(1-t)}</annotation></semantics></math></td></tr><tr><td>enc-dec</td><td><math><semantics><mrow><mrow><msup><mi>cos</mi> <mn>2</mn></msup> <mo>⁡</mo> <mrow><mo>(</mo><mrow><mi>π</mi> <mo></mo><mi>t</mi></mrow><mo>)</mo></mrow></mrow> <mo></mo><msub><mn>1</mn> <mrow><mo>[</mo><mn>0</mn><mo>,</mo><mfrac><mn>1</mn> <mn>2</mn></mfrac><mo>)</mo></mrow></msub> <mo></mo><mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></mrow> <apply><apply><apply><csymbol>superscript</csymbol> <cn>2</cn></apply> <apply><ci>𝜋</ci> <ci>𝑡</ci></apply></apply> <apply><csymbol>subscript</csymbol> <cn>1</cn> <interval><cn>0</cn> <apply><cn>1</cn> <cn>2</cn></apply></interval></apply> <ci>𝑡</ci></apply> <annotation>\cos^{2}(\pi t)1_{[0,\frac{1}{2})}(t)</annotation></semantics></math></td><td><math><semantics><mrow><mrow><msup><mi>cos</mi> <mn>2</mn></msup> <mo>⁡</mo> <mrow><mo>(</mo><mrow><mi>π</mi> <mo></mo><mi>t</mi></mrow><mo>)</mo></mrow></mrow> <mo></mo><msub><mn>1</mn> <mrow><mo>(</mo><mfrac><mn>1</mn> <mn>2</mn></mfrac><mo>,</mo><mn>1</mn><mo>]</mo></mrow></msub> <mo></mo><mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></mrow> <apply><apply><apply><csymbol>superscript</csymbol> <cn>2</cn></apply> <apply><ci>𝜋</ci> <ci>𝑡</ci></apply></apply> <apply><csymbol>subscript</csymbol> <cn>1</cn> <interval><apply><cn>1</cn> <cn>2</cn></apply> <cn>1</cn></interval></apply> <ci>𝑡</ci></apply> <annotation>\cos^{2}(\pi t)1_{(\frac{1}{2},1]}(t)</annotation></semantics></math></td><td><math><semantics><mrow><msup><mi>sin</mi> <mn>2</mn></msup> <mo>⁡</mo> <mrow><mo>(</mo><mrow><mi>π</mi> <mo></mo><mi>t</mi></mrow><mo>)</mo></mrow></mrow> <apply><apply><csymbol>superscript</csymbol> <cn>2</cn></apply> <apply><ci>𝜋</ci> <ci>𝑡</ci></apply></apply> <annotation>\sin^{2}(\pi t)</annotation></semantics></math></td></tr><tr><td rowspan="3">Gaussian <math><semantics><msub><mi>ρ</mi> <mn>0</mn></msub> <apply><csymbol>subscript</csymbol> <ci>𝜌</ci> <cn>0</cn></apply> <annotation>\rho_{0}</annotation></semantics></math> (one-sided)</td><td>linear</td><td><math><semantics><mrow><mn>1</mn> <mo>−</mo> <mi>t</mi></mrow> <apply><cn>1</cn> <ci>𝑡</ci></apply> <annotation>1-t</annotation></semantics></math></td><td><math><semantics><mi>t</mi> <ci>𝑡</ci> <annotation>t</annotation></semantics></math></td><td>0</td></tr><tr><td>trig</td><td><math><semantics><mrow><mi>cos</mi> <mo>⁡</mo> <mrow><mfrac><mi>π</mi> <mn>2</mn></mfrac> <mo></mo><mi>t</mi></mrow></mrow> <apply><apply><apply><ci>𝜋</ci> <cn>2</cn></apply> <ci>𝑡</ci></apply></apply> <annotation>\cos{\tfrac{\pi}{2}t}</annotation></semantics></math></td><td><math><semantics><mrow><mi>sin</mi> <mo>⁡</mo> <mrow><mfrac><mi>π</mi> <mn>2</mn></mfrac> <mo></mo><mi>t</mi></mrow></mrow> <apply><apply><apply><ci>𝜋</ci> <cn>2</cn></apply> <ci>𝑡</ci></apply></apply> <annotation>\sin{\tfrac{\pi}{2}t}</annotation></semantics></math></td><td>0</td></tr><tr><td>SBDM (VP)</td><td><math><semantics><msqrt><mrow><mn>1</mn> <mo>−</mo> <msup><mi>t</mi> <mn>2</mn></msup></mrow></msqrt> <apply><apply><cn>1</cn> <apply><csymbol>superscript</csymbol> <ci>𝑡</ci> <cn>2</cn></apply></apply></apply> <annotation>\sqrt{1-t^{2}}</annotation></semantics></math></td><td><math><semantics><mi>t</mi> <ci>𝑡</ci> <annotation>t</annotation></semantics></math></td><td>0</td></tr><tr><td>Mirror</td><td></td><td>0</td><td>1</td><td><math><semantics><msqrt><mrow><mi>a</mi> <mo></mo><mi>t</mi> <mo></mo><mrow><mo>(</mo><mrow><mn>1</mn> <mo>−</mo> <mi>t</mi></mrow><mo>)</mo></mrow></mrow></msqrt> <apply><apply><ci>𝑎</ci> <ci>𝑡</ci> <apply><cn>1</cn> <ci>𝑡</ci></apply></apply></apply> <annotation>\sqrt{at(1-t)}</annotation></semantics></math></td></tr></tbody></table>

Table 4: Linear interpolants. A table suggesting various linear interpolants. In general, this paper describes methods for arbitrary $\rho_{0}$ and $\rho_{1}$. In Section 4.4, we detail linear interpolants for one-sided generation, where $\rho_{0}$ is a Gaussian and the latent variable $z$ can be absorbed into $x_{0}$. Later, in Section 5.1, we discuss how to recast score-based diffusion models (SBDM) as linear one-sided interpolants, which leads to the expressions given in the table when using a variance-preserving parameterization. Linear mirror interpolants, where $\rho_{0}=\rho_{1}$ are equal, are defined by (3.25) with the choice $K(t,x_{1})=\alpha(t)x_{1}$.

### 4.2 Some specific design choices

It is useful to assume that both $\rho_{0}$ and $\rho_{1}$ have been scaled to have zero mean and identity covariance (which can be achieved in practice, for example, by an affine transformation of the data). In this case, the time-dependent mean and covariance of (4.1) are given by

$$
{\mathbb{E}}[x^{\mathsf{lin}}_{t}]=0,\qquad{\mathbb{E}}[x^{\mathsf{lin}}_{t}(x^{\mathsf{lin}}_{t})^{\mathsf{T}}]=(\alpha^{2}(t)+\beta^{2}(t)+\gamma^{2}(t))\text{\it Id}.
$$

Preserving the identity covariance at all times therefore leads to the constraint

$$
\forall t\in[0,1]\quad:\quad\alpha^{2}(t)+\beta^{2}(t)+\gamma^{2}(t)=1.
$$
![Refer to caption](https://ar5iv.labs.arxiv.org/html/2303.08797/assets/x4.png)

Figure 5: The functions α ( t ) 𝛼 𝑡 \\alpha(t), β 𝛽 \\beta(t), and γ 𝛾 \\gamma(t) for the linear ( 4.9 ), trigonometric ( 4.10 ) with = 2 1 − \\gamma(t)=\\sqrt{2t(1-t)}, Gaussian encoding-decoding ( 4.11 ), one-sided ( 3.15 ), and mirror ( 3.25 ) interpolants.

This choice is also sensible if $\rho_{0}$ and $\rho_{1}$ have covariances that are not the identity but are on a similar scale. In this case we no longer need to enforce (4.8) exactly, and could, for example, take three functions whose sum of squares is of order one. For definiteness, in the sequel we discuss possible choices that satisfy (4.8) exactly, with the understanding that the corresponding functions $\alpha$, $\beta$, and $\gamma$ could all be slightly modified without significantly affecting the conclusions.

#### Linear and trigonometric α𝛼\\alpha and β𝛽\\beta.

One way to ensure that (4.8) holds while maintaining the influence of $\rho_{0}$ and $\rho_{1}$ everywhere on $[0,1]$ except at the endpoints is to choose

$$
\alpha(t)=t,\qquad\beta(t)=1-t,\qquad\gamma(t)=\sqrt{2t(1-t)}.
$$

This choice was advocated in [^41], without the inclusion of the latent variable ($\gamma=0$). Another possibility that gives more leeway is to pick any $\gamma:[0,1]\to[0,1]$ and set

$$
\alpha(t)=\sqrt{1-\gamma^{2}(t)}\cos(\tfrac{1}{2}\pi t),\qquad\beta(t)=\sqrt{1-\gamma^{2}(t)}\sin(\tfrac{1}{2}\pi t).
$$

With $\gamma=0$, this was the choice preferred in [^2]. The PDF $\rho(t)$ obtained with the choices (4.9) and (4.10) when $\rho_{0}$ and $\rho_{1}$ are both Gaussian mixture densities are shown in Figure 6. As this example shows, when $\rho_{0}$ and $\rho_{1}$ have distinct complex features, these would be duplicated in $\rho(t)$ at intermediate times if not for the smoothing effect of the latent variable; this behavior is seen in Figure 6, where it is most prominent in the first row with $\gamma(t)=0$. From a statistical learning perspective, eliminating the formation of spurious features will simplify the estimation of the velocity field $b$, which becomes smoother as the formation of such features is suppressed.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2303.08797/assets/x5.png)

Figure 6: The effect of γ ( t ) z 𝛾 𝑡 𝑧 \\gamma(t)z on ρ 𝜌 \\rho(t). A visualization of how the choice of \\gamma(t) changes the density of x 𝗅𝗂𝗇 = α 0 + β 1 subscript superscript 𝑥 𝛼 𝛽 x^{\\mathsf{lin}}\_{t}=\\alpha(t)x\_{0}+\\beta(t)x\_{1}+\\gamma(t)z when \\rho\_{0} and \\rho\_{1} are Gaussian mixture densities with two modes and three modes, respectively. The first row depicts \\gamma(t)=0, which reduces to the stochastic interpolant developed in 2. This case forms a valid transport between, but produces spurious intermediate modes in. The second row depicts the choice of − \\gamma(t)=\\sqrt{2t(1-t)}. In this case, the spurious modes are partially damped by the addition of the latent variable, leading to a simpler. The final row shows the Gaussian encoding-decoding, which smoothly encodes into a standard normal distribution on the time interval \[, / \[0,1/2), which is then decoded into on the interval \] (1/2,1\]. In this case, no intermediate modes form in: the two modes in collide to form 𝖭 \\mathsf{N}(0,1) at t=\\tfrac{1}{2}, which then spreads into the three modes of. A visualization of individual sample trajectories from deterministic and stochastic generative models based on ODEs and SDEs whose solutions have density can be seen in Figure 7

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2303.08797/assets/x6.png)

Figure 7: The effect of ϵ italic-ϵ \\epsilon on sample trajectories. A visualization of how the choice of affects the sample trajectories obtained by solving the ODE ( 2.32 ) or the forward SDE ( 2.33 ). The set-up is the same as in Figure 6 ρ 0 subscript 𝜌 \\rho\_{0} and 1 \\rho\_{1} are taken to be the same Gaussian mixture densities, and the analytical expressions for b 𝑏 s 𝑠 are used. In the three panels in each column the value of γ 𝛾 \\gamma is the same, and each panel shows trajectories with different. Three specific trajectories from the same three initial conditions drawn from are also highlighted in white in every panel. As increases but stays the same, the density ( t ) 𝑡 \\rho(t) is unchanged, but the individual trajectories become increasingly stochastic. While all choices are equivalent with exact, Theorem 2.4 shows that nonzero values of provide control on the likelihood in terms of the error in when they are approximate.

#### Gaussian encoding-decoding.

A useful limiting case is to devolve the data from $\rho_{0}$ completely into noise by the halfway point $t=\tfrac{1}{2}$ and to reconstruct $\rho_{1}$ completely from noise starting from $t=\frac{1}{2}$. One choice that allows us to do so while satisfying (4.8) is

$$
\alpha(t)=\cos^{2}(\pi t)1_{[0,\frac{1}{2})}(t),\qquad\beta(t)=\cos^{2}(\pi t)1_{(\frac{1}{2},1]}(t),\qquad\gamma(t)=\sin^{2}(\pi t),
$$

where $1_{A}(t)$ is the indicator function of $A$, i.e. $1_{A}(t)=1$ if $t\in A$ and $1_{A}(t)=0$ otherwise. With this choice, it is easy to see that $x_{t=\frac{1}{2}}=\gamma(\tfrac{1}{2})z\sim{\sf N}(0,\gamma^{2}(\tfrac{1}{2}))$, which seamlessly glues together two interpolants: one between $\rho_{0}$ and a standard Gaussian, and one between a standard Gaussian and $\rho_{1}$.

Even though the choice (4.11) encodes $\rho_{0}$ into pure noise on the interval $[0,\tfrac{1}{2}]$, which is then decoded into $\rho_{1}$ on the interval $[\tfrac{1}{2},1]$ (and vice-versa when proceeding backwards in time), the resulting velocity $b$ still defines a single continuity equation that maps $\rho_{0}$ to $\rho_{1}$ on $[0,1]$. This is most clearly seen at the level of the probability flow (2.32), since its solution $X_{t}$ is a bijection between the initial and final conditions $X_{t=0}$ and $X_{t=1}$, but a similar pairing can also be observed in the solutions to the forward and backward SDEs (2.33) and (2.34), whose solutions at time $t=1$ or $t=0$ remain correlated with the initial or final condition used. This allows for a more direct means of image-to-image translation with diffusions when compared to the recent approach described in [^62]. The choice (4.11) is depicted in the final row of Figure 6, where no spurious modes form at all; individual sample trajectories of the deterministic and stochastic generative models based on ODEs and SDEs whose solutions have this $\rho(t)$ as density can be seen in the panels forming the third column in Figure 7. We note that the elimination of spurious intermediate modes can also be implemented by use of a data-adapted coupling $\nu(dx_{0},dx_{1})$, as considered in [^1].

Unsurprisingly, it is necessary to have $\gamma(t)>0$ for the choice (4.11): for $\gamma(t)=0$, the density $\rho(t)$ collapses to a Dirac measure at $t=\frac{1}{2}$. This consideration highlights that the inclusion of the latent variable $\gamma(t)z$ matters even for the deterministic dynamics (2.32), and its presence is distinct from the stochasticity inherent to the SDEs (2.33) and (2.34).

### 4.3 Impact of the latent variable γ(t)z𝛾𝑡𝑧\\gamma(t)z and the diffusion coefficient ϵ(t)italic-ϵ𝑡\\epsilon(t)

The stochastic interpolant framework enables us to discern the independent roles of the latent variable $\gamma(t)z$ and the diffusion coefficient $\epsilon(t)$ we use in a generative model. As shown in Theorem 2.2, the presence of the latent variable $\gamma(t)z$ for $\gamma\neq 0$ smooths both the density $\rho(t)$ and the velocity $b$ defined in (2.10) spatially. This provides a computational advantage at sample generation time because it simplifies the required numerical integration of (2.32), (2.33), and (2.34). Intuitively, this is because the density $\rho(t)$ of $x_{t}$ can be represented exactly as the density that would be obtained with $\gamma(t)=0$ convolved with $\mathsf{N}(0,\gamma^{2}(t)Id)$ at each $t\in(0,1)$. A comparison between the density $\rho(t)$ obtained with trigonometric interpolants with $\gamma(t)=0$ and $\gamma(t)=\sqrt{2t(1-t)}$ can be seen in the first and second row of Figure 6.

By contrast, the diffusion coefficient $\epsilon(t)$ leaves the density $\rho(t)$ unchanged, and only affects the way we sample it. In particular, the probability flow ODE (2.32) results in a map that pushes every $X_{t=0}=x_{0}$ onto a single $X_{t=1}=x_{1}$ and vice-versa. The forward SDE (2.33) maps each $X^{\mathsf{F}}_{t=0}=x_{0}$ onto an ensemble $X^{\mathsf{F}}_{t=1}$ whose spread is controlled by the amplitude of $\epsilon(t)$ (and similarly for the reversed SDE (2.34) that maps each $X^{\mathsf{B}}_{t=1}=x_{1}$ onto an ensemble $X^{\mathsf{B}}_{t=0}$). This ensemble is not distributed according to $\rho_{1}$ for finite $\epsilon(t)$ – like with the ODE, we need to sample initial conditions from $\rho_{0}$ to get solutions at time $t=1$ that sample $\rho_{1}$ – but its density converges towards $\rho_{1}$ as $\epsilon(t)\to\infty$. These features are illustrated in Figure 7.

###### Remark 4.1.

Another potential advantage of including the latent variable $\gamma(t)z$ is its impact on the velocity $b$ at the end points. Since $x_{t=0}=x_{0}$ and $x_{t=1}=x_{1}$, it is easy to see that the velocity $b$ of the linear interpolant $x_{t}$ defined in (4.1) satisfies

$$
\displaystyle b(0,x)
$$
 
$$
\displaystyle=\dot{\alpha}(0)x+\dot{\beta}(0){\mathbb{E}}[x_{1}]-\lim_{t\to 0}\gamma(t)\dot{\gamma}(t)s_{0}(x),
$$
$$
\displaystyle b(1,x)
$$
 
$$
\displaystyle=\dot{\alpha}(1){\mathbb{E}}[x_{0}]+\dot{\beta}(1)x-\lim_{t\to 1}\gamma(t)\dot{\gamma}(t)s_{1}(x),
$$

where $s_{0}=\nabla\log\rho_{0}$ and $s_{1}=\nabla\log\rho_{1}$. If $\gamma\in C^{2}([0,1])$, because $\gamma(0)=\gamma(1)=0$, the terms involving the scores $s_{0}$ and $s_{1}$ in these expressions vanish. Choosing $\gamma^{2}\in C^{1}([0,1])$ but $\gamma$ not differentiable at $t=0$ or $t=1$ leaves open the possibility that the limits remain nonzero. For example, if we take one of the choices discussed in Section 3.1, i.e.

$$
\gamma(t)=\sqrt{at(1-t)},\qquad a>0,
$$

we obtain

$$
\lim_{t\to 0}\gamma(t)\dot{\gamma}(t)=-\lim_{t\to 1}\gamma(t)\dot{\gamma}(t)=\frac{a}{2}.
$$

As a result, the choice (4.13) ensures that the velocity $b$ encodes information about the score of the densities $\rho_{0}$ and $\rho_{1}$ at the end points. We stress however that, while the choice of $\gamma(t)$ given in (4.13) is appealing because of its nontrivial influence on the velocity $b$ at the endpoints, the user is free to explore a variety of alternatives. We present some examples in Table 8, specifying the differentiability of $\gamma$ at $t=0$ and $t=1$. The function $\gamma(t)$ specified in (4.13) is the only featured case for which the contribution from the score is non-vanishing in the velocity $b$ at the endpoints. In Section 7, we illustrate on numerical examples that there are tradeoffs between different choices of $\gamma$, which might be directly related to this fact. When using the ODE as a generative model, the score is only felt through $b$, whereas it is explicit when using the SDE as a generative model.

| $\gamma(t):$ | $\sqrt{at(1-t)}$ | $t(1-t)$ | $\hat{\sigma}(t)$ | $\sin^{2}(\pi t)$ |
| --- | --- | --- | --- | --- |
| $C^{1}$ at $t=0,1$ | ✗ | ✓ | ✓ | ✓ |

Table 8: Differentiability of $\gamma(t)z$. A characterization of the possible choices of $\gamma(t)$ with respect to their differentiability. The column specified by $\hat{\sigma}(t)$ is sum of sigmoid functions, made compact by the notation $\hat{\sigma}(t)=\sigma(f(t-\tfrac{1}{2})+1)-\sigma(f(t-\tfrac{1}{2})-1)-\sigma(-\tfrac{f}{2}+1)+\sigma(-\tfrac{f}{2}-1)$, where $\sigma(t)=e^{t}/(1+e^{t})$ and $f$ is a scaling factor.

### 4.4 Spatially linear one-sided interpolants

Much of the discussion above generalizes to one-sided interpolants if we take the function $J(t,x_{1})$ in (3.15) to be linear in $x_{1}$ and define

$$
x^{\mathsf{os,lin}}_{t}=\alpha(t)z+\beta(t)x_{1},\qquad t\in[0,1]
$$

where $\alpha^{2},\beta\in C^{2}([0,1])$ and $\alpha(0)=\beta(1)=1$, $\alpha(1)=\beta(0)=0$, and $\alpha(t)>0$ for all $t\in[0,1)$. The velocity $b$ and the score $s$ defined in (2.10) and (2.14) can now be expressed as

$$
b(t,x)=\dot{\alpha}(t)\eta^{\mathsf{os}}_{z}(t,x)+\dot{\beta}(t)\eta^{\mathsf{os}}_{1}(t,x),\qquad s(t,x)=-\alpha^{-1}(t)\eta^{\mathsf{os}}_{z}(t,x),
$$

where the second expression holds for all $t\in[0,1)$ and we defined:

$$
\eta^{\mathsf{os}}_{z}(t,x)={\mathbb{E}}(z|x^{\mathsf{os,lin}}_{t}=x),\qquad\eta^{\mathsf{os}}_{1}(t,x)={\mathbb{E}}(x_{1}|x^{\mathsf{os,lin}}_{t}=x).
$$

Note that, by definition of the conditional expectation, $\eta^{\mathsf{os}}_{z}$ and $\eta^{\mathsf{os}}_{1}$ satisfy

$$
\forall(t,x)\in[0,1]\times{\mathbb{R}}^{d}\quad:\quad\alpha(t)\eta^{\mathsf{os}}_{z}(t,x)+\beta(t)\eta^{\mathsf{os}}_{1}(t,x)=x.
$$

As a result, only one of them needs to be estimated. For example, we can express $\eta^{\mathsf{os}}_{1}$ as a function of $\eta^{\mathsf{os}}_{z}$ for all $t$ such that $\beta(t)\not=0$, and use the result to express the velocity (4.16) as

$$
b(t,x)=\dot{\beta}(t)\beta^{-1}(t)x+\big{(}\dot{\alpha}(t)-\alpha(t)\dot{\beta}(t)\beta^{-1}(t)\big{)}\eta^{\mathsf{os}}_{z}(t,x)\qquad\forall t\ :\ \beta(t)\not=0.
$$

Assuming that $\beta(t)\not=0$ for all $t\in(0,1]$, this formula only needs to be supplemented at $t=0$ with

$$
b(0,x)=\dot{\alpha}(0)x+\dot{\beta}(0){\mathbb{E}}[x_{1}]
$$

which follows from (4.16) since $x^{\mathsf{os,lin}}_{t=0}=z$. Later in Section 5.2 we will show that using the velocity $b$ in (4.19) to solve the probability flow ODE (2.32) can be seen as using a denoiser to construct a generative model.

Finally note that $\eta_{z}$ and/or $\eta_{1}$ can be estimated using the following two objective functions, respectively:

$$
\displaystyle\mathcal{L}_{\eta_{z}}(\hat{\eta}^{\mathsf{os}}_{z})
$$
 
$$
\displaystyle=\int_{0}^{1}{\mathbb{E}}\left[\tfrac{1}{2}|\hat{\eta}^{\mathsf{os}}_{z}(t,x^{\mathsf{os,lin}}_{t})|^{2}-z\cdot\hat{\eta}^{\mathsf{os}}_{z}(t,x^{\mathsf{os,lin}}_{t})\right]dt,
$$
$$
\displaystyle\mathcal{L}_{\eta_{1}}(\hat{\eta}^{\mathsf{os}}_{1})
$$
 
$$
\displaystyle=\int_{0}^{1}{\mathbb{E}}\left[\tfrac{1}{2}|\hat{\eta}^{\mathsf{os}}_{1}(t,x^{\mathsf{os,lin}}_{t})|^{2}-x_{1}\cdot\hat{\eta}^{\mathsf{os}}_{1}(t,x^{\mathsf{os,lin}}_{t})\right]dt.
$$

## 5 Connections with other methods

In this section, we discuss connections between the stochastic interpolant framework and the score-based diffusion method [^60], the stochastic localization framework [^21] [^20] [^45]), denoising methods [^53] [^30] [^31] [^25], and the rectified flow method introduced in [^41].

### 5.1 Score-based diffusion models and stochastic localization

Score-based diffusion models (SBDM) are based on variants of the Ornstein-Uhlenbeck process

$$
dZ_{\tau}=-Z_{\tau}dt+\sqrt{2}dW_{\tau},\qquad Z_{\tau=0}\sim\rho_{1},
$$

which has the property that the marginal density of its solution at time $\tau$ converges to a standard normal as $\tau$ tends towards infinity. By learning the score of the density of $Z_{\tau}$, we can write the associated backward SDE for (5.1), which can then be used as a generative model – this backwards SDE is also the one that is used in the stochastic localization process, see [^45].

To see the connection with stochastic interpolants, notice that the solution of (5.1) from the initial condition $Z_{\tau=0}=x_{1}\sim\rho_{1}$ can be written exactly as

$$
Z_{\tau}=x_{1}e^{-\tau}+\sqrt{2}\int_{0}^{\tau}e^{-\tau+s}dW_{s}.
$$

As a result, the law of $Z_{\tau}$ conditioned on $Z_{\tau=0}=x_{1}$ is given by

$$
Z_{\tau}\sim{\sf N}(x_{1}e^{-\tau},(1-e^{-2\tau})),
$$

for any time $\tau\in[0,\infty)$. This is also the law of the process

$$
y_{\tau}=x_{1}e^{-\tau}+\sqrt{1-e^{-2\tau}}\,z,\qquad z\sim{\sf N}(0,\text{\it Id}),\qquad\tau\in[0,\infty).
$$

If we let $x_{1}\sim\rho_{1}$ with $x_{1}\perp z$, the process $y_{\tau}$ is similar to a one-sided stochastic interpolant, except the density of $y_{\tau}$ only converges to ${\sf N}(0,\text{\it Id})$ as $\tau\to\infty$; by contrast, the one-sided interpolants we introduced in Section 3.2 converge on the finite interval $[0,1]$. In SBDM, this is handled by capping the evolution of $Z_{\tau}$ to a finite time interval $[0,T]$ with $T<\infty$, and then by using the backward SDE associated with (5.1) restricted to $[0,T]$. However, this introduces a bias that is not present with one-sided stochastic interpolants, because the final condition used for the backwards SDE in SBDM is drawn from ${\sf N}(0,\text{\it Id})$ even though the density of the process (5.1) is not Gaussian at time $T$.

We can, however, turn (5.4) into a one-sided linear stochastic interpolant by defining $t=e^{-\tau}$ and by choosing $\alpha(t)$ and $\beta(t)$ in (4.15) to have a specific form. More precisely, evaluating (5.4) at $\tau=-\log t$,

$$
y_{\tau=-\log t}=\sqrt{1-t^{2}}z+tx_{1}\equiv x^{\mathsf{os,lin}}_{t}\quad\text{for}\quad\alpha(t)=\sqrt{1-t^{2}},\quad\beta(t)=t.
$$

With this choice of $\alpha(t)$ and $\beta(t)$, from (4.16) we get the velocity field

$$
b(t,x)=-\frac{t}{\sqrt{1-t^{2}}}\eta^{\mathsf{os}}_{z}(t,x)+\eta^{\mathsf{os}}_{1}(t,x)\equiv ts(t,x)+\eta^{\mathsf{os}}_{1}(t,x)
$$

where $\eta_{z}^{\mathsf{os}}$ and $\eta_{1}^{\mathsf{os}}$ are defined in (4.17). This expression shows that the velocity $b$ used in the probability flow ODE (2.32) is well-behaved at all times, including at $t=1$ where $\dot{\alpha}(t)$ is singular. The same is true for the drift $b_{\mathsf{F}}(t,x)=b(t,x)+\epsilon(t)s(t,x)$ used in the forward SDE (2.33), regardless of the choice of $\epsilon\in C^{0}([0,1])$ with $\epsilon(t)\geq 0$. This shows that casting SBDM into a one-sided linear stochastic interpolant (4.15) allows the construction of unbiased generative models that operate on $t\in[0,1]$. This comes at no extra computational cost, since only one of the two functions defined in (4.17) needs to be estimated, which is akin to estimating the score in SBDM.

It is worth comparing the above procedure to an equivalent change of time at the level of the diffusion process (5.1), which we now show leads to singular terms that pose numerical and analytical difficulties. Indeed, if we define $Z^{\mathsf{B}}_{t}=Z_{\tau=-\log t}$, from (5.1) we obtain

$$
dZ^{\mathsf{B}}_{t}=t^{-1}Z^{\mathsf{B}}_{t}dt+\sqrt{2t^{-1}}dW^{\mathsf{B}}_{t},\quad Z^{\mathsf{B}}_{t=1}\sim\rho_{1},
$$

to be solved backwards in time. Because of the factor $t^{-1}$, this SDE cannot easily be solved until $t=0$, which corresponds to $\tau=\infty$ in the original (5.1). For the same reason, the forward SDE associated with (5.7)

$$
dZ^{\mathsf{F}}_{t}=t^{-1}Z^{\mathsf{F}}_{t}dt+2t^{-1}s(t,Z^{\mathsf{F}}_{t})dt+\sqrt{2t^{-1}}dW_{t},
$$

cannot be solved from $t=0$, where formally $Z^{\mathsf{F}}_{t=0}\stackrel{{\scriptstyle d}}{{=}}Z^{\mathsf{B}}_{t=0}=Z_{\tau=\infty}\sim\mathsf{N}(0,\text{\it Id})$. This means it cannot be used as a generative model unless we start from some $t>0$, which introduces a bias. Importantly, this problem does not arise with the stochastic interpolant framework, because the construction of the density $\rho(t)$ connecting $\rho_{0}$ and $\rho_{1}$ is handled separately from the construction of the process that generates samples from $\rho(t)$. By contrast, SBDM combines these two operations into one, leading to the singularity at $t=0$ in the coefficients in (5.7) and (5.8).

###### Remark 5.1.

To emphasize the last point made above, we stress that there is no contradiction between having a singular drift and diffusion coefficient in (5.8), and being able to write a nonsingular SDE with stochastic interpolants. To see why, notice that the stochastic interpolant tells us that we can change the diffusion coefficient in (5.8) to any nonsingular $\epsilon\in C^{0}([0,1])$ with $\epsilon\geq 0$ and replace this SDE with

$$
dX^{\mathsf{F}}_{t}=t^{-1}Z^{\mathsf{F}}_{t}dt+(t^{-1}+\epsilon(t))s(t,X^{\mathsf{F}}_{t})dt+\sqrt{2\epsilon(t)}dW_{t},
$$

This SDE has the property that $X^{\mathsf{F}}_{t=1}\sim\rho_{1}$ if $X^{\mathsf{F}}_{t=0}\sim\rho_{0}$, and its drift is also nonsingular at $t=0$ and given precisely by (5.6). Indeed, using the constraint (4.18), which here reads $x=\sqrt{1-t^{2}}\eta^{\mathsf{os}}_{z}(t,x)+t\eta^{\mathsf{os}}_{1}(t,x)\equiv-(1-t^{2})s(t,x)+t\eta^{\mathsf{os}}_{1}(t,x)$, it is easy to see that

$$
t^{-1}x+t^{-1}s(t,x)=ts(t,x)+\eta_{1}^{\mathsf{os}}(t,x),
$$

which is nonsingular at $t=0$.

### 5.2 Denoising methods

Consider the spatially-linear one-sided stochastic interpolant defined in (4.15). By solving this equation for $x_{1}$, we obtain

$$
x_{1}=\beta^{-1}(t)\left(x^{\mathsf{os,lin}}_{t}-\alpha(t)z\right)\qquad t\in(0,1].
$$

Taking a conditional expectation at fixed $x^{\mathsf{os,lin}}_{t}$ and using (4.17) implies that

$$
{\mathbb{E}}(x_{1}|x^{\mathsf{os,lin}}_{t})=\eta^{\mathsf{os}}_{1}(t,x^{\mathsf{os,lin}}_{t})=\beta^{-1}(t)\left(x^{\mathsf{os,lin}}_{t}-\alpha(t)\eta^{\mathsf{os}}_{z}(t,x^{\mathsf{os,lin}}_{t})\right)\qquad t\in(0,1]
$$

while trivially ${\mathbb{E}}(x_{1}|x^{\mathsf{os,lin}}_{t=0})={\mathbb{E}}[x_{1}]$ since $x^{\mathsf{os,lin}}_{t=0}=z$. This expression is commonly used in denoising methods [^53] [^31], and it is Stein’s unbiased risk estimator (SURE) for $x_{1}$ given the noisy information in $x^{\mathsf{os,lin}}_{t}$ [^61]. Rather than considering the conditional expectation of $x_{1}$, we can consider an analogous quantity for $x_{s}^{\mathsf{os,lin}}$ for any $s\in[0,1]$; this leads to the following result.

###### Lemma 5.2 (SURE).

For $s\in[0,1]$, we have

$$
{\mathbb{E}}(x^{\mathsf{os,lin}}_{s}|x^{\mathsf{os,lin}}_{t})=\frac{\beta(s)}{\beta(t)}x^{\mathsf{os,lin}}_{t}+\left(\alpha(s)-\frac{\alpha(t)\beta(s)}{\beta(t)}\right)\eta^{\mathsf{os}}_{z}(t,x^{\mathsf{os,lin}}_{t})\qquad t\in(0,1]
$$

and ${\mathbb{E}}(x^{\mathsf{os,lin}}_{s}|x^{\mathsf{os,lin}}_{t=0})=\alpha(s)x^{\mathsf{os,lin}}_{t=0}+\beta(s){\mathbb{E}}[x_{1}]$.

###### Proof.

(5.13) follows from inserting (5.11) in the expression for $x^{\mathsf{os,lin}}_{s}$ and taking the conditional expectation using the definition of $\eta_{1}^{\mathsf{os}}$ and $\eta_{z}^{\mathsf{os}}$ in (4.17). ${\mathbb{E}}(x^{\mathsf{os,lin}}_{s}|x^{\mathsf{os,lin}}_{t=0})=\alpha(s)x^{\mathsf{os,lin}}_{t=0}+\beta(s){\mathbb{E}}[x_{1}]$ follows from $x^{\mathsf{os,lin}}_{t=0}=z$ together with (5.11). ∎

At this stage, equations (5.11) and (5.13) cannot be used as generative models: the random variable ${\mathbb{E}}(x_{1}|x^{\mathsf{os,lin}}_{t})$ is not a sample of $\rho_{1}$, and the random variable ${\mathbb{E}}(x^{\mathsf{os,lin}}_{s}|x^{\mathsf{os,lin}}_{t})$ is not a sample from $\rho(s)$, the density of $x^{\mathsf{os,lin}}_{s}$. However, the following result shows that if we iterate upon formula (5.13) by taking infinitesimal steps, we obtain a generative model consistent with the probability flow equation (2.32) associated with $x^{\mathsf{os,lin}}_{t}$. theoremdn Let $t_{j}=j/N$ with $j\in\{1,\ldots,N\}$, set $X^{\mathsf{den}}_{1}=z$, and define for $j=1,\ldots,N-1$,

$$
X^{\mathsf{den}}_{{j+1}}=\frac{\beta(t_{j+1})}{\beta(t_{j})}X^{\mathsf{den}}_{j}+\left(\alpha(t_{j+1})-\frac{\alpha(t_{j})\beta(t_{j+1})}{\beta(t_{j})}\right)\eta^{\mathsf{os}}_{z}(t_{j},X^{\mathsf{den}}_{j}).
$$

Then, (5.14) is a consistent integration scheme for the probability flow equation (2.32) associated with the velocity field (4.16) expressed as in (4.19). That is, if $N,j\to\infty$ with $j/N\to t\in[0,1]$, then $X^{\mathsf{den}}_{j}\to X_{t}$ where

$$
\dot{X}_{t}=b(t,X_{t})=\frac{\dot{\beta}(t)}{\beta(t)}X_{t}+\left(\dot{\alpha}(t)-\frac{\alpha(t)\dot{\beta}(t)}{\beta(t)}\right)\eta^{\mathsf{os}}_{z}(t,X_{t}),\quad X_{t=0}=z.
$$

In particular, if $z\sim{\sf N}(0,\text{\it Id})$, then $X^{\mathsf{den}}_{N}\to x_{1}\sim\rho_{1}$ in this limit.

The proof of this theorem is given in Appendix B.7, and proceeds by Taylor expansion of the right-hand side of (5.14).

### 5.3 Rectified flows

We now discuss how stochastic interpolants can be rectified according to the procedure proposed in [^41]. Suppose that we have perfectly learned the velocity field $b$ in the probability flow equation (2.32) for a given stochastic interpolant. Denote by $X_{t}(x)$ the solution to this ODE with the initial condition $X_{t=0}(x)=x$, i.e.

$$
\frac{d}{dt}X_{t}(x)=b(t,X_{t}(x)),\qquad X_{t=0}(x)=x.
$$

We can use the map $X_{t=1}:{\mathbb{R}}^{d}\to{\mathbb{R}}^{d}$ to define a new stochastic interpolant with $z\sim{\sf N}(0,\text{\it Id})$

$$
x^{\mathsf{rec}}_{t}=\alpha(t)z+\beta(t)X_{t=1}(z),
$$

where $\alpha^{2},\beta\in C^{2}([0,1])$ satisfy $\alpha(0)=\beta(1)=1$, $\alpha(1)=\beta(0)=0$, and $\alpha(t)>0$ for all $t\in[0,1)$. Clearly, we then have $x^{\mathsf{rec}}_{t=0}=z\sim\rho_{0}$ since $X_{t=0}(z)=z$ and $x^{\mathsf{rec}}_{t=1}=X_{t=1}(z)\sim\rho_{1}$ by definition of the probability flow equation. We can define a new probability flow equation associated with the velocity field

$$
b^{\mathsf{rec}}(t,x)={\mathbb{E}}[\dot{x}^{\mathsf{rec}}_{t}|x^{\mathsf{rec}}_{t}=x]=\dot{\alpha}(t){\mathbb{E}}[z|x^{\mathsf{rec}}_{t}=x]+\dot{\beta}(t){\mathbb{E}}[X_{t=1}(z)|x^{\mathsf{rec}}_{t}=x].
$$

It is easy to see that this velocity field is amenable to estimation, since it is the unique minimizer of

$$
\mathcal{L}_{b^{\mathsf{rec}}}[\hat{b}^{\mathsf{rec}}]=\int_{0}^{1}{\mathbb{E}}\big{[}\tfrac{1}{2}|\hat{b}^{\mathsf{rec}}(t,x^{\mathsf{rec}}_{t})|^{2}-(\dot{\alpha}(t)z+\dot{\beta}(t)X_{t=1}(z))\cdot\hat{b}^{\mathsf{rec}}(t,x^{\mathsf{rec}}_{t})\big{]}dt,
$$

where $x^{\mathsf{rec}}_{t}$ is given in (5.17) and the expectation is now only on $z\sim{\sf N}(0,\text{\it Id})$. Our next result show that the probability flow equation associated with the velocity field (5.18) has straight line solutions, but ultimately it leads to a generative model that is identical to the one based on (5.16). To phrase this result, we first make an assumption on the invertibility of $x_{t}^{\mathsf{rec}}$.

###### Assumption 5.3.

The map $x\to M(t,x)$ where $M(t,x)=\alpha(t)x+\beta(t)X_{t=1}(x)$ with $X_{t}(x)$ solution to (5.16) is invertible for all $(t,x)\in[0,1]\times{\mathbb{R}}^{d}$, i.e. $\exists N(t,\cdot,):{\mathbb{R}}^{d}\to{\mathbb{R}}^{d}$ such that

$$
\forall(t,x)\in[0,1]\times{\mathbb{R}}^{d}\quad:\quad N(t,M(t,x))=M(t,N(t,x))=x.
$$

This is equivalent to requiring that the determinant of the Jacobian of $M(t,x)$ is nonzero for all $(t,x)\in[0,1]\times{\mathbb{R}}^{d}$; put differently, Assumption 5.3 requires that the Jacobian of $X_{t=1}(x)$ never has eigenvalues precisely equal to $-\alpha(t)/\beta(t)$, which is generic. Under this assumption, we state the following theorem. theoremrec Consider the probability flow equation associated with (5.18),

$$
\frac{d}{dt}X^{\mathsf{rec}}_{t}(x)=b^{\mathsf{rec}}(t,X^{\mathsf{rec}}_{t}(x)),\qquad X^{\mathsf{rec}}_{t=0}(x)=x.
$$

Then, all solutions are such that $X^{\mathsf{rec}}_{t=1}(z)\sim\rho_{1}$ if $z\sim{\sf N}(0,\text{\it Id})$. In addition, if Assumption (5.3) holds, the velocity field defined in (5.18) reduces to

$$
b^{\mathsf{rec}}(t,x)=\dot{\alpha}(t)N(t,x)+\dot{\beta}(t)X_{t=1}(N(t,x))
$$

and the solution to the probability flow ODE (5.21) is simply

$$
X^{\mathsf{rec}}_{t}(x)=\alpha(t)x+\beta(t)X_{t=1}(x).
$$

The proof is given in Appendix B.8.

Theorem 5.3 implies that $X^{\mathsf{rec}}_{t}(x)$ is a simpler flow than $X_{t}(x)$, but we stress that they give the same map, $X^{\mathsf{rec}}_{t=1}=X_{t=1}$. In particular, $X^{\mathsf{rec}}_{t}(x)$ reduces to a straight line between $x$ and $X_{t=1}(x)$ for $\alpha(t)=1-t$ and $\beta(t)=t$. We also note that the approach can be used to learn a single-step map, since (5.22) and $N(t=0,x)=x$ give

$$
b^{\mathsf{rec}}(t=0,x)=\dot{\alpha}(0)x+\dot{\beta}(0)X_{t=1}(x),
$$

which expresses $X_{t=1}(x)$ in terms of known quantities as long as $\dot{\beta}(0)\not=0$. For example, if $\dot{\alpha}(0)=0$ and $\dot{\beta}(0)=1$, we obtain $b^{\mathsf{rec}}(t=0,x)=X_{t=1}(x)$.

###### Remark 5.4 (Optimal transport).

The discussion above highlights the fact that a probability flow equation can have straight line solutions and lead to a map that exactly pushes $\rho_{0}$ onto $\rho_{1}$ but is not the optimal transport map. That is, straight line solutions is a necessary condition for optimal transport, but it is not sufficient.

###### Remark 5.5 (Gradient fields).

The map is unaffected by the rectification procedure because we do not impose that the velocity $b^{\mathsf{rec}}(t,x)$ be a gradient field. If we do impose this structure by setting $b^{\mathsf{rec}}(t,x)=\nabla\phi(t,x)$ for some $\phi:{\mathbb{R}}^{d}\to{\mathbb{R}}$, then $X_{t=1}^{\mathsf{rec}}\not=X_{t=1}$. As shown in [^40], iterating over this procedure eventually gives the optimal transport map. That is, implemented over gradient fields and iterated infinitely, rectification computes Brenier’s polar decomposition of the map [^9].

###### Remark 5.6 (Consistency models).

Recent work has introduced the notion of consistency models [^56], which distill a velocity field learned via score-based diffusion into a single-step map. Section 5.1 and the previous discussion provide an alternative perspective on consistency models, and show how they may be computed in the framework of stochastic interpolants via rectification.

## 6 Algorithmic aspects

The methods described in the previous sections have efficient numerical realizations. Here, we detail algorithms and practical recommendations for an implementation. These suggestions can be split into two complementary tasks: learning the drift coefficients, and sampling with an ODE or an SDE.

### 6.1 Learning

As described in Section 2.2, there are a variety of algorithmic choices that can be made when learning the drift coefficients in (2.9), (2.20), and (2.22). While all choices lead to exact generative models in the absence of numerical and statistical errors, in practice, the presence of these errors ensures that different choices lead to different generative models, some of which may perform better for specific applications. Here, we describe the various realizations explicitly.

#### Deterministic generative modeling: Learning b𝑏b versus learning v𝑣v and s𝑠s.

Recall from Section 2.2 that the drift $b$ of the transport equation (2.9) can be written as $b(t,x)=v(t,x)-\gamma(t)\dot{\gamma}(t)s(t,x)$. This raises the practical question of whether it would be better to learn an estimate $\hat{b}$ of $b$ by minimizing the empirical risk

$$
\hat{\mathcal{L}}_{b}(\hat{b})=\frac{1}{N}\sum_{i=1}^{N}\left(\frac{1}{2}|\hat{b}(t_{i},x_{t_{i}}^{i})|^{2}-\hat{b}(t_{i},x_{t_{i}}^{i})\cdot\left(\partial_{t}I(t_{i},x_{0}^{i},x_{1}^{i})+\dot{\gamma}(t_{i})z^{i}\right)\right),
$$

or to learn estimates of $\hat{v}$ and $\hat{s}$ by minimizing the empirical risks

$$
\hat{\mathcal{L}}_{v}(\hat{v})=\frac{1}{N}\sum_{i=1}^{N}\left(\frac{1}{2}|\hat{v}(t_{i},x_{t_{i}}^{i})|^{2}-\hat{v}(t_{i},x_{t_{i}}^{i})\cdot\partial_{t}I(t_{i},x_{0}^{i},x_{1}^{i})\right)
$$

and

$$
\hat{\mathcal{L}}_{s}(\hat{s})=\frac{1}{N}\sum_{i=1}^{N}\left(\frac{1}{2}|\hat{s}(t_{i},x_{t_{i}}^{i})|^{2}+\gamma(t_{i})^{-1}\hat{s}(t_{i},x_{t_{i}}^{i})\cdot z^{i}\right)
$$

Input: Batch size $N$, interpolant function $I(t,x_{0},x_{1})$, coupling $\nu(dx_{0},dx_{1})$ to sample $x_{0},x_{1}$, noise function $\gamma(t)$, initial parameters $\theta_{b}$, gradient-based optimization algorithm to minimize $\mathcal{L}_{b}$ defined in (2.13), number of gradient steps $N_{g}$.

Returns: An estimate $\hat{b}$ of $b$.

for *$j=1,\ldots,N_{g}$* do

Draw $N$ samples $(t_{i},x_{0}^{i},x_{1}^{i},z^{i})\sim\mathsf{Unif}([0,1])\times\nu\times\mathsf{N}(0,1)$, for $i=1,\ldots,N$.

Construct samples $x_{t}^{i}=I(t_{i},x_{0}^{i},x_{1}^{i})+\gamma(t_{i})z^{i}$, for $i=1,\ldots,N$.

Take gradient step with respect to $\mathcal{L}_{b}\left(\theta_{b},\{x_{t}^{i}\}_{i=1}^{N}\right)$.

Return: $\hat{b}$.

Algorithm 1 Learning $b$ with arbitrary $\rho_{0}$ and $\rho_{1}$.

and construct the estimate $\hat{b}(t,x)=\hat{v}(t,x)-\gamma(t)\dot{\gamma}(t)\hat{s}(t,x)$. Above, $x_{t_{i}}^{i}=I(t_{i},x_{0}^{i},x_{1}^{i})+\gamma(t_{i})z^{i}$, and $N$ denotes the number of samples $t^{i}$, $x_{0}^{i}$, and $x_{1}^{i}$. If the practitioner is interested in a deterministic generative model (for example, to exploit adaptive integration or exact likelihood computation), learning the estimate $\hat{b}$ directly only requires learning a single model, and hence will typically lead to greater efficiency. This recommendation is captured in Algorithm 1. If both stochastic and deterministic generative models are of interest, it is necessary to learn two models for most choices of the interpolant; we discuss more suggestions for stochastic case below.

#### Antithetic sampling and capping.

In practice, the losses for $b$ (2.13) and $s$ (2.16) can become high-variance near the endpoints $t=0$ and $t=1$ due to the presence of the singular term $1/\gamma(t)$ and the (potentially) singular term $\dot{\gamma}(t)$. This issue can be eliminated by using antithetic sampling, which we found necessary for stable training of objectives involving $\gamma^{-1}(t)$. To show why, we consider the loss (2.16) for $s$, but an analogous calculation can be performed for the loss (2.13) for $b$ or (2.19) for $\eta_{z}$ (even though it is not necessary for this last quantity). We first observe that, by definition of $x_{t}$ and by Taylor expansion, as $t\to 0$ or $t\to 1$

$$
\displaystyle\frac{1}{\gamma(t)}z\cdot s(t,x_{t})
$$
 
$$
\displaystyle=\frac{1}{\gamma(t)}z\cdot s(t,I(t,x_{0},x_{1})+\gamma(t)z),
$$
$$
\displaystyle=\frac{1}{\gamma(t)}z\cdot\left(s(t,I(t,x_{0},x_{1}))+\gamma(t)\nabla s(t,I(t,x_{0},x_{1}))z+o(\gamma(t))\right),
$$
$$
\displaystyle=\frac{1}{\gamma(t)}z\cdot s(t,I(t,x_{0},x_{1}))+z\cdot\nabla s(t,I(t,x_{0},x_{1}))z+o(1).
$$

Even though the conditional mean of the first term at the right-hand side is finite in the limit as $t\to 0$ or $t\to 1$, its variance diverges. By contrast, let $x_{t}^{+}=I(t,x_{0},x_{1})+\gamma(t)z$ and $x_{t}^{-}=I(t,x_{0},x_{1})-\gamma(t)z$ with $x_{0},x_{1}$, and $z$ fixed. Then,

$$
\displaystyle\frac{1}{2\gamma(t)}\left(z\cdot s(t,x_{t}^{+})-z\cdot s(t,x_{t}^{-})\right)
$$
 
$$
\displaystyle=\frac{1}{2\gamma(t)}\left(z\cdot s(t,I(t,x_{0},x_{1})+\gamma(t)z))-z\cdot s(t,I(t,x_{0},x_{1})-\gamma(t)z)\right),
$$
$$
\displaystyle=\frac{1}{2\gamma(t)}z\cdot\left(s(t,I(t,x_{0},x_{1}))+\gamma(t)\nabla s(t,I(t,x_{0},x_{1}))z+o(\gamma(t))\right)
$$
 
$$
\displaystyle\qquad-\frac{1}{2\gamma(t)}z\cdot\left(s(t,I(t,x_{0},x_{1}))-\gamma(t)\nabla s(t,I(t,x_{0},x_{1}))z+o(\gamma(t))\right),
$$
$$
\displaystyle=z\cdot\nabla s(t,I(t,x_{0},x_{1}))z+o(1),
$$

so that both the conditional mean and variance are finite in the limit as $t\to 0$ or $t\to 1$ despite the singularity of $1/\gamma(t)$. In practice, this can be implemented by using $x_{t}^{+}$ and $x_{t}^{-}$ for every draw of $x_{0},x_{1}$, and $z$ in the empirical discretization of the population loss.

Input: Batch size $N$, interpolant function $I(t,x_{0},x_{1})$, a coupling $\nu(dx_{0},dx_{1})$ to sample $x_{0},x_{1}$, noise function $\gamma(t)$, initial parameters $\theta_{\eta_{z}}$, gradient-based optimization algorithm to minimize $\mathcal{L}_{\eta_{z}}$ defined in (2.19), number of gradient steps $N_{g}$.

Returns: An estimate $\hat{\eta}_{z}$ of $\eta_{z}$.

for *$j=1,\ldots,N_{g}$* do

Draw $N$ samples $(t_{i},x_{0}^{i},x_{1}^{i},z^{i})\sim\mathsf{Unif}([0,1])\times\nu\times\mathsf{N}(0,1)$, for $i=1,\ldots,N$.

Construct samples $x_{t}^{i}=I(t_{i},x_{0}^{i},x_{1}^{i})+\gamma(t_{i})z^{i}$, for $i=1,\ldots,N$.

Take gradient step with respect to $\mathcal{L}_{\eta_{z}}\left(\theta_{\eta_{z}},\{x_{t}^{i}\}_{i=1}^{N}\right)$.

Return: $\hat{\eta}_{z}$.

Algorithm 2 Learning $\eta_{z}$ with arbitrary $\rho_{0}$ and $\rho_{1}$.

#### Learning the score s𝑠s versus learning a denoiser ηzsubscript𝜂𝑧\\eta\_{z}.

When learning $s$, an alternative to antithetic sampling is to consider learning the denoiser $\eta_{z}$ defined in (2.17), which is related to the score by a factor of $\gamma$. Note that the objective function for the denoiser in (2.19) is well behaved for all $t\in[0,1]$, and can be thought of as a generalization of the DDPM loss introduced in [^25]. The empirical risk associated with this loss reads

$$
\hat{\mathcal{L}}_{\eta_{z}}(\hat{\eta}_{z})=\frac{1}{N}\sum_{i=1}^{N}\left(\frac{1}{2}|\hat{\eta}_{z}(t_{i},x_{t_{i}}^{i})|^{2}-\hat{\eta}_{z}(t_{i},x_{t_{i}}^{i})\cdot z^{i}\right)
$$

A detailed procedure for learning the denoiser $\eta_{z}$, e.g. for its use in an SDE-based generative model is given in Algorithm 2. For the case of one-sided spatially-linear interpolants, the procedure becomes particularly simple, which is highlighted in Algorithm 3.

Input: Batch size $N$, interpolant function $x^{\mathsf{os,lin}}_{t}$, a coupling $\nu(dx_{0},dx_{1})$ to sample $z,x_{1}$, initial parameters $\theta_{\eta_{z}^{\mathsf{os}}}$, gradient-based optimization algorithm to minimize $\mathcal{L}_{\eta_{z}^{\mathsf{os}}}$ defined in (2.19), number of gradient steps $N_{g}$.

Returns: An estimate $\hat{\eta}_{z}^{\mathsf{os}}$ of $\eta_{z}^{\mathsf{os}}$.

for *$j=1,\ldots,N_{g}$* do

Draw $N$ samples $(t_{i},z^{i},x_{1}^{i})\sim\mathsf{Unif}([0,1])\times\nu$, for $i=1,\ldots,N$.

Construct samples $x_{t}^{i}=\alpha(t_{i})z+\beta(t_{i})x_{1}^{i}$, for $i=1,\ldots,N$.

Take gradient step w.r.t $\mathcal{L}_{\eta_{z}^{\mathsf{os}}}\left(\theta_{\eta_{z}^{\mathsf{os}}},\{x_{t}^{i}\}_{i=1}^{N}\right)$.

Return: $\hat{\eta}_{z}^{\mathsf{os}}$.

Algorithm 3 Learning $\eta_{z}^{\mathsf{os}}$ with Gaussian $\rho_{0}$.

### 6.2 Sampling

We now discuss several practical aspects of sampling generative models based on stochastic interpolants. These are intimately related to the choice of objects that are learned, as well as to the specific interpolant used to build a path between $\rho_{0}$ and $\rho_{1}$. A general algorithm for sampling models built on either ordinary or stochastic differential equations is presented in Algorithm 4.

Input: Number of samples $n$, timestep $\Delta t$, drift estimates $\hat{b}$ and $\hat{\eta}_{z}$, initial time $t_{0}$, final time $t_{f}$, noise function $\gamma(t)$, diffusion coefficient $\epsilon(t)$, SDE or ODE timestepper TakeStep.

Returns: $\{\hat{x}_{1}^{(i)}\}_{i=1}^{n}$, a batch of model samples.

Initialize

Set time $t=t_{0}$.

Draw initial conditions $\hat{x}^{(i)}_{t_{0}}\sim\rho_{0}$ for $i=1,\ldots,n$.

Construct $\hat{s}(t,x)=-\hat{\eta}_{z}(t,x)/\gamma(t)$.

Construct $\hat{b}_{\mathsf{F}}(t,x)=\hat{b}(t,x)+\epsilon(t)\hat{s}(t,x)$. // Reduces to $\hat{b}$ for $\epsilon(t)=0$ (ODE).

while *$t<t_{f}$* do

Propagate $\hat{x}^{(i)}_{t+\Delta t}=\texttt{TakeStep}(t,\hat{x}^{(i)}_{t},b_{\mathsf{F}},\epsilon,\Delta t)$ for $i=1,\ldots,n$. // ODE or SDE integrator.

Update $t=t+\Delta t$.

Return: $\{\hat{x}^{(i)}\}_{i=1}^{n}$.

Algorithm 4 Sampling general stochastic interpolants.

#### Using the denoiser ηzsubscript𝜂𝑧\\eta\_{z} instead of the score s𝑠s.

We remarked in Section 6.1 that learning the denoiser $\eta_{z}$ is more numerically stable than learning the score $s$ directly. We note that while the objective for $\eta_{z}$ is well-behaved for all $t\in[0,1]$, the resulting drifts can become singular at $t=0$ and $t=1$ when using $s(t,x)=-\eta_{z}(t,x)/\gamma(t)$. There are several ways to avoid this singularity in practice. One method is to choose a time-varying $\epsilon(t)$ that vanishes in a small interval around the endpoints $t=0$ and $t=1$, which avoids this numerical instability. An alternative option is to integrate the SDE up to a final time $t_{f}$ with $t_{f}<1$, and then to perform a step of denoising using (5.13). We use this approach in Section 7 below when sampling the SDE.

#### A denoiser is all you need for spatially-linear one-sided interpolants.

Input: Number of samples $n$, timestep $\Delta t$, denoiser estimate $\hat{\eta}_{z}$, initial time $t_{0}$, final time $t_{f}$, noise function $\gamma(t)$, diffusion coefficient $\epsilon(t)$, interpolant functions $\alpha(t)$ and $\beta(t)$, SDE or ODE timestepper TakeStep.

Returns: $\{\hat{x}_{1}^{(i)}\}_{i=1}^{n}$, a batch of model samples.

Initialize

Set time $t=t_{0}$.

Draw initial conditions $\hat{x}^{(i)}_{t_{0}}\sim\rho_{0}$ for $i=1,\ldots,n$.

Construct $\hat{s}(t,x)=-\hat{\eta}_{z}(t,x)/\alpha(t)$.

Construct $\hat{b}(t,x)=\dot{\alpha}(t)\hat{\eta}_{z}^{\mathsf{os}}(t,x)+\frac{\dot{\beta}(t)}{\beta(t)}\left(x-\alpha(t)\hat{\eta}_{z}^{\mathsf{os}}(t,x)\right)$.

Construct $\hat{b}_{\mathsf{F}}(t,x)=\hat{b}(t,x)+\epsilon(t)\hat{s}(t,x)$. // Reduces to $\hat{b}$ for $\epsilon(t)=0$ (ODE).

while *$t<t_{f}$* do

Propagate $\hat{x}^{(i)}_{t+\Delta t}=\texttt{TakeStep}(t,\hat{x}^{(i)}_{t},b_{\mathsf{F}},\epsilon,\Delta t)$ for $i=1,\ldots,n$. // ODE or SDE integrator.

Update $t=t+\Delta t$.

Return: $\{\hat{x}^{(i)}\}_{i=1}^{n}$.

Algorithm 5 Sampling spatially-linear one-sided interpolants with Gaussian $\rho_{0}$.

As shown in (4.19), and as considered in Section 5.2, the denoiser $\eta_{z}^{\mathsf{os}}$ is sufficient to represent the velocity field $b$ appearing in the probability flow equation (2.32).

Using this definition for $b$ and the relationship $s(t,x)=-\eta_{z}(t,x)/\gamma(t)$, we state the following ordinary and stochastic differential equations for sampling

$$
\displaystyle\text{ODE}:
$$
$$
\displaystyle\quad\dot{X}_{t}=\dot{\alpha}(t)\eta^{\mathsf{os}}_{z}(t,X_{t})+\frac{\dot{\beta}(t)}{\beta(t)}\big{(}X_{t}-\alpha(t)\eta^{\mathsf{os}}_{z}(t,X_{t})\big{)}
$$
 
$$
\displaystyle\text{SDE}:
$$
$$
\displaystyle\quad dX^{\mathsf{F}}_{t}=\big{(}\dot{\alpha}(t)\eta^{\mathsf{os}}_{z}(t,X^{\mathsf{F}}_{t})+\frac{\dot{\beta}(t)}{\beta(t)}\big{(}X^{\mathsf{F}}_{t}-\alpha(t)\eta^{\mathsf{os}}_{z}(t,X^{\mathsf{F}}_{t})\big{)}-\frac{\epsilon(t)}{\alpha(t)}\eta^{\mathsf{os}}_{z}(t,X^{\mathsf{F}}_{t})\big{)}dt
$$
 
$$
\displaystyle\qquad\qquad+\sqrt{2\epsilon(t)}dW_{t}.
$$

Because $\beta(0)=0$, the drift is numerically singular in both equations. However, $b(t=0,x)$ has a finite limit

$$
b(t=0,x)=\dot{\alpha}(0)x+\dot{\beta}(0)\mathbb{E}[x_{1}],
$$

as originally given in (4.20). Equation (6.8) can be estimated using available data, which means that when learning a one-sided interpolant, ODE and SDE-based generative models can be defined exactly on the interval $t\in[0,1]$ using only a score or a denoiser without singularity.

The factor of $\alpha(t)^{-1}$ in the final term of the SDE could pose numerical problems at $t=1$, as $\alpha(1)=0$. As discussed in the paragraph above, a choice of $\epsilon(t)$ which is such that $\epsilon(t)/\alpha(t)\rightarrow C$ for some constant $C$ as $t\rightarrow 1$ avoids any issue.

An algorithm for sampling with only the denoiser $\eta_{z}^{\mathsf{os}}$ is given in Algorithm 5.

## 7 Numerical results

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2303.08797/assets/x7.png)

Figure 9: The effects of γ ( t ) 𝛾 𝑡 \\gamma(t) and ϵ italic-ϵ \\epsilon on sample quality: qualitative comparison. Kernel density estimates of ρ ^ 1 𝜌 \\hat{\\rho}(1) for models with different choices of \\gamma. Sampling with = 0 \\epsilon=0 corresponds to using the probability flow with the learned drift b v − ˙ s 𝑏 𝑣 𝑠 \\hat{b}=\\hat{v}-\\gamma\\dot{\\gamma}\\hat{s}, whereas sampling with > \\epsilon>0 corresponds to using the SDE with the learned drift \\hat{b} and score \\hat{s}. We find empirically that SDE sampling is generically better than ODE sampling for this target density, though the gap is smallest for the probability flow specified with \\gamma(t)=\\sqrt{t(1-t)}, in agreement with the Remark 4.1 regarding the influence of on at the endpoints. The SDE performs well at any noise level, though numerically integrating it for higher requires a smaller step size.

So far, we have been focused on the impact of $\alpha$, $\beta$, and $\gamma$ in (4.1) on the density $\rho(t)$, which we illustrated analytically. In this section, we study examples where the drift coefficients must be learned over parametric function classes. In particular, we explore numerically the tradeoffs between generative models based on ODEs and SDEs, as well as the various design choices introduced in Sections 3, 4, and 6. In Section 7.1, we consider simple two-dimensional distributions that can be visualized easily. In Section 7.2, we consider high-dimensional Gaussian mixtures, where we can compare our learned models to analytical solutions. Finally in Section 7.3 we perform some experiments in image generation.

### 7.1 Deterministic versus stochastic models: 2D

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2303.08797/assets/x8.png)

Figure 10: The effects of γ ( t ) 𝛾 𝑡 \\gamma(t) and ϵ italic-ϵ \\epsilon on sample quality: quantitative comparison. For each \\gamma and each specified in Figure 9, we compute the mean and variance of the absolute value of the difference of log ⁡ ρ 1 subscript 𝜌 \\log\\rho\_{1} (exact) and ^ \\log\\hat{\\rho}(1) (model). The model specified with = − \\gamma(t)=\\sqrt{t(1-t)} is the best performing probability flow ( 0 \\epsilon=0 ). At large, SDE sampling with the same learned drift b 𝑏 \\hat{b} and score s 𝑠 \\hat{s} performs better, complementing the observations in the previous figure.

As shown in Section 2.2, the evolution of $\rho(t)$ can be captured exactly by either the transport equation (2.9) or by the forward and backward Fokker-Planck equations (2.20) and (2.22). These perspectives lead to generative models that are either based on the deterministic dynamics (2.32) or the forward and backward stochastic dynamics (2.33) and (2.34), where the level of stochasticity can be tuned by varying the diffusion coefficient $\epsilon(t)$. We showed in Section 2.4 that setting a constant $\epsilon(t)=\epsilon>0$ can offer better control on the likelihood when using an imperfect velocity $b$ and an imperfect score $s$. Moreover, the optimal choice of $\epsilon$ is determined by the relative accuracy of the estimates $\hat{b}$ and $\hat{s}$. Having laid out the evolution of $\rho(t)$ for different choices of $\gamma$ in the previous section, we now show how different values of $\epsilon$ can build these densities up from individual trajectories. The stochasticity intrinsic to the sampling process increases with $\epsilon$, but by construction, the marginal density $\rho(t)$ for fixed $\alpha$, $\beta$ and $\gamma$ is independent of $\epsilon$.

#### The roles of γ(t)𝛾𝑡\\gamma(t) and ϵitalic-ϵ\\epsilon for 2D density estimation.

To explore the roles of $\gamma$ and $\epsilon$, we consider a target density $\rho_{1}$ whose mass concentrates on a two-dimensional checkerboard and a base density $\rho_{0}=\mathsf{N}(0,\text{\it Id})$; here, the target was chosen to highlight the ability of the method to learn a challenging density with sharp boundaries. The same model architecture and training procedure was used to learn both $v$ and $s$ for several choices of $\gamma$ given in Table 8. The feed-forward network was defined with $4$ layers, each of size $512$, and with the ReLU [^46] as an activation function.

After training, we draw 300,000 samples using either an ODE ($\epsilon=0$) or an SDE with $\epsilon=0.5$, $\epsilon=1.0$, or $\epsilon=2.5$. We compute kernel density estimates for each resulting density, which we compare to the exact density and to the original stochastic interpolant from [^2] (obtained by setting $\gamma=0$). Results are given in Figure 9 for each $\gamma$ and each $\epsilon$. Sampling with $\epsilon>0$ empirically performs better, though the gap is smallest when using the $\gamma$ specified in (4.13). Moreover, even when $\epsilon=0$, using the probability flow with $\gamma$ given by (4.13) performs better than the original interpolant from [^2]. Numerical comparisons of the mean and variance of the absolute value of the difference of $\log\rho_{1}$ (exact) from $\log\hat{\rho}(1)$ (model) for the various configurations are given in Figure 10, which corroborate the above observations.

### 7.2 Deterministic versus stochastic models: 128D Gaussian mixtures

We now study the performance of the stochastic interpolant method in the case where the target is a high-dimensional Gaussian mixture. Gaussian mixtures (GMs) are a convenient class of target distributions to study, because they can be made arbitrarily complex by increasing the number of modes, their separation, and the overall dimensionality. Moreover, by considering low-dimensional projections, we can compute quantitative error metrics such as the $\mathsf{KL}$ -divergence between the target and the model as a function of the (constant) diffusion coefficient $\epsilon$. This enables us to quantify the tradeoffs of ODE and SDE-based samplers.

#### Experimental details.

We consider the problem of mapping $\rho_{0}=\mathsf{N}(0,\text{\it Id})$ to a Gaussian mixture with five modes in dimension $d=128$. The mean $m_{i}\in{\mathbb{R}}^{d}$ of each mode is drawn i.i.d. $m_{i}\sim\mathsf{N}(0,\sigma^{2}\text{\it Id})$ with $\sigma=7.5$. Each covariance $C_{i}\in{\mathbb{R}}^{d\times d}$ is also drawn randomly with $C_{i}=\tfrac{1}{d}W_{i}^{\mathsf{T}}W_{i}+\text{\it Id}$ and $(W_{i})_{kl}\sim\mathsf{N}(0,1)$ for $k,l=1,\ldots,d$; this choice ensures that each covariance is a positive definite perturbation of the identity with diagonal entries that are $O(1)$ with respect to dimension $d$. For a fixed random draw of the means $\{m_{i}\}_{i=1}^{5}$ and covariances $\{C_{i}\}_{i=1}^{5}$, we study the four combinations of learning $b$ or $v$ and $s$ or $\eta_{z}$ to form a stochastic interpolant from $\rho_{0}$ to $\rho_{1}$. In each case, we consider a linear interpolant with $\alpha(t)=1-t$, $\beta(t)=t$, and $\gamma(t)=\sqrt{t(1-t)}$.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2303.08797/assets/x9.png)

Figure 11: Gaussian mixtures: target projection. Low-dimensional marginal of the target density ρ 1 subscript 𝜌 \\rho\_{1} for the Gaussian mixture experiment, visualized via KDE.

For visual reference, a projection of the target density $\rho_{1}$ onto the first two coordinates is depicted in Figure 11 – it contains significant multimodality, several modes that are difficult to distinguish, and one mode that is well-separated from the others, which requires nontrivial transport to resolve. In the following experiments, all samples were generated with the fourth-order Dormand-Prince adaptive ODE solver (dopri5) for $\epsilon=0$ and by using one thousand timesteps of the Heun SDE integrator introduced in [^32] for $\epsilon\neq 0$. To maximize performance at high $\epsilon$, the timestep should be adapted to $\epsilon$; here, we chose to use a fixed computational budget that performs well for moderate levels of $\epsilon$ to avoid computational effort that may become unreasonable in practice. When learning $\eta_{z}$, to avoid singularity at $t=0$ and $t=1$ when dividing by $\gamma(t)$ in the formula $s(t,x)=-\eta(t,x)/\gamma(t)$, we set $t_{0}=10^{-4}$ and $t_{f}=1-t_{0}$ in Algorithm 4. For all other cases, we set $t_{0}=0$ and $t_{f}=1$.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2303.08797/assets/x10.png)

Figure 12: Gaussian mixtures: density errors. Errors ρ ^ 1 ( x, y ) − subscript 𝜌 𝑥 𝑦 \\hat{\\rho}\_{1}(x,y)-\\rho\_{1}(x,y) in the marginals over the first two coordinates for all four variations of learning b 𝑏 or v 𝑣 and s 𝑠 η 𝜂 \\eta, computed via kernel density estimation. For small ϵ italic-ϵ \\epsilon, the model densities tend to be overly-concentrated, and overestimate the density within the modes and underestimate the densities in the tails. As increases, the model becomes less concentrated and more accurately represents the target density. For too large, the model becomes overly spread out and under-estimates the density within the modes. Note: visualizations show two-dimensional slices of a 128 -dimensional density.

#### Quantitative metric

To quantify performance, we make use of an error metric given by a $\mathsf{KL}$ -divergence between kernel density estimates (KDE) of low-dimensional marginals of $\rho_{1}$ and the model density $\hat{\rho}_{1}$; this error metric was chosen for computational tractability and interpretability. To compute it, we draw $50,000$ samples from $\rho_{1}$ and each $\hat{\rho}_{1}$. We obtain samples from the marginal density over the first two coordinates by projection, and then compute a Gaussian KDE with bandwidth parameter chosen by Scott’s rule. We then draw a fresh set of $N_{e}=50,000$ samples $\{x_{i}\}_{i=1}^{N_{e}}$ with each $x_{i}\sim\rho_{1}$ for evaluation. To compute the $\mathsf{KL}$ -divergence, we form a Monte-Carlo estimate with control variate

$$
\mathsf{KL}(\rho_{1}\>\|\>\hat{\rho}_{1})\approx\frac{1}{N_{e}}\sum_{i=1}^{N_{e}}\left(\log\rho_{1}(x_{i})-\log\hat{\rho}_{1}(x_{i})-\left(\frac{\hat{\rho}_{1}(x_{i})}{\rho_{1}(x_{i})}-1\right)\right).
$$

We found use of the control variate $\hat{\rho}_{1}/\rho_{1}-1$ helpful to reduce variance in the Monte-Carlo estimate; moreover, by concavity of the logarithm, use of the control variate ensures that the Monte-Carlo estimate cannot become negative.

#### Results.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2303.08797/assets/x11.png)

Figure 13: Gaussian mixtures: densities. A visualization of the marginal densities in the first two variables of the model density ρ ^ 1 subscript 𝜌 \\hat{\\rho}\_{1} computed for all four variations of learning b 𝑏 or v 𝑣 and s 𝑠 η 𝜂 \\eta, computed via kernel density estimation. Note: visualizations show two-dimensional slices of a 128 -dimensional density.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2303.08797/assets/x12.png)

Figure 14: Gaussian mixtures: quantitative comparison. 𝖪𝖫 ( ρ 1 ∥ ^ ϵ ) conditional subscript 𝜌 superscript italic-ϵ \\mathsf{KL}(\\rho\_{1}\\>\\|\\>\\hat{\\rho}\_{1}^{\\epsilon}) as a function of \\epsilon when learning each of the four possible sets of drift coefficients b, s 𝑏 𝑠 (b,s) η 𝜂 (b,\\eta) v 𝑣 (v,s) (v,\\eta). The best performance is achieved when learning and \\eta, along with a proper choice of > 0 \\epsilon>0.

Figures 12 and 13 display two-dimensional projections (computed via KDE) of the model density error $\hat{\rho}_{1}-\rho_{1}$ and the model density $\hat{\rho}_{1}$ itself, respectively, for different instantiations of Algorithms 1 and 2 and different choices of $\epsilon$ in Algorithm 4. Taken together with Figure 11, these results demonstrate qualitatively that small values of $\epsilon$ tend to over-estimate the density within the modes and under-estimate the density in the tails. Conversely, when $\epsilon$ is taken too large, the model tends to under-estimate the modes and over-estimate the tails. Somewhere in between (and for differing levels of $\epsilon$), every model obtains its optimal performance. Figure 14 makes these observations quantitative, and displays the $\mathsf{KL}$ -divergence from the target marginal to the model marginal $\mathsf{KL}(\rho_{1}\>\|\>\hat{\rho}_{1}^{\epsilon})$ as a function of $\epsilon$, with each data point on the curve matching the models depicted in Figures 12 and 13. We find that for each case, there is an optimal value of $\epsilon\neq 0$, in line with the qualitative picture put forth by Figures 12 and 13. Moreover, we find that learning $b$ generically performs better than learning $v$, and that learning $\eta$ generically performs better than learning $s$ (except when $\epsilon$ is taken large enough that performance starts to degrade). With proper treatment of the singularity in the sampling algorithm when using the denoiser in the construction of $s(t,x)=-\eta(t,x)/\gamma(t)$ – either by capping $t_{0}\neq 0$ and $t_{f}\neq 1$ or by properly tuning $\epsilon(t)$ as discussed in Section 6.2 – our results suggest that learning the denoiser is best practice.

### 7.3 Image generation

In the following, we demonstrate that the proposed method scales straightforwardly to high-dimensional problems like image generation. To this end, we illustrate the use of our approach on the $128\times 128$ Oxford flowers dataset [^47] by testing two different variations of the interpolant for image generation: the one-sided interpolant, using $\rho_{0}=\mathsf{N}(0,\text{\it Id})$, as well as the mirror interpolant, where $\rho_{0}=\rho_{1}$ both represent the data distribution. The purpose of this section is to demonstrate that our theory is well-motivated, and that it provides a framework that is both scalable and flexible. In this regard, image generation is a convenient exercise, but is not the main focus of this work, and we will leave a more thorough study on other datasets such as ImageNet with standard benchmarks such as the Frechet Inception Distance (FID) for a future study.

#### Generation from Gaussian ρ0subscript𝜌0\\rho\_{0}.

We train spatially-linear one-sided interpolants $x_{t}=(1-t)z+tx_{1}$ and $x_{t}=\cos({\tfrac{\pi}{2}}t)z+\sin({\tfrac{\pi}{2}t})x_{1}$ on the $128\times 128$ Oxford flowers dataset, where we take $z\sim\mathsf{N}(0,\text{\it Id})$ and $x_{1}$ from the data distribution. Based on our results for Gaussian mixtures, we learn the drift $b(t,x)$, the score $s(t,x)$, and the denoiser $\eta_{z}(t,x)$ to benchmark our generative models based on ODEs or SDEs. In all cases, we parameterize the networks representing $\hat{\eta}$, $\hat{s}$ and $\hat{b}$ using the U-Net architecture used in [^25]. Minimization of the objective functions given in Section 3.2 is performed using the Adam optimizer. Details of the architecture in both cases and all training hyperparameters are provided in Appendix C.1.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2303.08797/assets/x13.png)

Figure 15: Image generation: Oxford flowers. Example generated flowers from the same initial condition x 0 subscript 𝑥 x\_{0} using either the ODE with ϵ = italic-ϵ \\epsilon=0 and learned b 𝑏 or the SDE for various increasing values of \\epsilon with learned and s 𝑠. For, sampling is done using the dopri5 solver and therefore the number of steps is adaptive. Otherwise, 2000, 2500, and 4000 steps were taken using the Heun solver for 1.0 \\epsilon=1.0 2.0 4.0 respectively.

Like in the case of learning Gaussian mixtures, we use the fourth-order dopri5 solver when sampling with the ODE and the Heun method for the SDE, as detailed in Algorithm 4. When learning a denoiser $\eta_{z}$, we found it beneficial to complete the image generation with a final denoising step, in which we set $\epsilon=0$ and switch the integrator to the one given in (5.14).

Exemplary images generated from the model using the ODE and the SDE with various value of the diffusion coefficient $\epsilon$ are shown in Figure 15, starting from the same sample from $\rho_{0}$. The result illustrates that different images can be generated from the same sample when using the SDE, and their diversity increases as we increase the diffusion coefficient $\epsilon$. To highlight that the model does not memorize the training set, in Figure 16 we compare an example generated image to its five nearest neighbors in the training set (measured in $\ell_{1}$ norm), which appear visually quite different.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2303.08797/assets/x14.png)

Figure 16: No memorization. Top: An example generated image from a trained linear one-sided interpolant. Bottom: Five nearest neighbors (in ℓ 1 subscript \\ell\_{1} norm) to the generated image in the dataset. The nearest neighbors are visually distinct, highlighting that the interpolant does not overfit on the dataset.

#### Mirror interpolant.

We consider the mirror interpolant $x_{t}=x_{1}+\gamma(t)z$, for which (3.34) shows that the drift $b$ is given in terms the denoiser $\eta_{z}$ by $b(t,x)=\dot{\gamma}(t)\eta_{z}(t)$; this means that it is sufficient to only learn an estimate $\hat{\eta}_{z}$ to construct a generative model. Similar to the previous section, we demonstrate this on the Oxford flowers dataset, again making use of a U-Net parameterization for $\hat{\eta}_{z}(t,x)$. Further experimental details can be found in Appendix C.1. In this setup the output image at time $t=1$ is the same as the input image if we use the ODE (2.32); with the SDE, however, we can generate new images from the same input. This is illustrated in Figure 17, where we show how a sample image from the dataset $\rho_{1}$ is pushed forward through the SDE (2.33) with $\epsilon(t)=\epsilon=10$. As can be seen the original image is resampled to a proximal flower not seen in the dataset.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2303.08797/assets/x15.png)

Figure 17: Mirror interpolant. Example trajectory from a learned denoiser model b ^ = γ ˙ ( t ) η z 𝑏 𝛾 𝑡 subscript 𝜂 𝑧 \\hat{b}=\\dot{\\gamma}(t)\\hat{\\eta}\_{z} for the mirror interpolant x 1 + 𝑥 x\_{t}=x\_{1}+\\gamma(t)z on the 128 × 128\\times 128 Oxford flowers dataset with − \\gamma(t)=\\sqrt{t(1-t)}. The parametric form for \\hat{\\eta}\_{z} is the U-Net from 25, with hyperparameter details given in Appendix C. The choice of ϵ italic-ϵ \\epsilon in the generative SDE given in ( 2.33 ) influences the extent to which the generated image differs from the original. Here, 10.0 \\epsilon(t)=10.0.

## 8 Conclusion

The above exposition provides a full treatment of the stochastic interpolant method, as well as a careful consideration of its relation to existing literature. Our goal is to provide a general framework that can be used to devise generative models built upon dynamical transport of measure. To this end, we have detailed mathematical theory and efficient algorithms for constructing both deterministic and stochastic generative models that map between two densities exactly in finite time. Along the way, we have illustrated the various design parameters that can be used to shape this process, with connections, for example, to optimal transport and Schrödinger bridges. While we detail specific instantiations, such as the mirror and one-sided interpolants, we highlight that there is a much broader space of possible designs that may be relevant for future applications. Several candidate application domains include the solution of inverse problems such as image inpainting and super-resolution, spatiotemporal forecasting of dynamical systems, and scientific problems such as sampling of molecular configurations and machine learning-assisted Markov chain Monte-Carlo.

## Acknowledgements

We thank Joan Bruna, Jonathan Niles-Weed, Loucas Pillaud-Vivien, and Cédric Gerbelot for helpful discussions regarding stability estimates of dynamical transport. We are also grateful to Qiang Liu, Ricky Chen, and Yaron Lipman for feedback on previous and related work, and to Kyle Cranmer and Michael Lindsey for discussions on transport costs. We thank Mark Goldstein for insightful comments regarding practical considerations when training denoising-diffusion models. MSA is supported by the National Science Foundation under the award PHY-2141336. EVE is supported by the National Science Foundation under awards DMR-1420073, DMS-2012510, and DMS-2134216, by the Simons Collaboration on Wave Turbulence, Grant No. 617006, and by a Vannevar Bush Faculty Fellowship.

## Appendix A Bridging two Gaussian mixture densities

In this appendix, we consider the case where $\rho_{0}$ and $\rho_{1}$ are both Gaussian mixture densities. We denote by

$$
\displaystyle{\sf N}(x|m,C)
$$
 
$$
\displaystyle=(2\pi)^{-d/2}[\det C]^{-1/2}\exp\left(-\tfrac{1}{2}(x-m)^{\mathsf{T}}C^{-1}(x-m)\right),
$$
$$
\displaystyle=(2\pi)^{-d}\int_{{\mathbb{R}}^{d}}e^{ik\cdot(x-m)-\frac{1}{2}k^{\mathsf{T}}Ck}dk,
$$

the Gaussian probability density with mean vector $m\in{\mathbb{R}}^{d}$ and positive-definite symmetric covariance matrix $C=C^{\mathsf{T}}\in{\mathbb{R}}^{d\times d}$. We assume that

$$
\rho_{0}(x)=\sum_{i=1}^{N_{0}}p^{0}_{i}{\sf N}(x|m^{0}_{i},C^{0}_{i}),\qquad\rho_{1}(x)=\sum_{i=1}^{N_{1}}p^{1}_{i}{\sf N}(x|m^{1}_{i},C^{1}_{i})
$$

where $N_{0},N_{1}\in{\mathbb{N}}$, $p^{0}_{i}>0$ with $\sum_{i=1}^{N_{0}}p_{i}^{0}=1$, $m_{i}^{0}\in{\mathbb{R}}^{d}$, $C_{i}^{0}=(C_{i}^{0})^{\mathsf{T}}\in{\mathbb{R}}^{d\times d}$ positive-definite, and similarly for $p_{i}^{1}$, $m_{i}^{1}$, and $C_{i}^{1}$. We have: propositiongaussmixt Consider the process $x_{t}$ defined in (2.1) using the probability densities in (A.2) and the interpolant in (4.1), i.e.

$$
x^{\mathsf{lin}}_{t}=\alpha(t)x_{0}+\beta(t)x_{1}+\gamma(t)z,
$$

where $x_{0},\sim\rho_{0}$, $x_{1}\sim\rho_{1}$, and $z\sim{\sf N}(0,\text{\it Id})$ with $x_{0}\perp x_{1}\perp z$, and $\alpha,\beta,\gamma^{2}\in C^{2}([0,1])$ satisfy the conditions in (4.2). Denote

$$
m_{ij}(t)=\alpha(t)m^{0}_{i}+\beta(t)m^{1}_{j},\quad C_{ij}(t)=\alpha^{2}(t)C^{0}_{i}+\beta^{2}(t)C^{1}_{j}+\gamma^{2}(t)\text{\it Id},
$$

where $i=1,\ldots,N_{0},$ $j=1,\ldots,N_{1}$. Then the probability density $\rho$ of $x_{t}$ is the Gaussian mixture density

$$
\rho(t,x)=\sum_{i=1}^{N_{0}}\sum_{j=1}^{N_{1}}p^{0}_{i}p^{1}_{j}{\sf N}(x|m_{ij}(t),C_{ij}(t))
$$

and the velocity $b$ and the score $s$ defined in (2.10) and (2.14) are

$$
b(t,x)=\frac{\sum_{i=1}^{N_{0}}\sum_{j=1}^{N_{1}}p^{0}_{i}p^{1}_{j}\left(\dot{m}_{ij}(t)+\tfrac{1}{2}\dot{C}_{ij}(t)C^{-1}_{ij}(t)(x-m^{ij}(t))\right){\sf N}(x|m_{ij}(t),C_{ij}(t))}{\sum_{i=1}^{N_{0}}\sum_{j=1}^{N_{1}}p^{0}_{i}p^{1}_{j}{\sf N}(x|m_{ij}(t),C_{ij}(t))},
$$

and

$$
s(t,x)=-\frac{\sum_{i=1}^{N_{0}}\sum_{j=1}^{N_{1}}p^{0}_{i}p^{1}_{j}C^{-1}_{ij}(t)(x-m_{ij}(t)){\sf N}(x|m_{ij}(t),C_{ij}(t))}{\sum_{i=1}^{N_{0}}\sum_{j=1}^{N_{1}}p^{0}_{i}p^{1}_{j}{\sf N}(x|m_{ij}(t),C_{ij}(t))}.
$$

This proposition implies that $b$ and $s$ grow at most linearly in $x$, and are approximately linear in regions where the modes of $\rho(t,x)$ remain well-separated. In particular, if $\rho_{0}$ and $\rho_{1}$ are both Gaussian densities, $\rho_{0}=\mathsf{N}(m_{0},C_{0})$ and $\rho_{1}=\mathsf{N}(m_{1},C_{1})$, we have

$$
b(t,x)=\dot{m}(t)+\tfrac{1}{2}\dot{C}(t)C^{-1}(t)(x-m(t)),
$$

and

$$
s(t,x)=-C^{-1}(t)(x-m(t)),
$$

where

$$
m(t)=\alpha(t)m_{0}+\beta(t)m_{1},\quad C(t)=\alpha^{2}(t)C_{0}+\beta^{2}(t)C_{1}+\gamma^{2}(t)\text{\it Id}.
$$

Note that the probability flow ODE (2.32) associated with the velocity (A.7) is the linear ODE

$$
\frac{d}{dt}X_{t}=\dot{m}(t)+\tfrac{1}{2}\dot{C}(t)C^{-1}(t)(X_{t}-m(t)).
$$

This equation can only be solved analytically if $\dot{C}(t)$ and $C(t)$ commute (which is the case e.g. if $C_{0}=\text{\it Id}$), but it is easy to see that it always guarantees that

$$
{\mathbb{E}}_{0}X_{t}(x_{0})=m(t),\qquad{\mathbb{E}}_{0}\big{[}(X_{t}(x_{0})-m(t))(X_{t}(x_{0})-m(t))^{\mathsf{T}}\big{]}=C(t),
$$

where $X_{t}(x_{0})$ denotes the solution to (A.10) for the initial condition $X_{t=0}(x_{0})=x_{0}$ and ${\mathbb{E}}_{0}$ denotes expectation over $x_{0}\sim\rho_{0}$. A similar statement is true if we solve (A.10) with final conditions at $t=1$ drawn from $\rho_{1}$. Similarly, the forward SDE (2.33) associated with the velocity (A.7) and the score (A.8) is the linear SDE

$$
dX^{\mathsf{F}}_{t}=\dot{m}(t)dt+\big{(}\tfrac{1}{2}\dot{C}(t)-\epsilon\big{)}C^{-1}(t)(X^{\mathsf{F}}_{t}-m(t))dt+\sqrt{2\epsilon}dW_{t}.
$$

and its solutions are such that

$$
{\mathbb{E}}_{0}{\mathbb{E}}^{x_{0}}_{\mathsf{F}}X_{t}^{\mathsf{F}}=m(t),\qquad{\mathbb{E}}_{0}{\mathbb{E}}^{x_{0}}_{\mathsf{F}}\big{[}(X^{\mathsf{F}}_{t}-m(t))(X_{t}^{\mathsf{F}}-m(t))^{\mathsf{T}}\big{]}=C(t),
$$

where ${\mathbb{E}}_{\mathsf{F}}^{x_{0}}$ denotes expectation over the solution of (A.12) conditional on the event $X^{\mathsf{F}}_{t=0}=x_{0}$ and ${\mathbb{E}}_{0}$ denotes expectation over $x_{0}\sim\rho_{0}$. A similar statement also holds for the backward SDE (2.34).

###### Proof.

The characteristic function of $\rho(t,x)$ is given by

$$
g(t,k)={\mathbb{E}}e^{ik\cdot x_{t}}=\sum_{i=1}^{N_{0}}\sum_{j=1}^{N_{1}}p^{0}_{i}p^{1}_{j}e^{ik\cdot m_{ij}(t)-\tfrac{1}{2}k^{\mathsf{T}}C_{ij}(t)k}.
$$

whose inverse Fourier transform is (A.4). This automatically implies (A.6) since we know from 2.14 that $s=\nabla\log\rho$. To derive (A.7) use the function $m$ defined below in (B.12):

$$
m(t,k)=\sum_{i=1}^{N_{0}}\sum_{j=1}^{N_{1}}p^{0}_{i}p^{1}_{j}(\dot{m}_{ij}(t)+\tfrac{1}{2}i\dot{C}_{ij}(t)k)e^{ik\cdot m_{ij}(t)-\tfrac{1}{2}k^{\mathsf{T}}C_{ij}^{\gamma}(t)k}.
$$

From (B.13), we know that the inverse Fourier transform of this function is $b\rho$, so that we obtain

$$
b(t,x)\rho(t,x)=\sum_{i=1}^{N_{0}}\sum_{j=1}^{N_{1}}p^{0}_{i}p^{1}_{j}\left(\dot{m}_{ij}(t)+\tfrac{1}{2}\dot{C}_{ij}(t)C_{ij}^{-1}(t)(x-m_{ij}(t))\right){\sf N}(x|m_{ij}(t),C^{\epsilon}_{ij}(t)).
$$

This gives (A.5). ∎

## Appendix B Proofs

In this appendix, we provide the details for proofs omitted from the main text. For ease of reading, a copy of the original theorem statement is provided with the proof.

### B.1 Proof of Theorems,, and, and Corollary.

\*

###### Proof.

Let $g(t,k)={\mathbb{E}}e^{ik\cdot x_{t}}$, $k\in{\mathbb{R}}^{d}$, be the characteristic function of $\rho(t,x)$. From the definition of $x_{t}$ in (2.1),

$$
g(t,k)={\mathbb{E}}e^{ik\cdot(I(t,x_{0},x_{1})+\gamma(t)z)}.
$$

Using the independence between $(x_{0},x_{1})$ and $z$, we have

$$
g(t,k)={\mathbb{E}}\left(e^{ik\cdot I(t,x_{0},x_{1})}\right){\mathbb{E}}\left(e^{i\gamma(t)k\cdot z}\right)\equiv g_{0}(t,k)e^{-\tfrac{1}{2}\gamma^{2}(t)|k|^{2}}
$$

where we defined

$$
g_{0}(t,k)={\mathbb{E}}\left(e^{ik\cdot I(t,x_{0},x_{1})}\right)
$$

The function $g_{0}(t,k)$ is the characteristic function of $I(t,x_{0},x_{1})$ with $(x_{0},x_{1})\sim\nu$. From (B.2), we have

$$
|g(t,k)|=|g_{0}(t,k)|e^{-\tfrac{1}{2}\gamma^{2}(t)|k|^{2}}\leq e^{-\tfrac{1}{2}\gamma^{2}(t)|k|^{2}}
$$

Since $\gamma(t)>0$ for all $t\in(0,1)$ by assumption, this shows that

$$
\forall p\in{\mathbb{N}}\ \ \text{and}\ \ t\in(0,1)\quad:\quad\int_{{\mathbb{R}}^{d}}|k|^{p}|g(t,k)|dk<\infty,
$$

implying that $\rho(t,\cdot)$ is in $C^{p}({\mathbb{R}}^{d})$ for any $p\in{\mathbb{N}}$ and all $t\in(0,1)$. From (B.2), we also have

$$
\displaystyle|\partial_{t}g(t,k)|^{2}
$$
 
$$
\displaystyle=\left|{\mathbb{E}}[(ik\cdot\partial_{t}I_{t}(x_{0},x_{1})-\gamma(t)\dot{\gamma}(t)|k|^{2})e^{ik\cdot I_{t}(x_{0},x_{1})}]\right|^{2}e^{-\gamma^{2}(t)|k|^{2}}
$$
 
$$
\displaystyle\leq 2\left(|k|^{2}{\mathbb{E}}\big{[}|\partial_{t}I_{t}(x_{0},x_{1})|^{2}]+|\gamma(t)\dot{\gamma}(t)|^{2}|k|^{4}\right)e^{-\gamma^{2}(t)|k|^{2}}
$$
 
$$
\displaystyle\leq 2\left(|k|^{2}M_{1}+4|\gamma(t)\dot{\gamma}(t)|^{2}|k|^{4}\right)e^{-\gamma^{2}(t)|k|^{2}}
$$

and

$$
\displaystyle|\partial^{2}_{t}g(t,k)|^{2}
$$
 
$$
\displaystyle\leq 4\left(|k|^{2}{\mathbb{E}}\big{[}|\partial^{2}_{t}I_{t}(x_{0},x_{1})|^{2}]+(|\dot{\gamma}(t)|^{2}+\gamma(t)\ddot{\gamma}(t))^{2}|k|^{4}\right)e^{-\gamma^{2}(t)|k|^{2}}
$$
 
$$
\displaystyle\quad+8\left(|k|^{2}{\mathbb{E}}\big{[}|\partial_{t}I_{t}(x_{0},x_{1})|^{4}]+(\gamma(t)\dot{\gamma}(t))^{4}|k|^{8}\right)e^{-\gamma^{2}(t)|k|^{2}}
$$
 
$$
\displaystyle\leq 4\left(|k|^{2}M_{2}+(|\dot{\gamma}(t)|^{2}+\gamma(t)\ddot{\gamma}(t))^{2}|k|^{4}\right)e^{-\gamma^{2}(t)|k|^{2}}
$$
 
$$
\displaystyle\quad+8\left(|k|^{2}M_{1}+(\gamma(t)\dot{\gamma}(t))^{4}|k|^{8}\right)e^{-\gamma^{2}(t)|k|^{2}}
$$

where in both cases we used (2.8) in Assumption 2.5 to get the last inequalities. These imply that

$$
\forall p\in{\mathbb{N}}\ \ \text{and}\ \ t\in(0,1)\quad:\quad\int_{{\mathbb{R}}^{d}}|k|^{p}|\partial_{t}g(t,k)|dk<\infty;\quad\int_{{\mathbb{R}}^{d}}|k|^{p}|\partial^{2}_{t}g(t,k)|dk<\infty
$$

indicating that $\partial_{t}\rho(t,\cdot)$ and $\partial^{2}_{t}\rho(t,\cdot)$ are in $C^{p}({\mathbb{R}}^{d})$ for any $p\in{\mathbb{N}}$, i.e. $\rho\in C^{1}((0,1);C^{p}({\mathbb{R}}^{d}))$ as claimed. To show that $\rho$ is also positive, denote by $\mu_{0}(t,dx)$ the unique (by the Fourier inversion theorem) probability measure associated with $g_{0}(t,k)$, i.e. the measure such that

$$
g_{0}(t,k)=\int_{{\mathbb{R}}^{d}}e^{ik\cdot x}\mu_{0}(t,dx).
$$

From (B.2) and the convolution theorem it follows that we can express $\rho$ as

$$
\rho(t,x)=\int_{{\mathbb{R}}^{d}}\frac{e^{-|x-y|^{2}/(2\gamma^{2}(t))}}{(2\pi\gamma^{2}(t))^{d/2}}\mu_{0}(t,dy),
$$

This shows that $\rho>0$ for all $(t,x)\in(0,1)\times{\mathbb{R}}^{d}$. Since $x_{t=0}=x_{0}$ and $x_{t=1}=x_{1}$ by definition of the interpolant, we also have $\rho(0)=\rho_{0}$ and $\rho(1)=\rho_{1}$, which shows that $\rho$ is also positive and in $C^{p}({\mathbb{R}}^{d})$ at $t=0,1$ by Assumption 2.5. Note that since $\rho\in C^{1}((0,1);C^{p}({\mathbb{R}}^{d}))$ and is positive, we also immediately deduce that $s=\nabla\log\rho=\nabla\rho/\rho\in C^{1}((0,1);(C^{p}({\mathbb{R}}^{d}))^{d})$.

To show that $\rho$ satisfies the TE (2.9), we take the time derivative of (B.1) to deduce that

$$
\partial_{t}g(t,k)=ik\cdot m(t,k)
$$

where $m:[0,1]\times{\mathbb{R}}^{d}\to{\mathbb{C}}^{d}$ is the vector-valued function defined as

$$
m(t,k)={\mathbb{E}}\left((\partial_{t}I(t,x_{0},x_{1})+\dot{\gamma}(t)z)e^{ik\cdot x_{t}}\right).
$$

By definition of the conditional expectation, $m(t,k)$ can be expressed as

$$
\displaystyle m(t,k)
$$
 
$$
\displaystyle=\int_{{\mathbb{R}}^{d}}{\mathbb{E}}\left((\partial_{t}I(t,x_{0},x_{1})+\dot{\gamma}(t)z)e^{ik\cdot x_{t}}|x_{t}=x\right)\rho(t,x)dx
$$
 
$$
\displaystyle=\int_{{\mathbb{R}}^{d}}e^{ik\cdot x}{\mathbb{E}}\left((\partial_{t}I(t,x_{0},x_{1})+\dot{\gamma}(t)z)|x_{t}=x\right)\rho(t,x)dx
$$
 
$$
\displaystyle=\int_{{\mathbb{R}}^{d}}e^{ik\cdot x}b(t,x)\rho(t,x)dx
$$

where the last equality follows from the definition of $b$ in (2.10). Inserting (B.13) in (B.11), we deduce that this equation can be written in real space as the TE (2.9).

Let us now investigate the regularity of $b$. To that end, we go back to $m$ and use the independence between $x_{0}$, $x_{1}$, and $z$, as well as Gaussian integration by parts to deduce that

$$
m(t,k)={\mathbb{E}}\left((\partial_{t}I(t,x_{0},x_{1})-i\gamma(t)\dot{\gamma}(t)k)e^{ik\cdot I(t,x_{1},x_{0})}\right)e^{-\tfrac{1}{2}\gamma^{2}(t)|k|^{2}},
$$

As a result

$$
\displaystyle|m(t,k)|^{2}
$$
 
$$
\displaystyle=\left|{\mathbb{E}}\left((\partial_{t}I(t,x_{0},x_{1})-i\gamma(t)\dot{\gamma}(t)k)e^{ik\cdot I(t,x_{1},x_{0})}\right)\right|^{2}e^{-\gamma^{2}(t)|k|^{2}}
$$
 
$$
\displaystyle\leq 2\left({\mathbb{E}}\big{[}|\partial_{t}I(t,x_{0},x_{1})|^{2}\big{]}+|\gamma(t)\dot{\gamma}(t)|^{2}|k|^{2}\right)e^{-\gamma^{2}(t)|k|^{2}}
$$
 
$$
\displaystyle\leq 2M_{1}e^{-\gamma^{2}(t)|k|^{2}},
$$

and

$$
\displaystyle|\partial_{t}m(t,k)|^{2}
$$
 
$$
\displaystyle\leq 4\left({\mathbb{E}}\big{[}|\partial^{2}_{t}I(t,x_{0},x_{1})|^{2}+(\gamma(t)\ddot{\gamma}(t)+\dot{\gamma}^{2}(t))^{2}\right)e^{-\gamma^{2}(t)|k|^{2}}
$$
 
$$
\displaystyle\quad+8|k|^{2}\left({\mathbb{E}}\big{|}\partial_{t}I(t,x_{0},x_{1})|^{4}\big{]}+(\gamma(t)\dot{\gamma}(t))^{4}|k|^{4}\right)e^{-\gamma^{2}(t)|k|^{2}}
$$
 
$$
\displaystyle\leq 4\left(M_{1}+(\gamma(t)\ddot{\gamma}(t)+\dot{\gamma}^{2}(t))^{2}\right)e^{-\gamma^{2}(t)|k|^{2}}
$$
 
$$
\displaystyle\quad+8|k|^{2}\left(M_{2}+(\gamma(t)\dot{\gamma}(t))^{4}|k|^{4}\right)e^{-\gamma^{2}(t)|k|^{2}},
$$

where in both cases the last inequalities follow from (2.8). Therefore

$$
\forall p\in{\mathbb{N}}\ \ \text{and}\ \ t\in(0,1)\quad:\quad\int_{{\mathbb{R}}^{d}}|k|^{p}|m(t,k)|dk<\infty,\quad\int_{{\mathbb{R}}^{d}}|k|^{p}|\partial_{t}m(t,k)|dk<\infty,
$$

which implies that the inverse Fourier transform of $m$ is a function $j:[0,1]\times{\mathbb{R}}^{d}\to{\mathbb{R}}^{d}$ that satisfies $j(t,\cdot)\in(C^{p}({\mathbb{R}}^{d}))^{d}$ for any $p\in{\mathbb{N}}$ and any $t\in(0,1)$ and can be expressed as

$$
j(t,x)=(2\pi)^{-d}\int_{{\mathbb{R}}^{d}}e^{-ik\cdot x}m(t,k)dk={\mathbb{E}}\left(\partial_{t}I(t,x_{0},x_{1})+\dot{\gamma}(t)z|x_{t}=x\right)\rho(t,x)\equiv b(t,x)\rho_{t}(x)
$$

where the last equality follows from the definition of $b$ in (2.10). We deduce that $b\in C^{0}([0,1];(C^{p}({\mathbb{R}}^{d}))^{d})$ for any $p\in{\mathbb{N}}$ since $j\in C^{0}([0,1];(C^{p}({\mathbb{R}}^{d}))^{d})$ and $\rho\in C^{1}([0,1];C^{p}({\mathbb{R}}^{d}))$, and $\rho>0$.

Finally, let us establish (2.11). By (2.8) we have

$$
\displaystyle\int_{{\mathbb{R}}^{d}}|b(t,x)|^{2}\rho(t,x)dx
$$
 
$$
\displaystyle=\int_{{\mathbb{R}}^{d}}|{\mathbb{E}}\left(\partial_{t}I(t,x_{0},x_{1})+\dot{\gamma}(t)z|x_{t}=x\right)|^{2}\rho(t,x)dx
$$
 
$$
\displaystyle\leq 2\int_{{\mathbb{R}}^{d}}{\mathbb{E}}\left(|\partial_{t}I(t,x_{0},x_{1})|^{2}+|\dot{\gamma}(t)|^{2}|z|^{2}|x_{t}=x\right)\rho(t,x)dx
$$
 
$$
\displaystyle\leq 2{\mathbb{E}}\big{[}|\partial_{t}I(t,x_{0},x_{1})|^{2}+|\dot{\gamma}(t)|^{2}|z|^{2}\big{]}
$$
 
$$
\displaystyle<2M_{1}^{1/2}+2d|\dot{\gamma}(t)|^{2},
$$

so that this integral is bounded for all $t\in(0,1)$. To analyze its behavior at the end points, notice that the decomposition (2.27) implies that

$$
\displaystyle b_{0}(x)
$$
 
$$
\displaystyle\equiv\lim_{t\to 0}b(t,x)={\mathbb{E}}_{1}[\partial_{t}I(0,x,x_{1})]-\lim_{t\to 0}\dot{\gamma}(t)\gamma(t)s_{0}(x),
$$
$$
\displaystyle b_{1}(x)
$$
 
$$
\displaystyle\equiv\lim_{t\to 1}b(t,x)={\mathbb{E}}_{0}[\partial_{t}I(0,x_{0},x)]-\lim_{t\to 1}\dot{\gamma}(t)\gamma(t)s_{1}(x),
$$

where $s_{0}=\nabla\log\rho_{0}$, $s_{1}=\nabla\log\rho_{1}$, ${\mathbb{E}}_{0}$ and ${\mathbb{E}}_{1}$ denote expectations over $x_{0}\sim\rho_{0}$ and $x_{1}\sim\rho_{1}$, respectively, and we used the property that $x_{t=0}=x_{0}$ and $x_{t=1}=x_{1}$. Since $\lim_{t\to 0,1}\dot{\gamma}(t)\gamma(t)$ exists by our assumption that $\gamma^{2}\in C^{1}([0,1])$, $b_{0}$ and $b_{1}$ are well defined, and

$$
\int_{{\mathbb{R}}^{d}}|b_{0}(x)|^{2}\rho_{0}(x)dx<\infty,\qquad\int_{{\mathbb{R}}^{d}}|b_{1}(x)|^{2}\rho_{1}(x)dx<\infty.
$$

by Assumption 2.5. As a result, the integral in (B.19) is continuous at $t=0$ and $t=1$, $b$ must be integrable on $[0,1]$, and (2.11) holds. ∎

\*

###### Proof.

By definition of $\rho$, the objective $\mathcal{L}_{b}$ defined in (2.13) can also be written as

$$
\displaystyle\mathcal{L}_{b}[\hat{b}]
$$
 
$$
\displaystyle=\int_{{\mathbb{R}}^{d}}\left(\tfrac{1}{2}|\hat{b}(t,x)|^{2}-{\mathbb{E}}\left((\partial_{t}I(t,x_{0},x_{1})+\dot{\gamma}(t)z|x_{t}=x\right)\cdot\hat{b}(t,x)\right)\rho(t,x)dx
$$
 
$$
\displaystyle=\int_{{\mathbb{R}}^{d}}\left(\tfrac{1}{2}|\hat{b}(t,x)|^{2}-b(t,x)\cdot\hat{b}(t,x)\right)\rho(t,x)dx
$$

where we used the definition of $b$ in (2.10). This quadratic objective is bounded from below since

$$
\displaystyle\mathcal{L}_{b}[\hat{b}]
$$
 
$$
\displaystyle=\tfrac{1}{2}\int_{{\mathbb{R}}^{d}}\left|\hat{b}(t,x)-b(t,x)\right|^{2}\rho(t,x)dx-\tfrac{1}{2}\int_{{\mathbb{R}}^{d}}\left|b(t,x)\right|^{2}\rho(t,x)dx
$$
 
$$
\displaystyle\geq-\tfrac{1}{2}\int_{{\mathbb{R}}^{d}}\left|b(t,x)\right|^{2}\rho(t,x)dx>-\infty
$$

where the last inequality follows from (2.11). Since $\rho_{t}$ is positive the minimizer of (B.22) is unique and given by $\hat{b}=b$.

∎

\*

###### Proof.

Since $\rho\in C^{1}((0,1);C^{p}({\mathbb{R}}^{d}))$ and is positive by Theorem 2.2, we already know that $s=\nabla\log\rho=\nabla\rho/\rho\in C^{1}((0,1);(C^{p}({\mathbb{R}}^{d}))^{d})$. To establish (2.14), note that, for $t\in(0,1)$ where $\gamma(t)>0$, we have

$$
{\mathbb{E}}\left(ze^{i\gamma(t)k\cdot z}\right)=-\gamma^{-1}(t)(i\partial_{k}){\mathbb{E}}e^{i\gamma(t)k\cdot z}=-\gamma^{-1}(t)(i\partial_{k})e^{-\tfrac{1}{2}\gamma^{2}(t)|k|^{2}}=i\gamma(t)ke^{-\tfrac{1}{2}\gamma^{2}(t)|k|^{2}}.
$$

As a result, using the independence between $x_{0}$, $x_{1}$, and $z$, we have

$$
{\mathbb{E}}\left(ze^{ik\cdot x_{t}}\right)=i\gamma(t)kg(t,k)
$$

where $g$ is the characteristic function of $x_{t}$ defined in (B.1). Using the properties of the conditional expectation, the left-hand side of this equation can be written as

$$
{\mathbb{E}}\left(ze^{ik\cdot x_{t}}\right)=\int_{{\mathbb{R}}^{d}}{\mathbb{E}}\left(ze^{ik\cdot x_{t}}|x_{t}=x\right)\rho(t,x)dx=\int_{{\mathbb{R}}^{d}}{\mathbb{E}}\left(z|x_{t}=x\right)e^{ikx_{t}}\rho(t,x)dx
$$

Since the left hand side of (B.25) is the Fourier transform of $-\gamma(t)\nabla\rho(t,x)$, we deduce that

$$
{\mathbb{E}}\left(z|x_{t}=x\right)\rho(t,x)=-\gamma(t)\nabla\rho(t,x)=-\gamma(t)s(t,x)\rho(t,x).
$$

Since $\rho(t,x)>0$, this implies (2.14) for $t\in(0,1)$ where $\gamma(t)>0$.

To establish (2.15), notice that

$$
\displaystyle\int_{{\mathbb{R}}^{d}}|s(t,x)|^{2}\rho(t,x)dx
$$
 
$$
\displaystyle=\int_{{\mathbb{R}}^{d}}|{\mathbb{E}}\left((\gamma^{-1}(t)z|x_{t}=x\right)|^{2}\rho(t,x)dx
$$
 
$$
\displaystyle\leq\int_{{\mathbb{R}}^{d}}\gamma^{-2}(t){\mathbb{E}}\left(|z|^{2}|x_{t}=x\right)\rho(t,x)dx
$$
 
$$
\displaystyle\leq d\gamma^{-2}(t).
$$

This means that this integral is bounded for all $t\in(0,1)$. Since the integral is also continuous at $t=0$ and $t=1$, with values given by (2.7), it must be integrable on $[0,1]$ and (2.15) holds.

The objective $\mathcal{L}_{s}$ defined in (2.16) can also be written as

$$
\displaystyle\mathcal{L}_{s}[\hat{s}]
$$
 
$$
\displaystyle=\int_{{\mathbb{R}}^{d}}\left(\tfrac{1}{2}|\hat{s}(t,x)|^{2}+\gamma^{-1}(t){\mathbb{E}}\left(z|x_{t}=x\right)\cdot\hat{s}(t,x)\right)\rho(t,x)dx,
$$
$$
\displaystyle=\int_{{\mathbb{R}}^{d}}\left(\tfrac{1}{2}|\hat{s}(t,x)|^{2}-s(t,x)\cdot\hat{s}(t,x)\right)\rho(t,x)dx,
$$

where we used the definition of $s$ in (2.14). This quadratic objective is bounded from below since

$$
\displaystyle\mathcal{L}_{s}[\hat{s}]
$$
 
$$
\displaystyle=\tfrac{1}{2}\int_{{\mathbb{R}}^{d}}\left|\hat{s}(t,x)-s(t,x)\right|^{2}\rho(t,x)dx-\tfrac{1}{2}\int_{{\mathbb{R}}^{d}}\left|s(t,x)\right|^{2}\rho(t,x)dx,
$$
$$
\displaystyle\geq-\tfrac{1}{2}\int_{{\mathbb{R}}^{d}}\left|s(t,x)\right|^{2}\rho(t,x)dx>-\infty
$$

where the last inequality follows from (2.15). Since $\rho$ is positive the minimizer of (B.29) is unique and given by $\hat{s}=s$. ∎

\*

###### Proof.

The forward FPE (2.20) and the backward FPE (2.22) are direct consequences of the TE (2.9) and (2.14), since the equality

$$
\epsilon(t)\Delta\rho=\epsilon(t)\nabla\cdot(\rho\nabla\log\rho)=\epsilon(t)\nabla\cdot(s\rho)
$$

can be used to convert between these equations.

∎

### B.2 Proof of Lemma

\*

###### Proof.

The SDE (2.33) and the ODE (2.32) are the evolution equations for the processes whose densities solve (2.20) and (2.9), respectively. The equation that requires some explanation is the backwards SDE (2.34), which can be solved backwards in time from $t=1$ to $t=0$. As discussed in the main text, by definition, its solution is $X^{\mathsf{B}}_{t}=Z^{\mathsf{F}}_{1-t}$ where $Z^{\mathsf{F}}_{t}$ solves the forward SDE

$$
dZ^{\mathsf{F}}_{t}=-b_{\mathsf{B}}(1-t,Z^{\mathsf{F}}_{t})dt+\sqrt{2\epsilon}dW_{t}.
$$

To see how to write the backward Itô formula (2.36) note that given any $f\in C^{1}([0,1];C_{0}^{2}({\mathbb{R}}^{d}))$, we have

$$
\displaystyle df(1-t,Z^{\mathsf{F}}_{t})
$$
 
$$
\displaystyle=-\partial_{t}f(1-t,Z^{\mathsf{F}}_{t})dt+\nabla f(1-t,Z^{\mathsf{F}}_{t})\cdot dZ^{\mathsf{F}}_{t}+\epsilon\Delta f(1-t,Z^{\mathsf{F}}_{t})dt
$$
 
$$
\displaystyle=-\partial_{t}f(1-t,Z^{\mathsf{F}}_{t})dt+\left(-b_{\mathsf{B}}(1-t,Z^{\mathsf{F}}_{t})\cdot\nabla f(1-t,Z^{\mathsf{F}}_{t})+\epsilon\Delta f(1-t,Z^{\mathsf{F}}_{t})\right)dt
$$
 
$$
\displaystyle\quad+\sqrt{2\epsilon}\nabla f(1-t,Z^{\mathsf{F}}_{t})\cdot dW_{t}
$$

In integral form, this equation can be written as

$$
\displaystyle f(1,Z^{\mathsf{F}}_{1})
$$
 
$$
\displaystyle=f(1-t,Z^{\mathsf{F}}_{1-t})
$$
 
$$
\displaystyle\quad-\int_{1-t}^{1}\left(\partial_{t}f(1-s,Z^{\mathsf{F}}_{s})+b_{\mathsf{B}}(1-s,Z^{\mathsf{F}}_{s})\cdot\nabla f(1-s,Z^{\mathsf{F}}_{s})-\epsilon\Delta f(1-s,Z^{\mathsf{F}}_{s})\right)ds
$$
 
$$
\displaystyle\quad-\sqrt{2\epsilon}\int_{1-t}^{1}\nabla f(1-s,Z^{\mathsf{F}}_{s})\cdot dW_{s}.
$$

Using $X^{\mathsf{B}}_{t}=Z^{\mathsf{F}}_{1-t}$ and $W^{\mathsf{B}}_{t}=-W_{1-t}$ and changing integration variable from $s$ to $1-s$, this is

$$
\displaystyle f(1,X^{\mathsf{B}}_{0})
$$
 
$$
\displaystyle=f(1-t,X^{\mathsf{B}}_{t})-\int_{t}^{1}\left(\partial_{t}f(s,X^{\mathsf{B}}_{s})+b_{\mathsf{B}}(s,X^{\mathsf{B}}_{s})\cdot\nabla f(s,X^{\mathsf{B}}_{s})-\epsilon\Delta f(s,X^{\mathsf{B}}_{s})\right)ds
$$
 
$$
\displaystyle\quad-\sqrt{2\epsilon}\int_{t}^{1}\nabla f(s,X^{\mathsf{B}}_{s})\cdot dW^{\mathsf{B}}_{s},
$$

In differential form, this is equivalent to saying that

$$
\displaystyle df(t,X^{\mathsf{B}}_{t})
$$
 
$$
\displaystyle=\partial_{t}f(t,X^{\mathsf{B}}_{t})dt+\nabla f(t,X^{\mathsf{B}}_{t})\cdot dX^{\mathsf{B}}_{t}-\epsilon\Delta f(X^{\mathsf{B}}_{t})dt
$$
 
$$
\displaystyle=\left(\partial_{t}f(t,X^{\mathsf{B}}_{t})+b_{\mathsf{B}}(t,X^{\mathsf{B}}_{t})\cdot\nabla f(t,X^{\mathsf{B}}_{t})-\epsilon\Delta f(t,X^{\mathsf{B}}_{t})\right)dt+\sqrt{2\epsilon}\nabla f(t,X^{\mathsf{B}}_{t})\cdot dW_{t}
$$

which is the backward Itô formula (2.36). Similarly, by the Itô isometries we have: for any $g\in C^{0}([0,1];(C_{0}^{0}({\mathbb{R}}_{d}))^{d})$ and $t\in[0,1]$

$$
{\mathbb{E}}^{x}\int_{1-t}^{1}g(s,Z^{\mathsf{F}}_{s})\cdot dW_{s}=0,\qquad{\mathbb{E}}^{x}\left|\int_{1-t}^{1}g(s,Z^{\mathsf{F}}_{s})\cdot dW_{s}\right|^{2}=\int_{1-t}^{1}{\mathbb{E}}^{x}|g(s,Z^{\mathsf{F}}_{s})|^{2}ds.
$$

Written in terms of $X^{\mathsf{B}}_{t}$, these are (2.37). ∎

#### Derivation of ().

For any $t\in[0,1]$ the score is the minimizer of

$$
\displaystyle\int_{{\mathbb{R}}^{d}}|\hat{s}(t,x)-\nabla\log\rho(t,x)|^{2}\rho(t,x)dx
$$
 
$$
\displaystyle=\int_{{\mathbb{R}}^{d}}\left(|\hat{s}(t,x)|^{2}-2\hat{s}(t,x)\cdot\nabla\log\rho(t,x)+|\nabla\log\rho(t,x)|^{2}\right)\rho(t,x)dx
$$
 
$$
\displaystyle=\int_{{\mathbb{R}}^{d}}\left(|\hat{s}(t,x)|^{2}+2\nabla\cdot\hat{s}(t,x)+|\nabla\log\rho(t,x)|^{2}\right)\rho(t,x)dx,
$$

where we used the identity $\hat{s}\cdot\nabla\log\rho\,\rho=\hat{s}\cdot\nabla\rho$ and integration by parts to obtain the second equality. The last term involving $|\nabla\log\rho|^{2}$ is a constant in $\hat{s}$ that can be neglected for optimization. Expressing the remaining terms as an expectation over $x_{t}$ and integrating the result in time gives (2.30).

### B.3 Proofs of Lemmas and, and Theorem.

\*

###### Proof.

Using (2.38), we compute analytically

$$
\displaystyle\frac{d}{dt}\mathsf{KL}(\rho(t)\>\|\>\hat{\rho}(t))
$$
 
$$
\displaystyle=\frac{d}{dt}\int_{{\mathbb{R}}^{d}}\log\left(\frac{\rho}{\hat{\rho}}\right)\rho dx
$$
 
$$
\displaystyle=\int_{{\mathbb{R}}^{d}}\hat{\rho}\left(\frac{\partial_{t}\rho}{\hat{\rho}}-\frac{\rho}{(\hat{\rho})^{2}}\partial_{t}\hat{\rho}\right)dx+\int\log\left(\frac{\rho}{\hat{\rho}}\right)\partial_{t}\rho dx
$$
 
$$
\displaystyle=-\int_{{\mathbb{R}}^{d}}\left(\frac{\rho}{\hat{\rho}}\right)\partial_{t}\hat{\rho}dx+\int_{{\mathbb{R}}^{d}}\log\left(\frac{\rho}{\hat{\rho}}\right)\partial_{t}\rho dx
$$
 
$$
\displaystyle=\int_{{\mathbb{R}}^{d}}\left(\frac{\rho}{\hat{\rho}}\right)\nabla\cdot\left(\hat{b}\hat{\rho}\right)dx-\int\log\left(\frac{\rho}{\hat{\rho}}\right)\nabla\cdot\left(b\rho\right)dx
$$
 
$$
\displaystyle=-\int_{{\mathbb{R}}^{d}}\nabla\left(\frac{\rho}{\hat{\rho}}\right)\cdot\hat{b}\hat{\rho}dx+\int\left(\nabla\log\rho-\nabla\log\hat{\rho}\right)\cdot b\rho dx
$$
 
$$
\displaystyle=-\int_{{\mathbb{R}}^{d}}\left(\frac{\nabla\rho}{\hat{\rho}}-\frac{\rho\nabla\hat{\rho}}{\hat{\rho}^{2}}\right)\cdot\hat{b}\hat{\rho}dx+\int_{{\mathbb{R}}^{d}}\left(\nabla\log\rho-\nabla\log\hat{\rho}\right)\cdot b\rho dx
$$
 
$$
\displaystyle=\int_{{\mathbb{R}}^{d}}\left(\nabla\log\hat{\rho}-\nabla\log\rho\right)\cdot\hat{b}\rho dx+\int_{{\mathbb{R}}^{d}}\left(\nabla\log\rho-\nabla\log\hat{\rho}\right)\cdot b\rho dx
$$
 
$$
\displaystyle=\int_{{\mathbb{R}}^{d}}\left(\nabla\log\hat{\rho}-\nabla\log\rho\right)\cdot(\hat{b}-b)\rho dx.
$$

where we omitted the argument $(t,x)$ of all functions for simplicity of notation. Integrating both sides from $00$ to $1$ completes the proof. ∎

\*

###### Proof.

Similar to the proof of Lemma 2.4, we can use the FPE in (2.40) to compute $\frac{d}{dt}\mathsf{KL}(\rho(t)\>\|\>\hat{\rho}(t))$, which leads to the main result. Instead, we take a simpler approach, leveraging the result in Lemma 2.4. We re-write the Fokker-Planck equations in (2.40) as the (score-dependent) transport equations

$$
\displaystyle\partial_{t}\rho
$$
 
$$
\displaystyle=-\nabla\cdot((b_{\mathsf{F}}-\epsilon\nabla\log\rho)\rho),
$$
$$
\displaystyle\partial_{t}\hat{\rho}
$$
 
$$
\displaystyle=-\nabla\cdot((\hat{b}_{\mathsf{F}}-\epsilon\nabla\log\hat{\rho})\hat{\rho}).
$$

Applying Lemma 2.4 directly, we find that

$$
\displaystyle\mathsf{KL}(\rho(1)\>\|\>\hat{\rho}(1))=\int_{0}^{1}\int_{{\mathbb{R}}^{d}}\left[\left(\nabla\log\hat{\rho}-\nabla\log\rho\right)\cdot\left(\left[\hat{b}_{\mathsf{F}}-\epsilon\nabla\log\hat{\rho}\right]-\left[b_{\mathsf{F}}-\epsilon\nabla\log\rho\right]\right)\right]\rho dxdt.
$$

Expanding the above,

$$
\displaystyle\mathsf{KL}(\rho(1)\>\|\>\hat{\rho}(1))
$$
 
$$
\displaystyle=\int_{0}^{1}\int_{{\mathbb{R}}^{d}}\left[\left(\nabla\log\hat{\rho}-\nabla\log\rho\right)\cdot(\hat{b}_{\mathsf{F}}-b_{\mathsf{F}})\right]\rho dxdt
$$
 
$$
\displaystyle\qquad-\epsilon\int_{0}^{1}\int_{{\mathbb{R}}^{d}}\left[\left|\nabla\log\rho-\nabla\log\hat{\rho}\right|^{2}\right]\rho dxdt,
$$

which proves the first part of the lemma. Now, by Young’s inequality, it holds for any fixed $\eta>0$ that

$$
\displaystyle\mathsf{KL}(\rho_{1}\>\|\>\hat{\rho}(1))
$$
 
$$
\displaystyle\leq\frac{1}{2\eta}\int_{0}^{1}\int_{{\mathbb{R}}^{d}}\left|\hat{b}_{\mathsf{F}}-b_{\mathsf{F}}\right|^{2}\rho_{t}dxdt
$$
 
$$
\displaystyle\qquad+\left(\tfrac{1}{2}\eta-\epsilon\right)\int_{0}^{1}\int_{{\mathbb{R}}^{d}}\left|\nabla\log\rho_{t}-\nabla\log\hat{\rho}_{t}\right|^{2}\rho_{t}dxdt.
$$

Hence, for $\eta=2\epsilon$,

$$
\displaystyle\mathsf{KL}(\rho_{1}\>\|\>\hat{\rho}(1))
$$
 
$$
\displaystyle\leq\frac{1}{4\epsilon}\int_{0}^{1}\int_{{\mathbb{R}}^{d}}\left|\hat{b}_{\mathsf{F}}-b_{\mathsf{F}}\right|^{2}\rho dxdt,
$$

which proves the second part of the lemma. ∎

\*

###### Proof.

Observe that by Proposition 2.3, the target density $\rho_{1}=\rho(1,\cdot)$ is the density of the process $X_{t}$ that evolves according to SDE (2.33). By Lemma 2.4 and an additional application of Young’s inequality, we then have that

$$
\displaystyle\mathsf{KL}(\rho_{1}\>\|\>\hat{\rho}(1))
$$
 
$$
\displaystyle\leq\frac{1}{2\epsilon}\int_{0}^{1}\int_{{\mathbb{R}}^{d}}\left(|\hat{b}-b|^{2}+\epsilon^{2}\left|s-\hat{s}\right|^{2}\right)\rho dxdt.
$$

Using the definition of $\mathcal{L}_{b}[\hat{b}]$ and $\mathcal{L}_{s}[\hat{s}]$ in (2.13), and (2.16), we conclude that

$$
\mathsf{KL}(\rho_{1}\>\|\>\hat{\rho}(1))\leq\frac{1}{2\epsilon}\left(\mathcal{L}_{b}[\hat{b}]-\min_{\hat{b}}\mathcal{L}_{b}[\hat{b}]\right)+\frac{\epsilon}{2}\left(\mathcal{L}_{s}[\hat{s}]-\min_{\hat{s}}\mathcal{L}_{s}[\hat{s}]\right),
$$

which is (2.45). Using that $b=v-\gamma\dot{\gamma}s$ and $\hat{b}=\hat{v}-\gamma\dot{\gamma}\hat{s}$, we can write (B.43) in this case as

$$
\displaystyle\mathsf{KL}(\rho_{1}\>\|\>\hat{\rho}(1))
$$
 
$$
\displaystyle\leq\frac{1}{4\epsilon}\int_{0}^{1}\int_{{\mathbb{R}}^{d}}\left|\hat{v}-v-(\gamma(t)\dot{\gamma}(t)-\epsilon)(\hat{s}-s)\right|^{2}\rho dxdt,
$$
$$
\displaystyle\leq\frac{1}{2\epsilon}\int_{0}^{1}\int_{{\mathbb{R}}^{d}}\left(\left|\hat{v}-v\right|^{2}+(\gamma(t)\dot{\gamma}(t)-\epsilon)^{2}\left|\hat{s}-s\right|^{2}\right)\rho dxdt,
$$
$$
\displaystyle\leq\frac{1}{2\epsilon}\left(\mathcal{L}_{v}[\hat{v}]-\min_{\hat{v}}\mathcal{L}_{v}[\hat{v}]\right)+\frac{\max_{t\in[0,1]}(\gamma(t)\dot{\gamma}(t)-\epsilon)^{2}}{2\epsilon}\left(\mathcal{L}_{s}[\hat{s}]-\min_{\hat{v}}\mathcal{L}_{s}[\hat{s}]\right),
$$

which is the final part of the theorem. ∎

### B.4 Proofs of Lemma and Theorem

\*

###### Proof.

If $\hat{\rho}$ solves the TE (2.50) and $X_{s,t}$ solves the ODE (2.51), we have

$$
\displaystyle\frac{d}{dt}\hat{\rho}(t,X_{s,t}(x))
$$
 
$$
\displaystyle=\partial_{t}\hat{\rho}(t,X_{s,t}(x))+b(t,X_{s,t}(x))\cdot\nabla\hat{\rho}(t,X_{s,t}(x))
$$
 
$$
\displaystyle=-\nabla\cdot b(t,X_{s,t}(x))\hat{\rho}(t,X_{s,t}(x))
$$

This equation implies that

$$
\displaystyle\frac{d}{dt}\left(\exp\left(\int_{s}^{t}\nabla\cdot b(\tau,X_{s,\tau}(x))d\tau\right)\hat{\rho}(t,X_{s,t}(x))\right)=0.
$$

Integrating (B.50) over $[0,t]$ and setting $s=t$ in the result gives

$$
\displaystyle\hat{\rho}(t,x)=\exp\left(-\int_{0}^{t}\nabla\cdot b(\tau,X_{t,\tau}(x))d\tau\right)\hat{\rho}(0,X_{t,0}(x))
$$

If we use the initial condition $\hat{\rho}(0)=\rho_{0}$, this gives (2.52). Similarly, Integrating (B.50) on $[t,1]$ and setting $s=t$ in the result gives

$$
\displaystyle\hat{\rho}(t,x)=\exp\left(\int_{t}^{1}\nabla\cdot b(\tau,X_{t,\tau}(x))d\tau\right)\hat{\rho}(1,X_{t,1}(x))
$$

If we use the final condition $\hat{\rho}(1)=\rho_{1}$, this gives (2.53).

∎

\*

###### Proof.

Evaluating $d\hat{\rho}_{\mathsf{F}}(t,Y^{\mathsf{B}}_{t})$ via the backward Itô formula (2.36) we obtain

$$
\displaystyle d\hat{\rho}_{\mathsf{F}}(t,Y_{t}^{\mathsf{B}})
$$
 
$$
\displaystyle=\partial_{t}\hat{\rho}_{\mathsf{F}}(t,Y_{t}^{\mathsf{B}})dt+\nabla\hat{\rho}_{\mathsf{F}}(t,Y_{t}^{\mathsf{B}})\cdot dY^{\mathsf{B}}_{t}-\epsilon\Delta\hat{\rho}_{\mathsf{F}}(t,Y_{t}^{\mathsf{B}})dt
$$
 
$$
\displaystyle=\partial_{t}\hat{\rho}_{\mathsf{F}}(t,Y_{t}^{\mathsf{B}})dt+\nabla\hat{\rho}_{\mathsf{F}}(t,Y_{t}^{\mathsf{B}})\cdot\hat{b}_{\mathsf{F}}(t,Y^{\mathsf{B}}_{t})dt+\sqrt{2\epsilon}\nabla\hat{\rho}_{\mathsf{F}}(t,Y_{t}^{\mathsf{B}})\cdot dW^{\mathsf{B}}_{t}-\epsilon\Delta\hat{\rho}_{\mathsf{F}}(t,Y_{t}^{\mathsf{B}})dt
$$
 
$$
\displaystyle=-\nabla\cdot\hat{b}_{\mathsf{F}}(t,Y^{\mathsf{B}}_{t})\hat{\rho}_{\mathsf{F}}(t,Y^{\mathsf{B}}_{t})dt+\sqrt{2\epsilon}\nabla\hat{\rho}_{\mathsf{F}}(t,Y_{t}^{\mathsf{B}})\cdot dW^{\mathsf{B}}_{t}.
$$

where we used (2.56) in the second step and (2.57) in the last one. This equation can be written as a total differential in the form

$$
\displaystyle d\left(\exp\left(-\int_{t}^{1}\nabla\cdot\hat{b}_{\mathsf{F}}(\tau,Y^{\mathsf{B}}_{\tau})d\tau\right)\hat{\rho}_{\mathsf{F}}(t,Y_{t}^{\mathsf{B}})\right)
$$
 
$$
\displaystyle=\sqrt{2\epsilon}\exp\left(-\int_{t}^{1}\nabla\cdot\hat{b}_{\mathsf{F}}(\tau,Y^{\mathsf{B}}_{\tau})d\tau\right)\nabla\hat{\rho}_{\mathsf{F}}(t,Y_{t}^{\mathsf{B}})\cdot dW^{\mathsf{B}}_{t},
$$

which after integration over $t\in[0,1]$ becomes

$$
\displaystyle\hat{\rho}_{\mathsf{F}}(1,Y_{t=1}^{\mathsf{B}})-\exp\left(-\int_{0}^{1}\nabla\cdot\hat{b}_{\mathsf{F}}(t,Y^{\mathsf{B}}_{t})dt\right)\rho_{0}(Y^{\mathsf{B}}_{t=0})
$$
 
$$
\displaystyle=\sqrt{2\epsilon}\int_{0}^{1}\exp\left(-\int_{t}^{1}\nabla\cdot\hat{b}_{\mathsf{F}}(\tau,Y^{\mathsf{B}}_{\tau})d\tau\right)\nabla\hat{\rho}(t,Y_{t}^{\mathsf{B}})\cdot dW^{\mathsf{B}}_{t}.
$$

where we used $\hat{\rho}(0)=\rho_{0}$. Taking an expectation conditioned on the event $Y_{t=1}^{\mathsf{B}}=x$ and using that the term on the right-hand side has mean zero, we find that

$$
\hat{\rho}_{\mathsf{F}}(1,Y_{t=1}^{\mathsf{B}})-{\mathbb{E}}^{x}_{\mathsf{B}}\exp\left(-\int_{0}^{1}\nabla\cdot\hat{b}_{\mathsf{F}}(t,Y^{\mathsf{B}}_{t})dt\right)\rho_{0}(Y^{\mathsf{B}}_{t=0})=0.
$$

This gives (2.58).

Similarly, evaluating $d\hat{\rho}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})$ via Itô’s formula, we obtain

$$
\displaystyle d\hat{\rho}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})
$$
 
$$
\displaystyle=\partial_{t}\hat{\rho}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})dt+\nabla\hat{\rho}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})\cdot dY^{\mathsf{F}}_{t}+\epsilon\Delta\hat{\rho}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})dt
$$
 
$$
\displaystyle=\partial_{t}\hat{\rho}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})dt+\nabla\hat{\rho}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})\cdot\hat{b}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})dt+\sqrt{2\epsilon}\nabla\hat{\rho}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})\cdot dW_{t}+\epsilon\Delta\hat{\rho}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})dt
$$
 
$$
\displaystyle=-\nabla\cdot\hat{b}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})\hat{\rho}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})dt+\sqrt{2\epsilon}\nabla\hat{\rho}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})\cdot dW_{t}.
$$

where we used (2.55) in the second step and (2.59) in the last one. This equation can be written as a total differential in the form

$$
\displaystyle d\left(\exp\left(\int_{0}^{t}\nabla\cdot\hat{b}_{\mathsf{B}}(\tau,Y^{\mathsf{F}}_{\tau})d\tau\right)\hat{\rho}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})\right)
$$
 
$$
\displaystyle=\sqrt{2\epsilon}\exp\left(\int_{0}^{t}\nabla\cdot\hat{b}_{\mathsf{B}}(\tau,Y^{\mathsf{F}}_{\tau})d\tau\right)\nabla\hat{\rho}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})\cdot dW_{t}.
$$

Integrating the above on $t\in[0,1]$, we find that

$$
\displaystyle\exp\left(\int_{0}^{1}\nabla\cdot\hat{b}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})dt\right)\rho_{1}(Y^{\mathsf{F}}_{1})-\hat{\rho}_{\mathsf{B}}(0,Y^{\mathsf{F}}_{t=0})
$$
 
$$
\displaystyle=\sqrt{2\epsilon}\int_{0}^{1}\exp\left(\int_{0}^{t}\nabla\cdot\hat{b}_{\mathsf{B}}(\tau,Y^{\mathsf{F}}_{\tau})d\tau\right)\nabla\hat{\rho}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})\cdot dW_{t}.
$$

where we used $\hat{\rho}(1)=\rho_{1}$. Taking an expectation conditioned on the event $Y^{\mathsf{F}}_{t=0}=x$ and applying the Itô isometry, we deduce that

$$
{\mathbb{E}}_{\mathsf{F}}^{x}\left(\exp\left(\int_{0}^{1}\nabla\cdot\hat{b}_{\mathsf{B}}(t,Y^{\mathsf{F}}_{t})dt\right)\rho_{1}(Y^{\mathsf{F}}_{t=1})\right)-\hat{\rho}_{\mathsf{B}}(0,y)=0.
$$

This gives (2.60).

∎

### B.5 Proof of Theorem

\*

###### Proof.

Let us first consider what happens on the interval $t\in[0,\delta]$ where $I(t,x_{0},x_{1})=x_{0}$ by assumption and the stochastic interpolant (3.2) with $x_{0}$ fixed and $a(t)=a>0$ reduces to

$$
x_{t}=x_{0}+\sqrt{2at(1-t)}z\quad\text{with}\quad x_{0}\text{fixed},\ z\sim{\sf N}(0,\text{\it Id}).
$$

The law of this process is simply

$$
x_{t}\sim{\sf N}(x_{0},2at(1-t)\text{\it Id}),
$$

which means that its density satisfies for all $t>0$ the FPE

$$
\partial_{t}\rho+2at\nabla\cdot\left(s(t,x)\rho\right)=a\Delta\rho,
$$

where $s(t,x)=\nabla\log\rho(t,x)$ is the score, which is explicitly given by

$$
s(t,x)=-\frac{x-x_{0}}{2at(1-t)}\qquad\text{for}\ \ t\in(0,\delta].
$$

This means that the drift term in the FPE (B.63) can be written as

$$
2ats(t,x)=-\frac{x-x_{0}}{1-t}\qquad\text{for}\ \ t\in(0,\delta].
$$

and this equality also holds in the limit as $t\to 0$. It is also easy to check that

$$
-\frac{x-x_{0}}{2at(1-t)}=-\frac{\sqrt{2at}}{\sqrt{(1-t)}}{\mathbb{E}}(z|x_{t}=x)\qquad\text{for}\ \ t\in[0,\delta]
$$

which is consistent with the drift in (3.10) on $t\in[0,\delta]$ since $\partial_{t}I(t,x_{0},x_{1})=0$ on this interval. Since the right-hand side of (B.65) is nonsingular at $t=0$ it also means that the SDE (3.11) is well-defined for $t\in[0,\delta]$ and the law of its solutions coincide with that of the process $x_{t}$ defined (B.61), $X_{t}^{\mathsf{d}}\sim{\sf N}(x_{0},2at(1-t)\text{\it Id})$ for $t\in[0,\delta]$.

Considering next what happens on $t\in(\delta,1]$, since $\gamma(t)=\sqrt{2at(1-t)}$ is only zero at the endpoint $t=1$ where we assume that the density $\rho_{1}$ satisfies Assumption 2.5 we can mimick all the arguments in the proof of of Theorem 2.2 and Corollaries 2.6 and 2.3 to terminate the proof. ∎

Note that this proof shows that is enough to have $\partial_{t}I(t=0,x_{0},x_{1})=0$, since this implies that $I(t,x_{0},x_{1})=x_{0}+O(t^{2})$. It also shows that, while it is key to use an SDE on $t\in[0,\delta]$ so that the generative process can spread the mass away from $x_{0}$, the diffusive noise is no longer necessary and we could switch back to a probability flow ODE on $t\in(\delta,1]$ (using a time-dependent $a(t)$ to that effect with $a(t)=0$ for $t\in(\delta,1]$).

### B.6 Proof of Lemma and Theorem.

\*

###### Proof.

By definition of the map $T$ in (3.37), if $x_{0}\sim\rho_{0}$ and $x_{1}\sim\rho_{1}$, then $T^{-1}(0,x_{0})\sim{\sf N}(0,\text{\it Id})$ and $T^{-1}(1,x_{1})\sim{\sf N}(0,\text{\it Id})$. As a result, since $x_{0}$, $x_{1}$, and $z$ are independent, and $z\sim{\sf N}(0,\text{\it Id})$, we have

$$
\alpha(t)T^{-1}(0,x_{0})+\beta(t)T^{-1}(1,x_{1})+\gamma(t)z\sim{\sf N}(0,\alpha^{2}(t)\text{\it Id}+\beta^{2}(t)\text{\it Id}+\gamma^{2}(t)\text{\it Id})={\sf N}(0,\text{\it Id})
$$

where the second equality follows from the condition $\alpha^{2}(t)+\beta^{2}(t)+\gamma^{2}(t)=1$. Therfore, using again the definition of the map $T$

$$
T(t,\alpha(t)T^{-1}(0,x_{0})+\beta(t)T^{-1}(1,x_{1})+\gamma(t)z)\sim\rho(t),
$$

and we are done. ∎

\_\*

###### Proof.

Denote by $\hat{\rho}(t,x)$ the density of $\hat{x}_{t}$ and define the current $\hat{\jmath}:[0,1]\times{\mathbb{R}}^{d}\to{\mathbb{R}}^{d}$ as

$$
\hat{\jmath}(t,x)={\mathbb{E}}\big{(}\partial_{t}\hat{I}_{t}+\gamma(t)z|x=\hat{x}_{t}\big{)}\hat{\rho}(t,x).
$$

The max-min problem (3.39) can then be formulated as the constrained optimization problem:

$$
\displaystyle\max_{\hat{\rho},\hat{\jmath}}\min_{\hat{u}}
$$
 
$$
\displaystyle\int_{0}^{1}\int_{{\mathbb{R}}^{d}}\left(\tfrac{1}{2}|\hat{u}(t,x)|^{2}\hat{\rho}(t,x)-\hat{u}(t,x)\cdot\hat{\jmath}(t,x)\right)dxdt
$$
 
$$
\displaystyle\partial_{t}\hat{\rho}+\nabla\cdot\hat{\jmath}=\epsilon\Delta\hat{\rho},\quad\hat{\rho}(t=0)=\rho_{0},\quad\hat{\rho}(t=1)=\rho_{1}
$$

To solve this problem we can use the extended objective

$$
\displaystyle\max_{\hat{\rho},\hat{\jmath}}\min_{\hat{u}}
$$
 
$$
\displaystyle\Bigg{(}\int_{0}^{1}\int_{{\mathbb{R}}^{d}}\left(\tfrac{1}{2}|\hat{u}(t,x)|^{2}\hat{\rho}(t,x)-\hat{u}(t,x)\cdot\hat{\jmath}(t,x)\right)dxdt
$$
 
$$
\displaystyle-\int_{0}^{1}\int_{{\mathbb{R}}^{d}}\lambda(t,x)\left(\partial_{t}\hat{\rho}(t,x)+\nabla\cdot\hat{\jmath}(t,x)-\epsilon\Delta\hat{\rho}(t,x)\right)dxdt
$$
 
$$
\displaystyle+\int_{{\mathbb{R}}^{d}}\eta_{0}(x)\left(\hat{\rho}(0,x)-\rho_{0}(x)\right)dx-\int_{{\mathbb{R}}^{d}}\eta_{1}(x)\left(\hat{\rho}(1,x)-\rho_{1}(x)\right)dx\Bigg{)}
$$

where $\lambda(t,x)$, $\eta_{0}(x)$, and $\eta_{1}(x)$ are Lagrange multipliers used to enforce the constraints. The unique minimizer $(\rho,j,\lambda)$ of this optimization problem solves the Euler-Lagrange equations:

$$
\displaystyle\partial_{t}\rho+\nabla\cdot j=\epsilon\Delta\rho,\quad\rho(t=0)=\rho_{0}\quad\rho(t=1)=\rho_{1}
$$
 
$$
\displaystyle\partial_{t}\lambda+\tfrac{1}{2}|u|^{2}=-\epsilon\Delta\lambda,
$$
$$
\displaystyle j=u\rho
$$
 
$$
\displaystyle u=\nabla\lambda
$$

We can use the last two equations to write the first two as (3.36), with $u=\nabla\lambda$. Since under Assumption 3.10 there is an interpolant that realizes the density $\rho(t)$ that solves (3.36), we conclude that an optimizer $(I,u)$ of the the max-min problem (3.39) exists. For any optimizer, $I$ will be such that $\rho(t)$ is the density of $x_{t}=I(t,x_{0},x_{1})+\gamma(t)z$, and $u$ will satisfy $u=\nabla\lambda$. ∎

### B.7 Proof of Theorem

\*

###### Proof.

Use

$$
\beta(t_{j+1})\beta^{-1}(t_{j})=1+\dot{\beta}(t_{j})\beta^{-1}(t_{i})(t_{j+1}-t_{j})+O\big{(}(t_{j+1}-t_{j})^{2}\big{)}
$$

and

$$
\alpha(t_{j+1})-\alpha(t_{j})\beta(t_{j+1})\beta^{-1}(t_{j})=\big{(}\dot{\alpha}(t_{j})-\alpha(t_{j})\dot{\beta}(t_{j})\beta^{-1}(t_{i}))(t_{j+1}-t_{j})+O\big{(}(t_{j+1}-t_{j})^{2}\big{)}
$$

to deduce that (5.14) implies

$$
\displaystyle X^{\mathsf{den}}_{{j+1}}
$$
 
$$
\displaystyle=X^{\mathsf{den}}_{j}+\dot{\beta}(t_{j})\beta^{-1}(t_{j})X^{\mathsf{den}}_{j}(t_{j+1}-t_{j})
$$
 
$$
\displaystyle+\big{(}\dot{\alpha}(t_{j})-\alpha(t_{j})\dot{\beta}(t_{j})\beta^{-1}(t_{i}))\eta^{\mathsf{os}}_{z}(t_{j},X^{\mathsf{den}}_{j})(t_{j+1}-t_{j})+O\big{(}(t_{j+1}-t_{j})^{2}\big{)}.
$$

or equivalently

$$
\displaystyle\frac{X^{\mathsf{den}}_{{j+1}}-X^{\mathsf{den}}_{j}}{t_{j+1}-t_{j}}=\dot{\beta}(t_{j})\beta^{-1}(t_{j})X^{\mathsf{den}}_{j}+\big{(}\dot{\alpha}(t_{j})-\alpha(t_{j})\dot{\beta}(t_{j})\beta^{-1}(t_{i}))\eta^{\mathsf{os}}_{z}(t_{j},X^{\mathsf{den}}_{j})+O(t_{j+1}-t_{j}).
$$

Taking the limit as $N,j\to\infty$ with $j/N\to t\in[0,1]$, we recover (5.15) and deduce that $X^{\mathsf{den}}_{j}\to X_{t}$. ∎

Note that the proof shows that the result of Theorem 5.2 also holds if we use a nonuniform grid of times $t_{j}$, $j\in\{1,\ldots,N\}$.

### B.8 Proof of Theorem

\*

###### Proof.

The first part of the statement can be established by following the same steps as in the proof of Theorem 2.2 and Corollary 2.6. For the proof of the second part, use first (5.17) written as $x^{\mathsf{rec}}_{t}=M(t,z)$ in (5.22) to deduce that

$$
b^{\mathsf{rec}}(t,x^{\mathsf{rec}}_{t})=\dot{\alpha}(t)N(t,M(t,z))+\dot{\beta}(t)X_{t=1}(N(t,M(t,z)))=\dot{\alpha}(t)z+\dot{\beta}(t)X_{t=1}(z)=\dot{x}^{\mathsf{rec}}_{t}
$$

This shows that (5.22) is the unique minimizer of (5.19). Next, use (5.23) written as $X^{\mathsf{rec}}_{t}(x)=M(t,x)$ in (5.22) to deduce that

$$
b^{\mathsf{rec}}(t,X^{\mathsf{rec}}_{t}(x))=\dot{\alpha}(t)N(t,M(t,x))+\dot{\beta}(t)X_{t=1}(N(t,M(t,x)))=\dot{\alpha}(t)x+\dot{\beta}(t)X_{t=1}(x)
$$

This implies that (5.23) solves (5.21), and since the solution of this ODE is unique, we are done. ∎

## Appendix C Experimental Specifications

Details for the experiments in Section 7.1 are provided here. Feed forward neural networks of depth $4$ and width $512$ are used for each model of the velocities $b$, $v$, and $s$. Training was done for $7000$ iterations on batches comprised of $25$ draws from the base, $400$ draws from the target, and $100$ time slices. At each iteration, we used a variance reduction technique based on antithetic sampling, in which two samples $\pm z$ are used for each evaluation of the loss. The objectives given in (2.13) and (2.16) were optimized using the Adam optimizer. The learning rate was set to $.002$ and was dropped by a factor of $2$ every $1500$ iterations of training. To integrate the ODE/SDE when drawing samples, we used the Heun-based integrator as suggested in [^32].

### C.1 Image Experiments

#### Network Architectures.

For all image generation experiments, the U-Net architecture originally proposed in [^25] is used. The specification of architecture hyperparameters as well as training hyperparameters are given in Table 18. The same architecture is used regardless of whether learning $b,v,s,$ or $\eta$.

|  | ImageNet 32 $\times$ 32 | Flowers | Flowers (Mirror) |
| --- | --- | --- | --- |
| Dimension | 32 $\times$ 32 | 128 $\times$ 128 | 128 $\times$ 128 |
| \# Training point | 1,281,167 | 315,123 | 315,123 |
| Batch Size | 512 | 64 | 64 |
| Training Steps | 8 $\times$ 10 <sup>5</sup> | 3.5 $\times$ 10 <sup>5</sup> | 8 $\times$ 10 <sup>5</sup> |
| Hidden dim | 256 | 128 | 128 |
| Attention Resolution | 64 | 64 | 64 |
| Learning Rate (LR) | $0.0002$ | $0.0002$ | $0.0002$ |
| LR decay (1k epochs) | 0.995 | 0.995 | 0.985 |
| U-Net dim mult | \[1,2,2,2\] | \[1,1,2,3,4\] | \[1,1,2,3,4\] |
| Learned $t$ sinusoidal embedding | Yes | Yes | Yes |
| $t_{0},t_{f}$ for $t\sim\text{Unif}[t_{0},t_{f}]$, learning $\eta$ | \[0.0, 1.0\] | n/a | n/a |
| $t_{0},t_{f}$ for $t\sim\text{Unif}[t_{0},t_{f}]$, learning $s$ | n/a | \[.0002,.9998\] | \[.0002,.9998\] |
| $t_{0},t_{f}$ when sampling with ODE | \[0.0, 1.0\] | \[0.0001, 0.9999\] | \[.0001,.9999\] |
| $t_{0},t_{f}$ when sampling with SDE, $\eta$ | \[0.0, 0.97\] + denoising | n/a | n/a |
| $t_{0},t_{f}$ when sampling with SDE, $s$ | n/a | \[.0001,.9999\] | \[.0001,.9999\] |
| $\gamma(t)$ in $x_{t}$ | $\sqrt{(t(1-t)}$ | $\sqrt{(t(1-t)}$ | $\sqrt{(10t(1-t)}$ |
| EMA decay rate | 0.9999 | 0.9999 | 0.9999 |
| EMA start iteration | 10000 | 10000 | 10000 |
| $\#$ GPUs | 2 | 4 | 2 |

Table 18: Hyperparameters and architecture for image datasets.

When using the SDE and learning $\eta_{z}$, we found that integrating to a time slightly before $t_{f}=1.0$ and using the denoising formula (5.14) beyond this point provided the best results, as described in the main text.

[^1]: Michael S. Albergo, Mark Goldstein, Nicholas M. Boffi, Rajesh Ranganath, and Eric Vanden-Eijnden. Stochastic interpolants with data-dependent couplings. arXiv, October 2023.

[^2]: Michael S. Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In International Conference on Learning Representations, 2023.

[^3]: Brian D.O. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313–326, 1982.

[^4]: Dominique Bakry and Michel Émery. Diffusions hypercontractives. In Jacques Azéma and Marc Yor, editors, Séminaire de Probabilités XIX 1983/84, pages 177–206, Berlin, Heidelberg, 1985. Springer Berlin Heidelberg.

[^5]: Heli Ben-Hamu, Samuel Cohen, Joey Bose, Brandon Amos, Maximilian Nickel, Aditya Grover, Ricky T. Q. Chen, and Yaron Lipman. Matching normalizing flows and probability paths on manifolds. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 1749–1763. PMLR, 2022.

[^6]: Jean-David Benamou and Yann Brenier. A computational fluid mechanics solution to the monge-kantorovich mass transfer problem. Numerische Mathematik, 84(3):375–393, 2000.

[^7]: Nicholas M Boffi and Eric Vanden-Eijnden. Probability flow solution of the fokker–planck equation. Machine Learning: Science and Technology, 4(3):035012, jul 2023.

[^8]: Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion schrödinger bridge with applications to score-based generative modeling. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, 2021.

[^9]: Yann Brenier. Polar factorization and monotone rearrangement of vector-valued functions. Communications on Pure and Applied Mathematics, 44(4):375–417, 1991.

[^10]: Hongrui Chen, Holden Lee, and Jianfeng Lu. Improved analysis of score-based generative modeling: User-friendly bounds under minimal smoothness assumptions. ICML, 2022.

[^11]: Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018.

[^12]: Scott Chen and Ramesh Gopinath. Gaussianization. In T. Leen, T. Dietterich, and V. Tresp, editors, Advances in Neural Information Processing Systems, volume 13. MIT Press, 2000.

[^13]: Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru Zhang. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions. In The Eleventh International Conference on Learning Representations, 2023.

[^14]: Tianrong Chen, Guan-Horng Liu, and Evangelos Theodorou. Likelihood training of schrödinger bridge using forward-backward SDEs theory. In International Conference on Learning Representations, 2022.

[^15]: Yongxin Chen, Tryphon T. Georgiou, and Michele Pavon. Stochastic control liaisons: Richard sinkhorn meets gaspard monge on a schrödinger bridge. SIAM Review, 63(2):249–313, 2021.

[^16]: Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real NVP. In International Conference on Learning Representations, 2017.

[^17]: Tim Dockhorn, Arash Vahdat, and Karsten Kreis. Score-based generative modeling with critically-damped langevin diffusion. In International Conference on Learning Representations (ICLR), 2022.

[^18]: Joseph L. Doob. Classical potential theory and its probabilistic counterpart, volume 262 of Grundlehren der Mathematischen Wissenschaften \[Fundamental Principles of Mathematical Sciences\]. Springer-Verlag, New York, 1984.

[^19]: Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios. Neural spline flows. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

[^20]: Ahmed El Alaoui and Andrea Montanari. An information-theoretic view of stochastic localization. IEEE Trans. Inf. Theory, 68(11):7423–7426, 2022.

[^21]: Ronen Eldan. Thin shell implies spectral gap up to polylog via a stochastic localization scheme. Geometric and Functional Analysis, 23(2):532–569, 2013.

[^22]: Chris Finlay, Joern-Henrik Jacobsen, Levon Nurbekyan, and Adam Oberman. How to train your neural ODE: the world of Jacobian and kinetic regularization. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 3154–3164. PMLR, 13–18 Jul 2020.

[^23]: Jerome H. Friedman. Exploratory projection pursuit. Journal of the American Statistical Association, 82(397):249–266, 1987.

[^24]: Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, and David Duvenaud. Scalable reversible generative models with free-form continuous dynamics. In International Conference on Learning Representations, 2019.

[^25]: Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 6840–6851. Curran Associates, Inc., 2020.

[^26]: Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, 2023.

[^27]: Chin-Wei Huang, Ricky T. Q. Chen, Christos Tsirigotis, and Aaron Courville. Convex potential flows: Universal probability distributions with optimal transport and convex optimization. In International Conference on Learning Representations, 2021.

[^28]: Chin-Wei Huang, David Krueger, Alexandre Lacoste, and Aaron Courville. Neural autoregressive flows. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 2078–2087. PMLR, 10–15 Jul 2018.

[^29]: Aapo Hyvärinen. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6(24):695–709, 2005.

[^30]: Aapo Hyvärinen. Sparse code shrinkage: Denoising of nongaussian data by maximum likelihood estimation. Neural Computation, 11(7):1739–1768, 1999.

[^31]: Zahra Kadkhodaie and Eero P. Simoncelli. Solving linear inverse problems using the prior implicit in a denoiser, 2021.

[^32]: Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Proc. NeurIPS, 2022.

[^33]: Young-Heon Kim and Emanuel Milman. A generalization of caffarelli’s contraction theorem via (reverse) heat flow. Mathematische Annalen, 354:827–862, 2010.

[^34]: Diederik P Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. On density estimation with diffusion models. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, 2021.

[^35]: Yann LeCun, Sumit Chopra, and Raia Hadsell. A tutorial on energy-based learning. In Gökhan BakIr, Thomas Hofmann, Alexander J Smola, Bernhard Schölkopf, and Ben Taskar, editors, Predicting structured data, chapter 10. MIT press, 2007.

[^36]: Holden Lee, Jianfeng Lu, and Yixin Tan. Convergence for score-based generative modeling with polynomial complexity. arXiv:2206.06227, 2022.

[^37]: Holden Lee, Jianfeng Lu, and Yixin Tan. Convergence of score-based generative modeling for general data distributions. In Shipra Agrawal and Francesco Orabona, editors, Proceedings of The 34th International Conference on Algorithmic Learning Theory, volume 201 of Proceedings of Machine Learning Research, pages 946–985, 2023.

[^38]: Christian Léonard. A survey of the schrödinger problem and some of its connections with optimal transport. Discrete and Continuous Dynamical Systems, 34(4):1533–1574, 2014.

[^39]: Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.

[^40]: Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport, 2022.

[^41]: Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023.

[^42]: Xingchao Liu, Lemeng Wu, Mao Ye, and Qiang Liu. Let us build bridges: Understanding and extending diffusion generative models, 2022.

[^43]: Cheng Lu, Kaiwen Zheng, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Maximum likelihood training for score-based diffusion ODEs by high order denoising score matching. In Proceedings of the 39th International Conference on Machine Learning, 2022.

[^44]: Dimitra Maoutsa, Sebastian Reich, and Manfred Opper. Interacting particle solutions of fokker–planck equations through gradient–log–density estimation. Entropy, 22(8), 2020.

[^45]: Andrea Montanari. Sampling, diffusions, and stochastic localization, 2023.

[^46]: Vinod Nair and Geoffrey E. Hinton. Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th International Conference on International Conference on Machine Learning, ICML’10, page 807–814, Madison, WI, USA, 2010. Omnipress.

[^47]: Maria-Elena Nilsback and Andrew Zisserman. A visual vocabulary for flower classification. In IEEE Conference on Computer Vision and Pattern Recognition, volume 2, pages 1447–1454, 2006.

[^48]: Derek Onken, Samy Wu Fung, Xingjian Li, and Lars Ruthotto. Ot-flow: Fast and accurate continuous normalizing flows via optimal transport. Proceedings of the AAAI Conference on Artificial Intelligence, 35(10):9223–9232, May 2021.

[^49]: Félix Otto and Cédric Villani. Generalization of an inequality by talagrand and links with the logarithmic sobolev inequality. Journal of Functional Analysis, 173(2):361–400, 2000.

[^50]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked autoregressive flow for density estimation. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, page 2335–2344, Red Hook, NY, USA, 2017. Curran Associates Inc.

[^51]: Stefano Peluchetti. Non-denoising forward-time diffusions, 2022.

[^52]: Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In Francis Bach and David Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 1530–1538, Lille, France, 07–09 Jul 2015. PMLR.

[^53]: Eero P. Simoncelli and Edward H. Adelson. Noise removal via bayesian wavelet coring. In Proceedings of 3rd IEEE International Conference on Image Processing, volume 1, pages 379–382 vol.1, 1996.

[^54]: Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Francis Bach and David Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2256–2265, Lille, France, 07–09 Jul 2015. PMLR.

[^55]: Vignesh Ram Somnath, Matteo Pariset, Ya-Ping Hsieh, Maria Rodriguez Martinez, Andreas Krause, and Charlotte Bunne. Aligned diffusion schrödinger bridges. In The 39th Conference on Uncertainty in Artificial Intelligence, 2023.

[^56]: Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency Models. ICML, 2023.

[^57]: Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-based diffusion models. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 1415–1428. Curran Associates, Inc., 2021.

[^58]: Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

[^59]: Yang Song and Diederik P. Kingma. How to train your energy-based models, 2021.

[^60]: Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

[^61]: Charles M. Stein. Estimation of the Mean of a Multivariate Normal Distribution. The Annals of Statistics, 9(6):1135 – 1151, 1981.

[^62]: Xuan Su, Jiaming Song, Chenlin Meng, and Stefano Ermon. Dual diffusion implicit bridges for image-to-image translation. In The Eleventh International Conference on Learning Representations, 2023.

[^63]: Esteban G. Tabak and Cristina V. Turner. A family of nonparametric density estimation algorithms. Communications on Pure and Applied Mathematics, 66(2):145–164, 2013.

[^64]: Esteban G. Tabak and Eric Vanden-Eijnden. Density estimation by dual ascent of the log-likelihood. Communications in Mathematical Sciences, 8(1):217 – 233, 2010.

[^65]: Alexander Tong, Jessie Huang, Guy Wolf, David Van Dijk, and Smita Krishnaswamy. TrajectoryNet: A dynamic optimal transport network for modeling cellular dynamics. In Hal DaumÃ© III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 9526–9536. PMLR, 13–18 Jul 2020.

[^66]: Alexander Tong, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Kilian Fatras, Guy Wolf, and Yoshua Bengio. Conditional flow matching: Simulation-free dynamic optimal transport, 2023.

[^67]: Cédric Villani. Optimal transport: old and new, volume 338. Springer, 2009.

[^68]: Pascal Vincent. A Connection Between Score Matching and Denoising Autoencoders. Neural Computation, 23(7):1661–1674, 2011.

[^69]: Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma with denoising diffusion GANs. In International Conference on Learning Representations, 2022.

[^70]: Liu Yang and George Em Karniadakis. Potential flow generator with l2 optimal transport regularity for generative models. IEEE Transactions on Neural Networks and Learning Systems, 33(2):528–538, 2022.

[^71]: Linfeng Zhang, Weinan E, and Lei Wang. Monge-Ampère flow for generative modeling, 2018.