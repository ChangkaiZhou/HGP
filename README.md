Variational Model Selection of Inducing Points in Sparse Heteroscedastic Gaussian Process Regression
====

This is the implementation of the heteroscedastic GP (HGP) with greedy EM algorithm of adding inducing points developed in "*[Changkai Zhou, Wensheng Wang, Variational Model Selection of Inducing Points in Sparse Heteroscedastic Gaussian Process Regression].*" Please see the paper for further details. These codes based on the HGP in https://github.com/LiuHaiTao01.

We here focus on the heteroscedastic Gaussian process regression $y = f + \mathcal{N}(0, \exp(g))$ which integrates the latent function and the noise together in a unified non-parametric Bayesian framework. Though showing flexible and powerful performance, HGP suffers from the cubic time complexity, which strictly limits its application to big data. 

To improve the scalability of HGP, a variational sparse inference algorithm, named VSHGP, has been developed to handle large-scale datasets. This is performed by introducing $m$ latent variables $\mathbf{f}_m$ for $\mathbf{f}$, and $u$ latent variables $\mathbf{g}_u$ for $\mathbf{g}$. Furthermore, to enhance the model capability of capturing quick-varying features, the Bayesian committee machine (BCM) formalism were used to distribute the learning over $M$ local VSHGP experts $\{\mathcal{M}_i\}_{i=1}^M$ with many inducing points, and aggregate their predictive distributions. At the same time, the distributed mode scales DVSHGP up to arbitrary data size!

In order to figure out how many inducing points are enough for (D)VSHGP to summarize all the data, we proposed a posterior strategy. It iteratively adds inducing points and then trains. With the early stop criterion, the (D)VSHGP can stop at the right time.

To run the example file, execute:

```
Demo_DVSHGP_toy.m
```


