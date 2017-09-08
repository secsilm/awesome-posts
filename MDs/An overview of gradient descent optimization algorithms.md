# 梯度下降优化算法概述

> 原文作者简介：[Sebastian Ruder](http://ruder.io/) 是我非常喜欢的一个博客作者，是 NLP 方向的博士生，目前供职于一家做 NLP 相关服务的爱尔兰公司 [AYLIEN](http://aylien.com/)，博客主要是写机器学习、NLP和深度学习相关的文章。

> 本文原文是 [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/index.html)，同时作者也在 arXiv 上发了一篇同样内容的 [论文](https://arxiv.org/abs/1609.04747)。本文结合了两者来翻译，但是阅读原文我个人建议读博客中的，感觉体验更好点。水平有限，如有错误欢迎指出。翻译尽量遵循原文意思，但不意味着逐字逐句。

## Abstract

梯度下降算法虽然最近越来越流行，但是始终是作为一个「黑箱」在使用，因为对他们的优点和缺点的实际解释（practical explainations）很难实现。这篇文章致力于给读者提供这些算法工作原理的一个直观理解。在这篇概述中，我们将研究梯度下降的不同变体，总结挑战，介绍最常见的优化算法，介绍并行和分布式设置的架构，并且也研究了其他梯度下降优化策略。

## Introduction

梯度下降是最流行的优化算法之一，也是目前优化神经网络最常用的算法。同时，每一个最先进的深度学习库都包含了梯度下降算法的各种变体的实现（例如 [lasagne](http://lasagne.readthedocs.io/en/latest/modules/updates.html)，[caffe](http://caffe.berkeleyvision.org/tutorial/solver.html)，[keras](https://keras.io/optimizers/)）。然而始终是作为一个「黑箱」在使用，因为对他们的优点和缺点的实际解释很难实现。这篇文章致力于给读者提供这些算法工作原理的一个直观理解。我们首先介绍梯度下降的不同变体，然后简单总结下在训练中的挑战。接着，我们通过展示他们解决这些挑战的动机以及如何推导更新规则来介绍最常用的优化算法。我们也会简要介绍下在并行和分布式架构中的梯度下降。最后，我们会研究有助于梯度下降的其他策略。

梯度下降是一种最小化目标函数 $J(\theta)$ 的方法，其中 $\theta \in \mathbb{R^d}$ 是模型参数，而最小化目标函数是通过在其关于 $\theta$ 的 [梯度](https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6) $\nabla_\theta J(\theta)$ 的相反方向来更新 $\theta$ 来实现的。而学习率（learning rate）则决定了在到达（局部）最小值的过程中每一步走多长。换句话说，我们沿着目标函数的下坡方向来达到一个山谷。如果你对梯度下降不熟悉，你可以在 [这里](http://cs231n.github.io/optimization-1/) 找到一个很好的关于优化神经网络的介绍。

## Gradient descent variants

依据计算目标函数梯度使用的数据量的不同，有三种梯度下降的变体。根据数据量的大小，我们在参数更新的准确性和执行更新所需时间之间做了一个权衡。

### Batch gradient descent

标准的梯度下降，即批量梯度下降（batch gradient descent）（ *译者注：以下简称 BGD* ），在整个训练集上计算损失函数关于参数 $\theta$ 的梯度。

$$\theta = \theta - \eta \cdot \nabla_\theta J( \theta)$$

由于为了一次参数更新我们需要在整个训练集上计算梯度，导致 BGD 可能会非常慢，而且在训练集太大而不能全部载入内存的时候会很棘手。BGD 也不允许我们在线更新模型参数，即实时增加新的训练样本。

下面是 BGD 的代码片段：

```python
for i in range(nb_epochs):
    params_grad = evaluate_gradient(loss_function, data, params)
    params = params - learning_rate * params_grad
```

其中 `nb_epochs` 是我们预先定义好的迭代次数（epochs），我们首先在整个训练集上计算损失函数关于模型参数 `params` 的梯度向量 `params_grad`。其实目前最新的深度学习库都已经提供了关于一些参数的高效自动求导。如果你要自己求导求梯度，那你最好使用梯度检查（gradient checking），在 [这里](http://cs231n.github.io/neural-networks-3/) 查看关于如何进行合适的梯度检查的提示。

然后我们在梯度的反方向更新模型参数，而学习率决定了每次更新的步长大小。BGD 对于凸误差曲面（convex error surface）保证收敛到全局最优点，而对于非凸曲面（non-convex surface）则是局部最优点。

### Stochastic gradient descent