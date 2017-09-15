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

随机梯度下降（ *译者注：以下简称 SGD* ）则是每次使用一个训练样本 $x^{(i)}$ 和标签 $y^{(i)}$ 进行一次参数更新。

$$\theta = \theta - \eta \cdot \nabla_\theta J(\theta;x^{x(i)};y^{(i)})$$

BGD 对于大数据集来说执行了很多冗余的计算，因为在每一次参数更新前都要计算很多相似样本的梯度。SGD 通过一次执行一次更新解决了这种冗余。因此通常 SGD 的速度会非常快而且可以被用于在线学习。SGD 以高方差的特点进行连续参数更新，导致目标函数严重震荡，如图 1 所示。

![](https://upload.wikimedia.org/wikipedia/commons/f/f3/Stogra.png)
*图 1：SGD 震荡，来自 [Wikipedia](https://upload.wikimedia.org/wikipedia/commons/f/f3/Stogra.png)*

BGD 能够收敛到（局部）最优点，然而 SGD 的震荡特点导致其可以跳到新的潜在的可能更好的局部最优点。已经有研究显示当我们慢慢的降低学习率时，SGD 拥有和 BGD 一样的收敛性能，对于非凸和凸曲面几乎同样能够达到局部或者全局最优点。

代码片段如下，只是加了个循环和在每一个训练样本上计算梯度。注意依据 [这里](http://ruder.io/optimizing-gradient-descent/index.html#shufflingandcurriculumlearning) 的解释，我们在每次迭代的时候都打乱训练集。

```python
for i in range(nb_epochs):
    np.random.shuffle(data)
    for example in data:
        params_grad = evaluate_gradient(loss_function, example, params)
        params = params - learning_rate * params_grad
```

### Mini-batch gradient descent

Mini-batch gradient descent（ *译者注：以下简称 MBGD* ）则是在上面两种方法中采取了一个折中的办法：每次从训练集中取出 $n$ 个样本作为一个 mini-batch，以此来进行一次参数更新。

$$\theta = \theta - \eta \cdot \nabla_\theta J( \theta; x^{(i:i+n)}; y^{(i:i+n)})$$

这样做有两个好处：

- 减小参数更新的方差，这样可以有更稳定的收敛。
- 利用现在最先进的深度学习库对矩阵运算进行了高度优化的特点，这样可以使得计算 mini-batch 的梯度更高效。

通常来说 mini-batch 的大小为 50 到 256 之间，但是也会因为任务的差异而不同。MBGD 是训练神经网络时的常用方法，而且通常即使实际上使用的是 MBGD，也会使用 SGD 这个词来代替。注意：在本文接下来修改 SGD 时，为了简单起见我们会省略参数 $x^{(i:i+n)}; y^{(i:i+n)}$。

代码片段如下，我们每次使用 mini-batch 为 50 的样本集来进行迭代：

```python
for i in range(nb_epochs):
    np.random.shuffle(data)
    for batch in get_batches(data, batch_size=50):
        params_grad = evaluate_gradient(loss_function, batch, params)
        params = params - learning_rate * params_grad
```

## Challenges

标准的 MBGD 并不保证好的收敛，也提出了一下需要被解决的挑战：

- **选择一个好的学习率是非常困难的**。太小的学习率导致收敛非常缓慢，而太大的学习率则会阻碍收敛，导致损失函数在最优点附近震荡甚至发散。
- Learning rate schedules 试图在训练期间调整学习率即退火（annealing），根据先前定义好的一个规则来减小学习率，或者两次迭代之间目标函数的改变低于一个阈值的时候。然而这些规则和阈值也是需要在训练前定义好的，所以**也不能做到自适应数据的特点**。
- 另外，**相同的学习率被应用到所有参数更新中**。如果我们的数据比较稀疏，特征有非常多不同的频率，那么此时我们可能并不想要以相同的程度更新他们，反而是对更少出现的特征给予更大的更新。
- 对于神经网络来说，另一个最小化高度非凸误差函数的关键挑战是**避免陷入他们大量的次局部最优点（suboptimal）**。Dauphin 等人指出事实上困难来自于鞍点而不是局部最优点，即损失函数在该点的一个维度上是上坡（slopes up）（ *译者注：斜率为正* ），而在另一个维度上是下坡（slopes down）（ *译者注：斜率为负* ）。这些鞍点通常被一个具有相同误差的平面所包围，这使得对于 SGD 来说非常难于逃脱，因为在各个维度上梯度都趋近于 0 。

## Gradient descent optimization algorithms

接下来，我们将会概述一些在深度学习社区常用的算法，这些算法解决了我们前面提到的挑战。我们不会讨论实际上在高维数据集上不可行的算法，例如二阶方法中的 [牛顿法](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization)。

### Momentum

SGD 在遇到沟壑（ravines）会比较困难，即在一个维度上比另一个维度更陡峭的曲面，这些曲面通常包围着局部最优点。在这些场景中，SGD 震荡且缓慢的沿着沟壑的下坡方向朝着局部最优点前进，如图 2 所示。

![SGD without momentum](http://ruder.io/content/images/2015/12/without_momentum.gif)  
*图 2：不带动量的 SGD*

动量（Momentum）是一种在相关方向加速 SGD 的方法，并且能够减少震荡，如图 3 所示。

![SGD without momentum](http://ruder.io/content/images/2015/12/with_momentum.gif)  
*图 3：带动量的 SGD*

它在当前的更新向量中加入了先前一步的状态：

$$
\begin{aligned}
v_t &= \gamma v_{t-1} + \eta \nabla_\theta J( \theta) \\  
\theta &= \theta - v_t
\end{aligned}
$$

注意：一些实现可能改变了公式中的符号。动量项 $\gamma$ 通常设置为 0.9 或者相似的值。

本质上来说，当我们使用动量时，类似于我们把球推下山的过程。在球下山的过程中，球累积动量使其速度越来越快（直到达到其最终速度，如果有空气阻力的话，即 $\gamma \lt 1$）。相同的事情也发生在我们的参数更新中：对于梯度指向方向相同的维度动量项增大，对于梯度改变方向的维度动量项减小。最终，我们获得了更快的收敛并减少了震荡。

### Nesterov accelerated gradient

然而，一个球盲目的沿着斜坡下山，这不是我们希望看到的。我们希望有一个聪明的球，他知道将要去哪并可以在斜坡变成上坡前减速。Nesterov accelerated gradient（ *译者注：以下简称 NAG* ）就是这样一种给予我们的动量项预知能力的方法。我们知道我们使用动量项 $\gamma v_{t-1}$ 来更新 $\theta$。因此计算 $\theta-\gamma v_{t-1}$ 给了我们一个关于的参数下一个位置的估计（这里省略了梯度项），这是一个简单粗暴的想法。我们现在可以通过计算 $\theta$ 下一个位置而不是当前位置的梯度来实现「向前看」。

$$
\begin{aligned}
v_t &= \gamma v_{t-1} + \eta \nabla_\theta J( \theta - \gamma v_{t-1} ) \\  
\theta &= \theta - v_t
\end{aligned}
$$

我们仍然设置 $\gamma$ 为 0.9。动量法首先计算当前梯度（图 4 中的小蓝色向量）,然后在更新累积梯度（updated accumulated gradient）方向上大幅度的跳跃（图 4 中的大蓝色向量）。与此不同的是，NAG 首先在先前的累积梯度（previous accumulated gradient）方向上进行大幅度的跳跃（图 4 中的棕色向量），评估这个梯度并做一下修正（图 4 中的红色向量），这就构成一次完整的 NAG 更新（图 4 中的绿色向量）。这种预期更新防止我们进行的太快，也带来了更高的相应速度，这在一些任务中非常有效的提升了 RNN 的性能 [8]。

![Nesterov update](http://ruder.io/content/images/2016/09/nesterov_update_vector.png)  
*图 4：Nesterov 更新，来自 [G. Hinton's lecture 6c](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)*

可以在 [这里]() 查看对 NAG 的另一种直观解释，此外 Ilya Sutskever 在他的博士论文中也给出了详细解释 [9]。

现在我们已经能够依据误差函数的斜率来调整更新，并加快 SGD 的速度，此外我们也想根据每个参数的重要性来决定进行更大还是更小的更新。

### Adagrad

Adagrad 就是这样一种解决这个问题的基于梯度的优化算法：根据参数来调整学习率，对于不常见的参数给予更大的更新，而对于常见的给予更小的更新。因此，Adagrad 非常适用于稀疏数据。Dean 等人 [4] 发现 Adagrad 能够大幅提高 SGD 的鲁棒性，并在 Google 用其训练大规模神经网络，这其中就包括 [在 YouTube 中学习识别猫](http://www.wired.com/2012/06/google-x-neural-network/)。除此之外，Pennington 等人 [5] 使用 Adagrad 来训练 GloVe 词嵌入，因为罕见的词汇需要比常见词更大的更新。

前面我们在对所有参数 $\theta$ 更新时每个参数 $\theta_i$ 使用相同的学习率 $\eta$。Adagrad 在每个时间点 $t$ 对每个参数 $\theta_i$ 使用的学习率都不同，我们首先展现每个参数的更新，然后再向量化。简单起见，我们使用 $g_{t,i}$ 来表示目标函数关于参数 $\theta_i$ 在时间点 $t$ 时的梯度：

$$g_{t, i} = \nabla_\theta J( \theta_i )$$

SGD 在每个时间点 $t$ 对每个参数 $\theta_i$ 的更新变为：

$$\theta_{t+1,i} = \theta_{t,i} - \eta \cdot g_{t,i}$$

在这个更新规则里，Adagrad 在每个时间点 $t$ 对每个参数 $\theta_i$ 都会基于过去的梯度修改学习率 $\eta$。

$$\theta_{t+1, i} = \theta_{t, i} - \dfrac{\eta}{\sqrt{G_{t, ii} + \epsilon}} \cdot g_{t, i}$$

其中 $G_{t} \in \mathbb{R}^{d \times d}$ 是一个对角矩阵，对角元素 $G_t[i, i]$ 是参数 $\theta_i$ 从开始到时间点 $t$ 为止的梯度平方和，$\epsilon$ 是一个平滑项，用于防止分母为 0 ，通常为 $10^{-8}$ 左右。有趣的是，如果去掉开方操作，算法性能会大幅下降。

由于 $G_t$ 的对角元素是关于所有参数的过去的梯度的平方和，我们可以将上面的实现向量化，即使用点乘 $\odot$ ：

$$\theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{G_{t} + \epsilon}} \odot g_{t}$$

Adagrad 最大的一个优点是我们可以不用手动的调整学习率。大多数实现使用一个默认值 0.01 。

Adagrad 主要的缺点是分母中累积的平方和梯度：由于每一个新添加的项都是正的，导致累积和在训练期间不断增大。这反过来导致学习率不断减小，最终变成无限小，这时算法已经不能再继续学习新东西了。下面的这个算法就解决了这个问题。

### Adadelta

