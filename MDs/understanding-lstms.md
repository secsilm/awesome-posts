> 译者注：
>
> - 本文原文为 Christopher Olah 于 2015 年发表在自己[博客](http://colah.github.io/)上的经典文章：[Understanding LSTM Networks -- colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)。
> - 没有翻译原文中的 Acknowledgments 部分，此部分为致谢，私以为无关。
> - 文中括号或者引用块中的 *斜体字* 为对应的英文原文或者我自己注释的话（会标明 *译者注*），否则为原文中本来就有的话。
> - 本人水平有限，如有错误欢迎指出。

## Recurrent Neural Networks

人们不会每一秒都从头开始思考。当你阅读这篇文章的时候，你根据前面的单词来理解后面的单词。你不会扔掉他们然后重新开始思考。你的思考具有持续性。

传统的神经网络做不到这一点，而且这似乎是一个主要缺点。例如，想象一下你想要对一个电影中每一帧所发生的事件类型进行分类，而传统神经网络却不能够利用先前的事情来推理后来的事情。

Recurrent Neural Networks（*译者注：以下简称 RNN*）解决了这个问题，他们在网络内部有循环，可以让信息具有持续性。

<img src="https://i.imgur.com/A8appFv.png" height=180 alt="Recurrent Neural Networks have loops.")>
<center><font color="gray">RNN 中有循环</font></center>

上图是神经网络的一部分，其中 $A$ 观察输入 $x_t$ 并输出一个值 $h_t$，循环使得信息可以从网络中的一步传递到下一步。

这些循环使得 RNN 看起来有些神秘。然而如果你再进一步想想，就会发现他们与普通的神经网络并不是完全不同。一个 RNN 可以被想成是对同一个网络的多次复制，每次都把信息传递给下一个。考虑一下如果我们把循环展开（*unroll*）会发生什么：

![rnn-unrolled](https://i.imgur.com/0Wik9NF.png)
<center><font color="gray">一个展开的 RNN</font></center>

这种链状性质表明 RNN 与序列（*sequences*）和列表（*lists*）密切相关，在处理这种数据时他们是很自然的神经网络架构。

而且他们也确实在被使用！过去几年中，RNN 被应用于一系列的任务并取得了令人难以置信的成功：语音识别，语言模型，翻译，看图说话（*image captioning*）等等。你可以阅读 Andrej Karpathy 的精彩博文 —— [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) 来了解通过 RNN 可以取得的惊人成绩。他们真的很棒。

这些成功的基础是使用 LSTMs，一种非常特别地 RNN，在许多任务中要比标准版本好。几乎所有基于 RNN 的令人激动的成绩都是通过他们获得的。这篇文章讨论的就是这些 LSTMs。

## The Problem of Long-Term Dependencies

RNNs 其中一个有吸引力的地方是能够将以前的信息和现在的任务联系起来，例如使用视频中的前几帧信息可能会对当前帧的理解有帮助。如果 RNNs 能够做到这些，那么他们就是非常有用的。但是他们能吗？这不一定。

有时，我们可能只需要最近的信息来完成当前任务。例如，考虑一个试图基于前面的词来预测下一个词的语言模型，如果我们试图预测 “the clouds are in the *sky*” 这句话中的最后一个词，那么我们就不需要更多的信息，很明显下一个词就是 sky。在这种情况下，相关信息和需要的地方（*the place that it's needed*）之间的差距很小，那么这时候 RNNs 就可以学习到使用过去的信息。（*译者注：也就是短期依赖*）

<img src="https://i.imgur.com/HAvvUQV.png" height=180)>

但是也有其他情况是我们需要更多信息的。考虑我们需要预测 “I grew up in France... I speak fluent *French*” 这句话中的最后一个词。最近的信息表明这个词应该是一个语言的名字，但是如果我们想要知道哪个语言，那么我们需要结合更前面的 France 这个背景。这时相关信息和需要的点（*the point where it is needed*）之间的差距就会变得非常大。

然而不幸的是，随着这个差距的增大，RNNs 越来越难以学习使用以前的信息。

<img src="https://i.imgur.com/Whfo6UB.png" height=180)>

理论上来说，RNNs 完全可以处理这种“长期依赖”（*long-term dependencies*）。一个人可以很仔细的选择参数来解决这种形式的小问题（*toy problems*）。不过实际上，RNNs 似乎并不能学习到这种长期依赖。[Hochreiter (1991) [German]](http://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf) 和 [Bengio, et al. (1994)](http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf) 曾经深入探讨了这个问题，发现了一些相当根本的原因。

幸运的是，LSTMs 并没有这个问题！

## LSTM Networks

长短期记忆网络 —— 通常简称为“LSTMs” —— 是一种特别的 RNN，能够学习到长期依赖。LSTMs 由 [Hochreiter & Schmidhuber (1997)](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) 提出，其后有很多人都对其进行了改善和推广[^1]。LSTMs 在许多任务上效果都非常好，现在也被广泛使用。

LSTMs 就是被设计用来避免长期依赖问题的。记住长时间的信息实际上是他们的默认行为，而不需要可以去这样做！

所有的 RNN 都是在将一个神经网络的模块重复好多次并链式连接起来。在标准的 RNNs 中，这个被重复的模块是一个非常简单的结构，例如一个单层的 tanh 层。

<img src="https://i.imgur.com/KfFor4n.png" height=180)>
<center><font color="gray">标准 RNN 中的重复模块都含有一层</font></center>

LSTMs 也有这样一个链式结构，但是那个重复模块的结构是不一样的。与仅有一个单层神经网络不同的是，LSTMs 有 4 层，以一种非常特殊的方式连接起来。

<img src="https://i.imgur.com/UydD8qN.png" height=180)>
<center><font color="gray">LSTMs 中的重复模块含有 4 层</font></center>

不要担心这里面的细节。稍后我们将会逐步深入这个 LSTMs 图。现在，我们先来熟悉下我们将要使用的符号：

![](https://i.imgur.com/CbUDJll.png)

上图中，每一条线都表示一个向量从一个输出节点传到其他节点作为输入。粉色圆圈表示的是 pointwise 操作，例如向量加法，而黄色举行表示的是可学习的神经网络层。线合在一起表示连接（*concatenation*），线分叉表示其内容被复制成多份并且这些复制品流向不同的方向。

## The Core Idea Behind LSTMs

LSTMs 的核心是单元状态（*cell state*），就是顶部那条水平贯穿整个图的线。

单元状态就像是一个传送带，直接穿过整个链式结构，与其他部分仅有一些次要的线性交互，可以非常容易的传送信息而保持其不变。

<img src="https://i.imgur.com/n7yh66F.png" height=180)>

LSTMs 可以给单元状态移除或者增加信息，由一个称为门（*gate*）的结构来控制。

门是一种让信息选择性通过的方式，由一个 sigmoid 层和一个 pointwise 乘法操作组成。

<img src="https://i.imgur.com/1yDnZiD.png" height=180)>

sigmoid 层输出一个介于 0 和 1 之间的数字，表示每一个组件有多少信息可以穿过。0 意味着不让任何信息穿过，1 则意味着让所有信息穿过。

一个 LSTM 有 3 个这样的门，来保护和空值单元状态。

## Step-by-Step LSTM Walk Through

LSTM 的第一步就是决定我们要从单元状态中扔掉什么信息，这由一个叫失忆门（*forget gate layer*）的 sigmoid 层来控制。失忆门的输入为 $h_{t-1}$ 和 $x_t$，然后为单元状态 $C_{t-1}$ 中的每个数字输出一个 0 和 1 之间的数，1 表示完全保留信息，而 0 表示完全丢去信息。

让我们回到语言模型，我们想要基于所有前面的词来预测下一个词。在这样一个问题中，单元状态可能包括了当前主体（*subject*）的性别，因此可以使用正确的代词。当我们看到一个新的主体时，我们想要忘记旧主体的性别。

![](https://i.imgur.com/e8fiEJg.png)

下一步就是我们要决定要在单元状态中存入什么信息。这包括两部分。首先，一个叫做输入门（*input gate layer*）的 sigmoid 层决定我们要更新哪些值。然后一个 tanh 层创建一个新的候选向量 $\tilde{C}_t$，这个值会加到单元状态中。下一步我们将会利用这两个向量来更新单元状态。

在语言模型的例子中，我们想要用新主体的性别替换掉旧主体的性别，并加到单元状态中。

![LSTM3-focus-i](https://i.imgur.com/8JE0JQW.png)

现在可以把旧的单元状态 $\tilde{C}_{t-1}$ 更新为新的单元状态 $\tilde{C}_t$ 了，前面的步骤已经决定了要做什么，现在我们只需要真正的去做就行了。

我们要乘以 $f_t$，也就是之前在失忆门我们决定的要忘记的东西。然后我们加上 $i_t*\tilde{C}_t$，这是新的候选值，乘上我们决定的要为每个单元状态值更新多少。

在语言模型的例子中，这里实际做的就是丢弃旧主体的性别信息，加进新主体的性别信息，就像我们之前要做的那样。

![](https://i.imgur.com/1FXS5pB.png)

最后，我们要决定输出什么。这个输出将会是基于我们的单元状态的，但是是一个过滤版本（*filtered version*）。首先，我们运行一个 sigmoid 层来决定单元状态的哪些部分会被输出。然后，我们把单元状态输入给一个 tanh 层（把值映射到 $[-1,1]$ 区间内），并乘上 sigmoid 层的输出，然后这就是我们的输出。

在语言模型的例子中，由于仅仅有一个主体，以防接下来会发生什么事，LSTM 可能会输出与一个动词相关的信息。例如，可能会输出表示这个主体是单数还是复数的信息，以便我们如果知道接下来要发生什么，我们应该使用动词的什么形式。

![LSTM3-focus-o](https://i.imgur.com/lLxeyBy.png)

## Variants on Long Short Term Memory

我目前描述的是非常普通的 LSTM，但不是所有的 LSTMs 都和上面的一样。事实上，几乎所有与 LSTMs 相关的论文都会使用一个稍微不同的版本。这些区别是很小的，但是其中一些区别值得我们注意。

其中一个比较流行的 LSTM 变体由 [Gers & Schmidhuber (2000)](ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf) 引入，增加了一个「猫眼连接」（*peephole connections*），这意味着我们让门可以看到单元状态。

![LSTM3-var-peepholes](https://i.imgur.com/APS3oDp.png)

上图给所有的门都增加了「猫眼」，但是一些论文只会加一些。

另一个变体是使用耦合的（*coupled*）失忆门和输入门。与独立的决定我们要丢弃和增加哪些信息不同的是，我们一起做这两个决定。只有当我们输入一些新信息的时候才会丢弃一些信息，只有当我们丢弃一些旧信息的时候才会输入新信息。

![LSTM3-var-tied](https://i.imgur.com/gdm03LC.png)

稍微好点的（*dramatic*）LSTM 变体是 Gated Recurrent Unit，简称 GRU，由 [Cho, et al. (2014)](http://arxiv.org/pdf/1406.1078v3.pdf) 提出。GRU 将失忆门和输入门组合成一个更新门（*update gate*）。同时也合并了单元状态和隐藏状态，也做了一些其他改变。最终的模型比标准的 LSTM 模型更简单，而且越来越受欢迎。

![LSTM3-var-GRU](https://i.imgur.com/UEiY58u.png)

这只是一些最显著的 LSTM 变体，还有很多其他的，例如 [Yao, et al. (2015)](http://arxiv.org/pdf/1508.03790v2.pdf) 提出的 Depth Gated RNNs。也有一些方法用完全不同的方式来处理长期依赖的问题，比如 [Koutnik, et al. (2014)](http://arxiv.org/pdf/1402.3511v1.pdf) 提出的 Clockwork RNNs。

哪个变体是最好的？这些差别影响大吗？[Greff, et al. (2015)](http://arxiv.org/pdf/1503.04069.pdf) 对流行的变体做了一个很好的比较，发现他们都是一样的。[Jozefowicz, et al. (2015)](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf) 测试了超过 10000 种 RNN 架构，发现其中一些在某些具体任务上的表现优于 LSTMs。

## Conclusion

前面我提到人们使用 RNNs 取得了显著成果，基本上都是用的 LSTMs。他们在大多数任务上都表现的很好。

写下一堆有关 LSTMs 的方程，这让 LSTMs 看起来很吓人。希望这篇文章一步一步的走进 LSTMs 可以让你更好的理解他们。

在可以用 RNNs 完成的任务上使用 LSTMs 是前进的一大步。很自然我们会问：还有另外一个一大步吗？研究者中的一个普遍观点是“是的！有下一步而且这个下一步是注意力机制（*attention*）”。这个想法是让 RNN 的每一步都从更多的信息中挑选信息。例如，如果你用一个 RNN 来为一个图片加一些注释，每个输出的词都可能对应的是图片中的一部分。[Xu, et al. (2015)](http://arxiv.org/pdf/1502.03044v2.pdf) 就是这样做的，如果你想要继续探索注意力的话，那么这是一个很好地起点。使用注意力已经有了一些很好地结果，而且似乎还有更多的结果出来。注意力机制并不是 RNN 研究中唯一令人兴奋的部分，例如 [Kalchbrenner, et al. (2015)](http://arxiv.org/pdf/1507.01526v1.pdf) 提出的 Grid LSTMs 似乎也非常有前途。在生成模型中使用 RNNs 看起来也非常有趣，例如 [Gregor, et al. (2015)](http://arxiv.org/pdf/1502.04623.pdf)，[Chung, et al. (2015)](http://arxiv.org/pdf/1506.02216v3.pdf) 和 [Bayer & Osendorfer (2015)](http://arxiv.org/pdf/1411.7610v3.pdf)。过去几年对于 RNNs 来说是一个激动人心的时刻，接下来几年只会更是如此！

[^1]: 除了原作者，许多人都对现在的 LSTM 做出了贡献。 一个不完整名单：Felix Gers，Fred Cummins，Santiago Fernandez，Justin Bayer，Daan Wierstra，Julian Togelius，Faustino Gomez，Matteo Gagliolo，和 [Alex Graves](https://scholar.google.com/citations?user=DaFHynwAAAAJ&hl=en)。
