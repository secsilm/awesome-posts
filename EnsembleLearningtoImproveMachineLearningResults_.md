# 使用集成学习提升机器学习算法性能
  
  
> 这篇文章是对 PythonWeekly 推荐的一篇讲集成模型的文章的翻译，原文为 [Ensemble Learning to Improve Machine Learning Results](https://blog.statsbot.co/ensemble-learning-d1dcd548e936 )，由 Vadim Smolyakov 于 2017 年 8 月 22 日发表在 Medium 上，Vadim Smolyakov 是一名 MIT 的研究生，对数据科学和机器学习充满热情。
  
*集成学习（Ensemble Learning）通过联合几个模型来帮助提高机器学习结果。与单一模型相比，这种方法可以很好地提升模型的预测性能。这也是为什么集成模型在很多著名机器学习比赛中被优先使用的原因，例如 Netflix 比赛，KDD 2009 和 Kaggle。*
  
集成方法是一种将几种机器学习技术组合成一个预测模型的元算法（meta-algorithm），以减小方差（bagging），偏差（boosting），或者改进预测（stacking）。
  
集成方法可以分为两类：
  
- 序列集成方法（sequential ensemble methods），基学习器（base learner）顺序生成。序列方法的基本动机是**利用基学习器之间的依赖关系**。算法可以通过提高被分错样本的权重来提高性能。
- 并行集成方法（parallel ensemble methods），基学习器并行生成。并行方法的基本动机是**利用基学习器之间的独立性**，因为可以通过平均来显著降低误差。
  
大多数集成方法使用一个基学习算法来产生多个同质基学习器（homogeneous base learners），即相同类型的学习器，这就是同质集成（homogeneous ensembles）。
  
也有一些方法使用的是异质学习器（heterogeneous learner），即不同类型的学习器，这就是异质集成（heterogeneous ensembles）。为了使集成方法能够比任何构成它的单独的方法更准确，基学习器必须尽可能的准确和多样。
  
## Bagging
  
  
Bagging 表示的是 **b**ootstrap **agg**regat**ing**。降低一个估计的方差的一个方法就是平均多个估计。例如，我们可以在一个数据集的不同子集上（有放回的随机选取）训练 <img src="https://latex.codecogs.com/gif.latex?M"/> 个不同的树然后计算结果的平均值： <p align="center"><img src="https://latex.codecogs.com/gif.latex?f(x)=&#x5C;frac{1}{M}&#x5C;Sigma_{m=1}^M f_m(x)"/></p> bagging 使用[自助抽样法](https://en.wikipedia.org/wiki/Bootstrapping_(statistics ))（bootstrapping）来为每个基学习器获得一个数据集的子集。对于如何聚合多个基学习器的结果，bagging 在分类任务中使用投票，而在回归任务重使用平均。
  
我们可以通过在 Iris 数据集上执行分类任务来学习 bagging。我们选择两种基学习器：决策树（decision tree）和 kNN 分类器。图一显示了基学习器在 Iris 上学习到的决策边界和他们 bagging 集成之后学习到的决策边界。
  
- 决策树准确率：0.63（+/- 0.02）
- kNN 准确率：0.70（+/- 0.02）
- bagging 树准确率：0.64（+/- 0.01）
- bagging kNN准确率：0.59（+/- 0.07）
  
![图一](https://i.imgur.com/XA3aooQ.png )
*图一*
  
决策树的边界与轴平行，而 <img src="https://latex.codecogs.com/gif.latex?k=1"/> 时的 kNN 算法与数据点紧密贴合。该集成方法使用了 10 个基学习器，训练子集由原训练数据和特征的 80% 构成。
  
决策树集成相对于 kNN 集成达到了较高的准确率。kNN 对训练样本的扰动不敏感，因此也被称为稳定学习器（stable learner）。
  
> 稳定学习器的集成不太有利，因为这样的集成并不会提升泛化性能。
  
图一也显示了集成大小是如何提高测试准确率的。基于交叉验证的结果，我们可以看到在大约 10 个基学习器前准确率一直在增加，随后趋于平缓，在 0.7 左右上下波动。因此，再增加超过 10 个基学习器不仅没有得到准确率的提升，反而增加了计算复杂度。
  
我们也可以看到 bagging 树集成的学习曲线。注意到训练数据的平均误差为 0.3 和 测试数据的 U 型误差曲线。训练和测试误差差距最小时发生在 Training set size in percent 为 80% 左右。
  
> 一种常用的集成算法是随机森林。
  
在随机森林算法中，每个树都是基于从原训练数据集中有放回抽样（即自助抽样法）得到的子集训练的。另外，也对特征进行自助抽样，而不是使用全部特征。
  
最终随机森林的偏差可能会轻微增大，但是由于平均了几个不相关的树的结果，降低了方差，导致最终模型的整体性能更好。
  
![](https://i.imgur.com/iXTf6Ck.png )
  
在[极限随机树（extremely randomized tree）](https://en.wikipedia.org/wiki/Random_forest#ExtraTrees )算法中，随机性更近了一步：分裂阈值是随机选取的。与寻找最具有判别性的阈值不同，极限随机树为每个候选特征选取一个阈值，并将这些阈值的最佳值作为最终的分割阈值。这通常会降低方差，偏差会稍微增大。
  