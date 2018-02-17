---
layout: page
permalink: /understanding-cnn/
---

<a name='vis'></a>

(this page is currently in draft form)
## 显示`ConvNets`学习的结果

对于神经网络中学习特征不可解释的经常性批评，出于回应，在现代文献中，已经开发出了几种理解和显示卷积网络的方法。在本节中，我们将简要介绍这些方法和相关工作。

### 可视化激活和第一层权重

**图层激活**. 最直接的可视化技术是在正向传播期间显示网络的激活。对于`ReLU`网络来说，激活通常开始看起来相对比较臃肿和密集，但随着训练的进行，激活通常变得更加稀疏和局部集中。使用可视化很容易观察到一个危险的陷阱是，对于许多不同的输入，某些些激活映射可能全为零，这表明过滤器**死亡**，这正是过高学习速率的症状。

<div class="fig figcenter fighighlight">
  <img src="/assets/cnnvis/act1.jpeg" width="49%">
  <img src="/assets/cnnvis/act2.jpeg" width="49%">
  <div class="figcaption">
    识别猫的第一个卷积激活（左）图形和已经训练过的`AlexNet`模型第五层（右）。每个框都显示一个对应于某个过滤器的激活图。请注意，激活是稀疏的（大多数值为零，可视化显示为黑色），大多本地集中。
  </div>
</div>

**Conv / FC过滤器.** 第二种常用策略是将权重可视化。这些通常在第一个直接关注原始像素数据的CONV层上最易解释，但也可以在网络中更深地显示滤波器权重。权重对于可视化很有用，因为更好训练的网络通常会显示漂亮而平滑的过滤器，而不会出现任何噪音模式。噪声意味着网络可能未经足够长时间训练，或者正则化强度低可能导致过度拟合。

<div class="fig figcenter fighighlight">
  <img src="/assets/cnnvis/filt1.jpeg" width="49%">
  <img src="/assets/cnnvis/filt2.jpeg" width="49%">
  <div class="figcaption">
    第一个CONV层上的典型外观滤波器（左）以及良好训练的AlexNet的第二个CONV层（右）。请注意，第一层权重非常平滑，表明融合网络很好。由于AlexNet包含两个独立的处理流，所以颜色/灰度特征是聚类的，这种体系结构的明显结果是一个流开发高频灰度特征和其他低频彩色特征。第二个CONV层的权重不是可以解释的，但很明显，他们仍然是平滑的，格式良好的，并且没有噪音模式。
  </div>
</div>

### 检索最大程度地激活神经元的图像

另一种可视化技术是采集大量图像数据集，通过网络馈送它们并跟踪哪些图像最大程度地激活某个神经元。然后，我们可以将图像可视化，以了解神经元在其感受野中寻找的内容。一个案例参见Rich Girshick等人[在丰富的特征层次结构中精确的对象检测和语义分割](http://arxiv.org/abs/1311.2524)。

<div class="fig figcenter fighighlight">
  <img src="/assets/cnnvis/pool5max.jpeg" width="100%">
  <div class="figcaption">
    最大限度地激活AlexNet的某些POOL5（第5个池层）神经元的图像。白色显示特定神经元的激活值和感受野。（尤其要注意的是，POOL5神经元是输入图像相对较大部分的函数！）可以看出，一些神经元对上半身，文本或镜面高光敏感。
  </div>
</div>

这种方法的一个问题是，ReLU神经元本身不一定具有任何语义含义。相反，将多个ReLU神经元想象成图像块中表示的某些空间的基本向量是更合适的。换句话说，可视化显示沿着与过滤器权重相对应的（任意）轴的表示云表面边缘的补丁。这也可以通过ConvNet中的神经元在输入空间上线性运行的事实来看到，因此该空间的任意旋转是无操作的。Szegedy等人在[神经网络的Intriguing特性](http://arxiv.org/abs/1312.6199)中进一步论证了这一点，他们沿着表示空间中的任意方向执行类似的可视化。

### 使用`t-SNE`嵌入代码

ConvNets可以被解释为逐渐将图像转换为一种表达，在其中类可以通过线性分类器分离。通过将图像嵌入到两维中，我们可以粗略地了解这个空间的拓扑结构，以使它们的低维表示具有与它们的高维表示大致相等的距离。有许多嵌入方法是直接将高维矢量嵌入低维空间，同时保留点的成对距离。其中，[t-SNE](http://lvdmaaten.github.io/tsne/) 是产生视觉上令人满意的结果的最着名的方法之一。

为了产生嵌入，我们可以采用一组图像，并使用ConvNet提取CNN代码（例如，在AlexNet中，即在分类器之前的4096维矢量中，包括ReLU非线性至关重要）。然后，我们可以将这些插入到t-SNE中，并为每个图像获取二维矢量。相应的图像可以在网格中可视化：

<div class="fig figcenter fighighlight">
  <img src="/assets/cnnvis/tsne.jpeg" width="100%">
  <div class="figcaption">
    基于CNN代码的一组图像的t-SNE嵌入。彼此相邻的图像在CNN表示空间也很接近，这意味着CNN"看"它们非常相似。请注意，相似性通常是基于类和语义的，而不是基于像素和颜色。有关如何生成此可视化的更多详细信息，相关代码以及不同尺度下的更多相关可视化请参考<a href="http://cs.stanford.edu/people/karpathy/cnnembed/">`CNN`代码的`t-SNE`可视化</a>.
  </div>
</div>

### 遮挡图像的一部分

假设ConvNet将图像分类为狗。我们怎样才能确定它实际上是在图像中对狗进行拾取，而不是从背景或其他杂项对象中提取一些背景线索？一种调查图像分类预测究竟来自图像的哪一部分的方法是将感兴趣的类别（例如，狗类）的概率绘制为遮挡物对象的位置函数。也就是说，我们遍历图像的各个区域，将图像的一个片段设置为零，然后查看该类别的概率。我们可以将概率可视化为二维热图。这种方法应用于Matthew Zeiler的[可视化和理解卷积网络](http://arxiv.org/abs/1311.2901):

<div class="fig figcenter fighighlight">
  <img src="/assets/cnnvis/occlude.jpeg" width="100%">
  <div class="figcaption">
    三个输入图像（顶部）。请注意，遮盖区域显示为灰色。当我们在图像上滑动遮挡物时，我们记录了正确类的概率，然后将其可视化为热图（在每个图像下面显示）。例如，在最左侧的图像中，我们看到，当封堵器覆盖狗的脸部时，分类为博美犬的可能性直线下降，这提示我们，即狗的脸部承担了类评分的主要责任。相反，遮盖图像的其他部分只有轻微影响。
  </div>
</div>

### 可视化数据梯度和类似部分

**数据梯度**.

[深层卷积网络：可视化图像分类模型和显着性映射](http://arxiv.org/abs/1312.6034)

**DeconvNet**.

[可视化和理解卷积网络](http://arxiv.org/abs/1311.2901)

**引导式反向传播**.

[争取简单：所有卷积网络](http://arxiv.org/abs/1412.6806)

### 基于CNN编码重建原始图像

[通过反转理解深度图像表示](http://arxiv.org/abs/1412.0035)

### 多少空间信息被保存？

[ConvNets能进行相应学习吗？](http://papers.nips.cc/paper/5420-do-convnets-learn-correspondence.pdf) (tldr: yes)

### 使用图像属性函数绘制性能

[ImageNet大规模视觉识别挑战](http://arxiv.org/abs/1409.0575)

## 愚弄ConvNets

[解释和利用对抗性例子](http://arxiv.org/abs/1412.6572)

## 将ConvNets与人类标注相比较

[我从ImageNet上的ConvNet竞赛中学到了什么](http://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/)
