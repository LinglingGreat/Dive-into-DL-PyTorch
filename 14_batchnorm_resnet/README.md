### 批量归一化（BatchNormalization）

#### 对输入的标准化（浅层模型）

处理后的任意一个特征在数据集中所有样本上的均值为0、标准差为1。
标准化处理输入数据使各个特征的分布相近

#### 批量归一化（深度模型）

利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。



### 1.对全连接层做批量归一化

位置：全连接层中的仿射变换和激活函数之间。

### 2.对卷积层做批量归⼀化

位置：卷积计算之后、应⽤激活函数之前。
如果卷积计算输出多个通道，我们需要对这些通道的输出分别做批量归一化，且每个通道都拥有独立的拉伸和偏移参数。 计算：对单通道，batchsize=m,卷积计算输出=pxq 对该通道中m×p×q个元素同时做批量归一化,使用相同的均值和方差。

### 3.预测时的批量归⼀化

训练：以batch为单位,对每个batch计算均值和方差。
预测：用移动平均估算整个训练数据集的样本均值和方差。



###残差网络（ResNet）

深度学习的问题：深度CNN网络达到一定深度后再一味地增加层数并不能带来进一步地分类性能提高，反而会招致网络收敛变得更慢，准确率也变得更差。

### 残差块（Residual Block）

恒等映射：
左边：f(x)=x
右边：f(x)-x=0 （易于捕捉恒等映射的细微波动）

![Image Name](https://cdn.kesci.com/upload/image/q5l8lhnot4.png?imageView2/0/w/600/h/600)

### ResNet模型

卷积(64,7x7,3)
批量一体化
最大池化(3x3,2)

残差块x4 (通过步幅为2的残差块在每个模块之间减小高和宽)

全局平均池化

全连接



###稠密连接网络（DenseNet）

![Image Name](https://cdn.kesci.com/upload/image/q5l8mi78yz.png?imageView2/0/w/600/h/600)

### 主要构建模块：

稠密块（dense block）： 定义了输入和输出是如何连结的。
过渡层（transition layer）：用来控制通道数，使之不过大。

### 过渡层

1×1卷积层：来减小通道数
步幅为2的平均池化层：减半高和宽