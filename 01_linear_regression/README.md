### 线性回归

**生成数据集**：

```python
# set input feature number 
num_inputs = 2
# set example number
num_examples = 1000

# set true weight and bias in order to generate corresponded label
true_w = [2, -3.4]
true_b = 4.2

# 标准正态分布
features = torch.randn(num_examples, num_inputs,
                      dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# numpy类型的数据转成torch的tensor
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)
```

**torch的tensor转成numpy类型**：

```python
features[:, 1].numpy()
```

**转成long类型**：

```python
torch.LongTensor([0, 1, 2, 3])
```

**根据index选择数据，0表示按行选，1表示按列选**

```python
features.index_select(0, j)
```



**初始化模型参数，requires_grad_函数**

```python
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

# 当前量是否需要在计算中保留对应的梯度信息
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
```

使用backward()函数反向传播计算tensor的梯度时，并不计算所有tensor的梯度，而是只计算满足这几个条件的tensor的梯度：1.类型为叶子节点、2.requires_grad=True、3.依赖该tensor的所有tensor的requires_grad=True。

所有的tensor都有.requires_grad属性,可以设置这个属性

```python
x = tensor.ones(2,4,requires_grad=True)
```

如果想改变这个属性，就调用tensor.requires_grad_()方法：

```python
x.requires_grad_(False)
```

参考：https://zhuanlan.zhihu.com/p/85506092



**.view的用法**，比如a.view(1,6)，a.view(b.size())

把原先tensor中的数据按照行优先的顺序排成一个一维的数据（这里应该是因为要求地址是连续存储的），然后按照参数组合成其他维度的tensor。

a.view(1, -1)，-1代表的维度由a的维度和1推断出来，比如a tensor的数据个数是6，那么-1就代表6。

参考：https://blog.csdn.net/york1996/article/details/81949843



**sgd梯度下降**

```py
def sgd(params, lr, batch_size): 
    for param in params:
        param.data -= lr * param.grad / batch_size # ues .data to operate param without gradient track
```

param.data和 param 共享同一块数据，和param的计算历史无关，且其requires_grad = False

如果只是使用参数本身，那么在这个sgd参数更新其实可能也会变成网络结构的一部分，这时候网络就不是线性回归了，而使用.data可能就是隔断了梯度传导，让这里只是一个参数数值更新。

在损失函数进行求梯度时，为了保证参数值正常更新的同时又不影响梯度的计算，即使用param.data可以将更新的参数独立于计算图，阻断梯度的传播，当训练结束就可以得到最终的模型参数值。

参考：https://www.boyuai.com/elites/course/cZu18YmweLv10OeV/video/Rosi4tliobRSKaSVcsRx_

https://zhuanlan.zhihu.com/p/38475183



**训练**

```pyt
# super parameters init
lr = 0.03
num_epochs = 5

net = linreg
loss = squared_loss

# training
for epoch in range(num_epochs):  # training repeats num_epochs times
    # in each epoch, all the samples in dataset will be used once
    
    # X is the feature and y is the label of a batch sample
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  
        # calculate the gradient of batch sample loss 
        l.backward()  
        # using small batch random gradient descent to iter model parameters
        sgd([w, b], lr, batch_size)  
        # reset parameter gradient
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
```



**简洁实现**

```py
import torch
from torch import nn
import numpy as np
torch.manual_seed(1)

print(torch.__version__)
torch.set_default_tensor_type('torch.FloatTensor')

# 读取数据集
import torch.utils.data as Data

batch_size = 10

# combine featues and labels of dataset
dataset = Data.TensorDataset(features, labels)

# put dataset into DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,            # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # whether shuffle the data or not
    num_workers=2,              # read data in multithreading
)

# 定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()      # call father function to init 
        self.linear = nn.Linear(n_feature, 1)  # function prototype: `torch.nn.Linear(in_features, out_features, bias=True)`

    def forward(self, x):
        y = self.linear(x)
        return y
    
net = LinearNet(num_inputs)
```

**定义多层神经网络的方法**

```py
# ways to init a multilayer network
# method one
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # other layers can be added here
    )

# method two
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# method three
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))
```



```py
# 初始化模型参数
from torch.nn import init

init.normal_(net[0].weight, mean=0.0, std=0.01)
init.constant_(net[0].bias, val=0.0)  # or you can use `net[0].bias.data.fill_(0)` to modify it directly

# 定义损失函数
loss = nn.MSELoss()    # nn built-in squared loss function
                       # function prototype: `torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')`

# 定义优化函数
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.03)   # built-in random gradient descent function
print(optimizer)  # function prototype: `torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)`

# 训练
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # reset gradient, equal to net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
    
# result comparision
dense = net[0]
print(true_w, dense.weight.data)
print(true_b, dense.bias.data)
```



torch.mm 和 torch.mul 的区别？ torch.mm是矩阵相乘，torch.mul是按元素相乘

torch.manual_seed(1)的作用？ 设置随机种子，使实验结果可以复现

optimizer.zero_grad()的作用？使梯度置零，防止不同batch得到的梯度累加