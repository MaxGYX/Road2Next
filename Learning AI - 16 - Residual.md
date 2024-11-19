### 误差 & 残差
-  **误差Error**： “**模型预测值**”与“**真值**”之间的差异。这个差异可以是单个数据点的误差，也可以是整个数据集的平均误差。
-  **残差Residual**：“**实际观测值**”与“**模型预测值**”之间的差异，它反映了模型在特定数据点上的拟合程度。

对于单个数据点来说，这个点的真值就是观测值，所以误差与残差是一致的。

而对于数据集来说，误差通常使用平均误差。比如回归分析中，通常使用均方误差（MSE）或均方根误差（RMSE）。

### 残差网络 ResNet
ResNet论文网址：https://arxiv.org/abs/1512.03385

残差神经网络的主要贡献是发现了“**退化现象（Degradation）**”，并针对退化现象发明了 “**快捷连接（Shortcut connection）**”，极大的消除了深度过大的神经网络训练困难问题。
-  **Degradation**

   ResNet团队发现随着网络层不断的加深，模型的准确率先是不断的提高，达到最大值后，随着网络深度的继续增加，模型准确率毫无征兆的出现大幅度的降低。这个现象与“越深的网络准确率越高”的信念显然是冲突的。ResNet团队把这一现象称为“退化（Degradation）”。

   退化现象被归因为**深层**神经网络难以实现“恒等变换（y=x）”，深度学习的关键特征在于**网络层数更深**、**非线性转换**（激活）、**自动的特征提取和特征转换**，其中，非线性转换是关键目标，它将数据映射到高维空间以便于更好的完成“数据分类”。随着网络深度的不断增大，所引入的激活函数也越来越多，数据被映射到更加离散的空间，此时已经难以让数据回到原点（恒等变换）。
   
-  **Shortcut connection**

   通过增加一个线性的分支“桥梁”，把数据原封不动的送到某一层，就可以很容易实现“恒等变换”的能力。
<img width="675" alt="image" src="https://github.com/user-attachments/assets/a834adaa-a515-4a28-8a31-ab46d3178339">

   比如，我们把整个映射看成100%，则前面四层网络实现了98%的映射关系（直接通过一个分支送到FC层之前），而残余的映射由紫色层完成，Residual的含义就是让每一个残差块，只关注残余映射的一小部分。

   当我们并不会知道哪几层就能达到很好的效果，然后在它们的后面接一个跳连接，可以一开始便在两个层或者三个层之间添加跳连接，形成残差块，每个残差块只关注当前的残余映射，而不会关注前面已经实现的底层映射。



**这篇文章说的真清楚：**

https://zhuanlan.zhihu.com/p/101332297

一个例子
```python
class ResidualBlock(nn.Module):
    def __init__(self, num_hiddens):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 将输入数据x与经过2层卷积之后的数据进行连接，再经过relu激活函数后做为输出数据
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + residual)
```
