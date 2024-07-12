选择用**PyTorch**来实现后面的项目，开一篇来记录学习中遇到的问题

主要参考 https://pytorch.zhangxiann.com/ 的内容来学习，是我找到的比较适合自己的文章，有些看不懂的先跳过了，捡着对自己理解有帮助的内容来学习并做一些笔记便于回看加深理解。

## Why PyTorch
-  PyTorch是FAIR（Facebook AI Research）发布的深度学习框架。
-  PyTorch是在Torch基础上使用Python语言重新打造的（所以叫PyTorch），而Torch是使用Lua语言的机器学习框架，使用成本高。
-  PyTorch和TensorFlow的选择中，重点考虑的是PyTorch的资料更多，对于初学者更友好一些。

### PyTorch的主要Module
**数据**、**模型**、**损失函数**、**优化器**、**迭代训练**
  - “**数据**”指原始的数据文件和通过Dataset/DataLoader载入到内存中并进行预处理后的Tensor结构数据。
  - “**模型**”指各种网络模型（nn）以及对应的功能。
  - “**损失函数**”，“**优化器**”，“**迭代训练**”指通过训练数据迭代训练模型参数的判断依据和过程。        
   <img width="346" alt="image" src="https://github.com/MaxGYX/Road2Next/assets/158791943/77a7ce6c-b2f3-4164-bd13-b6b33b01e40b">

### 理解Tensor
Tensor张量，可以理解成一个**多维数组**，PyTorch中大多数操作都围绕着Tensor进行
<img width="764" alt="image" src="https://github.com/MaxGYX/Road2Next/assets/158791943/dfc3dbb3-15da-40a8-b34f-1071bed09661">

### 从数据集导入数据
<img width="749" alt="image" src="https://github.com/MaxGYX/Road2Next/assets/158791943/782812dd-aa93-48d1-9a17-a8dab91d34b8">

```python
# 载入数据集并进行处理

# 通过transforms.Compose()定义对数据的处理序列，包括：转换成Tensor结构，并进行归一化处理（Norm）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 1.MNIST手写数字数据集，包含60,000张训练图片，10,000张测试图片，数字包括0～9（共10类）。
#   图片都经过了处理，大小28×28的灰度图像（即只有一个channel，因此每个样本数据都是大小为784×1的矩阵)。
# 2.通过 torchvision.datasets.MNIST()载入MNIST数据（训练集/测试集，通过train参数指定）
#   存储在root指定目录下，没有数据则下载。
#   两个dataset都通过transform参数指定的方法进行预处理。
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 1.通过torch.utils.data.DataLoader()从DataSets加载数据，每批读取数据64个（batch_size参数），训练集数据打乱顺序（shuffle参数）
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
```

### 建立网络模型
建立一个包括2个卷积层和3个全连接层的网络。

```python
# 定义一个CNN网络（从nn.Module继承），需要自己实现两个方法
#   __init__：定义网络模型的层次结构
#   forward：定义前向计算的过程，在Module的训练过程中自动被调用
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        #网络中包括2个卷积层
        #   conv1： in_channels,输入维度1，灰度图只有一个channel，即输入就是1个28x28的tensor结构
        #           out_channels,输出维度32，即采用32个卷积核对输入进行卷积计算，输出会产生32个卷积的结果。
        #           kernel_size=3, 卷积核Filter尺寸3x3
        #           padding=1，对卷积后的结果外围补1圈数据
        #           卷积后的数据尺寸：
        #               h = (H-F+2*P)/S + 1 = (28-3+2*1)/1  + 1 = 28
        #               w = (W-F+2*P)/S + 1 = (28-3+2*1)/1  + 1 = 28
        #               即通过设置padding=1将卷积后的数据尺寸保持在28*28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        #   conv2： in_channels,输入维度32，即第一层卷积后的32个结果
        #           out_channels,输出维度64，即采用2个卷积核对输入进行卷积计算，输出会产生64个卷积的结果。
        #           kernel_size=3, 卷积核Filter尺寸3x3
        #           padding=1，对卷积后的结果外围补1圈数据
        #           卷积后的数据尺寸：
        #               h = (H-F+2*P)/S + 1 = (28-3+2*1)/1  + 1 = 28
        #               w = (W-F+2*P)/S + 1 = (28-3+2*1)/1  + 1 = 28
        #               即通过设置padding=1将第二层卷积后的数据尺寸继续保持在28*28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        #网络中包含1个池化层
        #   pool：采用最大池化策略 nn.MaxPool2d (2维数据的最大池化）
        #           kernel_size=2, 对2x2的区域进行池化
        #           stride=2，每次步长=2
        #           padding=0，结果不进行padding
        #           pooling前每个数据尺寸28x28，经过1次pooling后，尺寸变为14x14（kernel2x2，所以H/W都缩小一半）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #网络中包含3个线性层（全连接层）
        #   fc1: in_features：64x7x7，每个28x28数据经过2次pooling, 变成7x7, 第二层卷积输出维度64.
        #        out_features: 128, 将输入归类到128个输出结果上（即这个线性层有 128个节点）
        #   fc2: 将上一层128个结果作为输入，归类到64个输出结果上。
        #   fc3: 将上一层64个结果作为输入，归类到10个输出结果上。（我们最终需要的识别0～9手写字体共10类结果）
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # 先进行第1层卷积，结果经过relu非线性激活函数后，进行一次最大池化
        x = self.pool(torch.relu(self.conv1(x)))
        # 再进行第2层卷积，结果经过relu非线性激活函数后，进行一次最大池化
        x = self.pool(torch.relu(self.conv2(x)))
        # x.view()是将tensor结构数据进行reshape(改变尺寸)
        #   -1,表示不确认有多少行,由函数自己来计算
        #   64x7x7, 表示reshape后每一行数据有64x7x7列
        #   实际上经过这个函数后，每个输入(28x28)多次卷积后的数据会排列成一个64x7x7=3136的列，相当于把tensor展平
        x = x.view(-1, 64 * 7 * 7)
        # 进行第1次全连接，结果经过relu非线性激活函数
        x = torch.relu(self.fc1(x))
        # 进行第2次全连接，结果经过relu非线性激活函数
        x = torch.relu(self.fc2(x))
        # 进行第3次全连接
        x = self.fc3(x)
        return x
```

### 训练Training

```python
def train(model, train_loader, criterion, optimizer, epochs=5, save_interval=1):
    losses = []  # 记录每个epoch的损失值
    for epoch in range(epochs):
        # 迭代epochs轮数据
        running_loss = 0.0
        for inputs, labels in train_loader:
            # 迭代batch数据，每次循环处理一个batch的数据集
            inputs, labels = inputs.to(device), labels.to(device)
            # 重置optimizer里的梯度变量
            optimizer.zero_grad()
            # 执行forward计算，即根据当前模型参数计算输入对应的输出，其中会调用model.forward()
            outputs = model(inputs)
            # 计算loss
            loss = criterion(outputs, labels)
            # 计算loss的各参数对应的梯度；
            loss.backward()
            # 更新模型参数，用于下一次前向计算
            optimizer.step()
            # 累计loss
            running_loss += loss.item()

        # 计算并记录每个epoch的平均损失
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)

        # 打印每个epoch的平均损失
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

        # 每隔一定的epoch保存模型
        if ((epoch + 1) % save_interval == 0):
            save_model(model, epoch + 1)

    # 保存损失数据
    np.savetxt(os.path.join(save_dir, 'losses.txt'), np.array(losses))
```

### 推理
