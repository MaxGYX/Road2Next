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
Tensor张量，可以理解成一个多维数组，PyTorch中大多数操作都围绕着Tensor进行
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
