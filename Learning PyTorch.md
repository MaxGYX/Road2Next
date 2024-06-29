选择用**PyTorch**来实现后面的项目，开一篇来记录学习中遇到的问题
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

