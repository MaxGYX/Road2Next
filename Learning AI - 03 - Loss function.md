### 损失函数 Loss Function
* 损失函数是用来评估模型的**预测值**与**真实值**之间**误差大小**的**函数**。

* 理解：训练的目的是通过训练数据集来确定模型函数中的各个参数值，这些参数可以先从0开始（或者从随机值开始），通过每个训练数据的预测值（即通过模型函数得到的值）和真实值之间的差距，来确定Loss function的值。
通过找到这个loss function的最小值，以此就能确定模型函数的参数。

  损失函数可以有很多种，比如使用“均方误差（MSE）”做Loss function

  <img width="580" alt="image" src="https://github.com/MaxGYX/Road2Next/assets/158791943/a2a78ac6-a33d-4796-9381-c537f6e415e5">

