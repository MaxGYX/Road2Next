### VAE（变分自编码器 Variational Autoencoder）
VAE和AutoEncoder有类似的结构，即encoder和decoder这样的架构设计，也是**无监督**模型。

<img width="595" alt="image" src="https://github.com/user-attachments/assets/bd7af15a-aea2-46ad-b507-258d4b364645">

但与AE不同的是，VAE是一种概率模型（Probabilistic Model），其训练流程如下所示：
-  输入𝑥，encoder首先计算出**后验分布**的**均值**和**标准差**
-  然后通过**重采样方法采样得到隐变量𝑧**
-  然后送入decoder得到重建的数据𝑥′

encoder网络不直接得到隐含特征z，而是得到z的概率分布𝑝𝜃(𝑧)，所以VAE的encoder叫做Probabilistic encoder。

训练完成后，我们就得到生成模型𝑝𝜃(𝑥|𝑧)𝑝𝜃(𝑧)，其中𝑝𝜃(𝑥|𝑧)就是decoder网络，而先验𝑝𝜃(𝑧)为标准正态分布，我们从𝑝𝜃(𝑧)随机采样一个𝑧，送入decoder网络，就能生成与训练数据𝑋类似的样本。
