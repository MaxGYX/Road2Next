### VAE（变分自编码器 Variational Autoencoder）
VAE和AutoEncoder有类似的结构，即encoder和decoder这样的架构设计，也是**无监督**模型。

<img width="595" alt="image" src="https://github.com/user-attachments/assets/bd7af15a-aea2-46ad-b507-258d4b364645">

但与AE不同的是，VAE是一种概率模型（Probabilistic Model），其训练流程如下所示：
-  输入𝑥，encoder首先计算出**后验分布**的**均值**和**标准差**
-  然后通过**重采样方法采样得到隐变量𝑧**
-  然后送入decoder得到重建的数据𝑥′

encoder网络不直接得到隐含特征z，而是得到z的概率分布𝑝𝜃(𝑧)，所以VAE的encoder叫做Probabilistic encoder。

训练完成后，我们就得到生成模型𝑝𝜃(𝑥|𝑧)𝑝𝜃(𝑧)，其中𝑝𝜃(𝑥|𝑧)就是decoder网络，而先验𝑝𝜃(𝑧)为标准正态分布，我们从𝑝𝜃(𝑧)随机采样一个𝑧，送入decoder网络，就能生成与训练数据𝑋类似的样本。

<img width="562" alt="image" src="https://github.com/user-attachments/assets/190cdc29-2826-4a71-b7e2-934c2cb430aa">

encoder网络和decoder网络实际上就是几层全连接网络结构，encoder做降维（只是输出层是2个节点，一个均值，一个方差），decoder做升维重建。

**VAE的训练究竟学习到了什么**

VAE通过编码器从数据中学习到一个潜在变量的分布，然后通过解码器从这个潜在变量分布中进行采样并生成数据。编码器网络参数和解码器网络参数是通过优化ELBO的最大值来确定。
-  ELBO（Evidence Lower Bound）是VAE的优化目标，类似于损失函数的概念，不过ELBO的优化目标是让其最大。
-  ELBO包含两部分
  
    <img width="487" alt="image" src="https://github.com/user-attachments/assets/341d201e-8078-43e3-ab35-c5fe5e9668bf">
   
    - **重建误差**：，这部分表示在变分分布 𝑞(𝑧∣𝑥)，数据 𝑥 的对数似然 log⁡𝑝(𝑥∣𝑧)的期望值，衡量的是解码器从潜在变量 𝑧 重构出的数据与原始数据 𝑥 之间的差异。我们希望重构的数据尽可能接近原始数据，因此我们尝试最小化这个差异（最大化期望的对数似然）。
    - **KL散度**：编码器输出的潜在变量分布 𝑞(𝑧∣𝑥) 与先验分布 𝑝(𝑧) （实际是一个标准正态分布）之间的KL散度。KL散度用于两个概率分布之间的差异，这里我们希望最小化它，以确保潜在变量的分布尽可能接近我们预定义的先验分布。  

-  在VAE的训练过程中，我们的目标是最大化ELBO：

    -  最大化期望对数似然：提高模型重构数据的能力。
    -  最小化KL散度：确保潜在变量的分布与先验分布一致，这有助于控制潜在空间的复杂度和结构
    -  在反向传播过程中，通过梯度下降优化ELBO的最大值，迭代更新encoder/decoder的参数，以优化到ELBO的最大值。
        - 训练集中的一副图片x，通过encoder的计算到一个潜变量的分布，从这个分布随机采样出一个z，通过decoder的计算得到x’。计算ELBO，梯度下降，更新encoder/decoder的参数，重新走上面的过程。。。直到优化出ELBO最大值。




VAE的细节，之前看了几遍**苏剑林**老师的文章，现在想起来还是有点似懂非懂，继续多看几遍

https://kexue.fm/archives/5253

