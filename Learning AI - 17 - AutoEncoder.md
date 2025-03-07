**AutoEncoder（自动编码器AE）**，是一种**无监督**学习方法，原理：先将高维的原始数据映射到一个低维特征空间，然后从低维特征学习重建原始的数据。

一个AE模型包含两部分网络：
-  Encoder：将原始的高维数据映射到低维**特征空间**，这个特征维度一般比原始数据维度要小，所以起到压缩或者**降维**的目的，这个低维特征也往往成为中间隐含特征（latent representation）；
-  Decoder：基于压缩后的低维特征来重建原始数据；
<img width="537" alt="image" src="https://github.com/user-attachments/assets/1140f892-f376-4c88-a31b-44aa93802d68">

-  图中𝑔𝜙为encoder网络的映射函数（网络参数为𝜙），而𝑓𝜃为decoder网络的映射函数（网络参数为𝜃）。
-  对于输入𝑥，可以通过encoder得到隐含特征𝑧=𝑔𝜙(𝑥)，然后decoder从隐含特征对原始数据进行重建：𝑥′=𝑓𝜃(𝑧)=𝑓𝜃(𝑔𝜙(𝑥))。
-  我们希望重建的数据和原来的数据近似一致的，那么AE的训练损失函数可以采用MSE：

   <img width="348" alt="image" src="https://github.com/user-attachments/assets/e3cbcb42-e75b-4856-a371-6e8f4f4710f4">

-  **训练出来的参数就是𝜙/𝜃**，推理的时候使用这些参数进行encode/decode运算即可根据输入x生成类似的图片x’。

AE这种无监督的学习方式有效减少了很多人工打标签的工作。

#### 再理解一下“降维”
“降维”是指**减少**描述某些数据的**特征数量**的过程。

降维可以通过选择（只保留部分现有特征）或提取（根据旧特征创建数量减少的新特征）来完成，降维后对于数据存储、减少计算量都非常有帮助。
比如使用一个512维的向量描述一个数据（每个维度都代表着歌数据在某一方面的特征），降维后成为64维向量，就是说压缩或者去除了一些不太重要的特征，只保留了最核心最本质的特征。
