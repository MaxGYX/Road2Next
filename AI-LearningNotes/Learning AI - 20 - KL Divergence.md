在VAE和Diffusion里都用到了KL Divergence来衡量两个分布之间的“距离“。专门查了些资料，终于有点懂了。
### 信息量的衡量：熵
信息论中，概率越小的事件，其中蕴含的信息量越大。假设事件分布P下，事件信息量的平均值叫做“熵”。
如何表达概率越小的Event蕴含的信息量越大？方法是取对数 -log(P(X))，其平均值就是E(P(x).(-log(P(X))))。

<img width="261" alt="image" src="https://github.com/user-attachments/assets/c25d1699-85a2-4684-9461-5119f577b275">


### 交叉熵
衡量另外一个分布Q对于当前分布P得熵的差值（信息量的差距），叫做交叉熵，差距越小意味着两个分布越相似。两个分布的交叉熵就称为两个分布的KL散度。

### KL散度 （Kullback-Leibler Divergence）
又称为相对熵，或者信息散度。是衡量两个概率分布P和Q差异的一种度量方法。

<img width="643" alt="image" src="https://github.com/user-attachments/assets/826486fb-772e-4ed2-8684-ab157f9ed887">

公式表达：

<img width="348" alt="image" src="https://github.com/user-attachments/assets/2518c4a6-5747-4368-8868-dbf779ceb665">


