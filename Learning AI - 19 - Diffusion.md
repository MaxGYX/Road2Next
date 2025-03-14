### Diffusion Model 扩散模型
Diffusion模型中，首先是定义一个从数据变成噪声的正向过程（一步步加噪声），然后是一个从噪声变成新图片的反向过程（一步步去噪音）。

#### 加噪过程（前向）

此过程将图片数据映射为噪声，每一个时刻都要添加高斯噪声，后一个时刻都是由前一个时刻加噪声得到。
<img width="427" alt="image" src="https://github.com/user-attachments/assets/087909ae-cc08-49d5-b9cd-d26920fd4004">

这个过程可以直接计算，不需要训练。


#### 去噪过程
扩散模型训练的主要目标就是能够**预测去噪过程中每个时间步的噪声**。


#### 结构
<img width="783" alt="image" src="https://github.com/user-attachments/assets/70c6d013-643f-40dc-961d-7eba2e5e5081">
    
**Diffusion Model结构中包含一个Scheduler模块和一个U-Net模块**，其中：
-  Scheduler负责加噪过程输出噪声每个step的噪声系数，噪声 = 随机生成噪声 * scheduler输出的系数。
-  U-Net负责在去噪过程中预测噪声。结构特点是U型设计和跳层残差连接。
    - U型网络指包含encoder/decoder的网络，其中
    - Encoder利用卷积进行下采样，逐步提取特征并降低空间维度。
    - Decoder进行上采样（如转置卷积）逐步恢复特征图的空间维度，再与从encoder步骤中的跳层残差连接（对应的编码器特征图）结合，一层层重建数据表示。
    - U-Net 的最终输出是对噪声的预测。这一输出是一个与输入图像相同尺寸的特征图，表示在当前时间步预测的噪声。
    - 预测出噪声后，就可以计算这个时间步去噪后的图像。
      
        <img width="175" alt="image" src="https://github.com/user-attachments/assets/4446582c-d1f4-4bd1-b718-b8918e67d971">
-  Diffusion Model是最小化预测噪声和实际噪声之间的差异来学习，当每一步预测噪声与实际噪声差异最小时，也就代表了恢复出来的图片与原图最接近。

      <img width="363" alt="image" src="https://github.com/user-attachments/assets/fd03b750-3f2e-4e42-b9b2-8e115169c21e">

**Diffusion去噪过程中某个时间步具体的执行过程，Condition是如何加入的**

去噪过程中，从 X𝑡 到 X𝑡-1 的步骤中，使用交叉注意力（cross-attention）嵌入条件。
- 在时间步t，输入到模型的主要内容包括：
    - 带噪声图像 X𝑡：当前时间步的输入图像。
    - 条件信息：可以是文本描述、标签、图像等其他信息，用于指导生成过程。
- 将条件信息转化为特定维度的向量_**C**_，根据Xt的形状调整条件向量的形状，使其能进行下一步Attention的计算
- 在U-Net结构中，CrossAttention层 会进行Attention操作

   <img width="423" alt="image" src="https://github.com/user-attachments/assets/d636261c-d48a-4502-ae32-fa1b445b07af">
  
    - Q矩阵来自带噪声图像Xt
    - K，V矩阵来自条件向量C
    - Attention的结果就是结合了条件向量的带噪声图像，是一个新的特征表示
 - U-Net encoder对经过Attention融合后的带噪声数据进行一步步特征提取（降维）
 - U-Net decoder结合跳层连接，进行图像的逐步恢复。
 - U-Net的最后一层的输出，就是对这个时间步的噪声的预测。
 - 有了预测的噪声，就可以更新得到 X𝑡-1 的带噪图像，进行下一个时间步的计算。
  

**B站李宏毅老师的讲解**

https://www.bilibili.com/video/BV1734y1c7Hb/?p=3

一直不理解的一个问题： https://www.zhihu.com/question/574586781/answer/3001481574
