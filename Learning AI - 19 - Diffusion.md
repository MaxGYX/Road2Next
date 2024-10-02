### Diffusion Model 扩散模型
Diffusion模型中，首先是定义一个从数据变成噪声的正向过程（一步步加噪声），然后是一个从噪声变成新图片的反向过程（一步步去噪音）。

#### 加噪过程（前向）

此过程将图片数据映射为噪声，每一个时刻都要添加高斯噪声，后一个时刻都是由前一个时刻加噪声得到。
<img width="427" alt="image" src="https://github.com/user-attachments/assets/087909ae-cc08-49d5-b9cd-d26920fd4004">


#### 去噪过程


Diffusion Model也是一个U型网络。

B站李宏毅老师的讲解

https://www.bilibili.com/video/BV1734y1c7Hb/?p=3
