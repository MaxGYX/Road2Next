Deepseek也是基于Transformer网络模型，并在Transformer模型中的一些环节进行了极致的性能优化。这些优化之后，Deepseek可以用比较低的成本达到比较好的效果（也就是性价比高）。

### Transformer的结构
-	Transformer网络结构如下图，其中左边的部分称为encoder，右边称为decoder，由于Google最早将Transformer结构用于机器翻译任务（比如从中文翻译成英文），因此输入是一个中文的序列，输出是一个英文的序列，两个序列语言不同，因此设计了一个encoder-decoder结构的transformer，encoder一个词（token）一个词的接受中文输入，decoder一个词一个词的输出翻译后的英文。

-  而后来OpenAI将transformer用于了文本生成任务，文本生成任务只有一种语言序列，因此为了结构简单和减小计算量，OpenAI只使用了decoder结构（即decoder-only transformer）。

  后面也有人尝试encoder-only结构等，主流使用的是decoder-only transformer。
  
  <img width="310" alt="image" src="https://github.com/user-attachments/assets/ecb14dd8-7a99-4124-949f-0bd62491e4c5" />

-	Transformer结构中，是由N个顺序连接的Transformer block结构组成，每个Transformer Block里还饱含几个主要的结构：
    -  Attention：这部分结构模块负责注意力的计算。是整体结构中计算量最大的部分。
    -  Feed-Forward Network：FFN前向网络，这部分是一个很大深度的神经网络结构，是整体结构中参数量最大的部分。

 <img width="587" alt="image" src="https://github.com/user-attachments/assets/ca0b6814-746a-41df-810f-afa5a252b791" />

### DeepSeek的性能主要改进

-  **MLA：Multi-head Latent Attention多头潜空间注意力**

  是对Attention模块的改进，将数据降维到Latent space（潜空间）再进行Attention的计算，同时减少了很多重复计算（通过优化KV cache缓存技术），因此减小了数据的计算量。如上图所示。

-  **Deepseek-MOE：deepseek混合专家模型**
  是对FFN前向网络模块的改进，由于FFN模块的参数量很大，因此出现了一种对参数进行分组的机制（MOE），让某一类问题使用某个分组参数（称为Expert专家），不同的问题使用不同组的专家（参数组）来表征。Deepseek对MOE进行了改进，如上图所示。

  对参数分组以及不同问题路由到不同Expert的机制进行了改进，降低了对问题的推理成本（即回答某个问题调动的参数减小了）。

-  **MTP（Multiple token prediction）多Token预测**
  每次不只预测一个token，而是预测出多个token，加快了执行效率

-  **FP8混合精度**
  将一部分数据使用8bit浮点数来表示（通常浮点数使用32bits或者64bits来表示），降低了内存使用和计算量。

### DeepSeek的功能主要改进
-  Deepseek有V3/R1两个模型，V3模型参数量比较大（有6710亿个参数，参数尺寸671b）。R1模型参数量比较小，主要用于推理，擅长解决数学和编程问题。
-  R1模型的后训练（post-training）中，采用了一种全新的RL（Reinforcement learning强化学习）方式。后训练主要是使用精选的数据集来调优模型的推理效果（意思是让模型使用精选过的数学题和答案进行再次训练，这样就能在之后遇到新题的时候给出比较好的做题步骤和结果），这种方式称为SFT（Supervised Fine-Tuning，有监督微调）。准备这些优质数据的成本是很高的。

R1模型没有使用SFT，而是完全使用机器自己强化学习的方式产生做题的中间推导思路。这个是比较创新的做法。


