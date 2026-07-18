**Transformer**架构来自Google的论文： **Attention is all you need**  https://arxiv.org/pdf/1706.03762

Transformer擅长处理文本数据，ChatGPT就是使用Transformer架构实现的。实际上任何时间先后序列的数据都能通过Transformer来处理，比如文本翻译，文本生成，语音识别，等等等...

RNN/LSTM虽然也是用于处理长序列数据，但是有两个弱点：
- 当句子很长时，难以处理相距较远的词之间的长距离依赖关系；
- 处理当前token的时候，依赖之前的token处理结果（意味着在计算时间t的计算之前，它必须完成时间t-1的计算），必须串行处理数据，所以训练和推理速度会比较慢。
  
Transformer架构使用一种叫做Attention的机制（注意力机制）解决了这两个问题。

### Self-Attention自注意力模型
Self-Attention是Transformer的核心，本质是**描述输入的数据各元素（Token）之间的相关性，也就是输入数据本身的内在关联**。

下图左半部分，Scaled Dot-Product Attention描述了自注意力的实现过程，也就是使用矩阵的点积（Dot-Product）运算来实现token之间的相关性计算。

<img width="706" alt="image" src="https://github.com/user-attachments/assets/b14da76c-9f22-4a46-8a13-0ffaeff7af8c">

公式描述就是：

<img width="347" alt="image" src="https://github.com/user-attachments/assets/0921ab11-7167-4f5c-8a66-8b5bd34dc472">

**详细理解这个过程**
-  1，首先是对输入的数据进行数字化的表示，比如输入是一句话“you are welcome PAD”，对每一个token都表示成一个数值向量（一句话就表示成了一个矩阵， Embedding+Positional Encoding）
      <img width="776" alt="image" src="https://github.com/user-attachments/assets/0a0db67b-b7df-4094-aaff-5f2281c408ab">

     对这个输入矩阵做3个线性变换，和Wq矩阵做点积操作得到Q（Query），和Wk矩阵做点积操作得到K（Key），和Wv矩阵做点积操作得到V（Value）。
     -  Wq/Wk/Wv 三个矩阵都是通过训练得到的，这3个矩阵的维度相同 (维度都是Emb*Emb）。
     -  Q/K/V 都是原始输入数据通过线性变换得到的，因此可以理解都是原始数据的某种表示。
-  2， Q矩阵和K的转置矩阵做点积运算，Query矩阵每一行表示一个token，Key矩阵转置后每一列代表一个token，因此点积运算后实际上是得到的是每一个token和其他token（包括自己）之间的相似性。

      <img width="461" alt="image" src="https://github.com/user-attachments/assets/fb1e3c85-42df-4ce7-8b37-4762dc726097">
        
-  3， 通过K矩阵的维度 𝑑𝑘 进行scaling，进行缩放的原因是当 𝑑𝑘 比较大时，Q/K运算后会得到很大的值，而这将导致在经过sofrmax操作后产生非常小的梯度（输入softmax的值过大，会导致偏导数趋近于0，从而导致梯度消失），不利于网络的训练。

       ❝ We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients.❞
-  4， softmax操作将一个token所对应的其他token的相似度进行归一，即每一行的和为1。
-  5， 用得到的token相似度矩阵作为权重对V值进行加权平均，得到最终的编码输出。
  
      <img width="476" alt="image" src="https://github.com/user-attachments/assets/795e02e9-a69c-4257-987b-efd44e6939aa">
      
      换一个角度看这个过程，对于某一个token，

      <img width="472" alt="image" src="https://github.com/user-attachments/assets/941810a8-a338-41b7-af0b-447900c31995">

      对于最终输出“是”的编码向量来说，它其实就是原始“我/是/谁”3个向量的相关度的加权和，即最终得到的“是”这个token其实是包含了与这句话里全部token的相关性信息。

以上就是Self-Attention（自注意力模型）的实现过程，这也是**对输入数据进行编码并提取特征**的过程。
由于每个token的特征提取都用到了输入数据中所有的token，所以这是一种**全局**的特征提取，采用加权求和的方式，权重是tokne之间的相关性。

### Multi-Head Attention 多头注意力
Multi-head实际上就是把Wq/Wk/Wv矩阵拆成h个（head数）小矩阵，并行进行h个Self-Attention运算，然后把结果再拼接起来。

<img width="665" alt="image" src="https://github.com/user-attachments/assets/f2298589-6dba-4fe1-89c7-03fdaf407ea5">

上图表示的是2个head的计算过程。在《Attention is all you need》论文中，使用了h=8并行的自注意力模块（8个头）来构建一个注意力层，Wq/Wk/Wv矩阵列数dq=dk=dv=dmodel/h。

**实际上，多头注意力机制其实就是将一个大的高维单头拆分成了h个多头**。
<img width="581" alt="image" src="https://github.com/user-attachments/assets/19edecc0-7db6-4eff-842f-7d8009a32d96">

得到多个输出矩阵之后，Multi-Head Attention 将它们拼接在一起 (Concat)，然后传入一个Linear层【和一个Wo矩阵（列数dm）进行点积运算】，得到 Multi-Head Attention 最终的输出Z
<img width="713" alt="image" src="https://github.com/user-attachments/assets/aa6b4758-17d3-4c4e-ac72-7fce7827832c">

#### 为什么要用Multi-head
自注意力机制的缺陷就是：模型在对当前位置的信息进行编码时，会过度的将注意力集中于自身的位置，因此作者提出了通过多头注意力机制来解决这一问题。

同时，使用多头注意力机制还能够给予**注意力层的输出包含有不同子空间中的编码表示信息，从而增强模型的表达能力**。（不太懂，先放在这里）

❝ Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. ❞


### 参考
很多内容是从找到的比较适合自己理解的文章中摘取，**对初学者很友好的文章：**
-  这篇很详细讲解了Attention和Transformer的网络结构：https://zhuanlan.zhihu.com/p/720320507
-  https://www.zhihu.com/question/341222779/answer/2466825259
-  https://zhuanlan.zhihu.com/p/338817680
-  强烈推荐TransformerNeuralNetworks@**StatQuest** https://www.youtube.com/watch?v=zxQyTK8quyY




