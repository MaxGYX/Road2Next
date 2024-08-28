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

     对这个输入矩阵做3个线性变换，和Wq矩阵做点积操作得到Q，和Wk矩阵做点积操作得到K，和Wv矩阵做点积操作得到V。
     -  Wq/Wk/Wv 三个矩阵都是通过训练得到的，这3个矩阵的维度相同。
     -  Q/K/V 都是原始输入数据的某种表示。


