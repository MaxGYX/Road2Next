**Transformer**架构来自Google的论文： **Attention is all you need**  https://arxiv.org/pdf/1706.03762

Transformer擅长处理文本数据，ChatGPT就是使用Transformer架构实现的。实际上任何时间先后序列的数据都能通过Transformer来处理，比如文本翻译，文本生成，语音识别，等等等...

RNN/LSTM虽然也是用于处理长序列数据，但是有两个弱点：
- 当句子很长时，难以处理相距较远的词之间的长距离依赖关系；
- 处理当前token的时候，依赖之前的token处理结果（意味着在计算时间t的计算之前，它必须完成时间t-1的计算），必须串行处理数据，所以训练和推理速度会比较慢。
  
Transformer架构使用一种叫做Attention的机制（注意力机制）解决了这两个问题。
