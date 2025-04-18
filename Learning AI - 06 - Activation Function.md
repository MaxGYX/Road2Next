### 再理解一下神经网络中的神经元在做什么

一个神经元可能有多个input，然后经过计算输出一个output，这个计算实际上是根据参数（权重）做的线性运算。因此神经网络中每一层的输入输出都是一个线性求和的过程。

在完成这个线性计算后，都会使用一个非线性的函数对结果再进行一次计算。这个非线性函数就是**激活函数（Activation function）**。
<img width="1034" alt="image" src="https://github.com/MaxGYX/Road2Next/assets/158791943/94e27ad5-6cb2-416a-a2a7-454ef494a072">

所以激活函数的**作用**是**为函数添加非线性属性**。

* 如果没有激活函数，神经网络的多层计算实际上会等价于只有一层（线性计算的叠加），在每一层后面加入一个非线性的运算，相邻的两层无法等价于一层了。

* 如果没有激活函数，那么构造的神经网络多复杂，有多少层，最后的输出都是输入的线性组合，等价于只有一层。而纯粹的线性组合并不能够解决更为复杂的问题。

* 引入激活函数之后（常见的激活函数都是非线性的），因此也会给神经元引入非线性元素，在每一层后面加入一个非线性的运算，相邻的两层无法等价于一层了。这样就使得神经网络可以逼近其他的任何非线性函数，神经网络就可以应用到更多非线性模型中。


经常使用的激活函数有：

<img width="539" alt="image" src="https://github.com/MaxGYX/Road2Next/assets/158791943/7711dbcf-9b6e-4da3-b67e-604d3a5f25e7">

#### sigmoid
-  优点：
    -  函数输出值在0到1之间，适合表示概率。
    -  梯度平滑，便于求导。
-  场景：常用于二分类问题的输出层
-  缺点：
    -  在深层网络中容易造成梯度消失问题。
    -  输出不是以0为中心的，可能导致权重更新效率降低。
    -  包含指数运算，计算复杂度较高。

#### tanh 双曲正切激活函数
-  场景：类似于Sigmoid，但输出以0为中心，常用于隐藏层
-  优点：输出值在(-1,1)之间，以0为中心
-  缺点：同样存在梯度消失问题；计算时涉及指数运算

#### ReLU 修正线性单元
- 场景：计算效率高，通常作为隐藏层的激活函数。
- 优点：解决了梯度消失问题；计算速度快，不需要指数运算
- 缺点：存在Dead ReLU问题，即当输入为负时，梯度为0，导致部分神经元不再更新；输出不是以0为中心的。

#### Softmax
- 场景：多分类问题的输出层，将输出转换为概率分布。
- 优点：输出值在0到1之间，且加和为1，可以解释为概率分布。
- 缺点：对于具有大量类别的输出，计算量大；当输入值很大或很小的时候，会出现数值问题。

  <img width="221" alt="image" src="https://github.com/user-attachments/assets/0ed7e7bf-1f60-4aa4-9adf-182021504460">

