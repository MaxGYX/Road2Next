**1.Conda 建立 Env**

    Conda可以建立多个虚拟环境，每个环境中的python或者第三方库的版本可以不同。

- 下载Conda并安装：[Download Anaconda Distribution | Anaconda](https://www.anaconda.com/download)  

- 打开Terminal，执行

        //查看当前conda中已经创建的env列表

        #**conda env list**

        //创建一个名为”MyPyTorch“的env，环境的python版本为3.10

        #**conda create -n MyPytorch python=3.10**

- 进入刚刚建立的Env

        #**conda activate MyPytorch**

**2.通过Conda安装PyTorch**

- 选择Mac的版本 https://pytorch.org/get-started/locally/#mac-anaconda
      <img width="516" alt="image" src="https://github.com/MaxGYX/Road2Next/assets/158791943/b94c56e1-4229-4b1f-afc4-e75ac0092f45">

        //在Terminal中执行上面的命令

        #**conda install pytorch::pytorch torchvision torchaudio -c pytorch**

**3.测试PyTorch是否安装正常**

- 在Terminal中，进入python3

            #python3

            输入测试程序

```python
import torch
x = torch.rand(5, 3)
print(x)
exit() //退出python3
```

        输出正常，表示torch安装成功

**4.Conda其他命令**

    退出Env ： #**conda deactivate**

    删除某个Env： #**conda env remove -n MyPytorch —all**

**5.配置PyCharm的PyTorch环境**

- 打开PyCharm的Settings，选择Python解释器，选择Conda environment，选中刚刚创建的MyPytorch。

- 测试PyCharm中是否配置PyTorch完成。
  
  创建test.py，在PyCharm中运行，输出正常说明PyTorch配置正常。
  
  ```python
   import torch
   
   print(torch.__version__)
   x = torch.rand(5, 3)
   print(x)
  ```

       
