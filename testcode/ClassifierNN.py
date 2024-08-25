import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
def load_image(image_path, transform=None):
    image = Image.open(image_path)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

# 定义一个CNN模型（从nn.Module继承），需要自己实现两个方法
#   __init__：定义网络模型的层次结构
#   forward：定义前向计算的过程，在Module的训练过程中自动被调用
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 通过Sequential定义卷积层，结构为：卷积->非线性激活->最大池化->卷积->非线性激活->最大池化
        self.conv_layers = nn.Sequential(
            # 第1层卷积：in_channels,输入数据具有3个通道（彩色图像RGB），即输入数据是3个32x32的tensor结构。
            #           out_channels,输出通道6，即采用6个卷积核对输入数据进行卷积计算，输出6个特征图。
            #           kernel_size,卷积核尺寸 5x5
            #   第1层卷积后，数据3x32x32 -> 6x28x28
            nn.Conv2d(3, 6, 5),
            # 卷积之后接一个ReLU激活函数
            #   ReLU之后数据尺寸不变
            nn.ReLU(),
            # 最大池化层，采用最大池化策略 nn.MaxPool2d (2维数据的最大池化）
            #           kernel_size=2, 对2x2的区域进行池化
            #           stride=2，每次步长=2
            #   经过2x2最大池化之后，数据 6x28x28->6x14x14
            nn.MaxPool2d(2, 2),
            # 第2层卷积：in_channels,输入数据具有6个通道（即第1层卷积后的6个特征图）
            #           out_channels,输出维度16，即采用16个卷积核对输入进行卷积计算，输出16个特征图
            #           kernel_size,卷积核尺寸 5x5
            #   第1层卷积后，数据6x14x14 -> 16x10x10
            nn.Conv2d(6, 16, 5),
            #   ReLU之后，数据尺寸不变
            nn.ReLU(),
            # 最大池化层，采用最大池化策略 nn.MaxPool2d (2维数据的最大池化）
            #           kernel_size=2, 对2x2的区域进行池化，stride=2，每次步长=2
            #   经过2x2最大池化之后，数据 16x10x10 -> 16x5x5
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            # 第1层全连接线性层，将卷积之后的16x5x5数据，归类到120个输出结果上（即这个线性层有 120个节点）
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            # 第2层全连接线性层，将120个输入，归类到84个输出结果上（（即这个线性层有 84个节点）
            nn.Linear(120, 84),
            nn.ReLU(),
            # 第3层全连接线性层，将84个输入，归类到10个输出结果上（（即这个线性层有 10个节点）
            nn.Linear(84, 10)
        )

    def forward(self, x):
        # 先通过conv_layers中定义的卷积运算
        x = self.conv_layers(x)
        # x.view()是将tensor结构数据进行reshape(改变尺寸)
        #   -1,表示不确认有多少行,由函数自己来计算
        #   16x5x5, 表示reshape后每一行数据有16x5x5列
        #   实际上经过这个函数后，每个输入(3x32x32)多次卷积后的数据会排列成一个16x5x5的列，相当于把tensor展平
        x = x.view(-1, 16*5*5)
        # 最后通过fc_layers中定义的全连接线性层运算
        x = self.fc_layers(x)
        return x

# 通过transforms.Compose()定义对数据的处理序列，包括：转换成Tensor结构，并进行归一化处理（Norm）
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                ])
# CIFAR10数据集，每张图片格式：彩色32x32
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Net()
# LossFunction用交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 优化器使用SGD，将net.parameters传递给优化器，学习率learn_rate=0.001
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 通过trainset训练模型
# num_epochs = 20
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if i%2000 == 1999:
#             print('[%d, %5d] loss: %.3f' %(epoch+1, i+1, running_loss/2000))
#     if (epoch+1)%5 == 0:
#         torch.save(net.state_dict(), f'model_epoch_{epoch+1}.pt')
#         print(f'Saved model at epoch {epoch+1}')
# print('Finished Training!')

# 推理image的类型，image格式32x32
image_path = '/Users/MaxGYX/NNTest/data/1.jpg'
image = load_image(image_path, transform)

# load model
checkpoint = torch.load('model_epoch_20.pt')
net.load_state_dict(checkpoint)

net.eval()
with torch.no_grad():
    outputs = net(image)
    _, predicted = torch.max(outputs.data, 1)
    print(f'Predicted class: {classes[predicted]}')
