#复杂的网络结构可以由一些很简单的块（block）来搭建而成
import torch
import torch.nn as nn
import torch.nn.functional as F


#我们的第一个块叫Flatten，它的功能是将一个张量展平
#Flatten用于一系列的卷积操作之后，全连接层之前。因为深度学习模型的卷积操作往往在四维空间里，全连接层在二维空间里，Flatten能够将四维空间平铺成二维空间。
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

# 第二个块叫ConvBn，是卷积操作加一个BN层。
class ConvBn(nn.Module):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_c)
        )
        
    def forward(self, x):
        return self.net(x)

# 第三个块叫ConvBnPrelu，我们在刚刚搭建好的ConvBn块里又加了一个PReLu激活层。
class ConvBnPrelu(nn.Module):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBn(in_c, out_c, kernel, stride, padding, groups),
            nn.PReLU(out_c)
        )

    def forward(self, x):
        return self.net(x)
    
# 第四个块叫DepthWise，DepthWise层使用了上面定义的ConvBnPrelu和ConvBn，它按通道进行卷积操作，实现了高效计算。注意中间的ConvBnPrelu块的groups=groups。
class DepthWise(nn.Module):

    def __init__(self, in_c, out_c, kernel=(3, 3), stride=2, padding=1, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnPrelu(in_c, groups, kernel=(1, 1), stride=1, padding=0),
            ConvBnPrelu(groups, groups, kernel=kernel, stride=stride, padding=padding,groups=groups),
            ConvBn(groups, out_c, kernel=(1, 1), stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x)
    
# 第五个块叫DepthWiseRes，在第四个块的基础上，添加了一个原始输入，这是ResNet系列的精要。
class DepthWiseRes(nn.Module):
    """DepthWise with Residual"""

    def __init__(self, in_c, out_c, kernel=(3, 3), stride=2, padding=1, groups=1):
        super().__init__()
        self.net = DepthWise(in_c, out_c, kernel, stride, padding, groups)

    def forward(self, x):
        return self.net(x) + x
    
#第六个块叫MultiDepthWiseRes，与之前的不同，我多传入一个num_block的参数，由这个参数决定要堆多少个DepthWiseRes。由于这些DepthWiseRes的输入输出的通道数是一样的，所以堆多少都不会引起通道数的变化。
class MultiDepthWiseRes(nn.Module):

    def __init__(self, num_block, channels, kernel=(3, 3), stride=1, padding=1, groups=1):
        super().__init__()

        self.net = nn.Sequential(*[
            DepthWiseRes(channels, channels, kernel, stride, padding, groups)
            for _ in range(num_block)
        ])

    def forward(self, x):
        return self.net(x)
        

#由于前面已经定义好了六个块，现在这个FaceMobilenet的结构一目了然。我们先是堆叠了10种不同的卷积块，然后接一个Flatten块把输入展平，再接一个全连接层和1维的BatchNorm层。
class FaceMobileNet(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.conv1 = ConvBnPrelu(1, 64, kernel=(3, 3), stride=2, padding=1)
        self.conv2 = ConvBn(64, 64, kernel=(3, 3), stride=1, padding=1, groups=64)
        self.conv3 = DepthWise(64, 64, kernel=(3, 3), stride=2, padding=1, groups=128)
        self.conv4 = MultiDepthWiseRes(num_block=4, channels=64, kernel=3, stride=1, padding=1, groups=128)
        self.conv5 = DepthWise(64, 128, kernel=(3, 3), stride=2, padding=1, groups=256)
        self.conv6 = MultiDepthWiseRes(num_block=6, channels=128, kernel=(3, 3), stride=1, padding=1, groups=256)
        self.conv7 = DepthWise(128, 128, kernel=(3, 3), stride=2, padding=1, groups=512)
        self.conv8 = MultiDepthWiseRes(num_block=2, channels=128, kernel=(3, 3), stride=1, padding=1, groups=256)
        self.conv9 = ConvBnPrelu(128, 512, kernel=(1, 1))
        self.conv10 = ConvBn(512, 512, groups=512, kernel=(7, 7))
        self.flatten = Flatten()
        #由于我们的输入是1 x 128 x 128，经过多层卷积之后，其变成512 x 2 x 2，也就是2048。如果你不知道或者你懒得去算输入的图片经过卷积之后的维度是多少，你可以给网络传入一个假数据，报错信息会告诉你这个维度的值。
        #另外，这里的embedding_size由外部传入，它表示用多大的向量来表示一张人脸。像 Facenet 是使用了128维的向量来表征一张人脸，我们这里使用512。
        self.linear = nn.Linear(2048, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return out
    
class myFaceMobileNet(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.conv1 = ConvBnPrelu(1, 64, kernel=(3, 3), stride=2, padding=1)
        self.conv2 = ConvBn(64, 64, kernel=(3, 3), stride=1, padding=1, groups=64)
        self.conv3 = DepthWise(64, 64, kernel=(3, 3), stride=2, padding=1, groups=128)
        self.conv4 = MultiDepthWiseRes(num_block=2, channels=64, kernel=3, stride=1, padding=1, groups=128)
        self.conv5 = DepthWise(64, 128, kernel=(3, 3), stride=2, padding=1, groups=256)
        self.conv6 = MultiDepthWiseRes(num_block=4, channels=128, kernel=(3, 3), stride=1, padding=1, groups=256)
        self.conv7 = DepthWise(128, 128, kernel=(3, 3), stride=2, padding=1, groups=512)
        self.conv8 = MultiDepthWiseRes(num_block=2, channels=128, kernel=(3, 3), stride=1, padding=1, groups=256)
        self.conv9 = ConvBnPrelu(128, 512, kernel=(1, 1))
        self.conv10 = ConvBn(512, 512, groups=512, kernel=(7, 7))
        self.flatten = Flatten()
        #由于我们的输入是1 x 128 x 128，经过多层卷积之后，其变成512 x 2 x 2，也就是2048。如果你不知道或者你懒得去算输入的图片经过卷积之后的维度是多少，你可以给网络传入一个假数据，报错信息会告诉你这个维度的值。
        #另外，这里的embedding_size由外部传入，它表示用多大的向量来表示一张人脸。像 Facenet 是使用了128维的向量来表征一张人脸，我们这里使用512。
        self.linear = nn.Linear(2048, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return out
    
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    x = Image.open("../data/lfw/Thabo_Mbeki/Thabo_Mbeki_0004.jpg").convert('L')
    x = x.resize((128, 128))
    x = np.asarray(x, dtype=np.float32)
    x = x[None, None, ...]
    x = torch.from_numpy(x)
    net = myFaceMobileNet(512)
    net.eval()
    with torch.no_grad():
        out = net(x)
    
    print(out.shape)