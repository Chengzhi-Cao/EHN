import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

###### Layer 
def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 1,
                    stride =stride, padding=0,bias=False)

def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 3,
        stride =stride, padding=1,bias=False)

class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,):
        super(Bottleneck,self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False,dilation=2)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu= nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x) 
        return out



class Attention(nn.Module):
    def __init__(self,in_channels):
        super(Attention,self).__init__()
        self.out_channels = int(in_channels/2)
        self.conv1 = nn.Conv2d(in_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels,4,kernel_size=1,padding=0,stride=1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


###### Network
class SPANet(nn.Module):
    def __init__(self,in_nc,out_nc):
        super(SPANet,self).__init__()

        self.conv_in = nn.Sequential(
            conv3x3(3,32),
            nn.ReLU(True)
            )
        self.res_block1 = Bottleneck(32,32)
        self.res_block2 = Bottleneck(32,32)
        self.res_block3 = Bottleneck(32,32)
        self.res_block4 = Bottleneck(32,32)
        self.res_block5 = Bottleneck(32,32)
        self.res_block6 = Bottleneck(32,32)
        self.res_block7 = Bottleneck(32,32)
        self.res_block8 = Bottleneck(32,32)
        self.res_block9 = Bottleneck(32,32)
        self.res_block10 = Bottleneck(32,32)
        self.res_block11 = Bottleneck(32,32)
        self.res_block12 = Bottleneck(32,32)
        self.res_block13 = Bottleneck(32,32)
        self.res_block14 = Bottleneck(32,32)
        self.res_block15 = Bottleneck(32,32)
        self.res_block16 = Bottleneck(32,32)
        self.res_block17 = Bottleneck(32,32)
        self.conv_out = nn.Sequential(
            conv3x3(32,3)
        )
    def forward(self, x):

        out = self.conv_in(x)
        out = F.relu(self.res_block1(out) + out)
        out = F.relu(self.res_block2(out) + out)
        out = F.relu(self.res_block3(out) + out)
        
        Attention1 = out
        out = F.relu(self.res_block4(out) * Attention1  + out)
        out = F.relu(self.res_block5(out) * Attention1  + out)
        out = F.relu(self.res_block6(out) * Attention1  + out)
        
        Attention2 = out
        out = F.relu(self.res_block7(out) * Attention2 + out)
        out = F.relu(self.res_block8(out) * Attention2 + out)
        out = F.relu(self.res_block9(out) * Attention2 + out)
        
        Attention3 = out
        out = F.relu(self.res_block10(out) * Attention3 + out)
        out = F.relu(self.res_block11(out) * Attention3 + out)
        out = F.relu(self.res_block12(out) * Attention3 + out)
        
        Attention4 = out
        out = F.relu(self.res_block13(out) * Attention4 + out)
        out = F.relu(self.res_block14(out) * Attention4 + out)
        out = F.relu(self.res_block15(out) * Attention4 + out)
        
        out = F.relu(self.res_block16(out) + out)
        out = F.relu(self.res_block17(out) + out)
       
        out = self.conv_out(out)

        return out

# my_model = SPANet()
# _input = torch.zeros(2,3,50,50)
# _output = my_model(_input)
# print('_input=',_input.shape)
# print('_output=',_output.shape)