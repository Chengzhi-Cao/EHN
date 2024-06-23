import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import itertools
from PIL import Image
from torchvision import models
from torch.autograd import Variable

class SELayer(nn.Module):
    def __init__(self, channel, reduction=64):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, reduction),
                nn.ReLU(inplace=True),
                nn.Linear(reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y    

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# SE-ResNet Module    
class SEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=64, with_norm=False):
        super(SEBasicBlock, self).__init__()
        self.with_norm = with_norm
        
        self.conv1 = conv3x3(inplanes, planes, stride)                    
        self.conv2 = conv3x3(planes, planes, 1)        
        self.se = SELayer(planes, reduction)
        self.relu = nn.ReLU(inplace=True)        
        if self.with_norm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        if self.with_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.with_norm:
            out = self.bn2(out)
        out = self.se(out)        
        out += x        
        out = self.relu(out)
        return out


class InsNorm(nn.Module):    
    def __init__(self, dim, eps=1e-9):
        super(InsNorm, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()
        
    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):        
        flat_len = x.size(2)*x.size(3)
        vec = x.view(x.size(0), x.size(1), flat_len)
        mean = torch.mean(vec, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        var = torch.var(vec, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((flat_len - 1)/float(flat_len))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var+self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out
    
# DualUpDownLayer is DuRB_US, defined here:
class DualUpDownLayer(nn.Module):
    def __init__(self, in_dim, out_dim, res_dim, f_size=3, dilation=1, norm_type="instance", with_relu=True):
        super(DualUpDownLayer, self).__init__()
        
        self.conv1 = ConvLayer(in_dim, in_dim, 3, 1)
        self.conv2 = ConvLayer(in_dim, in_dim, 3, 1)
        
        # T^{l}_{1}: (up+conv.)
        # -- Up --
        self.conv_pre = ConvLayer(in_dim, 4*in_dim, 3, 1)
        self.upsamp = nn.PixelShuffle(2)
        # --------
        self.up_conv = ConvLayer(res_dim, res_dim, kernel_size=f_size, stride=1, dilation=dilation)

        # T^{l}_{2}: (se+conv.), stride=2 for down-scaling.
        self.se = SEBasicBlock(res_dim, res_dim, reduction=32)
        self.down_conv = ConvLayer(res_dim, out_dim, kernel_size=3, stride=2)

        self.with_relu = with_relu            
        self.relu = nn.ReLU()

    def forward(self, x, res):
        x_r = x
        
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x+= x_r
        x = self.relu(x)

        x = self.conv_pre(x)
        x = self.upsamp(x)
        x = self.up_conv(x)
        x+= res
        x = self.relu(x)
        res = x

        x = self.se(x)
        x = self.down_conv(x)
        x+= x_r        

        if self.with_relu:
            x = self.relu(x)
        else:
            pass

        return x, res               


# ------------------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(ConvLayer, self).__init__()
        self.dilation=dilation
        if dilation == 1:            
            reflect_padding = int(np.floor(kernel_size/2))
            self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
            self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation)
        else:
            self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, padding=dilation)

    def forward(self, x):
        if self.dilation == 1:
            out = self.reflection_pad(x)
            out = self.conv2d(out)
        else:
            out = self.conv2d(x)
        return out

        
class FeatNorm(nn.Module):
    def __init__(self, norm_type, dim):
        super(FeatNorm, self).__init__()
        if norm_type == "instance":
            self.norm = InsNorm(dim)
        elif norm_type == "batch_norm":
            self.norm = nn.BatchNorm2d(dim)
        else:
            raise Exception("Normalization type incorrect.")

    def forward(self, x):
        out = self.norm(x)        
        return out

class RLNet_event(nn.Module):
    def __init__(self,in_nc,out_nc):
        super(RLNet_event, self).__init__()
        
        # Initial convolutional layers
        self.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        
        # DuRBs, a DualUpDownLayer is a DuRB_US
        self.rud1 = DualUpDownLayer(64, 64, 64, f_size=5, dilation=1, norm_type='batch_norm')
        self.rud2 = DualUpDownLayer(64, 64, 64, f_size=5, dilation=1, norm_type='batch_norm')
        self.rud3 = DualUpDownLayer(64, 64, 64, f_size=7, dilation=1, norm_type='batch_norm')
        self.rud4 = DualUpDownLayer(64, 64, 64, f_size=7, dilation=1, norm_type='batch_norm')
        self.rud5 = DualUpDownLayer(64, 64, 64, f_size=11, dilation=1, norm_type='batch_norm')
        self.rud6 = DualUpDownLayer(64, 64, 64, f_size=11, dilation=1, norm_type='batch_norm')
        self.rud7 = DualUpDownLayer(64, 64, 64, f_size=11, dilation=1, norm_type='batch_norm')
        # self.rud8 = DualUpDownLayer(64, 64, 64, f_size=11, dilation=1, norm_type='batch_norm')
        # self.rud9 = DualUpDownLayer(64, 64, 64, f_size=11, dilation=1, norm_type='batch_norm')
        # self.rud10 = DualUpDownLayer(64, 64, 64, f_size=11, dilation=1, norm_type='batch_norm')
        # self.rud11 = DualUpDownLayer(64, 64, 64, f_size=11, dilation=1, norm_type='batch_norm')
        # self.rud12 = DualUpDownLayer(64, 64, 64, f_size=11, dilation=1, norm_type='batch_norm')

        # Last layers
        # -- Up1 --
        self.upconv1 = ConvLayer(64, 256, kernel_size=3, stride=1)
        self.upsamp1 = nn.PixelShuffle(2)
        #----------
        self.conv4 = ConvLayer(64, 64, kernel_size=3, stride=1)

        # -- Up2 --
        self.upconv2 = ConvLayer(64, 256, kernel_size=3, stride=1)
        self.upsamp2 = nn.PixelShuffle(2)        
        #----------
        self.conv5 = ConvLayer(64, 64, kernel_size=3, stride=1)

        self.end_conv = nn.Conv2d(64, 5, kernel_size=7, stride=1, padding=3)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()        
        self.last_conv = nn.Sequential(nn.Conv2d(5, 3, 3, 1, 1), nn.LeakyReLU(0.2))
        
    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        res = x
        x = self.relu(self.conv3(x))

        x, res = self.rud1(x, res)
        x, res = self.rud2(x, res)
        x, res = self.rud3(x, res)
        x, res = self.rud4(x, res)
        x, res = self.rud5(x, res)
        x, res = self.rud6(x, res)
        x, res = self.rud7(x, res)
        # x, res = self.rud8(x, res)
        # x, res = self.rud9(x, res)
        # x, res = self.rud10(x, res)
        # x, res = self.rud11(x, res)
        # x, res = self.rud12(x, res)

        x = self.upconv1(x)
        x = self.upsamp1(x)
        x = self.relu(self.conv4(x))

        x = self.upconv2(x)
        x = self.upsamp2(x)        
        x = self.relu(self.conv5(x))

        x = self.tanh(self.end_conv(x))
        x = x + residual
        x = self.last_conv(x)
        return x

# my_model = DuRN()
# _input = torch.zeros(2,3,32,32)
# _output = my_model(_input)
# print('_input=',_input.shape)
# print('_output=',_output.shape)