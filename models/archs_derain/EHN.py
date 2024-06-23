import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import slayerSNN as snn
netParams = snn.params('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/models/network.yaml')
from torchstat import stat
from thop import profile


from math import exp

import numpy as np
import torch.nn.init as init
from models.archs.arch_util import ConditionNet



def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out
    
class PAConv(nn.Module):

    def __init__(self, nf, k_size=3):

        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution

    def forward(self, x):

        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out
        
class SCPA(nn.Module):
    
    """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
        Github: https://github.com/MCG-NKU/SCNet
    """

    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(SCPA, self).__init__()
        group_width = nf // reduction
        
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        
        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        bias=False)
                    )
        
        self.PAConv = PAConv(group_width)
        
        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual

        return out


class Channel_Att(nn.Module):
    def __init__(self, reduction=16):
        super(Channel_Att, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 该情况相当于torch.mean(input)，所有的数累加起来求一个均值
        
        self.fc = nn.Sequential(
            nn.Linear(512, 40, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(40, 40, bias=False),
            nn.Sigmoid()
        )

    def forward(self, event_40C_feature,aggregrated_event):   # x=[2,40,128,128]
        
        _,event_c,_,_ = event_40C_feature.size()
        aggregrated_event = aggregrated_event[0]
        aggregrated_event = aggregrated_event.transpose(1,2)
        aggregrated_event = aggregrated_event.unsqueeze(3)
        b, c,_, _ = aggregrated_event.size()
        aggregrated_event = self.avg_pool(aggregrated_event)    # y=[2,40,1,1]
        aggregrated_event = aggregrated_event.view(b, c)
        
        
        channel_attention = self.fc(aggregrated_event)

        channel_attention = channel_attention.view(b,event_c,1,1)  # channel_attention = [2,40]    
        
        event_40C_feature = event_40C_feature * channel_attention.expand_as(event_40C_feature)
        

        return event_40C_feature

#############################################################################
#############################################################################
#correlation操作
#############################################################################
def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out



class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 3, 3, 1, 1, 1)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(3, 40, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(40, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)


        gwc_feature = x


        concat_feature = self.lastconv(gwc_feature)
        return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}




######################################################################
######################################################################
######################################################################

def getNeuronConfig(type: str='SRMALPHA',
                    theta: float=10.,
                    tauSr: float=1.,
                    tauRef: float=1.,
                    scaleRef: float=2.,
                    tauRho: float=0.3,  # Was set to 0.2 previously (e.g. for fullRes run)
                    scaleRho: float=1.):
    """
    :param type:     neuron type
    :param theta:    neuron threshold
    :param tauSr:    neuron time constant
    :param tauRef:   neuron refractory time constant
    :param scaleRef: neuron refractory response scaling (relative to theta)
    :param tauRho:   spike function derivative time constant (relative to theta)
    :param scaleRho: spike function derivative scale factor
    :return: dictionary
    """
    return {
        'type': type,
        'theta': theta,
        'tauSr': tauSr,
        'tauRef': tauRef,
        'scaleRef': scaleRef,
        'tauRho': tauRho,
        'scaleRho': scaleRho,
    }

class NetworkBasic(torch.nn.Module):
    def __init__(self, netParams,
                 theta=[30, 50, 100],
                 tauSr=[1, 2, 4],
                 tauRef=[1, 2, 4],
                 scaleRef=[1, 1, 1],
                 tauRho=[1, 1, 10],
                 scaleRho=[10, 10, 100]):
        super(NetworkBasic, self).__init__()

        self.neuron_config = []
        self.neuron_config.append(
            getNeuronConfig(theta=theta[0], tauSr=tauSr[0], tauRef=tauRef[0], scaleRef=scaleRef[0], tauRho=tauRho[0],
                            scaleRho=scaleRho[0]))
        self.neuron_config.append(
            getNeuronConfig(theta=theta[1], tauSr=tauSr[1], tauRef=tauRef[1], scaleRef=scaleRef[1], tauRho=tauRho[1],
                            scaleRho=scaleRho[1]))
        self.neuron_config.append(
            getNeuronConfig(theta=theta[2], tauSr=tauSr[2], tauRef=tauRef[2], scaleRef=scaleRef[2], tauRho=tauRho[2],
                            scaleRho=scaleRho[2]))

        self.slayer1 = snn.layer(self.neuron_config[0], netParams['simulation'])
        self.slayer2 = snn.layer(self.neuron_config[1], netParams['simulation'])
        self.slayer3 = snn.layer(self.neuron_config[2], netParams['simulation'])

        self.conv1 = self.slayer1.conv(1, 1, 5, padding=2)
        self.conv2 = self.slayer2.conv(1, 1, 3, padding=1)
        # self.upconv1 = self.slayer3.convTranspose(8, 2, kernelSize=2, stride=2)


    def forward(self, spikeInput):
        psp1 = self.slayer1.psp(spikeInput)
        spikes_layer_1 = self.slayer1.spike(self.conv1(psp1))
        # print('spikes_layer_1=',spikes_layer_1.shape)
        spikes_layer_2 = self.slayer2.spike(self.conv2(self.slayer2.psp(spikes_layer_1)))
        # print('spikes_layer_2=',spikes_layer_2.shape)
        # b, t, c, w, h = spikes_layer_2.size()    # [16,8,3,256,128] [b, t, c, w, h]
        # spikes_layer_2 = spikes_layer_2.contiguous().view(b * t, c, w, h)  # x=[128,3,256,128]
        # print('spikes_layer_2=',spikes_layer_2.shape)
        return spikes_layer_2

########################################################################
########################################################################
########################################################################



class CoupleLayer(nn.Module):
    def __init__(self, channels, substructor, condition_length,  clamp=5.):
        super().__init__()

        channels = channels
        # self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.conditional = True
        # condition_length = sub_len
        self.shadowpre = nn.Sequential(
            nn.Conv2d(4, channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.2))
        self.shadowpro = ShadowProcess(channels // 2)


        self.s1 = substructor(self.split_len1 + condition_length, self.split_len2*2)
        self.s2 = substructor(self.split_len2 + condition_length, self.split_len1*2)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, c, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))
        c_star = self.shadowpre(c)
        c_star = self.shadowpro(c_star)

        if not rev:
            # r2 = self.s2(torch.cat([x2, c_star], 1) if self.conditional else x2)
            r2 = self.s2(x2, c_star)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            # r1 = self.s1(torch.cat([y1, c_star], 1) if self.conditional else y1)
            r1 = self.s1(y1, c_star)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1
            # self.last_jac = self.log_e(s1) + self.log_e(s2)

        else: # names of x and y are swapped!
            # r1 = self.s1(torch.cat([x1, c_star], 1) if self.conditional else x1)
            r1 = self.s1(x1, c_star)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            # r2 = self.s2(torch.cat([y2, c_star], 1) if self.conditional else y2)
            r2 = self.s2(y2, c_star)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)
            # self.last_jac = - self.log_e(s1) - self.log_e(s2)

        return torch.cat((y1, y2), 1)

    # def jacobian(self, x, c=[], rev=False):
    #     return torch.sum(self.last_jac, dim=tuple(range(1, self.ndims+1)))
    def output_dims(self, input_dims):
        return input_dims




def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class ShadowProcess(nn.Module):
    def __init__(self, channels):
        super(ShadowProcess, self).__init__()
        self.process = UNetConvBlock(channels, channels)
        self.Attention = nn.Sequential(
            nn.Conv2d(channels,channels,3,1,1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.process(x)
        xatt = self.Attention(x)

        return xatt

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=16, bias=False):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc)
        self.conv2 = UNetConvBlock(gc, gc)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 1, 1, 0, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3


class MultiscaleDense(nn.Module):
    def __init__(self,channel_in, channel_out, init):
        super(MultiscaleDense, self).__init__()
        self.conv_mul = nn.Conv2d(channel_out//2,channel_out//2,3,1,1)
        self.conv_add = nn.Conv2d(channel_out//2, channel_out//2, 3, 1, 1)
        self.down1 = nn.Conv2d(channel_out//2,channel_out//2,stride=2,kernel_size=2,padding=0)
        self.down2 = nn.Conv2d(channel_out//2, channel_out//2, stride=2, kernel_size=2, padding=0)
        self.op1 = DenseBlock(channel_in, channel_out, init)
        self.op2 = DenseBlock(channel_in, channel_out, init)
        self.op3 = DenseBlock(channel_in, channel_out, init)
        self.fuse = nn.Conv2d(3 * channel_out, channel_out, 1, 1, 0)

    def forward(self, x, s):
        s_mul = self.conv_mul(s)
        s_add = self.conv_add(s)
        # x_trans = s_mul*x+s_add
        # x = torch.cat([x,x_trans],1)

        x1 = x
        x2,s_mul2,s_add2 = self.down1(x),\
                           F.interpolate(s_mul, scale_factor=0.5, mode='bilinear'),F.interpolate(s_add, scale_factor=0.5, mode='bilinear')
        x3, s_mul3, s_add3 = self.down2(x2), \
                             F.interpolate(s_mul, scale_factor=0.25, mode='bilinear'), F.interpolate(s_add,scale_factor=0.25,mode='bilinear')
        x1 = self.op1(torch.cat([x1,s_mul*x1+s_add],1))
        x2 = self.op2(torch.cat([x2,s_mul2*x2+s_add2],1))
        x3 = self.op3(torch.cat([x3,s_mul3*x3+s_add3],1))
        x2 = F.interpolate(x2, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x3 = F.interpolate(x3, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x = self.fuse(torch.cat([x1, x2, x3], 1))

        return x




def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            return MultiscaleDense(channel_in, channel_out, init)

        else:
            return None

    return constructor




class InvISPNet(nn.Module):
    def __init__(self, channel_in=3, subnet_constructor=subnet('DBNet'), block_num=4):
        super(InvISPNet, self).__init__()
        operations = []
        # level = 3
        self.condition = ConditionNet()
        for p in self.parameters():
            p.requires_grad = False

        channel_num = 16  # total channels at input stage
        self.CG0 = nn.Conv2d(channel_in, channel_num, 1, 1, 0)
        self.CG1 = nn.Conv2d(channel_num, channel_in, 1, 1, 0)
        self.CG2 = nn.Conv2d(channel_in, channel_num, 1, 1, 0)
        self.CG3 = nn.Conv2d(channel_num, channel_in, 1, 1, 0)

        for j in range(block_num):
            b = CoupleLayer(channel_num, substructor = subnet_constructor, condition_length=channel_num//2)  # one block is one flow step.
            operations.append(b)

        self.operations = nn.ModuleList(operations)
        self.initialize()


    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

    def forward(self, input, input2, rev=False):
        b, c, m, n = input.shape
        maskfea = self.condition(input)

        if not rev:
            x = input
            out = self.CG0(x)
            out_list = []
            for op in self.operations:
                out_list.append(out)
                out = op.forward(out, maskfea, rev)
            out = self.CG1(out)
        else:
            out = self.CG2(input2)
            out_list = []
            for op in reversed(self.operations):
                out = op.forward(out, maskfea, rev)
                out_list.append(out)
            out_list.reverse()
            out = self.CG3(out)
        # return out, out[:, :4, :, :]
        return out





######################################################################
######################################################################
######################################################################

class EHN(nn.Module):
    
    def __init__(self, in_nc, out_nc, nf, unf, nb, scale=4):
        super(EHN, self).__init__()
        # SCPA
        SCPA_block_f = functools.partial(SCPA, nf=nf, reduction=2)
        self.scale = scale
        
        ### first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        ### main blocks
        self.SCPA_trunk = make_layer(SCPA_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        
        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            
        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.snn = NetworkBasic(netParams=netParams).cuda()

        self.inverse = InvISPNet()

    def forward(self, x):
        
        event = x[:,3:5,:,:]
        feat = x[:,:3,:,:]
        event = event.unsqueeze(0)
        event = event.cuda()
        event = self.snn(event)

        
        event = event.squeeze()
        # print('event_out=',event.shape)
  
        x = self.inverse(feat,event)
        
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.SCPA_trunk(fea))
        fea = fea + trunk
        
        
        out = self.conv_last(fea)
        

        return out

