import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

##############################################################
##############################################################
##############################################################

class _UpProjection(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, use_bn=False):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

        self.use_bn = use_bn

    def forward(self, x, size):
        x = F.upsample(x, size=size, mode='bilinear', align_corners=True)

        if not self.use_bn:
            x_conv1 = self.relu(self.conv1(x))
            bran1 = self.conv1_2(x_conv1)
            bran2 = self.conv2(x)
        else:
            x_conv1 = self.relu(self.bn1(self.conv1(x)))
            bran1 = self.bn1_2(self.conv1_2(x_conv1))
            bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)

        return out


class E_resnet(nn.Module):

    def __init__(self, original_model, num_features=2048, use_bn=False):
        super(E_resnet, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

        self.use_bn = use_bn

    def forward(self, x):
        x = self.conv1(x)

        if self.use_bn:
            x = self.bn1(x)

        x = self.relu(x)
        x_block0 = x
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block0, x_block1, x_block2, x_block3, x_block4


class E_densenet(nn.Module):

    def __init__(self, original_model, num_features=2208):
        super(E_densenet, self).__init__()
        self.features = original_model.features

    def forward(self, x):
        x01 = self.features[0](x)
        x02 = self.features[1](x01)
        x03 = self.features[2](x02)
        x04 = self.features[3](x03)

        x_block1 = self.features[4](x04)
        x_block1 = self.features[5][0](x_block1)
        x_block1 = self.features[5][1](x_block1)
        x_block1 = self.features[5][2](x_block1)
        x_tran1 = self.features[5][3](x_block1)

        x_block2 = self.features[6](x_tran1)
        x_block2 = self.features[7][0](x_block2)
        x_block2 = self.features[7][1](x_block2)
        x_block2 = self.features[7][2](x_block2)
        x_tran2 = self.features[7][3](x_block2)

        x_block3 = self.features[8](x_tran2)
        x_block3 = self.features[9][0](x_block3)
        x_block3 = self.features[9][1](x_block3)
        x_block3 = self.features[9][2](x_block3)
        x_tran3 = self.features[9][3](x_block3)

        x_block4 = self.features[10](x_tran3)
        x_block4 = F.relu(self.features[11](x_block4))

        x_block0 = x03

        return x_block0, x_block1, x_block2, x_block3, x_block4


class E_senet(nn.Module):

    def __init__(self, original_model, num_features=2048):
        super(E_senet, self).__init__()
        self.base = nn.Sequential(*list(original_model.children())[:-3])

    def forward(self, x):
        x = self.base[0](x)
        x_block1 = self.base[1](x)
        x_block2 = self.base[2](x_block1)
        x_block3 = self.base[3](x_block2)
        x_block4 = self.base[4](x_block3)

        return x_block1, x_block2, x_block3, x_block4


class D_densenet(nn.Module):
    def __init__(self, num_features=2048, use_bn=False):
        super(D_densenet, self).__init__()
        # self.conv = nn.Conv2d(num_features, num_features //
        #                       2, kernel_size=1, stride=1, bias=False)
        # num_features = num_features // 2

        self.conv = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, bias=False)
        # num_features = num_features

        self.bn = nn.BatchNorm2d(num_features)

        self.up1 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)
        num_features = num_features // 2

        self.up2 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)
        num_features = num_features // 2

        self.up3 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)
        num_features = num_features // 2

        self.up4 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)
        num_features = num_features // 2

        self.up5 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)

        self.use_bn = use_bn

    def forward(self, x_block0, x_block1, x_block2, x_block3, x_block4):
        if self.use_bn:
            x_d0 = F.relu(self.bn(self.conv(x_block4))) + x_block4
        else:
            x_d0 = F.relu(self.conv(x_block4)) + x_block4

        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)]) + x_block3
        x_d2 = self.up2(x_d1, [x_block2.size(2), x_block2.size(3)]) + x_block2
        x_d3 = self.up3(x_d2, [x_block1.size(2), x_block1.size(3)]) + x_block1
        x_d4 = self.up4(x_d3, [x_block0.size(2), x_block0.size(3)]) + x_block0
        x_d5 = self.up5(x_d4, [x_block0.size(2) * 2, x_block0.size(3) * 2])

        return x_d5


class D_resnet(nn.Module):

    def __init__(self, num_features=2048, use_bn=False):
        super(D_resnet, self).__init__()
        # self.conv = nn.Conv2d(num_features, num_features //
        #                       2, kernel_size=1, stride=1, bias=False)
        # num_features = num_features // 2

        self.conv = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, bias=False)
        # num_features = num_features

        self.bn = nn.BatchNorm2d(num_features)

        self.up1 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)
        num_features = num_features // 2

        self.up2 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)
        num_features = num_features // 2

        self.up3 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)
        num_features = num_features // 2

        self.up4 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features, use_bn=use_bn)

        self.up5 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 4)

        self.use_bn = use_bn

    def forward(self, x_block0, x_block1, x_block2, x_block3, x_block4):
        if self.use_bn:
            x_d0 = F.relu(self.bn(self.conv(x_block4))) + x_block4
        else:
            x_d0 = F.relu(self.conv(x_block4)) + x_block4

        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)]) + x_block3
        x_d2 = self.up2(x_d1, [x_block2.size(2), x_block2.size(3)]) + x_block2
        x_d3 = self.up3(x_d2, [x_block1.size(2), x_block1.size(3)]) + x_block1
        x_d4 = self.up4(x_d3, [x_block0.size(2), x_block0.size(3)]) + x_block0
        x_d5 = self.up5(x_d4, [x_block0.size(2) * 2, x_block0.size(3) * 2])

        return x_d5


class MFF(nn.Module):

    def __init__(self, block_channel, num_features=64):
        super(MFF, self).__init__()

        self.up1 = _UpProjection(
            num_input_features=block_channel[0], num_output_features=16)

        self.up2 = _UpProjection(
            num_input_features=block_channel[1], num_output_features=16)

        self.up3 = _UpProjection(
            num_input_features=block_channel[2], num_output_features=16)

        self.up4 = _UpProjection(
            num_input_features=block_channel[3], num_output_features=16)

        self.conv = nn.Conv2d(
            num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x_block1, x_block2, x_block3, x_block4, size, use_bn=False):
        x_m1 = self.up1(x_block1, size)
        x_m2 = self.up2(x_block2, size)
        x_m3 = self.up3(x_block3, size)
        x_m4 = self.up4(x_block4, size)

        if use_bn:
            x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        else:
            x = self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1))
        x = F.relu(x)

        return x

##############################################################
##############################################################
##############################################################

def maps_2_cubes(x, b, d):
    x_b, x_c, x_h, x_w = x.shape
    x = x.contiguous().view(b, d, x_c, x_h, x_w)

    return x.permute(0, 2, 1, 3, 4)


def maps_2_maps(x, b, d):
    x_b, x_c, x_h, x_w = x.shape
    x = x.contiguous().view(b, d * x_c, x_h, x_w)

    return x


class R(nn.Module):
    def __init__(self, block_channel):
        super(R, self).__init__()

        num_features = 64 + block_channel[3] // 32
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        h = F.relu(x1)

        pred_depth = self.conv2(h)

        return h, pred_depth


class R_2(nn.Module):
    def __init__(self, block_channel):
        super(R_2, self).__init__()

        num_features = 64 + block_channel[3] // 32 + 4
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

        self.convh = nn.Conv2d(
            num_features, 4, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        h = self.convh(x1)
        pred_depth = self.conv2(x1)

        return h, pred_depth



class R_3(nn.Module):
    def __init__(self, block_channel, use_bn=False):
        super(R_3, self).__init__()

        num_features = 64 + block_channel[3] // 32 + 8
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        # self.conv2 = nn.Conv2d(
        #     num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

        self.conv2 = nn.Conv2d(
            num_features, 3, kernel_size=5, stride=1, padding=2, bias=True)

        self.convh = nn.Conv2d(
            num_features, 8, kernel_size=3, stride=1, padding=1, bias=True)

        self.use_bn = use_bn

    def forward(self, x):

        x0 = self.conv0(x)
        if self.use_bn:
            x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        if self.use_bn:
            x1 = self.bn1(x1)
        x1 = F.relu(x1)

        h = self.convh(x1)
        out = self.conv2(x1)

        return h, out


class R_CLSTM_5(nn.Module):
    def __init__(self, block_channel, use_bn=False):
        super(R_CLSTM_5, self).__init__()
        num_features = 64 + block_channel[3] // 32
        self.Refine = R_3(block_channel, use_bn=use_bn)
        self.F_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.I_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.C_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Tanh()
        )
        self.Q_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )

    def forward(self, input_tensor, b, d):
        input_tensor = maps_2_cubes(input_tensor, b, d)
        b, c, d, h, w = input_tensor.shape
        h_state_init = torch.zeros(b, 8, h, w).to('cuda')
        c_state_init = torch.zeros(b, 8, h, w).to('cuda')

        seq_len = d

        h_state, c_state = h_state_init, c_state_init
        output_inner = []
        for t in range(seq_len):
            input_cat = torch.cat((input_tensor[:, :, t, :, :], h_state), dim=1)
            c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)

            h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))

            output_inner.append(p_depth)

        layer_output = torch.stack(output_inner, dim=2)

        return layer_output


class R_4(nn.Module):
    def __init__(self, block_channel, use_bn=True):
        super(R_4, self).__init__()

        num_features = (64 + block_channel[3] // 32) * 1
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 3, kernel_size=5, stride=1, padding=2, bias=True)

        self.use_bn = use_bn

    def forward(self, x):

        # first conv layer (-> bn -> relu)
        x0 = self.conv0(x) + x
        if self.use_bn:
            x0 = self.bn0(x0)
        x0 = F.relu(x0)

        # second conv layer (-> bn -> relu)
        x1 = self.conv1(x0) + x0
        if self.use_bn:
            x1 = self.bn1(x1)
        x1 = F.relu(x1)

        # output conv layer
        out = self.conv2(x1)

        return out


class R_CLSTM_6(nn.Module):
    def __init__(self, block_channel, use_bn=True):
        super(R_CLSTM_6, self).__init__()
        num_features = 64 + block_channel[3] // 32
        self.num_features = num_features
        self.Refine = R_4(block_channel, use_bn=use_bn)
        self.F_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + num_features,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.I_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + num_features,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.C_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + num_features,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Tanh()
        )
        self.Q_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + num_features,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )

    def forward(self, input_tensor, b, d):
        input_tensor = maps_2_cubes(input_tensor, b, d)
        b, c, d, h, w = input_tensor.shape
        h_state_init = torch.zeros(b, self.num_features, h, w).to('cuda')
        c_state_init = torch.zeros(b, self.num_features, h, w).to('cuda')

        seq_len = d

        h_state, c_state = h_state_init, c_state_init
        output_inner = []
        for t in range(seq_len):
            input_cat = torch.cat((input_tensor[:, :, t, :, :], h_state), dim=1)
            c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)

            h_state = torch.tanh(c_state) * self.Q_t(input_cat)

            # output image
            o_state = self.Refine(h_state)
            output_inner.append(o_state)

        layer_output = torch.stack(output_inner, dim=2)

        return layer_output

refinenet_dict = {
    'R_CLSTM_5': R_CLSTM_5,
    'R_CLSTM_6': R_CLSTM_6
}
##############################################################
##############################################################
##############################################################

def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr



class ShortcutBlock_ZKH(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule1, submodule2):
        super(ShortcutBlock_ZKH, self).__init__()
        self.sub1 = submodule1
        self.sub2 = submodule2

    def forward(self, x):
        output = x + self.sub2(self.sub1(x))
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub1.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr



def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


####################
# Useful blocks
####################


class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class ResidualDenseBlock_5C(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc+4*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


####################
# Upsampler
####################


def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=None, act_type='relu'):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)



##############################################################
##############################################################
##############################################################
def cubes_2_maps(cubes):
    b, c, d, h, w = cubes.shape
    cubes = cubes.permute(0, 2, 1, 3, 4)

    return cubes.contiguous().view(b*d, c, h, w), b, d


def maps_2_cubes(maps, b, d):
    bd, c, h, w = maps.shape
    cubes = maps.contiguous().view(b, d, c, h, w)

    return cubes.permute(0, 2, 1, 3, 4)


def rev_maps(maps, b, d):
    """reverse maps temporarily."""
    cubes = maps_2_cubes(maps, b, d).flip(dims=[2])

    return cubes_2_maps(cubes)[0]


class CoarseNet(nn.Module):
    def __init__(self, encoder, decoder, block_channel, refinenet, bidirectional=False, input_residue=False):

        super(CoarseNet, self).__init__()

        self.use_bidirect = bidirectional
        self.input_residue = input_residue

        self.E = encoder
        self.D = decoder
        self.MFF = MFF(block_channel)

        self.R_fwd = refinenet_dict[refinenet](block_channel)

        if self.use_bidirect:
            self.R_bwd = refinenet_dict[refinenet](block_channel)
            self.bidirection_fusion = nn.Conv2d(6, 3, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x):
        x_cube = x

        x, b, d = cubes_2_maps(x)
        x_block0, x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block0, x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
        fwd_out = self.R_fwd(torch.cat((x_decoder, x_mff), 1), b, d)

        if self.use_bidirect:
            bwd_out = self.R_bwd(torch.cat((rev_maps(x_decoder, b, d),
                                            rev_maps(x_mff, b, d)), 1),
                                 b, d)

            concat_cube = cubes_2_maps(torch.cat((fwd_out, bwd_out.flip(dims=[2])), 1))[0]

            out = maps_2_cubes(self.bidirection_fusion(concat_cube), b=b, d=d)
        else:
            out = fwd_out

        # resolve odd number pixels
        if x_cube.shape[-1] % 2 == 1:
            out = out[:, :, :, :, :-1]

        if self.input_residue:
            return out + x_cube
        else:
            return out


class C_C3D_1(nn.Module):

    def __init__(self, out_channels=64, input_channels=5):
        self.inplanes = out_channels
        super(C_C3D_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, out_channels, kernel_size=5, stride=1, padding=[1, 2, 2], bias=False),
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=[0, 1, 1], bias=False),
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        x = self.conv4(x) + x

        x = x.squeeze(2)

        return x

class ESTIL_event(nn.Module):
    def __init__(self, num_features=64, num_blocks=9, out_nc=3, mode='CNA', act_type='relu', norm_type=None):
        super(ESTIL_event, self).__init__()

        nb, nf = num_blocks, num_features

        # 3d convolution to fuse sequence.
        c3d = C_C3D_1()

        rb_blocks = [RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        HR_conv0 = conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type)
        HR_conv1 = conv_block(nf, out_nc, kernel_size=3, norm_type=norm_type, act_type=None)

        self.net = sequential(c3d, ShortcutBlock_ZKH(sequential(*rb_blocks), LR_conv), HR_conv0, HR_conv1)

    def forward(self, c_out):
        # 6-channel = 3 (rgb) model_out_C + 3 (rgb) in_videos_C
        # inp_6c_F = torch.cat((c_out, c_in), 1)
        c_out = torch.unsqueeze(c_out,dim=2)
        c_out = torch.cat((c_out,c_out,c_out,c_out,c_out),dim=2)
        # print('c_out=',c_out.shape)
        inp_6c_F = c_out
        
        result1 = self.net(inp_6c_F)
        result2 = c_out[:, 0:3, c_out.shape[2] // 2, :, :]
        # print('result1=',result1.shape)
        # print('result2=',result2.shape)
        return result1 + result2



# if __name__ == '__main__':
#     s = b, c, d, h, w = 1, 3, 5, 224, 224
#     t = torch.ones(s).cuda()

#     net = FineNet().cuda()
#     print('t.shape=',t.shape)
#     output = net(t)
#     print('output.shape=',output.shape)



# my_model = FineNet()
# _input = torch.zeros(2,5,32,32)
# _output = my_model(_input)
# print('_input=',_input.shape)
# print('_output=',_output.shape)