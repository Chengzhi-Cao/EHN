import torch
from torch import nn
import torch.nn.functional as F


#######################  网络参数设置　　############################################################
channel = 20
Num_encoder = 16
ssim_loss = True
uint = "GRU"   #'RNN','GRU','LSTM'
cross_scale = True  # block
Net_cross = True   # network
single = False
conv_num = 4
scale_num = 4 
aug_data = True # Set as False for fair comparison
patch_size = 64
pic_is_pair = True  #input picture is pair or single
lr = 0.0005
########################################################################################


class Inner_scale_connection_block(nn.Module):
    def __init__(self):
        super(Inner_scale_connection_block, self).__init__()
        self.channel = channel
        self.scale_num = scale_num
        self.conv_num = conv_num
        self.scale1 = nn.ModuleList()
        self.scale2 = nn.ModuleList()
        self.scale4 = nn.ModuleList()
        self.scale8 = nn.ModuleList()
        if scale_num == 4:
            for i in range(self.conv_num):
                self.scale1.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
                self.scale2.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
                self.scale4.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
                self.scale8.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
            self.fusion84 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
            self.fusion42 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
            self.fusion21 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
            self.pooling8 = nn.MaxPool2d(8, 8)
            self.pooling4 = nn.MaxPool2d(4, 4)
            self.pooling2 = nn.MaxPool2d(2, 2)
            self.fusion_all = nn.Sequential(nn.Conv2d(4 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        elif scale_num == 3:
            for i in range(self.conv_num):
                self.scale1.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
                self.scale2.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
                self.scale4.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
            self.fusion42 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
            self.fusion21 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
            self.pooling4 = nn.MaxPool2d(4, 4)
            self.pooling2 = nn.MaxPool2d(2, 2)
            self.fusion_all = nn.Sequential(nn.Conv2d(3 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        elif scale_num == 2:
            for i in range(self.conv_num):
                self.scale1.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
                self.scale2.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
            self.fusion21 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
            self.pooling2 = nn.MaxPool2d(2, 2)
            self.fusion_all = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        elif scale_num == 1:
            for i in range(self.conv_num):
                self.scale1.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))

    def forward(self, x):
        if scale_num == 4:
            feature8 = self.pooling8(x)
            b8, c8, h8, w8 = feature8.size()
            feature4 = self.pooling4(x)
            b4, c4, h4, w4 = feature4.size()
            feature2 = self.pooling2(x)
            b2, c2, h2, w2 = feature2.size()
            feature1 = x
            b1, c1, h1, w1 = feature1.size()
            for i in range(self.conv_num):
                feature8 = self.scale8[i](feature8)
            scale8 = feature8
            feature4 = self.fusion84(torch.cat([feature4, F.upsample(scale8, [h4, w4])], dim=1))
            for i in range(self.conv_num):
                feature4 = self.scale4[i](feature4)
            scale4 = feature4
            feature2 = self.fusion42(torch.cat([feature2, F.upsample(scale4, [h2, w2])], dim=1))
            for i in range(self.conv_num):
                feature2 = self.scale2[i](feature2)

            scale2 = feature2
            feature1 = self.fusion21(torch.cat([feature1, F.upsample(scale2, [h1, w1])], dim=1))
            for i in range(self.conv_num):
                feature1 = self.scale1[i](feature1)
            scale1 = feature1
            fusion_all = self.fusion_all(torch.cat([scale1, F.upsample(scale2, [h1, w1]), F.upsample(scale4, [h1, w1]), F.upsample(scale8, [h1, w1])], dim=1))
            return fusion_all + x
        elif scale_num == 3:
            feature4 = self.pooling4(x)
            b4, c4, h4, w4 = feature4.size()
            feature2 = self.pooling2(x)
            b2, c2, h2, w2 = feature2.size()
            feature1 = x
            b1, c1, h1, w1 = feature1.size()

            for i in range(self.conv_num):
                feature4 = self.scale4[i](feature4)
            scale4 = feature4
            feature2 = self.fusion42(torch.cat([feature2, F.upsample(scale4, [h2, w2])], dim=1))
            for i in range(self.conv_num):
                feature2 = self.scale2[i](feature2)
            scale2 = feature2
            feature1 = self.fusion21(torch.cat([feature1, F.upsample(scale2, [h1, w1])], dim=1))
            for i in range(self.conv_num):
                feature1 = self.scale1[i](feature1)
            scale1 = feature1
            fusion_all = self.fusion_all(torch.cat([scale1, F.upsample(scale2, [h1, w1]), F.upsample(scale4, [h1, w1])],dim=1))
            return fusion_all + x
        elif scale_num == 2:
            feature2 = self.pooling2(x)
            b2, c2, h2, w2 = feature2.size()
            feature1 = x
            b1, c1, h1, w1 = feature1.size()

            for i in range(self.conv_num):
                feature2 = self.scale2[i](feature2)
            scale2 = feature2
            feature1 = self.fusion21(torch.cat([feature1, F.upsample(scale2, [h1, w1])], dim=1))
            for i in range(self.conv_num):
                feature1 = self.scale1[i](feature1)
            scale1 = feature1
            fusion_all = self.fusion_all(
                torch.cat([scale1, F.upsample(scale2, [h1, w1])], dim=1))
            return fusion_all + x
        elif scale_num == 1:
            feature1 = x
            b1, c1, h1, w1 = feature1.size()
            scale1 = self.scale1(feature1)
            fusion_all = scale1
            return fusion_all + x
            
class Cross_scale_fusion_block(nn.Module):
    def __init__(self):
        super(Cross_scale_fusion_block, self).__init__()
        self.channel_num = channel
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion3_1 = nn.Sequential(
            nn.Conv2d(3 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion3_2 = nn.Sequential(
            nn.Conv2d(3 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion3_3 = nn.Sequential(
            nn.Conv2d(3 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion2_1 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion2_2 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion2_3 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.pooling2 = nn.MaxPool2d(2, 2)
        self.pooling4 = nn.MaxPool2d(4, 4)

    def forward(self, x):
        input = x
        encoder1 = self.encoder_conv1(x)
        b1, c1, h1, w1 = encoder1.size()
        pooling1 = self.pooling2(encoder1)
        encoder2 = self.encoder_conv2(pooling1)
        b2, c2, h2, w2 = encoder2.size()
        pooling2 = self.pooling2(encoder2)
        encoder3 = self.encoder_conv3(pooling2)
        b3, c3, h3, w3 = encoder3.size()
        encoder1_resize1 = F.upsample(encoder1, [h3, w3])
        encoder2_resize1 = F.upsample(encoder2, [h3, w3])
        encoder3_resize1 = F.upsample(encoder3, [h3, w3])
        fusion3_1 = self.fusion3_1(torch.cat([encoder1_resize1, encoder2_resize1, encoder3_resize1], dim=1))
        encoder1_resize2 = F.upsample(encoder1, [h2, w2])
        encoder2_resize2 = F.upsample(encoder2, [h2, w2])
        encoder3_resize2 = F.upsample(encoder3, [h2, w2])
        fusion3_2 = self.fusion3_2(torch.cat([encoder1_resize2, encoder2_resize2, encoder3_resize2], dim=1))
        encoder1_resize3 = F.upsample(encoder1, [h1, w1])
        encoder2_resize3 = F.upsample(encoder2, [h1, w1])
        encoder3_resize3 = F.upsample(encoder3, [h1, w1])
        fusion3_3 = self.fusion3_3(torch.cat([encoder1_resize3, encoder2_resize3, encoder3_resize3], dim=1))

        decoder_conv1 = self.decoder_conv1(self.fusion2_1(torch.cat([fusion3_1, F.upsample(encoder3, [h3, w3])], dim=1)))
        decoder_conv2 = self.decoder_conv2(self.fusion2_2(torch.cat([fusion3_2, F.upsample(decoder_conv1, [h2, w2])], dim=1)))
        decoder_conv3 = self.decoder_conv3(self.fusion2_3(torch.cat([fusion3_3, F.upsample(decoder_conv2, [h1, w1])], dim=1)))
        return decoder_conv3 + input


class Encoder_decoder_block(nn.Module):
    def __init__(self):
        super(Encoder_decoder_block, self).__init__()
        self.channel_num = channel
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion2_1 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion2_2 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion2_3 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.pooling2 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        input = x
        encoder1 = self.encoder_conv1(x)
        b1, c1, h1, w1 = encoder1.size()
        pooling1 = self.pooling2(encoder1)
        encoder2 = self.encoder_conv2(pooling1)
        b2, c2, h2, w2 = encoder2.size()
        pooling2 = self.pooling2(encoder2)
        encoder3 = self.encoder_conv3(pooling2)

        decoder_conv1 = self.decoder_conv1(encoder3)
        decoder_conv2 = self.decoder_conv2(self.fusion2_2(torch.cat([F.upsample(encoder2, [h2, w2]), F.upsample(decoder_conv1, [h2, w2])], dim=1)))
        decoder_conv3 = self.decoder_conv3(self.fusion2_3(torch.cat([F.upsample(encoder1, [h1, w1]), F.upsample(decoder_conv2, [h1, w1])], dim=1)))
        return decoder_conv3 + input


Scale_block = Inner_scale_connection_block


class ConvDirec(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad = int(dilation * (kernel - 1) / 2)
        self.conv = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad, dilation=dilation)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, h=None):
        x = self.conv(x)
        x = self.relu(x)
        return x, None


class ConvRNN(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel - 1) / 2)
        self.conv_x = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)

        pad_h = int((kernel - 1) / 2)
        self.conv_h = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, h=None):
        if h is None:
            h = F.tanh(self.conv_x(x))
        else:
            h = F.tanh(self.conv_x(x) + self.conv_h(h))

        h = self.relu(h)
        return h, h


class ConvGRU(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel - 1) / 2)
        self.conv_xz = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xr = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xn = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)

        pad_h = int((kernel - 1) / 2)
        self.conv_hz = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hr = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hn = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, h=None):
        if h is None:
            z = F.sigmoid(self.conv_xz(x))
            f = F.tanh(self.conv_xn(x))
            h = z * f
        else:
            z = F.sigmoid(self.conv_xz(x) + self.conv_hz(h))
            r = F.sigmoid(self.conv_xr(x) + self.conv_hr(h))
            n = F.tanh(self.conv_xn(x) + self.conv_hn(r * h))
            h = (1 - z) * h + z * n

        h = self.relu(h)
        return h, h


class ConvLSTM(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel - 1) / 2)
        self.conv_xf = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xi = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xo = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xj = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)

        pad_h = int((kernel - 1) / 2)
        self.conv_hf = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hi = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_ho = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hj = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, pair=None):
        if pair is None:
            i = F.sigmoid(self.conv_xi(x))
            o = F.sigmoid(self.conv_xo(x))
            j = F.tanh(self.conv_xj(x))
            c = i * j
            h = o * c
        else:
            h, c = pair
            f = F.sigmoid(self.conv_xf(x) + self.conv_hf(h))
            i = F.sigmoid(self.conv_xi(x) + self.conv_hi(h))
            o = F.sigmoid(self.conv_xo(x) + self.conv_ho(h))
            j = F.tanh(self.conv_xj(x) + self.conv_hj(h))
            c = f * c + i * j
            h = o * F.tanh(c)

        h = self.relu(h)
        return h, [h, c]


RecUnit = {
    'Conv': ConvDirec,
    'RNN': ConvRNN,
    'GRU': ConvGRU,
    'LSTM': ConvLSTM,
}[uint]


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.unit_num = Num_encoder
        self.units = nn.ModuleList()
        self.channel_num = channel
        self.conv1x1 = nn.ModuleList()
        for i in range(self.unit_num):
            self.units.append(Scale_block())
            self.conv1x1.append(nn.Sequential(nn.Conv2d((i + 2) * self.channel_num, self.channel_num, 1, 1), nn.LeakyReLU(0.2)))

    def forward(self, x):
        catcompact = []
        catcompact.append(x)
        feature = []
        out = x
        for i in range(self.unit_num):
            tmp = self.units[i](out)
            feature.append(tmp)
            catcompact.append(tmp)
            out = self.conv1x1[i](torch.cat(catcompact, dim=1))
        return out, feature


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.unit_num = Num_encoder
        self.units = nn.ModuleList()
        self.channel_num = channel
        self.conv1x1 = nn.ModuleList()
        for i in range(self.unit_num):
            self.units.append(Scale_block())
            self.conv1x1.append(nn.Sequential(nn.Conv2d((i + 2) * self.channel_num, self.channel_num, 1, 1), nn.LeakyReLU(0.2)))

    def forward(self, x, feature):
        catcompact=[]
        catcompact.append(x)
        out = x
        for i in range(self.unit_num):
            tmp = self.units[i](out + feature[i])
            catcompact.append(tmp)
            out = self.conv1x1[i](torch.cat(catcompact, dim=1))
        return out


class DenseConnection(nn.Module):
    def __init__(self, unit, unit_num):
        super(DenseConnection, self).__init__()
        self.unit_num = unit_num
        self.channel = channel
        self.units = nn.ModuleList()
        self.conv1x1 = nn.ModuleList()
        for i in range(self.unit_num):
            self.units.append(unit())
            self.conv1x1.append(nn.Sequential(nn.Conv2d((i + 2) * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2)))

    def forward(self, x):
        cat = []
        cat.append(x)
        out = x
        for i in range(self.unit_num):
            tmp = self.units[i](out)
            cat.append(tmp)
            out = self.conv1x1[i](torch.cat(cat, dim=1))
        return out


class DCSFN_event(nn.Module):
    def __init__(self,in_nc,out_nc):
        super(DCSFN_event, self).__init__()
        self.channel = channel
        self.unit_num = channel
        self.enterBlock = nn.Sequential(nn.Conv2d(5, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.derain_net = DenseConnection(Cross_scale_fusion_block, self.unit_num)
        self.exitBlock = nn.Sequential(nn.Conv2d(self.channel, 5, 3, 1, 1), nn.LeakyReLU(0.2))

        self.last_conv = nn.Sequential(nn.Conv2d(5, 3, 3, 1, 1), nn.LeakyReLU(0.2))

    def forward(self, x):
        image_feature = self.enterBlock(x)
        rain_feature = self.derain_net(image_feature)
        rain = self.exitBlock(rain_feature)
        derain = x - rain
        derain = self.last_conv(derain)
        return derain


# my_model = DCSFN()
# _input = torch.zeros(2,3,50,50)
# _output = my_model(_input)
# print('_input=',_input.shape)
# print('_output=',_output.shape)