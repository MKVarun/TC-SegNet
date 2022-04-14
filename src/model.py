import torch
import torch.nn as nn
from torch.nn import functional as F
############################################################### Aux Functions ############################################################

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True))
        
    def forward(self, x):
        x = self.conv(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=3):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        

class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

        
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
        
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel, stride=stride)

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2




#################################################### UNET ########################################################
start_fm = 16

class Unet(nn.Module):
    
    def __init__(self, start_fm):
        super(Unet, self).__init__()    
        
        #(Double) Convolution 1        
        self.double_conv1 = double_conv(1, start_fm, 3, 1, 1)
        #Max Pooling 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        #Convolution 2
        self.double_conv2 = double_conv(start_fm, start_fm * 2, 3, 1, 1)
        #Max Pooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        #Convolution 3
        self.double_conv3 = double_conv(start_fm * 2, start_fm * 4, 3, 1, 1)
        #Max Pooling 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        
        #Convolution 4
        self.double_conv4 = double_conv(start_fm * 4, start_fm * 8, 3, 1, 1)
        #Max Pooling 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        
        #Convolution 5
        self.double_conv5 = double_conv(start_fm * 8, start_fm * 16, 3, 1, 1)
        
        #Transposed Convolution 4
        self.t_conv4 = nn.ConvTranspose2d(start_fm * 16, start_fm * 8, 2, 2)
        # Expanding Path Convolution 4 
        self.ex_double_conv4 = double_conv(start_fm * 16, start_fm * 8, 3, 1, 1)
        
        #Transposed Convolution 3
        self.t_conv3 = nn.ConvTranspose2d(start_fm * 8, start_fm * 4, 2, 2)
        #Convolution 3
        self.ex_double_conv3 = double_conv(start_fm * 8, start_fm * 4, 3, 1, 1)
        
        #Transposed Convolution 2
        self.t_conv2 = nn.ConvTranspose2d(start_fm * 4, start_fm * 2, 2, 2)
        #Convolution 2
        self.ex_double_conv2 = double_conv(start_fm * 4, start_fm * 2, 3, 1, 1)
        
        #Transposed Convolution 1
        self.t_conv1 = nn.ConvTranspose2d(start_fm * 2, start_fm, 2, 2)
        #Convolution 1
        self.ex_double_conv1 = double_conv(start_fm * 2, start_fm, 3, 1, 1)
        
        # One by One Conv
        self.one_by_one = nn.Conv2d(start_fm, 4, 1, 1, 0)
        #self.final_act = nn.Sigmoid()
        
        
    def forward(self, inputs):
        # Contracting Path
        conv1 = self.double_conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.double_conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.double_conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.double_conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
            
        # Bottom
        conv5 = self.double_conv5(maxpool4)
        
        # Expanding Path
        t_conv4 = self.t_conv4(conv5)
        cat4 = torch.cat([conv4 ,t_conv4], 1)
        ex_conv4 = self.ex_double_conv4(cat4)
        
        t_conv3 = self.t_conv3(ex_conv4)
        cat3 = torch.cat([conv3 ,t_conv3], 1)
        ex_conv3 = self.ex_double_conv3(cat3)

        t_conv2 = self.t_conv2(ex_conv3)
        cat2 = torch.cat([conv2 ,t_conv2], 1)
        ex_conv2 = self.ex_double_conv2(cat2)
        
        t_conv1 = self.t_conv1(ex_conv2)
        cat1 = torch.cat([conv1 ,t_conv1], 1)
        ex_conv1 = self.ex_double_conv1(cat1)
        
        one_by_one = self.one_by_one(ex_conv1)
        
        return one_by_one


#################################################################### Attention UNet #######################################################################

# https://github.com/LeeJunHyun/Image_Segmentation/blob/db34de21767859e035aee143c59954fa0d94bbcd/network.py#L46
class Attn_Unet(nn.Module):
    def __init__(self, start_fm):
        super(Attn_Unet,self).__init__()
        
        self.double_conv1 = double_conv(1, start_fm, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.double_conv2 = double_conv(start_fm, start_fm * 2, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.double_conv3 = double_conv(start_fm * 2, start_fm * 4, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        
        self.double_conv4 = double_conv(start_fm * 4, start_fm * 8, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        
        self.double_conv5 = double_conv(start_fm * 8, start_fm * 16, 3, 1, 1)

        # Upconvolution

        self.t_conv4 = nn.ConvTranspose2d(start_fm * 16, start_fm * 8, 2, 2)
        self.Att4 = Attention_block(F_g=start_fm * 8,F_l=start_fm * 8,F_int=start_fm * 4)
        self.ex_double_conv4 = double_conv(start_fm * 16, start_fm * 8, 3, 1, 1)

        self.t_conv3 = nn.ConvTranspose2d(start_fm * 8, start_fm * 4, 2, 2)
        self.Att3 = Attention_block(F_g=start_fm * 4,F_l=start_fm * 4,F_int=start_fm * 2)
        self.ex_double_conv3 = double_conv(start_fm * 8, start_fm * 4, 3, 1, 1)

        self.t_conv2 = nn.ConvTranspose2d(start_fm * 4, start_fm * 2, 2, 2)
        self.Att2 = Attention_block(F_g=start_fm * 2, F_l=start_fm * 2, F_int=start_fm)
        self.ex_double_conv2 = double_conv(start_fm * 4, start_fm * 2, 3, 1, 1)

        self.t_conv1 = nn.ConvTranspose2d(start_fm * 2, start_fm, 2, 2)
        self.Att1 = Attention_block(F_g=start_fm, F_l=start_fm, F_int=start_fm//2)
        self.ex_double_conv1 = double_conv(start_fm * 2, start_fm, 3, 1, 1)

        self.one_by_one = nn.Conv2d(start_fm, 4, 1, 1, 0)

    def forward(self,inputs):
        # encoding path
        conv1 = self.double_conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.double_conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.double_conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.double_conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
            
        conv5 = self.double_conv5(maxpool4)

        t_conv4 = self.t_conv4(conv5)
        conv4 = self.Att4(t_conv4,conv4)
        cat4 = torch.cat([conv4 ,t_conv4], 1)
        ex_conv4 = self.ex_double_conv4(cat4)

        t_conv3 = self.t_conv3(ex_conv4)
        conv3 = self.Att3(t_conv3,conv3)
        cat3 = torch.cat([conv3 ,t_conv3], 1)
        ex_conv3 = self.ex_double_conv3(cat3)

        t_conv2 = self.t_conv2(ex_conv3)
        conv2 = self.Att2(t_conv2,conv2)
        cat2 = torch.cat([conv2 ,t_conv2], 1)
        ex_conv2 = self.ex_double_conv2(cat2)

        t_conv1 = self.t_conv1(ex_conv2)
        conv1 = self.Att1(t_conv1,conv1)
        cat1 = torch.cat([conv1 ,t_conv1], 1)
        ex_conv1 = self.ex_double_conv1(cat1)

        one_by_one = self.one_by_one(ex_conv1)

        return one_by_one


############################################# Recurrent UNet ###############################################################

class R2U_Net(nn.Module):
    def __init__(self, start_fm, t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.RRCNN1 = RRCNN_block(1,start_fm,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=start_fm,ch_out=start_fm*2,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=start_fm*2,ch_out=start_fm*4,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=start_fm*4,ch_out=start_fm*8,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=start_fm*8,ch_out=start_fm*16,t=t)


        self.t_conv4 = nn.ConvTranspose2d(start_fm * 16, start_fm * 8, 2, 2)
        self.Up_RRCNN4 = RRCNN_block(ch_in=start_fm*16, ch_out=start_fm*8,t=t)
        
        self.t_conv3 = nn.ConvTranspose2d(start_fm * 8, start_fm * 4, 2, 2)
        self.Up_RRCNN3 = RRCNN_block(ch_in=start_fm*8, ch_out=start_fm*4,t=t)
        
        self.t_conv2 = nn.ConvTranspose2d(start_fm * 4, start_fm * 2, 2, 2)
        self.Up_RRCNN2 = RRCNN_block(ch_in=start_fm*4, ch_out=start_fm*2,t=t)
        
        self.t_conv1 = nn.ConvTranspose2d(start_fm * 2, start_fm, 2, 2)
        self.Up_RRCNN1 = RRCNN_block(ch_in=start_fm*2, ch_out=start_fm,t=t)

        self.one_by_one = nn.Conv2d(start_fm, 4, 1, 1, 0)


    def forward(self,inputs):
        # encoding path
        rnn1 = self.RRCNN1(inputs)
        maxpool1 = self.Maxpool(rnn1)

        rnn2 = self.RRCNN2(maxpool1)
        maxpool2 = self.Maxpool(rnn2)

        rnn3 = self.RRCNN3(maxpool2)
        maxpool3 = self.Maxpool(rnn3)

        rnn4 = self.RRCNN4(maxpool3)
        maxpool4 = self.Maxpool(rnn4)

        rnn5 = self.RRCNN5(maxpool4)
               
               
        t_conv4 = self.t_conv4(rnn5)
        cat4 = torch.cat([rnn4 ,t_conv4], 1)
        ex_rnn4 = self.Up_RRCNN4(cat4)
        
        t_conv3 = self.t_conv3(ex_rnn4)
        cat3 = torch.cat([rnn3 ,t_conv3], 1)
        ex_rnn3 = self.Up_RRCNN3(cat3)

        t_conv2 = self.t_conv2(ex_rnn3)
        cat2 = torch.cat([rnn2 ,t_conv2], 1)
        ex_rnn2 = self.Up_RRCNN2(cat2)
        
        t_conv1 = self.t_conv1(ex_rnn2)
        cat1 = torch.cat([rnn1 ,t_conv1], 1)
        ex_rnn1 = self.Up_RRCNN1(cat1)
        
        one_by_one = self.one_by_one(ex_rnn1)
        
        return one_by_one

####################################### Deep Residual UNet ###############################################

class ResUnet(nn.Module):
    def __init__(self,  start_fm):
        super(ResUnet, self).__init__()

        filters=[start_fm, start_fm*2, start_fm*4, start_fm*8]

        self.input_layer = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.one_by_one = nn.Conv2d(filters[0], 4, 1, 1, 0)


    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        one_by_one = self.one_by_one(x10)

        return one_by_one


################################################################### Deep ResUnet ##############################################################

class Deep_ResUnet(nn.Module):
    def __init__(self, start_fm):
        super(Deep_ResUnet, self).__init__()

        filters=[start_fm, start_fm*2, start_fm*4, start_fm*8, start_fm*16]

        self.input_layer = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.bridge = ResidualConv(filters[3], filters[4], 2, 1)

        self.upsample_1 = Upsample(filters[4], filters[4], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[3], filters[3], 1, 1)

        self.upsample_2 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_3 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_4 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv4 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)


        self.one_by_one = nn.Conv2d(filters[0], 4, 1, 1, 0)


    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)

        # Bridge
        x5 = self.bridge(x4)
        
        # Decode
        x5 = self.upsample_1(x5)
        x6 = torch.cat([x5, x4], dim=1)

        x7 = self.up_residual_conv1(x6)

        x7 = self.upsample_2(x7)
        x8 = torch.cat([x7, x3], dim=1)

        x9 = self.up_residual_conv2(x8)

        x9 = self.upsample_3(x9)
        x10 = torch.cat([x9, x2], dim=1)

        x11 = self.up_residual_conv3(x10)

        x11 = self.upsample_4(x11)
        x12 = torch.cat([x11, x1], dim=1)

        x13 = self.up_residual_conv4(x12)

        one_by_one = self.one_by_one(x13)

        return one_by_one



######################################################## Res UNet ++ #################################################################

class ResUnetPlusPlus(nn.Module):
    def __init__(self, start_fm):
        super(ResUnetPlusPlus, self).__init__()

        filters=[start_fm, start_fm*2, start_fm*4, start_fm*8, start_fm*16]

        self.input_layer = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.one_by_one = nn.Conv2d(filters[0], 4, 1, 1, 0)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        one_by_one = self.one_by_one(x9)

        return one_by_one


###################################################### ResUNet with Res Path ##################################################

class ResPath(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResPath, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, output_dim, 1, 1, 0)
        self.conv2 = nn.Conv2d(input_dim, output_dim, 3, 3, 0)
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(output_dim)
            )

    def forward(self, x):
        shortcut = x
        shortcut = self.conv1(shortcut)
        x = self.conv2(x)
        pd = (shortcut.shape[2] - x.shape[2])
        if(pd%2 != 0):
          x = torch.nn.functional.pad(x, (pd//2,pd//2+1,pd//2,pd//2+1), )
        else:
          x = torch.nn.functional.pad(x, (pd//2,pd//2,pd//2,pd//2), )

        x = shortcut + x
        x = self.linear(x)
        return x

class ResUnet_path(nn.Module):
    def __init__(self, start_fm):
        super(ResUnet_path, self).__init__()

        filters=[start_fm, start_fm*2, start_fm*4, start_fm*8]

        self.input_layer = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=3, padding=1)
        )

        self.res_path_1 = ResPath(filters[0], filters[0])

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.res_path_2 = ResPath(filters[1], filters[1])

        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.res_path_3 = ResPath(filters[2], filters[2])

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.one_by_one = nn.Conv2d(filters[0], 4, 1, 1, 0)


    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        
        x2 = self.residual_conv_1(x1)
        # Res Path skip connection
        x1 = self.res_path_1(x1)
        x1 = self.res_path_1(x1)
        x1 = self.res_path_1(x1)
        x1 = self.res_path_1(x1)

        x3 = self.residual_conv_2(x2)
        x2 = self.res_path_2(x2)
        x2 = self.res_path_2(x2)
        x2 = self.res_path_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        x3 = self.res_path_3(x3)
        x3 = self.res_path_3(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        one_by_one = self.one_by_one(x10)

        return one_by_one



######################################################## Deep Res Unet Path ####################################################################

class Deep_ResUnet_Path(nn.Module):
    def __init__(self, start_fm):
        super(Deep_ResUnet_Path, self).__init__()

        filters=[start_fm, start_fm*2, start_fm*4, start_fm*8, start_fm*16]

        self.input_layer = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=3, padding=1)
        )

        self.res_path_1 = ResPath(filters[0], filters[0])
        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.res_path_2 = ResPath(filters[1], filters[1])
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.res_path_3 = ResPath(filters[2], filters[2])
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.res_path_4 = ResPath(filters[3], filters[3])
        self.bridge = ResidualConv(filters[3], filters[4], 2, 1)

        self.upsample_1 = Upsample(filters[4], filters[4], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[3], filters[3], 1, 1)

        self.upsample_2 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_3 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_4 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv4 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)


        self.one_by_one = nn.Conv2d(filters[0], 4, 1, 1, 0)


    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x1 = self.res_path_1(x1)
        x1 = self.res_path_1(x1)
        x1 = self.res_path_1(x1)
        x1 = self.res_path_1(x1)

        x3 = self.residual_conv_2(x2)
        x2 = self.res_path_2(x2)
        x2 = self.res_path_2(x2)
        x2 = self.res_path_2(x2)

        x4 = self.residual_conv_3(x3)
        x3 = self.res_path_3(x3)
        x3 = self.res_path_3(x3)

        # Bridge
        x5 = self.bridge(x4)
        x4 = self.res_path_4(x4)

        # Decode
        x5 = self.upsample_1(x5)
        x6 = torch.cat([x5, x4], dim=1)

        x7 = self.up_residual_conv1(x6)

        x7 = self.upsample_2(x7)
        x8 = torch.cat([x7, x3], dim=1)

        x9 = self.up_residual_conv2(x8)

        x9 = self.upsample_3(x9)
        x10 = torch.cat([x9, x2], dim=1)

        x11 = self.up_residual_conv3(x10)

        x11 = self.upsample_4(x11)
        x12 = torch.cat([x11, x1], dim=1)

        x13 = self.up_residual_conv4(x12)

        one_by_one = self.one_by_one(x13)

        return one_by_one


####################################################### RES UNET++ PATH #########################################################################


class ResUnetPlusPlus_Path(nn.Module):
    def __init__(self, start_fm):
        super(ResUnetPlusPlus_Path, self).__init__()

        filters=[start_fm, start_fm*2, start_fm*4, start_fm*8, start_fm*16]

        self.input_layer = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=3, padding=1)
        )

        self.res_path_1 = ResPath(filters[0], filters[0])

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.res_path_2 = ResPath(filters[1], filters[1])


        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.res_path_3 = ResPath(filters[2], filters[2])

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.one_by_one = nn.Conv2d(filters[0], 4, 1, 1, 0)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x1 = self.res_path_1(x1)
        x1 = self.res_path_1(x1)
        x1 = self.res_path_1(x1)
        x1 = self.res_path_1(x1)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x2 = self.res_path_2(x2)
        x2 = self.res_path_2(x2)
        x2 = self.res_path_2(x2)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x3 = self.res_path_3(x3)
        x3 = self.res_path_3(x3)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        one_by_one = self.one_by_one(x9)

        return one_by_one


######################################################  MULTI RES UNET  ######################################





























##############    ABHISHEK THAKUR   #################
# import torch
# import torch.nn as nn
# def double_conv(in_c, out_c):
#   conv = nn.Sequential(
#       nn.Conv2d(in_c, out_c, kernel_size = 3),
#       nn.ReLU(inplace = True),
#       nn.Conv2d(out_c, out_c, kernel_size = 3),
#       nn.ReLU(inplace = True)
#   )
#   return conv

# def crop_img(tensor, target_tensor):
#   target_size = target_tensor.size()[2]
#   tensor_size = tensor.size()[2]
#   delta = tensor_size - target_size
#   delta = delta // 2
#   return tensor[:,:, delta:tensor_size-delta, delta:tensor_size-delta]

# class UNet(nn.Module):
#   def __init__(self):
#     super(UNet, self).__init__()

#     self.down_conv_1 = double_conv(1,64)
#     self.down_conv_2 = double_conv(64,128)
#     self.down_conv_3 = double_conv(128,256)
#     self.down_conv_4 = double_conv(256,512)
#     self.down_conv_5 = double_conv(512,1024)
#     self.max_pool_2x2 = nn.MaxPool2d(kernel_size = 2, stride = 2)


#     self.up_trans_1 = nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 2, stride = 2)
#     self.up_conv_1 = double_conv(1024,512)

#     self.up_trans_2 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 2, stride = 2)
#     self.up_conv_2 = double_conv(512,256)

#     self.up_trans_3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2)
#     self.up_conv_3 = double_conv(256,128)

#     self.up_trans_4 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2)
#     self.up_conv_4 = double_conv(128,64)

#     self.out = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1)


#   def forward(self,image):
#     #encoder
#     x1 = self.down_conv_1(image)
#     print(x1.size())
#     x2 = self.max_pool_2x2(x1)
#     x3 = self.down_conv_2(x2)
#     x4 = self.max_pool_2x2(x3)
#     x5 = self.down_conv_3(x4)
#     x6 = self.max_pool_2x2(x5)
#     x7 = self.down_conv_4(x6)
#     x8 = self.max_pool_2x2(x7)
#     x9 = self.down_conv_5(x8)

#     #decoder

#     x = self.up_trans_1(x9)
#     y = crop_img(x7,x)
#     x = self.up_conv_1(torch.cat([x,y], 1))

#     x = self.up_trans_2(x)
#     y = crop_img(x5,x)
#     x = self.up_conv_2(torch.cat([x,y], 1))

#     x = self.up_trans_3(x)
#     y = crop_img(x3,x)
#     x = self.up_conv_3(torch.cat([x,y], 1))

#     x = self.up_trans_4(x)
#     y = crop_img(x1,x)
#     x = self.up_conv_4(torch.cat([x,y], 1))

#     x = self.out(x)
#     print(x.size())
  
#     return x 
