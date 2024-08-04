import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from mmcv.cnn import ConvModule

############ 版本说明 ###################
# 2024.03.13
# output0305_grad_enhance_att
# GMTA fusion_loss 10
# 融合里面有三个特征加入融合
# 设置的学习率的参数
# SOLVER:
#   STEPS: (5000, 10000)
#   MAX_ITER: 15000
# out文件是train_final.out
###########################################

class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right % 2 == 0.0:
                left = int(lef_right / 2)
                right = int(lef_right / 2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot % 2 == 0.0:
                top = int(top_bot / 2)
                bot = int(top_bot / 2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out)
            # out = self.dropout(out)
        return out


class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        # out_channels_def = 16
        out_channels_def = int(in_channels / 2)
        # out_channels_def = out_channels
        denseblock = []
        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


class FusionBlock_res(torch.nn.Module):
    def __init__(self, channels, index):
        super(FusionBlock_res, self).__init__()
        ws = [3, 3, 3, 3]
        self.up = nn.Upsample(scale_factor=2)
        self.conv_fusion = ConvLayer(2 * channels, channels, ws[index], 1)

        self.conv_ir = ConvLayer(channels, channels, ws[index], 1)
        self.conv_vi = ConvLayer(channels, channels, ws[index], 1)

        block = []
        block += [ConvLayer(2 * channels, channels, 1, 1),
                  ConvLayer(channels, channels, ws[index], 1),
                  ConvLayer(channels, channels, ws[index], 1)]
        self.bottelblock = nn.Sequential(*block)

    def forward(self, x_ir, x_vi):
        f_cat = torch.cat([x_ir, x_vi], 1)
        f_init = self.conv_fusion(f_cat)
        f_init = self.up(f_init)

        out_ir = self.conv_ir(x_ir)
        out_vi = self.conv_vi(x_vi)
        out_ir = self.up(out_ir)
        out_vi = self.up(out_vi)
        out = torch.cat([out_ir, out_vi], 1)
        out = self.bottelblock(out)
        out = f_init + out
        return out


class Fusion_network(nn.Module):
    def __init__(self, nC):
        super(Fusion_network, self).__init__()

        self.fusion_block1 = FusionBlock_res(nC[0], 0)
        self.fusion_block2 = FusionBlock_res(nC[1], 1)
        self.fusion_block3 = FusionBlock_res(nC[2], 2)
        self.fusion_block4 = FusionBlock_res(nC[3], 3)

    def forward(self, en_ir, en_vi):
        f1_0 = self.fusion_block1(en_ir[0], en_vi[0])
        f2_0 = self.fusion_block2(en_ir[1], en_vi[1])
        f3_0 = self.fusion_block3(en_ir[2], en_vi[2])
        f4_0 = self.fusion_block4(en_ir[3], en_vi[3])
        return [f1_0, f2_0, f3_0, f4_0]


class Decoder(nn.Module):
    def __init__(self, nb_filter, output_nc=3):
        super().__init__()
        block = DenseBlock_light
        kernel_size = 3
        stride = 1
        self.up = nn.Upsample(scale_factor=2)
        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)
        self.up_eval = UpsampleReshape_eval()
        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride)

    def forward(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up_eval(f_en[0], f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up_eval(f_en[1], f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up_eval(f_en[0], x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up_eval(f_en[2], f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up_eval(f_en[1], x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up_eval(f_en[0], x2_2)], 1))
        x1_3 = self.up(x1_3)
        output = self.conv_out(x1_3)
        return output


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                if grad is not None:
                    # print(grad)
                    # TODO
                    tmp = param_t - lr_inner * grad
                    self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, MetaConv2d):
        init.kaiming_normal(m.weight)


class FusionBlock(MetaModule):
    def __init__(self, in_block, out_block, k_size=3):
        super(FusionBlock, self).__init__()
        self.conv1_1 = MetaConv2d(
            in_channels=in_block,
            out_channels=out_block,
            kernel_size=k_size,
            stride=1,
            padding=(k_size - 1) // 2,
            bias=True
        )
        self.conv1_2 = MetaConv2d(
            in_channels=out_block,
            out_channels=out_block,
            kernel_size=k_size,
            stride=1,
            padding=(k_size - 1) // 2,
            bias=True
        )
        self.conv1_3 = MetaConv2d(
            in_channels=out_block,
            out_channels=out_block,
            kernel_size=k_size,
            stride=1,
            padding=(k_size - 1) // 2,
            bias=True
        )

        self.conv1_0_00 = MetaConv2d(
            in_channels=out_block,
            out_channels=out_block,
            kernel_size=k_size,
            stride=1,
            padding=(k_size - 1) // 2,
            bias=True
        )
        self.conv1_0_01 = MetaConv2d(
            in_channels=out_block,
            out_channels=out_block,
            kernel_size=k_size,
            stride=1,
            padding=(k_size - 1) // 2,
            bias=True
        )
        self.relu = nn.ReLU()
        # self.tree1 = Tree(out_block)
        # self.tree2 = Tree(out_block)
        # self.tree3 = Tree(out_block)


    def forward(self, x):
        x = self.conv1_1(x)
        # x = self.tree1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        # x = self.tree2(x)
        x = self.relu(x)
        x = self.conv1_3(x)
        # x = self.tree3(x)
        x = self.relu(x)

        # x_1 = self.conv1_1(x)
        # x_1_t = self.tree1(x_1)
        # x_1_t = self.relu(x_1_t)
        # x_2 = self.conv1_2(x)
        # x_2_t = self.tree2(x_2)
        # x_2_t = self.relu(x_2_t)
        # x_3 = self.conv1_3(x)
        # x_3_t = self.tree3(x_3)
        # x_3_t = self.relu(x_3_t)
        # x = x_1_t+x_2_t+x_3_t

        x0 = self.conv1_0_00(x)
        # x0 = self.tree3(x0)
        x0 = self.relu(x0)
        x1 = self.conv1_0_01(x)
        # x1 = self.tree4(x1)
        x1 = self.relu(x1)

        return torch.cat([x0, x1], dim=1)



import math
class reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    
    def forward(self, x):
        return self.relu(x + 3) / 6

class Encoder(nn.Module):
    def __init__(self, c1,c2):
        super(Encoder, self).__init__()

        block1 = []
        
        block1.append(FusionBlock(in_block=4, out_block=32, k_size=3))#nn.Sequential(*block1)
        block1.append(FusionBlock(in_block=64, out_block=64, k_size=3))
        self.block1 =nn.Sequential(*block1)

        self.conv2 = nn.Conv2d(c2*2, c2//2, 1, bias=True)
        self.relu = nn.ReLU()
        self.att1 = Attentionregion(M=2, res_channels=256)
        self.att2 =Attentionregion(M=2, res_channels=256)
        self.att3 = Attentionregion(M=2, res_channels=256)

        self.conv_module = ConvModule(c2//2, c2//2, 1, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU'))
        self.conv_module_act = ConvModule(c2//2, c2//2, 1, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU'))
        self.conv_module_add = ConvModule(c2//2, c2//2, 1, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU'))
        self.sigmoid = h_sigmoid()
        self._init_weight()

    def forward(self, x, low_level_feat,factor=2):
        x1,x2,x3,x4 = x[0],x[1],x[2],x[3]

        low_level_feat = self.block1(low_level_feat)
        # import matplotlib.pylab as plt
        # for i in range(low_level_feat.shape[1]):
        #     x1_show = (low_level_feat[0,i,:,:]).cpu().detach().numpy()
        #     plt.imshow(x1_show)
        #     plt.savefig("./low_level_feat/{}.png".format(i))
        # import matplotlib.pylab as plt
        # for i in range(x[0].shape[1]):
        #     x1_show = (x[0][0,i,:,:]*255).cpu().detach().numpy()
        #     plt.imshow(x1_show)
        #     plt.savefig("./x1_before/{}.png".format(i))

        # import matplotlib.pylab as plt
        # for i in range(x[1].shape[1]):
        #     x1_show = (x[1][0,i,:,:]*255).cpu().detach().numpy()
        #     plt.imshow(x1_show)
        #     plt.savefig("./x2_before/{}.png".format(i))

        # import matplotlib.pylab as plt
        # for i in range(x[2].shape[1]):
        #     x1_show = (x[2][0,i,:,:]*255).cpu().detach().numpy()
        #     plt.imshow(x1_show)
        #     plt.savefig("./x3_before/{}.png".format(i))

        x1 = self.att1(x1)
        x2 = self.att2(x2)
        x3 = self.att3(x3)
        # import matplotlib.pylab as plt
        # for i in range(256,256+5):
        #     x1_show = (x1[0,i,:,:]*255).cpu().detach().numpy()
        #     plt.imshow(x1_show)
        #     plt.savefig("./fea_show/x1/{}.png".format(i))
        # # import matplotlib.pylab as plt
        # for i in range(256,256+5):
        #     x2_show = (x2[0,i,:,:]*255).cpu().detach().numpy()
        #     plt.imshow(x2_show)
        #     plt.savefig("./fea_show/x2/{}.png".format(i))
        # # import matplotlib.pylab as plt
        # for i in range(256,256+5):
        #     x3_show = (x3[0,i,:,:]*255).cpu().detach().numpy()
        #     plt.imshow(x3_show)
        #     plt.savefig("./fea_show/x3/{}.png".format(i))
        # import matplotlib.pylab as plt
        # for i in range(x2.shape[1]):
        #     x2_show = (x2[0,i,:,:]*255).cpu().detach().numpy()
        #     plt.imshow(x2_show)
        #     plt.savefig("./x2/{}.png".format(i))

        # import matplotlib.pylab as plt
        # for i in range(x2.shape[1]):
        #     x2_show = (x2[0,i,:,:]*255).cpu().detach().numpy()
        #     plt.imshow(x2_show)
        #     plt.savefig("./x2/{}.png".format(i))

        x1 = F.interpolate(x1, size=[i*(factor//2) for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=[i*(factor//2) for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=[i*(factor//2) for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
        x = x1+x2+x3
        x = self.relu(self.conv2(x))

        x_origin = self.conv_module(low_level_feat)
        x_act = self.conv_module_act(x)
        x_act = self.sigmoid(x_act)
        x_out = x_origin * x_act + x_origin
        # x_origin = self.conv_module(x)
        # x_act = self.conv_module_act(low_level_feat)
        # x_act = self.sigmoid(x_act)
        # x_out = x_origin * x_act + x_origin
        return x_out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, MetaConv2d):
                init.kaiming_normal(m.weight)



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class FusionNet(MetaModule):
    def __init__(self, block_num, feature_out):
        super(FusionNet, self).__init__()
        self.feature_out = feature_out
        self.block1 = Encoder(4,256)
        self.block2_in = 128
        self.block2 = nn.Sequential(
            nn.Conv2d(self.block2_in, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(32, 2, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x1,x2):
        x = self.block1(x1,x2)
        x = self.block2(x)

        return None, x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(SKConv, self).__init__()
        d = max(in_channels // r, L)
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, d),
            nn.ReLU(),
            nn.Linear(d, out_channels * M)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        feats = [conv(x) for conv in self.convs]
        U = sum(feats)  #4,3,512,512
        s = self.global_pool(U)      # 4,3,1,1
        z = self.fc(s.view(batch_size, -1))   #4,6
        z = z.view(batch_size, -1, len(self.convs))  #4,3,2
        a = self.softmax(z)
        a = a.unsqueeze(-1).unsqueeze(-1)  #4,3,2,1,1
        b1 = a[:, :, 0:1, :, :] * feats[0].unsqueeze(2)
        b2 = a[:, :, 1:, :, :] * feats[1].unsqueeze(2)
        V = torch.sum(torch.cat([b1,b2], dim=2), dim=2)
        return V

# 自定义 SRU（Spatial and Reconstruct Unit）类
class SRU(nn.Module):
    def __init__(self,
                 oup_channels:int,  # 输出通道数
                 group_num:int = 16,  # 分组数，默认为16
                 gate_treshold:float = 0.5,  # 门控阈值，默认为0.5
                 torch_gn:bool = True  # 是否使用PyTorch内置的GroupNorm，默认为False
                 ):
        super().__init__()  # 调用父类构造函数

         # 初始化 GroupNorm 层或自定义 GroupBatchnorm2d 层
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(c_num=oup_channels, group_num=group_num)
        self.gate_treshold  = gate_treshold  # 设置门控阈值
        self.sigomid        = nn.Sigmoid()  # 创建 sigmoid 激活函数

    def forward(self, x):
        gn_x        = self.gn(x)  # 应用分组批量归一化
        w_gamma     = self.gn.weight / sum(self.gn.weight)  # 计算 gamma 权重
        w_gamma     = w_gamma.view(1, -1, 1, 1)
        reweights   = self.sigomid(gn_x * w_gamma)  # 计算重要性权重

        # 门控机制
        w1          = torch.where(reweights > self.gate_treshold, torch.ones_like(reweights), reweights) # 大于门限值的设为1，否则保留原值
        w2          = torch.where(reweights > self.gate_treshold, torch.zeros_like(reweights), reweights) # 大于门限值的设为0，否则保留原值
        x_1         = w1 * x
        x_2         = w2 * x
        y           = self.reconstruct(x_1,x_2)
        return y
    
    def reconstruct(self,x_1,x_2):
        x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)

# 自定义 CRU（Channel Reduction Unit）类
class CRU(nn.Module):
    def __init__(self, op_channel:int, alpha:float = 1/2, squeeze_radio:int = 2, group_size:int = 2, group_kernel_size:int = 3):
        super().__init__()  # 调用父类构造函数

        self.up_channel     = up_channel = int(alpha * op_channel)  # 计算上层通道数
        self.low_channel    = low_channel = op_channel - up_channel  # 计算下层通道数
        self.squeeze1       = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层 无分变率变化
        self.squeeze2       = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层

        # 上层特征转换
        self.GWC            = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1, padding=group_kernel_size // 2, groups=group_size)  # 创建卷积层
        self.PWC1           = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)  # 创建卷积层

        # 下层特征转换
        self.PWC2           = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层
        self.advavg         = nn.AdaptiveAvgPool2d(1)  # 创建自适应平均池化层

    def forward(self, x):
        # 分割输入特征
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)

        # 上层特征转换
        Y1 = self.GWC(up) + self.PWC1(up)

        # 下层特征转换
        Y2 = torch.cat([self.PWC2(low), low], dim=1)

        # 特征融合
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


# 自定义 ScConv（Squeeze and Channel Reduction Convolution）模型
class ScConv(nn.Module):
    def __init__(self, op_channel:int, group_num:int = 16, gate_treshold:float = 0.5, alpha:float = 1/2, squeeze_radio:int = 2, group_size:int = 2, group_kernel_size:int = 3):
        super().__init__()  # 调用父类构造函数

        self.SRU = SRU(op_channel, group_num=group_num, gate_treshold=gate_treshold)  # 创建 SRU 层
        self.CRU = CRU(op_channel, alpha=alpha, squeeze_radio=squeeze_radio, group_size=group_size, group_kernel_size=group_kernel_size)  # 创建 CRU 层

    def forward(self, x):
        x = self.SRU(x)  # 应用 SRU 层
        x = self.CRU(x)  # 应用 CRU 层
        return x

class Attentionregion(nn.Module):

    def __init__(self, M=32, res_channels=2048, pooling_mode='GAP', add_lambda=0.8):
        super(Attentionregion, self).__init__()
        self.M = M
        self.base_channels = res_channels
        self.out_channels = M * res_channels
        # self.conv = BasicConv2d(res_channels, self.M, kernel_size=1)
        self.conv = SKConv(res_channels,self.M)
        # self.conv = ScConv(res_channels,self.M)
        # self.conv = MetaConv2d(
        #     in_channels=res_channels,
        #     out_channels=self.M,
        #     kernel_size=1,
        #     stride=1,
        #     padding=(1 - 1) // 2,
        #     bias=True
        # )
        # self.conv = ConvModule(res_channels, self.M, 1, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU'))
        self.EPSILON = 1e-6

    def bilinear_attention_pooling(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        feature_matrix = []
        for i in range(M):
            AiF = features * attentions[:, i:i + 1, ...]
            feature_matrix.append(AiF)
        feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        # feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + self.EPSILON)

        # l2 normalization along dimension M and C
        # feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        return feature_matrix

    def forward(self, x):

        attention_maps = self.conv(x)
        feature_matrix = self.bilinear_attention_pooling(x, attention_maps)

        return feature_matrix
    

# x = torch.rand(1,256,512,512)
# model = Attentionregion(M=2, res_channels=256)
# y = model(x)
# print(x.shape)
# print(y.shape)