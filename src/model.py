#1. DCNv2 because it is used in DLA
#2. DLA Model breakdown implementation

# DCNv2
import math
from typing import Tuple, Union
import torch
import torchvision.ops
from torch import nn
from torch.nn.modules.utils import _pair

#network.dla
import os
from os.path import join
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

#model.py
import torchvision.models as models
import torch.nn as nn

#losses
#from pytorch3d.ops import box3d_overlap
import iou3d_cuda

#detector
import cv2
import time

#utils
from src.utils import _transpose_and_gather_feat, boxes3d_to_bev_torch, _sigmoid
from src.utils import AverageMeter, car_pose_decode, car_pose_post_process
from src.utils import get_affine_transform, soft_nms_39, Debugger
from src.utils import flip_tensor, flip_lr, flip_lr_off, multi_pose_decode, multi_pose_post_process
from progress.bar import Bar

from src.utils import sel_locker
from copy import deepcopy

#https://github.com/liyier90/pytorch-dcnv2/blob/master/dcn.py
class DCNv2(nn.Module):
    """
    The below referenced link implements DCNv2 without need for compiling
    https://github.com/liyier90/pytorch-dcnv2/blob/master/dcn.py

    Attributes:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Tuple[int, int]): Size of the convolving kernel.
        stride (Tuple[int, int]): Stride of the convolution.
        padding (Tuple[int, int]): Padding added to all four sides of the input.
        dilation (Tuple[int, int]): Spacing between kernel elements.
        deformable_groups (int): Used to determine the number of offset groups.
        weight (torch.nn.Parameter): Convolution weights.
        bias (torch.nn.Parameter): Bias terms for convolution.
        conv_offset_mask (torch.nn.Conv2d): 2D convolution to generate the offset and mask.
    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Union[int, Tuple[int, int])): Size of the convolving kernel.
        stride (Union[int, Tuple[int, int]]): Stride of the convolution.
        padding (Union[int, Tuple[int, int]]): Padding added to all four sides of the input.
        dilation (Union[int, Tuple[int, int]]): Spacing between kernel elements.
        deformable_groups (int): Used to determine the number of offset groups.
    """
    num_chunks = 3  # Num channels for offset + mask

    def __init__(self, in_channels:int, out_channels:int, kernel_size:Union[int, Tuple[int, int]], stride:Union[int, Tuple[int, int]], padding:Union[int, Tuple[int, int]], dilation:Union[int, Tuple[int, int]]=1, deformable_groups:int = 1,) -> None:
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

        num_offset_mask_channels = (self.deformable_groups * self.num_chunks * self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(self.in_channels,num_offset_mask_channels,self.kernel_size,self.stride, self.padding,bias=True)
        self.init_offset()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call.
        Args:
            input (torch.Tensor): Input from the previous layer.
        Returns:
            (torch.Tensor): Result of convolution -> Tensor[batch_sz, out_channels, out_h, out_w]
        """
        out = self.conv_offset_mask(input)
        offset_1, offset_2, mask = torch.chunk(out, self.num_chunks, dim=1)
        offset = torch.cat((offset_1, offset_2), dim=1)
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(input=input, offset=offset, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, mask=mask,)

    def init_offset(self) -> None:
        """Initializes the weight and bias for `conv_offset_mask`."""
        self.conv_offset_mask.weight.data.zero_()
        if self.conv_offset_mask.bias is not None:
            self.conv_offset_mask.bias.data.zero_()

    def reset_parameters(self) -> None:
        """Re-initialize parameters using a method similar to He initialization with mode='fan_in' and gain=1."""
        fan_in = self.in_channels
        for k in self.kernel_size:
            fan_in *= k
        std = 1.0 / math.sqrt(fan_in)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()

#/src/lib/models/networks/pose_dla_dcn.py
BN_MOMENTUM = 0.1

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return 'http://dl.yf.io/dla/models/'+data+'/'+name+'-'+hash+'.pth'

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=3,stride=stride,padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out

class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out

class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d( in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x

class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,level_root=False, root_dim=0, root_kernel_size=1, dilation=1, root_residual=False):

        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, root_dim=0,
                              root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(nn.Conv2d(in_channels, out_channels,kernel_size=1, stride=1, bias=False),
                                         nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM))

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x

class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False), nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root)

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(nn.MaxPool2d(stride, stride=stride), nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1, padding=dilation, bias=False, dilation=dilation), nn.BatchNorm2d(planes, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(self.channels[-1], num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)

def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(nn.BatchNorm2d(cho, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.conv = DCNv2(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x

class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, padding=f // 2, output_padding=0, groups=o, bias=False)
            fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])

class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i), IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out

class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x

class DLASeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel, last_level, head_conv, out_channel=0):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)
        if out_channel == 0:
            out_channel = channels[self.first_level]
        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], [2 ** i for i in range(self.last_level - self.first_level)])
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(nn.Conv2d(channels[self.first_level], head_conv, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(head_conv, classes,  kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]

def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):
    model = DLASeg('dla{}'.format(num_layers), heads, pretrained=True, down_ratio=down_ratio, final_kernel=1, last_level=5, head_conv=head_conv)
    return model

#/src/lib/models/model.py
_model_factory = {'dla': get_pose_net}

def create_model(arch, heads, head_conv):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    return model

def load_model(model, model_path, optimizer=None, resume=False, lr=None, lr_step=None):
    start_epoch = 0
    print(model_path)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the pre-trained weight. Please make sure you have correctly specified --arch xxx or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, loaded shape{}. {}'.format(k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            ver = torch.__version__
            if ver == '1.12.0': #Added due to changes in adam optimizer params
                optimizer.param_groups[0]['capturable']=True
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model

def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch, 'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)

#/src/lib/models/losses.py
def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_iou: (M, N)
    """

    boxes_a_bev = boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = boxes3d_to_bev_torch(boxes_b)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_cuda.boxes_overlap_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev)

    # height overlap
    boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, 3]).view(-1, 1)
    boxes_a_height_max = boxes_a[:, 1].view(-1, 1)
    boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, 3]).view(1, -1)
    boxes_b_height_max = boxes_b[:, 1].view(1, -1)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)

    return iou3d

def _slow_neg_loss(pred, gt):
    '''focal loss from CornerNet'''
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)
    neg_weights = torch.pow(1 - gt[neg_inds], 4)
    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    num_pos = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)
    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()
    if num_pos > 0:
        all_loss /= num_pos
    loss -= all_loss
    return loss

def _slow_reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)
    regr = regr[mask]
    gt_regr = gt_regr[mask]
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum') #changed size_average=False to reduction ='sum' because size_average is deprecated
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    regr = regr * mask
    gt_regr = gt_regr * mask
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum') #changed size_average=False to reduction ='sum' because size_average is deprecated
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='mean') # changed reduction='elementwise_mean' because it is deprecated)

def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='mean') # changed reduction='elementwise_mean' because it is deprecated

def compute_rot_loss(output, target_bin, target_res, mask):
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res

def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper

class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)

class RegLoss(nn.Module):
    '''Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss

class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum') #changed size_average=False to reduction ='sum' because size_average is deprecated
        loss = loss / (mask.sum() + 1e-4)
        return loss

class NormRegL1Loss(nn.Module):
    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum') #changed size_average=False to reduction ='sum' because size_average is deprecated
        loss = loss / (mask.sum() + 1e-4)
        return loss

class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target,dep):
        dep=dep.squeeze(2)
        dep[dep<5]=dep[dep<5]*0.01
        dep[dep >= 5] = torch.log10(dep[dep >=5]-4)+0.1
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float()
        loss=torch.abs(pred * mask-target * mask)
        loss=torch.sum(loss,dim=2)*dep
        loss=loss.sum()
        loss = loss / (mask.sum() + 1e-4)

        return loss

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='mean') # changed reduction='elementwise_mean' because it is deprecated
        return loss

class depLoss(nn.Module):
    def __init__(self):
        super(depLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='mean') # changed reduction='elementwise_mean' because it is deprecated
        return loss

class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _transpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss

class Position_loss(nn.Module):
    def __init__(self, opt):
        super(Position_loss, self).__init__()
        const = torch.Tensor([[-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
                              [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1]])
        self.const = const.unsqueeze(0).unsqueeze(0)
        self.opt = opt
        self.num_joints = 9

    def forward(self, output, batch,phase=None):
        dim = _transpose_and_gather_feat(output['dim'], batch['ind'])
        rot = _transpose_and_gather_feat(output['rot'], batch['ind'])
        prob = _transpose_and_gather_feat(output['prob'], batch['ind'])
        kps = _transpose_and_gather_feat(output['hps'], batch['ind'])
        rot=rot.detach()
        b = dim.size(0)
        c = dim.size(1)
        mask = batch['hps_mask']
        mask = mask.float()
        calib = batch['calib']
        opinv = batch['opinv']
        cys = (batch['ind'] / self.opt.output_w).int().float()
        cxs = (batch['ind'] % self.opt.output_w).int().float()
        kps[..., ::2] = kps[..., ::2] + cxs.view(b, c, 1).expand(b, c, self.num_joints)
        kps[..., 1::2] = kps[..., 1::2] + cys.view(b, c, 1).expand(b, c, self.num_joints)
        opinv = opinv.unsqueeze(1)
        opinv = opinv.expand(b, c, -1, -1).contiguous().view(-1, 2, 3).float()
        kps = kps.view(b, c, -1, 2).permute(0, 1, 3, 2)
        hom = torch.ones(b, c, 1, 9).cuda()
        kps = torch.cat((kps, hom), dim=2).view(-1, 3, 9)
        kps = torch.bmm(opinv, kps).view(b, c, 2, 9)
        kps = kps.permute(0, 1, 3, 2).contiguous().view(b, c, -1)  # 16.32,18
        si = torch.zeros_like(kps[:, :, 0:1]) + calib[:, 0:1, 0:1]
        alpha_idx = rot[:, :, 1] > rot[:, :, 5]
        alpha_idx = alpha_idx.float()
        alpha1 = torch.atan(rot[:, :, 2] / rot[:, :, 3]) + (-0.5 * np.pi)
        alpha2 = torch.atan(rot[:, :, 6] / rot[:, :, 7]) + (0.5 * np.pi)
        alpna_pre = alpha1 * alpha_idx + alpha2 * (1 - alpha_idx)
        alpna_pre = alpna_pre.unsqueeze(2)
        rot_y = alpna_pre + torch.atan2(kps[:, :, 16:17] - calib[:, 0:1, 2:3], si)
        rot_y[rot_y > np.pi] = rot_y[rot_y > np.pi] - 2 * np.pi
        rot_y[rot_y < - np.pi] = rot_y[rot_y < - np.pi] + 2 * np.pi
        calib = calib.unsqueeze(1)
        calib = calib.expand(b, c, -1, -1).contiguous()
        kpoint = kps
        f = calib[:, :, 0, 0].unsqueeze(2)
        f = f.expand_as(kpoint)
        cx, cy = calib[:, :, 0, 2].unsqueeze(2), calib[:, :, 1, 2].unsqueeze(2)
        cxy = torch.cat((cx, cy), dim=2)
        cxy = cxy.repeat(1, 1, 9)
        kp_norm = (kpoint - cxy) / f
        l = dim[:, :, 2:3]
        h = dim[:, :, 0:1]
        w = dim[:, :, 1:2]
        cosori = torch.cos(rot_y)
        sinori = torch.sin(rot_y)
        B = torch.zeros_like(kpoint)
        C = torch.zeros_like(kpoint)
        kp = kp_norm.unsqueeze(3)
        const = self.const.cuda()
        const = const.expand(b, c, -1, -1)
        A = torch.cat([const, kp], dim=3)
        B[:, :, 0:1] = l * 0.5 * cosori + w * 0.5 * sinori
        B[:, :, 1:2] = h * 0.5
        B[:, :, 2:3] = l * 0.5 * cosori - w * 0.5 * sinori
        B[:, :, 3:4] = h * 0.5
        B[:, :, 4:5] = -l * 0.5 * cosori - w * 0.5 * sinori
        B[:, :, 5:6] = h * 0.5
        B[:, :, 6:7] = -l * 0.5 * cosori + w * 0.5 * sinori
        B[:, :, 7:8] = h * 0.5
        B[:, :, 8:9] = l * 0.5 * cosori + w * 0.5 * sinori
        B[:, :, 9:10] = -h * 0.5
        B[:, :, 10:11] = l * 0.5 * cosori - w * 0.5 * sinori
        B[:, :, 11:12] = -h * 0.5
        B[:, :, 12:13] = -l * 0.5 * cosori - w * 0.5 * sinori
        B[:, :, 13:14] = -h * 0.5
        B[:, :, 14:15] = -l * 0.5 * cosori + w * 0.5 * sinori
        B[:, :, 15:16] = -h * 0.5
        B[:, :, 16:17] = 0
        B[:, :, 17:18] = 0
        C[:, :, 0:1] = -l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 1:2] = -l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 2:3] = -l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 3:4] = -l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 4:5] = l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 5:6] = l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 6:7] = l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 7:8] = l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 8:9] = -l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 9:10] = -l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 10:11] = -l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 11:12] = -l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 12:13] = l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 13:14] = l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 14:15] = l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 15:16] = l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 16:17] = 0
        C[:, :, 17:18] = 0
        B = B - kp_norm * C
        kps_mask = mask
        AT = A.permute(0, 1, 3, 2)
        AT = AT.view(b * c, 3, 18)
        A = A.view(b * c, 18, 3)
        B = B.view(b * c, 18, 1).float()
        pinv = torch.bmm(AT, A)
        pinv = torch.inverse(pinv)
        mask2 = torch.sum(kps_mask, dim=2)
        loss_mask = mask2 > 15
        pinv = torch.bmm(pinv, AT)
        pinv = torch.bmm(pinv, B)
        pinv = pinv.view(b, c, 3, 1).squeeze(3)
        # change the center to kitti center. Note that the pinv is the 3D center point in the camera coordinate system
        pinv[:, :, 1] = pinv[:, :, 1] + dim[:, :, 0] / 2
        dim_mask = dim<0
        dim = torch.clamp(dim, 0 , 10)
        dim_mask_score_mask = torch.sum(dim_mask, dim=2)
        dim_mask_score_mask = 1 - (dim_mask_score_mask > 0).float() #added .float()
        dim_mask_score_mask = dim_mask_score_mask.float()
        box_pred = torch.cat((pinv, dim, rot_y), dim=2).detach()
        loss = (pinv - batch['location'])
        loss_norm = torch.norm(loss, p=2, dim=2)
        loss_mask = loss_mask.float()
        loss = loss_norm * loss_mask
        mask_num = (loss != 0).sum()
        loss = loss.sum() / (mask_num + 1)
        dim_gt = batch['dim'].clone()
        location_gt = batch['location']
        ori_gt = batch['ori']
        dim_gt[dim_mask] = 0
        gt_box = torch.cat((location_gt, dim_gt, ori_gt), dim=2)
        box_pred = box_pred.view(b * c, -1)
        gt_box = gt_box.view(b * c, -1)
        box_score = boxes_iou3d_gpu(box_pred, gt_box)
        box_score = torch.diag(box_score).view(b, c)
        prob = prob.squeeze(2)
        box_score = box_score * loss_mask * dim_mask_score_mask
        loss_prob = F.binary_cross_entropy_with_logits(prob, box_score.detach(), reduction='sum') #changed reducee=False to reduction ='sum' because reduce is deprecated
        loss_prob = loss_prob * loss_mask * dim_mask_score_mask
        loss_prob = torch.sum(loss_prob, dim=1)
        loss_prob = loss_prob.sum() / (mask_num + 1)
        box_score = box_score * loss_mask
        box_score = box_score.sum() / (mask_num + 1)
        return loss, loss_prob, box_score

class kp_align(nn.Module):
    def __init__(self):
        super(kp_align, self).__init__()
        self.index_x=torch.LongTensor([0,2,4,6,8,10,12,14])

    def forward(self, output, batch):
        kps = _transpose_and_gather_feat(output['hps'], batch['ind'])
        mask = batch['inv_mask']
        index=self.index_x.cuda()
        x_bottom=torch.index_select(kps,dim=2,index=index[0:4])
        bottom_mask = torch.index_select(mask,dim=2,index=index[0:4]).float()
        x_up=torch.index_select(kps,dim=2,index=index[4:8])
        up_mask = torch.index_select(mask, dim=2, index=index[4:8]).float()
        mask=bottom_mask*up_mask
        loss = F.l1_loss(x_up * mask, x_bottom * mask, reduction='sum') #changed size_average=False to reduction ='sum' because size_average is deprecated
        loss = loss / (mask.sum() + 1e-4)
        return loss

class kp_conv(nn.Module):
    def __init__(self):
        super(kp_conv, self).__init__()
        self.con1=torch.nn.Conv2d(18,18,3,padding=1)
        self.index_x=torch.LongTensor([0,2,4,6,8,10,12,14])

    def forward(self, output):
        kps = output['hps']
        kps=self.con1(kps)
        return kps

class BaseLoss(torch.nn.Module):
    def __init__(self, opt):
        super(BaseLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_hm_hp = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_kp = RegWeightedL1Loss() if not opt.dense_hp else torch.nn.L1Loss(reduction='sum')
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_rot = BinRotLoss()
        self.opt = opt
        self.position_loss=Position_loss(opt)

    def forward(self, outputs, batch,phase=None):
        opt = self.opt
        hm_loss, wh_loss, off_loss = 0, 0, 0
        hp_loss, off_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0, 0
        dim_loss, rot_loss, prob_loss = 0, 0, 0
        coor_loss =0
        box_score=0
        output = outputs[0]
        output['hm'] = _sigmoid(output['hm'])
        if opt.hm_hp and not opt.mse_loss:
            output['hm_hp'] = _sigmoid(output['hm_hp'])
        hm_loss = self.crit(output['hm'], batch['hm'])
        hp_loss = self.crit_kp(output['hps'],batch['hps_mask'], batch['ind'], batch['hps'],batch['dep'])
        if opt.wh_weight > 0:
            wh_loss = self.crit_reg(output['wh'], batch['reg_mask'],batch['ind'], batch['wh'])
        if opt.dim_weight > 0:
            dim_loss = self.crit_reg(output['dim'], batch['reg_mask'],batch['ind'], batch['dim'])
        if opt.rot_weight > 0:
            rot_loss = self.crit_rot(output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'], batch['rotres'])
        if opt.reg_offset and opt.off_weight > 0:
            off_loss = self.crit_reg(output['reg'], batch['reg_mask'], batch['ind'], batch['reg'])
        if opt.reg_hp_offset and opt.off_weight > 0:
            hp_offset_loss = self.crit_reg(output['hp_offset'], batch['hp_mask'], batch['hp_ind'], batch['hp_offset'])
        if opt.hm_hp and opt.hm_hp_weight > 0:
            hm_hp_loss = self.crit_hm_hp(output['hm_hp'], batch['hm_hp'])
        coor_loss, prob_loss, box_score = self.position_loss(output, batch,phase)
        loss_stats = {'loss': box_score, 'hm_loss': hm_loss, 'hp_loss': hp_loss,
                      'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss,'dim_loss': dim_loss,
                      'rot_loss':rot_loss,'prob_loss':prob_loss,'box_score':box_score,'coor_loss':coor_loss}
        return loss_stats, loss_stats

class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch,unlabel=False,phase=None):
        outputs = self.model(batch['input'])
        if unlabel:
            loss, loss_stats=outputs[-1]['dim'].mean(),{}
        else:
            loss, loss_stats = self.loss(outputs, batch,phase)
        return outputs[-1], loss, loss_stats

class BaseTrainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)
        self.rampup = exp_rampup(100)
        self.rampup_prob = exp_rampup(100)
        self.rampup_coor = exp_rampup(100)

    def set_device(self, device):
        self.model_with_loss = self.model_with_loss.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader,unlabel_loader1=None,unlabel_loader2=None, unlabel_set=None,iter_num=None,data_loder2=None):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            model_with_loss.eval()
        torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format("3D detection", opt.exp_id), max=num_iters)
        end = time.time()

        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            coor_weight=self.rampup_coor(epoch)
            if coor_weight< self.opt.coor_thresh:
                coor_weight=0
            output, loss, loss_stats = model_with_loss(batch,phase=phase)
            loss_agg  = (loss['hm_loss'].mean() if torch.is_tensor(loss['hm_loss']) else loss['hm_loss']) * opt.hm_weight
            loss_agg += (loss['wh_loss'].mean() if torch.is_tensor(loss['wh_loss']) else loss['wh_loss']) * opt.wh_weight
            loss_agg += (loss['off_loss'].mean() if torch.is_tensor(loss['off_loss']) else loss['off_loss']) * opt.off_weight
            loss_agg += (loss['hp_loss'].mean() if torch.is_tensor(loss['hp_loss']) else loss['hp_loss']) * opt.hp_weight
            loss_agg += (loss['hp_offset_loss'].mean() if torch.is_tensor(loss['hp_offset_loss']) else loss['hp_offset_loss']) * opt.off_weight
            loss_agg += (loss['hm_hp_loss'].mean() if torch.is_tensor(loss['hm_hp_loss']) else loss['hm_hp_loss']) * opt.hm_hp_weight
            loss_agg += (loss['dim_loss'].mean() if torch.is_tensor(loss['dim_loss']) else loss['dim_loss']) *opt.dim_weight
            loss_agg += (loss['rot_loss'].mean() if torch.is_tensor(loss['rot_loss']) else loss['rot_loss']) * opt.rot_weight
            loss_agg += (loss['prob_loss'].mean() if torch.is_tensor(loss['prob_loss']) else loss['prob_loss']) *self.rampup_prob(epoch)
            loss_agg += (loss['coor_loss'].mean() if torch.is_tensor(loss['coor_loss']) else loss['coor_loss']) *coor_weight

            if phase == 'train':
                self.optimizer.zero_grad()
                loss_agg.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(epoch,iter_id,num_iters, phase=phase, total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                #avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['input'].size(0)) #REPLACED WITH TWO LINES BELOW
                x = loss_stats[l].mean().item() if torch.is_tensor(loss_stats[l]) else loss_stats[l] #ADDED
                avg_loss_stats[l].update(x, batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) |Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    #print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
                    print('{}'.format(Bar.suffix))
            else:
                bar.next()

            if opt.debug > 0:
                self.debug(batch, output, iter_id)

            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats, loss_agg

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        hm_hp = output['hm_hp'] if self.opt.hm_hp else None
        hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
        dets = car_pose_decode(output['hm'], output['wh'], output['hps'],output['dim'], output['rot'], prob=output['prob'],reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

        dets_out = car_pose_post_process(dets.copy(), batch['meta']['c'].cpu().numpy(), batch['meta']['s'].cpu().numpy(), output['hm'].shape[2], output['hm'].shape[3])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'hp_loss', 'hm_hp_loss', 'hp_offset_loss', 'wh_loss', 'off_loss','dim_loss','rot_loss','prob_loss','coor_loss','box_score']
        loss = BaseLoss(opt)
        return loss_states, loss

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self,epoch,data_loader,unlabel_loader1=None,unlabel_loader2=None,unlabel_set=None,iter_num=None,uncert=None):
        return self.run_epoch('train', epoch, data_loader,unlabel_loader1,unlabel_loader2,unlabel_set,iter_num,uncert)

class BaseDetector(object):
    def __init__(self, opt):
        self.device = 'cpu'
        if opt.gpus[0] >= 0:
            #opt.device = torch.device('cuda')
            self.device = 'cuda'
        #else:
            #opt.device = torch.device('cpu')
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(self.device)#opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True
        self.image_path = opt.results_dir
        const = torch.Tensor([[-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1]])
        self.const = const.unsqueeze(0).unsqueeze(0)
        self.const=self.const.to(self.device)#self.opt.device)

        self.flip_idx = opt.flip_idx

    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width  = int(width * scale)
        if self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(resized_image, trans_input, (inp_width, inp_height),flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)

        meta = {'c': c, 's': s, 'out_height': inp_height // self.opt.down_ratio, 'out_width': inp_width // self.opt.down_ratio}
        trans_output_inv = get_affine_transform(c, s, 0, [meta['out_width'], meta['out_height']],inv=1)
        trans_output_inv = torch.from_numpy(trans_output_inv)
        trans_output_inv=trans_output_inv.unsqueeze(0)
        meta['trans_output_inv']=trans_output_inv
        return images, meta

    def process(self, images, meta, return_time=False):
        with torch.no_grad():
            if self.device == 'cuda': #Added
                torch.cuda.synchronize() #indented
            output = self.model(images)[-1]
            output['hm'] = output['hm'].sigmoid_()
            if self.opt.hm_hp and not self.opt.mse_loss:
                output['hm_hp'] = output['hm_hp'].sigmoid_()

            reg = output['reg'] if self.opt.reg_offset else None
            hm_hp = output['hm_hp'] if self.opt.hm_hp else None
            hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
            if self.device == 'cuda': #Added
                torch.cuda.synchronize() #indented
            forward_time = time.time()

            if self.opt.flip_test:
                output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
                output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
                output['hps'] = (output['hps'][0:1] + flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
                hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 if hm_hp is not None else None
                reg = reg[0:1] if reg is not None else None
                hp_offset = hp_offset[0:1] if hp_offset is not None else None
            if self.opt.faster==True:
                dets = car_pose_decode_faster(output['hm'], output['hps'], output['dim'], output['rot'], prob=output['prob'],K=self.opt.K, meta=meta, const=self.const)
            else:
                dets = car_pose_decode(output['hm'], output['wh'], output['hps'],output['dim'],output['rot'],prob=output['prob'], reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K,meta=meta,const=self.const)

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets = car_pose_post_process(dets.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'])
        for j in range(1,2):#, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 41)
            dets[0][j][:, :4] /= scale
            dets[0][j][:, 5:23] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        results[1] = np.concatenate([detection[1] for detection in detections], axis=0).astype(np.float32)
        if self.opt.nms or len(self.opt.test_scales) > 1:
            soft_nms_39(results[1], Nt=0.5, method=2)
        results[1] = results[1].tolist()
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy().copy()
        dets[:, :, :4] *= self.opt.down_ratio
        dets[:, :, 5:39] *= self.opt.down_ratio
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        if self.opt.hm_hp:
            pred = debugger.gen_colormap_hp(output['hm_hp'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')

    def show_results(self, debugger, image, results, calib):
        debugger.add_img(image, img_id='car_pose')
        for bbox in results[1]:
            if bbox[4] > self.opt.vis_thresh:
                #debugger.add_coco_bbox(bbox[:4], bbox[40], bbox[4], img_id='car_pose')
                #debugger.add_kitti_hp(bbox[5:23], img_id='car_pose')
                #debugger.add_bev(bbox, img_id='car_pose',is_faster=self.opt.faster)
                locker = sel_locker(self.opt,bbox)
                #locker = [l* 1.5 for l in locker]
                debugger.add_3d_detection(bbox, calib, img_id='car_pose',locker=locker)
                debugger.save_kitti_format(bbox,self.opt, locker, locker,img_id='car_pose',is_faster=self.opt.faster)
        #if self.opt.vis:
            #debugger.show_all_imgs(pause=self.pause)
        return deepcopy({'img':debugger.imgs['car_pose'], 'dict':deepcopy(debugger.results['car_pose'])}) #Added

    def read_clib(self,calib_path):
        f = open(calib_path, 'r')
        for i, line in enumerate(f):
            if i == 2:
                calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
                calib = calib.reshape(3, 4)
                return calib

    def run(self, image_or_path_or_tensor, calib_file_path=None, meta=None): #added calib_file_path=None
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3), theme=self.opt.debugger_theme, det_cats = self.opt.det_cats)
        start_time = time.time()
        pre_processed = False

        #ADDED
        if calib_file_path == None and type(image_or_path_or_tensor)==type(''):
            calib_path=self.opt.calib_dir+ str(image_or_path_or_tensor[-10:-3]+'txt')
        else:
            calib_path=calib_file_path
        calib_numpy=self.read_clib(calib_path)
        calib=torch.from_numpy(calib_numpy).unsqueeze(0).to(self.opt.device)

        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type (''):
            self.image_path=image_or_path_or_tensor
            image = cv2.imread(image_or_path_or_tensor)
            #calib_path=self.opt.calib_dir+ str(image_or_path_or_tensor[-10:-3]+'txt')
            #calib_numpy=self.read_clib(calib_path)
            #calib=torch.from_numpy(calib_numpy).unsqueeze(0).to(self.opt.device)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta)
                meta['trans_output_inv']=meta['trans_output_inv'].to(self.opt.device)
            else:
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}
            meta['calib']=calib
            images = images.to(self.opt.device)
            if self.device == 'cuda': #Added
                torch.cuda.synchronize() #indented
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            output, dets, forward_time = self.process(images,meta,return_time=True)

            if self.device == 'cuda': #Added
                torch.cuda.synchronize() #indented
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            #if self.opt.debug >= 2:
                #self.debug(debugger, images, dets, output, scale)

            dets = self.post_process(dets, meta, scale)
            if self.device == 'cuda': #Added
                torch.cuda.synchronize() #indented
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)

        results = self.merge_outputs(detections)
        if self.device == 'cuda': #Added
            torch.cuda.synchronize() #indented
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        #if self.opt.debug >= 1:
        result = self.show_results(debugger, image, results, calib_numpy)

        #return {'results': results, 'tot': tot_time, 'load': load_time, 'pre': pre_time, 'net': net_time, 'dec': dec_time,'post': post_time, 'merge': merge_time}
        return deepcopy(result)
