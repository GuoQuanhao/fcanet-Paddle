import math
import paddle
import paddle.nn as nn
# import torch.utils.model_zoo as model_zoo
from paddle.utils.download import get_weights_path_from_url

class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, dilation=dilation, padding=dilation, bias_attr=False)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias_attr=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2D(planes, planes * 4, kernel_size=1, bias_attr=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetSplit(nn.Layer):

    def __init__(self, block, layers, output_stride, BatchNorm,layer_range=[0,4],aux_channel=0):
        super(ResNetSplit, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        self.layer_range=layer_range
        self.aux_channel = aux_channel
        self.ori_inplanes=[3, 64, 64*block.expansion, 128*block.expansion, 256*block.expansion]
        self.inplanes = self.ori_inplanes[layer_range[0]]+aux_channel

        # Layers
        if layer_range[0]<=0 and 0<=layer_range[1]:
            self.conv1 = nn.Conv2D(self.inplanes, 64, kernel_size=7, stride=2, padding=3, bias_attr=False)
            self.bn1 = BatchNorm(64)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
            self.inplanes=64

        if layer_range[0]<=1 and 1<=layer_range[1]:
            self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)

        if layer_range[0]<=2 and 2<=layer_range[1]:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)

        if layer_range[0]<=3 and 3<=layer_range[1]:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)

        if layer_range[0]<=4 and 4<=layer_range[1]:
            self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
            # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self._init_weight()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input, if_final=True):
        x=input
        results=[]
        if self.layer_range[0]<=0 and 0<=self.layer_range[1]:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            l0=x
            results.append(l0)

        if self.layer_range[0]<=1 and 1<=self.layer_range[1]:
            x = self.layer1(x)
            l1=x
            results.append(l1)
            
        if self.layer_range[0]<=2 and 2<=self.layer_range[1]:
            x = self.layer2(x)
            l2=x
            results.append(l2)
            
        if self.layer_range[0]<=3 and 3<=self.layer_range[1]:
            x = self.layer3(x)
            l3=x
            results.append(l3)
            
        if self.layer_range[0]<=4 and 4<=self.layer_range[1]:
            x = self.layer4(x)
            l4=x
            results.append(l4)
            
        if if_final:
            return results[-1]
        else:
            return results

    def _init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32', default_initializer=nn.initializer.Normal(0, math.sqrt(2. / n)))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32', default_initializer=nn.initializer.Constant(1.0))
                m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32', default_initializer=nn.initializer.Constant(0.0))

    def _load_pretrained_model(self, local_weight_path):
        pretrain_dict = paddle.load(local_weight_path)
        model_dict = {}
        state_dict = self.state_dict()
        first_weight_names=['conv1.weight','layer1.0.conv1.weight','layer2.0.conv1.weight','layer3.0.conv1.weight','layer4.0.conv1.weight']
        ds_weight_names=['None','layer1.0.downsample.0.weight','layer2.0.downsample.0.weight','layer3.0.downsample.0.weight','layer4.0.downsample.0.weight']
        for k, v in pretrain_dict.items():
            if  self.aux_channel>0 and (k==first_weight_names[self.layer_range[0]]  or k==ds_weight_names[self.layer_range[0]]):
                v_tmp=state_dict[k]
                v_tmp[:,:self.ori_inplanes[self.layer_range[0]],:,:]=v
                model_dict[k] = v_tmp
                continue
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.set_state_dict(state_dict)

def ResNetSplit18(output_stride, BatchNorm, pretrained=False,layer_range=[0,4], aux_channel=0):
    model = ResNetSplit(BasicBlock, [2, 2, 2, 2], output_stride, BatchNorm,layer_range=layer_range,aux_channel=aux_channel)
    if pretrained:
        resnet18_pretrained_weight_url = 'https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams'
        local_weight_path = get_weights_path_from_url(resnet18_pretrained_weight_url)
        model._load_pretrained_model(local_weight_path)
    return model

def ResNetSplit34(output_stride, BatchNorm, pretrained=False,layer_range=[0,4], aux_channel=0):
    model = ResNetSplit(BasicBlock, [3, 4, 6, 3], output_stride, BatchNorm,layer_range=layer_range, aux_channel=aux_channel)
    if pretrained:
        resnet34_pretrained_weight_url = 'https://paddle-hapi.bj.bcebos.com/models/resnet34.pdparams'
        local_weight_path = get_weights_path_from_url(resnet34_pretrained_weight_url)
        model._load_pretrained_model(local_weight_path)
    return model

def ResNetSplit50(output_stride, BatchNorm, pretrained=False,layer_range=[0,4], aux_channel=0):
    model = ResNetSplit(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, layer_range=layer_range, aux_channel=aux_channel)
    if pretrained:
        resnet50_pretrained_weight_url = 'https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams'
        local_weight_path = get_weights_path_from_url(resnet50_pretrained_weight_url)
        model._load_pretrained_model(local_weight_path)
    return model

def ResNetSplit101(output_stride, BatchNorm, pretrained=False,layer_range=[0,4], aux_channel=0):
    model = ResNetSplit(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, layer_range=layer_range, aux_channel=aux_channel)
    if pretrained:
        resnet101_pretrained_weight_url = 'https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams'
        local_weight_path = get_weights_path_from_url(resnet101_pretrained_weight_url)
        model._load_pretrained_model(local_weight_path)
    return model

def ResNetSplit152(output_stride, BatchNorm, pretrained=False,layer_range=[0,4], aux_channel=0):
    model = ResNetSplit(Bottleneck, [3, 8, 36, 3], output_stride, BatchNorm, layer_range=layer_range, aux_channel=aux_channel)
    if pretrained:
        resnet152_pretrained_weight_url = 'https://paddle-hapi.bj.bcebos.com/models/resnet152.pdparams'
        local_weight_path = get_weights_path_from_url(resnet152_pretrained_weight_url)
        model._load_pretrained_model(local_weight_path)
    return model

 
    