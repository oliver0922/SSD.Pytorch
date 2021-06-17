import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os
from efficientnet_pytorch import EfficientNet


class SSD(nn.Module): # SSD 구조 설정.
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__() # 상속받은 pytorch의 nn.module에 ssd적용.
        self.phase = phase # train인지 test인지 설정.
        self.num_classes = num_classes # class의 개수.
        self.cfg = voc['SSD{}'.format(size)] # voc의 image  형식의 size.
        self.priorbox = PriorBox(self.cfg) # prior box의 shape.
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
            # prior box의 shape으로 랜덤하게 초기화.
        self.size = size
        
        self.base = nn.ModuleList(base) # SSD의 basemodel nn.Module에 넘겨주기.
        
        self.L2Norm = L2Norm(512, 20) # L2 Normalization 적용.
        self.extras = nn.ModuleList(extras) # loc conf layers에 추가되는 추가적인 layer.

        self.loc = nn.ModuleList(head[0]) # head에 저장되어있는 loc.
        self.conf = nn.ModuleList(head[1]) # head에 저장되어있는 conf.

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect()
            # SSD를 test할 때의 설정.

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        # 각각 변수들을 저장하기 위한 list 생성.
        
        for k in range(23):
            x = self.base[k](x)
            # basenet의 각각의 k번째 층의 요소를 nn.module로 가져와 저장.
        s = self.L2Norm(x) # L2 Normalization 적용.
        sources.append(s)

        for k in range(23, len(self.base)):
            x = self.base[k](x) # fc7로 vgg16 적용.
        sources.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
            # extra layer들 기존 레이어에 추가시키고 적용.

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            # loc, conf 각각 list에 추가하고 dimension변경, 배열들간에 연속적으로 나열.

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # 배열들간에 쌓여져 있는 내부 배열들 없애고 합치기.
        
        if self.phase == "test":
             output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45, # 사용자 정의 함수 적용.
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file): # 기존에 저장되어 있는 weight 파일 불러오기.
        other, ext = os.path.splitext(base_file) # 파일 split.
        if ext == '.pkl' or '.pth':
            print('Begin loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            # weight경로에 있는 파일 불러오기.
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
            # 지원하는 확장자와 다르다면 실행하지 않고 오류 출력.

# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py 참고.
def vgg(cfg, i = 3, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # v가 M일 때 Maxpooling을 실행.
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            # v기 C일 떼 Maxpooling을 실행하되 ceil_mode로 실행.
            # ceil mode는 Output Size에 대하여 바닥 함수대신, 천장 함수를 사용.
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            # v가 둘다 아니라면 convolution 실행.
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            # convolution 실행뒤에 batch normalization을 통해 batch의 평균과 분산을 조정.
            # Relu 활성화함수는 batch normalization과 관께없이 실행.
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    # 각각의 pooling층, convolution 층을 만든다.
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    # 각각의 층을 집어넣으면서 relu를 적용.
    print('VGG base:',layers)
    return layers

def efficientnet_base(batch_norm=False):
    base_model = EfficientNet.from_name('efficientnet-b4')
    layer1 = [base_model._conv_stem, base_model._bn0]
    layer2 = [base_model._blocks[0],base_model._blocks[1],base_model._blocks[2]]
    layer3 = [base_model._blocks[3],base_model._blocks[4],base_model._blocks[5],base_model._blocks[6]]
    layer4 = [base_model._blocks[7],base_model._blocks[8],base_model._blocks[9],base_model._blocks[10]]
    layer5 = [base_model._blocks[11],base_model._blocks[12],base_model._blocks[13],base_model._blocks[14],base_model._blocks[15],base_model._blocks[16],base_model._blocks[17],base_model._blocks[18],base_model._blocks[19],base_model._blocks[20],base_model._blocks[21],base_model._blocks[22]]
    print('base network:', layer1 + layer2 + layer3 + layer4 + layer5)
    return layer1 + layer2 + layer3 + layer4 + layer5
# basenet이 efficientnet일 때 block들을 여러개 넣어주면서 layer를 구성한다.

def add_extras(cfg, i, batch_norm=False):
    # SSD의 가장 큰 특징은 중간중간의 특징맵들을 모두 고려하는 것이기 때문에 vgg16 net의 extra layer가 추가된다.
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)] # convolution 연산 size(cfg)에 따라서 진행.
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
                # v가 S가 아니라면 stride 1로 진행.
            flag = not flag
        in_channels = v
    if len(cfg) == 13:
        print('input channels:',in_channels)
        layers += [nn.Conv2d(in_channels, 256, kernel_size=4,padding=1)]
        # caffe version에서 cfg의 길이가 13이라면 layer에 필터 사이즈가 4x4인 convolution 연산 layer를 넣어줌.
    print('extras layers:',layers)
    return layers

def add_efficientnet_extras(cfg, i = 272, batch_norm=False):
    # Extra layers added to EfficientNet for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    print('extras layers:',layers)
    return layers
# 기본적으로 add_extras와 같은데 efficientnet basenet일 때.

def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2] # conv4_3, conv7의 갯수
    print('VGG16 output size:',len(vgg))
    print('extra layer size:', len(extra_layers))
    for i, layer in enumerate(extra_layers):
        print('extra layer {} : {}'.format(i, layer))
        # extra layer 몇번째 layer가 어떤 layer인지 출력.
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
        # 각각의 loc_layers, conf_layers에 필터 사이즈 3, padding 1로 convolution layer 추가.
        # vgg의 conv4_3, conv7의 갯수에 따라서 진행.
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)

def efficientnet_multibox(efficientnet, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    efficientnet_source = [9, 13, -1]   #P3-p7
    print('EfficientNet output size:',len(efficientnet_source))
    print('extra layer size:', len(extra_layers))
    
    for i, layer in enumerate(extra_layers):
        print('extra layer {} : {}'.format(i, layer))
        # extra layer 몇번째 layer가 어떤 layer인지 출력.
    for k, v in enumerate(efficientnet_source):
        loc_layers += [nn.Conv2d(efficientnet[v]._project_conv.weight.size()[0],
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(efficientnet[v]._project_conv.weight.size()[0],
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return efficientnet, extra_layers, (loc_layers, conf_layers)
# efficientnet의 multibox 함수, 기본적으로 위의 multibox와 같다.

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
# 위에서 함수들에 들어갔던 input(output)사이즈 또는 if문으로 들어가 실행되기 위함.
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128],
}
# 위에서 함수들에 들어갔던 input(output)사이즈 또는 if문으로 들어가 실행되기 위함.
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  
    '512': [4, 6, 6, 6, 4, 4, 4],
}
# 특징맵에 따른 박스들의 개수 설정. 보통 앞부분 레이어에서 4개, 뒤로갈수록 6개 다시 4개 순으로 생성.
efficientnet_mbox = [4, 6, 6, 6, 4, 4]
efficientnet_axtras = [128, 'S', 256, 128, 256, 128, 256]
# efficientnet이 basenet일 때의 설정.

def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    # phase가 train, test 모드가 둘 다 아니라면 오류!
    if size not in [300, 512] :
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 and SSD512 is supported!")
        return
    # SSD의 size를 300, 512 중에 선택하지 않는 다면 오류! (지원하지 않음.)
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    # 위에서 만든 multibox를 통해 loc, conf layer들을 추가하고 반환.
    print('Begin to build SSD-VGG...\n')
    return SSD(phase, size, base_, extras_, head_, num_classes) # SSD 구성.

def build_ssd_efficientnet(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size not in [300, 512] :
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 and SSD512 is supported!")
        return
    base_, extras_, head_ = efficientnet_multibox(efficientnet_base(),
                                     add_efficientnet_extras(efficientnet_axtras),
                                     efficientnet_mbox, num_classes)
    # 위와 완전히 동일하지만 basenet으로 efficientnet을 쓰는 경우.
    print('Begin to build SSD-EfficientNet...')
    return SSD(phase, size, base_, extras_, head_, num_classes) # basenet이 efficientnet인 경우의 SSD 구성.
