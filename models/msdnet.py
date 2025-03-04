import torch.nn as nn
import torch
import math
import pdb
import numpy as np

class Myresnet_Basic(nn.Module):
    def __init__(self, nIn, nOut, kernel=(3, 1), stride=(1, 1),
                 padding=(1, 0)):
        super(Myresnet_Basic, self).__init__()
        self.shortcut = nn.Conv2d(nIn, nOut, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(nOut)
        self.relu = nn.ReLU()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(nOut),
            nn.ReLU(),
            nn.Conv2d(nOut, nOut, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(nOut)
        )
    def forward(self, x):
        shortcut = self.bn(self.shortcut(x))
        x = self.net(x)
        x = x + shortcut
        # res = self.relu(out + shortcut)
        # print(res.shape)
        return x

class Mynet_middle_layer(nn.Module):
    def __init__(self, nIn, nOut, Kernel_Size = (3, 1), Stride = (3, 1)) -> None:
        super(Mynet_middle_layer, self).__init__()
        # self.conv = nn.Conv2d(nIn, nOut, kernel_size=Kernel_Size, stride=Stride)
        # self.bn = nn.BatchNorm2d(nOut)
        # self.relu = nn.ReLU(inplace=True)
        # net_list = []
        # net_list.append(self.conv)
        # net_list.append(self.bn)
        # net_list.append(self.relu)
        # self.net = nn.Sequential(*net_list)
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=Kernel_Size, stride=Stride),
            nn.BatchNorm2d(nOut),
            nn.ReLU()
        )
    def forward(self, x):
        out = self.net(x)
        return out


class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=(3, 1), stride=(1, 1),
                 padding=(1, 0)):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(nOut),
            nn.ReLU()
        )

    def forward(self, x):
        # if x.dim() < 4:
        #     x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        print(x.shape)
        out = self.net(x)
        return out


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bottleneck,
                 bnWidth):
        """
        a basic conv in MSDNet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bottleneck: use bottleneck or not
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bottleneck is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.Conv2d(
                nIn, nInner, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU())

        if type == 'normal':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=(3,1),
                                   stride=(1, 1), padding=(1, 0), bias=False))
        elif type == 'down':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=(3,1),
                                   stride=(2, 1), padding=(1, 0), bias=False))
        else:
            raise ValueError

        layer.append(nn.BatchNorm2d(nOut))
        layer.append(nn.ReLU())

        self.net = nn.Sequential(*layer)
        #self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        return out


class ConvDownNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2):
        super(ConvDownNormal, self).__init__()
        self.conv_down = ConvBN(nIn1, nOut // 2, 'down',
                                bottleneck, bnWidth1)
        self.conv_normal = ConvBN(nIn2, nOut // 2, 'normal',
                                   bottleneck, bnWidth2)

    def forward(self, x):
        res = [x[1],
               self.conv_down(x[0]),
               self.conv_normal(x[1])]
        return torch.cat(res, dim=1)


class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, 'normal',
                                   bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0],
               self.conv_normal(x[0])]

        return torch.cat(res, dim=1)


class MyNET_FirstLayer(nn.Module):
    def __init__(self, nIn, nOut, args):
        super(MyNET_FirstLayer, self).__init__()
        self.layers = nn.ModuleList()
        conv = nn.Sequential(
                    nn.Conv2d(nIn, nOut * args.grFactor[0], (7, 1), (2, 1), (3, 0)),
                    nn.BatchNorm2d(nOut * args.grFactor[0]),
                    nn.ReLU(),
                    nn.MaxPool2d((3, 1), (2, 1), (1, 0)))
        self.layers.append(conv)
        # nIn = nOut * args.grFactor[0]
        # for i in range(1, args.nScales):
        #     self.layers.append(ConvBasic(nIn, nOut * args.grFactor[i],
        #                                 kernel=(3,1), stride=(2, 1), padding=(1, 0)))
        #     nIn = nOut * args.grFactor[i]
    def forward(self, x):
        res = []
        # for i in range(len(self.layers)):
            
        #     x = self.layers[i](x)
        #     res.append(x)
        res.append(self.layers[0](x))
        return res[0]


class MSDNFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, args):
        super(MSDNFirstLayer, self).__init__()
        self.layers = nn.ModuleList()
        if args.data.startswith('cifar'):
            self.layers.append(ConvBasic(nIn, nOut * args.grFactor[0],
                                         kernel=3, stride=1, padding=1))
        elif args.data == 'caltech256':
            conv = nn.Sequential(
                    nn.Conv2d(nIn, nOut * args.grFactor[0], 7, 2, 3),
                    nn.BatchNorm2d(nOut * args.grFactor[0]),
                    nn.ReLU(),
                    nn.MaxPool2d(3, 2, 1))
            self.layers.append(conv)
        elif args.data == 'ImageNet':
            conv = nn.Sequential(
                    nn.Conv2d(nIn, nOut * args.grFactor[0], 7, 2, 3),
                    nn.BatchNorm2d(nOut * args.grFactor[0]),
                    nn.ReLU(),
                    nn.MaxPool2d(3, 2, 1))
            self.layers.append(conv)
        else:
            conv = nn.Sequential(
                    nn.Conv2d(nIn, nOut * args.grFactor[0], (7, 1), (2, 1), (3, 0)),
                    nn.BatchNorm2d(nOut * args.grFactor[0]),
                    nn.ReLU(),
                    nn.MaxPool2d((3, 1), (2, 1), (1, 0)))
            self.layers.append(conv)

        nIn = nOut * args.grFactor[0]

        if args.data in ['UCI', 'UniMib', 'Wisdom'] :
            for i in range(1, args.nScales):
                self.layers.append(ConvBasic(nIn, nOut * args.grFactor[i],
                                            kernel=(3,1), stride=(2, 1), padding=(1, 0)))
                nIn = nOut * args.grFactor[i]
        else:
            for i in range(1, args.nScales):
                self.layers.append(ConvBasic(nIn, nOut * args.grFactor[i],
                                            kernel=3, stride=2, padding=1))
                nIn = nOut * args.grFactor[i]

    def forward(self, x):
        res = []
        for i in range(len(self.layers)):
            
            x = self.layers[i](x)
            res.append(x)

        return res

class MSDNLayer(nn.Module):
    def __init__(self, nIn, nOut, args, inScales=None, outScales=None):
        super(MSDNLayer, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.inScales = inScales if inScales is not None else args.nScales
        self.outScales = outScales if outScales is not None else args.nScales

        self.nScales = args.nScales
        self.discard = self.inScales - self.outScales

        self.offset = self.nScales - self.outScales
        self.layers = nn.ModuleList()

        if self.discard > 0:
            nIn1 = nIn * args.grFactor[self.offset - 1]
            nIn2 = nIn * args.grFactor[self.offset]
            _nOut = nOut * args.grFactor[self.offset]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, args.bottleneck,
                                              args.bnFactor[self.offset - 1],
                                              args.bnFactor[self.offset]))
        else:
            self.layers.append(ConvNormal(nIn * args.grFactor[self.offset],
                                          nOut * args.grFactor[self.offset],
                                          args.bottleneck,
                                          args.bnFactor[self.offset]))

        for i in range(self.offset + 1, self.nScales):
            nIn1 = nIn * args.grFactor[i - 1]
            nIn2 = nIn * args.grFactor[i]
            _nOut = nOut * args.grFactor[i]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, args.bottleneck,
                                              args.bnFactor[i - 1],
                                              args.bnFactor[i]))

    def forward(self, x):
        if self.discard > 0:
            inp = []
            for i in range(1, self.outScales + 1):
                inp.append([x[i - 1], x[i]])
        else:
            inp = [[x[0]]]
            for i in range(1, self.outScales):
                inp.append([x[i - 1], x[i]])

        res = []
        for i in range(self.outScales):
            res.append(self.layers[i](inp[i]))

        return res


class ParallelModule(nn.Module):
    """
    This module is similar to luatorch's Parallel Table
    input: N tensor
    network: N module
    output: N tensor
    """
    def __init__(self, parallel_modules):
        super(ParallelModule, self).__init__()
        self.m = nn.ModuleList(parallel_modules)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            res.append(self.m[i](x[i]))

        return res

class ClassifierModule(nn.Module):
    def __init__(self, m, channel, num_classes, cur_block = None):
        super(ClassifierModule, self).__init__()
        if cur_block == 1:
            channel = channel * 6
        self.m = m 
        self.linear = nn.Linear(384, num_classes)# default:channel UCI:124*1*9, UniMib:124*1*3, Wisdom:124*2*3
        

    def forward(self, x):
        res = self.m(x)
        # print(res.shape)
        res = res.view(res.size(0), -1)
        out = self.linear(res)
        return out


class MyNet(nn.Module):
    def __init__(self, args):
        super(MyNet, self).__init__()
        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.nBlocks = args.nBlocks
        self.steps = [args.base]
        self.args = args
        
        n_layers_all, n_layer_curr = args.base, 0
        for i in range(1, self.nBlocks):
            self.steps.append(args.step if args.stepmode == 'even'
                             else args.step * i + 1)
            n_layers_all += self.steps[-1]

        # print("building network of steps: ")
        # print(self.steps, n_layers_all)

        nIn = args.nChannels
        for i in range(self.nBlocks):
            print(' ********************** Block {} '
                  ' **********************'.format(i + 1))
            m, nIn = \
                self.My_build_block(nIn, args, self.steps[i],
                                  n_layers_all, n_layer_curr)
            self.blocks.append(m)
            n_layer_curr += self.steps[i]

            if args.data == 'UCI':
                self.classifier.append(
                    self.My_build_classifier_har(nIn * args.grFactor[-1], 6))
                self.n_classes = 6
            elif args.data == 'UniMib':
                self.classifier.append(
                    self.My_build_classifier_har(nIn * args.grFactor[-1], 17))
                self.n_classes = 17
            elif args.data == 'Wisdom':
                self.classifier.append(
                    self.My_build_classifier_har(nIn , 6, i + 1))
                # print(nIn, args.grFactor[-1])
                self.n_classes = 6
            else:
                raise NotImplementedError
        for m in self.blocks:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self.My_init_weights(_m)
            else:
                self.My_init_weights(m)

        for m in self.classifier:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self.My_init_weights(_m)
            else:
                self.My_init_weights(m)
                
    def My_init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            
    def My_build_block(self, nIn, args, step, n_layer_all, n_layer_curr):
        layers = [MyNET_FirstLayer(1, nIn, args)] \
            if n_layer_curr == 0 else []
        layers.append(Mynet_middle_layer(nIn, nIn))
        layers.append(Myresnet_Basic(nIn, nIn * 2))
        nIn = nIn * 2
        return nn.Sequential(*layers), nIn
    
    def My_build_classifier_har(self, nIn, num_classes, cur_Block):
        conv = nn.Sequential(
            ConvBasic(nIn, nIn, kernel=(3, 1), stride=(2, 1), padding=(1, 0)),
            ConvBasic(nIn, nIn, kernel=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.AvgPool2d((2, 1), padding=(0, 0))
        )
        return ClassifierModule(conv, nIn, num_classes, cur_Block)
    
    def forward(self, x):
        res = []
        for i in range(self.nBlocks):
            x = self.blocks[i](x)
            res.append(self.classifier[i](x))
        return res

    
    def predict(self, x):
        """
        This function was coded for MC integration of LA inference
        it outputs final output (res) and mean feature of the image (phi_out) 
        """
        res = []
        phi_out = []
        for i in range(self.nBlocks):
            x = self.blocks[i](x)
            # classifier module forward
            phi = self.classifier[i].m(x[-1])
            phi = phi.view(phi.size(0), -1)
            res.append(self.classifier[i].linear(phi))
            phi_out.append(phi)
        return res, phi_out
        
    def predict_until(self, x, until_block):
        """
        This function was coded for MC integration of LA inference
        it outputs final output (res) and mean feature of the image (phi_out) 
        """
        res = []
        phi_out = []
        for i in range(until_block):
            x = self.blocks[i](x)
            # classifier module forward
            phi = self.classifier[i].m(x[-1])
            phi = phi.view(phi.size(0), -1)
            res.append(self.classifier[i].linear(phi))
            phi_out.append(phi)
        return res, phi_out

class MSDNet(nn.Module):
    def __init__(self, args):
        super(MSDNet, self).__init__()
        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.nBlocks = args.nBlocks
        self.steps = [args.base]
        self.args = args
        
        n_layers_all, n_layer_curr = args.base, 0
        for i in range(1, self.nBlocks):
            self.steps.append(args.step if args.stepmode == 'even'
                             else args.step * i + 1)
            n_layers_all += self.steps[-1]

        print("building network of steps: ")
        print(self.steps, n_layers_all)

        nIn = args.nChannels
        for i in range(self.nBlocks):
            print(' ********************** Block {} '
                  ' **********************'.format(i + 1))
            m, nIn = \
                self._build_block(nIn, args, self.steps[i],
                                  n_layers_all, n_layer_curr)
            self.blocks.append(m)
            n_layer_curr += self.steps[i]

            if args.data.startswith('cifar100'):
                self.classifier.append(
                    self._build_classifier_cifar(nIn * args.grFactor[-1], 100))
                self.n_classes = 100
            elif args.data.startswith('cifar10'):
                self.classifier.append(
                    self._build_classifier_cifar(nIn * args.grFactor[-1], 10))
                self.n_classes = 10
            elif args.data == 'ImageNet':
                self.classifier.append(
                    self._build_classifier_imagenet(nIn * args.grFactor[-1], 1000))
                self.n_classes = 1000
            elif args.data == 'caltech256':
                self.classifier.append(
                    self._build_classifier_imagenet(nIn * args.grFactor[-1], 257))
                self.n_classes = 257
            elif args.data == 'UCI':
                self.classifier.append(
                    self._build_classifier_imagenet(nIn * args.grFactor[-1], 6))
                self.n_classes = 6
            elif args.data == 'UniMib':
                self.classifier.append(
                    self._build_classifier_imagenet(nIn * args.grFactor[-1], 17))
                self.n_classes = 17
            elif args.data == 'Wisdom':
                self.classifier.append(
                    self._build_classifier_imagenet(nIn * args.grFactor[-1], 6))
                # print(nIn, args.grFactor[-1])
                self.n_classes = 6
            else:
                raise NotImplementedError

        for m in self.blocks:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

        for m in self.classifier:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_block(self, nIn, args, step, n_layer_all, n_layer_curr):

        layers = [MSDNFirstLayer(1, nIn, args)] \
            if n_layer_curr == 0 else []
        for i in range(step):
            n_layer_curr += 1
            inScales = args.nScales
            outScales = args.nScales
            if args.prune == 'min':
                inScales = min(args.nScales, n_layer_all - n_layer_curr + 2)
                outScales = min(args.nScales, n_layer_all - n_layer_curr + 1)
            elif args.prune == 'max':
                interval = math.ceil(1.0 * n_layer_all / args.nScales)
                inScales = args.nScales - math.floor(1.0 * (max(0, n_layer_curr - 2)) / interval)
                outScales = args.nScales - math.floor(1.0 * (n_layer_curr - 1) / interval)
            else:
                raise ValueError

            layers.append(MSDNLayer(nIn, args.growthRate, args, inScales, outScales))
            print('|\t\tinScales {} outScales {} inChannels {} outChannels {}\t\t|'.format(inScales, outScales, nIn, args.growthRate))

            nIn += args.growthRate
            if args.prune == 'max' and inScales > outScales and \
                    args.reduction > 0:
                offset = args.nScales - outScales
                layers.append(
                    self._build_transition(nIn, math.floor(1.0 * args.reduction * nIn),
                                           outScales, offset, args))
                _t = nIn
                nIn = math.floor(1.0 * args.reduction * nIn)
                print('|\t\tTransition layer inserted! (max), inChannels {}, outChannels {}\t|'.format(_t, math.floor(1.0 * args.reduction * _t)))
            elif args.prune == 'min' and args.reduction > 0 and \
                    ((n_layer_curr == math.floor(1.0 * n_layer_all / 3)) or
                     n_layer_curr == math.floor(2.0 * n_layer_all / 3)):
                offset = args.nScales - outScales
                layers.append(self._build_transition(nIn, math.floor(1.0 * args.reduction * nIn),
                                                     outScales, offset, args))

                nIn = math.floor(1.0 * args.reduction * nIn)
                print('|\t\tTransition layer inserted! (min)\t|')
            print("")

        return nn.Sequential(*layers), nIn

    def _build_transition(self, nIn, nOut, outScales, offset, args):
        net = []
        for i in range(outScales):
            net.append(ConvBasic(nIn * args.grFactor[offset + i],
                                 nOut * args.grFactor[offset + i],
                                 kernel=(1, 1), stride=(1, 1), padding=(0, 0)))
        return ParallelModule(net)

    def _build_classifier_cifar(self, nIn, num_classes):
        interChannels1, interChannels2 = 128, 128
        conv = nn.Sequential(
            ConvBasic(nIn, interChannels1, kernel=(3, 1), stride=(2, 1), padding=(1, 0)),
            ConvBasic(interChannels1, interChannels2, kernel=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.AvgPool2d((2, 1),padding=(0, 0)),
        )
        return ClassifierModule(conv, interChannels2, num_classes)

    def _build_classifier_imagenet(self, nIn, num_classes):
        conv = nn.Sequential(
            ConvBasic(nIn, nIn, kernel=(3, 1), stride=(2, 1), padding=(1, 0)),
            ConvBasic(nIn, nIn, kernel=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.AvgPool2d((2, 1), padding=(0, 0))
        )
        return ClassifierModule(conv, nIn, num_classes)

    def forward(self, x):
        res = []
        for i in range(self.nBlocks):
            x = self.blocks[i](x)
            res.append(self.classifier[i](x))
        return res

    def predict(self, x):
        """
        This function was coded for MC integration of LA inference
        it outputs final output (res) and mean feature of the image (phi_out) 
        """
        res = []
        phi_out = []
        for i in range(self.nBlocks):
            x = self.blocks[i](x)
            # classifier module forward
            phi = self.classifier[i].m(x[-1])
            phi = phi.view(phi.size(0), -1)
            res.append(self.classifier[i].linear(phi))
            phi_out.append(phi)
        return res, phi_out
        
    def predict_until(self, x, until_block):
        """
        This function was coded for MC integration of LA inference
        it outputs final output (res) and mean feature of the image (phi_out) 
        """
        res = []
        phi_out = []
        for i in range(until_block):
            x = self.blocks[i](x)
            # classifier module forward
            phi = self.classifier[i].m(x[-1])
            phi = phi.view(phi.size(0), -1)
            res.append(self.classifier[i].linear(phi))
            phi_out.append(phi)
        return res, phi_out

