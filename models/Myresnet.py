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
        res = self.relu(x)
        # print(res.shape)
        return res
    
    
class Mynet_middle_layer(nn.Module):
    def __init__(self, nIn, nOut, Kernel_Size = (3, 1), Stride = (2, 1)) -> None:
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
        # print(x.shape)
        out = self.net(x)
        return out
    
    
    
class MyNET_FirstLayer(nn.Module):
    def __init__(self, nIn, nOut, args):
        super(MyNET_FirstLayer, self).__init__()
        self.layers = nn.ModuleList()
        conv = nn.Sequential(
                    nn.Conv2d(nIn, nOut * args.grFactor[0], (6, 1), (3, 1), (0, 0)), #UCI(5, 1) #Other(6, 1) 
                    nn.BatchNorm2d(nOut * args.grFactor[0]),
                    nn.ReLU())
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

class ClassifierModule(nn.Module):
    def __init__(self, m, channel, num_classes, cur_block = None, dataset_name = None):
        super(ClassifierModule, self).__init__()
        if dataset_name == 'Wisdm':
            if cur_block == 1:
                channel = channel*33*3 # large 32, other 33
            elif cur_block == 2:
                channel = channel*16*3
            elif cur_block == 3:
                channel = channel*7*3
            elif cur_block == 4:
                channel = channel*3*3
        elif dataset_name == 'UCI':
            if cur_block == 1:
                channel = channel*21*9
            elif cur_block == 2:
                channel = channel*10*9
            elif cur_block == 3:
                channel = channel*4*9
            elif cur_block == 4:
                channel = channel*2*9
        elif dataset_name == 'Usc':
            if cur_block == 1:
                channel = channel*84*6
            elif cur_block == 2:
                channel = channel*42*6
            elif cur_block == 3:
                channel = channel*20*6
            elif cur_block == 4:
                channel = channel*10*6
            elif cur_block == 5:
                channel = channel*4*6
            elif cur_block == 6:
                channel = channel*2*6
        elif dataset_name == 'UniMib':
            if cur_block == 1:
                channel = channel*24*3
            elif cur_block == 2:
                channel = channel*12*3
            elif cur_block == 3:
                channel = channel*5*3
            elif cur_block == 4:
                channel = channel*2*3
            
        self.m = m 
        self.linear = nn.Linear(channel, num_classes)# default:channel UCI:124*1*9, UniMib:124*1*3, Wisdm:124*2*3
        

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
                self.input_size_W = 128
                self.input_size_H = 9
                self.classifier.append(
                    self.My_build_classifier_har(nIn, 6, i + 1, args.data))
                self.n_classes = 6
            elif args.data == 'UniMib':
                self.input_size_W = 151
                self.input_size_H = 3
                self.classifier.append(
                    self.My_build_classifier_har(nIn, 17, i + 1, args.data))
                self.n_classes = 17
            elif args.data == 'Wisdm':
                self.input_size_W = 200
                self.input_size_H = 3
                self.classifier.append(
                    self.My_build_classifier_har(nIn , 6, i + 1, args.data))
                # print(nIn, args.grFactor[-1])
                self.n_classes = 6
            elif args.data == 'Usc':
                self.input_size_W = 512
                self.input_size_H = 6
                self.classifier.append(
                    self.My_build_classifier_har(nIn , 12, i + 1, args.data))
                # print(nIn, args.grFactor[-1])
                self.n_classes = 12
            elif args.data == 'Pampa2':
                self.input_size_W = 171
                self.input_size_H = 40
                self.classifier.append(
                    self.My_build_classifier_har(nIn , 12, i + 1, args.data))
                # print(nIn, args.grFactor[-1])
                self.n_classes = 12
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
        if layers == []:
            layers.append(Mynet_middle_layer(nIn, nIn))
        # layers.append(Mynet_middle_layer(nIn, nIn))
        layers.append(Myresnet_Basic(nIn, nIn * 2))
        nIn = nIn * 2
        return nn.Sequential(*layers), nIn
    
    def My_build_classifier_har(self, nIn, num_classes, cur_Block, dataset_name):
        conv = nn.Sequential(
            # ConvBasic(nIn, nIn, kernel=(3, 1), stride=(2, 1), padding=(1, 0)),
            # ConvBasic(nIn, nIn, kernel=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.AvgPool2d((2, 1), padding=(0, 0))
        )
        return ClassifierModule(conv, nIn, num_classes, cur_Block, dataset_name)
    
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
            phi = self.classifier[i].m(x)
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
            phi = self.classifier[i].m(x)
            phi = phi.view(phi.size(0), -1)
            res.append(self.classifier[i].linear(phi))
            phi_out.append(phi)
        return res, phi_out
