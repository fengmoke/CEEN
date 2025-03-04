import torch.nn as nn
import torch
import math
import pdb
import numpy as np
import sys

conv_list       =   {
                    'general':   [[ (5,1), (3,1), (0,0) ],
                                  [ (3,1), (2,1), (0,0) ],
                                  [ (3,1), (2,1), (0,0) ],
                                  [ (3,1), (2,1), (0,0) ]],
                        }

def get_data_size(data_name, is_sup = False ):

    Model_Seen_Sup_F_Size = {
        'UCI': (     1, 1, 128, 9     ) ,
        'Wisdm': (     1, 1, 200,  3     ) ,
        'UniMib': (     1, 1, 151,  3     ) ,
    }
    size_dict = Model_Seen_Sup_F_Size
    if data_name in size_dict:
        pass
    else:
        raise Exception( 'please input correct data name')
    return size_dict[data_name]

def GetFeatureMapSizeByConv(data_name,idex_layer):
    size        = get_data_size(data_name)[2:]
    conv_size   = conv_list['general']
    h,w         = size
    if idex_layer > 0:
        for i in range(idex_layer):
            # no pooling
            h   =  (h - conv_size[i][0][0])/(conv_size[i][1][0]) + 1
            w   =  (w - conv_size[i][0][1])/(conv_size[i][1][1]) + 1
        return ( int(h) , int(w) )
    else:
        raise  ValueError(f'check your idex_layer')
    
class DCL_first_layer(nn.Module):
    def __init__(self, nIn, nOut, data_name, Kernel_Size = (5, 1), Stride = (3, 1)) -> None:
        super(DCL_first_layer, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, Kernel_Size, Stride),
            nn.ReLU()
        )
    def forward(self, x):
        out = self.net(x)
        return out

class DCL_middle_layer(nn.Module):
    def __init__(self, nIn, nOut,  Kernel_Size = (3, 1), Stride = (2, 1), LSTM_units=128) -> None:
        super(DCL_middle_layer, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, Kernel_Size, Stride),
            nn.ReLU()
        )
    def forward(self, x):
        out = self.net(x)
        return out

class ClassifierModule(nn.Module):
    def __init__(self, m, channel, num_classes, cur_block = None, dataset_name = None):
        super(ClassifierModule, self).__init__()
        if dataset_name == 'Wisdm':
            if cur_block == 1:
                channel = 128*32*3
            elif cur_block == 2:
                channel = 256*16*3
            elif cur_block == 3:
                channel = 512*7*3
            elif cur_block == 4:
                channel = 1024*3*3
        elif dataset_name == 'UCI':
            if cur_block == 1:
                channel = 128*20*9
            elif cur_block == 2:
                channel = 256*10*9
            elif cur_block == 3:
                channel = 512*4*9
            elif cur_block == 4:
                channel = 1024*2*9
        elif dataset_name == 'UniMib':
            if cur_block == 1:
                channel = 128*24*3
            elif cur_block == 2:
                channel = 256*12*3
            elif cur_block == 3:
                channel = 512*5*3
            elif cur_block == 4:
                channel = 1024*2*3
        elif dataset_name == 'Usc':
            if cur_block == 1:
                channel = 128*84*6
            elif cur_block == 2:
                channel = 256*42*6
            elif cur_block == 3:
                channel = 512*20*6
            elif cur_block == 4:
                channel = 1024*10*6
        elif dataset_name == 'Pampa2':
            if cur_block == 1:
                channel = 128*28*40
            elif cur_block == 2:
                channel = 256*13*40
            elif cur_block == 3:
                channel = 512*6*40
            elif cur_block == 4:
                channel = 1024*3*40
        self.m = m
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(128, num_classes)# default:channel UCI:124*1*9, UniMib:124*1*3, Wisdm:124*2*3
        

    def forward(self, x):
        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.dropout(x)
        x, h = self.m(x)
        # x = x.permute(1,0,2)
        last_feature = x[-1, :, :]
        out = self.linear(last_feature)
        return out
    
class DCL(nn.Module):
    def __init__(self, args):
        super(DCL, self).__init__()
        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.nBlocks = args.nBlocks
        nIn = args.nChannels
        for i in range(self.nBlocks):
            print(' ********************** Block {} '
                  ' **********************'.format(i + 1))
            m, nIn = \
                self.My_build_block(nIn, i, args.data)
            self.blocks.append(m)

            if args.data == 'UCI':
                self.classifier.append(
                    self.My_build_classifier_har(nIn, 6, i + 1, args.data))
                self.n_classes = 6
            elif args.data == 'UniMib':
                self.classifier.append(
                    self.My_build_classifier_har(nIn, 17, i + 1, args.data))
                self.n_classes = 17
            elif args.data == 'Wisdm':
                self.classifier.append(
                    self.My_build_classifier_har(nIn , 6, i + 1, args.data))
                # print(nIn, args.grFactor[-1])
                self.n_classes = 6
            elif args.data == 'Usc':
                self.classifier.append(
                    self.My_build_classifier_har(nIn , 12, i + 1, args.data))
                # print(nIn, args.grFactor[-1])
                self.n_classes = 12
            elif args.data == 'Pampa2':
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
            
    def My_build_block(self, nIn, cur_block, data_name):
        layers = [DCL_first_layer(1, nIn, data_name, Kernel_Size=(5, 1))] if cur_block ==0 else []
        if layers == []:
            if data_name == 'UCI':
                layers.append(DCL_middle_layer(nIn, nIn*2))
                if cur_block > 0:
                    nIn *= 2
            else:
                layers.append(DCL_middle_layer(nIn, nIn))
        return nn.Sequential(*layers), nIn
    
    def My_build_classifier_har(self, nIn, num_classes, cur_Block, dataset_name, LSTM_units=128):
        if dataset_name == 'UCI':
            self.rep_feat = 9
            lstm =  nn.LSTM( self.rep_feat * nIn, LSTM_units, num_layers=1 )
        else:
            self.rep_feat = 3
            self.rep_feat = GetFeatureMapSizeByConv(dataset_name, cur_Block)
            lstm =  nn.LSTM( self.rep_feat[1] * nIn, LSTM_units, num_layers=1 )
        return ClassifierModule( lstm,  nIn, num_classes, cur_Block, dataset_name)
    
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
            phi = x.permute(2, 0, 3, 1)
            phi = phi.reshape(phi.shape[0], phi.shape[1], -1)
            phi = self.classifier[i].dropout(phi)
            phi, _ = self.classifier[i].m(phi)
            phi = phi[-1, :, :]
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
        phi_out =[]
        for i in range(until_block):
            x = self.blocks[i](x)
            # classifier module forward
            phi = x.permute(2, 0, 3, 1)
            phi = phi.reshape(phi.shape[0], phi.shape[1], -1)
            phi = self.classifier[i].dropout(phi)
            phi, _ = self.classifier[i].m(phi)
            phi = phi[-1, :, :]
            phi = phi.view(phi.size(0), -1)
            res.append(self.classifier[i].linear(phi))
            phi_out.append(phi)
        return res, phi_out
