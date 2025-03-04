import torch.nn as nn
import torch
import math
import pdb
import numpy as np


U = 128 # small 32, medium 64, large 128
class ClassifierModule(nn.Module):
    def __init__(self, channel, num_classes, cur_block = None, dataset_name = None):
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
            
        self.linear = nn.Linear(U, num_classes)# default:channel UCI:124*1*9, UniMib:124*1*3, Wisdm:124*2*3
        

    def forward(self, x):
        # x = x.permute(1,0,2)
        x = x[:, -1, :]
        out = self.linear(x)
        return out
    
class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.nBlocks = args.nBlocks
        if args.data == 'UCI':
            nIn = 9
        elif args.data == 'UniMib':
            nIn = 3
        elif args.data == 'Wisdm':
            nIn = 3
        for i in range(self.nBlocks):
            print(' ********************** Block {} '
                  ' **********************'.format(i + 1))
            m, nIn = \
                self.My_build_block(nIn)
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
            
    def My_build_block(self, nIn, LSTM_unit=U):# small 32, medium 64, large 128
        layers = []
        layers.append(nn.LSTM(int(nIn), LSTM_unit, num_layers=1, batch_first=True))
        nIn = LSTM_unit
        return nn.Sequential(*layers), nIn
    
    def My_build_classifier_har(self, nIn, num_classes, cur_Block, dataset_name):

        return ClassifierModule( nIn, num_classes, cur_Block, dataset_name)
    
    def forward(self, x):
        res = []
        # x = x.permute(2, 0, 3, 1)
        # x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.squeeze(1)
        for i in range(self.nBlocks):
            x, _ = self.blocks[i](x)
            res.append(self.classifier[i](x))
        return res

    
    def predict(self, x):
        """
        This function was coded for MC integration of LA inference
        it outputs final output (res) and mean feature of the image (phi_out) 
        """
        res = []
        phi_out = []
        x = x.squeeze(1)
        for i in range(self.nBlocks):
            x, _ = self.blocks[i](x)
            # classifier module forward
            phi = x[:, -1, :]
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
        x = x.squeeze(1)
        for i in range(until_block):
            x, _ = self.blocks[i](x)
            # classifier module forward
            phi = x[:, -1, :]
            res.append(self.classifier[i].linear(phi))
            phi_out.append(phi)
        return res, phi_out
