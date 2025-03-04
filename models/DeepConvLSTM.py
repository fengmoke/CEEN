import sys 
import os 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch.nn as nn
import torch.nn.functional as F
import torch

conv_list       =   {
                    'ucihar':   [ (5,1), (1,1), (2,0) ],
                    'uschad':   [ (5,1), (1,1), (2,0) ],
                        }


def get_data_size(data_name, is_sup = False ):

    Model_Seen_Sup_F_Size = {
        'ucihar': (     1, 1, 128, 9     ) ,
        'uschad': (     1, 1, 32,  6     ) ,
    }
    size_dict = Model_Seen_Sup_F_Size
    if data_name in size_dict:
        pass
    else:
        raise Exception( 'please input correct data name')
    return size_dict[data_name]


def get_classes(data_name):
    if data_name == 'ucihar':
        classes = 6
    elif data_name == 'pamap2':
        classes = 12
    elif data_name == 'dg':
        classes = 2
    elif data_name == 'motion':
        classes = 6
    elif data_name =='unimib':
        classes = 17
    elif data_name == 'wisdm':
        classes = 6
    elif data_name == 'uschad':
        classes = 12
    elif data_name == 'oppo40':
        classes = 18
    elif data_name == 'oppo':
        classes = 18
    elif data_name == 'dsads':
        classes = 19
    else:
        raise Exception( 'please input correct data name')
    return classes


def GetFeatureMapSizeByConv(data_name,idex_layer):
    size        = get_data_size(data_name)[2:]
    conv_size   = conv_list[data_name]
    h,w         = size
    if idex_layer > 0:
        for i in range(idex_layer):
            # no pooling
            h   =  h - conv_size[0][0] + 1
            w   =  w - conv_size[0][1] + 1
        return ( h , w )
    else:
        raise  ValueError(f'check your idex_layer')


class DeepConvLstm(nn.Module):
    #DeepConvLstm
    def __init__(self, data_name , fix_channel=64, kernel_size=5, LSTM_units=128, backbone=True):
        super(DeepConvLstm, self).__init__()        
        self.backbone = backbone

        self.conv1 = nn.Conv2d(1, fix_channel, (kernel_size, 1))
        self.conv2 = nn.Conv2d(fix_channel, fix_channel, (kernel_size, 1))
        self.conv3 = nn.Conv2d(fix_channel, fix_channel, (kernel_size, 1))
        self.conv4 = nn.Conv2d(fix_channel, fix_channel, (kernel_size, 1))
       
        self.dropout = nn.Dropout(0.4)

        self.lstm = nn.LSTM( GetFeatureMapSizeByConv(data_name,4)[1] * fix_channel, LSTM_units, num_layers=2 )
        self.out_dim = LSTM_units

        self.classifier = nn.Linear(LSTM_units, get_classes(data_name))

        self.activation = nn.ReLU(True)

    def forward(self, x):
        feature_list    = []
        self.lstm.flatten_parameters()
        # x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        feature_list.append(x)
        x = self.activation(self.conv2(x))
        feature_list.append(x)
        x = self.activation(self.conv3(x))
        feature_list.append(x)
        x = self.activation(self.conv4(x))
        feature_list.append(x)
        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.dropout(x)
        x, h = self.lstm(x)
        last_feature = x[-1, :, :]
        out = self.classifier(last_feature)
        res ={}
        res['output'] = out
        res['feature_list'] = feature_list
        return res 

class DeepConvLstmAttn(nn.Module):
    #DeepConvLstmAttn
    def __init__(self, data_name , fix_channel=64, kernel_size=5, LSTM_units=128, backbone=True):
        super(DeepConvLstmAttn, self).__init__()
        
        self.backbone = backbone

        self.conv1 = nn.Conv2d(1, fix_channel, (kernel_size, 1))
        self.conv2 = nn.Conv2d(fix_channel, fix_channel, (kernel_size, 1))
        self.conv3 = nn.Conv2d(fix_channel, fix_channel, (kernel_size, 1))
        self.conv4 = nn.Conv2d(fix_channel, fix_channel, (kernel_size, 1))
        self.dropout = nn.Dropout(0.1)

        self.rep_length, self.rep_feat = GetFeatureMapSizeByConv(data_name,4)

        self.lstm = nn.LSTM( self.rep_feat * fix_channel, LSTM_units, num_layers=2 )
        self.out_dim = LSTM_units

        self.classifier = nn.Linear(LSTM_units, get_classes(data_name))

        self.activation = nn.ReLU(True)

        # attention
        self.linear_1 = nn.Linear(LSTM_units, LSTM_units)
        self.tanh = nn.Tanh()
        self.dropout_2 = nn.Dropout(0.1)
        self.linear_2 = nn.Linear(LSTM_units, 1, bias=False)


    def forward(self, x):
        # x: B,1,L,C
        feature_list    = []
        self.lstm.flatten_parameters()
        # x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        feature_list.append(x)
        x = self.activation(self.conv2(x))
        feature_list.append(x)
        x = self.activation(self.conv3(x))
        feature_list.append(x)
        x = self.activation(self.conv4(x))
        feature_list.append(x)

        # x: B,N,L,C --> L, B,C*N
        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        # attn
        x = x.permute(1,0,2)
        context = x[:, :-1, :]
        out = x[:, -1, :]
        uit = self.linear_1(context)
        uit = self.tanh(uit)
        uit = self.dropout_2(uit)
        ait = self.linear_2(uit)
        attn = torch.matmul(F.softmax(ait, dim=1).transpose(-1, -2),context).squeeze(-2)

        out = self.classifier(out+attn)

        res ={}
        res['output'] = out
        res['feature_list'] = feature_list
        return res 

             
if __name__ == "__main__":
    x  = torch.randn(2,1,128,9)
    model = DeepConvLstm('ucihar')
    print(model(x)['output'].shape)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))