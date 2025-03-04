import torch
import torch.nn as nn

# 定义输入数据的维度
batch_size = 1
sequence_length = 200
input_size = 3

# 创建一个LSTM模型
lstm_model = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=1, batch_first=True)

# 生成随机输入数据
input_data = torch.randn(batch_size, sequence_length, input_size)

# 将输入数据输入到LSTM模型中
output, (h_n, c_n) = lstm_model(input_data)

# 输出的维度
print("Output shape:", output.shape)