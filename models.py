import math
from torch import nn

from layers import BaseMaskedLayer, MaskedConv2d, MaskedLinear, MaskedLSTMCell, LSTMCell, StateSelect, MaskedResidual


class BaseModel:
    def __init__(self, net):
        self.net = net
        self.init_parameters()

    def __call__(self, x):
        return self.net(x)

    def __getitem__(self, index):
        return self.net[index]

    def init_parameters(self, a=math.sqrt(5)):
        def init_weights(layer):
            if isinstance(layer, MaskedLinear):
                nn.init.kaiming_uniform_(layer.weight, a)
            elif isinstance(layer, MaskedConv2d):
                nn.init.xavier_normal_(layer.weight)

        self.net.apply(init_weights)

    def parameters(self):
        return self.net.parameters()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def get_sparse_term(self):
        term = 0
        for layer in self.net:
            if isinstance(layer, BaseMaskedLayer):
                term += layer.get_sparse_term()
        return term

    def get_remain(self):
        weight_remain, weight_numel = 0, 0
        remain_ratios = []
        for layer in self.net:
            if isinstance(layer, BaseMaskedLayer):
                weight_remain += layer.weight_numel * layer.remain_ratio
                weight_numel += layer.weight_numel
                remain_ratios.append(layer.remain_ratio)
        return weight_remain / weight_numel, remain_ratios


class LeNet_300_100(BaseModel):
    def __init__(self):
        net = nn.Sequential(
            MaskedLinear(28 * 28, 300), nn.ReLU(),
            MaskedLinear(300, 100), nn.ReLU(),
            MaskedLinear(100, 10))
        super(LeNet_300_100, self).__init__(net)


class LeNet_5_Caffe(BaseModel):
    def __init__(self):
        net = nn.Sequential(
            MaskedConv2d(1, 20, 5, padding=0, bias=True), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MaskedConv2d(20, 50, 5, padding=0, bias=True), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            MaskedLinear(50 * 4 * 4, 500), nn.ReLU(),
            MaskedLinear(500, 10))
        super(LeNet_5_Caffe, self).__init__(net)


class MaskedLSTM(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        net = nn.Sequential()
        for i in range(num_layers):
            net.add_module(f"lstm{i}", MaskedLSTMCell(input_size, hidden_size))
            input_size = hidden_size
        net.add_module("select", StateSelect())  # 选取最后一个时间步的状态作为linear的输入
        net.add_module("fc", MaskedLinear(hidden_size, output_size))

        super(MaskedLSTM, self).__init__(net)

    def init_lstm_state(self, batch_size, num_hiddens):
        for layer in self.net:
            if isinstance(layer, MaskedLSTMCell):
                layer.init_state(batch_size, num_hiddens)

    def reset_state(self):
        for layer in self.net:
            if isinstance(layer, MaskedLSTMCell):
                layer.reset_state()


class LSTM(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        net = nn.Sequential()
        for i in range(num_layers):
            net.add_module(f"lstm{i}", LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        net.add_module("select", StateSelect())  # 选取最后一个时间步的状态作为linear的输入
        net.add_module("fc", MaskedLinear(hidden_size, output_size))

        super(LSTM, self).__init__(net)

    def init_lstm_state(self, batch_size, num_hiddens):
        for layer in self.net:
            if isinstance(layer, LSTMCell):
                layer.init_state(batch_size, num_hiddens)

    def reset_state(self):
        for layer in self.net:
            if isinstance(layer, LSTMCell):
                layer.reset_state()


class MaskedVgg(BaseModel):
    def __init__(self, conv_arch=((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))):  # vgg16
        net = self._vgg(conv_arch)
        super(MaskedVgg, self).__init__(net)

    @staticmethod
    def _vgg_block(num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(MaskedConv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return layers

    def _vgg(self, conv_arch):
        conv_blks = []
        in_channels = 3
        # 卷积层部分
        for (num_convs, out_channels) in conv_arch:
            conv_blks.extend(self._vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(
            *conv_blks, nn.Flatten(),
            # 全连接层部分
            MaskedLinear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            # MaskedLinear(512, 512), nn.ReLU(), nn.Dropout(0.4), # 似乎只提到fc1和fc2
            MaskedLinear(512, 10))


class MaskedWideResNet(BaseModel):
    def __init__(self, widen_factor):
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        # 1st conv before any network block
        conv1 = MaskedConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        conv = []
        for i in range(1, 4):
            conv.extend(self._resnet_block(nChannels[i-1], nChannels[i], 2))
        net = nn.Sequential(
            conv1, nn.BatchNorm2d(nChannels[0]), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *conv,
            nn.BatchNorm2d(64 * widen_factor), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), MaskedLinear(512, 10))
        super(MaskedWideResNet, self).__init__(net)

    @staticmethod
    def _resnet_block(input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(MaskedResidual(input_channels, num_channels,
                                          use_1x1conv=True, strides=2))
            else:
                blk.append(MaskedResidual(num_channels, num_channels))
        return blk
