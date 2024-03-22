import math

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F


class UnitStepFunction(Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return (inp >= 0) * torch.ones_like(inp)

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        grad_input = grad_output.clone()

        abs_inp = torch.abs(inp)
        mask1 = (abs_inp <= 0.4)
        mask2 = (abs_inp > 0.4) & (abs_inp <= 1)
        res = (2 - 4 * abs_inp) * mask1 + 0.4 * mask2
        return grad_input * res


class UnitStepLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = UnitStepFunction.apply(x)
        return out


class BaseMaskedLayer(nn.Module):
    def __init__(self):
        super(BaseMaskedLayer, self).__init__()
        self.usl = UnitStepLayer()
        self._ratio = torch.tensor([1.])
        self._weight_num = 0

    def forward(self, *args):
        raise NotImplementedError

    def get_sparse_term(self):
        return torch.sum(torch.exp(-self.thresholds))

    @property
    def weight_numel(self):
        return self._weight_num

    @property
    def remain_ratio(self):
        return self._ratio.item()


class MaskedLinear(BaseMaskedLayer):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True):
        super(MaskLinear, self).__init__()
        self._weight_num = out_features * in_features

        self.thresholds = nn.Parameter(torch.zeros((out_features, 1)))
        self.weight = nn.Parameter(torch.zeros((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.randn((out_features,)))
        else:
            self.bias = None

    def forward(self, x):
        # t will be reset to zero if more than 99% elements in the mask are zero
        if self._ratio < 0.01:
            with torch.no_grad():
                self.thresholds.data.fill_(0.)

        Q = torch.abs(self.weight) - self.thresholds
        mask = self.usl(Q)

        self._ratio = torch.sum(mask) / self._weight_num

        masked_weight = self.weight * mask
        output = nn.functional.linear(x, masked_weight, self.bias)
        return output


class MaskedConv2d(BaseMaskedLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        super(MaskedConv2d, self).__init__()
        self._weight_num = out_channels * in_channels * kernel_size * kernel_size
        self.weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        self.thresholds = nn.Parameter(torch.zeros((out_channels, 1)))
        self.weight = nn.Parameter(torch.zeros(self.weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.randn((out_channels,)))
        else:
            self.bias = None

    def forward(self, x):
        # t will be reset to zero if more than 99% elements in the mask are zero
        if self._ratio < 0.01:
            with torch.no_grad():
                self.thresholds.data.fill_(0.)

        weight = torch.abs(self.weight).view(self.weight_shape[0], -1)
        Q = weight - self.thresholds
        mask = self.usl(Q)

        self._ratio = torch.sum(mask)/self._weight_num

        mask = mask.view(self.weight_shape)
        masked_weight = self.weight * mask
        output = nn.functional.conv2d(x, masked_weight, bias=self.bias, stride=self.stride, padding=self.padding)
        return output


class MaskedLSTMCell(BaseMaskedLayer):
    def __init__(self, input_size, hidden_size):
        super(MaskedLSTMCell, self).__init__()
        self._weight_num = 4 * (input_size * hidden_size + hidden_size * hidden_size)

        # self.params, self.bias, self.thresholds = self._get_lstm_params(input_size, hidden_size)
        stdv = 1.0 / math.sqrt(hidden_size)
        self.W_x = nn.Parameter(torch.zeros((4, hidden_size, input_size)).uniform_(-stdv, stdv))
        self.W_h = nn.Parameter(torch.zeros((4, hidden_size, hidden_size)).uniform_(-stdv, stdv))
        self.bias = nn.Parameter(torch.zeros((4, hidden_size)))
        self.thresholds = nn.Parameter(torch.zeros((hidden_size, 8)))
        self.state = (None, None)

    def init_state(self, batch_size, num_hiddens):
        self.state = torch.zeros((batch_size, num_hiddens)), torch.zeros((batch_size, num_hiddens))

    def reset_state(self):
        (H, C) = self.state
        H.fill_(0.)
        C.fill_(0.)
        self.state = (H, C)

    def forward(self, inputs):
        # t will be reset to zero if more than 99% elements in the mask are zero
        if self._ratio < 0.01:
            with torch.no_grad():
                self.thresholds.data.fill_(0.)

        cnt = 0
        masked_params = []
        for i in range(4):
            weight_x = self.W_x[i]
            thresholds = self.thresholds[:, i]
            Q = torch.abs(weight_x) - thresholds.view(-1, 1)
            mask = self.usl(Q)
            cnt += torch.sum(mask)
            masked_params.append((weight_x * mask).transpose(1, 0))  # 转置方便下面实现WT @ X

            weight_h = self.W_h[i]
            thresholds = self.thresholds[:, i+4]
            Q = torch.abs(weight_h) - thresholds.view(-1, 1)
            mask = self.usl(Q)
            cnt += torch.sum(mask)
            masked_params.append((weight_h * mask).transpose(1, 0))  # 转置方便下面实现WT @ X
        self._ratio = torch.tensor([cnt / self._weight_num])

        [W_xi, W_hi, W_xf, W_hf, W_xo, W_ho, W_xc, W_hc] = masked_params
        [b_i, b_f, b_o, b_c] = self.bias

        (H, C) = self.state
        outputs = []  # 各个时间步的输出
        for X in inputs:
            I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
            F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
            O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
            C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
            C = F * C + I * C_tilda
            H = O * torch.tanh(C)
            outputs.append(H)
        self.state = (H.detach(), C.detach())  # Wraps hidden states in new Tensors, to detach them from their history
        return torch.stack(outputs)


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self._weight_num = 4 * (input_size * hidden_size + hidden_size * hidden_size)

        # self.params, self.bias, self.thresholds = self._get_lstm_params(input_size, hidden_size)
        stdv = 1.0 / math.sqrt(hidden_size)
        self.W_x = nn.Parameter(torch.zeros((4, hidden_size, input_size)).uniform_(-stdv, stdv))
        self.W_h = nn.Parameter(torch.zeros((4, hidden_size, hidden_size)).uniform_(-stdv, stdv))
        self.bias = nn.Parameter(torch.zeros((4, hidden_size)))
        self.state = (None, None)

    def init_state(self, batch_size, num_hiddens):
        self.state = torch.zeros((batch_size, num_hiddens)), torch.zeros((batch_size, num_hiddens))

    def reset_state(self):
        (H, C) = self.state
        H *= 0
        C *= 0
        self.state = (H, C)

    def forward(self, inputs):
        masked_params = []
        for i in range(4):
            weight_x = self.W_x[i]
            masked_params.append(weight_x.transpose(1, 0))  # 转置方便下面实现WT @ X

            weight_h = self.W_h[i]
            masked_params.append(weight_h.transpose(1, 0))  # 转置方便下面实现WT @ X

        [W_xi, W_hi, W_xf, W_hf, W_xo, W_ho, W_xc, W_hc] = masked_params
        [b_i, b_f, b_o, b_c] = self.bias

        (H, C) = self.state
        outputs = []  # 各个时间步的输出
        for X in inputs:
            I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
            F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
            O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
            C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
            C = F * C + I * C_tilda
            H = O * torch.tanh(C)
            outputs.append(H)
        self.state = (H.detach(), C.detach())  # Wraps hidden states in new Tensors, to detach them from their history
        return torch.stack(outputs)


class StateSelect(nn.Module):
    def __init__(self):
        super(StateSelect, self).__init__()

    def forward(self, X):
        return X[-1]


class MaskedResidual(BaseMaskedLayer):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(MaskedResidual, self).__init__()
        del self.usl
        self.conv1 = MaskedConv2d(input_channels, num_channels, 3, padding=1, stride=strides)
        self.conv2 = MaskedConv2d(num_channels, num_channels, 3, padding=1)
        self._weight_num = self.conv1.weight_numel + self.conv2.weight_numel
        if use_1x1conv:
            self.conv3 = MaskedConv2d(input_channels, num_channels, 1, stride=strides)
            self._weight_num += self.conv3.weight_numel
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

    def get_sparse_term(self):
        R = self.conv1.get_sparse_term() + self.conv2.get_sparse_term()
        if self.conv3:
            R += self.conv3.get_sparse_term()
        return R

    @property
    def weight_numel(self):
        return self._weight_num

    @property
    def remain_ratio(self):
        cnt = self.conv1.weight_numel * self.conv1.remain_ratio + self.conv2.weight_numel * self.conv2.remain_ratio
        if self.conv3:
            cnt += self.conv3.weight_numel * self.conv3.remain_ratio
        return cnt / self._weight_num