import torch
from torch import nn

from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from torch.optim import lr_scheduler

from dataload import load_minst, load_cifar_10
from layers import UnitStepLayer
from models import LeNet_300_100, LeNet_5_Caffe, MaskedLSTM, LSTM, MaskedVgg, MaskedWideResNet
from train import train_net, train_1epoch, train_rnn, train_rnn_1epoch


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-b', '-C1', '-g', '-r', '-m', '-C5'), figsize=(3.5, 2.5), axes=None, twins=False, ylim2=None):
    backend_inline.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
    axes = axes if axes else plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    if twins:
        ax2 = axes.twinx()
        ax2.set_ylim(ylim2)
        ax2.set_ylabel(ylabel[1])
    i = 0
    ax = axes
    f = []
    for x, y, fmt in zip(X, Y, fmts):
        if twins and (i > 0):
            ax = ax2
        if len(x):
            h, = ax.plot(x, y, fmt)
        else:
            h, = ax.plot(y, fmt)
        f.append(h)
        i += 1
    ax.legend(f, legend)
    set_axes(axes, xlabel, ylabel[0], xlim, ylim, xscale, yscale)


def plot_all(epoch, acc, model_remain, layer_remain, xlabel, legend, figsize=(7.5, 3.5), ylim2=[0.93, 0.99]):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.55)
    plot(torch.arange(epoch) + 1, [model_remain, acc], xlabel=xlabel, ylabel=['model remain ratio', 'test acc'],
         legend=['model remain ratio', 'test acc'], xlim=[1, epoch], axes=axes[0], twins=True, ylim2=ylim2)
    plot(torch.arange(epoch) + 1, layer_remain, xlabel=xlabel, ylabel=['layer remain ratio'], legend=legend,
         xlim=[1, epoch], axes=axes[1])
    # plt.show()


class Tester:
    @staticmethod
    def test_usl():
        a = torch.tensor([-2, -0.7, -0.1, 0, 0.5, 1], requires_grad=True)
        usl = UnitStepLayer()
        out = usl(a)
        out.backward(torch.ones_like(a))
        assert torch.tensor([0, 0.4, 1.6, 2, 0.4, 0.4]).equal(a.grad)

    @staticmethod
    def test_lenet_300_100(alpha=0.0005):
        epoch, batch_size, lr = 20, 64, 0.01

        net = LeNet_300_100()
        loss = nn.CrossEntropyLoss()
        trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

        train_iter, test_iter = load_minst(batch_size, flatten=True)

        acc, model_remain, layer_remain = train_net(net, loss, trainer, alpha, train_iter, test_iter, epoch)
        plot_all(epoch, acc, model_remain, layer_remain, 'epoch', ['fc1', 'fc2', 'fc3'])
        plt.savefig("res/lenet-300-100.png")

    @staticmethod
    def test_lenet_5(alpha=0.0005):
        epoch, batch_size, lr = 20, 64, 0.01

        net = LeNet_5_Caffe()
        loss = nn.CrossEntropyLoss()
        trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

        train_iter, test_iter = load_minst(batch_size)

        acc, model_remain, layer_remain = train_net(net, loss, trainer, alpha, train_iter, test_iter, epoch)
        plot_all(epoch, acc, model_remain, layer_remain, 'epoch', ['conv1', 'conv2', 'fc1', 'fc2'], ylim2=[0.97, 1])
        plt.savefig("res/lenet-5.png")

    @staticmethod
    def test_lenet_1epoch(alpha=0.0005):
        batch_size, lr = 64, 0.01

        net = LeNet_300_100()
        loss = nn.CrossEntropyLoss()
        trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

        train_iter, _ = load_minst(batch_size, flatten=True)
        steps = len(train_iter)

        layer_remain = train_1epoch(net, loss, trainer, alpha, train_iter)
        plot(torch.arange(steps) + 1, layer_remain, xlabel='train steps', ylabel=['layer remain ratio'],
             legend=['fc1', 'fc2', 'fc3'], xlim=[1, steps])
        # plt.show()
        plt.savefig("res/1epoch.png")

    @staticmethod
    def test_masked_lstm(hidden_size, file, alpha=0.001):
        epoch, batch_size, lr = 20, 100, 0.001

        net = MaskedLSTM(28, hidden_size, 2, 10)
        net.init_lstm_state(batch_size, hidden_size)
        loss = nn.CrossEntropyLoss()
        trainer = torch.optim.Adam(net.parameters(), lr=lr)

        train_iter, test_iter = load_minst(batch_size, rnn=True)

        acc, model_remain, layer_remain = train_rnn(net, loss, trainer, alpha, train_iter, test_iter, epoch)
        plot_all(epoch, acc, model_remain, layer_remain, 'epoch', ['lstm1', 'lstm2', 'fc'], ylim2=[0.97, 0.99])
        plt.savefig(file)

    @staticmethod
    def test_masked_lstm_1epoch(hidden_size, file, alpha=0.001):
        batch_size, lr = 100, 0.001

        net = MaskedLSTM(28, hidden_size, 2, 10)
        net.init_lstm_state(batch_size, hidden_size)
        loss = nn.CrossEntropyLoss()
        trainer = torch.optim.Adam(net.parameters(), lr=lr)

        train_iter, test_iter = load_minst(batch_size, rnn=True)

        steps = 600
        acc, model_remain, layer_remain = train_rnn_1epoch(net, loss, trainer, alpha, train_iter, test_iter)
        plot(torch.arange(steps) + 1, layer_remain, xlabel='train steps', ylabel=['layer remain ratio'],
             legend=['lstm1', 'lstm2', 'fc'], xlim=[1, steps])
        plt.savefig(file)

    @staticmethod
    def test_lstm(hidden_size, file, alpha=0.001):
        epoch, batch_size, lr = 20, 100, 0.001

        net = LSTM(28, hidden_size, 2, 10)
        net.init_lstm_state(batch_size, hidden_size)
        loss = nn.CrossEntropyLoss()
        trainer = torch.optim.Adam(net.parameters(), lr=lr)

        train_iter, test_iter = load_minst(batch_size, rnn=True)

        acc, model_remain, layer_remain = train_rnn(net, loss, trainer, alpha, train_iter, test_iter, epoch)
        plot_all(epoch, acc, model_remain, layer_remain, 'epoch', ['lstm1', 'lstm2', 'fc'], ylim2=[0.97, 0.99])
        plt.savefig(file)

    @staticmethod
    def test_masked_vgg16(lr=0.01, alpha=5e-6):
        epoch, batch_size = 160, 64

        net = MaskedVgg()
        loss = nn.CrossEntropyLoss()
        trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[80, 120], gamma=0.1)

        train_iter, test_iter = load_cifar_10(batch_size)

        acc, model_remain, layer_remain = train_net(net, loss, trainer, alpha, train_iter, test_iter, epoch, scheduler)
        plot_all(epoch, acc, model_remain, layer_remain[[0, 2, 4, 7, 13, -1]], 'epoch',
                 ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2'], ylim2=[0.3, 1.])
        plt.savefig("res/vgg-16.png")

    @staticmethod
    def test_masked_wideres(wide_f, lr=0.1, alpha=5e-6):
        epoch, batch_size = 160, 64

        net = MaskedWideResNet(wide_f)
        loss = nn.CrossEntropyLoss()
        trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[80, 120], gamma=0.1)

        train_iter, test_iter = load_cifar_10(batch_size)

        acc, model_remain, layer_remain = train_net(net, loss, trainer, alpha, train_iter, test_iter, epoch,
                                                    scheduler)
        plot_all(epoch, acc, model_remain, layer_remain[[0, 1, 3, 5, 6, -1]], 'epoch',
                 ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc'], ylim2=[0.3, 1.])
        plt.savefig("res/wide_res.png")

    @staticmethod
    def test_diff_a(wide_f, lr=0.1):
        epoch, batch_size = 160, 64
        test_acc, model_remains = [], []

        train_iter, test_iter = load_cifar_10(batch_size)
        loss = nn.CrossEntropyLoss()

        alphas = torch.tensor([1e-7, 1e-6, 1e-5, 1e-4])
        for alpha in alphas:
            net = MaskedWideResNet(wide_f)
            trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
            scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[80, 120], gamma=0.1)

            acc, model_remain, _ = train_net(net, loss, trainer, alpha, train_iter, test_iter, epoch, scheduler)
            test_acc.append(acc[-1])
            model_remains.append(model_remain[-1])

        plot(alphas, [torch.tensor(model_remains), torch.tensor(test_acc)], xlabel='alpha',
             ylabel=['model remain ratio', 'test_acc'], legend=['model remain ratio', 'test_acc'],
             twins=True, ylim2=[0.8, 0.95], xscale='log')



    @staticmethod
    def test_model():
        net = MaskedWideResNet(8)
        X = torch.randn((2, 3, 32, 32))
        for layer in net:
            X = layer(X)
            print(X.shape)
            input()

