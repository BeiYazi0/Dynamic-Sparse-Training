import torch


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat.type(y.dtype) == y)
    return torch.sum(cmp)


def evaluate_accuracy(net, data_iter, rnn=False):
    net.eval()  # 设置为评估模式
    # 正确预测的数量，总预测的数量
    metric = torch.zeros(2)
    with torch.no_grad():
        for X, y in data_iter:
            if rnn:
                X = X.permute(1, 0, 2)
                net.reset_state()
            metric[0] += accuracy(net(X), y)
            metric[1] += y.numel()
    return metric[0] / metric[1]


def train_net(net, loss, trainer, alpha, train_iter, test_iter, epochs, scheduler=None):
    model_remain_ratios = []
    layer_remain_ratios = []
    test_acc_history = []
    for epoch in range(epochs):
        net.train()
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y) + alpha * net.get_sparse_term()

            trainer.zero_grad()  # 清除了优化器中的grad
            l.backward()  # 通过进行反向传播来计算梯度
            trainer.step()  # 通过调用优化器来更新模型参数
        if scheduler:
            scheduler.step()

        model_remain_ratio, layer_remain_ratio = net.get_remain()
        model_remain_ratios.append(model_remain_ratio)
        layer_remain_ratios.append(layer_remain_ratio)

        test_acc = evaluate_accuracy(net, test_iter)
        test_acc_history.append(test_acc)

        print("epoch: %d    model_remain: %.2f%%   test_acc: %.2f%%" % (epoch+1, model_remain_ratio * 100, test_acc * 100))
    return torch.tensor(test_acc_history), torch.tensor(model_remain_ratios), torch.tensor(layer_remain_ratios).T


def train_1epoch(net, loss, trainer, alpha, train_iter):
    layer_remain_ratios = []

    net.train()
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y) + alpha * net.get_sparse_term()

        trainer.zero_grad()  # 清除了优化器中的grad
        l.backward()  # 通过进行反向传播来计算梯度
        trainer.step()  # 通过调用优化器来更新模型参数

        _, layer_remain_ratio = net.get_remain()
        layer_remain_ratios.append(layer_remain_ratio)

    return torch.tensor(layer_remain_ratios).T


def train_rnn(net, loss, trainer, alpha, train_iter, test_iter, epochs):
    model_remain_ratios = []
    layer_remain_ratios = []
    test_acc_history = []
    for epoch in range(epochs):
        net.train()
        for X, y in train_iter:
            net.reset_state()
            y_hat = net(X.permute(1, 0, 2)) # (batch_size, time_step, feature) -> (time_step, batch_size, feature)
            l = loss(y_hat, y) + alpha * net.get_sparse_term()
            # print(net[1].thresholds[0].grad, net[3].thresholds.grad)

            trainer.zero_grad()  # 清除了优化器中的grad
            l.backward()  # 通过进行反向传播来计算梯度
            trainer.step()  # 通过调用优化器来更新模型参数

        model_remain_ratio, layer_remain_ratio = net.get_remain()
        model_remain_ratios.append(model_remain_ratio)
        layer_remain_ratios.append(layer_remain_ratio)

        test_acc = evaluate_accuracy(net, test_iter, rnn=True)
        test_acc_history.append(test_acc)

        print("model_remain: %.2f%%   test_acc: %.2f%%" % (model_remain_ratio * 100, test_acc * 100))
    return torch.tensor(test_acc_history), torch.tensor(model_remain_ratios), torch.tensor(layer_remain_ratios).T


def train_rnn_1epoch(net, loss, trainer, alpha, train_iter, test_iter):
    layer_remain_ratios = []

    net.train()
    for X, y in train_iter:
        net.reset_state()
        y_hat = net(X.permute(1, 0, 2)) # (batch_size, time_step, feature) -> (time_step, batch_size, feature)
        l = loss(y_hat, y) + alpha * net.get_sparse_term()
        # print(net[1].thresholds[0].grad, net[3].thresholds.grad)

        trainer.zero_grad()  # 清除了优化器中的grad
        l.backward()  # 通过进行反向传播来计算梯度
        trainer.step()  # 通过调用优化器来更新模型参数

        _, layer_remain_ratio = net.get_remain()
        layer_remain_ratios.append(layer_remain_ratio)
        print(layer_remain_ratio)

    return torch.tensor(layer_remain_ratios).T
