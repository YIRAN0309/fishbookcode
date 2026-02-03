import numpy as np
from collections import OrderedDict
import struct
def load_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)

def load_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
def data_load():
    train_images = load_images("../3/data/train-images-idx3-ubyte")
    train_labels = load_labels("../3/data/train-labels-idx1-ubyte")
    test_images = load_images("../3/data/t10k-images-idx3-ubyte")
    test_labels = load_labels("../3/data/t10k-labels-idx1-ubyte")

    print(train_images.shape, train_labels.shape)
    print(test_images.shape, test_labels.shape)
    train_images_flat = train_images.reshape(train_images.shape[0], -1)
    print(train_images_flat.shape, train_labels.shape)
    test_images_flat = test_images.reshape(test_images.shape[0], -1)
    print(test_images_flat.shape, test_labels.shape)
    return train_images_flat, train_labels, test_images_flat, test_labels
#5.6.1 Affine 层
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    #前向传播层
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    #这里的导数就是转置，书中没有给出详细的解释
    #求一个就是另外一个的转置
    #我们要做的就是更新参数
    #db这里是因为 前面做运算的时候把每一个参数都加了一个b，所以求导的时候 前面的都是b的系数 所以要把他们按照行加起来
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
def softmax(a):
    if a.ndim == 2:  # batch
        c = np.max(a, axis=1, keepdims=True)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
        y = exp_a / sum_exp_a
    else:  # 单样本
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
    return y
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    def backward(self, dout):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else :
            dx = self.y.copy()
            #又是这个花式索引 索引索引索引
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
#5.2.2 sigmoid层
class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    def backward(self,dout):
        # 这个还是比较好求导数的 主要是记一下当前的状态的参数，看一下求导数跟哪些值有关
        dx = dout * (1.0 - self.out) * self.out
        return dx
class Relu:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        #先找出x<=0的部分的序号
        self.mask = (x <= 0)
        #然后对于这一部分 变成0 其他部分不变
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self, dout):
        #<=0的部分保持求导就是0 其他部分因为导数是1 所以相当于还是dout，这个花式索引到底在干什么
        #以为在滑轮滑吗
        #所以这里的self.mask存的是 输入是否小于0
        #求导数就是
        dout[self.mask] = 0
        dx = dout
        return dx
 #mini batch版交叉熵误差的实现
def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t] + delta))/ batch_size

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # 生成和x形状相同的数组
    # 这里是一个元素一个元素的遍历吗
    for idx in np.ndindex(x.shape):  # 支持任意维度
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #初始化权重
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = weight_init_std * np.random.randn(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = weight_init_std * np.random.randn(output_size)

        #生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params["W1"], self.params["b1"])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params["W2"], self.params["b2"])
        self.lastLayer = SoftmaxWithLoss()
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 :t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / y.shape[0]
        return accuracy
    def gradient(self, x, t):
        #forward
        self.loss(x, t)
        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        #一起返回
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads



    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads[ 'W1' ] = numerical_gradient( loss_W, self.params[ 'W1' ] )
        grads[ 'b1' ] = numerical_gradient( loss_W, self.params[ 'b1' ] )
        grads[ 'W2' ] = numerical_gradient( loss_W, self.params[ 'W2' ] )
        grads[ 'b2' ] = numerical_gradient( loss_W, self.params[ 'b2' ] )
        return grads

if __name__ == "__main__":

    x_train, t_train, x_test, t_test = data_load()
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    #从下标 0 开始，到下标 3 之前（不包含 3）为止。
    """    x_batch = x_train[:3]
        t_batch = t_train[:3]
        grad_numerical = network.numerical_gradient(x_batch, t_batch)
        grad_backprop = network.gradient(x_batch, t_batch)
        for k in grad_backprop.keys():
            diff = np.average(np.abs(grad_backprop[k] - grad_numerical[k]))
            print(k,diff)"""
    #下面是使用迷你batch 进行训练
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    """这里的类是网络结构类 所以存储的是有关网络的参数，其实就是网络的权重 之前考虑过的为什么不能存y这种目标值也在此得到解答"""
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    for i in range(iters_num):
        # 获取minibatch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        # 计算梯度
        grad = network.gradient(x_batch, t_batch)
        # 更新参数
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # ✅ 同步参数到各层（关键）这里是应该是没用的 实际验证效果也是没用的 不知道为什么让这样更改 gpt
        #gpt说的这样改
        """network.layers['Affine1'].W = network.params['W1']
        network.layers['Affine1'].b = network.params['b1']
        network.layers['Affine2'].W = network.params['W2']
        network.layers['Affine2'].b = network.params['b2']"""

        # 记录学习过程
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % 100 == 0:
            print(f"迭代次数：{i:5d}|当前loss:{loss:.4f}")
