#再整体梳理比较一下四五章 数值微分法 和 链式求导的区别和联系
#整体的思路都是先前向传播
#前向传播需要激活函数 涉及到的激活函数有sigmoid ReLu
#求loss 涉及到的额外的函数有交叉熵损失函数
#再反向传播 反向传播这里应该是没有直接进行反向传播 而是直接用了数值微分求导数
import numpy as np
import struct
import matplotlib.pyplot as plt
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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
#这里交叉熵损失函数的内容就是 -tk logyk
#但是因为tk的标签肯定为1 所以就直接约掉 然后取出来对应索引的yk就可以
def cross_entropy_loss(y,t):
    delta = 1e-7
    #如果他是一个一维度的，就得把他变成一个二维的【】变成【【】】才能运算
    #但是为什么不行呢 没看懂
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1,y.size )
    batch_size = y.shape[0]
    #这里基本上就能看懂了
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
#数值微分来求导数
#这里？？？？需要再仔细看一下
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)#生成和x形状相同的数组
    #这里是一个元素一个元素的遍历吗
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
    def __init__(self, input_size, hidden_size, output_size,weight_init_std = 0.01):
        self.params = {}
        self.params[ 'W1' ] = weight_init_std * np.random.randn( input_size, hidden_size)
        self.params[ 'b1' ] = np.zeros( hidden_size)
        self.params[ 'W2' ] = weight_init_std * np.random.randn( hidden_size, output_size)
        self.params[ 'b2' ] = np.zeros( output_size)
    def predict(self,x):
        W1, W2 = self.params[ 'W1' ], self.params[ 'W2' ]
        b1, b2 = self.params[ 'b1' ], self.params[ 'b2' ]
        a1 = np.dot( x, W1 ) + b1
        z1 = sigmoid( a1 )
        a2 = np.dot( z1, W2 ) + b2
        y = softmax(a2)
        return y
    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy_loss(y,t)
    def accuarcy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        acc = np.sum(y==t) / float(x.shape[0])
        return acc
    #最后一个就是更新梯度的函数
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads[ 'W1' ] = numerical_gradient( loss_W, self.params[ 'W1' ] )
        grads[ 'b1' ] = numerical_gradient( loss_W, self.params[ 'b1' ] )
        grads[ 'W2' ] = numerical_gradient( loss_W, self.params[ 'W2' ] )
        grads[ 'b2' ] = numerical_gradient( loss_W, self.params[ 'b2' ] )
        return grads

"""这里由于需要传入巨额的参数，比如说改变W1，那么通过数值微分法 需要向里面传入巨额的参数重新运算两遍，写
一个函数统计一下收敛到比较低的loss需要的参数量，数值微分法梯度计算次数 = 参数总量 × 2
W1: shape=(784, 100), count=78400
b1: shape=(100,), count=100
W2: shape=(100, 10), count=1000
b2: shape=(10,), count=10
Total parameters: 79510
参数总量为 79,510 个
数值微分需要计算 2×79510=159,020
2×79510=159,020 次 loss
每次 loss 都要一次前向传播！所以速度极慢。"""

"""写一个minibatch"""
if __name__ == '__main__':
    """查看一下参数"""
    """ print("net.params[W1].shape:", net.params[ 'W1' ].shape)
    print("net.params[b1].shape:", net.params[ 'b1' ].shape)
    print("net.params[W2].shape:", net.params[ 'W2' ].shape)
    print("net.params[b2].shape:", net.params[ 'b2' ].shape)"""

    """训练需要加载的数据集包装成一个函数了 麻烦得很"""
    x_train, t_train, x_test, t_test = data_load()
    train_loss_list = []
    #超参数的设置
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    """这里的类是网络结构类 所以存储的是有关网络的参数，其实就是网络的权重 之前考虑过的为什么不能存y这种目标值也在此得到解答"""
    network = TwoLayerNet(784, 10, 10)
    for i in range(iters_num):
        #获取minibatch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        #计算梯度
        grad = network.numerical_gradient(x_batch, t_batch)
        #更新参数
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        #记录学习过程
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        if i % 100 == 0:
            print(f"迭代次数：{i:5d}|当前loss:{loss:.4f}")
    #显示图像
    plt.plot(np.arange(len(train_loss_list)), train_loss_list)
    plt.show()
