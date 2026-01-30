import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        #这个创建的是一个空字典
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    #补充两个激活函数的代码

    def sigmoid(self,a):
        return 1 / (1 + np.exp(-a))

    #这里跟维度没关系吗
    #这里相当于对每一个可能性都做了softmax
    def softmax(self,a):
        #a = np.array([[1,2,3],[4,5,6],[7,8,9]])
        if a.ndim == 2:
            #在同一行的不同列上取最大值
            c = np.max(a,axis = 1, keepdims = True)
            #这里相当于对每一个元素都做了e的这个元素次方
            exp_a = np.exp(a - c)
            sum_exp_a = np.sum(exp_a, axis = 1, keepdims = True)
            return exp_a / sum_exp_a
        else:
            c = np.max(a)
            exp_a = np.exp(a - c)
            return exp_a / np.sum(exp_a)


    #前向传播的代码
    def predict(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        a1 = np.dot(X, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = self.softmax(a2)
        return y
    #得有一个计算交叉熵损失函数的
    def cross_entropy_error(self,y, t):
        delta = 1e-7
        if y.ndim == 1:
            #如果是单样本的情况就把他们🙆变成多维度的，我理解 比如y =[1,2,3]然后 就变成1行三列[[1,2,3]]这样的？
            #保证“批次维”存在（方便统一矩阵运算）
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)#y也同理
        batch_size = y.shape[0]
        #花式索引 直接取出来每行的对应标签处的元素 这里相当于对于one hot编码做了适应
        #本质上是一样的无所谓
        return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
    #前向传播完了得求准确率吧
    def accuracy(self, X, t):
        y = self.predict(X)
        #np.argmax(y.axis =1) 返回的是 最大值的索引
        #和对应标签能对得上的 加起来 就是能匹配上的数量了，/总数 也就是总的批次，总的个数
        acc = np.sum(np.argmax(y, axis = 1) == np.argmax(t, axis = 1)) / y.shape[0]
        return acc
    def loss(self, X, t):
        y = self.predict(X)
        loss = self.cross_entropy_error(y, t)
        return loss

    def numerical_gradient(self,f, x):
        h = 1e-4
        grad = np.zeros_like(x)  # 生成和x形状相同的数组

        for idx in range(x.size):
            tmp_val = x[idx]
            x[idx] = tmp_val + h
            fxh1 = f(x)

            x[idx] = tmp_val - h
            fxh2 = f(x)
            grad[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = tmp_val
        return grad

    def numerical_gradient_all(self, x, t):
        loss_w =lambda W:self.loss(x, t)
        grads = {}
        grads['W1'] = self.numerical_gradient(loss_w, self.params['W1'])
        grads['b1'] = self.numerical_gradient(loss_w, self.params['b1'])
        grads['W2'] = self.numerical_gradient(loss_w, self.params['W2'])
        grads['b2'] = self.numerical_gradient(loss_w, self.params['b2'])
        return grads
#3.6.2神经网络的推理处理
#接下来对这个数据集实现神经网络的推理处理，输入层有784个神经元 输出层有10个神经元
#两个隐藏层 一层100个 一层50 个
import numpy as np
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




if __name__ == '__main__':
    #加载数据
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

    train_loss_list = []
    #超参数
    iters_num = 10000
    train_size = train_images_flat.shape[0]
    batch_size = 100
    learning_rate = 0.1
    net = TwoLayerNet(input_size = 784, hidden_size = 100, output_size = 10)
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = train_images_flat[batch_mask]
        t_batch = train_labels[batch_mask]

        #计算梯度
        grad = net.numerical_gradient_all(x_batch, t_batch)

        #更新参数
        for key in net.params.keys():
            net.params[key] -= learning_rate * grad[key]
        loss = net.loss(x_batch, t_batch)
        train_loss_list.append(loss)