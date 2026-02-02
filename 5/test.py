#5.4.1乘法层的实现
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self, x, y):
        # 前向传播的时候会保存值 然后反向传播的时候用
        self.x = x
        self.y = y
        out = x * y
        return out
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x#翻转x和y
        return dx, dy
#5.4.2 加法层的实现
class AddLayer:
    def __init__(self):
        pass
    def forward(self, x, y):
        return x + y
    def backward(self, dout):
        dx = dout * 1.0
        dy = dout * 1.0
        return dx, dy

#实现买苹果和橘子
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()
#forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)
#backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print("price:", price)
print(dapple_num, dapple, dorange, dorange_price, dtax)