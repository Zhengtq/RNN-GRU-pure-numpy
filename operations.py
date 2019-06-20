import numpy as np
import math


class MultiplyGate:
    def forward(self,W,Wb, x):

        res = np.dot(W,x) + Wb


        return res

    def backward(self, W,Wb, x, dz):
        dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)), np.asmatrix(x)))
        dWb = dz
        dx = np.dot(np.transpose(W), dz)
        return dW,dWb, dx

class AddGate:
    def forward(self, x1, x2):
        return x1 + x2

    def backward(self, x1, x2, dz):
        dx1 = dz * np.ones_like(x1)
        dx2 = dz * np.ones_like(x2)
        return dx1, dx2


class Eltwise_mul:
    def forward(self, x1, x2):
        return np.multiply(x1, x2)

    def backward(self, x1, x2, dz):
        dx1 = np.multiply(dz, x2)
        dx2 = np.multiply(dz, x1)
        
        return dx1, dx2
class Sigmoid:
    def forward(self, x):


        res = np.zeros(x.shape, dtype=float) 
        for ind, item in enumerate(x):

            item = float(item)
            if item >0:
                res[ind] = 1.0 / (1.0 + np.exp(-item))
            else:
                res[ind] = np.exp(item) / (1.0 + np.exp(item))


 
#          def fun1(x):
            #  res = 1.0 / (1.0 + np.exp(-x))
            #  return res

        #  def fun2(x):
            #  res = np.exp(x) / (1.0 + np.exp(x))
            #  return res

        #  res = np.where(x>0, fun1(x), fun2(x))

        return res

    def backward(self, x, top_diff):
        output = self.forward(x)
        return (1.0 - output) * output * top_diff

class Tanh:
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x, top_diff):
        output = self.forward(x)
        return (1.0 - np.square(output)) * top_diff

class Softmax:
    def predict(self, x, y = -1):

        if x>0:
            res = 1.0 / (1.0 + math.exp(-x))
        else:
            res = math.exp(x)/(1.0+math.exp(x))


        return res

    def loss(self, x, y):
        x = x[0]
        loss = max(x, 0) - x * float(y) + math.log(1 + math.exp(-math.fabs(x)))
  #      print('%.3f,%.3f,%.3f' %(loss, y,x))
        loss = np.array(loss)

        return loss

    def diff(self, x, y):
        x = x[0]
        probs = self.predict(x, y)
        diff = (probs - y)/(probs*(1-probs) + 1e-20)

        diff = min(10, diff)
        diff = max(-10, diff)
        diff = np.array(diff)
        
        return diff
