import numpy as np
import sys
from layer import RNNLayer, GRULayer
from operations import Softmax
import os
np.random.seed(5)

class Model:
    def __init__(self, feature_dim, hidden_dim=100,time_step = 5, bptt_truncate=4):
        self.time_step = time_step
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.output_num = 1
        self.W = np.random.uniform(-np.sqrt(1. / (hidden_dim + feature_dim)), np.sqrt(1. / (hidden_dim + feature_dim)), (hidden_dim * 2, hidden_dim + feature_dim))
        self.Wb = np.ones((hidden_dim * 2), dtype=float) * 0.01
        self.C = np.random.uniform(-np.sqrt(1. / (hidden_dim + feature_dim)), np.sqrt(1. / (hidden_dim + feature_dim)), (hidden_dim, hidden_dim + feature_dim))
        self.Cb = np.ones((hidden_dim), dtype=float) * 0.01
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (self.output_num, hidden_dim))
        self.Vb = np.ones((self.output_num), dtype=float) * 0.01

        self.W1 = np.random.uniform(-np.sqrt(1. / (hidden_dim * 2)), np.sqrt(1. / (hidden_dim * 2)), (hidden_dim * 2, hidden_dim *2))
        self.W1b = np.ones((hidden_dim * 2), dtype=float) * 0.01
        self.C1 = np.random.uniform(-np.sqrt(1. / (hidden_dim * 2)), np.sqrt(1. / (hidden_dim *2)), (hidden_dim, hidden_dim * 2))
        self.C1b = np.ones((hidden_dim), dtype=float) * 0.01
        self.V1 = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (self.output_num, hidden_dim))
        self.V1b = np.ones((self.output_num), dtype=float) * 0.01

   #     self.load_weight()


#          self.W = np.ones((hidden_dim * 2, hidden_dim + feature_dim), dtype=float) * 0.6
        #  self.C = np.ones((hidden_dim, hidden_dim + feature_dim), dtype=float) * 0.6
        #  self.V = np.ones((self.output_num, hidden_dim), dtype=float) * 0.6

        #  self.W1 = np.ones((hidden_dim * 2, hidden_dim *2), dtype=float) * 0.6
        #  self.C1 = np.ones((hidden_dim, hidden_dim * 2), dtype=float) * 0.6
        #  self.V1 = np.ones((self.output_num, hidden_dim), dtype=float) * 0.6


    def load_weight(self):


        weight_file = open('./RNN_WT/wt.txt', 'r')
        all_weight = []
        for item1 in weight_file:
            for item2 in item1.strip().split(','):
                try:
                    the_weight = float(item2)
                    all_weight.append(the_weight)
                except:
                    continue
        weight_file.close()

        
        weight_file_c = 0
        for out_num in range(self.V1.shape[0]):
            for in_num in range(self.V1.shape[1]):
                self.V1[out_num, in_num] = all_weight[weight_file_c]
                weight_file_c += 1
        
        for out_num in range(self.V1b.shape[0]):
            self.V1b[out_num] = all_weight[weight_file_c]
            weight_file_c += 1

        for out_num in range(self.W.shape[0]):
            for in_num in range(self.W.shape[1]):
                self.W[out_num, in_num] = all_weight[weight_file_c]
                weight_file_c += 1
        
        for out_num in range(self.Wb.shape[0]):
            self.Wb[out_num] = all_weight[weight_file_c]
            weight_file_c += 1

        for out_num in range(self.C.shape[0]):
            for in_num in range(self.C.shape[1]):
                self.C[out_num, in_num] = all_weight[weight_file_c]
                weight_file_c += 1
        
        for out_num in range(self.Cb.shape[0]):
            self.Cb[out_num] = all_weight[weight_file_c]
            weight_file_c += 1

        for out_num in range(self.W1.shape[0]):
            for in_num in range(self.W1.shape[1]):
                self.W1[out_num, in_num] = all_weight[weight_file_c]
                weight_file_c += 1
        
        for out_num in range(self.W1b.shape[0]):
            self.W1b[out_num] = all_weight[weight_file_c]
            weight_file_c += 1

        for out_num in range(self.C1.shape[0]):
            for in_num in range(self.C1.shape[1]):
                self.C1[out_num, in_num] = all_weight[weight_file_c]
                weight_file_c += 1
        
        for out_num in range(self.C1b.shape[0]):
            self.C1b[out_num] = all_weight[weight_file_c]
            weight_file_c += 1



    def forward_propagation(self, x):
        # The total number of time steps
        layers = []
        layer1s = []
        prev_s = np.zeros(self.hidden_dim)
        prev1_s = np.zeros(self.hidden_dim)
        # For each time step...
        for t in range(self.time_step):
            layer = GRULayer()
            layer1 = GRULayer()
            input = x[t]
            layer.forward(input, prev_s, self.W, self.Wb,self.C,self.Cb,  self.V, self.Vb)
            prev_s = layer.s
            layer1.forward(layer.s, prev1_s, self.W1, self.W1b,self.C1,self.C1b, self.V1, self.V1b)
            prev1_s = layer1.s
            layers.append(layer)
            layer1s.append(layer1)
        return layers, layer1s


    def calculate_loss(self, x, y):
        output = Softmax()
        layers, layer1s = self.forward_propagation(x)
        loss = output.loss(layer1s[-1].mulv, y)
        return loss


    def bptt(self, x, y):



        output = Softmax()
        layers, layer1s = self.forward_propagation(x)
        dW = np.zeros(self.W.shape)
        dC = np.zeros(self.C.shape)
        dV = np.zeros(self.V.shape)
        dW1 = np.zeros(self.W1.shape)
        dC1 = np.zeros(self.C1.shape)
        dV1 = np.zeros(self.V1.shape)


        dWb = np.zeros(self.Wb.shape)
        dCb = np.zeros(self.Cb.shape)
        dVb = np.zeros(self.Vb.shape)
        dW1b = np.zeros(self.W1b.shape)
        dC1b = np.zeros(self.C1b.shape)
        dV1b = np.zeros(self.V1b.shape)



        predict = output.predict(layer1s[-1].mulv)

        diff1_s = np.zeros(self.hidden_dim)
        dmulv_zero = np.zeros(self.output_num)
        if True:
            t = self.time_step -1
            dmulv1 = output.diff(layer1s[t].mulv, y)
            dmulv1 = np.array([dmulv1])
            input = x[t]

         #   print(dmulv1)

            dprev1_s, dW1_t,dW1b_t, dC1_t,dC1b_t,  dV1_t, dV1b_t = layer1s[t].backward(layers[t].s, layer1s[t-1].s, self.W1, self.W1b, self.C1,self.C1b, self.V1,self.V1b, diff1_s, dmulv1)
            dprev_s, dW_t,dWb_t,  dC_t,dCb_t, dV_t,dVb_t = layers[t].backward(input, layers[t-1].s, self.W, self.Wb,self.C, self.Cb,self.V,self.Vb, dprev1_s, dmulv_zero)


            for i in range(t-1, 0, -1):
                input = x[i]
                prev1_s_i = np.zeros(self.hidden_dim) if i == 0 else layer1s[i-1].s
                prev_s_i = np.zeros(self.hidden_dim) if i == 0 else layers[i-1].s

                dprev1_s, dW1_i,dW1b_i,dC1_i,dC1b_i, dV1_i,dV1b_i = layer1s[i].backward(layers[t].s, prev1_s_i, self.W1, self.W1b,self.C1, self.C1b,self.V1,self.V1b, dprev_s, dmulv_zero)
                dprev_s, dW_i,dWb_i, dC_i,dCb_i, dV_i,dVb_i = layers[i].backward(input, prev_s_i, self.W, self.Wb,self.C,self.Cb,self.V, self.Vb,dprev1_s, dmulv_zero)

                dW_t += dW_i
                dW1_t += dW1_i
                dC_t += dC_i
                dC1_t += dC1_i

                dWb_t += dWb_i
                dW1b_t += dW1b_i
                dCb_t += dCb_i
                dC1b_t += dC1b_i



            dV += dV_t
            dW += dW_t
            dC += dC_t
            dV1 += dV1_t
            dW1 += dW1_t
            dC1 += dC1_t



            dVb += dVb_t
            dWb += dWb_t
            dCb += dCb_t
            dV1b += dV1b_t
            dW1b += dW1b_t
            dC1b += dC1b_t


        return dW, dWb, dC,dCb, dV,dVb, dW1,dW1b, dC1,dC1b,dV1,dV1b, predict

    def sgd_step(self, x, y, learning_rate):
        dW,dWb,dC,dCb, dV,dVb, dW1,dW1b,dC1, dC1b,dV1,dV1b, predict = self.bptt(x, y)
        self.V -= learning_rate * dV
        self.V1 -= learning_rate * dV1
        self.W -= learning_rate * dW
        self.W1 -= learning_rate * dW1
        self.C -= learning_rate * dC
        self.C1 -= learning_rate * dC1

        self.Vb -= learning_rate * dVb
        self.V1b -= learning_rate * dV1b
        self.Wb -= learning_rate * dWb
        self.W1b -= learning_rate * dW1b
        self.Cb -= learning_rate * dCb
        self.C1b -= learning_rate * dC1b
        return predict



    def train(self, X, Y,time_step, learning_rate=0.1):
        self.time_step = time_step
        predict = self.sgd_step(X, Y, learning_rate)
        loss2 = self.calculate_loss(X, Y)
        return loss2



    def save_wt(self, wt_file = './RNN_WT/wt.txt'):

        
        wt_folder = os.path.dirname(wt_file) + '/'
        if not os.path.exists(wt_folder):
            os.makedirs(wt_folder)
       
       
        all_weight = []
        for out_num in range(self.V1.shape[0]):
            for in_num in range(self.V1.shape[1]):
                all_weight.append(self.V1[out_num, in_num])
        
        for out_num in range(self.V1b.shape[0]):
            all_weight.append(self.V1b[out_num])

        for out_num in range(self.W.shape[0]):
            for in_num in range(self.W.shape[1]):
                all_weight.append(self.W[out_num, in_num])
        
        for out_num in range(self.Wb.shape[0]):
            all_weight.append(self.Wb[out_num])

        for out_num in range(self.C.shape[0]):
            for in_num in range(self.C.shape[1]):
                all_weight.append(self.C[out_num, in_num])
        
        for out_num in range(self.Cb.shape[0]):
            all_weight.append(self.Cb[out_num])

        for out_num in range(self.W1.shape[0]):
            for in_num in range(self.W1.shape[1]):
                all_weight.append(self.W1[out_num, in_num])
        
        for out_num in range(self.W1b.shape[0]):
            all_weight.append(self.W1b[out_num])

        for out_num in range(self.C1.shape[0]):
            for in_num in range(self.C1.shape[1]):
                all_weight.append(self.C1[out_num, in_num])
        
        for out_num in range(self.C1b.shape[0]):
            all_weight.append(self.C1b[out_num])

        all_weight.append(233.0)
        wt_file_out = open(wt_file, 'w')

        for ind, item in enumerate(all_weight):
            if ind != 0 and ind % 10 == 0:
                wt_file_out.write('\n')
            wt_file_out.write(str(item) + ',')
        
        
