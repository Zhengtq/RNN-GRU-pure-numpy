from operations import Tanh, Sigmoid
from operations import AddGate, MultiplyGate, Eltwise_mul
import numpy as np


mulGate = MultiplyGate()
addGate = AddGate()
activation = Tanh()
sig_act = Sigmoid()
tan_act = Tanh()
eltwise_mul = Eltwise_mul()

class RNNLayer:
    def forward(self, x, prev_s, W, V):

        
        self.hidden_mum = len(prev_s)
        self.x_prev_s = np.concatenate((x, prev_s), axis = 0)
        self.mulw = mulGate.forward(W, self.x_prev_s)
        self.s = activation.forward(self.mulw)
        self.mulv = mulGate.forward(V, self.s)

    def backward(self, x, prev_s, W, V, diff_s, dmulv):
        self.forward(x, prev_s, W, V)
        dV, dsv = mulGate.backward(V, self.s, dmulv)
        ds = dsv + diff_s
        dadd = activation.backward(self.mulw, ds)
        dW, dprev_s = mulGate.backward(W, self.x_prev_s, dadd)
        dprev_s = dprev_s[-self.hidden_mum:]
        return (dprev_s, dW, dV)


class GRULayer:


    def forward(self, x, prev_s, W,Wb,C,Cb, V, Vb):
        
        self.x_prev_s = np.concatenate((x, prev_s), axis = 0)
        self.hidden_mum = len(prev_s)
        self.mulw = mulGate.forward(W, Wb,self.x_prev_s)

        self.mulwsig = sig_act.forward(self.mulw) 

        self.r = self.mulwsig[:len(self.mulwsig)/2]
        self.u = self.mulwsig[len(self.mulwsig)/2:]

        self.r_state = eltwise_mul.forward(self.r, prev_s)

        self.x_r_state = np.concatenate((x, self.r_state), axis = 0)
        self.x_r_state_mulc =mulGate.forward(C,Cb, self.x_r_state)


        self.x_r_state_mulc_tan = tan_act.forward(self.x_r_state_mulc)

        self.tmpadd1 = eltwise_mul.forward(self.u, prev_s)
        self.u_fu = 1 - self.u
        self.tmpadd2 = eltwise_mul.forward(self.u_fu, self.x_r_state_mulc_tan)
        self.s = addGate.forward(self.tmpadd1, self.tmpadd2)
        self.mulv = mulGate.forward(V,Vb, self.s)
     


    def backward(self, x, prev_s, W, Wb, C, Cb, V, Vb,diff_s, dmulv):

   #     self.forward(x, prev_s, W,Wb,C,Cb, V,Vb)

        dV,dVb, dsv = mulGate.backward(V,Vb, self.s, dmulv)
        ds = dsv + diff_s

        dtmpadd1, dtmpadd2 = addGate.backward(self.tmpadd1, self.tmpadd2, ds)
       
        du_fu, dx_r_state_mulc_tan = eltwise_mul.backward(self.u_fu, self.x_r_state_mulc_tan, dtmpadd2)

        du1 = -du_fu
        
        du2, dprev_s0 = eltwise_mul.backward(self.u, prev_s, dtmpadd1)

        du = du1 + du2

        dx_r_state_mulc = tan_act.backward(self.x_r_state_mulc, dx_r_state_mulc_tan)


        dC,dCb, dx_x_r_state = mulGate.backward(C,Cb, self.x_r_state, dx_r_state_mulc)


        dx = dx_x_r_state[:len(x)]
        dr_state = dx_x_r_state[len(x):]
       
        dr, dprev_s1 = eltwise_mul.backward(self.r, prev_s, dr_state)

        dmulwsig = np.concatenate((dr, du), axis = 0)
        dmulw = sig_act.backward(self.mulw, dmulwsig)

        dW,dWb, dx_prev_s = mulGate.backward(W, Wb, self.x_prev_s, dmulw)
        dprev_s = dx_prev_s[-self.hidden_mum:] + dprev_s1 + dprev_s0


  

        return (dprev_s, dW,dWb, dC,dCb, dV,dVb)

