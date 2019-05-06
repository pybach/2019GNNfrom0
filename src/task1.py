import numpy as np
import math

# ReLU activation function
def relu(x):
    return x if x > 0 else 0

class GNN1:
    def __init__(self, D, T, W):
        self.D = D
        self.T = T
        self.W = W

    def x0(self, N):
        # x0: NxD np.array
        x =  np.zeros((N, self.D))
        if N <= self.D:
            for i in range(self.D):
                x[i%N, i] = i
        else:
            for i in range(N):
                x[i, i%self.D] = i
        return x
    # x0
    # example: (N=3, D=5)
    #  1 0 0 4 0
    #  0 2 0 0 5
    #  0 0 3 0 0

    @staticmethod
    def f(x):
        return np.frompyfunc(relu, 1, 1)(x)

    @staticmethod
    def aggregate1(G, x):
        a = np.dot(G, x)
        return a

    def aggregate2(self, a):
        x = self.f(np.dot(a, self.W))
        return x

    def readout(self, G):
        N = len(G)
        x = self.x0(N)
        for _ in range(self.T):
            a = self.aggregate1(G, x)
            x = self.aggregate2(a)

        return x.sum(axis=0)
# end of class GNN1


################################################################################
### 課題1　動作テスト
################################################################################
if __name__ == '__main__':
    D = 3
    W = np.array([[1,1,0],[1,-1,0],[0,0,1]])
    G = np.array([[0,1,0,0],[1,0,1,1],[0,1,0,1],[0,1,1,0]])
    for T in range(3):
        gnn1 = GNN1(D,T,W)
        print(gnn1.readout(G))
    # 確かにOK。
