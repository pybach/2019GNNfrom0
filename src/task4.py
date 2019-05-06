################################################################################
### 課題4 Adamの実装と多層ニューラルネットワーク化
################################################################################
from task3 import *


################################################################################
### Adamの実装
### Based on: https://arxiv.org/abs/1412.6980
### ハイパーパラメータのデフォルト値は全て原論文に倣った。
class Adam(Optimizer):
    def __init__(self,
                 alpha=1.0e-3, beta1=0.9, beta2=0.999,
                 epsilon=1.0e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.pw_beta1 = 1
        self.pw_beta2 = 1

    def update(self, grad):
        grad2 = grad**2
        if self.m is None:
            self.m = np.zeros(len(grad))
            self.v = np.zeros(len(grad))
        self.m *= self.beta1
        self.m += (1-self.beta1)*grad
        self.v *= self.beta2
        self.v += (1-self.beta2)*grad2
        self.pw_beta1 *= self.beta1
        self.pw_beta2 *= self.beta2
        mhat = self.m/(1-self.pw_beta1)
        vhat = self.v/(1-self.pw_beta2)
        dtheta = -self.alpha*mhat/(np.sqrt(vhat)+self.epsilon)
        return dtheta


################################################################################
### 多層ニューラルネットワーク化
### 集約2で、Wによるニューラルネットを多層化する。
class GNN4(GNN3):
    def __init__(self, D, T,
                 W0=None, A0=None, b0=None, sigma=0.4,
                 epsilon=1.0e-3,
                 Wlayers=2):
        self.Wlayers=Wlayers
        super().__init__(D, T,
                         W0=W0, A0=A0, b0=b0, sigma=sigma,
                         epsilon=epsilon)

    ############################################################################
    ### W.shape = (Wlayers,D,D) である。
    ### それに合わせ、パラメータ初期化に関係する関数を上書きする。
    def init_Theta(self, sigma, W0, A0, b0):
        # W0: Wlayers x D x D
        if W0 is None:
            W0 = np.random.normal(0,sigma,(self.Wlayers,self.D,self.D))
        if A0 is None:
            A0 = np.random.normal(0,sigma,self.D)
        if b0 is None:
            b0 = np.random.normal(0,sigma)
        self.Theta = self.encode_Theta(W0,A0,b0)
        self.W, self.A, self.b = self.decode_Theta()

    def decode_Theta(self):
        Wsize=self.Wlayers*self.D*self.D
        Asize=self.D
        return (self.Theta[:Wsize].reshape((self.Wlayers,self.D,self.D)),
                self.Theta[Wsize:Wsize+Asize],
                self.Theta[Wsize+Asize])

    ############################################################################
    ### 集約2を多層ニューラルネットに変更。
    def aggregate2(self, a):
        x = a
        for i in range(self.Wlayers):
            x = self.f(np.dot(x, self.W[i]))
        return x


################################################################################
### 課題4　動作テスト
################################################################################
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ### 学習用データ
    Ndata = 2000
    datasetsdir='../../datasets/train/'
    graph_filename = datasetsdir + '%d_graph.txt'
    label_filename = datasetsdir + '%d_label.txt'
    graphs = []
    labels = []
    for i in range(Ndata):
        with open(label_filename % i, "r") as label_file:
            labels.append(int(label_file.readline()[0]))
        with open(graph_filename % i, "r") as graph_file:
            lines = graph_file.readlines()
        N = int(lines[0])
        graph = np.empty((N,N))
        for k in range(N):
            graph[k] = list(map(int,lines[k+1].split()))
        graphs.append(graph)
    ### データ分割
    # 1600組を学習用(x,y)、400組を検証用(vx,vy)とする。
    Ntrain = 1600
    x, y, vx, vy = graphs[:Ntrain], labels[:Ntrain], graphs[Ntrain:], labels[Ntrain:]

    ############################################################################
    ### Adamの動作確認と性能評価（GNN3）
    D, T = 8, 2
    epochs = 10

    ### SGD、Momentum SGD、Adam
    ### 比較のため、同一初期値で行う。
    gnn3 = GNN3(D,T)
    # 初期値を保存
    Theta0 = np.copy(gnn3.Theta)
    ### SGD
    losses_SGD = gnn3.fit(x, y, validation=(vx,vy),
                          epochs=epochs,
                          optimizer=SGD())
    # 初期値を復元
    gnn3.Theta[:] = Theta0
    # Momentum SGD
    losses_mSGD = gnn3.fit(x, y, validation=(vx,vy),
                           epochs=epochs,
                           optimizer=MomentumSGD())
    # 初期値を復元
    gnn3.Theta[:] = Theta0
    # Adam
    losses_Adam = gnn3.fit(x, y, validation=(vx,vy),
                           epochs=epochs,
                           optimizer=Adam())
    ### 学習曲線の描画
    n = len(losses_SGD[0])
    x_arr = np.array(range(n))*(epochs/n)
    loss_max = 10
    margin = 0.25
    plt.ylim(-margin,loss_max+margin)
    plt.xlim(-margin,epochs)
    p1=plt.plot(x_arr,losses_SGD[0])  # loss, SGD
    p2=plt.plot(x_arr,losses_SGD[2])  # vloss, SGD
    p3=plt.plot(x_arr,losses_mSGD[0])    # loss, Momentum SGD
    p4=plt.plot(x_arr,losses_mSGD[2])    # vloss, Momentum SGD
    p5=plt.plot(x_arr,losses_mSGD[0])    # loss, Adam
    p6=plt.plot(x_arr,losses_mSGD[2])    # vloss, Adam
    plt.grid(True)
    plt.legend((p1[0],p2[0],p3[0],p4[0],p5[0],p6[0]),
               ("loss, SGD", "vloss, SGD",
                "loss, momentum SGD", "vloss, momentum SGD",
                "loss, Adam", "vloss, Adam"
                ),
                loc=1)
    # 学習曲線プロットをファイルに保存
    plt.savefig("task4_Adam_plot.pdf")
    # lossデータをファイルに保存
    np.savez_compressed("task4_Adam_losses.npz",losses_SGD,losses_mSGD,losses_Adam)
