################################################################################
### 課題4(b) 多層ニューラルネットワーク化
################################################################################
from task3 import *

################################################################################
### 多層ニューラルネットワーク化したクラスGNN4を作成。
### 集約2で、Wによるニューラルネットを多層化する。そのためWを3次元配列に。
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
                self.Theta[Wsize+Asize:Wsize+Asize+1])

    ############################################################################
    ### 集約2を多層ニューラルネットに変更。
    def aggregate2(self, a):
        x = a
        for i in range(self.Wlayers):
            x = self.f(np.dot(x, self.W[i]))
        return x


################################################################################
### 課題4(b)　動作テストと性能評価
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
    ### 動作確認と性能評価（GNN4とGNN3の比較）
    D, T = 8, 2
    epochs = 30

    # GNN3とGNN4
    # あまり意味はないかもしれないが、気休め程度に初期値を合わせておく。
    gnn4 = GNN4(D,T)
    gnn3 = GNN3(D,T,W0=gnn4.W[0], A0=gnn4.A, b0=gnn4.b[0])

    # それぞれ学習。
    losses_gnn3 = gnn3.fit(x, y, validation=(vx,vy),
                           epochs=epochs,
                           optimizer=MomentumSGD())
    losses_gnn4 = gnn4.fit(x, y, validation=(vx,vy),
                           epochs=epochs,
                           optimizer=MomentumSGD())
    # lossデータをファイルに保存
    np.savez("task4b_losses.npz",losses_gnn3,losses_gnn4)
    np.savez("task4b_theta.npz", gnn3.Theta, gnn4.Theta)
    
    ### 学習曲線の描画
    n = len(losses_gnn3[0])
    x_arr = np.array(range(n))*(epochs/n)
    loss_max = 2
    margin = 0.25
    plt.ylim(-margin,loss_max+margin)
    plt.xlim(-margin,epochs)
    p1=plt.plot(x_arr,losses_gnn3[0])  # loss, GNN3
    p2=plt.plot(x_arr,losses_gnn3[2])  # vloss, GNN3
    p3=plt.plot(x_arr,losses_gnn4[0])    # loss, GNN4
    p4=plt.plot(x_arr,losses_gnn4[2])    # vloss, GNN4
    plt.grid(True)
    plt.legend((p1[0],p2[0],p3[0],p4[0]),
               ("loss, GNN3", "vloss, GNN3",
                "loss, GNN4", "vloss, GNN4"
                ),
                loc=1)
    # 学習曲線プロットをファイルに保存
    plt.savefig("task4b_plot.pdf")
