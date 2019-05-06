################################################################################
### 課題4(a) Adamの実装
################################################################################
from task3 import *

################################################################################
### SGDと同じく、Optimizerの派生クラスとして実装する。
###
### 使用例: gnn3.fit(x,y,optimizer=Adam())
###
### 操作は全て原論文 https://arxiv.org/abs/1412.6980 より。
### ハイパーパラメータのデフォルト値も全て原論文に倣った。
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
### 課題4(a)　Adamの動作テストと性能評価（SGD, momentum SGDとの比較）
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

    ### ハイパーパラメータ
    D, T = 8, 2
    epochs = 30

    ### SGD、Momentum SGD、Adam
    ### 比較のため、全て同一初期値で行う。
    gnn3 = GNN3(D,T)
    # 初期値を保存
    Theta0 = np.copy(gnn3.Theta)
    ### SGD
    losses_SGD = gnn3.fit(x, y, validation=(vx,vy),
                          epochs=epochs,
                          optimizer=SGD())
    Theta_SGD = np.copy(gnn3.Theta)
    # 初期値を復元
    gnn3.Theta[:] = Theta0
    # Momentum SGD
    losses_mSGD = gnn3.fit(x, y, validation=(vx,vy),
                           epochs=epochs,
                           optimizer=MomentumSGD())
    Theta_mSGD = np.copy(gnn3.Theta)
    # 初期値を復元
    gnn3.Theta[:] = Theta0
    # Adam
    losses_Adam = gnn3.fit(x, y, validation=(vx,vy),
                           epochs=epochs,
                           optimizer=Adam())
    Theta_Adam = np.copy(gnn3.Theta)
    # lossデータをファイルに保存
    np.savez("task4a_losses.npz", losses_SGD, losses_mSGD, losses_Adam)
    # 学習済みパラメータを保存
    np.savez("task4a_theta.npz", Theta_SGD, Theta_mSGD, Theta_Adam)

    ### 学習曲線の描画 (Matplotlib)
    n = len(losses_SGD[0])
    x_arr = np.array(range(n))*(epochs/n)
    loss_max = 2
    margin = 0.025
    plt.ylim(loss_max*(-margin),loss_max*(1+margin))
    plt.xlim(-margin*epochs,epochs)
    p1=plt.plot(x_arr,losses_SGD[0],zorder=0)  # loss, SGD
    p2=plt.plot(x_arr,losses_SGD[3],zorder=3)  # vloss, SGD
    p3=plt.plot(x_arr,losses_mSGD[0],zorder=1)    # loss, Momentum SGD
    p4=plt.plot(x_arr,losses_mSGD[3],zorder=4)    # vloss, Momentum SGD
    p5=plt.plot(x_arr,losses_Adam[0],zorder=2)    # loss, Adam
    p6=plt.plot(x_arr,losses_Adam[3],zorder=5)    # vloss, Adam
    plt.grid(True)
    plt.legend((p1[0],p2[0],p3[0],p4[0],p5[0],p6[0]),
               ("loss, SGD", "vloss, SGD",
                "loss, momentum SGD", "vloss, momentum SGD",
                "loss, Adam", "vloss, Adam"
                ),
                loc=1)
    # 学習曲線プロットをファイルに保存
    plt.savefig("task4a_plot.pdf")
