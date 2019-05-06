################################################################################
### 課題3 損失・勾配の計算と勾配降下
################################################################################
from task2 import *

# GNN2にデータセットからの学習機能を追加。
# SGDなどのオプティマイザを使って学習する。
class GNN3(GNN2):
    def __init__(self, D, T,
                 W0=None, A0=None, b0=None, sigma=0.4,
                 epsilon=1.0e-3):
        super().__init__(D, T,
                         W0=W0, A0=A0, b0=b0, sigma=sigma,
                         epsilon=epsilon)
    def predict(self,x):
        return np.array([ self.p(g) for g in x ])


    ############################################################################
    ### データセット(x,y) からの学習メソッド
    ### 
    ### GNN3.sgd(self,x,y) や GNN3.msgd(self,x,y) は実装しない。
    ### その代わり、それらの機能はfit()にoptimizerとしてオプティマイザオブジェクトを渡す。
    ###
    ### 例: gnn3.fit(x, y, optimizer=MomentumSGD())
    ###
    ###　詳しくは下の class Optimizer以下を参照。
    def fit(self,
            x, y,   # 学習データ。len(y)<=len(x)であること。
            epochs=1, batchsize=16, shuffle=True,
            validation=None,    # (vx,vy) のタプルで与える。len(vy)<=len(vx) であること。
            optimizer=None  # Optimizer派生クラスのオブジェクトを与える。
            ):
        if optimizer is None:
            optimizer = SGD()   # デフォルトはSGD

        n = len(y)
        if batchsize > n:
            batchsize = n
        if validation is None:
            vld = False
        else:
            vld = True
            vx, vy = validation
            vn = len(vy)
            if batchsize > vn:
                batchsize = vn
            valid_marker = vn   # vloss計算のための準備

        # 平均損失値ログの初期化。
        # 学習前全データ平均をepoch毎のログに格納。
        batch_losses = []
        epoch_losses = []
        epoch_losses.append(np.mean([self.loss(x[i],y[i]) for i in range(n)]))
        if vld:
            batch_vlosses = []
            epoch_vlosses = []
            epoch_vlosses.append(np.mean([self.loss(vx[i],vy[i]) for i in range(vn)]))        

        for e in range(epochs):
            if shuffle:
                # epoch毎にランダムに並べ替える。
                order = np.random.permutation(n)
            else:
                order = range(n)

            for b in range((n-1)//batchsize+1):
                start = batchsize * b
                end = min(start+batchsize, n)

                # ミニバッチごとに、更新前の平均損失値を計算して記録。
                batch_loss = np.mean([ self.loss(x[i],y[i]) for i in order[start:end]])
                batch_losses.append(batch_loss)
                if vld:
                    # validationも同じ数だけ抽出して評価。
                    if valid_marker + (end-start) > vn:
                        valid_marker = 0
                        valid_order = np.random.permutation(vn)
                    batch_vloss = np.mean([ self.loss(vx[i],vy[i])
                                            for i in valid_order[valid_marker:valid_marker+(end-start)]])
                    batch_vlosses.append(batch_vloss)

                ### Thetaの更新。
                # 勾配ミニバッチ平均
                grad = np.mean([ self.grad(x[i],y[i])
                                 for i in order[start:end]],axis=0)
                # optimizerで変化量を求める。（optimizerに依存する箇所はここだけ）
                self.shift_Theta(optimizer.update(grad))

            # epoch終了時はtrain/validationとも全データで評価。
            # 勾配計算のコストが巨大なので、ここの計算量は特に考慮しなくてよい。
            epoch_loss = np.mean([self.loss(x[i],y[i]) for i in range(n)])
            epoch_losses.append(epoch_loss)
            if vld:
                epoch_vloss = np.mean([self.loss(vx[i],vy[i]) for i in range(vn)])
                epoch_vlosses.append(epoch_vloss)

        # lossの履歴を返す。
        if vld:
            return batch_losses, epoch_losses, batch_vlosses, epoch_vlosses
        else:
            return batch_losses, epoch_losses
# end of GNN3

################################################################################
### オプティマイザクラス群

### オプティマイザ基底クラス。
### ここから派生したクラスのオブジェクトをGNN3.fit()にoptimizerとして渡す。
class Optimizer():
    @staticmethod
    def update(grad):
        # 勾配を受け取り、内部状態を更新し、変化量を返すメソッド。
        # （派生クラスではそのような関数に再定義する。）
        return -grad

### 確率的勾配降下法
class SGD(Optimizer):
    def __init__(self, alpha=1.0e-4):
        self.alpha = alpha

    def update(self, grad):
        # SGDは更新される内部状態を持たない。
        return -self.alpha * grad

class MomentumSGD(Optimizer):
    def __init__(self, alpha=1.0e-4, eta=0.9):
        self.alpha = alpha
        self.eta = eta
        self.w = None

    def update(self, grad):
        # 内部のwを更新しながら変化量を返す。
        if self.w is None:
            self.w = np.zeros(len(grad))
        self.w *= self.eta
        self.w -= self.alpha * grad
        return self.eta * self.w


################################################################################
### 課題3　動作テストと学習曲線描画
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
    x, y, vx, vy = graphs[:1600], labels[:1600], graphs[1600:], labels[1600:]

    ### ハイパーパラメータ設定
    D, T = 8, 2
    epochs = 10

    ### SGD と Momentum SGD での学習曲線
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
    plt.grid(True)
    plt.legend((p1[0],p2[0],p3[0],p4[0]),
               ("loss, SGD", "vloss, SGD",
                "loss, momentum SGD", "vloss, momentum SGD"),
               　loc=1)

    # 学習曲線プロットをファイルに保存
    plt.savefig("task3_plot.pdf")
    # lossデータをファイルに保存
    np.saves_compressed("task3_losses.npz",losses_SGD,losses_mSGD)
