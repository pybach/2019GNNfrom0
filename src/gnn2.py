################################################################################
### 課題2 損失・勾配の計算と勾配降下
################################################################################
from gnn1 import *

# GNN1に損失関数・勾配を計算して勾配降下を行う機能を追加。
class GNN2(GNN1):
    def __init__(self, D, T,
                 W0=None, A0=None, b0=None,
                 sigma=0.4, # (W,A,b)のランダム初期値標準偏差
                 epsilon=1.0e-3 # 数値微分の微少変分
                 ):
        # (W, A, b) の初期値を設定する機能を念のため残した。
        # 通常、使う必要はない。省略した場合、sigmaの正規分布となる。
        self.D = D
        self.T = T
        self.init_Theta(sigma,W0,A0,b0)
        self.epsilon = epsilon
        # GNN1.__init__() は不要。

    ############################################################################
    ### Theta = (W,A,b) の初期化/変換メソッド群。
    ### Theta と (W,A,b) はメモリを共有する。
    def init_Theta(self, sigma, W0, A0, b0):
        if W0 is None:
            W0 = np.random.normal(0,sigma,(self.D,self.D))
        if A0 is None:
            A0 = np.random.normal(0,sigma,self.D)
        if b0 is None:
            b0 = np.random.normal(0,sigma)
        self.Theta = self.encode_Theta(W0,A0,b0)
        self.W, self.A, self.b = self.decode_Theta()

    @staticmethod
    def encode_Theta(W,A,b):
        return np.concatenate((W.flatten(),A,np.array([b])))

    # bも要素数1の配列としてb[0]でアクセスする。
    # メモリ共有したまま値を更新するため。
    def decode_Theta(self):
        return (self.Theta[:self.D*self.D].reshape((self.D,self.D)),
                self.Theta[self.D*self.D:self.D*(self.D+1)],
                self.Theta[self.D*(self.D+1):self.D*(self.D+1)+1])
    ############################################################################


    ############################################################################
    ### 課題2-(1/3)：予言値（確率）と損失関数
    def s(self, G):
        h = self.readout(G)
        return np.dot(h,self.A) + self.b[0]

    def p(self, G):
        return sigmoid(self.s(G))

    def loss(self,G,y):
        return binary_cross_entropy(y, self.s(G))
    ############################################################################


    ############################################################################
    ### 課題2-(2/3)： 数値微分による勾配計算
    # W,A,b の区別は気にせず、Thetaの各成分で数値微分すればよい。
    def grad(self, G, y, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        L0 = self.loss(G,y)
        K = len(self.Theta)
        dL = np.empty(K)
        for i in range(K):
            self.Theta[i] += epsilon
            dL[i] = (self.loss(G,y) - L0)/epsilon
            self.Theta[i] -= epsilon
        return dL
    ############################################################################


    ############################################################################
    ### 課題2-(3/3)： 勾配降下
    def shift_Theta(self, dTheta):
        self.Theta += dTheta
    
    def descendant_update(self, G, y, alpha, epsilon=None):
        self.shift_Theta(-alpha*self.grad(G,y,epsilon))
    ############################################################################
# end of GNN2

# sigmoid function
def sigmoid(x):
    return 1/(1 + math.exp(-x))

# binary-crossentropy
# （overflow回避のため、sの正負で等価な式に場合分けして計算。）
def binary_cross_entropy(y,s):
    # p = sigmoid(s)
    if s >= 0:
        return (1-y)*s + math.log(1+math.exp(-s))
    else:
        return -y*s + math.log(1+math.exp(s))


################################################################################
### 課題2　動作テスト
################################################################################
if __name__ == '__main__':
    ### 学習用データ
    Ndata = 2000
    datasetsdir='../../datasets/train/'
    graph_filename = datasetsdir + '%d_graph.txt'
    label_filename = datasetsdir + '%d_label.txt'

    # データセットの中からランダムに選んで(G,y)取り込み。
    n = np.random.randint(Ndata)
    with open(label_filename % n, "r") as label_file:
        y = int(label_file.readline()[0])
    with open(graph_filename % n, "r") as graph_file:
        lines = graph_file.readlines()
        N = int(lines[0])
        G = np.empty((N,N))
        for k in range(N):
            G[k] = list(map(int,lines[k+1].split()))

    ### ハイパーパラメータ設定。ここではヒントに倣った。
    D, T = 8, 2
    alpha = 1.0e-4

    ### GNN2の定義と勾配降下学習。
    # ステップごとに損失関数loss()の値をリストlossesに追記。
    gnn2 = GNN2(D,T)
    losses = [gnn2.loss(G,y)]
    for i in range(32):
        gnn2.descendant_update(G,y,alpha)
        loss = gnn2.loss(G,y)
        losses.append(loss)

    ### 結果出力
    # ファイルに保存する場合。
    # np.save('task2_losses.npy', np.array(losses))

    # 単に表示する。
    print('Data No.%d' % n)
    print(np.array(losses))
    # 32ステップの間だけでも明らかに減少していくことがわかる。
    # （データ番号のほかパラメータ初期値が乱数なので、具体的な結果は毎回違う。）
