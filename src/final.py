from task4b import *
from task4a import Adam

if __name__ == '__main__':
#    import matplotlib.pyplot as plt
    import sys
    import os
    pid = os.getpid()
 
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
    (D,T,Nw,epochs) = map(int,sys.argv[1:5])

    # GNN3とGNN4
    # あまり意味はないかもしれないが、気休め程度に初期値を合わせておく。
    if Nw == 1:
        gnn = GNN3(D,T)
    else:
        gnn = GNN4(D,T,Nw=Nw)

    # それぞれ学習。
    losses = gnn.fit(x, y, validation=(vx,vy),
                           epochs=epochs,
                           optimizer=Adam())
    # lossデータをファイルに保存
    np.save("task4b_losses_%d_%d_%d_%d_%d.npz" % (D,T,Nw,epochs,pid), losses)
    np.save("task4b_theta_%d_%d_%d_%d_%d.npz" % (D,T,Nw,epochs,pid), gnn.Theta)
    # 的中率をテキストで
    np.savetxt("task4b_acc_%d_%d_%d_%d_%d.csv" % (D,T,Nw,epochs,pid), losses[5])
