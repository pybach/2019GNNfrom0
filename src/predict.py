from task4a import *
from task4b import *

if __name__ == '__main__':
    # 未知データ
    Ndata = 500
    datasetsdir='../../datasets/test/'
    graph_filename = datasetsdir + '%d_graph.txt'
    # label_filename = datasetsdir + '%d_label.txt'
    graphs = []
    # labels = []
    for i in range(Ndata):
        # with open(label_filename % i, "r") as label_file:
        #     labels.append(int(label_file.readline()[0]))
        with open(graph_filename % i, "r") as graph_file:
            lines = graph_file.readlines()
        N = int(lines[0])
        graph = np.empty((N,N))
        for k in range(N):
            graph[k] = list(map(int,lines[k+1].split()))
        graphs.append(graph)


    data = np.load('task4a_theta01.npz')
    data.allow_pickle=True
    Theta_SGD = data['arr_0']
    Theta_mSGD = data['arr_1']
    Theta_Adam = data['arr_2']

    D, T = 8, 2
    gnn3 = GNN3(D,T)
    

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
    p5=plt.plot(x_arr,losses_Adam[0])    # loss, Adam
    p6=plt.plot(x_arr,losses_Adam[2])    # vloss, Adam
    plt.grid(True)
    plt.legend((p1[0],p2[0],p3[0],p4[0],p5[0],p6[0]),
               ("loss, SGD", "vloss, SGD",
                "loss, momentum SGD", "vloss, momentum SGD",
                "loss, Adam", "vloss, Adam"
                ),
                loc=1)
    # 学習曲線プロットをファイルに保存
    plt.savefig("task4a_plot.pdf")
