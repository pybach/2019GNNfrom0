from task4a import *
#from task4b import *

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


    data = np.load('task4a_theta03.npz')
    data.allow_pickle=True
    Theta_SGD = data['arr_0']
    Theta_mSGD = data['arr_1']
    Theta_Adam = data['arr_2']

    D, T = 8, 2
    gnn3 = GNN3(D,T)
    gnn3.Theta[:] = Theta_Adam

    pred = np.round(gnn3.predict(graphs)).astype(int)
    for i in range(Ndata):
        print(pred[i])