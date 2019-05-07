if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    epochs = 50

#    data.allow_pickle=True
    losses_gnn3 = np.load('task4b_losses_8_2_1_50_12690.npz.npy',allow_pickle=True)
    losses_gnn4 = np.load('task4b_losses_8_2_2_50_12772.npz.npy',allow_pickle=True)

    optstr = ['SGD', 'mSGD', 'Adam']
    # opts = [SGD, MomentumSGD, Adam]
    optn = 2
    # opt = opts[optn]

    margin = 0.025
    plt.xlim(-margin*epochs,epochs)

    flag = 0 # 0: acc, 1: loss
    if flag == 0:
        x_arr = np.array(range(epochs+1))
        acc_max = 1
        plt.ylim(acc_max*(-margin),acc_max*(1+margin))
        p1=plt.plot(x_arr,losses_gnn3[5])
        p2=plt.plot(x_arr,losses_gnn4[5])
        plt.legend((p1[0],p2[0]),
                   ("vacc, GNN3 (with %s)" % optstr[optn], "vacc, GNN4 (with %s)" % optstr[optn]),
                   loc=1)
    else:
        ### 学習曲線の描画
        n = len(losses_gnn3[0])
        loss_max = 2
        x_arr = np.array(range(n))*(epochs/n)
        plt.ylim(loss_max*(-margin),loss_max*(1+margin))
        p1=plt.plot(x_arr,losses_gnn3[0],zorder=0)  # loss, GNN3
        p2=plt.plot(x_arr,losses_gnn3[3],zorder=3)  # vloss, GNN3
        p3=plt.plot(x_arr,losses_gnn4[0],zorder=1)    # loss, GNN4
        p4=plt.plot(x_arr,losses_gnn4[3],zorder=4)    # vloss, GNN4
        plt.legend((p1[0],p2[0],p3[0],p4[0]),
                   ("loss, GNN3 (with %s)" % optstr[optn], "vloss, GNN3 (with %s)" % optstr[optn],
                    "loss, GNN4 (with %s)" % optstr[optn], "vloss, GNN4 (with %s)" % optstr[optn]),
                   loc=1)
    plt.grid(True)
    # plt.set_axisbelow(True)
    # 学習曲線プロットをファイルに保存
    plt.savefig("task4b_plot04a.pdf")
