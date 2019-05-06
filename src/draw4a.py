if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    epochs = 30

    data = np.load('task4a_losses03.npz')
    data.allow_pickle=True
    losses_SGD = data['arr_0']
    losses_mSGD = data['arr_1']
    losses_Adam = data['arr_2']


    margin = 0.025
    plt.xlim(-margin*epochs,epochs)

    flag = 1 # 0: acc, 1: loss
    if flag == 0:
        x_arr = np.array(range(epochs+1))
        acc_max = 1
        plt.ylim(acc_max*(-margin),acc_max*(1+margin))
        p1=plt.plot(x_arr,losses_SGD[5])
        p2=plt.plot(x_arr,losses_mSGD[5])
        p3=plt.plot(x_arr,losses_Adam[5])
        plt.legend((p1[0],p2[0],p3[0]),
                   ("acc, SGD", "acc, momentum SGD", "acc, Adam"),
                   loc=1)
    else:
        ### 学習曲線の描画
        n = len(losses_SGD[5])
        loss_max = 2
        x_arr = np.array(range(n))*(epochs/n)
        plt.ylim(loss_max*(-margin),loss_max*(1+margin))
        p1=plt.plot(x_arr,losses_SGD[0])  # loss, SGD
        p2=plt.plot(x_arr,losses_SGD[3])  # vloss, SGD
        p3=plt.plot(x_arr,losses_mSGD[0])    # loss, Momentum SGD
        p4=plt.plot(x_arr,losses_mSGD[3])    # vloss, Momentum SGD
        p5=plt.plot(x_arr,losses_Adam[0])    # loss, Adam
        p6=plt.plot(x_arr,losses_Adam[3])    # vloss, Adam
        plt.legend((p1[0],p2[0],p3[0] ,p4[0],p5[0],p6[0]),
                   ("loss, SGD", "vloss, SGD",
                    "loss, momentum SGD", "vloss, momentum SGD",
                    "loss, Adam", "vloss, Adam"
                    ),
                    loc=1)
    plt.grid(True)
    # 学習曲線プロットをファイルに保存
    plt.savefig("task4a_plot03b.pdf")
