if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    epochs = 10

    data = np.load('task4a_losses01.npz')
    data.allow_pickle=True
    losses_SGD = data['arr_0']
    losses_mSGD = data['arr_1']
    losses_Adam = data['arr_2']

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
