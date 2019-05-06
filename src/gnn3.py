################################################################
### 課題3 損失・勾配の計算と勾配降下
################################################################
from gnn2 import *

# GNN1に損失関数・勾配を計算して勾配降下を行う機能を追加。
class GNN3(GNN2):
    def __init__(self, D, T,
                 W0=None, A0=None, b0=None, sigma=0.4,
                 epsilon=1.0e-3):
        super().__init__(D, T,
                         W0=W0, A0=A0, b0=b0, sigma=sigma,
                         epsilon=epsilon)
    def predict(self,x):
        return np.array([ self.p(g) for g in x ])

    def fit(self, x, y,
            epochs=1, batchsize=16, shuffle=True,
            validation=None,
            optimizer=None):
        if optimizer is None:
            optimizer = GNN3.SGD()

        if validation is None:
            vld = False
        else:
            vld = True
            vx, vy = validation
            vn = len(vy)
        n = len(y)

        batch_losses = []
        epoch_losses = []
        epoch_losses.append(np.mean([self.loss(x[i],y[i]) for i in range(n)]))
        if vld:
            batch_vlosses = []
            epoch_vlosses = []
            epoch_vlosses.append(np.mean([self.loss(vx[i],vy[i]) for i in range(vn)]))        

        for e in range(epochs):
            if shuffle:
                order = np.random.permutation(n)
            else:
                order = range(n)

            for b in range((n-1)//batchsize+1):
                start = batchsize * b
                end = min(start+batchsize, n)

                batch_loss = np.mean([ self.loss(x[i],y[i]) for i in order[start:end]])
                batch_losses.append(batch_loss)
                if vld:
                    batch_vloss = np.mean([ self.loss(vx[i],vy[i]) for i in np.random.randint(vn,size=end-start)])
                    batch_vlosses.append(batch_vloss)

                # Update Theta
                grad = np.mean([ self.grad(x[i],y[i]) for i in order[start:end]],axis=0)
                self.shift_Theta(optimizer.update(grad))

            epoch_loss = np.mean([self.loss(x[i],y[i]) for i in range(n)])
            epoch_losses.append(epoch_loss)
            if vld:
                epoch_vloss = np.mean([self.loss(vx[i],vy[i]) for i in range(vn)])
                epoch_vlosses.append(epoch_vloss)

        if vld:
            return batch_losses, epoch_losses, batch_vlosses, epoch_vlosses
        else:
            return batch_losses, epoch_losses

