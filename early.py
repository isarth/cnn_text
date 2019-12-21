from fastai.text import *

class EarlyStopping(Callback):
    def __init__(self,learner,save_path, opt='loss',pos=0):
        super().__init__()
        self.learner=learner
        self.save_path=save_path
        self.opt = opt
        self.pos = pos
    def on_train_begin(self):
        self.best_val_loss=100.0
        self.best_met = 0.0
    def on_epoch_end(self,metrics):
        val_loss = metrics[0]
        #m = metric[self.pos]
        if self.opt=='loss':
            if val_loss < self.best_val_loss:
               self.best_val_loss=val_loss
               print (f'Saving model with loss: {val_loss}')
               torch.save(self.learner.state_dict(),self.save_path)
        else:
            m = metrics[self.pos]
            if m > self.best_met:
               self.best_met = m
               print (f'Saving model with metric: {m}')
               torch.save(self.learner.state_dict(),str(self.save_path)+str(np.round(m,2))+'.pth')
