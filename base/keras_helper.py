from keras import backend as K
import numpy as np
from keras.callbacks import Callback
class ModelCheckpointPlus(Callback):
    '''
    定义最优代数为val_loss-val_acc的最小值
    '''
    def __init__(self, filepath, monitor='val_loss+', verbose=0,
                 save_best_only=True, save_weights_only=False,
                 mode='auto',verbose_show=5):
        super(ModelCheckpointPlus, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.verbose_show=verbose_show
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if self.save_best_only:
            if self.monitor=='val_loss+':
                loss_val = logs.get('val_loss')
                acc_val = logs.get('val_acc')
                if loss_val is not None and acc_val is not None:
                    current = loss_val-acc_val
                else:
                    current=None
            else:
                current=logs.get(self.monitor)
                    
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch, self.monitor, self.best,
                                 current, filepath))
                    self.best = current
                    self.best_loss = logs.get('val_loss')
                    self.best_acc = logs.get('val_acc')
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)
            else:
                self.model.save(filepath, overwrite=True)
        
        if self.verbose_show>0 and epoch%self.verbose_show==0:
            print("epoch: %d - loss: %.4f - acc: %.4f - val_loss: %.4f - val_acc: %.4f"
                  %(epoch,logs.get('loss'),logs.get('acc'),logs.get('val_loss'),logs.get('val_acc')))
            