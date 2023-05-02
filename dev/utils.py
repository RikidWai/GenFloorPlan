import numpy as np


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False


    def add_counter(self):
        self.counter += 1
        print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
        if self.counter >= self.patience:
            print('INFO: Early stopping')
            self.early_stop = True


    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif val_loss == np.nan:
            self.add_counter()
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0 # reset counter if validation loss improves
        elif self.best_loss - val_loss < self.min_delta:
            self.add_counter()
        else:
            # unknown situation
            self.add_counter()