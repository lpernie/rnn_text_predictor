from tensorflow.keras.callbacks import Callback

class NBatchLogger(Callback):
    """
    A Logger that log average performance per minibatch
    """

    def __init__(self, validation_data, freq=2, val=False, verbose=False):
        super(NBatchLogger, self).__init__()
        self.validation_data = validation_data
        self.val = val
        if not isinstance(freq, int):
            print('Warning: select an integer frequency. Converting freq to int.')
            freq = int(freq)
        if freq < 2:
            print(
                'You want to save the history at least twice per epoch. Otherwise it make no sense to use this Callback')
        self.freq = freq
        self.n_baches_done = 0
        self._batches_where_i_save_history = []
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        logs = {} if logs is None else logs
        self.my_metrics = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        n_batches = int(self.params['samples'] / self.params['batch_size'])

        for i in range(1, self.freq + 1)[::-1]:
            self._batches_where_i_save_history.append(int(n_batches / i))

        if self.verbose:
            print(f'You have {n_batches} minibatch in each epoch.')
            print(f'I will save the history at batch n:{self._batches_where_i_save_history}')

    def on_epoch_begin(self, epoch, logs=None):
        self.n_baches_done = 0

    def on_batch_end(self, batch, logs=None):
        logs = {} if logs is None else logs
        self.n_baches_done += 1
        if self.n_baches_done in self._batches_where_i_save_history:
            self.my_metrics['loss'].append(logs.get('loss'))
            self.my_metrics['accuracy'].append(logs.get('accuracy'))
            if self.val:
                # Validation data is not available at batch level, you have to compute it
                x = self.validation_data[0]
                y = self.validation_data[1]
                val_loss, val_acc = self.model.evaluate(x, y, verbose=0)
                self.my_metrics['val_loss'].append(val_loss)
                self.my_metrics['val_accuracy'].append(val_acc)

