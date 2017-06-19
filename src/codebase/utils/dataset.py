class dataset:
    '''
    Data loader class

    TODO
        - shuffle methods
    '''

    def __init__(self, data, labels, opt):
        '''
        inputs:
            - data: n_data x im_dim x im_dim x n_channels
        '''
        assert data.shape[1] == opt.im_dim and data.shape[2] == opt.im_dim
        assert data.shape[-1] == opt.n_channels
        assert data.shape[0] == labels.shape[0]
        self.data = data
        self.labels = labels
        self.n_data = data.shape[0]
        self.batch_size = opt.batch_size
        self.n_batches = data.shape[0] / opt.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        return (data[idx*batch_size:(idx+1)*batch_size], 
                labels[idx*batch_size:(idx+1)*batch_size])
