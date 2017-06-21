import pdb

class Dataset:
    '''
    Data loader class.

    TODO
        - shuffle methods
    '''

    def __init__(self, ins, outs, args):
        '''
        inputs:
            - path: path to HDF5 file containing {training,validation} {inputs,outputs}
        '''
        assert ins.shape[1] == args.im_size and ins.shape[2] == args.im_size
        assert ins.shape[-1] == args.n_channels
        self.ins = ins
        self.outs = outs
        self.n_ins = ins.shape[0]
        self.batch_size = args.batch_size
        self.n_batches = ins.shape[0] / args.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        return (self.ins[idx*batch_size:(idx+1)*batch_size], 
                self.outs[idx*batch_size:(idx+1)*batch_size])
