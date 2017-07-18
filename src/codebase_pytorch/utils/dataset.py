import pdb
import math
import torch

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
        assert len(ins.shape) == 4
        assert ins.shape[1] == args.n_channels
        assert ins.shape[2] == args.im_size and ins.shape[3] == args.im_size
        self.ins = torch.FloatTensor(ins)
        self.outs = torch.LongTensor(outs.astype(int))
        self.n_ins = ins.shape[0]
        self.batch_size = args.batch_size
        self.n_batches = int(math.ceil(1.0 * ins.shape[0] / args.batch_size))

    def __getitem__(self, idx):
        batch_size = self.batch_size
        return (self.ins[idx*batch_size:(idx+1)*batch_size], 
                self.outs[idx*batch_size:(idx+1)*batch_size])
