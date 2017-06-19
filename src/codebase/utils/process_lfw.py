import os
import sys
import argparse
import h5py
import pdb
import pickle
import numpy as np
from PIL import Image
from scipy.misc import imread, imresize, imsave

def open_image(path):
    im = Image.open(path)
    if args.resize:
        im = imresize(im, size=(args.resize, args.resize)) # might need to transpose?
    return np.array(im, dtype=np.float32) / 255.

def load_data(path):
    '''
    Load data from raw images
    Inputs:
        - path to LFW data 
    Outputs:
        - ims: the images
        - targs: the class for the images
    '''

    n_ims = 0
    for person in os.listdir(path):
        n_ims += len(os.listdir(path+person))
    sample_im = open_image(path+person+'/'+os.listdir(path+person)[0])
    args.im_size = im_size = sample_im.shape[0]
    args.n_channels = n_channels = sample_im.shape[-1]

    ims = np.zeros((n_ims, im_size, im_size, n_channels))
    targs = np.zeros((n_ims))
    counter = 0
    for i, person in enumerate(os.listdir(path)):
        raw_ims = os.listdir(path+person)
        person_ims = []
        for raw_im in raw_ims:
            person_ims.append(open_image(path+person+'/'+raw_im))
        targs[counter:counter+len(person_ims)] = 1. * i
        ims[counter:counter+len(person_ims)] = np.array(person_ims)
        counter += len(person_ims)

    assert counter == n_ims
    print '\tLoaded %d images for %d classes' % (n_ims, len(path))
    return ims, targs

def augment(data):
    '''
    TODO

    Augment face data with
        - reflections
        - crops
    '''
    try:
        for person, ims in data.iteritems(): # probably better to do this while creating episodes
            data[person] = np.vstack((data[person], np.array([np.fliplr(im) for im in ims])))
    except Exception as e:
        pdb.set_trace()
    return augmented

def create_dataset(ins, outs, split):
    with h5py.File(args.out_path + split + ".hdf5", 'w') as f:
        f['ins'] = ins
        f['outs'] = outs
    print '\tFinished %s data' % split

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_path', help='path to data', type=str, default='')
    parser.add_argument('--load_data_from', help='optional path to hdf5 file containing loaded data', type=str, default='')
    parser.add_argument('--save_data_to', help='optional path to save hdf5 file containing loaded data', type=str, default='')
    parser.add_argument('--out_path', help='path to folder to contain output files', type=str, default='')

    parser.add_argument('--n_tr_shards', help='number of shards to split tr data amongst', type=int, default=1)
    parser.add_argument('--n_val_shards', help='number of shards to split val data amongst', type=int, default=1)
    parser.add_argument('--n_te_shards', help='number of shards to split te data amongst', type=int, default=1)
    parser.add_argument('--n_tr_classes', help='fraction of classes (before augmentation) for training, \
            remaining classes are split evenly among validation and test', type=int, default=.8)
    parser.add_argument('--normalize', help='1 if normalize', type=int, default=0)
    parser.add_argument('--reflections', help='1 if augment with reflections', type=int, default=0)
    parser.add_argument('--crops', help='1 if augment with crops', type=int, default=0)
    parser.add_argument('--resize', help='dimension (along a side) to resize to, 0 for none', type=int, default=0)
    args = parser.parse_args(arguments)
    
    print 'Loading data...'
    if args.load_data_from:
        print '\tReading data from %s' % args.load_data_from
        f = h5py.File(args.load_data_from, 'r')
        ins = f['ins'][:]
        outs =  f['outs'][:]
        f.close()
        args.im_size = ins.shape[1]
        args.n_channels = ins.shape[-1]
    else:
        print '\tReading data from images'
        if args.data_path[-1] != '/':
            args.data_path += '/'
        ins, outs = load_data(args.data_path)

        if args.save_data_to:
            with h5py.File(args.save_data_to, 'w') as f:
                f['ins'] = ins
                f['outs'] = outs
            print '\tSaved loaded data to %s' % args.save_data_to
    n_ims = ins.shape[0]
    n_classes = np.max(outs) + 1

    if args.normalize:
        mean = np.mean(ins, axis = 0)
        std = np.std(ins, axis = 0)
        ins = (ins - mean) / std

    perm = np.random.permutation(n_ims)
    ins, outs = ins[perm], outs[perm]
    tr_split_pt = int(args.n_tr_classes * n_ims)
    val_split_pt = int(n_ims * (args.n_tr_classes + (1. - args.n_tr_classes)/2))
    print '\tData loaded!'
    print '\tSplit sizes: %d, %d' % (tr_split_pt, val_split_pt - tr_split_pt)

    # TODO
    if args.normalize or args.reflections or args.crops:
        raise NotImplementedError

    # create episodes
    if args.out_path[-1] != '/':
        args.out_path += '/'
    print 'Creating data...'
    create_dataset(ins[:tr_split_pt], outs[:tr_split_pt], "tr")
    create_dataset(ins[tr_split_pt:val_split_pt], outs[tr_split_pt:val_split_pt], "val")
    create_dataset(ins[val_split_pt:], outs[val_split_pt:], "te")
    with h5py.File(args.out_path + "params.hdf5", 'w') as f:
        f['n_classes'] = np.array([n_classes], dtype=np.int32)
        f['im_size'] = np.array([args.im_size], dtype=np.int32)
        f['n_channels'] = np.array([args.n_channels], dtype=np.int32)
        if args.normalize:
            f['mean'] = mean
            f['std'] = std
    print 'Done!'

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
