import os
import sys
import pdb
import h5py
import argparse
import numpy as np
from PIL import Image
from scipy.misc import imread, imresize, imsave

def load_images(path):
    '''
    Load data from raw images

    Inputs:
        - path to data, structured as follow:
            - data
                - person1
                    - person1_im1.png
                    - person1_im2.png
                    - ...
                - person2
                    - person2_im1.png
                ...
    Outputs:
        - name2idx: dict from string name to idx
        - idx2ims: dict from idx to ims
    '''
    if path[-1] != '/':
        path = path + '/'
    if not args.n_classes:
        args.n_classes = len(os.listdir(path))
    name2idx = {name:idx for (idx, name) in \
            enumerate(os.listdir(path)[:args.n_classes])}
    n_ims = sum([len(os.listdir(path+person)) for person in \
            name2idx.keys()])
    ins = np.zeros((n_ims, args.im_size, args.im_size, args.n_channels))
    outs = np.zeros((n_ims, ))
    counter = 0
    for person, idx in name2idx.iteritems():
        n_person_ims = len(os.listdir(path+person))
        for i, im_path in enumerate(os.listdir(path+person)):
            ins[counter+i] = load_im(path+person+'/'+im_path)
        outs[counter:counter+n_person_ims] = idx
        counter += n_person_ims
    return ins, outs, name2idx

def load_im(path):
    '''
    Load an image from path
    '''
    try:
        im = Image.open(path)
        return np.array(im, dtype=np.float32) #/ 255.
    except IOError as e:
        print "\tUnable to open %s" % path

def augment(ins, outs):
    return ins, outs

def create_dataset(ins, outs, split):
    with h5py.File(args.out_path + split + ".hdf5", 'w') as f:
        f['ins'] = ins
        f['outs'] = outs

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_path', help='path to raw data', type=str, default='')
    parser.add_argument('--load_data_from', help='optional path to hdf5 file containing loaded data', type=str, default='')
    parser.add_argument('--save_data_to', help='optional path to save hdf5 file containing loaded data', type=str, default='')
    parser.add_argument('--out_path', help='path to folder to contain output files', type=str, default='')

    parser.add_argument('--normalize', help='1 if normalize', type=int, default=0)

    parser.add_argument('--n_classes', help='number of classes to use', type=int, default=0)
    parser.add_argument('--n_tr_classes', \
            help='number of classes (before augmentation) for training, \
            remaining classes are split evenly among validation and test', \
            type=float, default=.8)
    parser.add_argument('--n_te_exs', help='number of test examples', type=int, default=0)
    
    parser.add_argument('--im_size', help='size of image along a side (assuming square images)', type=int, default=128)
    parser.add_argument('--n_channels', help='number of image channels', type=int, default=3)
    parser.add_argument('--thresh', help='threshold (in (0,1)) for image binarization, 0 for none', type=float, default=0.)
    parser.add_argument('--resize', help='size (along a side) to resize to, 0 for none', type=int, default=0)
    args = parser.parse_args(arguments)
    
    print 'Loading data...'
    if args.load_data_from:
        print '\tReading data from %s' % args.load_data_from
        with h5py.File(args.load_data_from, 'r') as f:
            ins = f['ins'][:]
            outs = f['outs'][:]
            args.n_classes = f['n_classes'][:]
    else:
        print '\tReading data from images'
        ins, outs, name2idx = load_images(args.data_path)

        pairs = [(idx, name) for (name, idx) in name2idx.iteritems()]
        pairs.sort(key=lambda x: x[0])
        if args.out_path[-1] != '/':
            args.out_path += '/'
        with open(args.out_path + 'classes.txt', 'w') as fh:
            for idx, name in pairs:
                fh.write('%d\t%s\n' % (idx, name))

        if args.save_data_to:
            with h5py.File(args.save_data_to, 'w') as f:
                f['ins'] = ins
                f['outs'] = outs
                f['n_classes'] = np.array([args.n_classes])
            print '\tSaved loaded data to %s' % args.save_data_to
        else:
            print '\tAre you sure you don\'t want to save images to HDF5?'
    ins = ins.transpose((0,3,1,2))
    n_ims, n_classes = ins.shape[0], args.n_classes
    print '\tLoaded %d images for %d classes' % (n_ims, n_classes)

    if args.normalize:
        mean = np.mean(ins, axis=(0,2,3), keepdims=1)
        std = np.std(ins, axis=(0,2,3), keepdims=1)
        ins = (ins - mean) / std
    if augment:
        ins, outs = augment(ins, outs)

    p = np.random.permutation(n_ims)
    tr_split_pt = int(args.n_tr_classes * n_ims)
    if not args.n_te_exs:
        val_split_pt = int((1 + args.n_tr_classes)/2 * n_ims)
    else:
        val_split_pt = tr_split_pt + args.n_te_exs
    ins, outs = ins[p], outs[p]
    tr_ins, tr_outs = ins[:tr_split_pt], outs[:tr_split_pt]
    te_ins, te_outs = ins[tr_split_pt:val_split_pt], outs[tr_split_pt:val_split_pt]
    val_ins, val_outs = ins[val_split_pt:], outs[val_split_pt:]
    print '\tSplit sizes: %d, %d, %d' % (len(tr_ins), len(val_ins), len(te_ins))
    print 'Data loaded!'

    # create episodes
    if args.out_path[-1] != '/':
        args.out_path += '/'
    print 'Creating data...'
    create_dataset(tr_ins, tr_outs, "tr")
    print '\tFinished training data'
    create_dataset(val_ins, val_outs, "val")
    print '\tFinished validation data'
    create_dataset(te_ins, te_outs, "te")
    print '\tFinished test data'
    with h5py.File(args.out_path + "params.hdf5", 'w') as f:
        f['n_classes'] = np.array([n_classes], dtype=np.int32)
        f['n_channels'] = np.array([args.n_channels], dtype=np.int32)
        f['im_size'] = np.array([args.im_size], dtype=np.int32)
        if args.normalize:
            f['mean'] = mean
            f['std'] = std
    print 'Done!'

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
