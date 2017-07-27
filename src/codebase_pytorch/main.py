import os
import pdb
import sys
import h5py
import torch
import argparse
import numpy as np
from scipy.misc import imsave

# Helper stuff
from src.codebase_pytorch.utils.dataset import Dataset
from src.codebase.utils.utils import log as log
from src.codebase_pytorch.utils.hooks import print_outputs, print_grads

# Classifiers
from src.codebase_pytorch.models.ModularCNN import ModularCNN
from src.codebase_pytorch.models.mnistCNN import MNISTCNN
from src.codebase_pytorch.models.squeezeNet import SqueezeNet, squeezenet1_0, squeezenet1_1
from src.codebase_pytorch.models.openFace import openFaceClassifier

# Generators
from src.codebase_pytorch.generators.random import RandomNoiseGenerator
from src.codebase_pytorch.generators.fgsm import FGSMGenerator
from src.codebase_pytorch.generators.carlini_l2 import CarliniL2Generator

def main(arguments):
    '''
    Main logic
    '''

    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # General options
    parser.add_argument("--no_cuda", help="disables CUDA training", type=int, default=0)
    parser.add_argument("--log_file", help="Path to file to log progress", type=str)
    parser.add_argument("--data_path", help="Path to hdf5 files containing training data", type=str, default='')
    parser.add_argument("--im_file", help="Path to h5py? file containing images to obfuscate", type=str, default='')
    parser.add_argument("--out_file", help="Optional hdf5 filepath to write obfuscated images to", type=str)
    parser.add_argument("--out_path", help="Optional path to folder to save images to", type=str)

    # Model options
    parser.add_argument("--model", help="Model architecture to use", type=str, default='modular')
    parser.add_argument("--n_kerns", help="Number of convolutional filters", type=int, default=64)
    parser.add_argument("--kern_size", help="Kernel size", type=int, default=3)
    parser.add_argument("--init_scale", help="Initialization scale (std around 0)", type=float, default=.1)
    parser.add_argument("--init_dist", help="Initialization distribution", type=str, default='normal')
    parser.add_argument("--load_model_from", help="Path to load model from. When loading a model, this argument must match the import model type.", type=str, default='')
    parser.add_argument("--save_model_to", help="Path to save model to", type=str, default='')
    parser.add_argument("--load_openface", help="Path to load pretrained openFace model from.", type=str, default='src/codebase_pytorch/models/openFace.ckpt')

    # Training options
    parser.add_argument("--n_epochs", help="Number of epochs to train for", type=int, default=5)
    parser.add_argument("--optimizer", help="Optimization algorithm to use", type=str, default='adam')
    parser.add_argument("--batch_size", help="Batch size", type=int, default=200)
    parser.add_argument("--lr", help="Learning rate", type=float, default=.1)
    parser.add_argument("--momentum", help="Momentum", type=float, default=.5)
    parser.add_argument("--weight_decay", help="Weight decay", type=float, default=0.0)
    parser.add_argument("--nesterov", help="Momentum", type=bool, default='false')

    # ModularCNN options
    parser.add_argument("--n_modules", help="Number of convolutional modules to stack (shapes must match)", type=int, default=6)

    # Generator options
    parser.add_argument("--generate", help="1 if should build generator and obfuscate images", type=int, default=1)
    parser.add_argument("--generator", help="Type of noise generator to use", type=str, default='fast_gradient')
    parser.add_argument("--target", help="Method for selecting class generator should 'push' image towards.", type=str, default='none')
    parser.add_argument("--eps", help="Magnitude of the noise", type=float, default=.1)
    parser.add_argument("--alpha", help="Magnitude of random initialization for noise, 0 for none", type=float, default=.0)
    parser.add_argument("--generator_opt_const", help="Optimization constant for Carlini generator", type=float, default=.1)
    parser.add_argument("--generator_confidence", help="Confidence in obfuscated image for Carlini generator", type=float, default=0.)

    # Generator training options
    parser.add_argument("--generator_optimizer", help="Optimizer to use for Carlini generator", type=str, default='adam')
    parser.add_argument("--generator_lr", help="Learning rate for generator optimization when necessary", type=float, default=.1)
    parser.add_argument("--n_generator_steps", help="Number of iterations to run generator for", type=int, default=1)


    args = parser.parse_args(arguments)

    log_fh = open(args.log_file, 'w')
    if args.data_path[-1] != '/':
        args.data_path += '/'
    with h5py.File(args.data_path+'params.hdf5', 'r') as f:
        args.n_classes = int(f['n_classes'][0])
        args.im_size = int(f['im_size'][0])
        args.n_channels = int(f['n_channels'][0])
    log(log_fh, "Processing %d types of images of size %d and %d channels" % (args.n_classes, args.im_size, args.n_channels))
    # TODO log more parameters
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        log(log_fh, "Using CUDA")

    # Build the model
    log(log_fh, "Building model...")
    if args.model == 'modular':
        model = ModularCNN(args)
        log(log_fh, "\tBuilt modular CNN with %d modules" % (args.n_modules))
    elif args.model == 'mnist':
        model = MNISTCNN(args)
        log(log_fh, "\tBuilt MNIST CNN")
    elif args.model == 'squeeze':
        #model = squeezenet1_1(pretrained=True, n_classes=args.n_classes, use_cuda=args.cuda)
        model = SqueezeNet(n_classes=args.n_classes, use_cuda=args.cuda)
    elif args.model == 'openface':
        model = openFaceClassifier(args)
    else:
        raise NotImplementedError
    if args.cuda:
        model.cuda()
    log(log_fh, "\tDone!")

    log(log_fh, "Training...")
    with h5py.File(args.data_path+'tr.hdf5', 'r') as fh:
        tr_data = Dataset(fh['ins'][:] - .5, fh['outs'][:] - .5, args)
    with h5py.File(args.data_path+'val.hdf5', 'r') as fh:
        val_data = Dataset(fh['ins'][:] - .5, fh['outs'][:] - .5, args)
    model.train_model(args, tr_data, val_data, log_fh)
    log(log_fh, "Done!")

    if args.generate:
        assert args.im_file

        # Load images to obfuscate
        log(log_fh, "Generating noise for images...")
        with h5py.File(args.im_file, 'r') as fh:
            te_data = Dataset(fh['ins'][:] - .5, fh['outs'][:] - .5, args)
        log(log_fh, "\tLoaded images!")
        _, clean_acc_old = model.evaluate(te_data)

        # Generate the noise
        if args.generator == 'random':
            generator = RandomNoiseGenerator(args)
            generator_s = 'random noise'
        elif args.generator == 'carlini_l2':
            generator = CarliniL2Generator(args, te_data.n_ins)
            generator_s = 'Carlini L2'
        elif args.generator == 'fgsm':
            generator = FGSMGenerator(args)
            generator_s = 'FGSM'
        else:
            raise NotImplementedError

        if args.target == 'random':
            data = Dataset(te_data.ins, np.random.randint(args.n_classes, size=te_data.n_ins), args)
            target_s = 'random'
        elif args.target == 'least':
            preds = model.predict(te_data)
            targs = np.argmin(preds, axis=1)
            data = Dataset(te_data.ins.numpy(), targs, args)
            target_s = 'least likely'
        elif args.target == 'next_likely':
            preds = model.predict(te_data)
            one_hot = np.zeros((te_data.n_ins, args.n_classes))
            one_hot[np.arange(te_data.n_ins), te_data.outs.numpy().astype(int)] = 1
            targs = np.argmax(preds * (1. - one_hot), axis=1)
            data = Dataset(te_data.ins.numpy(), targs, args)
            target_s = 'second most likely'
        elif args.target == 'none':
            data = te_data
            target_s = 'no'
        else:
            raise NotImplementedError
        log(log_fh, "\tBuilt %s generator targeting %s class with eps %.3f" %
                (generator_s, target_s, args.eps))
        noise = generator.generate(data, model, args, log_fh)
        log(log_fh, "Done!")

        # TODO handle out of range pixels
        # Compute the corruption rate
        log(log_fh, "Computing corruption rate...")
        corrupt_ins = te_data.ins.numpy() + noise 
        #corrupt_ins = np.max(np.min(te_data.ins + noise, 1.0), 0.0)
        corrupt_data = Dataset(corrupt_ins, te_data.outs, args)    
        _, clean_acc = model.evaluate(te_data)
        _, corrupt_acc = model.evaluate(corrupt_data)
        log(log_fh, "\tOriginal accuracy: %.3f \tnew accuracy: %.3f" % 
                (clean_acc, corrupt_acc))

        te_data.ins += .5
        corrupt_data.ins += .5

        # Save noise and images
        if args.out_file:
            with h5py.File(args.out_file, 'w') as fh:
                fh['noise'] = noise
                fh['ims'] = te_data.ins.numpy()
                fh['noisy_ims'] = corrupt_data.ins.numpy()
            log(log_fh, "Saved image and noise data to %s" % args.out_file)

        if args.out_path:
            for i, (clean_im, corrupt_im) in \
                    enumerate(zip(te_data.ins.numpy(), corrupt_data.ins.numpy())):
                imsave("%s/%d_clean.png" % (args.out_path, i), np.squeeze(clean_im))
                imsave("%s/%d_corrupt.png" % (args.out_path, i), np.squeeze(corrupt_im))
            log(log_fh, "Saved images to %s" % args.out_path)

        log(log_fh, "Done!")

    log_fh.close()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
