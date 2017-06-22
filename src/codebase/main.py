import pdb
import sys
import h5py
import argparse
from models.simple_cnn import SimpleCNN
from generators.fast_gradient import FastGradientGenerator
from utils.dataset import Dataset
from utils.utils import log as log

def main(arguments):
    '''
    Main logic
    '''

    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # General options
    parser.add_argument("--log_file", help="Path to file to log progress", type=str)
    parser.add_argument("--data_path", help="Path to hdf5 files containing training data", type=str, default='')
    parser.add_argument("--im_file", help="Path to h5py? file containing images to obfuscate", type=str, default='')
    parser.add_argument("--out_file", help="Optional hdf5 filepath to write obfuscated images to", type=str)

    # Model options
    parser.add_argument("--model", help="Model architecture to use", type=str, default='simple')
    parser.add_argument("--n_kernels", help="Number of convolutional filters", type=int, default=64)
    parser.add_argument("--kern_size", help="Kernel size", type=int, default=3)
    parser.add_argument("--init_scale", help="Initialization scale (std around 0)", type=float, default=.1)

    # Model logging options
    parser.add_argument("--load_model_from", help="Path to load model from", type=str, default='')
    parser.add_argument("--save_model_to", help="Path to save model to", type=str, default='')

    # Training options
    parser.add_argument("--n_epochs", help="Number of epochs to train for", type=int, default=5)
    parser.add_argument("--optimizer", help="Optimization algorithm to use", type=str, default='adam')
    parser.add_argument("--batch_size", help="Batch size", type=int, default=200)
    parser.add_argument("--learning_rate", help="Learning rate", type=float, default=1.)

    # SimpleCNN options
    parser.add_argument("--n_modules", help="Number of convolutional modules to stack (shapes must match)", type=int, default=6)

    # Generator options
    parser.add_argument("--generator", help="Type of noise generator to use", type=str, default='fast_gradient')
    parser.add_argument("--eps", help="Max l_inf norm of noise", type=float, default=.1)

    args = parser.parse_args(arguments)

    log_fh = open(args.log_file, 'w')
    if args.data_path[-1] != '/':
        args.data_path += '/'
    with h5py.File(args.data_path+'params.hdf5', 'r') as f:
        args.n_classes = f['n_classes'][0]
        args.im_size = f['im_size'][0]
        args.n_channels = f['n_channels'][0]
    log(log_fh, "Processing %d types of images of size %d and %d channels" % (args.n_classes, args.im_size, args.n_channels))

    # Train or load a model
    log(log_fh, "Building model...")
    if args.load_model_from:
        raise NotImplementedError
    else:
        if args.model == 'simple':
            model = SimpleCNN(args)
    log(log_fh, "\tDone!")

    log(log_fh, "Training...")
    with h5py.File(args.data_path+'tr.hdf5', 'r') as fh:
        tr_data = Dataset(fh['ins'][:], fh['outs'][:], args)
    with h5py.File(args.data_path+'val.hdf5', 'r') as fh:
        val_data = Dataset(fh['ins'][:], fh['outs'][:], args)
    model.train(tr_data, val_data, args, log_fh)
    if args.save_model_to:
        raise NotImplementedError
    log(log_fh, "\tDone!")
    _, val_acc = model.validate(val_data)
    log(log_fh, "Validation accuracy: %.3f" % val_acc)

    # Load image to obfuscate
    # TODO: clean up this code to be consistent with Dataset class
    #   - rework generator signature
    #   - generator is breaking shit
    log(log_fh, "Generating noise for images...")
    with h5py.File(args.im_file, 'r') as fh:
        test_ins = fh['ins'][:]
        test_outs = fh['outs'][:]
        assert test_ins.shape[1] == args.im_size and test_ins.shape[2] == args.im_size
        assert test_ins.shape[-1] == args.n_channels
    log(log_fh, "\tLoaded images!")
    clean_data = Dataset(test_ins, test_outs, args)    
    _, clean_acc = model.validate(clean_data)
    log(log_fh, "Clean accuracy: %.3f" % clean_acc)

    # Get the noise (either image-specific or universal)
    if args.generator == 'deepfool':
        generator = DeepFool()
    elif args.generator == 'fast_gradient':
        generator = FastGradientGenerator(args)
    log(log_fh, "\tGenerator built!")
    noise = generator.generate(test_ins, test_outs, model)
    log(log_fh, "\tDone!")

    # Compute the corruption rate
    log(log_fh, "Computing corruption rate...")
    clean_data = Dataset(test_ins, test_outs, args)    
    _, clean_acc = model.validate(clean_data)
    corrupt_data = Dataset(test_ins + noise, test_outs, args)    
    _, corrupt_acc = model.validate(corrupt_data)
    log(log_fh, "\tOriginal accuracy: %.3f, new accuracy: %.3f" % 
            (clean_acc, corrupt_acc))
    log(log_fh, "\tDone!")

    # Save noise and images
    if args.out_file:
        with h5py.File(args.out_file, 'w') as fh:
            fh['noise'] = noise
            fh['ims'] = test_ins
            fh['noisy_ims'] = test_ins + noise
        log(log_fh, "Saved images to %s" % args.out_file)

    log_fh.close()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
