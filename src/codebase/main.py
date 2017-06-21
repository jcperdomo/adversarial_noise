import sys
import h5py
import argparse
from models.simple_cnn import Simple_CNN
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
    parser.add_argument("--log_file", help="Path to file to log progress", type=str, default='')
    parser.add_argument("--im_path", help="Path to h5py? file containing images to obfuscate", type=str, default='')
    # out_file?

    # Data options
    parser.add_argument("--data_path", help="Path to hdf5 files containing training data", type=str, default='')

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
            model = Simple_CNN(args)
    log(log_fh, "\tDone!")

    log(log_fh, "Training...")
    tr_data = Dataset(args.data_path+'tr.hdf5', args)
    val_data = Dataset(args.data_path+'val.hdf5', args)
    model.train(tr_data, val_data, args, log_fh)
    if args.save_model_to:
        raise NotImplementedError

    '''
    # Load image to obfuscate
    with h5py.File(args.im_path, 'r') as fh:
        ims = fh['ims'][:]
        assert ims.shape[1] == args.im_dim and ims.shape[2] == args.im_dim
        assert ims.shape[-1] == args.n_channels

    # Get the noise (either image-specific or universal)
    if args.generator == 'deepfool':
        generator = deepfool()
    elif args.generator == 'fast_gradient':
        generator = fast_gradient()

    # Generate adversarial image and save
    noise = generator.generate(ims, model)
    '''

    log_fh.close()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
