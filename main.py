import sys
import h5py
import argparse

def main(arguments):
    '''
    Main logic
    '''

    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # General options
    parser.add_argument("--log_file", help="Path to file to log progress", type=str, default='')
    parser.add_argument("--im_path", help="Path to h5py file containing images to obfuscate", type=str, default='')

    # Data options
    parser.add_argument("--data_path", help="Path to hdf5 files containing training data", type=str, default='')
    parser.add_argument("--im_dim", help="Image dimension along a side (assuming square currently)", type=int)
    parser.add_argument("--n_channels", help="Number of image channels (1 for B/W, 3 for color)", type=int, default=3)

    # Model options
    parser.add_argument("--model", help="Model architecture to use", type=str, default='simple')
    parser.add_argument("--n_kernels", help="Number of convolutional filters", type=int, default=64)
    parser.add_argument("--load_model_from", help="Path to load model from", type=str, default='')
    parser.add_argument("--save_model_to", help="Path to save model to", type=str, default='')

    args = parser.parse_args(arguments)

    # Train or load a model
    if args.load_model_from:
        raise NotImplementedError
    else:
        if args.model == 'simple':
            model = simple_cnn(args)

        model.train(args.data_path)

    # Load image
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

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
