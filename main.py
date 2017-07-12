import os
import sys
import pdb
import h5py
import argparse
import StringIO
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imsave

from flask import Flask, render_template, request, redirect, \
        send_from_directory, flash, url_for, jsonify, send_file
from werkzeug.utils import secure_filename

from src.codebase.models.simple_cnn import SimpleCNN
from src.codebase.generators.fast_gradient import FastGradientGenerator
from src.codebase.utils.utils import log
from src.codebase.utils.dataset import Dataset

# constants
API_NAME = 'illnois'
VERSION = 'v0.1'

# web app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'test'

# globals for hacky / lazy stuff
true_class = -1

def encode_arr(arr):
    fh = StringIO.StringIO()
    imsave(fh, np.squeeze(arr), format='png')
    return fh.getvalue().encode('base64')

@app.route('/')
def index():
    return render_template('index.html')

###################
### API METHODS ###
###################
# Long term, probably not sustainable to send the entire image over the server

@app.route('/%s/api/%s/obfuscate' % (API_NAME, VERSION), methods=['POST'])
def obfuscate():
    #global true_class
    im = np.array(request.json).reshape((1,32,32,3))
    noise = generator.generate((im, np.array([true_class])), model, args)
    preds = model.predict(im+noise)
    enc_noise = encode_arr(noise / (2*generator.eps) + .5)
    enc_im = encode_arr(im+noise)
    return jsonify(preds=preds[0].tolist(),
        noise_src='data:image/png;base64,'+enc_noise,
        obf_src='data:image/png;base64,'+enc_im)

@app.route('/%s/api/%s/predict' % (API_NAME, VERSION), methods=['POST'])
def predict():
    global true_class
    im = np.array(request.json).reshape((1,32,32,3))
    preds = model.predict(im)
    true_class=  np.argmax(preds[0])
    return jsonify(preds=preds[0].tolist())

if __name__ == '__main__':
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
    parser.add_argument("--load_model_from", help="Path to load model from. \
                                                    When loading a model, \
                                                    this argument must match \
                                                    the import model type.", 
                                                    type=str, default='')
    parser.add_argument("--save_model_to", help="Path to save model to", type=str, default='')

    # Training options
    parser.add_argument("--n_epochs", help="Number of epochs to train for", type=int, default=5)
    parser.add_argument("--optimizer", help="Optimization algorithm to use", type=str, default='adam')
    parser.add_argument("--batch_size", help="Batch size", type=int, default=200)
    parser.add_argument("--learning_rate", help="Learning rate", type=float, default=1.)

    # SimpleCNN options
    parser.add_argument("--n_modules", help="Number of convolutional modules to stack (shapes must match)", type=int, default=6)

    # Generator options
    parser.add_argument("--generate", help="1 if should build generator and obfuscate images", type=int, default=1)
    parser.add_argument("--generator", help="Type of noise generator to use", type=str, default='fast_gradient')
    parser.add_argument("--generator_optimizer", help="Optimizer to use for Carlini generator", type=str, default='adam')
    parser.add_argument("--eps", help="Magnitude of the noise", type=float, default=.1)
    parser.add_argument("--alpha", help="Magnitude of random initialization for noise, 0 for none", type=float, default=.0)
    parser.add_argument("--n_generator_steps", help="Number of iterations to run generator for", type=int, default=1)
    parser.add_argument("--generator_opt_const", help="Optimization constant for Carlini generator", type=float, default=.1)
    parser.add_argument("--generator_confidence", help="Confidence in obfuscated image for Carlini generator", type=float, default=0)
    parser.add_argument("--generator_learning_rate", help="Learning rate for generator optimization when necessary", type=float, default=.1)

    args = parser.parse_args(sys.argv[1:])

    if args.data_path[-1] != '/':
        args.data_path += '/'
    with h5py.File(args.data_path+'params.hdf5', 'r') as f:
        args.n_classes = f['n_classes'][0]
        args.im_size = f['im_size'][0]
        args.n_channels = f['n_channels'][0]

    generator = FastGradientGenerator(args)
    model = SimpleCNN(args)
    model.load_weights(args.load_model_from)
    print('Loaded model from %s' % args.load_model_from)

    app.run()

def setup_app():

    # General options
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--data_path", help="Path to hdf5 files containing training data", type=str, default='')

    # Model options
    parser.add_argument("--model", help="Model architecture to use", type=str, default='simple')
    parser.add_argument("--n_kernels", help="Number of convolutional filters", type=int, default=128)
    parser.add_argument("--kern_size", help="Kernel size", type=int, default=3)
    parser.add_argument("--init_scale", help="Initialization scale (std around 0)", type=float, default=.1)
    parser.add_argument("--load_model_from", help="Path to load model from. \
                                                    When loading a model, \
                                                    this argument must match \
                                                    the import model type.", 
                                                    type=str, default='')

    # SimpleCNN options
    parser.add_argument("--n_modules", help="Number of convolutional modules to stack (shapes must match)", type=int, default=5)

    # Generator options
    parser.add_argument("--generator", help="Type of noise generator to use", type=str, default='fast_gradient')
    parser.add_argument("--generator_optimizer", help="Optimizer to use for Carlini generator", type=str, default='adam')
    parser.add_argument("--eps", help="Magnitude of the noise", type=float, default=.1)
    parser.add_argument("--alpha", help="Magnitude of random initialization for noise, 0 for none", type=float, default=.0)
    parser.add_argument("--n_generator_steps", help="Number of iterations to run generator for", type=int, default=1)
    parser.add_argument("--generator_opt_const", help="Optimization constant for Carlini generator", type=float, default=.1)
    parser.add_argument("--generator_confidence", help="Confidence in obfuscated image for Carlini generator", type=float, default=0)
    parser.add_argument("--generator_learning_rate", help="Learning rate for generator optimization when necessary", type=float, default=.1)

    args = parser.parse_args(['--data_path', 'src/data/cifar10',
                                '--load_model_from', 'src/checkpoints/06-26/cifar10.ckpt'])

    if args.data_path[-1] != '/':
        args.data_path += '/'
    with h5py.File(args.data_path+'params.hdf5', 'r') as f:
        args.n_classes = f['n_classes'][0]
        args.im_size = f['im_size'][0]
        args.n_channels = f['n_channels'][0]

    generator = FastGradientGenerator(args)
    model = SimpleCNN(args)
    model.load_weights(args.load_model_from)
    print('Loaded model from %s' % args.load_model_from)

    return model, generator, args

model, generator, args = setup_app()