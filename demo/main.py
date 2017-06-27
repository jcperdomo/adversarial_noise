import os
import sys
import h5py
import argparse
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, redirect, \
        send_from_directory, flash, url_for
from werkzeug.utils import secure_filename

from src.codebase.models.simple_cnn import SimpleCNN
from src.codebase.generators.fast_gradient import FastGradientGenerator
from src.codebase.utils.utils import log

UPLOAD_FOLDER = 'temp/'
ALLOWED_EXTENSIONS = set(['jpg', 'png'])

# define web app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/test')
def home():
    return render_template('test.html')

# need to be able to upload image

# helper function
def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        fh = request.files['file']
        if not fh.filename:
            flash('No file selected')
            return redirect(request.url)
        if fh and allowed_file(fh.filename):
            filename = secure_filename(fh.filename)
            fh.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# return new image

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
    parser.add_argument("--eps", help="Magnitude of the noise", type=float, default=.3)
    parser.add_argument("--alpha", help="Magnitude of random initialization for noise, 0 for none", type=float, default=.0)

    args = parser.parse_args(sys.argv[1:])

    if args.data_path[-1] != '/':
        args.data_path += '/'
    with h5py.File(args.data_path+'params.hdf5', 'r') as f:
        args.n_classes = f['n_classes'][0]
        args.im_size = f['im_size'][0]
        args.n_channels = f['n_channels'][0]

    generator = FastGradientGenerator(args)
    model = SimpleCNN(args)
    saver = tf.train.Saver()
    saver.restore(model.session, args.load_model_from)
    print('Loaded model from %s' % args.load_model_from)

    app.run()