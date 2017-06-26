import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, send_from_directory, flash, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'temp/'
ALLOWED_EXTENSIONS = set(['jpg', 'png'])

# define web app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

'''
@app.route('/')
def main():
    return render_template('index.html')
'''

# need to be able to upload image
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
    return '''
    <!doctype html>
    <title>Upload new file</title>
    <h1>Upload new file</h1>
    <form method=post enctype=multipart/form-data>
        <p><input type=file name=file>
            <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# return new image

if __name__ == '__main__':
    app.run()
