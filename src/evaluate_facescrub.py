import os
import pdb
import sys
import h5py
import string
import boto3
import base64
import random
import argparse
import numpy as np
from scipy.misc import imsave

def random_string(length=10):
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(length))

def main(arguments):
    '''
    Main logic
    '''

    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--data_path", help="Path to hdf5 containing images to evaluate", type=str)
    parser.add_argument("--tmp_path", help="Path to temporary location to store images", type=str, default='tmp')

    args = parser.parse_args(arguments)

    with h5py.File(args.data_path, 'r') as fh:
        clean = fh['ims'][:].transpose((0,2,3,1))
        corrupt = fh['noisy_ims'][:].transpose((0,2,3,1))
    if not os.path.exists(args.tmp_path):
        os.makedirs(args.tmp_path)

    client = boto3.client(service_name='rekognition')

    n_clean_unk, n_corrupt_unk = 0, 0
    n_corrupt_wrong, n_corrupt_right = 0, 0
    wrong_conf, right_conf_change = 0., 0.
    for (clean_im, corrupt_im) in zip(clean, corrupt):
        clean_path = '%s/%s.png' % (args.tmp_path, random_string())
        imsave(clean_path, clean_im)
        with open(clean_path, 'r') as fh:
            im = fh.read()
            im_bytes = bytearray(im)
        clean_resp = client.recognize_celebrities(Image={'Bytes':im_bytes})
        if not clean_resp['CelebrityFaces']:
            n_clean_unk += 1
            os.remove(clean_path)
            continue
        clean_celeb = clean_resp['CelebrityFaces'][0]

        corrupt_path = '%s/%s.png' % (args.tmp_path, random_string())
        imsave(corrupt_path, corrupt_im)
        with open(corrupt_path, 'r') as fh:
            im = fh.read()
            im_bytes = bytearray(im)
        corrupt_resp = client.recognize_celebrities(Image={'Bytes':im_bytes})
        if not corrupt_resp['CelebrityFaces']:
            n_corrupt_unk += 1
        else:
            corrupt_celeb = corrupt_resp['CelebrityFaces'][0]
            if clean_celeb['Name'] != corrupt_celeb['Name']:
                n_corrupt_wrong += 1
                wrong_conf += corrupt_celeb['MatchConfidence']
            else:
                n_corrupt_right += 1
                right_conf_change += clean_celeb['MatchConfidence'] - \
                        corrupt_celeb['MatchConfidence']

        os.remove(clean_path)
        os.remove(corrupt_path)
    os.rmdir(args.tmp_path)

    print("Total n ims: %d, n clean unks: %d, n corrupt unks: %d" % 
            (clean.shape[0], n_clean_unk, n_corrupt_unk))
    if n_corrupt_wrong:
        print("# corrupt wrong: %d, wrong confidence: %.3f" % 
                (n_corrupt_wrong, wrong_conf / n_corrupt_wrong))
    else:
        print("# corrupt wrong: %d" % n_corrupt_wrong)
    if n_corrupt_right:
        print("# corrupt right: %d, avg confidence change: %3f" %
                (n_corrupt_right, right_conf_change / n_corrupt_right))
    else:
        print("# corrupt right: %d" % n_corrupt_right)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
