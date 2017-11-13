#!/usr/bin/python2

import subprocess
import numpy as np
import random
from skimage import io
from skimage import transform

data_dir = 'latex_data/'

def get_symbols():
    with open(data_dir + 'symbols.txt') as f:
        return [x[:-1] for x in f.readlines()]


def gen_images(symbols):
    for index, symbol in enumerate(symbols):
        print index, symbol
        cmd = 'l2p -i \'{}\' -o {}image_{}.png'.format(symbol, data_dir, index)
        subprocess.check_call(cmd, shell=True)

def gen_processed_images(symbols, images):
    with open(data_dir + 'processed_symbols.txt', 'w') as f:
        index = len(symbols)
        for symbol, image in zip(symbols, images):
            _index = [index]
            def write_image(data):
                print _index[0], symbol
                io.imsave('{}image_{}.png'.format(data_dir, _index[0]), data)
                f.write(symbol + '\n')
                _index[0] += 1

            def apply_tf(tf):
                return transform.warp(image, inverse_map=tf, cval=1)

            for i in range(0, 10):
                x, y = 12*(np.random.rand(2)-0.5)
                tf = transform.AffineTransform(translation=(int(x),int(y)))
                write_image(apply_tf(tf))

            for i in range(0, 10):
                l, h = np.log(0.8), np.log(1.2)
                sx, sy = np.exp(np.random.rand(2)*(h-l) + l)
                tf = transform.AffineTransform(scale=(sx, sy))
                write_image(apply_tf(tf))

            for i in range(0, 10):
                ang = 20*(np.random.rand()-0.5)*np.pi/180
                tf = transform.AffineTransform(shear=ang)
                write_image(apply_tf(tf))

            index = _index[0]

def get_processed_symbols():
    with open(data_dir + 'processed_symbols.txt') as f:
        return [x[:-1] for x in f.readlines()]

def get_images(symbols):
    data = []
    for index in range(0, len(symbols)):
        data.append(io.imread(data_dir + 'image_{}.png'.format(index)))
    return data

def get_data():
    symbols = get_processed_symbols()
    images = get_images(symbols)
    n = len(symbols)
    test = set(random.sample(range(n), n/5))
    train = set(range(0, n))-test
    def get_subset_data(subset):
        return [symbols[x] for x in subset], [images[x] for x in subset]
    train_data = get_subset_data(train)
    test_data = get_subset_data(test)
    return train_data, test_data

if __name__ == '__main__':
    symbols = get_symbols()
    #gen_images(symbols)
    images = get_images(symbols)
    gen_processed_images(symbols, images)
