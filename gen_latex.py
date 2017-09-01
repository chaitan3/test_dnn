#!/usr/bin/python2

import subprocess
import numpy as np
import matplotlib.pyplot as plt

data_dir = 'latex/'

def get_symbols():
    with open(data_dir + 'symbols.txt') as f:
        return [x[:-1] for x in f.readlines()]

def gen_images(symbols):
    for index, symbol in enumerate(symbols):
        print symbol
        cmd = 'l2p -i \'{}\' -o {}image_{}.png'.format(symbol, data_dir, index)
        subprocess.check_call(cmd, shell=True)

def get_images(symbols):
    data = []
    for index in range(0, len(symbols)):
        data.append(plt.imread(data_dir + 'image_{}.png'.format(index)))
    return data

def get_data():
    symbols = get_symbols()
    images = get_images(symbols)
    return {x: y for x, y in zip(symbols, images)}

if __name__ == '__main__':
    symbols = get_symbols()
    gen_images(symbols)
