#!/usr/bin/python2

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import random

data_dir = 'latex_data/'

def get_symbols():
    with open(data_dir + 'symbols.txt') as f:
        return [x[:-1] for x in f.readlines()]

def gen_images(symbols):
    for index, symbol in enumerate(symbols):
        print index, symbol
        cmd = 'l2p -i \'{}\' -o {}image_{}.png'.format(symbol, data_dir, index)
        subprocess.check_call(cmd, shell=True)

def gen_processed_images(symbols):
    with open(data_dir + 'processed_symbols.txt', 'w') as f:
        index = len(symbols)
        for symbol in symbols:
            print index, symbol
            cmd = 'l2p -i \'{}\' -o {}image_{}.png'.format(symbol, data_dir, index)
            subprocess.check_call(cmd, shell=True)
            f.write(symbol)
            index += 1

def get_processed_symbols():
    with open(data_dir + 'processed_symbols.txt') as f:
        return [x[:-1] for x in f.readlines()]

def get_images(symbols):
    data = []
    for index in range(0, len(symbols)):
        data.append(plt.imread(data_dir + 'image_{}.png'.format(index)))
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
    gen_images(symbols)
    gen_processed_images(symbols)
