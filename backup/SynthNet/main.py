from pyo import *
from jackServer import JackServer
from audioStream import AudioServerManager
from predictionNetwork import PredictionNetwork
from jacktripServer import JacktripServer
from VAEnet import VAE
from VAEnetFFT import VAE_FFT
from synthAE import SynthAutoencoder
import numpy as np
import multiprocessing
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remote Jacktrip \
                                    Tensorflow Model Training')
    parser.add_argument('-b', '--backend', default='coreaudio')
    parser.add_argument('-p', '--block_size', type=int, default=1024)
    parser.add_argument('-c', '--channels', type=int, default=2)
    parser.add_argument('-r', '--samplerate', type=int, default=44100)

    parser.add_argument('-o', '--port_offset', type=int, default=0)
    parser.add_argument('-bd', '--bit_depth', type=int, default=16)
    parser.add_argument('-q', '--queue', type=int, default=4)
    parser.add_argument('-rd', '--redundancy', type=int, default=1)
    parser.add_argument('-a', '--autoconnect', default=False)
    parser.add_argument('-z', '--zero_underrun', default=False)

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    args = parser.parse_args()

    init_arr = np.zeros((args.block_size,))
    in_block = multiprocessing.Array('d', init_arr)
    out_block = multiprocessing.Array('d', init_arr)
    lock = multiprocessing.Lock()

    time.sleep(3.0)

    pyoManager = AudioServerManager(args.samplerate, 
                                    args.block_size,
                                    args.channels)
    pyoManager.start_server(in_block, out_block, lock)

    param_controls = np.array([ 40, 24, 82, 35, 41, 81, 34, 20, 39,
                  19, 33, 83, 21, 50, 36, 37, 22, 49,
                  80, 44, 48, 51, 44, 58, 42,
                  84, 43, 45, 56, 23, 88, 27 ])

    model = SynthAutoencoder(block_size=args.block_size,
                in_block=in_block,
                out_block=out_block,
                lock=lock,
                lr=args.learning_rate,
                param_controls=param_controls)
    model.start_thread()
    time.sleep(3600)

    pyoManager.kill_server()
