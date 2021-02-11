from pyo import *
from jackServer import JackServer
from audioStream import AudioServerManager, PyoServer
from predictionNetwork import PredictionNetwork
from latentVisualizer import LatentVisualizer
# from predictionNetworkOffline import PredictionNetwork
# from jacktripServer import JacktripServer
# from VAEnet import VAE
# from VAEnetFFT import VAE_FFT
# from synthAE import SynthAutoencoder
import numpy as np
import multiprocessing
import subprocess
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
    parser.add_argument('-k', '--kernel_size', type=int, default=9)
    args = parser.parse_args()

    init_arr = np.zeros((args.block_size,))
    in_block = multiprocessing.Array('d', init_arr)
    out_block = multiprocessing.Array('d', init_arr)

    in_block[:] = np.zeros((args.block_size,))
    out_block[:] = np.zeros((args.block_size,))
    lock = multiprocessing.Lock()

    # =========PYO SERVER==========================

    pyoServer = PyoServer(
        sr=args.samplerate,
        block_size=args.block_size,
        channels=args.channels,
        in_block=in_block,
        out_block=out_block,
        lock=lock)
    pyoServer.start()

    # ========TF MODEL========================

    model = PredictionNetwork(block_size=args.block_size,
                in_block=in_block,
                out_block=out_block,
                lock=lock,
                lr=args.learning_rate,
                kernel_size=args.kernel_size)
    model.start_thread()

    # =======MAKE JACK CONNECTIONS=============
    subprocess.Popen(['./jack_connections_quasi.sh'])

    # ========LATENT VISUALIZER================

    vis = LatentVisualizer(
        model=model,
        dims=(256,64)
        )
    vis.start()

    time.sleep(10000)