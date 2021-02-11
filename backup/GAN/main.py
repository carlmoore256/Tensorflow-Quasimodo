# import tensorflow as tf
from pyo import *
from jackServer import JackServer
from audioStream import AudioServerManager
from predictionNetwork import PredictionNetwork
import numpy as np
import multiprocessing
import time


sr = 44100
block_size = 1024
channels = 2

init_arr = np.zeros((block_size,))

in_block = multiprocessing.Array('d', init_arr)
out_block = multiprocessing.Array('d', init_arr)


predictionNetwork = PredictionNetwork(block_size=block_size,
                                    in_block=in_block,
                                    out_block=out_block,
                                    lr=0.0001)
predictionNetwork.start_thread()

jackManager = JackServer(sr, block_size, channels)
jackManager.system_kill_jack()
jackManager.start_server()

pyoManager = AudioServerManager(sr, block_size, channels)
pyoManager.start_server(predictionNetwork, in_block, out_block)


time.sleep(3600)

pyoManager.kill_server()
jackManager.kill_server()
jackManager.system_kill_jack()
