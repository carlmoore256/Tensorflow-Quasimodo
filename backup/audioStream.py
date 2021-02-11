from pyo import *
from threading import Thread
import multiprocessing
import numpy as np


class PyoServer(multiprocessing.Process):
    def __init__(self, sr, block_size, channels, predictionNetwork, in_block, out_block):
        super(PyoServer, self).__init__()
        self.daemon = True
        self._terminated = False
        self.sr = sr
        self.block_size = block_size
        self.channels = channels
        # idea - have a running ring buffer that is filled with samples from a neural network
        self.ringBuff = None
        # this class will request buffers from the prediction network to fill the ring buffer
        self.predictionNetwork = predictionNetwork

        self.in_block = in_block
        self.out_block = out_block

    def pyo_callback(self):
        self.in_block[:] = self.input_buff.copy()[:]
        self.arr[:] = self.out_block[:]
        # self.arr[:] = np.random.normal(0.0, 0.5, size=self.block_size)
        
    def run(self):
        self.server = Server(sr=self.sr, 
                            nchnls=self.channels, 
                            buffersize=self.block_size, 
                            duplex=1,
                            audio='jack')
        self.server.setJackAuto(xin=False,xout=False)
        self.server.boot().start()

        t = DataTable(size=self.block_size)
        osc = TableRead(t, freq=t.getRate(), loop=True, mul=0.1).out()
        
        self.arr = np.asarray(t.getBuffer()) # Share table's memory with np arr
        
        self.in_stream = Input().play()
        t2 = DataTable(size=self.block_size)
        self.input_buff = np.asarray(t2.getBuffer())

        fill = TableFill(self.in_stream, table=t2)

        self.server.setCallback(self.pyo_callback)

        # Keeps the process alive...
        while not self._terminated:
            time.sleep(0.001)

        self.server.stop()

    def stop(self):
        self._terminated = True

class AudioServerManager():
    def __init__(self, sr, block_size, channels):
        self.sr = sr
        self.block_size = block_size
        self.channels = channels

    def start_server(self, predictionNetwork, in_block, out_block):
        self.pyo_proc = PyoServer(self.sr, self.block_size, self.channels, 
                                    predictionNetwork, in_block, out_block)
        print(f'\n starting pyo audio server {self.pyo_proc} \n')
        self.pyo_proc.start()

    def kill_server(self):
        self.pyo_proc.stop()
        print("\n Stopped pyo SERVER!!! \n")
        exit()
