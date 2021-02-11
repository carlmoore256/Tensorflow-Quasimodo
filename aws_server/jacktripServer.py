import subprocess
import multiprocessing
import time

class JacktripServer(multiprocessing.Process):
    # manages the running jack connection server
    def __init__(self, channels, port_offset=0, 
                bit_depth=16, queue=4, redundancy=1, 
                autoconnect=False, zero_underrun=False):
        super(JacktripServer, self).__init__()
        self.channels = channels
        self.port_offset = port_offset
        self.bit_depth = bit_depth
        self.queue = queue
        self.redundancy = redundancy
        self.autoconnect = autoconnect
        self.zero_underrun = zero_underrun

    def start_server(self):
        try:
            self.system_kill_jacktrip()
        except:
            print('failed to terminate any existing server')
            
        print('starting jacktrip server!')

        command = ["jacktrip",
                    "-s",
                    "-n", str(self.channels),
                    "-o", str(self.port_offset),
                    "-b", str(self.bit_depth),
                    "-q", str(self.queue),
                    "-r", str(self.redundancy)]

        if self.autoconnect==False:
            command.append("--nojackportsconnect")
        if self.zero_underrun:
            command.append("-z")

        self.jacktrip = subprocess.Popen(command,
                        stderr=subprocess.PIPE) 
        stdout, stderr = self.jacktrip.communicate()

    def run(self):
        self.start_server()

    def kill_server(self):
        self.jacktrip.kill()
        print('killed jacktrip server!')

    def system_kill_jacktrip(self):
        kill_proc = subprocess.Popen(['killall', 'jacktrip'])
        stdout, stderr = kill_proc.communicate()
        print(stdout)
