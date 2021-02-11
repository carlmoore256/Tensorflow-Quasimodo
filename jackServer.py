import subprocess
import multiprocessing

class JackServer(multiprocessing.Process):
    # manages the running jack connection server
    def __init__(self, sr, block_size, channels, backend='coreaudio'):
        super(JackServer, self).__init__()
        self.sr = sr
        self.block_size = block_size
        self.channels = channels
        self.backend = backend

    def start_server(self):
        try:
            self.system_kill_jack()
        except:
            print('failed to terminate any existing server')

        self.jackd = subprocess.Popen(['sudo', 'jackd',
                    # '--autoconnect', 'a'
                    '-d', self.backend,
                    '-d' 'AppleUSBAudioEngine:Audient:EVO4:14620000:1,2',
                    '-r', str(self.sr), 
                    '-p', str(self.block_size),
                    '-c', str(self.channels)],
                    stderr=subprocess.PIPE)
                    # stderr=subprocess.DEVNULL)

        stdout, stderr = self.jackd.communicate()
                    
        print('started jack server!')

    # def make_connections(self):

    def run(self):
        self.start_server()

    def kill_server(self):
        self.jackd.kill()
        print('killed jack server!')

    def system_kill_jack(self):
        kill_proc = subprocess.Popen(['killall', 'jackd'])
        stdout, stderr = kill_proc.communicate()
        print(stdout)
