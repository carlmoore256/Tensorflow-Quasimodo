import subprocess

class JackServer():
    # manages the running jack connection server
    def __init__(self, sr, block_size, channels):
        self.sr = sr
        self.block_size = block_size
        self.channels = channels
        # self.system_kill_jack() #clear any existing jackd

    def start_server(self):
        try:
            self.system_kill_jack()
        except:
            print('failed to terminate any existing server')

        self.jackd = subprocess.Popen(['jackd',
                    # '--autoconnect', 'a'
                    '-d', 'coreaudio',
                    '-r', str(self.sr), 
                    '-p', str(self.block_size),
                    '-c', str(self.channels)],stderr=subprocess.DEVNULL)
                    
        print('started jack server!')

    def kill_server(self):
        self.jackd.kill()
        print('killed jack server!')

    def system_kill_jack(self):
        kill_proc = subprocess.Popen(['killall', 'jackd'])
        stdout, stderr = kill_proc.communicate()
        print(stdout)
