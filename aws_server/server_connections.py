import subprocess
from jackServer import JackServer
from jacktripServer import JacktripServer
import multiprocessing
import argparse
import time


def jacktrip_server():
    print('starting jacktrip server')
    subprocess.Popen(['killall', 'jacktrip'])
    jacktrip = subprocess.Popen(['jacktrip',
                                '-s',
                                '--nojackportsconnect'])

    # jack_connect JackTrip:receive_1 JackTrip:send_1

def jack_connections():
    print('creating jack connections')
    connections = subprocess.Popen(['jack_connect',
                                    'JackTrip:receive_1',
                                    'JackTrip:send_1'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set up jack server')
    parser.add_argument('-b', '--backend', default='dummy')
    parser.add_argument('-p', '--block_size', default=1024)
    parser.add_argument('-c', '--channels', default=2)
    parser.add_argument('-r', '--samplerate', default=48000)

    parser.add_argument('-o', '--port_offset', default=0)
    parser.add_argument('-bd', '--bit_depth', default=16)
    parser.add_argument('-q', '--queue', default=4)
    parser.add_argument('-rd', '--redundancy', default=1)
    parser.add_argument('-a', '--autoconnect', default=False)
    parser.add_argument('-z', '--zero_underrun', default=False)

    args = vars(parser.parse_args())

    jack_server = JackServer(args['samplerate'], 
                            args['block_size'], 
                            args['channels'],
                            args['backend'])
    jack_server.start()

    time.sleep(3.0)

    jt_server = JacktripServer(args['channels'],
                                args['port_offset'],
                                args['bit_depth'],
                                args['queue'],
                                args['redundancy'],
                                args['autoconnect'],
                                args['zero_underrun'])
    jt_server.start()

    running=True
    while running:
        time.sleep(0.01)
    # jt_server.start_server()
    # jack_connections()