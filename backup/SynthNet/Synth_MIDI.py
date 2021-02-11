# manages all midi output to hardware device
import mido
from mido import Message

class MidiManager():
    def __init__(self, channel, param_controls, device_name='minilogue SOUND'):
        self.channel = channel
        # to reveal device name, run mido.get_output_names()
        self.outport = mido.open_output(device_name)
        self.param_controls = param_controls


    def set_control_values(self, values):
        for c, v in zip(self.param_controls, values):
            msg = Message('control_change', channel=self.channel, control=c, value=v)
            self.outport.send(msg)

    def play_note(self, note):
        msg = Message('note_on', note=note, 
                        velocity=120, time=1, 
                        channel=self.channel)
        self.outport.send(msg)

    def train_step(self, values):
        self.play_note()
        self.set_control_values(values)