#!/bin/bash


# synth->pd
jack_connect system:capture_1 pure_data:input0

# pd->send to noah
jack_connect pure_data:output1 69.117.42.215:send_1

# recieve from artem->pd
jack_connect 147.9.76.229:receive_1 pure_data:input2

# pd->pyo
jack_connect pure_data:output2 pyo:input_1
# pyo->pd
jack_connect pyo:output_1 pure_data:input1


jack_connect pure_data:output0 system:playback_1
jack_connect pure_data:output1 system:playback_2

