#!/bin/bash

# ===== Make global disconnections ======

jack_disconnect system:capture_2 pure_data:input1

# synth -> Pd
# jack_connect system:capture_1 pure_data:input0
# jack_connect system:capture_1 pyo:input_1 

jack_connect pure_data:output2 pyo:input_1


jack_connect pyo:output_1 pure_data:input1

# here we could connect pyo:output_1 to the jt call as well

jack_connect pure_data:output0 system:playback_1
jack_connect pure_data:output1 system:playback_2
jack_connect pure_data:output2 pyo:input_1