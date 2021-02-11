#!/bin/bash

# ===== Make global disconnections ======

# jack_connect pure_data:output2 pyo:input_1


jack_connect pyo:output_1 pure_data:input1
jack_connect pure_data:output2 pyo:input1


jack_connect pure_data:output0 system:playback_1
jack_connect pure_data:output1 system:playback_2

