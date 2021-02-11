#!/bin/bash

# ===== Make global disconnections ======

jack_disconnect system:capture_1 pure_data:input0
jack_disconnect system:capture_2 pure_data:input1


jack_connect pyo:output_1 52.25.61.105:send_1

jack_connect 52.25.61.105:receive_1 pure_data:input0

jack_connect pure_data:output2 pyo:input_1
