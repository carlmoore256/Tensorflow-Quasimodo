#!/bin/bash

# ===== Make global disconnections ======

# jack_disconnect system:capture_1 pure_data:input0
jack_disconnect system:capture_2 pure_data:input1

jack_disconnect system:capture_1 52.25.61.105:send_1
jack_disconnect system:capture_2 52.25.61.105:send_2

# jack_disconnect system:playback_1 pure_data:output0
# jack_disconnect system:playback_2 pure_data:output1

# ===== Connect system cap to Pd ===============
# system [1 indexed] -> puredata [0 indexed]
jack_connect system:capture_1 pure_data:input0

# ===== Connect net inputs & outputs to Pd ======

# puredata [0 indexed] -> pyo [1 indexed]
jack_connect pure_data:output0 pyo:input_1
# pyo [1 indexed] -> puredata [0 indexed]; pyo[0] -> pd[1]
jack_connect pyo:output_1 pure_data:input1

# ===== Connect Pd output to speakers ======

# jack_connect pure_data:output0 system:playback_1
# jack_disconnect pure_data:output0 system:playback_1

jack_connect pure_data:output1 system:playback_2
jack_connect pure_data:output1 system:playback_1

jack_disconnect pure_data:output0 system:playback_1