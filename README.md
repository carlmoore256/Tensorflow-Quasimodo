# Tensorflow-Quasimodo
A slightly insane way of training neural networks in real-time with live network streamed audio data

A small, unsupervised generative audio CNN autoencoder is trained in real-time with audio data buffered from a low-latency JackTrip connection via Pyo and Jack. A spin on Alvin Lucier's "Quasimodo The Great Lover" (1970), where several participants form a feedback loop over audio transmitted at a distance. In this case, we had several people connected through JackTrip, and at one node, this autoencoder was inserted into the loop as an input (training) and output (inference.) This was an attempt to direct some creative energy towards a realtime audio TensorFlow pipeline for servers. It became more practical to focus on client training and inference for the scope of this project however. 

This is not an example on how to actually train deep learning models for practical reasons! It's pretty raw with very specific platfrom requirements and dependencies, I might get around to cleaning it up.
