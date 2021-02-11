import cv2
import numpy as np
import time
import multiprocessing

class LatentVisualizer(multiprocessing.Process):
    def __init__(self, model, dims=(256,256)):
        super(LatentVisualizer, self).__init__()
        self.model = model
        self.dims = dims
        self.latent = self.model.latent_activation

    def run(self):
        while True:
            # time.sleep(0.1)
            image = np.asarray(self.latent)
            image = image.reshape((1,image.shape[0]))
            image +=1 
            image *= 128
            # print(image)
            image = image.astype(np.uint8)
            # print(image)

            image = cv2.resize(image, (self.dims[0],self.dims[1]), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('latent visualization', image)
            cv2.waitKey(1)