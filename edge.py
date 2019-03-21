import cv2 as cv
import numpy as np
import os

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

cv.dnn_registerLayer('Crop', CropLayer)

# Load the model.
net = cv.dnn.readNet('deploy.prototxt', 'hed_pretrained_bsds.caffemodel')


li_dir = os.listdir('dataset')
for dir in li_dir:
    li_photos = os.listdir('dataset/{}'.format(dir))
    for photo in li_photos:
        print('dataset/{0}/{1}'.format(dir, photo))
        frame = cv.imread('dataset/{0}/{1}'.format(dir, photo))
        frame = cv.resize(frame, (192, 192))

        average = frame.mean(axis=0).mean(axis=0)
        average = (average[2], average[1], average[0])


        print(average)

        inp = cv.dnn.blobFromImage(frame, scalefactor=1.0, size=(192, 192),
                                   mean=average,
                                   swapRB=False, crop=False)
        net.setInput(inp)
        out = net.forward()
        out = out[0, 0]
        out = cv.resize(out, (frame.shape[1], frame.shape[0]))
        out = 255 * out
        out = out.astype(np.uint8)
        out=cv.cvtColor(out,cv.COLOR_GRAY2BGR)

        if not os.path.exists('dataset-Edged/{0}'.format(dir)):
            os.mkdir('dataset-Edged/{0}'.format(dir))

        cv.imwrite('dataset-Edged/{0}/{1}'.format(dir, photo), out)