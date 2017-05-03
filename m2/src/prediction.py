
import glob
import cv2
import caffe

import numpy as np
from caffe.proto import caffe_pb2

caffe.set_mode_gpu() 

IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227


'''
Image processing helper function
'''
def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


'''
Reading mean image, caffe model and its weights 
'''
#Load mean image
mean_blob = caffe_pb2.BlobProto()
with open('../data/mean.binaryproto', 'rb') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

net = caffe.Net('../caffe_model/deploy.prototxt',
                '../caffe_model/snapshot/model_transfer_iter_400.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))
'''
Making predicitions
'''
##Reading image paths
test_img_paths = [img_path for img_path in glob.glob("../data/test/*jpg")]

test_ids = []
preds = []
i = 0
results = []
#Making predictions
for img_path in test_img_paths:
    if i % 100 == 0:
        print i, 'has been predicted'
    i+= 1
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']

    test_ids = img_path.split('/')[-1][:-4]
    test_id = test_ids.split('\\')[-1]
    ratio = pred_probas[0][1] if pred_probas.argmax() == 1 else pred_probas[0][0]
    results.append([int(test_id), float(pred_probas[0][1])])

results = sorted(results, key = lambda x:x[0])

'''
Making submission file
'''
with open("../result.csv","w") as f:
    f.write("id,label\n")
    for result in results:
        f.write(str(result[0])+","+str(result[1])+"\n")
f.close()
