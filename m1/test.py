import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
# Wei Yang 2015-08-19
# Source
#   Read LevelDB/LMDB
#   ==================
#       http://research.beenfrog.com/code/2015/03/28/read-leveldb-lmdb-for-caffe-with-python.html
#   Plot image
#   ==================
#       http://www.pyimagesearch.com/2014/11/03/display-matplotlib-rgb-image/
#   Creating LMDB in python
#   ==================
#       http://deepdish.io/2015/04/28/creating-lmdb-in-python/

lmdb_file = "features"
lmdb_env = lmdb.open(lmdb_file)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()
i = 0
for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    i += 1
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    print key, label, data
    # im = data.astype(np.uint8)
    # im = np.transpose(im, (2, 1, 0)) # original (dim, col, row)
    # # print "label ", label, data
    #
    # plt.imshow(im)
    # plt.show()

print i