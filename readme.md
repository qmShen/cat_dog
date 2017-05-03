This is a project for the competition: Dog vs Cat https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition.

There are two version of the implementation, which distributed in two different folders m1 and m2, m1 is a shallow network and m2 is methods based on BAIR Reference CaffeNet (inspired by blog http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/)

Compared with these two methods, the m2 has a significant better performance for both accuracy and convergence rate.

This project is tested on windows10, cuda 8.0, vs2015 and python2.7

The caffe project should be complied and the directory caffe-windows/Build/Build/x64/Release into path, and add caffe-windows/Build/x64/Release/pycaffe/caffe to python \Lib\site-packages

M1.
1. We transform the data into leveldb and you can download the leveldb data from the http://pan.baidu.com/s/1pKGPvlT, directly put the three folders into the  dog_cat/m1/.

2. Run Train.bat in the current directory and the model will be put into the snapshot folder.

To be continued...

M2.
1. Decompress the train.zip data from the https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data, and put all the images into m2/data/train; decompress test.zip and put all the images into n2/data/test/

2. Enter the directory m2/code and run python generate_lmdb.py to generate lmdb files, the lmdb files will be found into m2/data/

3. Enter the directory m2/code/ run: generate_mean.bat to generate the mean image from each input image, the mean.binaryprot could be found in m2/data/.
compute_image_mean in the generate_mean.bat is an executable file from the caffe project.

4. Download bvlc_reference_caffenet.caffemodel(http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel) and put it into m2/caffe_model/

5. Enter the directory m2/code/ run Train.bat, the model will generated into the directory of m2/caffe_model/snapshot/

6. Enter the directory m2/code/ edit the prediction.py modify the line 42 to select one modeo from the directory m2/caffe_model/snapshot  python prediction.py and result.csv will be generated into the directory m2/
