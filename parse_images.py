import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
from sklearn.utils import shuffle
import os
import os.path
import pickle

datasets_path = os.path.expanduser("~/Documents/datasets/")

data_path = datasets_path + "catdog/"
pictures = os.listdir(data_path)
images = []
labels = []
i = -1
for picture in pictures:
    i += 1;
    img_path = data_path + picture

    img = imread(img_path)
    img = imresize(img, (32,32))
    #x = np.expand_dims(x, axis=0)
    #print image.shape
    #images[int(picture.split(".")[0])-1] = img
    images.append(img)
    #labels.append(1 if picture.split(".")[0] == "dog" else 0)
    if i % 1000 == 0:
        print i;

data = np.array(images).astype('uint8')
#labeldata = np.array(labels).astype('uint8')

print data.shape


#data, labeldata = shuffle(data,labeldata)

pickle.dump(data,open(datasets_path + "catdogall.data",'w'))
#pickle.dump(labeldata,open("./data/catdog_train.lbl",'w'))

exit(0)

data_path = "./data/catdog/test/"
pictures = os.listdir(data_path)
pictures.sort()

images = []
labels = []
i = -1
for k in range(1,12501):
    i += 1;
    img_path = data_path + str(k) + '.jpg'

    img = imread(img_path)
    img = imresize(img, (128,128))
    #x = np.expand_dims(x, axis=0)
    #print image.shape
    #images[int(picture.split(".")[0])-1] = img
    images.append(img)
    #labels.append(picture.split(".")[0])
    if i % 100 == 0:
        print i;

data = np.array(images).astype('uint8')
#labeldata = np.array(labels).astype('int32')

# fix order of test samples
#p = labeldata.argsort()
#labeldata = labeldata[p]
#data = data[p]
#print labeldata
print data.shape

pickle.dump(data,open("./data/catdog_test_128.data",'w'))
#pickle.dump(labeldata,open("./data/catdog_test.lbl",'w'))
