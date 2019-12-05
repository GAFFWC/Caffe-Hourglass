import caffe
import cv2
import lmdb
from caffe.proto import caffe_pb2
import numpy as np
import scipy.misc
from random import shuffle
from threading import Thread
from PIL import Image

from tools import SimpleTransformer

class PoseDataLayer(caffe.Layer):
    def setup(self, bottom, top):
 
        self.params = eval(self.param_str)       

        self.batch_size = self.params[&apos;batch_size&apos;]

        
        self.batch_loader = BatchLoader(self.params)

        top[0].reshape(self.batch_size, 3, 256, 256)
        
        top[1].reshape(self.batch_size, 16, 64, 64) # target(16 * 64 * 64)

        top[2].reshape(self.batch_size, 16) # target_weight(16)



    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            im, target, target_weight = self.batch_loader.load_next_image(self.params)
#            print(im.shape)
#            print(target.shape)
#            print(target_weight.shape)
#	    print(im)
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = target
            top[2].data[itt, ...] = target_weight


    def backward(self, bottom, propagate_down, top):
        pass



class BatchLoader(object):
   
    def __init__(self, params):
        #weight_file = open(params[&apos;weight_source&apos;])
	#target_weights = weight_file.read().split(&apos; &apos;)
        #print(target_weights)
        
        self.batch_size = params[&apos;batch_size&apos;]
        self.im_shape = [256, 256]
        self.cur = 0
        self.transformer = SimpleTransformer()
            
        lmdb_file = params[&apos;source&apos;]
       
        lmdb_env = lmdb.open(lmdb_file)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        #self.dataset_size = sum(1 for _ in lmdb_txn.cursor())
        self.groups = []    
        
        #print(&quot;total &quot; + str(self.dataset_size) + &quot; images.&quot;)
	count = 0        
        for key, value in lmdb_cursor:
	    count += 1
            group = []
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value)
	    
	    #print(datum)
            data = caffe.io.datum_to_array(datum)
	    label = datum.target_weight
	    #print(len(datum.data))
            #im = np.fromstring(datum.data, dtype=np.uint8).reshape(3, 256, 256)
	    im = data 
            #im = np.transpose(data, (2, 1, 0))
	    
            img = Image.fromarray(im, &apos;RGB&apos;)
	    #print(im)
	    im = im.astype(np.float)
	    im /= 255.0
	    #print(im)
	    #print(im) 
            #im -= np.uint8(params[&apos;mean_value&apos;])
	    #im = im.astype(&apos;float32&apos;)
	    #im = 2. * (im - np.min(im)) / np.ptp(im) - 1.
	    #print(np.max(im))
	    #print(np.min(im))
	    #print(im)
	    #img.save(&quot;../docker_share/&quot; + str(int(key)) + &quot;.jpg&quot;)
            #im = scipy.misc.imresize(im, self.im_shape)
	    #print(im)
            target = np.fromstring(datum.target, dtype=np.float32).reshape(16, 64, 64)
            target_weight = []
	    for idx in range(16):
                if label / pow(2, 15 - idx) > 0:
                    target_weight.append(1)
                else:
                    target_weight.append(0)

		label %= pow(2, 15 - idx)
	    #print(target_weight)
            target_weight = np.asarray(target_weight, dtype=np.float32).reshape(16)
            group.append(im)
            group.append(target)
            group.append(target_weight)
            
            self.groups.append(group)

            if count > 0 and count % 1000 == 0:
                print(&quot;loaded &quot; + str(count) + &quot; images.&quot;)
	    #print(count)
	    #if int(key) > 10:
	#	break

        self.dataset_size = len(self.groups)
        self.indexlist = []
        for i in range(self.dataset_size):
            self.indexlist.append(i)

    def load_next_image(self, params):
        
        if self.cur == self.dataset_size:
            self.cur = 0
            shuffle(self.indexlist)

        index = self.indexlist[self.cur]
        im = self.groups[index][0]
        target = self.groups[index][1]
        target_weight = self.groups[index][2]

        #print(im)
        im = im[:, :, ::-1]
        im = im.transpose((2, 0, 1))
        #print(im) 
	target = np.float32(target)
        target = target[:, :, ::-1]
        #target = target.transpose((2, 0, 1))
        
        
        self.cur += 1
        return im, target, target_weight
