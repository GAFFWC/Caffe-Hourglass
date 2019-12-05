import caffe
import cv2
import lmdb
from caffe.proto import caffe_pb2
import numpy as np
import scipy.misc
from random import shuffle
from threading import Thread
from PIL import Image
import sys
from tools import SimpleTransformer

np.set_printoptions(threshold=sys.maxsize, precision=3, suppress=True)
class ShowImageLayer(caffe.Layer):
    &quot;&quot;&quot; 
    bottom : prediction, target, target_weight (should be this order !)
    &quot;&quot;&quot;

    def setup(self, bottom, top):        
	pass            

    def reshape(self, bottom, top):
        pass
    

    def forward(self, bottom, top):
        inputs = bottom[0].data
	target = bottom[1].data
	output = bottom[2].data

	#print(inputs[0].shape)
	#print(inputs[0])
	#img.save(&quot;../docker_share/inp.jpg&quot;)
	gt_batch_img = Image.fromarray(self.batch_with_heatmap(inputs, target), &apos;RGB&apos;)
	pred_batch_img = Image.fromarray(self.batch_with_heatmap(inputs, output), &apos;RGB&apos;)

	gt_batch_img.save(&quot;../../docker_share/sample_inp.jpg&quot;)
	pred_batch_img.save(&quot;../../docker_share/sample_out.jpg&quot;)


    def backward(self, top, propagate_down, bottom):
        pass


    def color_heatmap(self, x):
        color = np.zeros((x.shape[0],x.shape[1],3))
        color[:,:,0] = self.gauss(x, .5, .6, .2) + self.gauss(x, 1, .8, .3)
        color[:,:,1] = self.gauss(x, 1, .5, .3)
        color[:,:,2] = self.gauss(x, 1, .2, .3)
        color[color > 1] = 1
        color = (color * 255).astype(np.uint8)
        return color

    def gauss(self, x, a, b, c, d = 0):
	return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

    def sample_with_heatmap(self, inp, out, num_rows = 2, parts_to_show=None):
	inp *= 255
        
	img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
        for i in range(3):
	    img[:, :, i] = inp[i, :, :]

	if parts_to_show is None:
	    parts_to_show = np.arange(out.shape[0])

	num_cols = int(np.ceil(float(len(parts_to_show)) / num_rows))
	size = img.shape[0] // num_rows

	full_img = np.zeros((img.shape[0], size * (num_cols + num_rows), 3), np.uint8)
	full_img[:img.shape[0], :img.shape[1]] = img

	#inp_small = scipy.misc.imresize(img, [size, size])
	inp_small = cv2.resize(img, dsize = (size, size))

	for i, part in enumerate(parts_to_show):
	    part_idx = part
            #out_resized = scipy.misc.imresize(out[part_idx], [size, size])
	    out_resized = cv2.resize(out[part_idx], dsize = (size, size))
            out_resized = out_resized.astype(float)/255
            out_img = inp_small.copy() * .3
            color_hm = self.color_heatmap(out_resized)
            out_img += color_hm * .7

            col_offset = (i % num_cols + num_rows) * size
            row_offset = (i // num_cols) * size
            full_img[row_offset:row_offset + size, col_offset:col_offset + size] = out_img

        return full_img


    def batch_with_heatmap(self, inputs, outputs, mean = np.array([0.0, 0.0, 0.0]), num_rows = 2, parts_to_show = None):
	batch_img = []
	for n in range(min(inputs.shape[0], 4)):
	    inp = inputs[n] + mean.view().reshape(3, 1, 1)
	    batch_img.append(self.sample_with_heatmap(inp.clip(0, 1), outputs[n], num_rows = num_rows, parts_to_show = parts_to_show))

	return np.concatenate(batch_img)
