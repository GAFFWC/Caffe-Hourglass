import sys
import caffe
import numpy as np
from caffe.proto import caffe_pb2

np.set_printoptions(threshold=sys.maxsize, precision=3, suppress=True)
class JointMseLossLayer(caffe.Layer):
    &quot;&quot;&quot; 
    bottom : prediction, target, target_weight (should be this order !)
    &quot;&quot;&quot;

    def setup(self, bottom, top):
        
        if len(bottom) != 3:
            raise Exception(&quot;Need 3 inputs to compute Joint Mse Loss.&quot;)
            

    def reshape(self, bottom, top):
	#print
        self.diff = bottom[0].data - bottom[1].data
	#print(self.diff)
        top[0].reshape(1)
	top[1].reshape(1)
        
    

    def forward(self, bottom, top):
	&quot;&quot;&quot;
	im = 2. * (im - np.min(im)) / np.ptp(im) - 1.
	&quot;&quot;&quot;
	output = bottom[0].data
	#print(np.min(output))
	#print(np.max(output))
	#print(np.max(output[0, 0, :]))
	#print(str(np.argmax(output[0, 0, :]) / 64) + &quot; &quot; + str(np.argmax(output[0, 0, :]) % 64))
	#output = (output - np.min(output)) / np.ptp(output)
	#print(output)
	#print(np.max(bottom[1].data))
        target = bottom[1].data
	#print(str(np.argmax(target[0, 0, :]) / 64) + &quot; &quot; + str(np.argmax(target[0, 0, :]) % 64))
	#print(target[0, 0])
        target_weight = bottom[2].data
	
        batch_size = bottom[0].data.shape[0]
        num_joints = bottom[0].data.shape[1]
        
	
        heatmaps_pred = output.reshape(batch_size, num_joints, -1)
        heatmaps_pred = np.transpose(heatmaps_pred, (1, 0, 2))
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))
        heatmaps_gt = np.transpose(heatmaps_gt, (1, 0, 2))
        loss = 0
        #print(target_weight.shape)
	
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
	    for batch in range(batch_size):
		temp = heatmap_pred[batch, :] * target_weight[batch, idx] - heatmap_gt[batch, :] * target_weight[batch, idx]
                loss += 0.5 * np.mean(temp**2)

        top[0].data[...] = loss / num_joints / batch_size
	
	#print(np.sum(bottom[0].data*bottom[1].data))
        
	idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]	

	acc = self.accuracy(output, target, idx)
	#for val in range(len(acc)):
	#    print(&quot;accuracy for point #&quot; + str(val) + &quot; : &quot; + str(acc[val])) 
	top[1].data[...] = acc[0]

    def backward(self, top, propagate_down, bottom):
	#print(bottom)
	#print(bottom[0].data)
	bottom[0].diff[...] = self.diff[...]

    def get_preds(self, scores):
	shape = scores.shape
	scores = scores.reshape(shape[0], shape[1], -1)
	scores *= 100000000.0
	scores = np.int32(scores)
	#print(scores)
	maxval= np.max(scores, 2)
	idx = np.argmax(scores, 2)
	maxval = maxval.astype(np.float)
	idx = idx.astype(np.float)
	maxval /= 100000000.0
	#print(maxval)
	maxval = maxval.view().reshape(shape[0], shape[1], -1) + 1
	idx = idx.view().reshape(shape[0], shape[1], 1) + 1
	preds = np.repeat(idx, 2, 2)
	#preds = np.float((preds))
	#print(scores)
	preds[:,:,0] = (preds[:,:,0] - 1) % shape[3] + 1
	preds[:,:,1] = np.floor((preds[:,:,1] - 1) / shape[3]) + 1

	#print(maxval.__gt__(0))
	pred_mask = np.repeat(maxval.__gt__(0), 2, 2)

	preds *= pred_mask
	#print(preds.shape)
	return preds

    def calc_dists(self, preds, target, normalize):
	
	preds = preds.astype(np.float)
	target = target.astype(np.float)
	dists = np.zeros((preds.shape[1], preds.shape[0]))
	for n in range(preds.shape[0]):
	    for c in range(preds.shape[1]):
		if target[n, c, 0] > 1 and target[n, c, 1] > 1:
		    dists[c, n] = np.linalg.norm(preds[n, c, :] - target[n, c, :], 2)/normalize[n]
		else:
		    dists[c, n] = -1

	    	#print(dists[c, n])

	return dists

    def accuracy(self, output, target, idxs, thr=0.5):
	
	preds = self.get_preds(output)
	gts   = self.get_preds(target)
	norm  = np.ones(preds.shape[0]) * output.shape[3] / 10
	dists = self.calc_dists(preds, gts, norm)

	#print(norm)
	acc = np.zeros(len(idxs) + 1)
	avg_acc = 0
	cnt = 0
	#print(idxs)
	for i in range(len(idxs)):
	    acc[i + 1] = self.dist_acc(dists[idxs[i] - 1])
	    if acc[i + 1] >= 0:
		avg_acc = avg_acc + acc[i + 1]
		cnt += 1

	if cnt != 0:
	    acc[0] = avg_acc / cnt

	return acc

    def dist_acc(self, dist, thr=0.5):
	
	dist = dist[dist != -1]

	if len(dist) > 0:
	    ret = 1.0 * (dist < thr).sum().item() / len(dist)
	    #print(ret)
	    return ret
	
	else:
	    return -1
