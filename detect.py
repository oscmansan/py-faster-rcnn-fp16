#!/usr/bin/python
import tools._init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
from os import walk
import time
import math

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

MODELS_DIR = 'models'
DATA_DIR = 'data'


def vis_detections(ax, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    inds = filter(lambda x: x >= thresh, dets[:, -1])
    if len(inds) == 0:
        return

    for i in inds:
    	i = int(i)
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    plt.axis('off')
    plt.tight_layout()


def detect(net, im_files):
    # Detect all object classes and regress object bounds
    N = len(im_files)
    sqrtN = int(math.ceil(math.sqrt(N)))
    fig, axarr = plt.subplots(sqrtN, sqrtN, sharey=True, figsize=(16,12))
    axarr = axarr.flatten()
    imarr = []
    avg = 0
    for i, im_file in enumerate(im_files):
	    imarr.append(cv2.imread(im_file))
	    start = time.clock()
	    scores, boxes = im_detect(net, imarr[i])
	    end = time.clock()
	    diff = end - start
	    avg += diff
	    print ('{:s} took {:.3f}s').format(im_file, diff)

	    print 'boxes.spape:', boxes.shape
	    print 'scores.shape:', scores.shape

	    CONF_THRESH = 0.5
	    NMS_THRESH = 0.3
	    
	    for cls_ind, cls in enumerate(CLASSES[1:]):
	        cls_ind += 1
	        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
	        cls_scores = scores[:, cls_ind]
	        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
	        keep = nms(dets, NMS_THRESH)
	        dets = dets[keep, :]
	        vis_detections(axarr[i], cls, dets, thresh=CONF_THRESH)

	    imarr[i] = imarr[i][:, :, (2, 1, 0)]

    avg /= N
    print ('Detection took {:.3f}s\n').format(avg) 

    for i in range(N):
		axarr[i].imshow(imarr[i], aspect='equal')


def usage():
    print 'Usage:', sys.argv[0], 'path'
    sys.exit(0)


if __name__ == '__main__':
    if len(sys.argv) < 2: usage()

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    prototxt = os.path.join(MODELS_DIR, 'pascal_voc', 'VGG16', 'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(DATA_DIR, 'faster_rcnn_models', 'VGG16_faster_rcnn_final.caffemodel')
    #prototxt = os.path.join(MODELS_DIR, 'pascal_voc', 'ZF', 'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    #caffemodel = os.path.join(DATA_DIR, 'faster_rcnn_models', 'ZF_faster_rcnn_final.caffemodel')

    caffe.set_mode_gpu()
    caffe.set_device(0)

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}\n'.format(caffemodel)

    path = sys.argv[1]
    for (_, _, im_files) in os.walk(path):
    	print 'input images:', str(im_files) + '\n'
    	im_files = map(lambda im: path + '/' + im, im_files)

    detect(net, im_files)
    plt.show()
