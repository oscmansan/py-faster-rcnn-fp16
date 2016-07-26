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


def detect(net, im_file):
    # Load image
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    N = 10
    avg = 0
    for i in range(N):
	    timer = Timer()
	    timer.tic()
	    scores, boxes = im_detect(net, im)
	    timer.toc()
	    avg += timer.total_time
	    print ('Iteration {:d} took {:.3f}s').format(i, timer.total_time)
    avg /= N
    print ('\nDetection took {:.3f}s for {:d} object proposals').format(avg, boxes.shape[0])

    print 'boxes.spape:', boxes.shape
    print 'scores.shape:', scores.shape

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    '''
    fig, ax = plt.subplots(figsize=(12, 12))
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1
        print 'Searching for class', CLASSES[cls_ind]
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(ax, cls, dets, thresh=CONF_THRESH)

    im = im[:, :, (2, 1, 0)]
    ax.imshow(im, aspect='equal')
    '''


def usage():
    print 'Usage:', sys.argv[0], 'image_file'
    sys.exit(0)


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    #prototxt = os.path.join(MODELS_DIR, 'pascal_voc', 'VGG16', 'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    #caffemodel = os.path.join(DATA_DIR, 'faster_rcnn_models', 'VGG16_faster_rcnn_final.caffemodel')
    prototxt = os.path.join(MODELS_DIR, 'pascal_voc', 'ZF', 'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(DATA_DIR, 'faster_rcnn_models', 'ZF_faster_rcnn_final.caffemodel')

    caffe.set_mode_gpu()
    caffe.set_device(0)

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}\n'.format(caffemodel)

    if len(sys.argv) < 2: usage()
    im_file = sys.argv[1]
    detect(net, im_file)
    plt.show()
