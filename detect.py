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
    # Draw detected bounding boxes
    inds = np.where(dets[:, -1] >= thresh)[0]
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
                          edgecolor='#00ff00', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='black', alpha=0.5),
                fontsize=14, color='yellow')

    plt.axis('off')
    plt.tight_layout()


def draw_img_with_dets(img, boxes, scores):
    CONF_THRESH = 0.9
    NMS_THRESH = 0.3
    
    fig, ax = plt.subplots(figsize=(16,12))
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(ax, cls, dets, thresh=CONF_THRESH)
    img = img[:, :, (2, 1, 0)] 
    ax.imshow(img, aspect='equal')
    #fig.savefig('img{:d}.png'.format(i))


def total_area(boxes):
    x1 = boxes[:, 0::4].flatten()
    y1 = boxes[:, 1::4].flatten()
    x2 = boxes[:, 2::4].flatten()
    y2 = boxes[:, 3::4].flatten()
    assert x1.size == boxes.size/4
    area = 0
    for i in range(boxes.size/4):
        w = x2[i]-x1[i]
        h = y2[i]-y1[i]
        area += w*h
    return int(area)


def detect(net, im_files):
    # Detect all object classes and regress object bounds
    N = len(im_files)
    
    imgarr = [None] * N
    scores = [None] * N
    boxes  = [None] * N
    avg = 0
    for i, im_file in enumerate(im_files):
        imgarr[i] = (cv2.imread(im_file))
        start = time.time()
        scores_, boxes_ = im_detect(net, imgarr[i])
        end = time.time()
        diff = end - start
        avg += diff
        scores[i] = np.copy(scores_)
        boxes[i] = np.copy(boxes_)
        area = total_area(boxes_)
        print ('{:s} took {:.3f}s for {:d} detections in {:d} pixels').format(im_file, diff, boxes[i].shape[0], area)
    avg /= N
    print ('Detection took {:.3f}s\n').format(avg)

    for i, im_file in enumerate(im_files):
    	print 'drawing img', str(i) + ':', im_file + '...'
        draw_img_with_dets(imgarr[i], boxes[i], scores[i])


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
        #im_files = [im_files[0]] * 10
        #im_files = im_files[0:5]
        print 'Input images:', str(im_files) + '\n'
        im_files = map(lambda im: path + '/' + im, im_files)

    # Caching data to accelerate forward pass for real images
    print 'Caching data...\n'
    dummy_image = np.zeros((600,1000,3), np.uint8)
    im_detect(net, dummy_image)

    detect(net, im_files)

    # Show images in separate windows
    #plt.show()
