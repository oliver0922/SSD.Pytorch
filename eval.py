"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# argumnet parser 선언.
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default= '/content/drive/MyDrive/SSD.Pytorch/weights/VOC.pth', type=str,
                    help='Trained state_dict file path to open')
# trained된 모델을 가져와서 쓸 수 있게함. 경로를 입력.
parser.add_argument('--input',default=512, type=int, choices=[300, 512], help='ssd input size, currently support ssd300 and ssd512')
# SSD version 300, 512중 선택.
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
# 검증에 대한 결과 저장하는 경로.
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
# Confidence threshold hyperparameter를 정할 수 있다.
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
# 제한할 예측 갯수 정하기.
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
# GPU를 사용할지에 대한 boolean값 입력.
parser.add_argument('--voc_root', default='/content/drive/MyDrive/SSD.Pytorch/kitti',
                    help='Location of VOC root directory')
# dataset인 voc의 경로 입력.
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
# evaluation의 결과로 저장했던 것들을 지워버림.

args = parser.parse_args() # args로 인자값들 저장.

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
    # 결과값을 저장할 save_folder를 입력해주지않으면 새로 만든다.

if torch.cuda.is_available(): # GPU를 사용할 수 있을때.
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # GPU 사용시 사용.
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
        # GPU 사용가능하지만 사용하지 못할 때 문구 출력.
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    # GPU사용불가할 때 CPU 사용.

annopath = os.path.join(args.voc_root, 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'ImageSets',
                          'Main', '{:s}.txt')
# VOC dataset에 들어있는 xml, jpg, txt 파일들 불러오기.

devkit_path = args.voc_root # devkit  경로 입력.
dataset_mean = (104, 117, 123) 
# dataset 평균.
set_type = 'test' # test용.


class Timer(object): # 시간 재기위한 함수.
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        # 각 변수 초기화.
        
    def tic(self):
        self.start_time = time.time()
        # 시작시간 설정.
        
    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        # 현재시간에서 시작시간 뺌으로써 걸린 시간 계산.
        self.total_time += self.diff
        # 총 걸리는 시간 계산.
        self.calls += 1
        self.average_time = self.total_time / self.calls
        # 하나당 걸리는 시간 계산.
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename) # xml 파일 피싱하기 위해 parsing.
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)
        # obj 즉 object들에서 내부에 들어가 있는 요소들 추출하고(name, pose, bbox 등) 따로 list에 저장.

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
        # output 디렉토리가 없으면 만들어준다. 
        # 경로가 존재하면 만들지 않고 반환.
    return filedir


def get_voc_results_file_template(image_set, cls):
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    # results 경로안에 경로를 설정. 즉 경로 병합.
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        # 위에서 만든 결과값을 저장할 수 있는 경로를 가져온다.
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                # 내부 경로안에 있는 파일을 쓴다.
                # 바운딩 박스에 대한 정보 입력.
                if dets == []:
                    continue
                    # dets가 아무것도 없는 배열이라면 계속 실행.
                for k in range(dets.shape[0]): # dets의 갯수만큼 for 반복문 실행.
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))
                    # dets안의 정보를 가지고 k번째의 정보들을 나열하면서 write한다.
                    # box들에 대한 정보를 write.


def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    # devkit경로의 annotations_cache를 연다.
    aps = [] # list 생성.
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07 # boolean 값 그대로 입력.
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        # output 디렉토리 없으면 만들기.
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        # label에 대한 txt 파일을 set_type과 cls의 이름으로 생성.
        rec, prec, ap = voc_eval(
           filename, annopath, imgsetpath.format(set_type), cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        # SSD 모델에서 VOC dataset에 대한 검증 결과 뽑아내기.
        aps += [ap] # average precision 구하기.
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
            # output 디렉토리를 넣어서 각 검증결과 정보들을 pickle을 통해서 입력한다.(쓴다)
    print('Mean AP = {:.4f}'.format(np.mean(aps))) # mAP 출력.
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap)) # 각각의 ap 출력.
    print('{:.3f}'.format(np.mean(aps))) # mAP 출력.
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True): # 검증결과 정보들이 인자로 들어감.
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric: # 11point metric 이라고함.
        ap = 0.
        for t in np.arange(0., 1.1, 0.1): # 0, 0.1, 0.2 ... 1까지.
            if np.sum(rec >= t) == 0: # rec가 모두 t보다 작으면 p = 0 대입.
                p = 0
            else:
                p = np.max(prec[rec >= t]) # rec가 모두 t보다 작은 것은 아닐때.
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        # AP 계산을 정확히 한다.
        # rec와 prec의 배열을 내부적으로 모두 합친다.

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            # 정밀도를 전체적으로 계산한다.

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # np.where를 통해 위에서 계산한 부분을 가지고 확인.
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        # ap 계산.
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    # VOC dataset으로 성능평가를 위해 각 경로와 class 이름, 저장할 경로, threshold 등 입력.
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # cache 디렉토리가 없다면 만들어주고 cache 디렉토리안에 annots.pk1를 가져온다.
    
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
        # image set 파일내부 읽어오기.
    imagenames = [x.strip() for x in lines]
    # line by line으로 가져온다.
    if not os.path.isfile(cachefile):
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
                # 100번 마다 하나씩 annotation 이름 출력.
        
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
            # pickle로 recs 쓰고 저장.
    else:
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)
            # cache file이 있다면 pickle로 load.

    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        # recs에서의 imagename으로 for문돌림.
        bbox = np.array([x['bbox'] for x in R])
        # bounding box를 numpy의 array 형태로 가져온다.
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        # x dictionary의 difficult을 bool형태로 가져온다.
        det = [False] * len(R) # obj의 길이 만큼 false로 채운다.
        npos = npos + sum(~difficult) # difficult 배열의 boolean을 모두 반대로 바꾼다음 
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
        # class_recs 베얄 bbox, difficult, det를 dictionary 형태로 저장.

    detfile = detpath.format(classname) 
    with open(detfile, 'r') as f:
        lines = f.readlines()
        # det 파일 읽기.
    if any(lines) == 1:
        splitlines = [x.strip().split(' ') for x in lines]
        # det 안의 line들을 line끼리 분할 후 splitlines에 넣는다.
        image_ids = [x[0] for x in splitlines]
        # splitlines의 각 요소의 첫번째를 가져와 id로 저장.
        confidence = np.array([float(x[1]) for x in splitlines])
        # line의 두번째 요소를 가져와서 confidence score로 지정.
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
        # line의 두번째 요소이후는 모두 bounding box에 대한 정보이므로 가져와서 BB에 넣는다.

        sorted_ind = np.argsort(-confidence)
        # confidence score를 작은 순서대로 index를 반환시킨다.
        sorted_scores = np.sort(-confidence)
        # confidence score를 작은 순서대로 score를 반환시킨다.
        BB = BB[sorted_ind, :] # bounding box를 다시 sorted_ind 순으로 정렬.
        image_ids = [image_ids[x] for x in sorted_ind]
        # imageid도 마찬가지로 sorted_ind 순으로 정렬한다.

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf # 마이너스 무한대.
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0: # IOU 계산.
                ixmin = np.maximum(BBGT[:, 0], bb[0]) # 둘 중 max인것 추출
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2]) # 둘 중 min인것 추출
                iymax = np.minimum(BBGT[:, 3], bb[3])
                # x의 최대 최소, y의 최대 최소점을 구해서 사각형을 알아낼 수 있다.
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                # x축의 길이를 구해 w, y축의 길이를 구해 h를 구함.
                inters = iw * ih # 넓이 구하기.
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                # 두 bounding box의 합에서 교차지역을 한번 뺌으로써 두 bounding box의 합집합 넓이를 구함.
                overlaps = inters / uni
                # IOU 계산.
                ovmax = np.max(overlaps) # overlaps 중 가장 큰 것 추출. ovmax갱신.
                jmax = np.argmax(overlaps) # overlaps가 가장큰 index

            if ovmax > ovthresh: # threshold보다 클 때.
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # precision 꼐산.
        fp = np.cumsum(fp) # row와 column 구분 없이 모든 원소들의 합 반환.
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        # precision 구하기.
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.
        # lines가 없다면 구하지 않는다.
    return rec, prec, ap


def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    num_images = len(dataset) # image 개수.
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)] 
    # 이미지마다 좌표와 score를 넣어주기 위해 배열 선언.
    
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    # output 디렉토리 선언.
    det_file = os.path.join(output_dir, 'detections.pkl')
    # output 디렉토리의 detections.pk1 파일 갖고오기.

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0)) # image의 shape 만큼 랜덤을 초기화.
        if args.cuda:
            x = x.cuda() # 계산을 위해 GPU에 올림.
        _t['im_detect'].tic() # 시간 측정 시작.
        detections = net(x).data # x를 network에 넣어 예측값 얻기.
        detect_time = _t['im_detect'].toc(average=False)
        # 시간 측정 종료.

        for j in range(1, detections.size(1)): # background class는 제외.
            dets = detections[0, j, :] # 예측값의 정보 가져오기.
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1  # background class 포함 -> +1.
    net = build_ssd('test', args.input, num_classes) # SSD 구조 초기화.
    net.load_state_dict(torch.load(args.trained_model)) # trained된 모델이 있다면 가져오기.
    net.eval() # 모델 평가.
    print('Finished loading model!')
    
    dataset = VOCDetection(args.voc_root, set_type,
                           BaseTransform(args.input, dataset_mean),
                           VOCAnnotationTransform()) # VOC dataset 및 정보 가져오기
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
        # GPU 사용시 network GPU에 올리기.
    
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, args.input,
             thresh=args.confidence_threshold) # 모델 평가로부터 계산되는 output의 저장위치, dataset, basenet 변경 여부, input, threshold 등
    # 검증평가에 필요한 parameter 입력.
    
