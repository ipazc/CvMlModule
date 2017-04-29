#!/usr/bin/env python
# -*- coding: utf-8 -*-

import caffe
import cv2
import numpy

from main.model.algorithm.detection.face.mtcnn.mtcnn_image_processor import MTCNNImageProcessor
from main.model.stdfile_redirector import stdfile_redirector

__author__ = "Ivan de Paz Centeno"


class MTCNNFaceDetector(object):
    """
    Performs a detection of faces in an image, based on a CNN in Caffe.
    """

    def __init__(self,
                 det1_model=("main/data/caffe/mtcnn/det1.caffemodel", "main/data/caffe/mtcnn/det1.prototxt"),
                 det2_model=("main/data/caffe/mtcnn/det2.caffemodel", "main/data/caffe/mtcnn/det2.prototxt"),
                 det3_model=("main/data/caffe/mtcnn/det3.caffemodel", "main/data/caffe/mtcnn/det3.prototxt"),
                 use_gpu=-1):
        """
        Initializes the detector with the specified caffe models.
        :param det1_model: pair of model-prototxt regarding the first detector.
        :param det2_model: pair of model-prototxt regarding the second detector.
        :param det3_model: pair of model-prototxt regarding the third detector.
        """

        if use_gpu > -1:
            caffe.set_device(use_gpu)
            caffe.set_mode_gpu()

        with stdfile_redirector():
            self.p_net = caffe.Net(det1_model[1], det1_model[0], caffe.TEST)
            self.r_net = caffe.Net(det2_model[1], det2_model[0], caffe.TEST)
            self.o_net = caffe.Net(det3_model[1], det3_model[0], caffe.TEST)

    def detect_faces(self, image, minsize=20, threshold=None, fastresize=False, factor=0.709):
        """
        Performs a detection of faces in the given image.
        :param image: image to analyze.
        :param minsize:
        :param threshold:
        :param fastresize:
        :param factor:
        :return:
        """
        if threshold is None:
            threshold = [0.6, 0.7, 0.7]

        translated_image = image.copy()
        tmp = translated_image[:, :, 2].copy()
        translated_image[:, :, 2] = translated_image[:, :, 0]
        translated_image[:, :, 0] = tmp

        total_boxes = numpy.zeros((0, 9), numpy.float)

        mtcnn_image_processor = MTCNNImageProcessor(translated_image, minsize, threshold, factor)

        scaled_images = mtcnn_image_processor.get_scales(fastresize)

        # first stage
        for scale_pack in scaled_images:
            [scaled_image, scale, scaled_width, scaled_height] = scale_pack

            boxes = self._perform_first_stage(scaled_image, scale, scaled_width, scaled_height, threshold)

            if boxes.shape[0] != 0:
                total_boxes = numpy.concatenate((total_boxes, boxes), axis=0)


        # Now we have all the boxes detected by the first stage. It is time to pass them to the second stage:
        total_boxes = self._perform_second_stage(translated_image, total_boxes, threshold)

        total_boxes, points = self._perform_third_stage(translated_image, total_boxes, threshold)

        return total_boxes, points

    @staticmethod
    def _generate_bounding_box(map, reg, scale, threshold):
        """
        Generates a bounding box for the specified map, reg scale and threshold.
        :param map:
        :param reg:
        :param scale:
        :param threshold:
        :return:
        """
        stride = 2
        cellsize = 12
        map = map.T
        dx1 = reg[0, :, :].T
        dy1 = reg[1, :, :].T
        dx2 = reg[2, :, :].T
        dy2 = reg[3, :, :].T
        (x, y) = numpy.where(map >= threshold)

        yy = y
        xx = x

        score = map[x, y]
        reg = numpy.array([dx1[x, y], dy1[x, y], dx2[x, y], dy2[x, y]])

        boundingbox = numpy.array([yy, xx]).T

        bb1 = numpy.fix((stride * boundingbox + 1) / scale).T
        bb2 = numpy.fix((stride * boundingbox + cellsize - 1 + 1) / scale).T
        score = numpy.array([score])

        boundingbox_out = numpy.concatenate((bb1, bb2, score, reg), axis=0)

        return boundingbox_out.T

    @staticmethod
    def _nms(boxes, threshold, type):
        """
        nms
        :boxes: [:,0:5]
        :threshold: 0.5 like
        :type: 'Min' or others
        :returns: TODO
        """
        if boxes.shape[0] == 0:
            return numpy.array([])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s = boxes[:, 4]

        area = numpy.multiply(x2 - x1 + 1, y2 - y1 + 1)
        I = numpy.array(s.argsort())  # read s using I

        pick = []

        while len(I) > 0:
            xx1 = numpy.maximum(x1[I[-1]], x1[I[0:-1]])
            yy1 = numpy.maximum(y1[I[-1]], y1[I[0:-1]])
            xx2 = numpy.minimum(x2[I[-1]], x2[I[0:-1]])
            yy2 = numpy.minimum(y2[I[-1]], y2[I[0:-1]])
            w = numpy.maximum(0.0, xx2 - xx1 + 1)
            h = numpy.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h

            if type == 'Min':
                o = inter / numpy.minimum(area[I[-1]], area[I[0:-1]])
            else:
                o = inter / (area[I[-1]] + area[I[0:-1]] - inter)

            pick.append(I[-1])

            I = I[numpy.where(o <= threshold)[0]]

        return pick

    @staticmethod
    def _convert_to_square(bounding_box):
        """
        Converts the specified bounding box into a squared box.
        :return: bounding box converted.
        """
        # convert bboxA to square
        w = bounding_box[:, 2] - bounding_box[:, 0]
        h = bounding_box[:, 3] - bounding_box[:, 1]
        l = numpy.maximum(w,h).T

        bounding_box[:, 0] = bounding_box[:, 0] + w * 0.5 - l * 0.5
        bounding_box[:, 1] = bounding_box[:, 1] + h * 0.5 - l * 0.5
        bounding_box[:, 2:4] = bounding_box[:, 0:2] + numpy.repeat([l], 2, axis = 0).T

        return bounding_box

    @staticmethod
    def _pad(bounding_boxes, w, h):

        boxes = bounding_boxes.copy()  # shit, value parameter!!!

        tmph = boxes[:, 3] - boxes[:, 1] + 1
        tmpw = boxes[:, 2] - boxes[:, 0] + 1
        num_boxes = boxes.shape[0]

        dx = numpy.ones(num_boxes)
        dy = numpy.ones(num_boxes)
        edx = tmpw
        edy = tmph

        x = boxes[:, 0:1][:, 0]
        y = boxes[:, 1:2][:, 0]
        ex = boxes[:, 2:3][:, 0]
        ey = boxes[:, 3:4][:, 0]

        tmp = numpy.where(ex > w)[0]
        if tmp.shape[0] != 0:
            edx[tmp] = -ex[tmp] + w - 1 + tmpw[tmp]
            ex[tmp] = w - 1

        tmp = numpy.where(ey > h)[0]
        if tmp.shape[0] != 0:
            edy[tmp] = -ey[tmp] + h - 1 + tmph[tmp]
            ey[tmp] = h - 1

        tmp = numpy.where(x < 1)[0]
        if tmp.shape[0] != 0:
            dx[tmp] = 2 - x[tmp]
            x[tmp] = numpy.ones_like(x[tmp])

        tmp = numpy.where(y < 1)[0]
        if tmp.shape[0] != 0:
            dy[tmp] = 2 - y[tmp]
            y[tmp] = numpy.ones_like(y[tmp])

        # for python index from 0, while matlab from 1
        dy = numpy.maximum(0, dy - 1)
        dx = numpy.maximum(0, dx - 1)
        y = numpy.maximum(0, y - 1)
        x = numpy.maximum(0, x - 1)
        edy = numpy.maximum(0, edy - 1)
        edx = numpy.maximum(0, edx - 1)
        ey = numpy.maximum(0, ey - 1)
        ex = numpy.maximum(0, ex - 1)

        return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]

    @staticmethod
    def _normalize_bounding_boxes(bounding_boxes, width, height):
        """
        Normalizes the bounding boxes retrieved after the first stage.
        :param bounding_boxes: bboxes to normalize
        :return: [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        """
        total_boxes = numpy.array([])
        num_boxes = bounding_boxes.shape[0]
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if num_boxes > 0:
            # nms
            pick = MTCNNFaceDetector._nms(bounding_boxes, 0.7, 'Union')
            total_boxes = bounding_boxes[pick, :]

            # revise and convert to square
            regh = total_boxes[:, 3] - total_boxes[:, 1]
            regw = total_boxes[:, 2] - total_boxes[:, 0]
            t1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
            t2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
            t3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
            t4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
            t5 = total_boxes[:, 4]

            total_boxes = numpy.array([t1, t2, t3, t4, t5]).T

            total_boxes = MTCNNFaceDetector._convert_to_square(total_boxes)  # convert box to square

            total_boxes[:, 0:4] = numpy.fix(total_boxes[:, 0:4])

            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = MTCNNFaceDetector._pad(total_boxes, width, height)

        return total_boxes, [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]

    @staticmethod
    def _bbreg(boundingbox, reg):
        reg = reg.T

        # calibrate bouding boxes
        w = boundingbox[:,2] - boundingbox[:,0] + 1
        h = boundingbox[:,3] - boundingbox[:,1] + 1

        bb0 = boundingbox[:,0] + reg[:,0]*w
        bb1 = boundingbox[:,1] + reg[:,1]*h
        bb2 = boundingbox[:,2] + reg[:,2]*w
        bb3 = boundingbox[:,3] + reg[:,3]*h

        boundingbox[:,0:4] = numpy.array([bb0, bb1, bb2, bb3]).T

        return boundingbox

    def _perform_first_stage(self, scaled_image, scale, scaled_width, scaled_height, threshold):
        """
        Performs the first stage of the detection.
        :param scaled_image: already scaled image.
        :param scale: scale number applied to the image.
        :param scaled_width: width of the scaled image.
        :param scaled_height: height of the scaled image.
        :return: total boxes detected on the scaled image.
        """
        with stdfile_redirector():
            self.p_net.blobs['data'].reshape(1, 3, scaled_width, scaled_height)
            self.p_net.blobs['data'].data[...] = scaled_image
            out = self.p_net.forward()

        boxes = self._generate_bounding_box(out['prob1'][0, 1, :, :], out['conv4-2'][0], scale, threshold[0])

        if boxes.shape[0] != 0:
            pick = self._nms(boxes, 0.5, 'Union')

            if len(pick) > 0:
                boxes = boxes[pick, :]

        return boxes

    def _perform_second_stage(self, image, total_boxes, threshold):
        """
        Performs the second stage of the detection
        :param total_boxes: boxes from the first stage concatenated.
        :return:
        """

        (width, height) = (image.shape[1], image.shape[0])

        total_boxes, [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self._normalize_bounding_boxes(total_boxes, width,
                                                                                                   height)

        num_boxes = total_boxes.shape[0]
        if num_boxes > 0:

            # construct input for RNet
            temp_image = numpy.zeros((num_boxes, 24, 24, 3))  # (24, 24, 3, num_boxes)
            for k in range(num_boxes):
                tmp = numpy.zeros((int(tmph[k]), int(tmpw[k]), 3))

                tmp[int(dy[k]):int(edy[k]) + 1, int(dx[k]):int(edx[k]) + 1] = image[int(y[k]):int(ey[k]) + 1,
                                                                              int(x[k]):int(ex[k]) + 1]

                temp_image[k, :, :, :] = cv2.resize(tmp, (24, 24))

            temp_image = (temp_image - 127.5) * 0.0078125

            # RNet

            temp_image = numpy.swapaxes(temp_image, 1, 3)

            with stdfile_redirector():
                self.r_net.blobs['data'].reshape(num_boxes, 3, 24, 24)
                self.r_net.blobs['data'].data[...] = temp_image
                out = self.r_net.forward()

            score = out['prob1'][:, 1]

            pass_t = numpy.where(score > threshold[1])[0]

            score = numpy.array([score[pass_t]]).T
            total_boxes = numpy.concatenate((total_boxes[pass_t, 0:4], score), axis=1)

            mv = out['conv5-2'][pass_t, :].T

            if total_boxes.shape[0] > 0:
                pick = self._nms(total_boxes, 0.7, 'Union')

                if len(pick) > 0:
                    total_boxes = total_boxes[pick, :]
                    total_boxes = self._bbreg(total_boxes, mv[:, pick])
                    total_boxes = self._convert_to_square(total_boxes)

        return total_boxes

    def _perform_third_stage(self, image, total_boxes, threshold):
        """
        Performs the third stage of the face detection.
        """
        (width, height) = (image.shape[1], image.shape[0])

        num_boxes = total_boxes.shape[0]
        points = []

        if num_boxes > 0:
            total_boxes = numpy.fix(total_boxes)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self._pad(total_boxes, width, height)

            temp_image = numpy.zeros((num_boxes, 48, 48, 3))
            for k in range(num_boxes):
                tmp = numpy.zeros((int(tmph[k]), int(tmpw[k]), 3))
                tmp[int(dy[k]):int(edy[k]) + 1, int(dx[k]):int(edx[k]) + 1] = image[int(y[k]):int(ey[k]) + 1, int(x[k]):int(ex[k]) + 1]
                temp_image[k, :, :, :] = cv2.resize(tmp, (48, 48))

            temp_image = (temp_image - 127.5) * 0.0078125  # [0,255] -> [-1,1]

            # ONet
            temp_image = numpy.swapaxes(temp_image, 1, 3)

            with stdfile_redirector():
                self.o_net.blobs['data'].reshape(num_boxes, 3, 48, 48)
                self.o_net.blobs['data'].data[...] = temp_image

                out = self.o_net.forward()

            score = out['prob1'][:, 1]
           
            points = out['conv6-3']
            pass_t = numpy.where(score > threshold[2])[0]
            points = points[pass_t, :]
            score = numpy.array([score[pass_t]]).T
            total_boxes = numpy.concatenate((total_boxes[pass_t, 0:4], score), axis=1)

            mv = out['conv6-2'][pass_t, :].T
            w = total_boxes[:, 3] - total_boxes[:, 1] + 1
            h = total_boxes[:, 2] - total_boxes[:, 0] + 1

            points[:, 0:5] = numpy.tile(w, (5, 1)).T * points[:, 0:5] + numpy.tile(total_boxes[:, 0], (5, 1)).T - 1
            points[:, 5:10] = numpy.tile(h, (5, 1)).T * points[:, 5:10] + numpy.tile(total_boxes[:, 1], (5, 1)).T - 1

            if total_boxes.shape[0] > 0:
                total_boxes = self._bbreg(total_boxes, mv[:, :])
                
                pick = self._nms(total_boxes, 0.7, 'Min')

                if len(pick) > 0:
                    total_boxes = total_boxes[pick, :]
                    points = points[pick, :]

        return total_boxes, points

