import cv2
import numpy as np

import time
from colors_utils import *
import sys
sys.path.append("mrcnn")
from m_rcnn import *
from visualize import *
import tensorflow as tf
import keras
from keras import backend as K
import os
from PIL import ImageFont, ImageDraw, Image

sift = cv2.SIFT_create()  # make sift feature recognition
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")


def rotate(img,angle):

    w, h, cX, cY = get_boarders(img)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))
    nW,nH = get_new_boarders(M,h,w)
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return img
def get_boarders(img):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    return w,h,cX,cY
def get_new_boarders(M,h,w):
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    return nW,nH
def water_marked(img,text):
    h,w,c = img.shape
    img_text = np.uint8(np.zeros_like(img))
    fontpath = "./simsun.ttc" # <== 这里是宋体路径 
    font = ImageFont.truetype(fontpath, 128)
    img_pil = Image.fromarray(img_text)
    draw = ImageDraw.Draw(img_pil)
    draw.text((0, h//2),text, font = font, fill = (255, 0, 0, 0))
    img_text = np.array(img_pil)

    img_text = rotate(img_text,-45)
    img_water_marked = cv2.addWeighted(img,0.6,img_text,0.6,0)
    return brighten(img_water_marked,40)

def concate(imgs):
    im_v = cv2.vconcat(imgs)
    return im_v

def create_folder(dirr):
    try:
        os.mkdir(dirr)
    except:
        pass
def get_text_boxes(img,target):
    height, width, channels = img.shape
    classes = ['char']
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in
                     net.getUnconnectedOutLayers()]  # get the output layers which are 82,94,106 indexes
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    # Loading image
    font = cv2.FONT_HERSHEY_PLAIN
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True,
                                 crop=False)  # src,scale factor(standard deviation), size of image wanted,means of R,G,B, swapRB=True,no croping
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
    for i in range(len(boxes)):
        if i in indexes:
            img_zeros = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            #cv2.rectangle(target, (x, y), (x + w, y + h), color, 2)
            img_zeros[y:y+h,x:x+w]  = np.ones_like(img_zeros[y:y+h,x:x+w])
            img_src = sharpen(simplest_cb(BrightnessContrast(simplest_cb(img,5), 186, 125),5))
            center_face = (int((x + x + w) / 2), int((y + y + h) / 2))
            target = cv2.seamlessClone(target, img_src, img_zeros, center_face, cv2.NORMAL_CLONE)
    return target
def get_face_mask(img,target):
    try:
        img_zeros= np.zeros_like(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
        sess = K.get_session()
        test_model, inference_config = load_inference_model(1, "mask_rcnn_object_0005.h5")  # for api usage

        h,w,c = img.shape
        r = test_model.detect(np.array(img).reshape(-1, h, w, c))[0]
        colors = random_colors(80)
        object_count = len(r["class_ids"])
        for i in range(object_count):
            mask = r["masks"][:, :, i]
            contours = get_mask_contours(mask)
            for cnt in contours:
                cv2.fillPoly(img_zeros, [cnt], 255)

        mask_img_not = cv2.bitwise_not(img_zeros)
        target_no_face = cv2.bitwise_and(target, target, mask=mask_img_not)
        res = cv2.bitwise_and(img, img, mask=img_zeros)
        res = brighten(res,40)
        target = cv2.add(target_no_face, res)
        K.clear_session()
    except:
        pass
    return target,img_zeros



def get_arg(pts, X, Y):
    min_dist = 0
    arg = 0
    for i, data in enumerate(pts):
        x, y = data
        dist = ((X - x) ** 2 + (Y - y) ** 2) ** 0.5
        if dist < min_dist or i == 0:
            min_dist = dist
            arg = i
    if min_dist < 100:
        return arg


def get_pts(query, img):
    kp_query, desc_query = sift.detectAndCompute(query, None)  # None stands for mask
  
    kp_train, desc_train = sift.detectAndCompute(img, None)
    good_p = []
    matches = flann.knnMatch(desc_query, desc_train,
                             k=2)  # try to find the matches between the decriptors in he first img  and the second one
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_p.append(m)

    if (len(good_p) > 15):
        q_p = np.float32([kp_query[m.queryIdx].pt for m in good_p]).reshape(-1, 1, 2)
        train_p = np.float32([kp_train[m.trainIdx].pt for m in good_p]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(q_p, train_p, cv2.RANSAC, 5.0)
        h, w, c = query.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)  # applies homography and returns a vector containing the point
        return dst
    else:
        return None


def get_img_alligned(img, dst):
    q_p = np.float32(dst).reshape(-1, 2)
    t_p = np.float32([[0, 0], [0, 500], [700, 500], [700, 0]])
    matrix = cv2.getPerspectiveTransform(q_p, t_p)
    img_aligned = cv2.warpPerspective(img, matrix, (700, 500))
    return img_aligned


def get_scanned(query, img):
    dst = get_pts(query, img)
    img_aligned = get_img_alligned(img, dst)

    return img_aligned


def show(label, img):
    if img is not None:
        cv2.imshow(label, img)
    else:
        pass


def get_text_mask(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
    return mask

def remove_back_ground(img):
    lower = np.array([245, 245, 245])
    upper = np.array([255, 255, 255])

    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # invert morp image
    mask = 255 - morph

    # apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)
    return mask
