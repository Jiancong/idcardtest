import cv2
import math
import numpy as np
import sys


def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    return img
def brighten(img,val):
    h, w, c = img.shape
    mtx = np.ones((h, w, c), dtype='uint8') * val
    brightened = cv2.add(img, mtx)
    return brightened

def clahe_equalization(img):
    GLARE_MIN = np.array([0, 0, 50], np.uint8)
    GLARE_MAX = np.array([0, 0, 225], np.uint8)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV
    frame_threshed = cv2.inRange(hsv_img, GLARE_MIN, GLARE_MAX)
    result = cv2.inpaint(img, frame_threshed, 0.1, cv2.INPAINT_TELEA)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Converting image from LAB Color model to BGR color space
    enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    enhanced_img = brighten(enhanced_img,40)
    # Stacking the original image with the enhanced image

    return enhanced_img

def BrightnessContrast(img, brightness=0, contrast=0):
    effect = controller(img, brightness, contrast)
    # The function imshow displays an image
    # in the specified window
    # cv2.imshow('Effect', effect)
    return effect


def controller(img, brightness=255,
               contrast=127):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))

    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightness != 0:

        if brightness > 0:

            shadow = brightness

            max = 255

        else:

            shadow = 0
            max = 255 + brightness

        al_pha = (max - shadow) / 255
        ga_mma = shadow

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)

    else:
        cal = img

    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)

    return cal
def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]

        #print("Lowval: ", low_val)
        #print ("Highval: ", high_val)

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)
