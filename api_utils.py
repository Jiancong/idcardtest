
import requests
import json
import cv2
import numpy as np
from PIL import Image
import io
import base64
import warnings
from flask import Flask, request, Response
from flask import send_file


def encode(val):
    val=val.replace("+","BGABG")
    val=val.replace("=","AGAAGA")
    val=val.replace("/","OPAWQQQQ")
    return val
def decode(val):
    val=val.replace("BGABG","+")
    val=val.replace("AGAAGA","=")
    val=val.replace("OPAWQQQQ","/")
    return val
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))
# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)