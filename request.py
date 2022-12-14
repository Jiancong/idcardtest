import cv2
import numpy as np
import glob
import os
import requests
import json
from PIL import Image
import io
import base64
import warnings
#from flask import Flask, request, Response
#from flask import send_file
#from colors_utils import *
#from utils import *
import warnings
#from main_code import *
#from api_utils import *
from PIL import ImageFont, ImageDraw, Image

warnings.filterwarnings('ignore')

REMOTE_SERVER_URL="http://142.47.103.151:8081/"

def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

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

def create_folder(dirr):
    try:
        os.mkdir(dirr)
    except:
        pass
        
def show(label, img):
    if img is not None:
        cv2.imshow(label, img)
    else:
        pass        

X=0
Y=0
drag =False
global pts 
arg = 0        

def get_img_scanned(img, pts, img_type):
    print("get_img_scanned called.")
    _,buffer=cv2.imencode('.jpg', img)
    img_encoded=base64.b64encode(buffer)
    sttt=(str(img_encoded)[2:])[:-1]
    print("pts shape originally:" + str(pts.shape))
    pts = np.int32(pts).reshape(-1,).tolist()
    print("pts list length:" + str(len(pts)))
    pts = ','.join(map(str,pts))
    my_dict ={'pts':pts,'img':sttt,'type':img_type}
    resp = requests.post(url=REMOTE_SERVER_URL + '/final_img', data=my_dict)
    data = resp.json()
    final_img = toRGB(stringToImage(data['img']))
    
    return final_img

def get_img_pts(img, img_type):
    print("get_img_pts called.")
    _,buffer=cv2.imencode('.jpg',img)
    
    img_encoded=base64.b64encode(buffer)
    
    print("image encoded.")
    encoded_img=(str(img_encoded)[2:])[:-1]
    #print("encoded image shape: {}".format(encoded_img.shape))
    #print("encoded_img:{}".format(encoded_img))

    print("image type:{}".format(img_type))
    my_dict ={'type':img_type, 'img':encoded_img}
    
    resp = requests.post(url=REMOTE_SERVER_URL+'/pts', data=my_dict)
    print("get resp")
    data = resp.json()
    print("data:")
    if data is None:
        print("data is none. Error happened.")
        return
    pts= data['pts']
    return pts

def combine_images(img_1,img_2):
    print("combine_images called.")
    _,buffer=cv2.imencode('.jpg',img_1)
    img_encoded=base64.b64encode(buffer)
    img_1_st=(str(img_encoded)[2:])[:-1]
    _,buffer=cv2.imencode('.jpg',img_2)
    img_encoded=base64.b64encode(buffer)
    img_2_st=(str(img_encoded)[2:])[:-1]
    my_dict = {'img_front':img_1_st,'img_back':img_2_st}
    resp = requests.post(url=REMOTE_SERVER_URL + '/final_result', data=my_dict)
    data = resp.json()
    img =  toRGB(stringToImage(data['img']))
    img_water_marked = toRGB(stringToImage(data['img_water_marked']))
    return img,img_water_marked

def select_point(event, x, y, flags, param):
    print("select_point called.")
    global X
    global Y
    global drag
    global arg
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts)>=4:
            drag=True
            X=x
            Y=y
            arg = get_arg(pts,X,Y)
        else:
            pts.append([x,y])
        
    if event == cv2.EVENT_MOUSEMOVE:
        if drag ==True:
            if arg is not None:
                pts[arg] = [x,y]
    if event == cv2.EVENT_LBUTTONUP:
        drag=False

if __name__ == "__main__":

    print("start running...")

    create_folder("outputs")  
    DIR = 'trainings_samples'#images directory that contains jpg images

    X=0
    Y=0
    drag =False
    global pts 
    arg = 0

    # linux
    #imgs_dir = glob.glob(f'{DIR}/*.jpg')
    # windows 
    imgs_dir = glob.glob(f'{DIR}\*.jpg')

    for img_filename in imgs_dir:
    
        print("img_filename:" + img_filename) 
    
        try:
            if '_front' in img_filename:
                #print('file type:' + type(img_filename))
                imgs = []
                
                img_front_file = cv2.imread(str(img_filename))
                print(type(img_front_file))
                print(img_front_file.shape)
                print(img_front_file.dtype)
                print('img_front_filename:' + str(img_filename))
                img_front_resized = cv2.resize(img_front_file,(860,540))
                
                print ("----------------------")
                img_back_filename = os.path.abspath(str(img_filename)).split('_front')[0] + "_back.jpg"
                img_back_file = cv2.imread(str(img_back_filename))
                print('img_back_filename:' + str(img_back_filename))
                print(type(img_back_file))
                print(img_back_file.shape)
                print(img_back_file.dtype)
                #print('img_back_file:' + str(img_back_filename))
                
                img_back_resized = cv2.resize(img_back_file,(860,540))

        
                cv2.namedWindow("img")
                cv2.setMouseCallback("img", select_point)
                
                ## ?????????front side
                pts = get_img_pts(
                    img_front_resized,
                    'front')
                print("pts:{}".format(pts))
                my_img = img_front_resized.copy()
                
                
                # ??????looping 
                while True:
                
                    image=my_img.copy()
                    
                    if len(pts)>=4:
                        pts = np.int32(pts).reshape(-1,2)
                        cv2.polylines(image,[pts],True,(255,0,0),2)
        
                    #print("Show processed image.")
                    show('img',image)
                    
                    # ??????1??????
                    key=cv2.waitKey(1)
                    if key & 0xFF == ord('f'):
                    
                        if len(pts) < 4:
                            print("ERROR: the points in front side is invalid. Drop the request.")
                            continue
                        else:
                            print("Front side scanned...")
                    
                        output_front = get_img_scanned(
                                        img_front_resized,
                                        pts,
                                        'front')
                        print("Front side finished....") 
                        
                        if len(img_back_resized) > 0 :
                            print("Back side scanned....")
                            my_img = img_back_resized.copy()
                            pts = []
                            
                            pts = get_img_pts(
                                img_back_resized,
                                'back'
                                )
                        else:
                            print("back image is Null. Break.")
                            break
                            
                    if key & 0xFF ==ord('b'):
                    
                        if len(pts) < 4:
                            print("ERROR: the points in back side is invalid. Drop the request.")
                            continue 
                        else:
                            print("Back side scanned....")
                    
                        if len(img_back_resized)>0:
                        
                            
                            print("pts shape:" + str(pts.shape))
                            output_back = get_img_scanned(
                                            img_back_resized,
                                            pts,
                                            'back')
                            
                            pts = []
                        # ?????????back?????????
                        break
                        
                    if key & 0xFF ==ord('c'):
                        print("Clear the points.")
                        pts = []
                        
                print("Destroy the windows.")
                cv2.destroyAllWindows()
                
                if len(output_front)>0 and len(output_back)>0:
                    print("Both images are scanned.")
                else:
                    print("Something wrong happened.")
                    continue
                img_final,    img_final_water_marked = combine_images(
                                                        output_front,
                                                        output_back)
                #show('img_final',cv2.resize(img_final,(700,800)))
                #show('img_final_water_marked',cv2.resize(img_final_water_marked,(700,800)))
                cv2.imshow('front',output_front)
                cv2.imshow('back',output_back)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                name= os.path.join('outputs',os.path.basename(img_filename))
                #cv2.imwrite(name,img_final)
                #cv2.imwrite(name.split('.jpg')[0]+'_water.jpg',img_final_water_marked)
                cv2.imwrite(name,output_front)
                cv2.imwrite(name.split('.jpg')[0]+'back.jpg',output_back)
        except Exception as e:
            print(str(e))
