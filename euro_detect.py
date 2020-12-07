from  tkinter import filedialog
from tkinter import *
import cv2
import numpy as np
import os

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "C:/",title = "choose your file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
coinfile = root.filename
scriptfile = os.path.dirname( os.path.abspath( __file__ ) )
fileroute = os.path.relpath(coinfile, scriptfile)#이미지 파일 불러오기 

imagearray = []

img = cv2.imread(fileroute)
rows, cols = img.shape[:2]

height, width, channel = img.shape
white = [255,255,255]
black = [0,0,0]
for x in range(0,width):
    for y in range(0,height):
        channel_new = img[y,x]
        if all(channel_new == white):
            img[y,x] = black
#이미지 리사이징 및 전처리



mean = cv2.pyrMeanShiftFiltering(img, 20, 50)


gray = cv2.cvtColor(mean, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3,3), 0)

_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU )


dst = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
dst = (dst/(dst.max() - dst.min()) * 255).astype(np.uint8)
#cv2.imshow('dst', dst)

localMx = cv2.dilate(dst, np.ones((50,50), np.uint8))
lm= np.zeros((rows ,cols), np.uint8)
lm[(localMx==dst) & (dst!=0)] = 255
#cv2.imshow('localMx', lm)

seeds = np.where(lm==255)
seed = np.stack((seeds[1], seeds[0]), axis=-1)
fill_mask = np.zeros((rows+2, cols+2), np.uint8)
for x,y in seed:
    ret = cv2.floodFill(mean, fill_mask, (x,y), (255,255,255), \
        (10,10,10),(10,10,10))
#cv2.imshow('floodFill',mean)

gray = cv2.cvtColor(mean, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
dst = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
dst = ((dst/ (dst.max() - dst.min())) * 255).astype(np.uint8)
#cv2.imshow('dst2', dst)

ret, sure_fg = cv2.threshold(dst, 0.5*dst.max(),255,0)
#cv2.imshow('sure_fg', sure_fg)

_, bg_th = cv2.threshold(dst, 0.3*dst.max(), 255, cv2.THRESH_BINARY_INV)
bg_dst = cv2.distanceTransform(bg_th, cv2.DIST_L2, 5)
bg_dst = ((bg_dst / (bg_dst.max() - bg_dst.min()))*255).astype(np.uint8)
ret, sure_bg = cv2.threshold(bg_dst, 0.3*bg_dst.max(), 255,cv2.THRESH_BINARY)
#cv2.imshow('sure_bg', sure_bg)

ret, inv_sure_bg = cv2.threshold(sure_bg, 127, 255, cv2.THRESH_BINARY_INV)
unkown = cv2.subtract(inv_sure_bg, sure_fg)
#cv2.imshow('unkown', unkown)

_, markers = cv2.connectedComponents(sure_fg)

markers = markers+1
markers[unkown==255] = 0
colors = []
marker_show = np.zeros_like(img)
for mid in np.unique(markers): 
    color = [int(j) for j in np.random.randint(0,255,3)]
    colors.append((mid, color))
    marker_show[markers==mid] = color
    coords = np.where(markers==mid)
    x, y = coords[1][0], coords[0][0]
    cv2.putText(marker_show, str(mid), (x+20, y+20), cv2.FONT_HERSHEY_PLAIN, \
        2,(255,255,255))
#cv2.imshow('before', marker_show)

markers = cv2.watershed(img, markers)

for mid, color in colors: 
    marker_show[markers==mid] = color
    coords = np.where(markers==mid)
    if coords[0].size <= 0 :
        continue
    x,y = coords[1][0], coords[0][0]
    cv2.putText(marker_show, str(mid), (x+20, y+20), cv2.FONT_HERSHEY_PLAIN, \
        2,(255,255,255))

marker_show[markers==1] = (0,255,0)
#cv2.imshow('watershed marker', marker_show)

img[markers==-1] = (0,255,0)
#cv2.imshow('watershed',img)
#워터세드 적용

mask = np.zeros((rows, cols), np.uint8)
mask[markers!= 1] =255
nobg=cv2.bitwise_and(img,img, mask=mask)
coin_label = [l for l in np.unique(markers) if (l != 1 and l != -1)]
for i, label in enumerate(coin_label):
    mask[:,:] = 0
    mask[markers == label] =255
    coins = cv2.bitwise_and(img, img, mask=mask)
    _, contour,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contour[0])
    coin = coins[y:y+h, x:x+w]
    #cv2.imshow('coin%d'%(i+1), coin)
    cv2.imwrite('./coin%d.jpg'%(i+1), coin)
    imagearray.append('./coin%d.jpg'%(i+1))
#워터세드로 분할한 알고리즘 저장





categories =  ['1cent', '2cent','5cent','50cent','20cent','2euro','1euro' ]
dict_file = './coin/coin_dict200.npy'

svm_model_file = './coin/coin_svm200.xml'


for image in imagearray :
    imgs = [image]
    detector = cv2.xfeatures2d.SIFT_create()
    bowextractor = cv2.BOWImgDescriptorExtractor(detector, \
                                    cv2.BFMatcher(cv2.NORM_L2))
    bowextractor.setVocabulary(np.load(dict_file))
    svm  = cv2.ml.SVM_load(svm_model_file)

    for i, path in enumerate(imgs):
        bowimg = cv2.imread(path)
        gray = cv2.cvtColor(bowimg, cv2.COLOR_BGR2GRAY)
        hist = bowextractor.compute(gray, detector.detect(gray))
        ret, result = svm.predict(hist)
        name = categories[int(result[0][0])]
        txt, base = cv2.getTextSize(name, cv2.FONT_HERSHEY_PLAIN, 2, 3)
        x,y = 10, 50
        cv2.rectangle(bowimg, (x,y-base-txt[1]), (x+txt[0], y+txt[1]), (30,30,30), -1)
        cv2.putText(bowimg, name, (x,y), cv2.FONT_HERSHEY_PLAIN, \
                                    2, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow(path, bowimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
