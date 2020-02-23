import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from palette4 import palette
from skimage import color as skcolor
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976,delta_e_cie2000,delta_e_cmc ,delta_e_cie1994
from numba import autojit, jit

lab_matrix= pd.Series([LabColor(lab_l=c[0], lab_a=c[1], lab_b=c[2]) for c in palette['lab'].values])

func=  delta_e_cmc

def get_s_channel(img):
	#input: BGR image
	img_hsv= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	h,s,v= cv2.split(img_hsv)
	return s

def get_mask(s,img):
	#input: S channel of HSV image
	s=cv2.medianBlur(s,5)
	ret,th1 = cv2.threshold(s,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	mask= cv2.bitwise_not(th1)
	roi= cv2.bitwise_and(s,s,mask=mask)
	ret,th2 = cv2.threshold(roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	gray=cv2.medianBlur(gray,5)
	ret,th3 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	th3= cv2.bitwise_not(th3)

	th= th1|th2|th3
	return th

def preprocess_image(img,th):
	#input: BGR image, binary image(0,255)
	#remove background : replace it with bgr(0,0,0)
	img=cv2.bitwise_and(img,img,mask=th)
	#convert image to rgb
	img_rgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	#covert image to lab
	img_lab= cv2.cvtColor(img_rgb,cv2.COLOR_RGB2LAB)

	return img,img_rgb,img_lab

def image_to_thumbs(img):
	"""Create thumbnail from image"""
	height, width, channels = img.shape
	thumbs = {"original": img}
	#sizes = [640, 320, 160]
	sizes = [200]
	for size in sizes:
		if (width >= size):
			r = (size + 0.0) / width
			max_size = (size, int(height * r))
			thumbs[str(size)] = cv2.resize(img, max_size, interpolation=cv2.INTER_AREA)

	return thumbs

def get_data_points(img):
	#input: LAB image
	indices = np.dstack(np.indices(img.shape[:2]))
	xycolors = np.concatenate((img, indices), axis=-1)     
	return np.reshape(xycolors, [-1,5])

def get_cluster_labels(feature_image,eps=7, min_samples=5):
	db = DBSCAN(eps=eps, min_samples=min_samples, metric = 'euclidean',algorithm ='auto',n_jobs=2)
	db.fit(feature_image)
	labels = db.labels_
	
	return labels

def get_label_masks(label_map, labels,img):
	label_masks=[]
	for i in set(labels):
		mask= np.zeros([img.shape[0],img.shape[1]])
		mask=label_map==i
		mask= np.uint8(mask)
		label_masks.append(mask)

	return label_masks

def get_roi_maps(label_masks,img_rgb):
	roi_maps=[]
	for mask in label_masks:
		roi= cv2.bitwise_and(img_rgb,img_rgb,mask=mask)
		roi_maps.append(roi)

	return roi_maps

def rgb2hex(r,g,b):
	hex = "#{:02x}{:02x}{:02x}".format(r,g,b)
	return hex

def hex2rgb(hexcode):
	rgb = tuple(map(ord,hexcode[1:].decode('hex')))
	return rgb


def get_dominant_color(roi,label_masks, th,index):
    dominance_level= np.sum(label_masks[index])/(np.sum(th))
    flat= np.reshape(roi,[-1,3])


    temp=  np.array_equal

    back= np.asarray([0,0,0], dtype= np.uint8)
    ''''
    pix=[]
    for arr in list(flat):
        if not np.array_equal(arr,[0,0,0]):
            pix.append(arr)
            '''
    #_pix= filter(lambda x: not temp(x,back), flat)
    #pix= [p for p in _pix]
    #pix=np.array([pix])
    #convert rgb2hsv

    df= pd.DataFrame(flat)
    _pix= df.loc[(df[0]!=0) & (df[1]!=0) & (df[2]!=0)]

    _pix= np.asarray(_pix)

    pix=np.expand_dims(_pix, axis=0)


    pix_hsv= cv2.cvtColor(pix, cv2.COLOR_RGB2HSV)
    h,s,v= cv2.split(pix_hsv)

    hist, bin_edges = np.histogram(h, bins = range(256))
    
    rgb_arr= cv2.cvtColor(np.array([[[np.argmax(hist), np.max(s), np.mean(v)]]]).astype(np.uint8), cv2.COLOR_HSV2RGB)[0][0]
    
    r,g,b=list(rgb_arr)

    _hex=rgb2hex(r,g,b)

    return {'rgb':rgb_arr, 'hex':_hex , 'dominance_level':dominance_level}


def map_to_palette(color_dict):
	rgb_arr= color_dict['rgb']
	color_dict['rgb']=np.array_str(color_dict['rgb'])
	rgb_arr= np.array([[rgb_arr]])
	_l,_a,_b= list(skcolor.rgb2lab(rgb_arr)[0][0])
	color1 = LabColor(lab_l=_l, lab_a=_a, lab_b=_b)

	'''delta_e_matrix = vfunc(color1, lab_matrix)
				e=np.argmin(delta_e_matrix)'''

	#delta_e_matrix= lab_matrix.apply(lambda x: func(color1,x))
	@np.vectorize
	def vfunc(x):
		return func(color1,x)

	@jit(nopython=True, parallel=True)
	def final_func(x):
		return vfunc(x)

	delta_e_matrix= vfunc(lab_matrix.values)
	e=  delta_e_matrix.argmin()

	return dict(palette.iloc[e])
