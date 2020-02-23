"""This script acts as the entry point of the repo.
	python driver.py <input_image_folder> <output_folder>
"""

__author__ = "Rahul Kumar Chaurasia"
__email__ = "rahul.8d@gmail.com"
__status__ = "Testing"


from helper import *
import sys
import json
import codecs
import os
import traceback
#import matplotlib.pyplot as plt
from palette4 import palette
from collections import defaultdict
import time

def run(path, save_path):

	#load image ---> default channel config : BGR
	#path=sys.argv[1]
	#save_path= sys.argv[2]
	img= cv2.imread(path,1)
	img= image_to_thumbs(img)['200']

	s= get_s_channel(img)

	th= get_mask(s,img)
	'''
	plt.imshow(np.hstack([s,th]), cmap='gray')
	plt.show()
	'''

	img,img_rgb,img_lab=preprocess_image(img,th)
	'''
	plt.imshow(img_rgb)
	plt.show()
	'''

	feature_image= get_data_points(img_lab)

	labels= get_cluster_labels(feature_image)


	n_clusters=len(set(labels)) -2

	print('n_clusters: {}'.format(n_clusters))

	label_map=np.reshape(labels, [img.shape[0], img.shape[1]])
	
	'''plt.imshow(label_map)
				plt.show()'''
	

	label_masks= get_label_masks(label_map,labels, img)

	roi_maps= get_roi_maps(label_masks,img_rgb)

	extracted_color= []
	th[th==255]=1
	for index,roi in enumerate(roi_maps):
		if index==0:
			continue
		try:
			color_dict= get_dominant_color(roi,label_masks, th,index)

			matched_color= map_to_palette(color_dict)

			color_dict['mapped_color']= matched_color['color_name']
			color_dict['mapped_color_hex']= matched_color['hex']
			extracted_color.append(color_dict)

		except:
			print(traceback.format_exc())
			continue

	base_color= defaultdict(list)
	color_list=set([color_dict['mapped_color'] for color_dict in extracted_color])

	temp_dict= {}
	for c in color_list:
			temp_dict['mapped_color']=c
			temp_dict['mapped_color_hex']= palette.loc[palette['color_name'] == c]['hex'].values[0]
			temp_dict['dominance_level']= sum([color_dict['dominance_level'] if color_dict['mapped_color']==c else 0 for color_dict in extracted_color])
			temp_dict['base_color']= palette.loc[palette['color_name'] == c]['base_color'].values[0]
			temp_dict['base_color_hex']= palette.loc[palette['color_name'] == c]['base_color_hex'].values[0]
			base_color[temp_dict['base_color']].append(temp_dict)
			temp_dict={}

	out=[]
	for k,v in base_color.items():
		a= {'base_color':k, 'dominance_level': str(sum([each['dominance_level'] for each in v])), 'base_color_hex':v[0]['base_color_hex']}
		out.append(a)
			
	out=sorted(out, key= lambda x: np.float32(x['dominance_level']), reverse=True)

	output_dict= {'filename' :path.split('\\')[-1], 'colors': out}

	

	with open(save_path,'wb') as f:
		json.dump(output_dict,  codecs.getwriter('utf-8')(f), ensure_ascii=False)

	return
if __name__ == '__main__':

	path=sys.argv[1]
	save_path= sys.argv[2]
	
	for p in os.listdir(path):
		t= time.time()
		_p=path+p
		s= save_path+p+'.json'
		print("{} {} \n".format(_p,s))
		run(_p,s)
		print(time.time()-t)
		

