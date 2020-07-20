import os
import numpy as np
import pandas as pd
from tqdm import tqdm

dataset=pd.read_csv('data/newest.csv')
# df_train=pd.DataFrame(columns=['ImageId','XMin', 'XMax', 'Ymin', 'Ymax', 'ClassName'])
# df_test=pd.DataFrame(columns=['ImageId','XMin', 'XMax', 'Ymin', 'Ymax', 'ClassName'])

rows_list_train=[]
rows_list_test=[]


dataset_processed={}
num_points=25
int_nd = 0 # integer for not defined
j=0
for index, row in tqdm(dataset.iterrows()):
    j=j+1
    dict_temp={}
    left_x=[]
    left_y=[]
    right_x=[]
    right_y=[]
    img_name=row['url'].split('/')[-1]
#     print(img_name)
    dataset_processed[img_name]={
            'l_prob' : row['l_probability'],
            'r_prob' : row['r_probability']
        }
    for i in (range(num_points)):
        if (str(row['l_'+str(i)+'_v'])=='True'):
            left_x.append(512 * (0.5 + float(row['l_' + str(i) + '_x'])))
            left_y.append(512 * (1 - (0.5 + float(row['l_' + str(i) + '_y']))))
#             left_x.append(float(row['l_'+str(i)+'_x']))
#             left_y.append(float(row['l_'+str(i)+'_y']))
            
        if (str(row['r_'+str(i)+'_v'])=='True'):
            right_x.append(512 * (0.5 + float(row['r_' + str(i) + '_x'])))
            right_y.append(512 * (1 - (0.5 + float(row['r_' + str(i) + '_y']))))
#             print('yes')
#             right_x.append(float(row['r_'+str(i)+'_x']))
#             right_y.append(float(row['r_'+str(i)+'_y']))

#in line 12284 str is present in l_0_x that's why did float to all.       
    if row['l_probability'] != 0.0:
        
        assert(len(left_x) == len(left_y))
        #removed those images that cause width or height to be 0
        if len(left_x) > 1 and min(left_x)!=max(left_x) and min(left_y)!=max(left_y):
            dict_temp={
                'ImageID' : img_name,
                'XMin' : min(left_x),
                'YMin' : min(left_y),
                'XMax' : max(left_x),
                'YMax' : max(left_y),#had left_y here. such a shitty mistake
                'ClassName' : str("left_foot")
            }
            if (j<int(0.8*dataset.shape[0])):
                rows_list_train.append(dict_temp)

            else:
                rows_list_test.append(dict_temp)
                
        else:
            print('Left Annotation Error:', img_name)
                
        
        
        
    if row['r_probability'] != 0.0:
        assert(len(right_x) == len(right_y))
        if len(right_x) > 1 and min(right_x)!=max(right_x) and min(right_y)!=max(right_y):
            dict_temp={
                'ImageID' : img_name,
                'XMin' : min(right_x),
                'YMin' : min(right_y),
                'XMax' : max(right_x),
                'YMax' : max(right_y),
                'ClassName' : str("right_foot")
            }
            if (j<int(0.8*dataset.shape[0])):
                rows_list_train.append(dict_temp)

            else:
                rows_list_test.append(dict_temp)
                
        else:
            print('Right Annotation Error:', img_name)
            
        
            
        
            
        
df_train = pd.DataFrame(rows_list_train)
df_test = pd.DataFrame(rows_list_test)

df_train=df_train[df_train.ImageID != "crop1002nhyui"] #faulty
df_train.to_csv('./data/sub-train-annotations-bbox.csv', index=False)
df_test.to_csv('./data/sub-test-annotations-bbox.csv',index=False)
        
#         if row['l_probability'] == 0.0:
#             dict_temp={
#                 'left_x_min' : int_nd,
#                 'left_y_min' : int_nd,
#                 'left_x_max' : int_nd,
#                 'left_y_max' : int_nd
#             }
#             l_width = 0
#             l_height = 0
#             l_center_x = 0
#             l_center_y = 0
#             dict_new={
#                 'l_width' : l_width,
#                 'l_height' : l_height,
#                 'l_center_x' : l_center_x,
#                 'l_center_y' : l_center_y
#             }
#             dataset_processed[img_name].update(dict_new)
                
#         else:
            
#             dict_temp={
#                 'left_x_min' : min(left_x),
#                 'left_y_min' : min(left_y),
#                 'left_x_max' : max(left_x),
#                 'left_y_max' : max(left_x)
#             }
# #             l_width = dict_temp['left_x_max']-dict_temp['left_x_min']
# #             l_height = dict_temp['left_y_max']-dict_temp['left_y_min']
# #             l_center_x = (dict_temp['left_x_max']+dict_temp['left_x_min'])/2
# #             l_center_y = (dict_temp['left_y_max']+dict_temp['left_y_min'])/2
            
# #             dict_new={
# #                 'l_width' : l_width,
# #                 'l_height' : l_height,
# #                 'l_center_x' : l_center_x,
# #                 'l_center_y' : l_center_y
# #             }
#             dataset_processed[img_name].update(dict_new)
                
                
                
#         if row['r_probability'] == 0.0:
            
#             dict_temp={
#                 'right_x_min' : int_nd,
#                 'right_y_min' : int_nd,
#                 'right_x_max' : int_nd,
#                 'right_y_max' : int_nd
#             }
#             r_width = 0
#             r_height = 0
#             r_center_x = 0
#             r_center_y = 0
#             dict_new={
#                 'r_width' : l_width,
#                 'r_height' : l_height,
#                 'r_center_x' : l_center_x,
#                 'r_center_y' : l_center_y
#             }
#             dataset_processed[img_name].update(dict_new)
                
#         else:
# #             print(right_x, index, row['r_probability'])
            
#             dict_temp={
#                 'right_x_min' : min(right_x),
#                 'right_y_min' : min(right_y),
#                 'right_x_max' : max(right_x),
#                 'right_y_max' : max(right_y)
        
#             }
#             r_width = dict_temp['right_x_max']-dict_temp['right_x_min']
#             r_height = dict_temp['right_y_max']-dict_temp['right_y_min']
#             r_center_x = (dict_temp['right_x_max']+dict_temp['right_x_min'])/2
#             r_center_y = (dict_temp['right_y_max']+dict_temp['right_y_min'])/2
            
#             dict_new={
#                 'r_width' : r_width,
#                 'r_height' : r_height,
#                 'r_center_x' : r_center_x,
#                 'r_center_y' : r_center_y
#             }
#             dataset_processed[img_name].update(dict_new)
            
#             if(j<int(0.8*dataset.shape[0])):
                