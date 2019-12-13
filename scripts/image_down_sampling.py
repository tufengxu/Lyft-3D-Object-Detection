# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 21:28:49 2019

@author: zephyrus
"""

import cv2
import threading
import os
import winsound

def file_abs_path_folder(folder_path):
    return([folder_path+os.sep+name for name in os.listdir(folder_path)])
    
def resize_img(img_path,save_folder):
    img=cv2.imread(img_path)
    img_size=img.shape
    _,img_name=os.path.split(img_path)
    resized_img=cv2.resize(img,(img_size[1]//4,img_size[0]//4),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(save_folder+os.sep+img_name,resized_img)
    return

def down_sampling_img_in_folder(input_folder,output_folder,thread_num=200):
    file_paths=file_abs_path_folder(input_folder)
    img_num=len(file_paths)
    rounds=img_num//thread_num+1
    for i in range(rounds):
        threads_lis=[]
        for j in range(thread_num):
            img_index=i*thread_num+j
            if img_index<img_num and img_index>12800:
                temp_thread=threading.Thread(target=resize_img,args=(file_paths[img_index],output_folder))
                threads_lis.append(temp_thread)
        if len(threads_lis)>0:
            for t in threads_lis:
                t.setDaemon(True)
                t.start()
            t.join()
    return

test_input_path=r'G:\Download\3d-object-detection-for-autonomous-vehicles\3d-object-detection-for-autonomous-vehicles\train_images'
test_output_path=r'G:\Download\3d-object-detection-for-autonomous-vehicles\3d-object-detection-for-autonomous-vehicles\train_imgs_small'

down_sampling_img_in_folder(test_input_path,test_output_path)

duration = 1000  # millisecond
freq = 440  # Hz
winsound.Beep(freq, duration)

#file_abs_path_lis=file_abs_path_folder(test_input_path)
#print(file_abs_path_lis[0])
#img=cv2.imread(file_abs_path_lis[0],1)
#cv2.namedWindow("Image",cv2.WINDOW_NORMAL) 
#cv2.imshow("Image", img) 
#cv2.waitKey (0)
#cv2.destroyAllWindows()


































































































