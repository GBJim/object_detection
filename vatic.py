# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

#This is negative_ignore version of imdb class for Caltech Pedestrian dataset


import os
import numpy as np


import json
from os.path import isfile, join, basename

import glob
#from datasets.config import CLASS_SETS
from natsort import natsorted


def get_data_map(path="/root/data", prefix="data-"):
    data_map = {} 
    data_paths = glob.glob("{}/{}*".format(path, prefix))
    for data_path in data_paths:
        name = basename(data_path)[5:]
        data_map[name] = data_path
    return data_map    

data_map = get_data_map()
data_names = data_map.keys()
    
    

def has_data(name):
    return name in data_names

def load_meta(meta_path):
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
    else:
       
        meta = {"format":"jpg"}
        meta["train"] = {"start":None, "end":None, "stride":1, "sets":[0]}
        meta["test"] = {"start":None, "end":None, "stride":30, "sets":[1]}
        print("Meta data path: {} does not exist. Use Default meta data".format(meta_path))
    return meta
 

        


class VaticData():
    def __init__(self, name, train_split="train", test_split="test"):
        
        assert data_map.has_key(name),\
        'The {} dataset does not exist. The available dataset are: {}'.format(name, data_map.keys())
            
        self._data_path = data_map[name]  
        assert os.path.exists(self._data_path), \
        'Path does not exist: {}'.format(self._data_path)
        
 
        
        annotation_path = os.path.join(self._data_path, "annotations.json")         
        assert os.path.exists(annotation_path), \
                'Annotation path does not exist.: {}'.format(annotation_path)
        self._annotation = json.load(open(annotation_path))   
        
        self.class_set = self.get_class_set()
        
        
        
        meta_data_path = os.path.join(self._data_path, "meta.json") 
        
       
            
        self._meta = load_meta(meta_data_path)
        
        if train_split == "train" or train_split ==  "test":
            pass
        elif train_split == "all":
            print("Use both split for training")
            self._meta["train"]["sets"] +=  self._meta["test"]["sets"]
        else:
            raise("Options except train and test are not supported!")
            
            
        if test_split == "train" or test_split ==  "test":
            pass
        elif test_split == "all":
            print("Use both split for testing")
            self._meta["test"]["sets"] +=  self._meta["train"]["sets"]
        else:
            raise("Options except train and test are not supported!")
        

        self._image_ext = self._meta["format"]    
        self._image_ext = '.jpg'
        self._image_index = self._get_image_index()
        self.size = len(self._image_index)
    
    
    def get_class_set(self):
        class_set = set()
        for set_num in self._annotation:
                for bboxes in self._annotation[set_num].values():
                    for bbox in bboxes.values():
                        class_set.add(bbox['label'])  
        
        
        
        return class_set
       
    
   

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        set_num, v_num, frame = index.split("_")
        image_path = os.path.join(self._data_path, 'images', set_num, v_num, index+self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path
    
    def bbox_from_index(self, index):
        img_name = os.path.basename(index)
        set_num, v_num, img_num = img_name.split("_")
        set_num = set_num[-1]
        img_num = img_num[:-4]
        return self._annotation[set_num].get(img_num, {})
    
    
    
   
    def _load_image_set_list(self):
        image_set_file = os.path.join(self._data_path,
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        f = open(image_set_file)
        return  [line.strip() for line in f]
    
    
   
        

    def _get_image_index(self):
      
        """
        Load the indexes listed in this dataset's image set file.
        """
        
        
        image_path = os.path.join(self._data_path, 'images')
        assert os.path.exists( image_path), \
                'Path does not exist: {}'.format( image_path)
        target_imgs = []
        
        sets = self._meta["train"]["sets"]
        start = self._meta["train"]["start"]
        end = self._meta["train"]["end"]
        stride = self._meta["train"]["stride"]
        
        
        if start is None:
            start = 0
            
        for set_num in self._meta["train"]["sets"]:
            img_pattern = "{}/set0{}/V000/set0{}_V*.jpg".format(image_path,set_num,set_num)       
            img_paths = natsorted(glob.glob(img_pattern))
            #print(img_paths)
            
            first_ind = start
            last_ind = end if end else len(img_paths)
            for i in range(first_ind, last_ind, stride):
                img_path = img_paths[i]
                img_name = os.path.basename(img_path)
                target_imgs.append(img_name[:-4])
        print(self._meta)    
        print("Total: {} images".format(len(target_imgs)))            
        return target_imgs                  
            
                                
    
#Unit Test
if __name__ == '__main__':
   
    
  
    name_a = "chruch_street"
    
    A = VaticData("chruch_street")
    B = VaticData("YuDa")
    img_path = A.image_path_at(5)
    bbox = A.bbox_from_index(img_path)
    
    
    
    
    from IPython import embed; embed()