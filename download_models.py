# A python script to download the five tensorflow models
import six.moves.urllib as urllib
import tarfile
import os

def download_model(model_name, file_format='.tar.gz',\
                   base_url='http://download.tensorflow.org/models/object_detection/'):
 
    model_file =  model_name + file_format
    opener = urllib.request.URLopener()
    print("Downloading model: {}".format(model_name))
    opener.retrieve(base_url + model_file, model_file)
    tar_file = tarfile.open(model_file)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
    print("Extraction Done: {}".format(model_name+file_format)) 
            
if __name__ == "__main__":
    
    model_names = ['ssd_mobilenet_v1_coco_11_06_2017', 'ssd_inception_v2_coco_11_06_2017',\
                  'rfcn_resnet101_coco_11_06_2017', 'faster_rcnn_resnet101_coco_11_06_2017',\
                   'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017']
    
    for model_name in model_names:    
        download_model(model_name)