import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from PIL import Image
import time
#sys.path.append("..")  #Not sure if I need this line if I run it in the parental directory
from utils import label_map_util

MODELS = ['ssd_mobilenet_v1_coco_11_06_2017', 'ssd_inception_v2_coco_11_06_2017',\
                  'rfcn_resnet101_coco_11_06_2017', 'faster_rcnn_resnet101_coco_11_06_2017',\
                   'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017']




NUM_CLASSES = 90
PATH_TO_LABELS = "/root/models/object_detection/data/mscoco_label_map.pbtxt"
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


#Get the session from the specified model
def load_model(model, dynamic_memory=True):
    
    #Dynamically allocating memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=dynamic_memory
    sess = tf.Session(config=config)    
    PATH_TO_CKPT = os.path.join("/root/models/object_detection", model , 'frozen_inference_graph.pb')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
    return tf.Session(graph=detection_graph, config=config)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def translate_result(boxes, scores, classes, num_detections, im_width, im_height, thresh):
    #Normalizing the detection result
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)    
    
    outputs = []        
    for i, score in enumerate(scores):      
    #Stop when score is lower than threshold since the score is sorted.
        if score < thresh:
          break

        class_name = category_index[classes[i]]['name']
        ymin, xmin, ymax, xmax = boxes[i]
        left, right, top, bottom = (xmin * im_width, xmax * im_width,\
                                  ymin * im_height, ymax * im_height)          
        #Allocating result into ouput dict
        output = {}        
        output['score'] = score
        output['class'] = class_name
        output['x'] = left
        output['y'] = top
        output['width'] = right-left
        output['height'] = bottom-top
        #Append each detection into a list
        outputs.append(output)
    return outputs


def detect(sess, img_path, thresh=0.7):
    img = Image.open(img_path)
    img_width, img_height = img.size
    img_np = load_image_into_numpy_array(img)
    img_np_expanded = np.expand_dims(img_np, axis=0)
    
    #Initalization of output and input tensors for session
    img_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
    boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
    scores = sess.graph.get_tensor_by_name('detection_scores:0')
    classes = sess.graph.get_tensor_by_name('detection_classes:0')
    num_detections = sess.graph.get_tensor_by_name('num_detections:0')
    
    outputs = [boxes, scores, classes, num_detections]
    feed_dict = {img_tensor: img_np_expanded}
    boxes, scores, classes, num_detections = sess.run(outputs,feed_dict=feed_dict) 
 
    return translate_result(boxes, scores, classes, num_detections, img_width,\
                            img_height,thresh)
    
if __name__ == "__main__":
    TEST_IMAGE_PATHS = [os.path.join("/root/models/object_detection/test_images",\
                                     'image{}.jpg'.format(i)) for i in range(1,3)]
    THRESHOLD = 0.7
    model = MODELS[3]
    sess = load_model(model)
    for img_path in TEST_IMAGE_PATHS:
        result = detect(sess, img_path, thresh=THRESHOLD)
        outputs = result
        for output in outputs:                     
            score = output['score'] 
            class_name = output['class']
            x = output['x']
            y = output['y']
            width = output['width']
            height = output['height']                       
            print("'{}' detected with confidence {} in [{}, {}, {}, {}]".format(class_name.upper(),\
                                                                              score, x, y, width,\
                                                                              height))    