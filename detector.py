import numpy as np
import os
import tensorflow as tf
import time
import label_map_util
import argparse
import cv2
import glob

MODELS = ['faster_rcnn_inception_v2_coco_2017_11_08']


NUM_CLASSES = 90
PATH_TO_LABELS = "data\mscoco_label_map.pbtxt"
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

'''
def parse_args():
    parser = argparse.ArgumentParser(description='Tensor Flow Detection Models')
    parser.add_argument('--net', dest='net', help='Which model to specify:  1.SSD-Mobilenet, 2.SSD-Inception, 3.RFCN-ResNet, 4.Faster-RCNN-Resnet, 5.Faster-RCNN-Inception-Resnet',
                        default=1, type=int, choices=[1,2,3,4,5])
    args = parser.parse_args()
    return args
'''




#Get the session from the specified model
def load_model(model, dynamic_memory=True):
    
    #Dynamically allocating memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=dynamic_memory
    sess = tf.Session(config=config)    
    PATH_TO_CKPT = os.path.join(model , 'frozen_inference_graph.pb')
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


def translate_result(boxes, scores, classes, num_detections, im_width, im_height, thresh, roi):
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
        output['x'] = left + roi[0]
        output['y'] = top + roi[1]
        output['width'] = right - left
        output['height'] = bottom - top
        #Append each detection into a list
        outputs.append(output)
    return outputs


def detect(sess, img_path, roi=(0,0,0,0), thresh=0.7):
    #img = Image.open(img_path)
    #
    #img_np = load_image_into_numpy_array(img)
    img = cv2.imread(img_path)
     #img_height, img_width, _ = .shape
    
    if roi[2] > 0:
        roiImage = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] # y1, y2, x1, x2
    else:
        roiImage = img.copy()

    roiImage = cv2.cvtColor(roiImage, cv2.COLOR_RGB2BGR)
    
    img_height, img_width, _ = roiImage.shape
    img_np_expanded = np.expand_dims(roiImage, axis=0)
    
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
                            img_height, thresh, roi)


    
if __name__ == "__main__":

  

    TEST_IMAGE_PATHS = glob.glob("test_images/*.jpg")
    THRESHOLD = 0.7
    model = MODELS[0]
    sess = load_model(model) 
   
    for img_path in TEST_IMAGE_PATHS:
        tic = time.time()
        outputs = detect(sess, img_path, thresh=THRESHOLD)
        
        for output in outputs:                     
            score = output['score'] 
            class_name = output['class']
            x = output['x']
            y = output['y']
            width = output['width']
            height = output['height']   
            print("Detection Time {0:.2f} sec".format(time.time()-tic))       
            print("'{}' detected with confidence {} in [{}, {}, {}, {}]\n".format(class_name.upper(),\
                                                                               score, x, y, width,\
                                                                               height))    

            
            
            
            
            
