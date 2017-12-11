import numpy as np
import os
import sys
import tensorflow as tf
import label_map_util
import cv2
import time
from vatic import VaticData
from config import CLASS_SETS
import glob
import re
from timer import Timer


MODELS = ['ssd_mobilenet_v1_coco_11_06_2017', 'ssd_inception_v2_coco_11_06_2017',\
                  'rfcn_resnet101_coco_11_06_2017', 'faster_rcnn_resnet101_coco_11_06_2017',\
                   'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017']


NUM_CLASSES = 90
PATH_TO_LABELS = "/root/object_detection/data/mscoco_label_map.pbtxt"
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


#Get the session from the specified model
def load_model(model, dynamic_memory=True):
    
    #Dynamically allocating memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=dynamic_memory
    sess = tf.Session(config=config)    
    PATH_TO_CKPT = os.path.join("/root/object_detection", model , 'frozen_inference_graph.pb')
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
    
    thresh_mask = scores > thresh
    
    scores = scores[thresh_mask]
    boxes = boxes[thresh_mask]
    classes = classes[thresh_mask]
    
    outputs = []        
    for i, score in enumerate(scores):      
        #Stop when score is lower than threshold since the score is sorted
        #!!!!Performance of this line can be improved!!!
       

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
    #img = Image.open(img_path)
    #
    #img_np = load_image_into_numpy_array(img  OUTPUT_DIR= os.path.join(imdb._data_path ,"res" ,output_name)    
  
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    
    img_height, img_width, _ = img.shape
    img_np_expanded = np.expand_dims(img, axis=0)
    
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








def write_testing_results_file(net, imdb, skip, OUTPUT_DIR, CLASSES):
    
  


    # The follwing nested fucntions are for smart sorting
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split('(\d+)', text) ]
    
    
    

    def insert_frame(target_frames, file_path,start_frame, frame_stride, end_frame):
        file_name = file_path.split("/")[-1]
        set_num, v_num, frame_num = file_name[:-4].split("_")
        condition = int(frame_num) >= start_frame and (int(frame_num)+1) % frame_stride == 0 and int(frame_num) < end_frame
        #print(frame_num,start_frame, frame_stride, end_frame, condition)

        if condition:
            target_frames.setdefault(set_num,{}).setdefault(v_num,[]).append(file_path)
            return 1
        else:
            return 0

 


    def get_target_frames(image_set_list, imdb):
        image_path = os.path.join(imdb._data_path, "images")
        start_frame = imdb._meta["test"]["start"]
        end_frame = imdb._meta["test"]["end"]
        frame_stride = skip if skip else imdb._meta["test"]["stride"]
        
        if start_frame is None:
            start_frame = 0
        
        target_frames = {}
        total_frames = 0 
        for set_num in image_set_list:
            file_pattern = "{}/set{}/V000/set{}_V*".format(image_path,set_num,set_num)
            #print(file_pattern)
            #print(file_pattern)
            file_list = sorted(glob.glob(file_pattern), key=natural_keys)
            
            if end_frame is None:
                last_file = file_list[-1]
                end_frame =  int(last_file.split("_")[-1].split(".")[0])
                
            #print(file_list)
            for file_path in file_list:
                total_frames += insert_frame(target_frames, file_path, start_frame, frame_stride, end_frame)

        return target_frames, total_frames 
    
    

    def detection_to_file(target_path, v_num, file_list ,total_frames, current_frames, max_proposal=100, thresh=0):
        timer = Timer()
        w = open("{}/{}.txt".format(target_path, v_num), "w")
        for file_index, file_path in enumerate(file_list):
            file_name = file_path.split("/")[-1]
            set_num, v_num, frame_num = file_name[:-4].split("_")
            frame_num = str(int(frame_num) +1)
            im = cv2.imread(file_path)
            timer = Timer()
            timer.tic()
            outputs = detect(sess, file_path, thresh=0)
            timer.toc()
            print('Detection Time:{:.3f}s on {}  {}/{} images'.format(timer.average_time,\
                                                   file_name ,current_frames+file_index+1 , total_frames))
            
           
           
            for output in outputs:
                score = output['score'] * 100
                label = output['class']
                x = output['x']
                y = output['y']
                width = output['width']
                height = output['height']    
                if label in TARGET_CLASSES:
                    w.write("{},{},{},{},{},{},{}\n".format(frame_num, x, y, width, height, score, label))
          


        w.close()
        print("Evalutaion file {} has been writen".format(w.name))   
            
        
        return file_index + 1
        
        
    image_set_list = [ str(set_num).zfill(2) for set_num in imdb._meta["test"]["sets"]]
    target_frames, total_frames = get_target_frames(image_set_list,  imdb)
    #print(target_frames)
    #print(total_frames)imdb


    current_frames = 0 
    if not os.path.exists(OUTPUT_DIR ):
        os.makedirs(OUTPUT_DIR ) 
    for set_num in target_frames:
        target_path = os.path.join(OUTPUT_DIR , set_num)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        for v_num, file_list in target_frames[set_num].items():
            current_frames += detection_to_file(target_path, v_num, file_list, total_frames, current_frames)


    
















if __name__ == '__main__':
    #TARGET_CLASSES = ["person", "backpack", "handbag", "suitcase"]
    TARGET_CLASSES = ["car", "person", "bus", "van", "truck", "scooter", "bike", "pickup"]
 
    data_name = "Thailand"  
    output_name = "tf-fasterrcnn"        
    
    THRESHOLD = 0
    model = MODELS[4]
    sess = load_model(model)
     
    FPS_rate = 1
    GPU_ID = 3

    classes = CLASS_SETS["coco"]
    imdb = VaticData(data_name, classes)
    OUTPUT_DIR= os.path.join(imdb._data_path ,"res" ,output_name)    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR ) 
     
    print("Model Loaded")
    print("Start Detecting")    
 
    write_testing_results_file(sess, imdb, FPS_rate, OUTPUT_DIR, classes)      


  
 
