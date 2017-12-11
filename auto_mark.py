from demo import load_model, detect
import glob
import time

MODEL = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'


sess = load_model(MODEL) 
THRESHOLD = 0.7

TEST_IMAGE_PATHS = glob.glob("/root/data/demo/van_big/*.jpg")
                    
                    
              
for img_path in TEST_IMAGE_PATHS:
        tic = time.time()
        outputs = detect(sess, img_path, thresh=THRESHOLD)
        print("Detection Time {0:.2f} sec".format(time.time()-tic))      
        for output in outputs:                     
            score = output['score'] 
            class_name = output['class']
            x = output['x']
            y = output['y']
            width = output['width']
            height = output['height']   
             
            print("'{}' detected with confidence {} in [{}, {}, {}, {}]\n".format(class_name.upper(),\
                                                                               score, x, y, width,\
                                                                               height))    