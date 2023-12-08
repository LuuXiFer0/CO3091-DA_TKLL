from .label import classNames
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import time
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import os

count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0
count_5 = 0
not_found = 0
MODE = "Detection"
SIGN = "unknown"

def reset():
    global count_1, count_2, count_3, count_4, count_5, MODE, SIGN
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    count_5 = 0
    MODE = "Detection"
    SIGN = "unknown"


    


def sign_detect(image):
    global count_1, count_2, count_3, count_4, count_5, MODE, SIGN, not_found
    
    model = load_model(os.path.abspath('src/prius_sdc_pkg/prius_sdc_pkg/Detection/Signs/sign_detection_reducev2.h5'))
    # path_to_weights =  os.path.abspath('src/prius_sdc_pkg/prius_sdc_pkg/Detection/Signs/yolov3_ts_train_5000.weights')
    path_to_cfg = os.path.abspath('src/prius_sdc_pkg/prius_sdc_pkg/Detection/Signs/yolov3_ts_test.cfg')
    labels = pd.read_csv(os.path.abspath('src/prius_sdc_pkg/prius_sdc_pkg/Detection/Signs/label_names_reducev2.csv'))
    with open(os.path.abspath('src/prius_sdc_pkg/prius_sdc_pkg/Detection/Signs/mean_image_rgb.pickle'), 'rb') as f:
        mean = pickle.load(f, encoding='latin1') 
    # Loading trained YOLO v3 weights and cfg configuration file by 'dnn' library from OpenCV
    network = cv2.dnn.readNetFromDarknet(path_to_cfg, path_to_weights)

    # To use with GPU
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

    layers_all = network.getLayerNames()
    layers_names_output = [layers_all[i-1] for i in network.getUnconnectedOutLayers()]

    # Minimum probability to eliminate weak detections
    probability_minimum = 0.2

    # Setting threshold to filtering weak bounding boxes by non-maximum suppression
    threshold = 0.2
    # Generating colours for bounding boxes
    # randint(low, high=None, size=None, dtype='l')
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Getting spatial dimension of input image
    h, w = image.shape[:2]

    # Variable for counting total processing time
    t = 0

    # Blob from current frame
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Forward pass with blob through output layers  
    network.setInput(blob)
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    # Time
    t += end - start
    print('Total amount of time {:.5f} seconds'.format(t))

    # Lists for detected bounding boxes, confidences and class's number
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]

            # Eliminating weak predictions by minimum probability
            if confidence_current > probability_minimum:
                # Scaling bounding box coordinates to the initial frame size
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Getting top left corner coordinates
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    # Implementing non-maximum suppression of given bounding boxes
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

    # Checking if there is any detected object been left
    if len(results) > 0:
        # Going through indexes of results
        for i in results.flatten():
            # Bounding box coordinates, its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                
                
            # Cut fragment with Traffic Sign
            c_ts = image[y_min:y_min+int(box_height), x_min:x_min+int(box_width), :]
            # print(c_ts.shape)
                
            if c_ts.shape[:1] == (0,) or c_ts.shape[1:2] == (0,):
                pass
            else:
                # Getting preprocessed blob with Traffic Sign of needed shape
                blob_ts = cv2.dnn.blobFromImage(c_ts, 1 / 255.0, size=(32, 32), swapRB=True, crop=False)
                blob_ts[0] = blob_ts[0, :, :, :] - mean['mean_image_rgb']
                blob_ts = blob_ts.transpose(0, 2, 3, 1)
                # plt.imshow(blob_ts[0, :, :, :])
                # plt.show()

                # Feeding to the Keras CNN model to get predicted label among 43 classes
                scores = model.predict(blob_ts)

                # Scores is given for image with 43 numbers of predictions for each class
                # Getting only one class with maximum value
                prediction = np.argmax(scores)
                print(type(prediction))
                # print(labels['SignName'][prediction])
                # print(confidences[i])

                # Colour for current bounding box
                colour_box_current = colours[class_numbers[i]].tolist()
                
                # Green BGR
                colour_box_current = [0, 255, 61]

                # Drawing bounding box on the original current frame
                cv2.rectangle(image, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box_current, 6)
                
                if prediction == 3:
                    count_1 += 1
                    # Preparing text with label and confidence for current bounding box
                    text_box_current = '{}: {:.4f}'.format('Speed limit: 60km/h', confidences[i])

                    # Putting text with label and confidence on the original image
                    cv2.putText(image, text_box_current, (x_min - 110, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour_box_current, 2)
                    
                elif prediction == 14:            
                    count_2 += 1
                    # Preparing text with label and confidence for current bounding box
                    text_box_current = '{}: {:.4f}'.format('Stop', confidences[i])

                    # Putting text with label and confidence on the original image
                    cv2.putText(image, text_box_current, (x_min - 110, y_min + box_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour_box_current, 2)
                elif prediction == 23:  
                    count_3 += 1
                    # Preparing text with label and confidence for current bounding box
                    text_box_current = '{}: {:.4f}'.format('Slippery road', confidences[i])

                    # Putting text with label and confidence on the original image
                    cv2.putText(image, text_box_current, (x_min - 110, y_min + box_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour_box_current, 2)
                elif prediction == 33:            
                    count_4 += 1

                    # Preparing text with label and confidence for current bounding box
                    text_box_current = '{}: {:.4f}'.format('Turn right ahead', confidences[i])

                    # Putting text with label and confidence on the original image
                    cv2.putText(image, text_box_current, (x_min - 110, y_min + box_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour_box_current, 2)
                elif prediction == 34:     
                    count_5 += 1
                    # Preparing text with label and confidence for current bounding box
                    text_box_current = '{}: {:.4f}'.format('Turn left ahead', confidences[i])

                    # Putting text with label and confidence on the original image
                    cv2.putText(image, text_box_current, (x_min - 110, y_min + box_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour_box_current, 2)
    else: 
        not_found += 1

    if(count_1 > 5):
        MODE = "Tracking"
        SIGN = "speed_sign_60"
        reset()

    if(count_2 > 5):
        MODE = "Tracking"
        SIGN = "stop"
        reset()

    if(count_3 > 5):
        MODE = "Tracking"
        SIGN = "slippery_road"
        reset()

    if(count_4 > 5):
        MODE = "Tracking"
        SIGN = "right_turn"
        reset()

    if(count_5 > 5):
        MODE = "Tracking"
        SIGN = "left_turn"
        reset()

    if(not_found > 5):
        MODE = "Detection"
        SIGN = "unknown"
        reset()

    # if
    return MODE, SIGN
