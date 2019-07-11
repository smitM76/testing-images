from __future__ import absolute_import, division, print_function

import cv2
import imutils
import numpy as np
import os, os.path
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pylab as plt
import tensorflow_datasets as tfds
import tensorflow.keras.backend as K
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator as image_generator
from config.class_names import labels as lb
from config.make_model_config import feature_extractor_url,saved_model_path,tmp_masked_image_dir
from config.mask_rcnn_model_config import CLASS_NAME,weights



"""
importing and creating instance for mask-rcnn
"""
CLASS_NAMES = open(CLASS_NAME).read().strip().split("\n")


class SimpleConfig(Config):
	NAME = "coco_inference"
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	NUM_CLASSES = len(CLASS_NAMES)

config = SimpleConfig()

print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config,
	model_dir=os.getcwd())
model.load_weights(weights, by_name=True)

# loading custom trained model for make model prediction
new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)

# initializing lables according to classes [car names]
labels = np.array(lb)


def pridict_make_model(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(299,299))
    img = np.reshape(img,[1,299,299,3])

    sess = K.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)
    tmp_list = []

    result = new_model.predict(img)

    #getting the result with highest confidence
    y_classes = result.argmax(axis=-1)

    i = 0
    for res in result:
        while i < len(lb):
            a = {'make_model':lb[i],'confidence':res[i]}
            # skipping the result with highest confidence
            # storing rest of the results in a list
            if i == y_classes:
                pass
            else:
                if round(a['confidence'],6) >= 0.000001:
                    tmp_list.append(a)
            i += 1

    # getting label for highest confidence result
    predicted_label_ = labels[np.argmax(result[0])]
    return {'make_model':predicted_label_,'confidence':result[0][np.argmax(result[0])],'candidates':tmp_list}


def apply_filter_and_predict(img_path):
    # make_model_results = []
    """getting image and preprocessing for object detection with mask rcnn
    """
    image = cv2.imread(img_path)
    name = img_path.split('/')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = imutils.resize(image, width=512)
    print("[INFO] making predictions with Mask R-CNN...")
    """runnung object detection with mask rcnn
    returns all objects found into that image
    """
    r = model.detect([image])[0]
    """checking if car is present into objects,
    """
    i = 0
    while i < len(r["scores"]):
        """class id for car is 3
        """
        if r["class_ids"][i] == 3 or r["class_ids"][i] == 6 or r["class_ids"][i] == 8:
            print('car found')
            try:
                """getting pixel values of car object
                """
                (startY, startX, endY, endX) = r["rois"][i]
                classID = r["class_ids"][i]
                label = CLASS_NAMES[classID]
                score = r["scores"][i]
                """applying mask on car object
                """
                mask = np.zeros(image.shape[:2],np.uint8)
                bgdModel = np.zeros((1,65),np.float64)
                fgdModel = np.zeros((1,65),np.float64)
                rect = (startX,startY,endX,endY)
                cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                img = image*mask2[:,:,np.newaxis]
                """storing masked image as new image,
                if multiple car object found multiple image will store with image name plus prefix number
                """
                cv2.imwrite('{}{}{}'.format(tmp_masked_image_dir,i,name[-1]),img)
                new_image_path = '{}{}{}'.format(tmp_masked_image_dir,i,name[-1])

                """passing newly created masked images to pridict_make_model function for make model detection
                """
                pridiction_results = pridict_make_model(new_image_path)

                """storing results with higher confidence only
                """
                for candidate in pridiction_results['candidates']:
                    candidate['confidence'] = str(round(candidate['confidence'],5))

                if pridiction_results['confidence'] > 0 :
                    # make_model_results.append(pridiction_results)
                    pridiction_results['confidence'] = str(pridiction_results['confidence'])
                    result = pridiction_results
                else:
                    result = None
                    print('skipping')
            except Exception as e:
                print('too small object to be masked skipping.... \n {}'.format(e))
        else:
            result = None
            print('no car found in current frame')
        i += 1
    return result



def apply_crop(imageDir):
    image_path_list = []
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
    valid_image_extensions = [item.lower() for item in valid_image_extensions]

    for file in os.listdir(imageDir):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list.append(os.path.join(imageDir, file))

    for imagePath in image_path_list:
        image = cv2.imread(imagePath)
        name = imagePath.split('/')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = imutils.resize(image, width=512)

        print("[INFO] making predictions with Mask R-CNN...")
        r = model.detect([image], verbose=1)[0]

        i = 0
        while i < len(r["scores"]):
            if r["class_ids"][i] == 3 or r["class_ids"][i] == 6 or r["class_ids"][i] == 8:
                #print (f'image name : {i}{name[-1]} class Id : {r["class_ids"][i]}')
                startX, startY, endX, endY = r["rois"][i]
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                crop = image[startX:endX,startY:endY]
                #print (f'image name : {i}{name[-1]} class Id : {r["class_ids"][i]}')
                height, width = crop.shape[:2]
                if height > 150 and width > 150:
                     cv2.imwrite('{}{}{}'.format(imageDir,i,name[-1]),crop)
            else:
                print("\nName : \n" + name[-2])
                print("\nClass : \n", r["class_ids"][i])
            i += 1
