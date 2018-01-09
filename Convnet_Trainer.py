# import libraries
from __future__ import print_function
from __future__ import division

import numpy as np 
import pandas as pd

import os
import re
import cv2

import tensorflow as tf
import tflearn


from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected

from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

import random






from timeit import default_timer as timer

#import tsahelper as tsa
import tsahelper.tsahelper as tsa


# Setting up of all folders needed to run the convnet model with training and testing followed up the generation of predictions 

INPUT_FOLDER = 'stage1_aps/'
PREPROCESSED_DATA_FOLDER = 'preprocessed/'
PREPROCESSED_NEW_DATA_FOLDER = 'preprocessed_new/'

# Need all these files and this folder structure set up in order to run this code 
STAGE1_LABELS = 'stage1_labels.csv'
STAGE1_NEW_LABELS = 'stage1_all.csv'

THREAT_ZONE = 1 # This number is changed based on what threat zone we want to train and get predictions for

BATCH_SIZE = 16

EXAMPLES_PER_SUBJECT = 182

FILE_LIST = []
TRAIN_TEST_SPLIT_RATIO = 0.2


TRAIN_SET_FILE_LIST = []
TEST_SET_FILE_LIST = []

IMAGE_DIM = 250
LEARNING_RATE = 1e-3
N_TRAIN_STEPS = 1
TRAIN_PATH = 'tsalogs/train/'
MODEL_PATH = 'tsalogs/model/'
MODEL_NAME = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('alexnet-v0.1', LEARNING_RATE, IMAGE_DIM, 
                                                IMAGE_DIM, THREAT_ZONE )) 

def threshold(a, threshmin=None, threshmax=None, newval=0):
    a = np.ma.array(a, copy=True)
    mask = np.zeros(a.shape, dtype=bool)
    if threshmin is not None:
        mask |= (a < threshmin).filled(False)

    if threshmax is not None:
        mask |= (a > threshmax).filled(False)

    a[mask] = newval
    return a

def spread_spectrum(img):
    #img = stats.threshold(img, threshmin=12, newval=0)
    img = threshold(img, threshmin=12, newval=0)
    
    # see http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img= clahe.apply(img)
    
    return img
  

# Generates the list of subjects that are required in the final submission file 
# This is so we know that we have to generate predictions for these 100 subjects  
input_v = 'stage1_sample_submission.csv'

sub_list = pd.read_csv(input_v)
sub_list['Subject'], sub_list['Zone'] = sub_list['Id'].str.split('_',1).str

sub_list = sub_list['Subject'].unique()

sub_list = sub_list.tolist()

sub_list[0:20]


def preprocess_tsa_data(): # Preprocessing all subjects with labels
    
    # Type 1: get a list of all subjects for whom there is data
    #SUBJECT_LIST = [os.path.splitext(subject)[0] for subject in os.listdir(INPUT_FOLDER)]
    
    # Type  2: get a list of subjects for small bore test purposes
    #SUBJECT_LIST = ['00360f79fd6e02781457eda48f85da90','0043db5e8c819bffc15261b1f1ac5e42',
    #                '0050492f92e22eed3474ae3a6fc907fa','006ec59fa59dd80a64c85347eef810c7',
    #                '0097503ee9fa0606559c56458b281a08','011516ab0eca7cad7f5257672ddde70e']
    

    # METHOD 3: get a list of all subjects for which there are labels (1200 subjects)

    df = pd.read_csv(STAGE1_LABELS)

    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    
    SUBJECT_LIST = df['Subject'].unique()


    # intialize tracking and saving items
    batch_num = 1

    threat_zone_examples = []
    
    start_time = timer()
    
    for subject in SUBJECT_LIST: # for each subject

        # read in the images

        print('t+> {:5.3f} |Reading images for subject #: {}'.format(timer()-start_time, subject))
        
        images = tsa.read_data(INPUT_FOLDER + '/' + subject + '.aps')

        # transpose so that the slice is the first dimension shape(16, 620, 512)

        images = images.transpose()

        # for each threat zone, loop through each image, mask off the zone and then crop it

        for tz_num, threat_zone_x_crop_dims in enumerate(zip(tsa.zone_slice_list, 
                                                             tsa.zone_crop_list)):

            threat_zone = threat_zone_x_crop_dims[0]
            crop_dims = threat_zone_x_crop_dims[1]

            # get label
            label = np.array(tsa.get_subject_zone_label(tz_num, tsa.get_subject_labels(STAGE1_LABELS, subject)))

            for img_num, img in enumerate(images):

                print('Threat Zone:Image -> {}:{}'.format(tz_num, img_num))
                
                print('Threat Zone Label -> {}'.format(label))
                
                if threat_zone[img_num] is not None: # If there is a threat zone observable

                    # correct the orientation of the image
                    print('-> reorienting base image') 

                    base_img = np.flipud(img)
                    
                    print('-> shape {}|mean={}'.format(base_img.shape, base_img.mean()))

                    # convert to grayscale
                    print('-> converting to grayscale')

                    
                    rescaled_img = tsa.convert_to_grayscale(base_img)
                    
                    print('-> shape {}|mean={}'.format(rescaled_img.shape, rescaled_img.mean()))

                    # spread the spectrum to improve contrast

                    print('-> spreading spectrum')
                    
                    #high_contrast_img = tsa.spread_spectrum(rescaled_img)
                    
                    high_contrast_img = spread_spectrum(rescaled_img)
                    
                    print('-> shape {}|mean={}'.format(high_contrast_img.shape, high_contrast_img.mean()))

                    # get the masked image
                    print('-> masking image')

                    masked_img = tsa.roi(high_contrast_img, threat_zone[img_num])
                    
                    print('-> shape {}|mean={}'.format(masked_img.shape, masked_img.mean()))

                    # crop the image
                    print('-> cropping image')

                    cropped_img = tsa.crop(masked_img, crop_dims[img_num])

                    print('-> shape {}|mean={}'.format(cropped_img.shape, cropped_img.mean()))

                    # normalize the image
                    print('-> normalizing image')

                    
                    normalized_img = tsa.normalize(cropped_img)
                    
                    print('-> shape {}|mean={}'.format(normalized_img.shape, normalized_img.mean()))

                    # zero center the image
                    print('-> zero centering')

                    zero_centered_img = tsa.zero_center(normalized_img)
                    
                    print('-> shape {}|mean={}'.format(zero_centered_img.shape, zero_centered_img.mean()))

                    # append the features and labels to this threat zone's example array
                    print ('-> appending example to threat zone {}'.format(tz_num))

                    threat_zone_examples.append([[tz_num], zero_centered_img, label])
                    
                    print ('-> shape {:d}:{:d}:{:d}:{:d}:{:d}:{:d}'.format(
                                                         len(threat_zone_examples),
                                                         len(threat_zone_examples[0]),
                                                         len(threat_zone_examples[0][0]),
                                                         len(threat_zone_examples[0][1][0]),
                                                         len(threat_zone_examples[0][1][1]),
                                                         len(threat_zone_examples[0][2])))
                else:
                    print('-> No view of tz:{} in img:{}. Skipping to next...'.format(tz_num, img_num))


        # This writes the the data once there is a full minibatch completed

        if ((len(threat_zone_examples) % (BATCH_SIZE * EXAMPLES_PER_SUBJECT)) == 0):
         
            for tz_num, tz in enumerate(tsa.zone_slice_list):

                tz_examples_to_save = []

                # write out the batch and reset
                print(' -> writing: ' + PREPROCESSED_DATA_FOLDER + 'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format( tz_num+1, len(threat_zone_examples[0][1][0]), len(threat_zone_examples[0][1][1]), batch_num))

                # get this tz's examples
                tz_examples = [example for example in threat_zone_examples if example[0] == [tz_num]]

                # drop unused columns
                tz_examples_to_save.append([[features_label[1], features_label[2]] for features_label in tz_examples])

                np.save(PREPROCESSED_DATA_FOLDER +  'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, len(threat_zone_examples[0][1][0]),len(threat_zone_examples[0][1][1]),batch_num), tz_examples_to_save)
                del tz_examples_to_save

            #reset for next batch 

            del threat_zone_examples

            threat_zone_examples = []
            batch_num += 1
    
    # we may run out of subjects before we finish a batch, so we write out the last batch stub

    if (len(threat_zone_examples) > 0):
        for tz_num, tz in enumerate(tsa.zone_slice_list):

            tz_examples_to_save = []

            # write out the batch and reset
            print(' -> writing: ' + PREPROCESSED_DATA_FOLDER + 'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, len(threat_zone_examples[0][1][0]), len(threat_zone_examples[0][1][1]), batch_num))

            # get this tz's examples
            tz_examples = [example for example in threat_zone_examples if example[0] == [tz_num]]

            # drop unused columns
            tz_examples_to_save.append([[features_label[1], features_label[2]] for features_label in tz_examples])

            #save batch
            np.save(PREPROCESSED_DATA_FOLDER +  'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, len(threat_zone_examples[0][1][0]), len(threat_zone_examples[0][1][1]), batch_num), tz_examples_to_save)

preprocess_tsa_data()


#Once preprocessing is done for all subjects for a particular tzone we split the files into training and testing

def get_train_test_file_list():
    
    global FILE_LIST
    global TRAIN_SET_FILE_LIST
    global TEST_SET_FILE_LIST

    if os.listdir(PREPROCESSED_DATA_FOLDER) == []:
        print ('No preprocessed data available.  Skipping preprocessed data setup..')
    else:
        #FILE_LIST = [f for f in os.listdir(PREPROCESSED_DATA_FOLDER) 
        #             if re.search(re.compile('-tz' + str(THREAT_ZONE) + '-'), f)]
        FILE_LIST = [f for f in os.listdir(PREPROCESSED_DATA_FOLDER)if re.search(re.compile('-tz' + str(THREAT_ZONE) + '-'), f)]

        train_test_split = len(FILE_LIST) -  max(int(len(FILE_LIST)*TRAIN_TEST_SPLIT_RATIO),1)

        TRAIN_SET_FILE_LIST = FILE_LIST[:train_test_split]
        TEST_SET_FILE_LIST = FILE_LIST[train_test_split:]
        
        TRAIN_SET_FILE_LIST = TEST_SET_FILE_LIST
        
        print('Train/Test Split -> {} file(s) of {} used for testing'.format(len(FILE_LIST) - train_test_split, len(FILE_LIST)))
        

get_train_test_file_list()

# Gets the file and the path and generates the feature batch and label batch to be used by our model for training and testing 
def gen_features_labels(filename, path):

    preprocessed_tz_scans = []
    feature_batch = []
    label_batch = []
    
    #Load a batch of preprocessed tz scans
    preprocessed_tz_scans = np.load(os.path.join(path, filename))
        
    #Shuffle to randomize for input into the model
    np.random.shuffle(preprocessed_tz_scans)
    
    # separate features and labels
    for example_list in preprocessed_tz_scans:
        for example in example_list:
            feature_batch.append(example[0])
            label_batch.append(example[1])
    
    feature_batch = np.asarray(feature_batch, dtype=np.float32)
    label_batch = np.asarray(label_batch, dtype=np.float32)
    
    return feature_batch, label_batch
  
print ('Train Set ')

for f_in in TRAIN_SET_FILE_LIST:
    feature_batch, label_batch = gen_features_labels(f_in, PREPROCESSED_DATA_FOLDER)

    print (' -> features shape {}:{}:{}'.format(len(feature_batch), len(feature_batch[0]),len(feature_batch[0][0])))

    print (' -> labels shape   {}:{}'.format(len(label_batch), len(label_batch[0])))
    




print ('Test Set ')

for f_in in TEST_SET_FILE_LIST:

    feature_batch, label_batch = gen_features_labels(f_in, PREPROCESSED_DATA_FOLDER)

    print (' -> features shape {}:{}:{}'.format(len(feature_batch), len(feature_batch[0]), len(feature_batch[0][0])))

    print (' -> labels shape   {}:{}'.format(len(label_batch), len(label_batch[0])))

#Shuffle for some randomization
def shuffle_train_set(train_set):
    sorted_file_list = random.shuffle(train_set)
    TRAIN_SET_FILE_LIST = sorted_file_list
    
print ('Before Shuffling ->', TRAIN_SET_FILE_LIST)

shuffle_train_set(TRAIN_SET_FILE_LIST)

print ('After Shuffling ->', TRAIN_SET_FILE_LIST)



# The Alexnet was first put to the real world test during the ImageNet Large Scale Visual Recognition Challenge in 2012.
# The performance of this network was a quantum shift for its time as the model achieved a top-5 error of 15.3%, more than 10.8 percentage points ahead of the runner up. 
# But in short the network consists of 7 layers, 5 convolutions/maxpools, plus 2 regression layers at the end.  

# Alexnet network is used by our model

def alexnet(width, height, lr):

    network = input_data(shape=[None, width, height, 1], name='features')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    
    network = max_pool_2d(network, 3, strides=2)
    
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    
    network = max_pool_2d(network, 3, strides=2)
    
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    
    network = max_pool_2d(network, 3, strides=2)
    
    network = local_response_normalization(network)
    
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    
    network = fully_connected(network, 2, activation='softmax')
    
    network = regression(network, optimizer='momentum', loss='categorical_crossentropy', 
                         learning_rate=lr, name='labels')

    model = tflearn.DNN(network, checkpoint_path=MODEL_PATH + MODEL_NAME, 
                        tensorboard_dir=TRAIN_PATH, tensorboard_verbose=3, max_checkpoints=1)

    return model


# Training
# 
# Loop to read in minibatches for test and train and run the fit method. 
# TFLearn treats each "minibatch" as an epoch.   
    
val_features = []
val_labels = []

# get train and test batches
get_train_test_file_list()

# instantiate model
model = alexnet(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE)

# read in the validation test set
for j, test_f_in in enumerate(TEST_SET_FILE_LIST):
    if j == 0:
        val_features, val_labels = gen_features_labels(test_f_in, PREPROCESSED_DATA_FOLDER)
    else:
        tmp_feature_batch, tmp_label_batch = gen_features_labels(test_f_in, 
                                                            PREPROCESSED_DATA_FOLDER)
        val_features = np.concatenate((tmp_feature_batch, val_features), axis=0)
        val_labels = np.concatenate((tmp_label_batch, val_labels), axis=0)

val_features = val_features.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)

# start training process
for i in range(N_TRAIN_STEPS):

    # shuffle the train set files before each step
    shuffle_train_set(TRAIN_SET_FILE_LIST)

    # run through every batch in the training set
    for f_in in TRAIN_SET_FILE_LIST:

        # read in a batch of features and labels for training
        feature_batch, label_batch = gen_features_labels(f_in, PREPROCESSED_DATA_FOLDER)
        feature_batch = feature_batch.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
        print ('Feature Batch Shape ->', feature_batch.shape)                

        # run the fit operation
        print(model.fit({'features': feature_batch}, {'labels': label_batch}, n_epoch=1, 
                  validation_set=({'features': val_features}, {'labels': val_labels}), 
                  shuffle=True, snapshot_step=None, show_metric=True, 
                  run_id=MODEL_NAME))

        
# This is to preprocess test data for the 100 subjects

def preprocess_tsa_test_data():
    
    SUBJECT_LIST = sub_list[0:100]
    
    # intialize tracking and saving items
    batch_num = 1
    threat_zone_examples = []
    start_time = timer()
    
    for subject in SUBJECT_LIST:

        # read in the images
        print('t+> {:5.3f} |Reading images for subject #: {}'.format(timer()-start_time, 
                                                                     subject))

        images = tsa.read_data(INPUT_FOLDER + '/' + subject + '.aps')

        # transpose so that the slice is the first dimension shape(16, 620, 512)
        images = images.transpose()

        # for each threat zone, loop through each image, mask off the zone and then crop it
        for tz_num, threat_zone_x_crop_dims in enumerate(zip(tsa.zone_slice_list, 
                                                             tsa.zone_crop_list)):

            threat_zone = threat_zone_x_crop_dims[0]
            crop_dims = threat_zone_x_crop_dims[1]

            # get label
            label = np.array(tsa.get_subject_zone_label(tz_num, 
                             tsa.get_subject_labels(STAGE1_NEW_LABELS, subject)))

            for img_num, img in enumerate(images):

                print('Threat Zone:Image -> {}:{}'.format(tz_num, img_num))
                print('Threat Zone Label -> {}'.format(label))
                
                if threat_zone[img_num] is not None:

                    # correct the orientation of the image
                    print('-> reorienting base image') 

                    base_img = np.flipud(img)
                    
                    print('-> shape {}|mean={}'.format(base_img.shape, 
                                                       base_img.mean()))

                    # convert to grayscale
                    print('-> converting to grayscale')
                    
                    rescaled_img = tsa.convert_to_grayscale(base_img)

                    print('-> shape {}|mean={}'.format(rescaled_img.shape, 
                                                       rescaled_img.mean()))

                    # spread the spectrum to improve contrast
                    print('-> spreading spectrum')
                    
                    #high_contrast_img = tsa.spread_spectrum(rescaled_img)
                    
                    high_contrast_img = spread_spectrum(rescaled_img)
                    print('-> shape {}|mean={}'.format(high_contrast_img.shape,
                                                       high_contrast_img.mean()))

                    # get the masked image
                    print('-> masking image')
                    masked_img = tsa.roi(high_contrast_img, threat_zone[img_num])
                    print('-> shape {}|mean={}'.format(masked_img.shape, 
                                                       masked_img.mean()))

                    # crop the image
                    print('-> cropping image')
                    cropped_img = tsa.crop(masked_img, crop_dims[img_num])
                    print('-> shape {}|mean={}'.format(cropped_img.shape, 
                                                       cropped_img.mean()))

                    # normalize the image
                    print('-> normalizing image')
                    
                    normalized_img = tsa.normalize(cropped_img)

                    print('-> shape {}|mean={}'.format(normalized_img.shape, 
                                                       normalized_img.mean()))

                    # zero center the image
                    print('-> zero centering')
                    
                    zero_centered_img = tsa.zero_center(normalized_img)
                    
                    #high_contrast_img = tsa.spread_spectrum(rescaled_img)
                    
                    print('-> shape {}|mean={}'.format(zero_centered_img.shape, 
                                                       zero_centered_img.mean()))

                    # append the features and labels to this threat zone's example array
                    print ('-> appending example to threat zone {}'.format(tz_num))

                    
                    threat_zone_examples.append([[tz_num], zero_centered_img, label])
                    
                    print ('-> shape {:d}:{:d}:{:d}:{:d}:{:d}:{:d}'.format(len(threat_zone_examples),len(threat_zone_examples[0]),len(threat_zone_examples[0][0]),len(threat_zone_examples[0][1][0]),len(threat_zone_examples[0][1][1]),len(threat_zone_examples[0][2])))
                else:
                    print('-> No view of tz:{} in img:{}. Skipping to next...'.format( 
                                tz_num, img_num))

        if ((len(threat_zone_examples) % (BATCH_SIZE * EXAMPLES_PER_SUBJECT)) == 0):
            for tz_num, tz in enumerate(tsa.zone_slice_list):

                tz_examples_to_save = []

                # write out the batch and reset
                print(' -> writing: ' + PREPROCESSED_NEW_DATA_FOLDER + 
                                        'preprocessed_new_TSA_scans-tz{}-{}-{}-b{}.npy'.format( 
                                        tz_num+1,
                                        len(threat_zone_examples[0][1][0]),
                                        len(threat_zone_examples[0][1][1]), 
                                        batch_num))

                # get this tz's examples
                
                tz_examples = [example for example in threat_zone_examples if example[0] == [tz_num]]

                # drop unused columns
                
                tz_examples_to_save.append([[features_label[1], features_label[2]] 
                                            for features_label in tz_examples])


                np.save(PREPROCESSED_NEW_DATA_FOLDER +  'preprocessed_new_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, len(threat_zone_examples[0][1][0]),len(threat_zone_examples[0][1][1]),batch_num), tz_examples_to_save)
                
                del tz_examples_to_save

            #reset for next batch 
            del threat_zone_examples
            threat_zone_examples = []
            batch_num += 1
    
    if (len(threat_zone_examples) > 0):
        for tz_num, tz in enumerate(tsa.zone_slice_list):

            
            tz_examples_to_save = []

            

            # write out the batch and reset
            
            print(' -> writing: ' + PREPROCESSED_NEW_DATA_FOLDER 
                    + 'preprocessed_new_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, 
                      len(threat_zone_examples[0][1][0]),
                      len(threat_zone_examples[0][1][1]),batch_num))

            # get this tz's examples
            
            tz_examples = [example for example in threat_zone_examples if example[0] == 
                           [tz_num]]

            # drop unused columns

            tz_examples_to_save.append([[features_label[1], features_label[2]] 
                                        for features_label in tz_examples])

            #save batch



            np.save(PREPROCESSED_NEW_DATA_FOLDER + 
                    'preprocessed_new_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, 
                                                     len(threat_zone_examples[0][1][0]),
                                                     len(threat_zone_examples[0][1][1]), 
                                                     batch_num), 
                                                     tz_examples_to_save)


# Generate the test data prepocessed folder
preprocess_tsa_test_data()


FILE_LIST = [f for f in os.listdir(PREPROCESSED_NEW_DATA_FOLDER)if re.search(re.compile('-tz' + str(THREAT_ZONE) + '-'), f)]

TRAIN_SET_FILE_LIST[0]

#f_in = FILE_LIST[1]
f_in = FILE_LIST[0]
# Generate all the features and labels for new test set

feature_batch, label_batch = gen_features_labels(f_in, PREPROCESSED_NEW_DATA_FOLDER)
len(feature_batch)

feature_batch = feature_batch.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)

# This gives us the aray for predictions for the given threatzone
x_predi = model.predict(feature_batch)

# Predictions printed after this call 
x_predi



