import os
import csv 
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation , Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Cropping2D, Lambda
from keras.utils import np_utils
from keras.models import Model
from random import shuffle
import matplotlib.pyplot as plt
import sklearn

# Download the dataset from the following location
# https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip 

samples = []
## Read the CSV File
with open ('./data/driving_log.csv') as csv_file :
    reader = csv.reader(csv_file)
    for sample in reader: 
        samples.append(sample)

# Delete the first element which is the header name of each column
del samples[0]

print(len(samples))

## Split the training sample and validationn sample 
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

## Generator for loading and preprocessing the image
def generator (samples, batch_size=128) :
    num_samples = len(samples)
    shuffle(samples)
    while 1 : # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size) :
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples :

                # For each sample, 6 images are added to the dataset
                # 1. centre_image, 
                # 2. flipped_centre_image
                # 3. left_image, 
                # 4. flipped_left_image
                # 5. right_image, 
                # 6. flipped_right_image

                for i in range(3) :
                        source_path = batch_sample[i]
                        filename = os.path.basename(source_path)
                        complete_filename = './data/IMG/' + filename
                
                        # Read the image
                        image = cv2.imread(complete_filename)
                
                        # Based on the image(center/left/right), compute the steering angle
                        # For left image , add correction value 0.2
                        # For right image, subract correction value of 0.2
                        angle = float(batch_sample[3])
                        angle += 0.2 if (i==1) else angle  # Left image
                        angle -= 0.2 if (i==2) else angle  # Right Image

                        # Add to the image and angle
                        images.append(image)
                        angles.append(angle)
                        
                        # Data Augmentation: Flipping images 
                        # Flip the image and angle and add to the dataset
                        images.append(cv2.flip(image,1))
                        angles.append(angle*-1.0)

                X_train = np.array(images)
                Y_train = np.array(angles)
                #print (X_train.shape)
                #print (Y_train.shape)
                yield sklearn.utils.shuffle(X_train, Y_train)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


#ch, row, col = 3, 80, 320
ch, row, col = 3, 160, 320

# Create the model
model = Sequential()

# Normalize the image 
#model.add(Lambda(lambda x: x/255.0 - 0.5, 
#                input_shape=(ch, row, col), 
#                output_shape=(ch, row, col)))

model.add(Lambda(lambda x: x/255.0 - 0.5, 
                input_shape=(160, 320, 3), 
                output_shape=(160, 320, 3)))

# Corp the image
#model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Flatten())
model.add(Dense(1))

# Compile and fit the model
model.compile(loss='mse', optimizer='adam')
model.fit_generator( 
                    train_generator, 
                    samples_per_epoch=len(train_samples), 
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples), 
                    nb_epoch=3 )

## Save the model
model.save('model.h5')

