# @article{,
# title= {AVA: A Large-Scale Database for Aesthetic Visual Analysis},
# keywords= {images, aesthetics, semantic, quality, AVA, DPChallenge},
# journal= {},
# author= {Naila Murray and Luca Marchesotti and Florent Perronnin},
# year= {},
# url= {},
# license= {},
# abstract= {Aesthetic Visual Analysis (AVA) contains over 250,000 images along with a rich variety of meta-data including a large number of aesthetic scores for each image, semantic labels for over 60 categories as well as labels related to photographic style for high-level image quality categorization.},
# superseded= {},
# terms= {}
# }

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load a pre-trained MobileNet model
base_model = MobileNet(weights='imagenet', include_top=False)

# Add new layers
x = GlobalAveragePooling2D()(base_model.output)
predictions = Dense(10, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Base model
# model.summary()

# Path of AVA dataset
ava_images_path = 'AVA_dataset/images/images.7z'

# Preprocess the images to match MobileNet's expected input
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
    validation_split=0.2, # Splitting the data into training (80%) and validation (20%)
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create data generators for training and validation
train_generator = train_datagen.flow_from_directory(
    ava_images_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical', # Adjust based on your task (binary, categorical, etc.)
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    ava_images_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical', # Adjust based on your task
    subset='validation'
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // 32
)

# Trained model
model.summary()
