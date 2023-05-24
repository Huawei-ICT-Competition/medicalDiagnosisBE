import tensorflow as tf
import numpy as np
import os

# Optimizing TensorFlow performance to my current machine
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Building CNN layers (3 convolutional layers with max pooling, 1 dense layer, and a softmax output layer)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D(2, 2, ),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Using generator to input the training and validation images. Also, used image augmentation on the training data.
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
).flow_from_directory('huawei/train',
                      color_mode='grayscale',
                      target_size=(100, 100),
                      batch_size=128,
                      class_mode='categorical')

val_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
).flow_from_directory('huawei/val',
                      color_mode='grayscale',
                      target_size=(100, 100),
                      batch_size=16,
                      class_mode='categorical')

# Using checkpoints to save models
checkpoint = tf.keras.callbacks.ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss', verbose=0,
                                                save_best_only=True, mode='auto')

# Fitting the model on the data. Saved the model in a variable to compare results
history = model.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=[checkpoint])
