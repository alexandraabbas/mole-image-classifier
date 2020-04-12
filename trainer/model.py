"""Defines a Keras model and input function for training."""
import logging
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    Dropout,
    MaxPooling2D,
    GlobalAveragePooling2D
)


def train_input_fn(
    train_dir,
    num_epochs,
    batch_size,
    img_height,
    img_width,
    classes,
    shuffle=True,
):
    """
    Generates an input function to be used for model training.

    Args:
        train_dir: directory where train images are stored
        shuffle: boolean for whether to shuffle the data or not
        num_epochs: number of epochs to provide the data for
        batch_size: batch size for training
        img_height: target image height
        img_width:target image width
        classes: list of categorical labels
    Returns:
        ImageDataGenerator that can provide data to the Keras model
        for training
    """

    train_image_generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.5
    )

    train_dataset = train_image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=train_dir,
        shuffle=shuffle,
        target_size=(img_height, img_width),
        class_mode='categorical',
        classes=classes
    )

    return train_dataset


def valid_input_fn(
    valid_dir,
    num_epochs,
    batch_size,
    img_height,
    img_width,
    classes
):
    """
    Generates an input function to be used for model validation.

    Args:
        valid_dir: directory where validation images are stored
        num_epochs: number of epochs to provide the data for
        batch_size: batch size for training
        img_height: target image height
        img_width:target image width
        classes: list of categorical labels
    Returns:
        ImageDataGenerator that can provide data to the Keras model
        for validation
    """
    validation_image_generator = ImageDataGenerator(rescale=1./255)

    val_dataset = validation_image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=valid_dir,
        target_size=(img_height, img_width),
        class_mode='categorical',
        classes=classes
    )

    return val_dataset


def create_keras_model(num_classes, img_height, img_width):

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(
            img_height, img_width, 3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
    )

    logging.info(model.summary())

    return model
