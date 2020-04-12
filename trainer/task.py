"""Trains a Keras model to predict cancer class from other Skin Cancer data."""
import os
import argparse
from datetime import datetime

import tensorflow as tf

from util import download_unwrap
from model import (
    train_input_fn,
    valid_input_fn,
    create_keras_model
)

LESION_TYPES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
IMG_HEIGHT = 150
IMG_WIDTH = 150
NUM_CLASSES = 7


def get_args():
    """
    Argument parser.

    Returns: Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--kaggle-key',
        type=str,
        required=True,
        help='Path to kaggle JSON key file.')
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='Local or GCS location for writing checkpoints and exporting models')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=15,
        help='Number of times to go through the data, default=15')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='Number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float,
        help='Learning rate for gradient descent, default=.01')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()

    return args


def get_file_number(dir):
    return sum([len(files) for r, d, files in os.walk(dir)])


def create_class_weights(classes, train_dir):
    total_train = get_file_number(train_dir)

    class_weight = {}
    for index, a_class in enumerate(classes):
        total_class = get_file_number(os.path.join(train_dir, a_class))

        weight = (1 / total_class)*(total_train / 2.0)
        class_weight[index] = weight

    return class_weight


def train_and_evaluate(args):
    """
    Trains and evaluates the Keras model.
    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in util.py. Saves the trained model in TensorFlow SavedModel
    format to the path defined in part by the --job-dir argument.

    Args:
      args: dictionary of arguments - see get_args() for details
    """
    train_path, valid_path, test_path = download_unwrap(
        key_file=args.kaggle_key,
        classes=LESION_TYPES
    )

    total_train = get_file_number(train_path)
    total_valid = get_file_number(valid_path)

    keras_model = create_keras_model(
        num_classes=NUM_CLASSES,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH
    )

    training_dataset = train_input_fn(
        train_dir=train_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        classes=LESION_TYPES,
        shuffle=True
    )

    validation_dataset = valid_input_fn(
        valid_dir=valid_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        classes=LESION_TYPES
    )

    lr_decay_cb = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: args.learning_rate + 0.02 * (0.5 ** (1 + epoch)),
        verbose=True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        os.path.join(args.job_dir, 'keras_tensorboard'),
        histogram_freq=1)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        verbose=1,
        patience=2,
        mode='max',
        restore_best_weights=True
    )

    class_weight = create_class_weights(
        classes=LESION_TYPES, train_dir=train_path)

    keras_model.fit(
        training_dataset,
        steps_per_epoch=total_train // args.batch_size,
        epochs=args.num_epochs,
        validation_data=validation_dataset,
        validation_steps=total_valid // args.batch_size,
        callbacks=[early_stopping, lr_decay_cb, tensorboard_cb],
        class_weight=class_weight
    )

    dt_now = datetime.now().strftime("%d%m%Y_%H%M%S")
    export_path = os.path.join(args.job_dir, dt_now)
    keras_model.save(export_path)
    print('Model exported to: {}'.format(export_path))


if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
