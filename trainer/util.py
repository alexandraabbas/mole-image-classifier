"""Utilities to download and preprocess the Skin Cancer MNIST: HAM10000 data."""
import os
from shutil import copyfile, move, rmtree
import logging

import pandas as pd
import numpy as np

DIR_PATH = os.path.dirname(os.path.abspath(__file__))


def copy_key(key_file):
    os.mkdir('~/.kaggle')
    copyfile(key_file, '~/.kaggle/')


def download():
    import kaggle

    download_path = f'{DIR_PATH}/data/'

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        'kmader/skin-cancer-mnist-ham10000',
        path=download_path,
        unzip=True
    )

    return download_path


def unwrap(data_path, lesion_types):

    # create folders for train, validation and test sets
    for dataset in ['train', 'valid', 'test']:
        os.mkdir(f'{data_path}/{dataset}')
        for lesion_type in lesion_types:
            os.mkdir(f'{data_path}/{dataset}/{lesion_type}')

    metadata = pd.read_csv(f'{data_path}/HAM10000_metadata.csv')

    # move all images to relevant train folder
    for _, row in metadata.iterrows():
        try:
            src = f'{data_path}/HAM10000_images_part_1/{row["image_id"]}.jpg'
            dst = f'{data_path}/train/{row["dx"]}/{row["image_id"]}.jpg'
            copyfile(src, dst)
        except FileNotFoundError:
            src = f'{data_path}/HAM10000_images_part_2/{row["image_id"]}.jpg'
            dst = f'{data_path}/train/{row["dx"]}/{row["image_id"]}.jpg'
            copyfile(src, dst)

    # sample valid and test sets from all
    for lesion in lesion_types:
        train_path = f'{data_path}/train/{lesion}/'
        valid_path = f'{data_path}/valid/{lesion}/'
        test_path = f'{data_path}/test/{lesion}/'

        logging.info(f'Creating {lesion} valid and test dataset...')

        files = os.listdir(train_path)
        num_files = len(files)

        logging.info(f'Number of images for lesion: {num_files}')

        portion = 0.1
        k = int(num_files * portion)

        file_array = np.array(files)
        np.random.shuffle(file_array)
        valid_files = file_array[0:k]
        test_files = file_array[k + 1: 2*k]

        for f in valid_files:
            move(train_path + f, valid_path)

        for f in test_files:
            move(train_path + f, test_path)

    # remove unused folders and files
    rmtree(f'{data_path}/HAM10000_images_part_1/')
    rmtree(f'{data_path}/HAM10000_images_part_2/')
    rmtree(f'{data_path}/hmnist_28_28_L.csv')
    rmtree(f'{data_path}/hmnist_28_28_RGB.csv')
    rmtree(f'{data_path}/hmnist_8_8_RGB.csv')
    rmtree(f'{data_path}/hmnist_8_8_L.csv')

    return train_path, valid_path, test_path


def download_unwrap(key_file, classes):

    copy_key(key_file)

    data_path = download()
    train_path, valid_path, test_path = unwrap(data_path, classes)

    return train_path, valid_path, test_path
