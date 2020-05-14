import math
import pathlib

import numpy as np
import tensorflow.compat.v1 as tf
from absl import flags
from monty.collections import AttrDict

flags.DEFINE_string('data_dir', '~/westworld-data',
                    'Data directory for WestWorld dataset.')


def create_image_dataset_from_files(image_files,
                                    image_size=None,
                                    shuffle=False,
                                    buffer_size=1024):
    image_filepaths = [str(fp) for fp in image_files]

    ds = tf.data.Dataset.from_tensor_slices((image_filepaths,))
    if shuffle:
        ds = ds.shuffle(buffer_size)

    ds = ds.map(lambda image_fp: tf.image.decode_jpeg(tf.io.read_file(image_fp), channels=3))

    if image_size:
        image_size = image_size[0], image_size[1]
        ds = ds.map(lambda image: tf.image.resize(image, image_size))
    else:
        ds = ds.map(lambda image: tf.cast(image, tf.float32))

    return ds


def make_westworld_dataset_from_folder(config):
    data_dir = pathlib.Path(config.data_dir)
    obs_size = config.canvas_size
    batch_size = config.batch_size
    n_channels = config.n_channels

    def postpro(batch_img):
        batch_img.set_shape((batch_size, obs_size, obs_size, n_channels))
        return {"image": batch_img}

    image_dir = data_dir / 'images'
    image_files = np.array(list(image_dir.glob('**/*.jpg')))
    indices = np.random.permutation(image_files.shape[0])
    train_idx, val_idx = split(indices, (0.8, 0.2))
    train_files = image_files[train_idx]
    val_files = image_files[val_idx]

    tds = create_image_dataset_from_files(train_files,
                                          image_size=(obs_size, obs_size))
    if n_channels == 1:
        tds = tds.map(lambda image: tf.image.rgb_to_grayscale(image))
    tds = tds.map(lambda image: tf.to_float(image) / 255.)
    tds = tds.repeat().batch(batch_size)
    tds = tds.map(postpro)

    vds = create_image_dataset_from_files(val_files,
                                          image_size=(obs_size, obs_size))
    if n_channels == 1:
        vds = vds.map(lambda image: tf.image.rgb_to_grayscale(image))
    vds = vds.map(lambda image: tf.to_float(image) / 255.)
    vds = vds.repeat().batch(batch_size)
    vds = vds.map(postpro)

    res = AttrDict(
        trainset=tds,
        validset=vds
    )

    return res


def split(sequence, rates):
    N = len(sequence)
    total = sum(rates)
    ratios = [rate / total for rate in rates]

    break_points = [0]
    for ratio in ratios:
        break_points.append(break_points[-1] + ratio)

    indices = [math.floor(i * N) for i in break_points]
    indice_pairs = [(start, end) for start, end in zip(indices[:-1], indices[1:])]

    for start, end in indice_pairs:
        yield sequence[start:end]
