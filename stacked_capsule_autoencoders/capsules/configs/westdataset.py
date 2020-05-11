import pathlib
import random

import numpy as np
import tensorflow.compat.v1 as tf
from absl import flags
from gym_miniworld.envs import WestWorld
from monty.collections import AttrDict

flags.DEFINE_string('data_dir', '~/westworld-data',
                    'Data directory for WestWorld dataset.')


def make_dataset(seed, obs_size, size=1000, reset_every=100, num_max_random_actions=10):
    env = WestWorld(
        seed=seed,
        obs_width=obs_size,
        obs_height=obs_size,
    )

    def fn():
        for i in range(size):
            if i % reset_every == 0:
                obs = env.reset()
            else:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                for j in range(random.randint(0, num_max_random_actions)):
                    obs, reward, done, info = env.step(action)

            yield obs

    return tf.data.Dataset.from_generator(
        fn,
        output_types=tf.uint8,
        output_shapes=(obs_size, obs_size, 3)
    )


def make_westworld_dataset(config):
    """Creates the WestWorld dataset."""
    obs_size = config.canvas_size
    batch_size = config.batch_size

    def postpro(batch_img):
        batch_img.set_shape((batch_size, obs_size, obs_size, 1))
        return {"image": batch_img,
                "label": tf.ones((batch_size,), dtype=tf.int64)}

    tds = make_dataset(seed=None, obs_size=obs_size, size=1000)
    tds = tds.map(lambda img: tf.image.rgb_to_grayscale(img))
    tds = tds.map(lambda img: tf.to_float(img) / 255.)
    tds = tds.repeat().batch(batch_size)
    tds = tds.map(postpro)

    vds = make_dataset(seed=None, obs_size=obs_size, size=100)
    vds = vds.map(lambda img: tf.image.rgb_to_grayscale(img))
    vds = vds.map(lambda img: tf.to_float(img) / 255.)
    vds = vds.repeat().batch(batch_size)
    vds = vds.map(postpro)

    res = AttrDict(
        trainset=tds,
        validset=vds
    )

    return res


def create_image_dataset_from_files(image_files, image_size=None, shuffle=False, buffer_size=1024):
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

    def postpro(batch_img):
        batch_img.set_shape((batch_size, obs_size, obs_size, 1))
        return {"image": batch_img,
                "label": tf.ones((batch_size,), dtype=tf.int64)}

    image_dir = data_dir / 'images'
    image_files = np.array(list(image_dir.glob('**/*.jpg')))
    indices = np.random.permutation(image_files.shape[0])
    train_idx, val_idx = indices[:80], indices[80:]
    train_files = image_files[train_idx]
    val_files = image_files[val_idx]

    tds = create_image_dataset_from_files(train_files,
                                          image_size=(obs_size, obs_size))
    tds = tds.map(lambda image: tf.image.rgb_to_grayscale(image))
    tds = tds.map(lambda image: tf.to_float(image) / 255.)
    tds = tds.repeat().batch(batch_size)
    tds = tds.map(postpro)

    vds = create_image_dataset_from_files(val_files,
                                          image_size=(obs_size, obs_size))
    vds = vds.map(lambda image: tf.image.rgb_to_grayscale(image))
    vds = vds.map(lambda image: tf.to_float(image) / 255.)
    vds = vds.repeat().batch(batch_size)
    vds = vds.map(postpro)

    res = AttrDict(
        trainset=tds,
        validset=vds
    )

    return res
