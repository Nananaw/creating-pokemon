import tensorflow as tf
import models as md

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize_with_crop_or_pad(img, 64, 64)
    img = tf.cast(img, tf.float32)
    img = (img - 127.5) / 127.5
    return img

def load_data(path):
    images = tf.data.Dataset.list_files(path)
    images = images.map(load_image)
    images = images.batch(md.BATCH_SIZE)
    return images

def unnormPix(x):
    return 127.5 * (x + 1)
    




