import models as md
import preprocessing as pp
import tensorflow as tf
import numpy as np
import cv2

pokemon = pp.load_data('pokemon/*.jpg')

generator = md.make_generator_model()
discriminator = md.make_discriminator_model()

md.train(generator, discriminator, pokemon, md.EPOCHS)
print('Done!')

for i in range(50):
    noise = tf.random.uniform([1, 100])
    generated_image = generator(noise, training=False)
    RGB_img = np.vectorize(pp.unnormPix)(generated_image[0].numpy()) 
    cv2.imwrite(f'generated/image_{i+1}.jpg', RGB_img)

