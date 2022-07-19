import numpy as np
import tensorflow as tf
import models as md
import time
import cv2

pokemon = tf.keras.utils.image_dataset_from_directory('pokemon_jpg', labels='inferred', image_size = (64, 64))

generator = md.make_generator_model()
discriminator = md.make_discriminator_model()


EPOCHS = 20

@tf.function
def train_step(images):
    noise = tf.random.normal([50, 500])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = md.generator_loss(fake_output)
      disc_loss = md.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    md.generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    md.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for images, labels in dataset:
      train_step(images)
      print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

def toPixel(x):
    return 127.5 * (x + 1)      

train(pokemon, EPOCHS)

for i in range(50):
    noise = tf.random.normal([1, 500])
    generated_image = generator(noise, training=False)
    RGB_img = np.vectorize(toPixel)(generated_image[0].numpy()) 
    cv2.imwrite(f'generated/image_{i+1}.jpg', RGB_img)
    




