import tensorflow as tf
import time
import diffAugment as da

EPOCHS = 5
noise_dim = 100
BATCH_SIZE = 32

def make_generator_model():
    model = tf.keras.Sequential([
    
    tf.keras.layers.Dense(1024 * 4 * 4, use_bias = False, input_shape = (noise_dim,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    
    tf.keras.layers.Reshape((4, 4, 1024)),
    
    tf.keras.layers.Conv2DTranspose(512, (5,5), strides = (2,2), padding = 'same', use_bias = False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    
    tf.keras.layers.Conv2DTranspose(256, (5,5), strides = (2,2), padding = 'same', use_bias = False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    
    tf.keras.layers.Conv2DTranspose(128, (5,5), strides = (2,2), padding = 'same', use_bias = False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    
    tf.keras.layers.Conv2DTranspose(3, (5,5), strides = (2,2), padding = 'same', use_bias = False, activation = 'tanh')
    ])
    
    return model


def make_discriminator_model():
    model = tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(8, (5,5), activation = 'relu', input_shape = (64, 64, 3)),
    tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = 2),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Conv2D(16, (5,5), activation = 'relu'),
    tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = 2),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Conv2D(32, (5,5), activation = 'relu'),
    tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = 2),
    tf.keras.layers.BatchNormalization(),
    
    #classifying layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy()
relu = tf.nn.relu
norm = tf.norms

def generator_loss(fake_output):
    return relu(tf.ones_like(fake_output)) - relu(fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
pol = 'color,translation,cutout'

@tf.function
def train_step(generator, discriminator, images):
    noise = tf.random.uniform([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      #real_output = discriminator(da.DiffAugment(images, policy = pol), training=True)
      #fake_output = discriminator(da.DiffAugment(generated_images, policy = pol), training=True)
      
      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(generator, discriminator, dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(generator, discriminator, image_batch)

    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))