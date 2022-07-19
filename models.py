import tensorflow as tf

def make_generator_model():
    model = tf.keras.Sequential([
    
    tf.keras.layers.Dense(256 * 8 * 8 * 3, use_bias = False, input_shape = (500,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ELU(),
    
    tf.keras.layers.Reshape((3, 8, 8, 256)),
    
    tf.keras.layers.Conv3DTranspose(128, (1,2,2), strides = (1,2,2), input_shape = (3,8,8)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ELU(),
    
    
    tf.keras.layers.Conv3DTranspose(64, (1,2,2), strides = (1,2,2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ELU(),
    
    tf.keras.layers.Conv3DTranspose(1, (1,2,2), strides = (1,2,2), activation = 'tanh'),
    
    tf.keras.layers.Reshape((64, 64, 3))
    ])
    
    return model


def make_discriminator_model():
    model = tf.keras.Sequential([
    
    #first covolution and pooling layer
    tf.keras.layers.Conv2D(128, (4,4), input_shape = (64,64,3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ELU(),
    tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = 2),
    
    #second convolution and pooling layer
    tf.keras.layers.Conv2D(256, (4,4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ELU(),
    tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = 2),
    
    #third convolution and pooling layer
    tf.keras.layers.Conv2D(512, (4,4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ELU(),
    tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = 2),
    
    #classifying layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'elu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy()

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

generator_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adam()