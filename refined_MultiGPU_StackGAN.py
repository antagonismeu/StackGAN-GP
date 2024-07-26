import os
from char_train import CharCnnRnn
from char_train_incre import CharCnnRnn as CharCnnRnnII 
try:
    import tensorflow as tf
    if tf.__version__.startswith('1'):
        raise ImportError("Please upgrade your TensorFlow to version 2.x")
    from tensorflow.keras.layers import *
    from tensorflow.keras import layers
    import pandas as pd
    import argparse
    from PIL import Image
    import numpy as np
except Exception as e:
    print(f"Error encountered during loading required packages: {e}")
    print('Attempt to reinitialize the interface.....')
    requirements = ['numpy', 'tensorflow', 'pandas', 'Pillow']
    for item in requirements :
        os.system(f'pip3 install {item}')
        print('Done!')

'''
technological upgrade: StackGAN-GP
'''

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)




global WIDTH, HEIGHT, CHANNEL
strategy = tf.distribute.MirroredStrategy()
width, height = 256, 256
assert width >= 128; height >= 128                                  # INFERIOR BOUNDARY : width, height = 128, 128  
WIDTH , HEIGHT = width, height
BATCH_SIZE = 32  
BATCH_SIZE_2 = 16
'''
Note: the relationship between GPUs and SIZE is complicated, 
meaning that larger size will guarantee the lower storing usage of GPUS, 
but increase the rate of occupation of GPU in the meantime 
'''                             
channel = 3
assert BATCH_SIZE >= 1; channel == 3; BATCH_SIZE_2 >= 1
CHANNEL = channel
GLOBAL_BATCH_SIZE = BATCH_SIZE * tf.distribute.MirroredStrategy().num_replicas_in_sync
GLOBAL_BATCH_SIZE_2 = BATCH_SIZE_2 * tf.distribute.MirroredStrategy().num_replicas_in_sync




def configuration() :
    module_path_1 = 'models/CharCNNRnn280.index'
    module_path_2 = 'models/checkpoint'
    module_path_3 = 'models/CharCNNRnn280.data-00000-of-00001'
    module_path_4 = 'models/CharCNNRnn150.data-00000-of-00001'
    module_path_5 = 'models/CharCNNRnn150.index'
    module_list = [module_path_1, module_path_2, module_path_3, module_path_4, module_path_5]
    for element in module_list :
        flag = os.path.exists(element)
        if flag :
            continue
        raise FileNotFoundError(f"Insufficient Resource. The required file '{element}' does not exist.")
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./samples', exist_ok=True)
    os.makedirs('./StageI', exist_ok=True)





class CA(tf.keras.Model):
    def __init__(self, output_dim, char):
        super(CA, self).__init__()
        self.Char = char
        self.fc = layers.Dense(output_dim * 2)
        self.leaky_relu = layers.LeakyReLU(alpha=0.2)

    def call(self, x):
        x_ = self.Char(x, training=False)
        y = self.fc(x_)
        y = self.leaky_relu(y)
        mean, logvar = tf.split(y, num_or_size_splits=2, axis=1)
        return mean, logvar, x_




class CA2(tf.keras.Model):
    def __init__(self, output_dim, char):
        super(CA2, self).__init__()
        self.Char = char
        self.fc = layers.Dense(output_dim * 2)
        self.leaky_relu = layers.LeakyReLU(alpha=0.2)

    def call(self, x):
        x_ = self.Char(x, training=False)
        y = self.fc(x_)
        y = self.leaky_relu(y)
        mean, logvar = tf.split(y, num_or_size_splits=2, axis=1)
        return mean, logvar, x_




class HierarchicalAttention(tf.keras.Model):
    def __init__(self, d_model, num_heads=8, num_layers=3, affiliation=512, width = WIDTH // 16, height = HEIGHT // 16):
        super(HierarchicalAttention, self).__init__()
        channel = d_model + affiliation
        self.dense = layers.Dense(channel)
        self.concate = layers.Concatenate(axis=-1)
        self.reshape = layers.Reshape(((width * height, channel)))
        self.inversed_reshape = layers.Reshape((width, height, channel))
        self.attention_layers = [layers.MultiHeadAttention(num_heads, channel) for _ in range(num_layers)]

    def call(self, inputs):
        txt_feat, img_feat = inputs
        
        combined_features = self.concate([txt_feat, img_feat]) 
        combined_features = self.dense(combined_features)
        combined_features = self.reshape(combined_features) 
        
        for attention_layer in self.attention_layers:
            combined_features = attention_layer(combined_features, combined_features)  
        output = self.inversed_reshape(combined_features)
        return output




class ResidualBlock(tf.keras.Model):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.bn2 = layers.BatchNormalization()

    def call(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + residual


class StageI_Generator(tf.keras.Model):
    def __init__(self):
        super(StageI_Generator, self).__init__()
        self.fc1 = layers.Dense(128 * 8 * 4 * 4,use_bias=False)
        self.activation = layers.ReLU()
        
        self.upsampling1 = layers.UpSampling2D(size=(2,2))
        self.conv1 = layers.Conv2D(512,kernel_size=3,strides=1,padding='same',use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.ac1 = layers.ReLU()
        
        self.upsampling2 = layers.UpSampling2D(size=(2,2))
        self.conv2 = layers.Conv2D(256,kernel_size=3,strides=1,padding='same',use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.ac2 = layers.ReLU()

        self.upsampling3 = layers.UpSampling2D(size=(2,2))
        self.conv3 = layers.Conv2D(128,kernel_size=3,strides=1,padding='same',use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.ac3 = layers.ReLU()

        self.upsampling4 = layers.UpSampling2D(size=(2,2))
        self.conv4 = layers.Conv2D(64,kernel_size=3,strides=1,padding='same',use_bias=False)
        self.bn4 = layers.BatchNormalization()
        self.ac4 = layers.ReLU()

        self.conv5 = layers.Conv2D(32,kernel_size=3,strides=1,padding='same',use_bias=False)
        self.bn5 = layers.BatchNormalization()
        self.ac5 = layers.ReLU()

        self.conv6 = layers.Conv2D(3,kernel_size=3,strides=1,padding='same',use_bias=False)
        self.tanh = layers.Activation('tanh')
        
    def call(self, z):
        z = self.fc1(z)
        z = self.activation(z)
        z = tf.reshape(z, (-1, 4, 4, 128 * 8))
        z = self.upsampling1(z)
        z = self.conv1(z)
        z = self.bn1(z)
        z = self.ac1(z)
        z = self.upsampling2(z)
        z = self.conv2(z)
        z = self.bn2(z)
        z = self.ac2(z)
        z = self.upsampling3(z)
        z = self.conv3(z)
        z = self.bn3(z)
        z = self.ac3(z)
        z = self.upsampling4(z)
        z = self.conv4(z)
        z = self.bn4(z)
        z = self.ac4(z)
        z = self.conv5(z)
        z = self.bn5(z)
        z = self.ac5(z)
        z = self.conv6(z)
        return self.tanh(z)


class StageI_Discriminator(tf.keras.Model):
    def __init__(self, dimension):
        super(StageI_Discriminator, self).__init__()
        self.reshape = layers.Reshape((1, 1, dimension))                            
        self.tile = layers.Lambda(lambda x: tf.tile(x, [1, WIDTH // 64, HEIGHT // 64, 1]))

        self.conv1 = layers.Conv2D(32,kernel_size=(4,4),padding='same',strides=1,use_bias=False)
        self.ac1 = layers.LeakyReLU(alpha=0.2)
        
        self.conv2 = layers.Conv2D(64,kernel_size=(4,4),padding='same',strides=2,use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.ac2 = layers.LeakyReLU(alpha=0.2)

        self.conv3 = layers.Conv2D(128,kernel_size=(4,4),padding='same',strides=2,use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.ac3 = layers.LeakyReLU(alpha=0.2)

        self.conv4 = layers.Conv2D(256,kernel_size=(4,4),padding='same',strides=2,use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.ac4 = layers.LeakyReLU(alpha=0.2)

        self.conv5 = layers.Conv2D(512,kernel_size=(4,4),padding='same',strides=2,use_bias=False)
        self.bn4 = layers.BatchNormalization()
        self.ac5 = layers.LeakyReLU(alpha=0.2)

        self.conv6 = layers.Conv2D(512,kernel_size=1,padding='same',strides=1)
        self.bn5 = layers.BatchNormalization()
        self.ac6 = layers.LeakyReLU(alpha=0.2)

        self.concat = layers.Concatenate(axis=-1)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        img, aux_input = inputs
        x = self.conv1(img)
        x = self.ac1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.ac2(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.ac3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.ac4(x)
        x = self.conv5(x)
        x = self.bn4(x)
        x = self.ac5(x)
        aux = self.reshape(aux_input)
        aux = self.tile(aux)
        x = self.concat([x, aux])
        x = self.conv6(x)
        x = self.bn5(x)
        x= self.ac6(x)
        x = self.flatten(x)
        x = self.fc(x)        
        return x


class StageII_Generator(tf.keras.Model):
    def __init__(self, dimension):
        super(StageII_Generator, self).__init__()
        self.reshape = layers.Reshape((1, 1, dimension))                                  
        self.tile = layers.Lambda(lambda x: tf.tile(x, [1, WIDTH // 16, HEIGHT // 16, 1]))
        self.concat = Concatenate(axis=-1)
        self.h_att = HierarchicalAttention(dimension)
        self.conv1 = layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.ac1 = layers.ReLU()
        self.conv2 = layers.Conv2D(256, kernel_size=(4, 4), strides=2, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn1 = layers.BatchNormalization()
        self.ac2 = layers.ReLU()
        self.conv3 = layers.Conv2D(512, kernel_size=(4, 4), strides=2, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn2 = layers.BatchNormalization()
        self.ac3 = layers.ReLU()

        self.conv4 = layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn3 = layers.BatchNormalization()
        self.ac4 = layers.ReLU()

        self.conv5 = layers.Conv2D(1024, kernel_size=(3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn4 = layers.BatchNormalization()
        self.ac5 = layers.ReLU()

        self.rb1 = ResidualBlock(1024)
        self.rb2 = ResidualBlock(1024)
        self.rb3 = ResidualBlock(1024)
        self.rb4 = ResidualBlock(1024)
        
        self.upsampling1 = layers.UpSampling2D(size=(2, 2))
        self.conv6 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn5 = layers.BatchNormalization()
        self.ac6 = layers.ReLU()

        self.upsampling2 = layers.UpSampling2D(size=(2, 2))
        self.conv7 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn6 = layers.BatchNormalization()
        self.ac7 = layers.ReLU()

        self.upsampling3 = layers.UpSampling2D(size=(2, 2))
        self.conv8 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn7 = layers.BatchNormalization()
        self.ac8 = layers.ReLU()

        self.upsampling4 = layers.UpSampling2D(size=(2, 2))
        self.conv9 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn8 = layers.BatchNormalization()
        self.ac9 = layers.ReLU()

        self.conv10 = layers.Conv2D(3, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.tanh = layers.Activation('tanh')

        self.a = self.add_weight(
            shape=(BATCH_SIZE_2, WIDTH // 16, HEIGHT // 16, dimension + 512),       
            initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1),
            trainable=True,
            name='alpha'
        )


    def weight(self, inputs) :
        x, y = inputs
        x = self.a * x + (1 - self.a) * y 
        return x

    def call(self, inputs):
        c, img = inputs
        img = self.conv1(img)
        img = self.ac1(img)
        img = self.conv2(img)
        img = self.bn1(img)
        img = self.ac2(img)
        img = self.conv3(img)
        img = self.bn2(img)
        img = self.ac3(img)
        c = self.reshape(c)
        c = self.tile(c)
        y = self.concat([c, img])
        x = self.h_att([c, img])
        x = self.weight([x, y])
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.ac4(x)
        x = self.conv5(x)
        x = self.bn4(x)
        x = self.ac5(x)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.upsampling1(x)
        x = self.conv6(x)
        x = self.bn5(x)
        x = self.ac6(x)
        x = self.upsampling2(x)
        x = self.conv7(x)
        x = self.bn6(x)
        x = self.ac7(x)
        x = self.upsampling3(x)
        x = self.conv8(x)
        x = self.bn7(x)
        x = self.ac8(x)
        x = self.upsampling4(x)
        x = self.conv9(x)
        x = self.bn8(x)
        x = self.ac9(x) 
        x = self.conv10(x)                       
        return self.tanh(x)



class StageII_Discriminator(tf.keras.Model):
    def __init__(self, dimension):
        super(StageII_Discriminator, self).__init__()
        self.reshape = layers.Reshape((1, 1, dimension))     
        self.h_att = HierarchicalAttention(dimension, affiliation=512, width = WIDTH // 64, height = HEIGHT // 64)                              
        self.tile = layers.Lambda(lambda x: tf.tile(x, [1, WIDTH // 64, HEIGHT // 64, 1]))

        self.conv1 = layers.Conv2D(64, kernel_size=(4, 4), strides=2, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.ac1 = layers.LeakyReLU(alpha=0.2)
        
        self.conv2 = layers.Conv2D(128, kernel_size=(4, 4), strides=2, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn1 = layers.BatchNormalization()
        self.ac2 = layers.LeakyReLU(alpha=0.2)

        self.conv3 = layers.Conv2D(256, kernel_size=(4, 4), strides=2, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn2 = layers.BatchNormalization()
        self.ac3 = layers.LeakyReLU(alpha=0.2)

        self.conv4 = layers.Conv2D(512, kernel_size=(4, 4), strides=2, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn3 = layers.BatchNormalization()
        self.ac4 = layers.LeakyReLU(alpha=0.2)

        self.conv5 = layers.Conv2D(1024, kernel_size=(4, 4), strides=2, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn4 = layers.BatchNormalization()
        self.ac5 = layers.LeakyReLU(alpha=0.2)

        self.conv6 = layers.Conv2D(2048, kernel_size=(4, 4), strides=2, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn5 = layers.BatchNormalization()
        self.ac6 = layers.LeakyReLU(alpha=0.2)

        self.conv7 = layers.Conv2D(1024, kernel_size=(1, 1), strides=1, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn6 = layers.BatchNormalization()
        self.ac7 = layers.LeakyReLU(alpha=0.2)

        self.conv8 = layers.Conv2D(512, kernel_size=(1, 1), strides=1, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn7 = layers.BatchNormalization()

        self.conv9 = layers.Conv2D(128, kernel_size=(1, 1), strides=1, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn8 = layers.BatchNormalization()
        self.ac8 = layers.LeakyReLU(alpha=0.2)

        self.conv10 = layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn9 = layers.BatchNormalization()
        self.ac9 = layers.LeakyReLU(alpha=0.2)

        self.conv11 = layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn10 = layers.BatchNormalization()
        self.ac10 = layers.LeakyReLU(alpha=0.2)

        self.add = layers.Add()
        self.conv12 = layers.Conv2D(64 * 8, kernel_size=1, strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn11 = layers.BatchNormalization()
        self.ac11 = layers.LeakyReLU(alpha=0.2)
        self.flatten = layers.Flatten()
        self.concat = layers.Concatenate(axis=-1)
        self.fc = layers.Dense(1, activation='sigmoid')


        self.a = self.add_weight(
            shape=(BATCH_SIZE_2, WIDTH // 64, HEIGHT // 64, dimension + 512),       
            initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1),
            trainable=True,
            name='alpha'
        )


    def weight(self, inputs) :
        x, y = inputs
        x = self.a * x + (1 - self.a) * y 
        return x



    def call(self, inputs):
        img, aux_input = inputs
        img = self.conv1(img)
        img = self.ac1(img)
        img = self.conv2(img)
        img = self.bn1(img)
        img = self.ac2(img)
        img = self.conv3(img)
        img = self.bn2(img)
        img = self.ac3(img)
        img = self.conv4(img)
        img = self.bn3(img)
        img = self.ac4(img)
        img = self.conv5(img)
        img = self.bn4(img)
        img = self.ac5(img)
        img = self.conv6(img)
        img = self.bn5(img)
        img = self.ac6(img)
        img = self.conv7(img)
        img = self.bn6(img)
        img = self.ac7(img)  
        img = self.conv8(img)
        img = self.bn7(img)
        img_ = self.conv9(img)
        img_ = self.bn8(img_)
        img_ = self.ac8(img_)
        img_ = self.conv10(img_)
        img_ = self.bn9(img_)
        img_ = self.ac9(img_)
        img_ = self.conv11(img_)
        img_ = self.bn10(img_)
        x = self.add([img, img_])  
        x = self.ac10(x)
        aux = self.reshape(aux_input)
        aux = self.tile(aux)
        z = self.concat([x, aux])
        z0 = self.h_att([x, aux])
        z = self.weight([z, z0]) 
        z = self.conv12(z)
        z = self.bn11(z)
        z = self.ac11(z)  
        z = self.flatten(z)
        z = self.fc(z)                                                   
        return z               


class StageI(tf.keras.Model):
    def __init__(self, output_dim, loss_fn, optimizer, char, erroneous_weight=5.3, gp_weight=10.0):
        super(StageI, self).__init__()
        self.generator = StageI_Generator()
        self.discriminator = StageI_Discriminator(output_dim)
        self.ca = CA(output_dim, char)
        self.cross_entropy = loss_fn
        self.generator_optimizer = optimizer
        self.discriminator_optimizer = optimizer
        self.g_erroneous_weight = erroneous_weight
        self.d_erroneous_weight = erroneous_weight 
        self.mean = 0.0
        self.stddev = 1.0       
        self.gp_weight = gp_weight
        self.generator.compile(optimizer=self.generator_optimizer, loss=self.cross_entropy)
        self.discriminator.compile(optimizer=self.generator_optimizer, loss=self.cross_entropy)
    
    def gradient_penalty(self, real_images, fake_images, embeddings):
        alpha = tf.random.uniform(shape=[real_images.shape[0], 1, 1, 1], minval=0., maxval=1.)
        alpha = tf.broadcast_to(alpha, real_images.shape)
        interpolated_images = alpha * real_images + (1 - alpha) * fake_images
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated_images)
            d_interpolated = self.discriminator([interpolated_images, embeddings], training=True)
        gradients = tape.gradient(d_interpolated, [interpolated_images])[0]
        gradients_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((gradients_l2 - 1.0) ** 2)
        return gradient_penalty

    def grant_weight(self, index):
        lower_bound = index / 4 if index > 0 else 0
        upper_bound = (index + 1) / 4 if index < BATCH_SIZE - 1 else np.inf
        
        cdf_lower = 0.5 * (1 + np.math.erf((lower_bound - self.mean) / (self.stddev * np.sqrt(2))))
        cdf_upper = 0.5 * (1 + np.math.erf((upper_bound - self.mean) / (self.stddev * np.sqrt(2))))
        
        interval_prob = (cdf_upper - cdf_lower) * 2
        return interval_prob

    def discriminator_loss(self, real_output, fake_output, wrong_outputs, real_images, fake_images, embeddings):
        real_loss = self.cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
        fake_loss = self.cross_entropy(tf.ones_like(fake_output) * 0.1, fake_output)

        wrong_losses = []
        for i, wrong_output in enumerate(wrong_outputs):
            weight = self.grant_weight(i)
            wrong_loss = self.cross_entropy(tf.ones_like(wrong_output) * 0.1, wrong_output)
            wrong_losses.append(weight * wrong_loss)

        total_wrong_loss = tf.reduce_sum(wrong_losses)
        gp = self.gradient_penalty(real_images, fake_images, embeddings)
        return real_loss + fake_loss + self.d_erroneous_weight * total_wrong_loss + self.gp_weight * gp

    def generator_loss(self, fake_output, wrong_outputs, mu, logvar):
        gen_loss = self.cross_entropy(tf.ones_like(fake_output) * 0.9, fake_output)

        wrong_losses = []
        for i, wrong_output in enumerate(wrong_outputs):
            weight = self.grant_weight(i)
            wrong_loss = self.cross_entropy(tf.ones_like(wrong_output) * 0.1, wrong_output)
            wrong_losses.append(weight * wrong_loss)

        total_wrong_loss = tf.reduce_sum(wrong_losses)
        kl_div = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
        return gen_loss + self.g_erroneous_weight * total_wrong_loss + kl_div

    def call(self, text_embeddings, real_images, noise_size):
        noise = tf.random.normal([real_images.shape[0], noise_size])
        mu, logvar, _ = self.ca(text_embeddings, training=True)
        c0 = mu + tf.exp(logvar * 0.5) * tf.random.normal(shape=mu.shape)
        c0_ = tf.concat([c0, noise], axis=1)
        generated_images = self.generator(c0_, training=True)
        real_output = self.discriminator([real_images, c0], training=True)
        fake_output = self.discriminator([generated_images, c0], training=True)

        d_wrong_outputs = []
        g_wrong_outputs = []
        for i in range(1, BATCH_SIZE + 1):
            wrong_output_1 = self.discriminator([real_images[i:], c0[:-i]], training=True)
            wrong_output_2 = self.discriminator([real_images[:i], c0[-i:]], training=True)
            wrong_output_concat_1_2 = tf.concat([wrong_output_1, wrong_output_2], axis=0)
            d_wrong_outputs.append(wrong_output_concat_1_2)

            wrong_output_3 = self.discriminator([generated_images[i:], c0[:-i]], training=True)
            wrong_output_4 = self.discriminator([generated_images[:i], c0[-i:]], training=True)
            wrong_output_concat_3_4 = tf.concat([wrong_output_3, wrong_output_4], axis=0)
            g_wrong_outputs.append(wrong_output_concat_3_4)

        return real_output, fake_output, d_wrong_outputs, g_wrong_outputs, mu, logvar, generated_images, c0

    def train_step(self, text_embeddings, real_images, noise_size):
        with tf.GradientTape(persistent=True) as tape:
            real_output, fake_output, d_wrong_output, g_wrong_output, mu, logvar, generated_images, embeddings = self(text_embeddings, real_images, noise_size)
            d_loss = self.discriminator_loss(real_output, fake_output, d_wrong_output, real_images, generated_images, embeddings)
            g_loss = self.generator_loss(fake_output, g_wrong_output, mu, logvar)
        gradients_of_generator = tape.gradient(g_loss, self.generator.trainable_variables + self.ca.trainable_variables)
        gradients_of_discriminator = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables + self.ca.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        return d_loss, g_loss


class StageII(tf.keras.Model):
    def __init__(self, output_dim, loss_fn, optimizer, char, GI, legacy_ca, noise_size, erroneous_weight=5.3, gp_weight=10.0):
        super(StageII, self).__init__()
        self.generator = StageII_Generator(output_dim)
        self.g1 = GI
        self.ca0 = legacy_ca
        self.noise_size = noise_size
        self.ca1 = CA2(output_dim, char)
        self.discriminator = StageII_Discriminator(output_dim)
        self.cross_entropy = loss_fn
        self.generator_optimizer = optimizer
        self.discriminator_optimizer = optimizer
        self.g_erroneous_weight = erroneous_weight
        self.d_erroneous_weight = erroneous_weight
        self.mean = 0.0
        self.stddev = 1.0
        self.gp_weight = gp_weight      
        self.generator.compile(optimizer=self.generator_optimizer, loss=self.cross_entropy)
        self.discriminator.compile(optimizer=self.generator_optimizer, loss=self.cross_entropy)
    
    def gradient_penalty(self, real_images, fake_images, embeddings):
        alpha = tf.random.uniform(shape=[real_images.shape[0], 1, 1, 1], minval=0., maxval=1.)
        alpha = tf.broadcast_to(alpha, real_images.shape)
        interpolated_images = alpha * real_images + (1 - alpha) * fake_images
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated_images)
            d_interpolated = self.discriminator([interpolated_images, embeddings], training=True)
        gradients = tape.gradient(d_interpolated, [interpolated_images])[0]
        gradients_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((gradients_l2 - 1.0) ** 2)
        return gradient_penalty

    def grant_weight(self, index):
        lower_bound = index / 4 if index > 0 else 0
        upper_bound = (index + 1) / 4 if index < BATCH_SIZE_2 - 1 else np.inf
        
        cdf_lower = 0.5 * (1 + np.math.erf((lower_bound - self.mean) / (self.stddev * np.sqrt(2))))
        cdf_upper = 0.5 * (1 + np.math.erf((upper_bound - self.mean) / (self.stddev * np.sqrt(2))))
        
        interval_prob = (cdf_upper - cdf_lower) * 2
        return interval_prob
    
    def discriminator_loss(self, real_output, fake_output, wrong_outputs, real_images, fake_images, embeddings):
        real_loss = self.cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
        fake_loss = self.cross_entropy(tf.ones_like(fake_output) * 0.1, fake_output)

        wrong_losses = []
        for i, wrong_output in enumerate(wrong_outputs):
            weight = self.grant_weight(i)
            wrong_loss = self.cross_entropy(tf.ones_like(wrong_output) * 0.1, wrong_output)
            wrong_losses.append(weight * wrong_loss)

        total_wrong_loss = tf.reduce_sum(wrong_losses)
        gp = self.gradient_penalty(real_images, fake_images, embeddings)
        return real_loss + fake_loss + self.d_erroneous_weight * total_wrong_loss + self.gp_weight * gp

    def generator_loss(self, fake_output, wrong_outputs, mu, logvar):
        gen_loss = self.cross_entropy(tf.ones_like(fake_output) * 0.9, fake_output)

        wrong_losses = []
        for i, wrong_output in enumerate(wrong_outputs):
            weight = self.grant_weight(i)
            wrong_loss = self.cross_entropy(tf.ones_like(wrong_output) * 0.1, wrong_output)
            wrong_losses.append(weight * wrong_loss)

        total_wrong_loss = tf.reduce_sum(wrong_losses)
        kl_div = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
        return gen_loss + self.g_erroneous_weight * total_wrong_loss + kl_div

    def train_step(self, text, real_images):
        with tf.GradientTape(persistent=True) as tape:
            noise = tf.random.normal([real_images.shape[0], self.noise_size])
            mu, logvar, _ = self.ca1(text, training=True)
            mu0, logvar0, _ = self.ca0(text, training=False)
            c0 = mu + tf.exp(logvar * 0.5) * tf.random.normal(shape=mu.shape)
            pre_c0 = mu0 + tf.exp(logvar0 * 0.5) * tf.random.normal(shape=mu0.shape)
            c0_ = tf.concat([pre_c0, noise], axis=1)
            preliminary_images = self.g1(c0_, training=False)
            generated_images = self.generator([c0, preliminary_images], training=True)
            real_output = self.discriminator([real_images, c0], training=True)
            fake_output = self.discriminator([generated_images, c0], training=True)
            d_wrong_outputs = []
            g_wrong_outputs = []
            for i in range(1, BATCH_SIZE_2 + 1):
                wrong_output_1 = self.discriminator([real_images[i:], c0[:-i]], training=True)
                wrong_output_2 = self.discriminator([real_images[:i], c0[-i:]], training=True)
                wrong_output_concat_1_2 = tf.concat([wrong_output_1, wrong_output_2], axis=0)
                d_wrong_outputs.append(wrong_output_concat_1_2)

                wrong_output_3 = self.discriminator([generated_images[i:], c0[:-i]], training=True)
                wrong_output_4 = self.discriminator([generated_images[:i], c0[-i:]], training=True)
                wrong_output_concat_3_4 = tf.concat([wrong_output_3, wrong_output_4], axis=0)
                g_wrong_outputs.append(wrong_output_concat_3_4)  
            d_loss = self.discriminator_loss(real_output, fake_output, d_wrong_outputs, real_images, generated_images, c0)
            g_loss = self.generator_loss(fake_output, g_wrong_outputs, mu, logvar)       
        gradients_of_generator = tape.gradient(g_loss, self.generator.trainable_variables + self.ca1.trainable_variables + self.g1.trainable_variables)
        gradients_of_discriminator = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables + self.ca1.trainable_variables + self.g1.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        return d_loss, g_loss



class DataProcessor:
    def __init__(self, description_file, image_directory, batch_size, height, width, max_len=256):
        self.description_file = description_file
        self.image_directory = image_directory
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.max_len = max_len
        
        self.df = pd.read_csv(description_file)
        self.descriptions = [desc.replace('"', '') for desc in self.df['description']]
        self.image_paths = [f"{image_directory}/{image_id}" for image_id in self.df['image_id']]
        
        self.tokenizer = None
        self.vocabulary = None

    def str_to_labelvec(self, string, max_str_len):
        string = string.lower()
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
        alpha_to_num = {k: v + 1 for k, v in zip(alphabet, range(len(alphabet)))}
        labels = tf.zeros(max_str_len, dtype=tf.int32)
        max_i = min(max_str_len, len(string))
        for i in range(max_i):
            char_index = alpha_to_num.get(string[i], alpha_to_num[' '])
            labels = tf.tensor_scatter_nd_update(labels, [[i]], [char_index])    
        return labels, alpha_to_num

    def labelvec_to_onehot(self, labels):
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        one_hot = tf.one_hot(labels, depth=71)    
        one_hot = one_hot[:, 1:]   
        one_hot = tf.transpose(one_hot, perm=[1, 0])    
        return one_hot

    def preparation_txt(self, string, max_str_len):
        labels, tokenizer = self.str_to_labelvec(string, max_str_len)
        one_hot = self.labelvec_to_onehot(labels)
        return one_hot, tokenizer

    def preprocess_image_I(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)  
        img = tf.image.resize(img, [self.width//4, self.height//4])
        img_array = tf.image.convert_image_dtype(img, tf.float32) / 127.5 - 1.0
        return img_array        

    def preprocess_image_II(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)  
        img = tf.image.resize(img, [self.width, self.height])
        img_array = tf.image.convert_image_dtype(img, tf.float32) / 127.5 - 1.0
        return img_array        

    def preprocess_text(self):
        txt_tensors = []
        for sentence in self.descriptions:
            tensor, tokenizer = self.preparation_txt(sentence, self.max_len)
            txt_tensors.append(tensor)
        text_dataset = tf.data.Dataset.from_tensor_slices(txt_tensors)
        self.tokenizer = tokenizer
        return text_dataset
    

    def postprocedure(self, img, path, signature):
        img = ((img + 1.0) * 127.5).numpy().astype(np.uint8)
        img = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(f'{path}/{signature}.png')



    def preprocedure(self):
        image_dataset_I = tf.data.Dataset.from_tensor_slices(self.image_paths).map(self.preprocess_image_I)
        image_dataset_II = tf.data.Dataset.from_tensor_slices(self.image_paths).map(self.preprocess_image_II)
        text_dataset = self.preprocess_text()
        
        self.vocabulary = len(self.tokenizer) + 1 
        
        dataset = tf.data.Dataset.zip((image_dataset_I, image_dataset_II, text_dataset))
        dataset = dataset.shuffle(buffer_size=max(len(self.df)+1, 1024), reshuffle_each_iteration=True).batch(self.batch_size)
        return dataset


    def validate(self, validate_descriptions, CA, CA2, G_I, G_II, noise_size, path, signature, stage2=True):
        try:
            subfolder = os.path.join(path, str(signature))
            os.makedirs(subfolder, exist_ok=True)           

            for idx, sentence in enumerate(validate_descriptions):
                cluster = sentence
                one_hot, _ = self.preparation_txt(cluster, self.max_len)
                one_hot = tf.expand_dims(one_hot, axis=0) 
                mu, logvar, _ = CA(one_hot)
                noise = tf.random.normal([1, noise_size])
                c0 = mu + tf.exp(logvar * 0.5) * tf.random.normal(shape=mu.shape)
                c0_ = tf.concat([c0, noise], axis=1)
                generated_images = G_I(c0_)
                
                if stage2:
                    mu, logvar, _ = CA2(one_hot)
                    c0 = mu + tf.exp(logvar * 0.5) * tf.random.normal(shape=mu.shape)
                    generated_images = G_II([c0, generated_images])
                
                final_image = generated_images[0]
                nickname = f"GI_{idx + 1}"
                self.postprocedure(final_image, subfolder, nickname)
        except Exception as e:
            print(f"Error encountered during generating images from the given descriptions: {e}")







def main_stage1(latent_dim, flag, path) :
    print('''
        -----------------------------
        ---Stage 1 Is Initialized-----  
        -----------------------------

    ''')
    configuration()
    coversion_log_path = './log/StageI.log'
    epochs_stage = 5000
    csv_path = 'descriptions.csv'
    images_path = './images'
    noise_size = 200
    learning_rate = 2e-4
    save_path = './StageI'
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=2650,
        decay_rate=0.96,
        staircase=True
    )
    save_interval = 50

    with strategy.scope() :
        load_dataset = DataProcessor(csv_path, images_path, GLOBAL_BATCH_SIZE, height, width)
        dataset = load_dataset.preprocedure()
        optimizer_ = tf.keras.optimizers.legacy.RMSprop(learning_rate=lr_schedule)
        '''
        Warning:
        if the previous version is implemented under tf(2.11)(not include 2.11)
        the restoring line should be modified like this tf.keras.optimizers.legacy.RMSprop
        '''
        char = CharCnnRnn(optimizer_)
        char.load_weights('models/CharCNNRnn280')
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.5)
        s1 = StageI(latent_dim, cross_entropy, optimizer, char)
        if flag :
            s1.generator.load_weight(f'modles/{path[1]}')
            s1.discriminator.load_weight(f'modles/{path[2]}')
            s1.ca.load_weight(f'modles/{path[0]}')
        s1.compile(optimizer=optimizer, loss=cross_entropy)



    print(f'Number of available GPUs: {strategy.num_replicas_in_sync}')


    @tf.function
    def train_step_stage1(batch) :
        shrinked_target, _, text = batch
        print(shrinked_target.shape, text.shape)
        d_loss, g_loss = s1.train_step(text, shrinked_target, noise_size)
        return g_loss, d_loss  


    @tf.function
    def distributed_train_stage1(datum) :
        g_loss, d_loss = strategy.run(train_step_stage1, args=(datum,))
        g_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, g_loss, axis=None)
        d_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, d_loss, axis=None)
        return g_loss, d_loss




    for epoch in range(epochs_stage) :
        sparse_tensorized_data = strategy.experimental_distribute_dataset(dataset)
        iterator = iter(sparse_tensorized_data)

        num, total_g_losses, total_d_losses = 0, 0, 0
        for num_, batch in enumerate(iterator):
            g_loss, d_loss = distributed_train_stage1(batch)
            num += 1
            total_g_losses += g_loss
            total_d_losses += d_loss
            print(f'per_batch_GI_loss:{g_loss} per_batch_DI_loss:{d_loss} epoch:{epoch + 1} batch_index:{num_+1}')
        train_g_loss = total_g_losses / num
        train_d_loss = total_d_losses / num


        print(f'Epoch {epoch + 1}/{epochs_stage}, Generator_I Loss: {train_g_loss.numpy()} Discriminator_I Loss: {train_d_loss.numpy()}')
        with open(coversion_log_path, 'a') as log_file:
            log_file.write(f'Epoch {epoch + 1}/{epochs_stage}, Generator_I Loss: {train_g_loss.numpy()} Discriminator_I Loss: {train_d_loss.numpy()}\n')

        if (epoch + 1) % save_interval == 0 or epoch == epochs_stage - 1:
            sentences_group = [
                'a pixel art character with black glasses, a toothbrush-shaped head and a redpinkish-colored body on a warm background',
                'a pixel art character with square yellow and orange glasses, a beer-shaped head and a gunk-colored body on a cool background'
            ]
            load_dataset.validate(sentences_group, s1.ca, None, s1.generator, None, noise_size, save_path, f'N{epoch + 1}', False)
            s1.ca.save_weights(f'models/CA{epoch + 1}')
            s1.ca.save_weights(f'models/CA_backup')
            s1.generator.save_weights(f'models/G1{epoch + 1}')
            s1.generator.save_weights(f'models/G1_backup')
            s1.discriminator.save_weights(f'models/D1{epoch + 1}')

    print('''
        -----------------------------
        ---Stage 1 Is Terminated-----  
        -----------------------------

    ''')
    return s1.ca, s1.generator


def main_stage2(latent_dim, ca, g1, flag, path) :
    print('''
        -----------------------------
        ---Stage 2 Is Initialized-----  
        -----------------------------

    ''')
    configuration()
    coversion_log_path = './log/StageII.log'
    epochs_stage = 500
    csv_path = 'descriptions.csv'
    images_path = './images'
    noise_size = 200
    learning_rate = 2e-4
    save_path = './samples'
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=2650,
        decay_rate=0.96,
        staircase=True
    )    
    save_interval = 30

    with strategy.scope() :
        optimizer_ = tf.keras.optimizers.legacy.RMSprop(learning_rate=lr_schedule)
        '''
        Warning:
        if the previous version is implemented under tf(2.11)(not include 2.11)
        the restoring line should be modified like this tf.keras.optimizers.legacy.RMSprop
        '''
        char = CharCnnRnnII(optimizer_)
        char.load_weights('models/CharCNNRnn150')        
        load_dataset = DataProcessor(csv_path, images_path, GLOBAL_BATCH_SIZE_2, height, width)
        dataset = load_dataset.preprocedure()
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.5)
        s2 = StageII(latent_dim, cross_entropy, optimizer, char, g1, ca ,noise_size)
        if flag :
            s2.generator.load_weights(f'models/{path[1]}')
            s2.discriminator.load_weights(f'models/{path[2]}')
            s2.ca1.load_weights(f'models/{path[0]}')
        s2.compile(optimizer=optimizer, loss=cross_entropy)



    print(f'Number of available GPUs: {strategy.num_replicas_in_sync}')


    @tf.function
    def train_step_stage2(batch) :
        _, final_target, text = batch
        print(text.shape, final_target.shape)
        d_loss, g_loss = s2.train_step(text, final_target)
        return g_loss, d_loss  


    @tf.function
    def distributed_train_stage2(datum) :
        g_loss, d_loss = strategy.run(train_step_stage2, args=(datum,))
        g_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, g_loss, axis=None)
        d_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, d_loss, axis=None)
        return g_loss, d_loss




    for epoch in range(epochs_stage) :
        sparse_tensorized_data = strategy.experimental_distribute_dataset(dataset)
        iterator = iter(sparse_tensorized_data)

        num, total_g_losses, total_d_losses = 0, 0, 0
        for num_, batch in enumerate(iterator):
            g_loss, d_loss = distributed_train_stage2(batch)
            num += 1
            total_g_losses += g_loss
            total_d_losses += d_loss
            print(f'per_batch_GII_loss:{g_loss} per_batch_DII_loss:{d_loss} epoch:{epoch + 1} batch_index:{num_+1}')
        train_g_loss = total_g_losses / num
        train_d_loss = total_d_losses / num


        print(f'Epoch {epoch + 1}/{epochs_stage}, Generator_II Loss: {train_g_loss.numpy()} Discriminator_II Loss: {train_d_loss.numpy()}')
        with open(coversion_log_path, 'a') as log_file:
            log_file.write(f'Epoch {epoch + 1}/{epochs_stage}, Generator_II Loss: {train_g_loss.numpy()} Discriminator_II Loss: {train_d_loss.numpy()}\n')

        if (epoch + 1) % save_interval == 0 or epoch == epochs_stage - 1 :
            sentences_group = [
                'a pixel art character with black glasses, a toothbrush-shaped head and a redpinkish-colored body on a warm background',
                'a pixel art character with square yellow and orange glasses, a beer-shaped head and a gunk-colored body on a cool background'
            ]
            load_dataset.validate(sentences_group, ca, s2.ca1, s2.g1, s2.generator, noise_size, save_path, f'N{epoch + 1}')
            s2.generator.save_weights(f'models/G2{epoch + 1}')
            s2.ca1.save_weights(f'models/CA2{epoch + 1}')
            s2.discriminator.save_weights(f'models/D2{epoch + 1}')

    print('''
        -----------------------------
        ---Stage 2 Is Terminated-----  
        -----------------------------

    ''')




def main(flag1, flag2, path1, path2, mode="restart"):
    latent_dim = 1500 
    latent_dim_2 = 2503


    def load_state(flag1, path1):
        try:
            initial_learning_rate = 0.001
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=2650,
                decay_rate=0.96,
                staircase=True
            )    
            optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=lr_schedule) 
            '''
            Warning:
            if the previous version is implemented under tf(2.11)(not include 2.11)
            the restoring line should be modified like this tf.keras.optimizers.legacy.RMSprop
            '''             
            char = CharCnnRnn(optimizer)
            char.load_weights('models/CharCNNRnn280')
            if not flag1 :
                ca = CA(latent_dim, char)
                ca.load_weights('./models/CA_backup')
                g1 = StageI_Generator()
                g1.load_weights('./models/G1_backup') 
            if flag1 :
                ca = CA(latent_dim, char)
                ca.load_weights(f'./models/{path1[0]}')
                g1 = StageI_Generator()
                g1.load_weights(f'./models/{path1[1]}')                          
            return ca, g1                       
        except Exception as e:
            print(f"Error encountered during reactivating repository: {e}")
            return None



    if mode == 'restart':
        ca, g1 = main_stage1(latent_dim, flag1, path1)
        main_stage2(latent_dim_2, ca, g1, flag2, path2)
    elif mode == 'recover':
        ca, g1 = load_state(flag1, path1)
        main_stage2(latent_dim_2, ca, g1, flag2, path2)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main stages of the program.")
    parser.add_argument("mode", type=str, choices=["restart", "recover"],
                        help="Mode to run the program. 'train' for training mode and 'recover' for recovery mode.", nargs='?', default='restart')
    parser.add_argument('--flag1', action='store_true', help='A flag to recover training StageI')
    parser.add_argument('-path1', type=str, nargs=3, default=None, help='if flag is True, add three arguments CA_path, G1_path, D1_path')
    parser.add_argument('--flag2', action='store_true', help='A flag to recover training StageII')
    parser.add_argument('-path2', type=str, nargs=3, default=None, help='if flag is True, add three arguments CA2_path, G2_path, D2_path')
    args = parser.parse_args()
    main(args.flag1, args.flag2, args.path1, args.path2, args.mode)