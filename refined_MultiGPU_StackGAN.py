import os
from char_train import CharCnnRnn
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
    requirements = ['numpy', 'tensorflow', 'pandas', 'Pillow', 'transformers']
    for item in requirements :
        os.system(f'pip3 install {item}')
        print('Done!')



gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)




global WIDTH, HEIGHT, CHANNEL
strategy = tf.distribute.MirroredStrategy()
width, height = 256, 256
assert width >= 128; height >= 128                                  #INFERIOR BOUNDARY : width, height = 128, 128  
WIDTH , HEIGHT = width, height
BATCH_SIZE = 64  
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
    module_list = [module_path_1, module_path_2, module_path_3]
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

    def call(self, x):
        x_ = self.Char(x, training=False)
        y = self.fc(x_)
        mean, logvar = tf.split(y, num_or_size_splits=2, axis=1)
        return mean, logvar, x_


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
        self.fc = layers.Dense(4 * 4 * 64 * 8)
        self.upsample1 = layers.Conv2DTranspose(64 * 4, 4, strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.upsample2 = layers.Conv2DTranspose(64 * 2, 4, strides=2, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.upsample3 = layers.Conv2DTranspose(64, 4, strides=2, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.upsample4 = layers.Conv2DTranspose(3, 4, strides=2, padding='same')
        self.tanh = layers.Activation('tanh')

    def call(self, z):
        z = self.fc(z)
        z = tf.reshape(z, (-1, 4, 4, 64 * 8))
        z = self.upsample1(z)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.upsample2(z)
        z = self.bn2(z)
        z = self.relu(z)
        z = self.upsample3(z)
        z = self.bn3(z)
        z = self.relu(z)
        z = self.upsample4(z)
        return self.tanh(z)


class StageI_Discriminator(tf.keras.Model):
    def __init__(self):
        super(StageI_Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(64, 4, strides=2, padding='same')
        self.lrelu = layers.LeakyReLU(0.2)
        self.conv2 = layers.Conv2D(128, 4, strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(256, 4, strides=2, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(512, 4, strides=2, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.concat = layers.Concatenate()
        self.final_dense = layers.Dense(1)
        self.aux_dense = layers.Dense(512, activation='relu')

    def call(self, inputs):
        img, aux_input = inputs
        x = self.conv1(img)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.lrelu(x)
        x = self.flatten(x)
        aux = self.aux_dense(aux_input)
        x = self.concat([x, aux])
        x = self.final_dense(x)        
        return tf.nn.sigmoid(x)


class StageII_Generator(tf.keras.Model):
    def __init__(self):
        super(StageII_Generator, self).__init__()
        self.conv1 = layers.Conv2DTranspose(64, 4, strides=2, padding='same')
        self.lrelu = layers.LeakyReLU(0.2)
        self.conv2 = layers.Conv2DTranspose(128, 4, strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.resblock1 = ResidualBlock(128)
        self.resblock2 = ResidualBlock(128)
        self.resblock3 = ResidualBlock(128)
        self.resblock4 = ResidualBlock(128)
        self.deconv1 = layers.Conv2DTranspose(64, 4, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.deconv2 = layers.Conv2DTranspose(3, 4, strides=1, padding='same')
        self.tanh = layers.Activation('tanh')
        
    def spatial_replication(self, c, height, width):
        c = tf.expand_dims(tf.expand_dims(c, 1), 1)
        c = tf.tile(c, [1, height, width, 1])
        return c

    def call(self, inputs):
        c, img = inputs
        
        c = self.spatial_replication(c, img.shape[1], img.shape[2])
        
        x = tf.concat([img, c], axis=-1)
        
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.deconv2(x)
        return self.tanh(x)



class StageII_Discriminator(tf.keras.Model):
    def __init__(self):
        super(StageII_Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(64, 4, strides=2, padding='same')
        self.lrelu = layers.LeakyReLU(0.2)
        self.conv2 = layers.Conv2D(128, 4, strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(256, 4, strides=2, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(512, 4, strides=2, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.concat = layers.Concatenate()
        self.final_dense = layers.Dense(1)
        self.aux_dense = layers.Dense(512, activation='relu')

    def call(self, inputs):
        img, aux_input = inputs
        x = self.conv1(img)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.lrelu(x)
        x = self.flatten(x)
        aux = self.aux_dense(aux_input)
        x = self.concat([x, aux])
        x = self.final_dense(x)
        
        return tf.nn.sigmoid(x)



import tensorflow as tf

class StageI(tf.keras.Model):
    def __init__(self, output_dim, optimizer, char, gp_weight=10.0, n_critic=5):
        super(StageI, self).__init__()
        self.generator = StageI_Generator()
        self.discriminator = StageI_Discriminator()
        self.ca = CA(output_dim, char)
        self.generator_optimizer = optimizer
        self.discriminator_optimizer = optimizer
        self.gp_weight = gp_weight
        self.n_critic = n_critic  
        self.generator.compile(optimizer=self.generator_optimizer, loss=self.generator_loss)
        self.discriminator.compile(optimizer=self.discriminator_optimizer, loss=self.discriminator_loss)
    
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

    def discriminator_loss(self, real_output, fake_output, real_images, fake_images, embeddings):
        real_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        gp = self.gradient_penalty(real_images, fake_images, embeddings)
        return real_loss + self.gp_weight * gp

    def generator_loss(self, fake_output, mu, logvar):
        gen_loss = -tf.reduce_mean(fake_output)
        kl_div = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
        return gen_loss + kl_div

    def call(self, text_embeddings, real_images, noise_size):
        noise = tf.random.normal([real_images.shape[0], noise_size])
        mu, logvar, embeddings = self.ca(text_embeddings, training=True)
        c0 = mu + tf.exp(logvar * 0.5) * tf.random.normal(shape=mu.shape)
        c0 = tf.concat([c0, noise], axis=1)
        generated_images = self.generator(c0, training=True)
        real_output = self.discriminator([real_images, embeddings], training=True)
        fake_output = self.discriminator([generated_images, embeddings], training=True)
        return real_output, fake_output, mu, logvar, generated_images, embeddings

    def train_step(self, text_embeddings, real_images, noise_size):
        for _ in range(self.n_critic):
            with tf.GradientTape(persistent=True) as tape:
                real_output, fake_output, mu, logvar, generated_images, embeddings = self(text_embeddings, real_images, noise_size)
                d_loss = self.discriminator_loss(real_output, fake_output, real_images, generated_images, embeddings)
            gradients_of_discriminator = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        with tf.GradientTape(persistent=True) as tape:
            real_output, fake_output, mu, logvar, generated_images, embeddings = self(text_embeddings, real_images, noise_size)
            g_loss = self.generator_loss(fake_output, mu, logvar)
        gradients_of_generator = tape.gradient(g_loss, self.generator.trainable_variables + self.ca.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables + self.ca.trainable_variables))
        
        return d_loss, g_loss


class StageII(tf.keras.Model):
    def __init__(self, optimizer, CAI, GI, noise_size, gp_weight=10.0, n_critic=5):
        super(StageII, self).__init__()
        self.generator = StageII_Generator()
        self.g1 = GI
        self.noise_size = noise_size
        self.ca1 = CAI
        self.discriminator = StageII_Discriminator()
        self.generator_optimizer = optimizer
        self.discriminator_optimizer = optimizer
        self.gp_weight = gp_weight
        self.n_critic = n_critic  
        self.generator.compile(optimizer=self.generator_optimizer, loss=self.generator_loss)
        self.discriminator.compile(optimizer=self.discriminator_optimizer, loss=self.discriminator_loss)
    
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

    def discriminator_loss(self, real_output, fake_output, real_images, fake_images, embeddings):
        real_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        gp = self.gradient_penalty(real_images, fake_images, embeddings)
        return real_loss + self.gp_weight * gp

    def generator_loss(self, fake_output, mu, logvar):
        gen_loss = -tf.reduce_mean(fake_output)
        kl_div = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
        return gen_loss + kl_div

    def train_step(self, text, real_images):
        for _ in range(self.n_critic):
            with tf.GradientTape(persistent=True) as tape:
                mu, logvar, embeddings = self.ca1(text, training=True)
                noise = tf.random.normal([BATCH_SIZE_2, self.noise_size])
                c0 = mu + tf.exp(logvar * 0.5) * tf.random.normal(shape=mu.shape)
                c0_ = tf.concat([c0, noise], axis=1)
                preliminary_images = self.g1(c0_, training=True)
                generated_images = self.generator([c0_, preliminary_images], training=True)
                real_output = self.discriminator([real_images, embeddings], training=True)
                fake_output = self.discriminator([generated_images, embeddings], training=True)
                d_loss = self.discriminator_loss(real_output, fake_output, real_images, generated_images, embeddings)
            gradients_of_discriminator = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        with tf.GradientTape(persistent=True) as tape:
            mu, logvar, embeddings = self.ca1(text, training=True)
            noise = tf.random.normal([BATCH_SIZE_2, self.noise_size])
            c0 = mu + tf.exp(logvar * 0.5) * tf.random.normal(shape=mu.shape)
            c0_ = tf.concat([c0, noise], axis=1)
            preliminary_images = self.g1(c0_, training=True)
            generated_images = self.generator([c0_, preliminary_images], training=True)
            fake_output = self.discriminator([generated_images, embeddings], training=True)
            g_loss = self.generator_loss(fake_output, mu, logvar)
        gradients_of_generator = tape.gradient(g_loss, self.generator.trainable_variables + self.ca1.trainable_variables + self.g1.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables + self.ca1.trainable_variables + self.g1.trainable_variables))
        
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


    def validate(self, validate_descriptions, CA, G_I, G_II, noise_size, path, signature, stage2=True):
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
                    generated_images = G_II([c0_, generated_images])
                
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
    save_interval = 150

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
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.5)
        s1 = StageI(latent_dim, optimizer, char)
        if flag :
            s1.generator.load_weight(f'modles/{path[1]}')
            s1.discriminator.load_weight(f'modles/{path[2]}')
            s1.ca.load_weight(f'modles/{path[0]}')
        s1.compile(optimizer=optimizer)



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
            load_dataset.validate(sentences_group, s1.ca, s1.generator, None, noise_size, save_path, f'N{epoch + 1}', False)
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


def main_stage2(ca, g1, flag, path) :
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
    save_interval = 50

    with strategy.scope() :
        load_dataset = DataProcessor(csv_path, images_path, GLOBAL_BATCH_SIZE_2, height, width)
        dataset = load_dataset.preprocedure()
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.5)
        s2 = StageII(optimizer, ca, g1, noise_size)
        if flag :
            s2.generator.load_weights(f'models/{path[0]}')
            s2.discriminator.load_weights(f'models/{path[1]}')
        s2.compile(optimizer=optimizer)



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
            load_dataset.validate(sentences_group, s2.ca1, s2.g1, s2.generator, noise_size, save_path, f'N{epoch + 1}')
            s2.generator.save_weights(f'models/G2{epoch + 1}')
            s2.discriminator.save_weights(f'models/D2{epoch + 1}')

    print('''
        -----------------------------
        ---Stage 2 Is Terminated-----  
        -----------------------------

    ''')




def main(flag1, flag2, path1, path2, mode="restart"):
    latent_dim = 385 


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
        main_stage2(ca, g1, flag2, path2)
    elif mode == 'recover':
        ca, g1 = load_state(flag1, path1)
        main_stage2(ca, g1, flag2, path2)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main stages of the program.")
    parser.add_argument("mode", type=str, choices=["restart", "recover"],
                        help="Mode to run the program. 'train' for training mode and 'recover' for recovery mode.", nargs='?', default='restart')
    parser.add_argument('--flag1', action='store_true', help='A flag to recover training StageI')
    parser.add_argument('-path1', type=str, nargs=3, default=None, help='if flag is True, add three arguments CA_path, G1_path, D1_path')
    parser.add_argument('--flag2', action='store_true', help='A flag to recover training StageII')
    parser.add_argument('-path2', type=str, nargs=2, default=None, help='if flag is True, add three arguments G2_path, D2_path')
    args = parser.parse_args()
    main(args.flag1, args.flag2, args.path1, args.path2, args.mode)