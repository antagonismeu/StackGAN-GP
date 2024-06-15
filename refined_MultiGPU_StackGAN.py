import os
try:
    import tensorflow as tf
    if tf.__version__.startswith('1'):
        raise ImportError("Please upgrade your TensorFlow to version 2.x")
    from tensorflow.keras.layers import *
    from tensorflow.keras import layers
    from tensorflow.keras.models import Model
    from tensorflow.keras.initializers import HeNormal
    from tensorflow.keras.optimizers.legacy import Adam
    from tensorflow.keras.losses import MeanSquaredError
    import pandas as pd
    import pickle, argparse, glob
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
width, height = 256, 256
assert width >= 128, height >= 128                                  #INFERIOR BOUNDARY : width, height = 128, 128  
WIDTH , HEIGHT = width, height
BATCH_SIZE = 64  
'''
Note: the relationship between GPUs and SIZE is complicated, 
meaning that larger size will guarantee the lower storing usage of GPUS, 
but increase the rate of occupation of GPU in the meantime 
'''                             
channel = 3
assert BATCH_SIZE >= 1, channel == 3
CHANNEL = channel
GLOBAL_BATCH_SIZE = BATCH_SIZE * tf.distribute.MirroredStrategy().num_replicas_in_sync




def configuration() :
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./samples', exist_ok=True)
    os.makedirs('./StageI', exist_ok=True)



class TextEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, gru_units):
        super(TextEncoder, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.gru = layers.GRU(gru_units)
    
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.gru(x)
        return x




class CA(tf.keras.Model):
    def __init__(self, output_dim, vocab_size, embedding_dim, gru_units):
        super(CA, self).__init__()
        self.text_encoder = TextEncoder(vocab_size, embedding_dim, gru_units)
        self.fc = layers.Dense(output_dim * 2)

    def call(self, x):
        x_ = self.text_encoder(x)
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
        c = tf.reshape(c[0], 1, 1, c[1])
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



class StageI(tf.keras.Model):
    def __init__(self, output_dim, loss_fn, optimizer, vocab_size, embedding_dim, gru_units):
        super(StageI, self).__init__()
        self.generator = StageI_Generator()
        self.discriminator = StageI_Discriminator()
        self.ca = CA(output_dim, vocab_size, embedding_dim, gru_units)
        self.cross_entropy = loss_fn
        self.generator_optimizer = optimizer
        self.discriminator_optimizer = optimizer
        self.generator.compile(optimizer=self.generator_optimizer, loss=self.cross_entropy)
        self.discriminator.compile(optimizer=self.generator_optimizer, loss=self.cross_entropy)
    
    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def generator_loss(self, fake_output, mu, logvar):
        gen_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        kl_div = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
        return gen_loss + kl_div
    

    def call(self, text_embeddings, real_images, noise_size) :
        noise = tf.random.normal([real_images.shape[0], noise_size])
        mu, logvar, embeddings = self.ca(text_embeddings)
        c0 = mu + tf.exp(logvar * 0.5) * tf.random.normal(shape=mu.shape)
        c0 = tf.concat([c0, noise], axis=1)
        generated_images = self.generator(c0, training=True)
        real_output = self.discriminator([real_images, embeddings], training=True)
        fake_output = self.discriminator([generated_images, embeddings], training=True)
        return real_output, fake_output, mu, logvar
    
    
    def train_step(self, text_embeddings, real_images, noise_size) :
        with tf.GradientTape(persistent=True) as tape:
            real_output, fake_output, mu, logvar = self(text_embeddings, real_images, noise_size)
            d_loss = self.discriminator_loss(real_output, fake_output)
            g_loss = self.generator_loss(fake_output, mu, logvar)
        gradients_of_generator = tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        return d_loss, g_loss



class StageII(tf.keras.Model):
    def __init__(self, loss_fn, optimizer, CAI, GI, noise_size):
        super(StageII, self).__init__()
        self.generator = StageII_Generator()
        self.g1 = GI
        self.noise_size = noise_size
        self.ca1 = CAI
        self.discriminator = StageII_Discriminator()
        self.cross_entropy = loss_fn
        self.generator_optimizer = optimizer
        self.discriminator_optimizer = optimizer
        self.generator.compile(optimizer=self.generator_optimizer, loss=self.cross_entropy)
        self.discriminator.compile(optimizer=self.generator_optimizer, loss=self.cross_entropy)
    
    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def generator_loss(self, fake_output, mu, logvar):
        gen_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        kl_div = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
        return gen_loss + kl_div
    

    def train_step(self, text, real_images):
        with tf.GradientTape(persistent=True) as tape:
            mu, logvar, embeddings = self.ca1(text, training=True)
            noise = tf.random.normal([BATCH_SIZE, self.noise_size])
            c0 = mu + tf.exp(logvar * 0.5) * tf.random.normal(shape=mu.shape)
            c0_ = tf.concat([c0, noise], axis=1)
            preliminary_images = self.g1(c0_, training=True)
            generated_images = self.generator([c0_, preliminary_images], training=True)
            real_output = self.discriminator([real_images, embeddings], training=True)
            fake_output = self.discriminator([generated_images, embeddings], training=True)
            d_loss = self.discriminator_loss(real_output, fake_output)
            g_loss = self.generator_loss(fake_output, mu, logvar)
        gradients_of_generator = tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        return d_loss, g_loss

        



def load_dataset(description_file, image_directory, batch_size, height, width):
    df = pd.read_csv(description_file)
    descriptions = [desc.replace('"', '') for desc in df['description']]
    image_paths = [f"{image_directory}/{image_id}" for image_id in df['image_id']]

    def preprocess_image_I(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)  
        img = tf.image.resize(img, [width//4, height//4])
        img_array = tf.image.convert_image_dtype(img, tf.float32) / 127.5 - 1.0
        return img_array        


    def preprocess_image_II(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)  
        img = tf.image.resize(img, [width, height])
        img_array = tf.image.convert_image_dtype(img, tf.float32) / 127.5 - 1.0
        return img_array        

    def preprocess_text(descriptions):
        global tokenizer
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")
        tokenizer.fit_on_texts(descriptions)
	
        voc_li = tokenizer.texts_to_sequences(descriptions)

        voc_li = tf.keras.preprocessing.sequence.pad_sequences(voc_li, padding="post")


        text_dataset = tf.data.Dataset.from_tensor_slices(voc_li)        
        return text_dataset, voc_li
        
    image_dataset_I = tf.data.Dataset.from_tensor_slices(image_paths).map(preprocess_image_I)
    image_dataset_II = tf.data.Dataset.from_tensor_slices(image_paths).map(preprocess_image_II)
    text_dataset, voc_li = preprocess_text(descriptions)
    vocabulary = len(tokenizer.word_index) + 1

    dataset = tf.data.Dataset.zip((image_dataset_I, image_dataset_II ,text_dataset))
    dataset = dataset.shuffle(buffer_size=max(len(df)+1, 1024), reshuffle_each_iteration=True).batch(batch_size)
    return dataset,  voc_li, vocabulary, tokenizer
    






def generate_images_from_text(descriptions, CA, G_I, G_II, noise_size, path, gross_range, signature, stage2=True):
    try:
        subfolder = os.path.join(path, str(signature))
        os.makedirs(subfolder, exist_ok=True)
        
        def postprocedure(img, path, signature):
            img = ((img + 1.0) * 127.5).numpy().astype(np.uint8)
            img = np.clip(img, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(f'{path}/{signature}.png')

        for idx, sentence in enumerate(descriptions):
            inputs = [tokenizer.word_index.get(i, 0) for i in sentence.split(" ")]  
            inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=gross_range, padding="post")
            inputs = tf.convert_to_tensor(inputs)

            mu, logvar, _ = CA(inputs)
            noise = tf.random.normal([1, noise_size])
            c0 = mu + tf.exp(logvar * 0.5) * tf.random.normal(shape=mu.shape)
            c0_ = tf.concat([c0, noise], axis=1)
            generated_images = G_I(c0_)
            
            if stage2:
                generated_images = G_II([c0, generated_images])
            
            final_image = generated_images[0]
            nickname = f"GI_{idx + 1}"
            postprocedure(final_image, subfolder, nickname)
    except Exception as e:
        print(f"Error encountered during generating images from the given descriptions: {e}")






def main_stage1(latent_dim) :
    print('''
        -----------------------------
        ---Stage 1 Is Initialized-----  
        -----------------------------

    ''')
    configuration()
    coversion_log_path = './log/StageI.log'
    strategy = tf.distribute.MirroredStrategy()
    epochs_stage = 200
    csv_path = 'descriptions.csv'
    images_path = './images'
    noise_size = 200
    learning_rate = 2e-4
    embedding_dim = 200
    gru_units = latent_dim
    save_path = './StageI'
    save_interval = 150

    with strategy.scope() :
        dataset, vocab, vocab_size, _  = load_dataset(csv_path, images_path, GLOBAL_BATCH_SIZE, height, width)
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.5)
        s1 = StageI(latent_dim, cross_entropy, optimizer, vocab_size, embedding_dim, gru_units)
        s1.compile(optimizer=optimizer, loss=cross_entropy)



    print(f'Number of available GPUs: {strategy.num_replicas_in_sync}')

    def max_length(vectors) :
        return max(len(vector) for vector in vectors)


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
            print(f'per_batch_GI_loss:{total_g_losses} per_batch_DI_loss:{total_d_losses} epoch:{epoch + 1} batch_index:{num_+1}')
        train_g_loss = total_g_losses / num
        train_d_loss = total_d_losses / num


        print(f'Epoch {epoch + 1}/{epochs_stage}, Generator_I Loss: {train_g_loss.numpy()} Discriminator_I Loss: {train_d_loss.numpy()}')
        with open(coversion_log_path, 'a') as log_file:
            log_file.write(f'Epoch {epoch + 1}/{epochs_stage}, Generator_I Loss: {train_g_loss.numpy()} Discriminator_I Loss: {train_d_loss.numpy()}\n')

        if (epoch + 1) % save_interval == 0 or epoch == epochs_stage :
            sentences_group = [
                'a pixel art character with black glasses, a toothbrush-shaped head and a redpinkish-colored body on a warm background',
                'a pixel art character with square yellow and orange glasses, a beer-shaped head and a gunk-colored body on a cool background'
            ]
            generate_images_from_text(sentences_group, s1.ca, s1.generator, None, noise_size, save_path, max_length(vocab), f'N{epoch + 1}', False)
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


def main_stage2(ca, g1) :
    print('''
        -----------------------------
        ---Stage 2 Is Initialized-----  
        -----------------------------

    ''')
    configuration()
    coversion_log_path = './log/StageII.log'
    strategy = tf.distribute.MirroredStrategy()
    epochs_stage = 200
    csv_path = 'descriptions.csv'
    images_path = './images'
    noise_size = 200
    learning_rate = 2e-4
    save_path = './samples'
    save_interval = 50

    with strategy.scope() :
        dataset, vocab, vocab_size, _ = load_dataset(csv_path, images_path, GLOBAL_BATCH_SIZE, height, width)
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.5)
        s2 = StageII(cross_entropy, optimizer, ca, g1, noise_size)
        s2.compile(optimizer=optimizer, loss=cross_entropy)



    print(f'Number of available GPUs: {strategy.num_replicas_in_sync}')


    def max_length(vectors) :
        return max(len(vector) for vector in vectors)


    @tf.function
    def train_step_stage2(batch) :
        _, final_target, text = batch
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
            print(f'per_batch_GII_loss:{total_g_losses} per_batch_DII_loss:{total_d_losses} epoch:{epoch + 1} batch_index:{num_+1}')
        train_g_loss = total_g_losses / num
        train_d_loss = total_d_losses / num


        print(f'Epoch {epoch + 1}/{epochs_stage}, Generator_II Loss: {train_g_loss.numpy()} Discriminator_II Loss: {train_d_loss.numpy()}')
        with open(coversion_log_path, 'a') as log_file:
            log_file.write(f'Epoch {epoch + 1}/{epochs_stage}, Generator_II Loss: {train_g_loss.numpy()} Discriminator_II Loss: {train_d_loss.numpy()}\n')

        if (epoch + 1) % save_interval == 0 or epoch == epochs_stage :
            sentences_group = [
                'a pixel art character with black glasses, a toothbrush-shaped head and a redpinkish-colored body on a warm background',
                'a pixel art character with square yellow and orange glasses, a beer-shaped head and a gunk-colored body on a cool background'
            ]
            generate_images_from_text(sentences_group, ca, g1, s2.generator, noise_size, save_path, max_length(vocab), f'N{epoch + 1}')
            s2.generator.save_weights(f'models/G2{epoch + 1}')
            s2.disciminator.save_weights(f'models/D2{epoch + 1}')

    print('''
        -----------------------------
        ---Stage 2 Is Terminated-----  
        -----------------------------

    ''')




def main(mode="restart"):
    latent_dim = 256 


    def load_state():
        try:
            ca = CA(latent_dim)
            ca.load_weights('./models/CA_backup')
            g1 = StageI_Generator()
            g1.load_weights('./models/G1_backup')            
            return ca, g1                       
        except Exception as e:
            print(f"Error encountered during reactivating repository: {e}")
            return None



    if mode == 'restart':
        ca, g1 = main_stage1(latent_dim)
        main_stage2(ca, g1)
    elif mode == 'recover':
        ca, g1 = load_state()
        main_stage2(ca, g1)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main stages of the program.")
    parser.add_argument("mode", type=str, choices=["restart", "recover"],
                        help="Mode to run the program. 'train' for training mode and 'recover' for recovery mode.")
    args = parser.parse_args()
    main(args.mode)