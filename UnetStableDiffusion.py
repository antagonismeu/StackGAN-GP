import os
try :
    import tensorflow as tf
    if tf.__version__.startswith('1'):
        raise ImportError("Please upgrade your TensorFlow to version 2.x")
    from tensorflow.keras.layers import *
    from tensorflow.keras import layers
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers.legacy import Adam
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from transformers import TFBertModel, BertTokenizer
    import pandas as pd
    from PIL import Image
    import numpy as np
except :
    requirements = ['numpy', 'tensorflow', 'pandas', 'Pillow', 'transformers']
    for item in requirements :
        os.system(f'pip3 install {item}')
        print('Done!')

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)


global WIDTH, HEIGHT
width, height = 128, 128
WIDTH , HEIGHT = width, height
BATCH_SIZE = 4







def configuration() :
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./samples', exist_ok=True)


class TextEncoder(tf.keras.Model):
    def __init__(self, vocab_size, output_dim=512, embed_dim=512):
        super(TextEncoder, self).__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.text_projection = Dense(output_dim, activation='relu')


    def call(self, input_ids):
        outputs = self.embedding(input_ids)
        text_embeddings = self.text_projection(outputs)
        return text_embeddings


class ImageEncoder(tf.keras.Model):
    def __init__(self, input_shape=(WIDTH, HEIGHT, 3), output_dim=512):
        super(ImageEncoder, self).__init__()
        self.conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape)
        self.pooling1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pooling2 = layers.MaxPooling2D((2, 2))
        self.conv3 = layers.Conv2D(256, (4, 4), activation='relu', padding='same')
        self.pooling3 = layers.MaxPooling2D((2, 2))
        self.conv4 = layers.Conv2D(512, (4, 4), activation='relu', padding='same')
        self.pooling4 = layers.MaxPooling2D((2, 2))
        self.conv5 = layers.Conv2D(1024, (4, 4), activation='relu', padding='same')
        self.pooling5 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.image_projection = layers.Dense(output_dim)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.pooling2(x)
        x = self.conv3(x)
        x = self.pooling3(x)
        x = self.conv4(x)
        x = self.pooling4(x)
        x = self.conv5(x)
        x = self.pooling5(x)
        x = self.flatten(x)
        latent_representation = self.image_projection(x)
        return latent_representation



class UNetDiffusionModule(tf.keras.Model):
    def __init__(self, num_batch, width, height, time_embedding_dim=128, text_embedding_dim=16):
        super(UNetDiffusionModule, self).__init__()
        self.batch_size = num_batch 
        self.width = width
        self.height = height          
        self.time_embedding = Embedding(input_dim=time_embedding_dim, output_dim=self.batch_size)
        self.text_projection = Dense(units=text_embedding_dim * self.batch_size)
        self.time_embedding_dim = time_embedding_dim
        self.text_embedding_dim = text_embedding_dim       

            
        self.final_conv = Conv2D(filters=3, kernel_size=(7, 7), strides=(1, 1), padding='same')
    
            
    def _build_unet_block(self, dim, dim2, width, height):
        inputs = Input(shape=(width, height, dim2 + dim * self.text_embedding_dim + self.time_embedding_dim))

        def conv_block(x, filters, kernel_size, strides=1, padding='same', activation='relu'):
            x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
            x = BatchNormalization()(x)
            x = Activation(activation)(x)
            return x

        def deconv_block(x, filters, kernel_size, strides=2, padding='same', activation='relu'):
            x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
            x = BatchNormalization()(x)
            x = Activation(activation)(x)
            return x

        def residual_block(x, filters, kernel_size=3, strides=1, padding='same', activation='relu'):
            x1 = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
            x1 = BatchNormalization()(x1)
            x1 = Activation(activation)(x1)
            x1 = Conv2D(filters, kernel_size, strides=1, padding='same')(x1)
            x1 = BatchNormalization()(x1)
            x = Add()([x, x1])
            x = Activation(activation)(x)
            return x

        conv1 = conv_block(inputs, 64, 4, 2)  
        conv2 = conv_block(conv1, 128, 4, 2)  
        conv3 = conv_block(conv2, 256, 4, 2)  

            
        conv3 = residual_block(conv3, 256)
        conv3 = residual_block(conv3, 256)

        conv4 = conv_block(conv3, 512, 4, 2)  
        conv5 = conv_block(conv4, 512, 4, 2)  
        conv6 = conv_block(conv5, 512, 4, 2)  
        conv7 = conv_block(conv6, 512, 4, 2)  

        deconv8 = deconv_block(conv7, 512, 4, 2)  
        deconv8 = Concatenate()([deconv8, conv6])

    
        deconv8 = residual_block(deconv8, 1024)
        deconv8 = residual_block(deconv8, 1024)

        deconv9 = deconv_block(deconv8, 512, 4, 2)  
        deconv9 = Concatenate()([deconv9, conv5])

        
        deconv9 = residual_block(deconv9, 1024)
        deconv9 = residual_block(deconv9, 1024)

        deconv10 = deconv_block(deconv9, 512, 4, 2)  
        deconv10 = Concatenate()([deconv10, conv4])


        deconv10 = residual_block(deconv10, 1024)
        deconv10 = residual_block(deconv10, 1024)

        deconv11 = deconv_block(deconv10, 256, 4, 2) 
        deconv11 = Concatenate()([deconv11, conv3])

        
        deconv11 = residual_block(deconv11, 512)
        deconv11 = residual_block(deconv11, 512)

        deconv12 = deconv_block(deconv11, 128, 4, 2)  
        deconv12 = Concatenate()([deconv12, conv2])

        
        deconv12 = residual_block(deconv12, 256)
        deconv12 = residual_block(deconv12, 256)

        deconv13 = deconv_block(deconv12, 64, 4, 2)  
        deconv13 = Concatenate()([deconv13, conv1])

        
        deconv13 = residual_block(deconv13, 128)
        deconv13 = residual_block(deconv13, 128)

        deconv14 = deconv_block(deconv13, 64, 4, 2)

        outputs = Conv2D(3, kernel_size=7, strides=1, padding='same', activation='sigmoid')(deconv14)
        generator = Model(inputs=inputs, outputs=outputs)

        return generator

            
    def call(self, noisy_images, time_step, text_embeddings):
            
        time_embedding = self.time_embedding(time_step)
        text_embedding = self.text_projection(text_embeddings)
        dim_1 = tf.TensorShape(text_embedding.shape).as_list()[1]
        dim_2 = tf.TensorShape(noisy_images.shape).as_list()[-1]
        dim_3 = tf.TensorShape(text_embedding.shape).as_list()[0]

        self.unet_block = self._build_unet_block(dim_3, dim_2, self.width, self.height)
            
        time_embedding_reshaped = tf.reshape(time_embedding, [self.batch_size, 1, 1, self.time_embedding_dim])
            
        time_embedding_tiled = tf.tile(time_embedding_reshaped, [1, self.width, self.height, 1]) 

        noisy_images_reshaped = tf.reshape(noisy_images, [self.batch_size, 1, 1, dim_2])
        noisy_images_tiled = tf.tile(noisy_images_reshaped, [1, self.width, self.height, 1])            
            
            
        text_embedding_reshaped = tf.reshape(text_embedding, [self.batch_size, 1, 1, self.text_embedding_dim*dim_3])
        text_embedding_tiled = tf.tile(text_embedding_reshaped, [1, self.width, self.height, 1]) 
        
            
        d1 = Concatenate(axis=-1)([noisy_images_tiled, time_embedding_tiled, text_embedding_tiled]) 
            
        d1 = self.unet_block(d1)
        denoised_images = self.final_conv(d1)
        return denoised_images

    
class Text2ImageDiffusionModel(tf.keras.Model):
    def __init__(self, vocab_size, num_batch, width, height):
        super(Text2ImageDiffusionModel, self).__init__()
        self.text_encoder = TextEncoder(vocab_size)
        self.image_encoder = ImageEncoder()
        self.batch_size = num_batch
        self.width = width
        self.height = height
        self.diffusion_module = UNetDiffusionModule(self.batch_size, self.width, self.height)
        
    def call(self, text_inputs, image_inputs, time_steps):
        text_embeddings = self.text_encoder(text_inputs)
        latent_images = self.image_encoder(image_inputs)
        generated_images = self.diffusion_module(latent_images, time_steps, text_embeddings)
        return generated_images
        



def load_dataset(description_file, image_directory, batch_size, height, width):
    df = pd.read_csv(description_file)

    image_paths = [f"{image_directory}/{image_id}" for image_id in df['image_id']]
    descriptions = [desc.replace('"', '') for desc in df['description']]

    def preprocess_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)  
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
        
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(preprocess_image)
    text_dataset, voc_li = preprocess_text(descriptions)

    dataset = tf.data.Dataset.zip((image_dataset, text_dataset))
    dataset = dataset.shuffle(buffer_size=len(df)).batch(batch_size)
    return dataset, len(tokenizer.word_index) + 1, voc_li


def generate_image_from_text(sentence, model, width, height, time_steps, path, gross_range, initial_image=None):
    try :                       
        inputs = [tokenizer.word_index[i] for i in sentence.split(" ")]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=gross_range, padding="post")
        inputs = tf.convert_to_tensor(inputs)

        if initial_image is None:
            initial_image = tf.random.normal(shape=[1, width, height, 3])  
            initial_image = tf.clip_by_value(initial_image, -1, 1)
        def postprocedure(img, path) :
            img = np.clip(img, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(path)
                    
        time_steps_ = tf.range(0, time_steps, dtype=tf.float32)
        generated_images = model(inputs, initial_image, time_steps_)
        final_image = generated_images[-1]
        postprocedure(final_image, path)
    except :
        print('ERROR OCCURED')




def main() :
    configuration()
    epochs = 10000
    implemented_coefficient = 0.25
    time_embedding_dim = 128
    csv_path = 'descriptions.csv'
    images_path = './images'



    
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of available GPUs: {strategy.num_replicas_in_sync}')

    
    def max_len(vectors) :
        length = max(len(vec) for vec in vectors)
        return length

    
    dataset, vocab_size, magnitude = load_dataset(csv_path, images_path, BATCH_SIZE, height, width)
    gross_magnitude = max_len(magnitude)


    text2image_model = Text2ImageDiffusionModel(vocab_size, BATCH_SIZE, width, height)
    optimizer = Adam(learning_rate=0.001)
    loss_fn = MeanSquaredError()


        
    text2image_model.compile(optimizer=optimizer, loss=loss_fn)

    
    

    log_file_path = './log/StackGAN++.log'
    save_path = './samples'
    save_interval = 50



    def data_augmentation(images, noise_factor=0.1):
        noise = tf.random.normal(shape=(BATCH_SIZE, width, height, 3), mean=0.0, stddev=noise_factor)
        return images + noise



    
    def train(batch) :
        with tf.GradientTape() as tape:
            image_inputs, text_inputs = batch[0], batch[1][0]
            time_steps = tf.range(0, time_embedding_dim, dtype=tf.float32)
            image_inputs = data_augmentation(image_inputs, implemented_coefficient)
            print(text_inputs.shape, image_inputs.shape, time_steps.shape)
            generated_images = text2image_model(text_inputs, image_inputs, time_steps)
            loss = loss_fn(image_inputs, generated_images)
        gradients = tape.gradient(loss, text2image_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, text2image_model.trainable_variables))
        return loss


    for epoch in range(epochs):
        
        dataset = dataset.shuffle(buffer_size=vocab_size, reshuffle_each_iteration=True)

        iterator = iter(dataset)        
        
        num, total_losses = 0, 0
        for batch in iterator:
            per_loss = train(batch)
            num += 1
            total_losses += per_loss
        train_loss = total_losses / num


        print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss.numpy()}')
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Epoch {epoch + 1}, Batch Losses: {train_loss.numpy()}\n")

        if (epoch + 1) % save_interval == 0:
            sentence = "a galaxy in the cosmos."
            generate_image_from_text(sentence, text2image_model, width, height, time_embedding_dim, save_path, gross_magnitude)
            text2image_model.save_weights(f'models/UnetSD{epoch + 1}')

            converter = tf.lite.TFLiteConverter.from_keras_model(text2image_model)
            tflite_model = converter.convert()

            with open(f'models/UnetSD{epoch + 1}.tflite', 'wb') as f:
                f.write(tflite_model)


if __name__ == '__main__' :
    main()