import os
try :
    import tensorflow as tf
    if tf.__version__.startswith('1'):
        raise ImportError("Please upgrade your TensorFlow to version 2.x")
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from transformer import TFBertModel, BertTokenizer
    import pandas as pd
    from PIL import Image
    import numpy as np
except :
    requirements = ['numpy', 'tensorflow', 'pandas', 'Pillow', 'transfomers']
    for item in requirements :
        os.system(f'pip3 install {item}')
        print('Done!')

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

strategy = tf.distribute.MirroredStrategy()
print(f'Number of available GPUs: {strategy.num_replicas_in_sync}')

with strategy.scope() :

    def configuration() :
        os.makedirs('./log', exist_ok=True)
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./samples', exist_ok=True)


    class TextEncoder(tf.keras.Model):
        def __init__(self, model_url, output_dim=512):
            super(TextEncoder, self).__init__()
            self.transformer_model = TFBertModel.from_pretrained(model_url)
            self.text_projection = Dense(output_dim, activation='relu')

        def call(self, input_ids, attention_mask=None):
            outputs = self.transformer_model(input_ids, attention_mask=attention_mask)
            text_embeddings = self.text_projection(outputs.last_hidden_state)
            return text_embeddings


    class ImageEncoder(tf.keras.Model):
        def __init__(self, output_dim=512, weights_path=None):
            super(ImageEncoder, self).__init__()
            self.resnet_model = tf.keras.applications.ResNet50(include_top=False, weights=None, pooling='avg')
            if weights_path:
                self.resnet_model.load_weights(weights_path)
            self.image_projection = Dense(output_dim)

        def call(self, inputs):
            x = self.resnet_model(inputs)
            latent_representation = self.image_projection(x)
            return latent_representation

    class UNetDiffusionModule(tf.keras.Model):
        def __init__(self, time_embedding_dim=64, text_embedding_dim=512):
            super(UNetDiffusionModule, self).__init__()
            
            self.time_embedding = Embedding(input_dim=10000, output_dim=time_embedding_dim)
            self.text_projection = Dense(units=text_embedding_dim)
            self.time_embedding_dim = time_embedding_dim
            self.text_embedding_dim = text_embedding_dim       
            self.unet_block = self._build_unet_block()

            
            self.final_conv = Conv2D(filters=3, kernel_size=(7, 7), strides=(1, 1), padding='same')
    
            
        def _build_unet_block():
            inputs = Input(shape=(256, 256, 3))

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

            batch_size = tf.shape(noisy_images)[0]
            height = tf.shape(noisy_images)[1]
            width = tf.shape(noisy_images)[2]

            
            time_embedding_reshaped = tf.reshape(time_embedding, [batch_size, 1, 1, self.time_embedding_dim])
            
            time_embedding_tiled = tf.tile(time_embedding_reshaped, [1, height, width, 1]) 
            
            
            text_embedding_reshaped = tf.reshape(text_embedding, [batch_size, 1, 1, self.text_embedding_dim])
            text_embedding_tiled = tf.tile(text_embedding_reshaped, [1, height, width, 1]) 
        
            
            d1 = Concatenate(axis=-1)([noisy_images, time_embedding_tiled, text_embedding_tiled]) 
            
            d1 = self.unet_block(d1)
            denoised_images = self.final_conv(d1)
            return denoised_images

    
    class Text2ImageDiffusionModel(tf.keras.Model):
        def __init__(self, text_encoder_model_url, image_weights):
            super(Text2ImageDiffusionModel, self).__init__()
            self.text_encoder = TextEncoder(text_encoder_model_url)
            self.image_encoder = ImageEncoder(weights_path=image_weights)
            self.diffusion_module = UNetDiffusionModule()
        
        def call(self, text_inputs, image_inputs, time_steps, attentions):
            text_embeddings = self.text_encoder(text_inputs, attentions)
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
            tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=BERT_PATH)
            tokenized_outputs = tokenizer(descriptions, 
                                        padding=True, 
                                        truncation=True, 
                                        return_tensors="tf", 
                                        max_length=512)
            
            input_ids = tokenized_outputs["input_ids"]
            attention_mask = tokenized_outputs["attention_mask"]  

            text_dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_mask))        
            return text_dataset
        
        image_dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(preprocess_image)
        text_dataset = preprocess_text(descriptions)

        dataset = tf.data.Dataset.zip((image_dataset, text_dataset))
        dataset = dataset.shuffle(buffer_size=len(df)).batch(batch_size)
        return dataset, len(df)


    def generate_image_from_text(sentence, model, width, height, time_steps, path, initial_image=None):
                    
        inputs = tokenizer(sentence, return_tensors="tf", padding="max_length", truncation=True, max_length=512)
        tokenized_inputs = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        if initial_image is None:
            initial_image = tf.random.normal(shape=[1, width, height, 3])  
            initial_image = tf.clip_by_value(initial_image, -1, 1)
        def postprocedure(img, path) :
            img = np.clip(img, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(path)
                    
        time_steps_ = tf.range(0, time_steps, dtype=tf.float32)
        generated_images = model(tokenized_inputs, initial_image, time_steps_, attention_mask)
        final_image = generated_images[-1].numpy()
        postprocedure(final_image, path)




    def main() :
        configuration()
        epochs = 10000  
        csv_path = 'descriptions.csv'
        images_path = './images'
        width, height = 256, 256
        BATCH_SIZE = 16
        global BERT_PATH, RESNET50_PATH
        RESNET50_PATH = './ResNet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        BERT_PATH ='./BERT'
        text2image_model = Text2ImageDiffusionModel(BERT_PATH, RESNET50_PATH)
        dataset, memory_size = load_dataset(csv_path, images_path, BATCH_SIZE, height, width)
        loss_fn = MeanSquaredError()
        log_file_path = './log/StackGAN++.log'
        save_path = './samples'
        save_interval = 50
        optimizer = Adam(learning_rate=0.001)


        text2image_model.compile(optimizer=optimizer, loss=loss_fn)


        
        for epoch in range(epochs):
            dataset = dataset.shuffle(buffer_size= memory_size, reshuffle_each_iteration=True)
            time_steps = tf.range(0, epochs, dtype=tf.float32)  
            iterator = iter(dataset)

            for batch in iterator :
                with tf.GradientTape() as tape:
                    text_inputs, image_inputs, attention = batch[0], batch[1][0], batch[1][1]
                    generated_images = text2image_model(text_inputs, image_inputs, time_steps, attention)
                    loss = loss_fn(image_inputs, generated_images)

                gradients = tape.gradient(loss, text2image_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, text2image_model.trainable_variables))

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}')
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"Epoch {epoch + 1}, Batch Losses: {loss.numpy()}\n")

            if (epoch + 1) % save_interval == 0 :
                sentence = "A beautiful landscape with mountains and clear blue sky"
                generate_image_from_text(sentence, text2image_model, width, height, epochs, save_path)
                text2image_model.save_weights(f'models/UnetSD{epoch+1}')
                converter = tf.lite.TFLiteConverter.from_keras_model(text2image_model) 
                tflite_model = converter.convert()

                
                with open(f'models/UnetSD{epoch+1}.tflite', 'wb') as f:
                    f.write(tflite_model)

    if __name__ == '__main__' :
        main()