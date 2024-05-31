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
channel = 3
assert BATCH_SIZE >= 1, channel == 3
CHANNEL = channel
GLOBAL_BATCH_SIZE = BATCH_SIZE * tf.distribute.MirroredStrategy().num_replicas_in_sync




def configuration() :
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./samples', exist_ok=True)
    os.makedirs('./temporary_checkpoints', exist_ok=True)
    os.makedirs('./VAE_results', exist_ok=True)



class TextEncoder(tf.keras.Model):
    def __init__(self, vocab_size, output_dim=128, embed_dim=512):
        super(TextEncoder, self).__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.text_projection = Dense(output_dim, activation='relu')


    def call(self, input_ids):
        outputs = self.embedding(input_ids)
        text_embeddings = self.text_projection(outputs)
        return text_embeddings


class GroupNorm(layers.Layer):
    def __init__(self, groups=32, axis=-1, epsilon=1e-5):
        super(GroupNorm, self).__init__()
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None

    def _initialize_weights(self, input_shape):
        param_shape = [input_shape[self.axis]]
        self.gamma = self.add_weight(name='gamma',
                                     shape=param_shape,
                                     initializer='ones',
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=param_shape,
                                    initializer='zeros',
                                    trainable=True)
        self.groups = min(self.groups, input_shape[self.axis])

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        tensor_shape = inputs.shape.as_list()

        if self.gamma is None or self.beta is None:
            self._initialize_weights(tensor_shape)

        
        group_shape = [tensor_shape[0], tensor_shape[1], tensor_shape[2], self.groups, tensor_shape[3] // self.groups]
        inputs = tf.reshape(inputs, group_shape)

        
        mean, variance = tf.nn.moments(inputs, [1, 2, 4], keepdims=True)
        inputs = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        inputs = tf.reshape(inputs, input_shape)
        return self.gamma * inputs + self.beta


class Swish(layers.Layer):
    def call(self, inputs):
        return inputs * tf.nn.sigmoid(inputs)


class GSC(layers.Layer):
    def __init__(self, filters, kernel_size=3):
        super(GSC, self).__init__()
        self.group_norm = GroupNorm(groups=32)
        self.swish = Swish()
        self.conv = Conv2D(filters, kernel_size, padding='same')

    def call(self, inputs):
        x = self.group_norm(inputs)
        x = self.swish(x)
        x = self.conv(x)
        return x

class ResidualBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(filters, kernel_size, padding='same')
        self.bn1 = BatchNormalization()
        self.relu = LeakyReLU(alpha=0.2)
        self.conv2 = Conv2D(filters, kernel_size, padding='same')
        self.bn2 = BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += inputs
        return x


class VAEResidualBlock(layers.Layer):
    def __init__(self, filters):
        super(VAEResidualBlock, self).__init__()
        self.gsc1 = GSC(filters)
        self.gsc2 = GSC(filters)

    def call(self, inputs):
        x = self.gsc1(inputs)
        x = self.gsc2(x)
        return x + inputs





class DownBlock(layers.Layer):
    def __init__(self, filters, number=2):
        super(DownBlock, self).__init__()
        self.num = number
        self.conv = Conv2D(filters, 3, strides=2, padding='same')
        self.resblocks = [VAEResidualBlock(filters) for _ in range(self.num)]

    def call(self, inputs):
        x = self.conv(inputs)
        for resblock in self.resblocks:
            x = resblock(x)
        return x

class UpBlock(layers.Layer):
    def __init__(self, filters, number=3):
        super(UpBlock, self).__init__()
        self.num = number
        self.interpolate = UpSampling2D(size=(2, 2), interpolation='nearest')
        self.conv = Conv2D(filters, 3, padding='same')
        self.resblocks = [VAEResidualBlock(filters) for _ in range(self.num)]

    def call(self, inputs):
        x = self.interpolate(inputs)
        x = self.conv(x)
        for resblock in self.resblocks:
            x = resblock(x)        
        return x



class SelfAttention(layers.Layer):
    def __init__(self, filters):
        super(SelfAttention, self).__init__()
        self.query = Conv2D(filters, 1)
        self.key = Conv2D(filters, 1)
        self.value = Conv2D(filters, 1)
        self.scale = Lambda(lambda x: x / tf.math.sqrt(tf.cast(filters, tf.float32)))

    def call(self, inputs):
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)
        score = tf.matmul(query, key, transpose_b=True)
        score = self.scale(score)
        weights = tf.nn.softmax(score)
        attention = tf.matmul(weights, value)
        return attention

class MidBlock2D(tf.keras.Model):
    def __init__(self, filters):
        super(MidBlock2D, self).__init__()
        self.resnet1 = VAEResidualBlock(filters)
        self.attn = SelfAttention(filters)
        self.resnet2 = VAEResidualBlock(filters)

    def call(self, inputs):
        x = self.resnet1(inputs)
        x = self.attn(x)
        x = self.resnet2(x)
        return x



class VAE(tf.keras.Model):
    def __init__(self, loss, optimizer, latent_dim, batch_size, width, height, channel, num_up_res_blocks=3, num_down_res_blocks=2):
        super(VAE, self).__init__()
        self.width = width
        self.latent_dim = latent_dim
        self.height = height
        self.batch_size = batch_size
        self.channel = channel

        self.encoder = [
            Conv2D(32, 3, strides=1, padding='same', kernel_initializer=HeNormal()),
            DownBlock(64, num_down_res_blocks),
            DownBlock(128, num_down_res_blocks),
            DownBlock(256, num_down_res_blocks),
            *[VAEResidualBlock(256) for _ in range(num_down_res_blocks)],
            MidBlock2D(256),
            GSC(256),
            Flatten(),
            Dense(self.latent_dim + self.latent_dim, kernel_initializer=HeNormal())
        ]
        self.decoder = [
            Dense(32 * 32 * 256, activation='relu', kernel_initializer=HeNormal()),
            Reshape((32, 32, 256)),
            Conv2D(256, 3, strides=1, padding='same', kernel_initializer=HeNormal()),
            MidBlock2D(256),
            UpBlock(128, num_up_res_blocks),
            UpBlock(64, num_up_res_blocks),
            UpBlock(32, num_up_res_blocks),
            *[VAEResidualBlock(32) for _ in range(num_up_res_blocks)],
            GSC(32),
            Conv2DTranspose(3, 3, activation='sigmoid', padding='same', kernel_initializer=HeNormal())
        ]
        self.optimizer = optimizer
        self.mse_loss = loss

    def encode(self, inputs):
        x = inputs
        for layer in self.encoder:
            x = layer(x)
        z_mean, z_log_var = tf.split(x, num_or_size_splits=2, axis=1)
        z = self.reparameterize(z_mean, z_log_var)
        return z, z_mean, z_log_var

    def decode(self, inputs):
        x = inputs
        for layer in self.decoder:
            x = layer(x)
        return x

    def call(self, inputs):
        z, _, _ = self.encode(inputs)
        x = self.decode(z)
        return x

    def reparameterize(self, mean, log_var):
        epsilon = tf.random.normal(shape=(tf.shape(mean)[0], self.latent_dim))
        return mean + tf.exp(0.5 * log_var) * epsilon

    def compute_loss(self, inputs, l2_reg_coeff=0.03, reconstruction_loss_weight=1.0):
        with tf.GradientTape() as tape:
            z, z_mean, z_log_var = self.encode(inputs)
            reconstructed = self.decode(z)
            reconstruction_loss = self.mse_loss(inputs, reconstructed)
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.nn.softplus(kl_loss))  
            loss = reconstruction_loss_weight * reconstruction_loss + kl_loss
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables if 'kernel' in v.name])
            regularization_loss = l2_reg_coeff * l2_loss
            total_loss = loss + regularization_loss
            total_loss /= tf.cast(tf.reduce_prod(tf.shape(inputs)[:]), tf.float32)
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return total_loss






class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.apply_attention(q, mask)
        k = self.apply_attention(k, mask)
        v = self.apply_attention(v, mask)
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot(q, k, v)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, self.d_model))
        
        output = self.dense(concat_attention)
        
        return output, attention_weights
    

    def apply_attention(self, text_inputs, mask) :
        if mask is None:
            mask = tf.cast(tf.math.equal(text_inputs, 0), tf.float32) 
            text_inputs += (mask * -1e9)
        return text_inputs    


    def scaled_dot(self, q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
            
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights




class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size, strides, activation='relu'):
        super(ConvBlock, self).__init__()
        self.conv = Conv2D(filters, kernel_size, strides=strides, padding='same')
        self.batch_norm = BatchNormalization()
        self.activation = Activation(activation)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x




class DeconvBlock(layers.Layer):
    def __init__(self, filters, kernel_size, strides, activation='relu'):
        super(DeconvBlock, self).__init__()
        self.deconv = Conv2DTranspose(filters, kernel_size, strides=strides, padding='same')
        self.batch_norm = BatchNormalization()
        self.activation = Activation(activation)

    def call(self, inputs):
        x = self.deconv(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x




class UNetDiffusionModule(tf.keras.Model):
    def __init__(self, num_batch, width, height, d_model, num_heads=8):
        super(UNetDiffusionModule, self).__init__()
        self.batch_size = num_batch 
        self.width = width
        self.height = height          
        self.num_heads = num_heads
        self.d_model = d_model
        self.multi_head_attention = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)     
        self.project = Dense(width*height*32)
        self.reshape = Reshape((width, height, 32))
        self.compression = Dense(1024)      #  latent_dim
        self.concatenate = Concatenate()
        self.exaggeration = Conv2DTranspose(32, kernel_size=7, strides=1, padding='same', activation='sigmoid')
        self.flatten = Flatten()

        self.conv_block1 = ConvBlock(filters=64, kernel_size=4, strides=2)
        self.conv_block2 = ConvBlock(filters=128, kernel_size=4, strides=2)
        self.conv_block3 = ConvBlock(filters=256, kernel_size=4, strides=2)
        self.conv_block4 = ConvBlock(filters=512, kernel_size=4, strides=2)
        self.conv_block5 = ConvBlock(filters=1024, kernel_size=4, strides=2)
        
        self.deconv_block1 = DeconvBlock(filters=32, kernel_size=4, strides=2)
        self.deconv_block2 = DeconvBlock(filters=64, kernel_size=4, strides=2)
        self.deconv_block3 = DeconvBlock(filters=128, kernel_size=4, strides=2)
        self.deconv_block4 = DeconvBlock(filters=256, kernel_size=4, strides=2)
        self.deconv_block5 = DeconvBlock(filters=512, kernel_size=4, strides=2)

        self.residual_block1 = ResidualBlock(filters=128, kernel_size=3)
        self.residual_block2 = ResidualBlock(filters=256, kernel_size=3)
        self.residual_block3 = ResidualBlock(filters=512, kernel_size=3)
        self.residual_block4 = ResidualBlock(filters=1024, kernel_size=3)


    def call(self, noisy_images, text_embedding):
        eigenvector, _ = self.multi_head_attention(noisy_images, text_embedding, text_embedding)
        d1 = self.reshape(self.project(eigenvector))

        d1 = self.conv_block1(d1)
        d2 = self.conv_block2(d1)
        d2 = self.residual_block1(d2)
        d2 = self.residual_block1(d2)
        d3 = self.conv_block3(d2)
        d3 = self.residual_block2(d3)
        d3 = self.residual_block2(d3)
        d4 = self.conv_block4(d3)
        d4 = self.residual_block3(d4)
        d4 = self.residual_block3(d4)
        d5 = self.conv_block5(d4)
        d5 = self.residual_block4(d5)
        d5 = self.residual_block4(d5)

        d6 = self.deconv_block5(d5)
        d6 = self.concatenate([d6, d4])
        d6 = self.residual_block4(d6)
        d6 = self.residual_block4(d6)
        d7 = self.deconv_block4(d6)
        d7 = self.concatenate([d7, d3])
        d7 = self.residual_block3(d7)
        d7 = self.residual_block3(d7)
        d8 = self.deconv_block3(d7)
        d8 = self.concatenate([d8, d2])
        d8 = self.residual_block2(d8)
        d8 = self.residual_block2(d8)
        d9 = self.deconv_block2(d8)
        d9 = self.concatenate([d9, d1])
        d9 = self.residual_block1(d9)
        d9 = self.residual_block1(d9)
        d10 = self.deconv_block1(d9)
        d10 = self.exaggeration(d10)
        d10 = self.flatten(d10)
        outputs = self.compression(d10)                            
        return outputs




    
class Text2ImageDiffusionModel(tf.keras.Model):
    def __init__(self, VAE, vocab_size, num_batch, width, height, channel, alpha, d_model, input_shape = 512):
        super(Text2ImageDiffusionModel, self).__init__()
        self.text_encoder = TextEncoder(vocab_size)
        self.batch_size = num_batch
        self.flatten = Flatten()
        self.width = width
        self.vae = VAE
        self.channel = channel
        self.height = height
        self.alpha = alpha
        self.diffusion_module = UNetDiffusionModule(self.batch_size, self.width//8, self.height//8, d_model)
        
    def call(self, text_inputs, image_inputs, time_steps_vector):
        text_embeddings = self.text_encoder(text_inputs)
        latent_images, _, _ = self.vae.encode(image_inputs)
        generated_images_list = []
        gaussian_list = []
        generated_images_list.append(latent_images)
        for index in range(len(time_steps_vector) - 1) :
            gaussian_vector = tf.random.normal(shape=(self.batch_size, self.width, self.height, self.channel), mean=0.0, stddev=1.0)
            latent_gaussian_vector, _, _ = self.vae.encode(gaussian_vector)
            latent_images = np.sqrt(self.alpha**index) * latent_images + np.sqrt(1 - self.alpha**index) * latent_gaussian_vector
            generated_images_list.append(latent_images)
            gaussian_list.append(latent_gaussian_vector)
        images_tensor = tf.stack(generated_images_list, axis=0)
        latent_gaussian_tensor  = tf.stack(gaussian_list, axis=0)
        generated_list = []
        generated_images = self.diffusion_module(images_tensor[-1], self.flatten(text_embeddings))
        generated_list.append(generated_images)
        varied_tensor = 1/np.sqrt(self.alpha)*(images_tensor[-1] - (1 - self.alpha)/np.sqrt(1 - self.alpha**len(images_tensor)) * generated_images) + np.sqrt(1 - self.alpha) * latent_gaussian_tensor[len(images_tensor) - 2]
        for index in reversed(range(1, len(images_tensor) - 1)) :
            generated_images = self.diffusion_module(varied_tensor, self.flatten(text_embeddings))
            generated_list.append(generated_images)
            varied_tensor = 1/np.sqrt(self.alpha)*(varied_tensor - (1 - self.alpha)/np.sqrt(1 - self.alpha**index) * generated_images) + np.sqrt(1 - self.alpha) * latent_gaussian_tensor[index - 1]
        generated_tensor = tf.stack(generated_list, axis=0)
        return varied_tensor, generated_tensor, latent_gaussian_tensor
    

    def compute_loss(self, labels, predictions, loss_fn, l2_reg_coeff=0.01):
        per_example_loss = loss_fn(labels[..., None], predictions[..., None])
        loss = tf.nn.compute_average_loss(per_example_loss)

        
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables])
        regularization_loss = l2_reg_coeff * l2_loss

        total_loss = loss + regularization_loss

        
        total_loss /= tf.cast(tf.reduce_prod(tf.shape(labels)[:]), tf.float32)

        return total_loss


    
    def train_step(self, optimizer, targets, generated_images, loss_fn):
        with tf.GradientTape() as tape:
            scaled_loss = self.compute_loss(targets, generated_images, loss_fn)
        gradients = tape.gradient(scaled_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return scaled_loss

        



def load_dataset(description_file, image_directory, batch_size, height, width, vae_mode=False):
    df = pd.read_csv(description_file)
    descriptions = [desc.replace('"', '') for desc in df['description']]
    if vae_mode :
        portfolio = image_directory
        image_paths = glob.glob(os.path.join(portfolio, '*.jpg'))
    if not vae_mode :
        image_paths = [f"{image_directory}/{image_id}" for image_id in df['image_id']]

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
    if not vae_mode :
        text_dataset, voc_li = preprocess_text(descriptions)

        dataset = tf.data.Dataset.zip((image_dataset, text_dataset))
        dataset = dataset.shuffle(buffer_size=max(len(df)+1, 512), reshuffle_each_iteration=True).batch(batch_size)
        return dataset,  len(tokenizer.word_index) + 1, voc_li
    dataset = image_dataset.shuffle(buffer_size=max(len(df)+1, 512), reshuffle_each_iteration=True).batch(batch_size)
    text_dataset, voc_li = preprocess_text(descriptions)
    return dataset,  len(tokenizer.word_index) + 1, voc_li
    




def vae_validation(description_file, model, image_directory, save_path, signature) :
    try :
        df = pd.read_csv(description_file)

        test_image_path = f"{image_directory}/{df['image_id'][201]}"
        def preprocess_image(image_path):
            img = tf.io.read_file(image_path)
            img = tf.image.decode_png(img, channels=3)  
            img = tf.image.resize(img, [width, height])
            img_array = tf.image.convert_image_dtype(img, tf.float32) / 127.5 - 1.0
            return tf.expand_dims(img_array, axis=0)
        def postprocedure(img, path, signature) :
            img = ((img + 1.0) * 127.5).numpy().astype(np.uint8)
            img = np.clip(img, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(f'{path}/{signature}.png')
        image = preprocess_image(test_image_path)
        fake = model(image)
        postprocedure(fake[-1], save_path, signature)
    except Exception as e:
        print(f"Error encountered during VAE validation: {e}")



def generate_image_from_text(sentence, model1, model2, width, height, time_steps, path, gross_range, signature, initial_image=None):
    try :                       
        inputs = [tokenizer.word_index[i] for i in sentence.split(" ")]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=gross_range, padding="post")
        inputs = tf.convert_to_tensor(inputs)

        if initial_image is None:
            clean_image = tf.random.uniform(shape=[1, width, height, CHANNEL], minval=0, maxval=1, dtype=tf.float32)
            initial_image = tf.random.normal(shape=[1, width, height, CHANNEL]) 
            initial_image += clean_image 
            initial_image = tf.clip_by_value(initial_image, -1, 1)
        def postprocedure(img, path, signature) :
            img = ((img + 1.0) * 127.5).numpy().astype(np.uint8)
            img = np.clip(img, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(f'{path}/{signature}.png')
        time_steps_vector = tf.range(0, time_steps, dtype=tf.float32)
        text_embeddings = model1.text_encoder(inputs)
        latent_images, _, _ = model2.encode(initial_image)                   
        generated_gaussian = model1.diffusion_module(latent_images, model1.flatten(text_embeddings))
        varied_tensor = 1/np.sqrt(model1.alpha)*(latent_images - (1 - model1.alpha)/np.sqrt(1 - model1.alpha**len(time_steps_vector)) * generated_gaussian) + np.sqrt(1 - model1.alpha) * generated_gaussian
        for index in reversed(range(1, len(time_steps_vector) - 1)) :
            generated_gaussian= model1.diffusion_module(varied_tensor, model1.flatten(text_embeddings))
            varied_tensor = 1/np.sqrt(model1.alpha)*(varied_tensor - (1 - model1.alpha)/np.sqrt(1 - model1.alpha**index) * generated_gaussian) + np.sqrt(1 - model1.alpha) * generated_gaussian
        generated_images = model2.decode(varied_tensor)
        final_image = generated_images[0]
        postprocedure(final_image, path, signature)
    except Exception as e:
        print(f"Error encountered during generating image from the given text: {e}")





def main_stage1(latent_dim) :
    print('''
        -----------------------------
        ---Stage 1 Is Initialized-----  
        -----------------------------

    ''')
    configuration()
    coversion_log_path = './log/VAE.log'
    strategy = tf.distribute.MirroredStrategy()
    epochs_stage = 20000
    csv_path = 'descriptions.csv'
    images_path = './images'
    save_path = './VAE_results'
    save_interval = 150
    initial_learning_rate = 1e-4
    temporary_interval = 100


    with strategy.scope() :
        dataset, vocab_size, magnitude = load_dataset(csv_path, images_path, GLOBAL_BATCH_SIZE, height, width, vae_mode=True)
        loss_fn = MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=10000,
            decay_rate=0.9,
            staircase=True
        )
        optimizer = Adam(learning_rate=lr_schedule, beta_1=0.5, clipnorm=1.0)
        vae = VAE(loss_fn, optimizer, latent_dim, BATCH_SIZE, width, height, channel)
        vae.compile(optimizer=optimizer, loss=loss_fn)

    print(f'Number of available GPUs: {strategy.num_replicas_in_sync}')


    def insurance(model1, x1, x2) :
        with open("./temporary_checkpoints/last_latent_vector.pkl", "wb") as in_f :
            pickle.dump((x1, x2), in_f)
        model1.save_weights(f'./temporary_checkpoints/InsuranceModel')


    @tf.function
    def train_step_stage1(batch) :
        with tf.GradientTape() as tape :
            image_inputs = batch
            print(image_inputs.shape)
            scaled_loss = vae.compute_loss(image_inputs)
        gradients = tape.gradient(scaled_loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))

        return scaled_loss  


    @tf.function
    def distributed_train_stage1(datum) :
        per_replicas_losses = strategy.run(train_step_stage1, args=(datum,))
        per_replicas_losses = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replicas_losses, axis=None)
        return per_replicas_losses




    for epoch in range(epochs_stage) :
        sparse_tensorized_data = strategy.experimental_distribute_dataset(dataset)
        iterator = iter(sparse_tensorized_data)

        num, total_losses = 0, 0
        for num_, batch in enumerate(iterator):
            per_loss = distributed_train_stage1(batch)
            num += 1
            total_losses += per_loss
            print(f'per_batch_loss:{per_loss} epoch:{epoch + 1} batch_index:{num_+1}')
        if (epoch + 1) % temporary_interval == 0 :
            insurance(vae, vocab_size, len(magnitude))
        train_loss = total_losses / num


        print(f'Epoch {epoch + 1}/{epochs_stage}, Loss: {train_loss.numpy()}')
        with open(coversion_log_path, 'a') as log_file:
            log_file.write(f"Epoch {epoch + 1}, Batch Losses: {train_loss.numpy()}\n")

        if (epoch + 1) % save_interval == 0:
            vae_validation(csv_path, vae, images_path,
                                    save_path, epoch + 1)
            vae.save_weights(f'models/VAE{epoch + 1}')

    print('''
        -----------------------------
        ---Stage 1 Is Terminated-----  
        -----------------------------

    ''')
    return vae, vocab_size, len(magnitude)






def main_stage2(vae, vocab_size, gross_magnitude, latent_dim):
    print('''
        -----------------------------
        ---Stage 2 Is Initialized-----  
        -----------------------------

    ''')
    epochs2 = 20000
    alpha = 0.828
    csv_path_2 = 'descriptions.csv'
    images_path_2 = './images'
    save_path = './samples'
    initial_learning_rate = 1e-4
    time_embedding_dim = 512

    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of available GPUs: {strategy.num_replicas_in_sync}')



    with strategy.scope():
        tensorized_data, _, _ = load_dataset(csv_path_2, images_path_2, GLOBAL_BATCH_SIZE, height, width)
        text2image_model = Text2ImageDiffusionModel(vae, vocab_size, BATCH_SIZE, width, height, channel, alpha, latent_dim)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=10000,
            decay_rate=0.9,
            staircase=True
        )        
        optimizer = Adam(learning_rate=lr_schedule, beta_1=0.5)
        loss_fn = MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        text2image_model.compile(optimizer=optimizer, loss=loss_fn)

    log_file_path = './log/UnetSD_Latent.log'
    save_interval = 50




    @tf.function
    def train_step(batch):
        image_inputs, text_inputs = batch[0], batch[1]
        time_steps = tf.range(0, time_embedding_dim, dtype=tf.float32)
        print(text_inputs.shape, image_inputs.shape, time_steps.shape)
        output, predictions, targets = text2image_model(text_inputs, image_inputs, time_steps)
        counter, total_loss = 0, 0
        for index in reversed(range(len(targets))) :
            solo_loss = text2image_model.train_step(optimizer, targets[index], predictions[index], loss_fn)
            counter += 1
            total_loss += solo_loss
            print(f'per_diffusion_loss:{total_loss} epoch:{epoch} batch_index:{num_+1} diffusion_step:{index}')
        loss = total_loss / counter
        return output, loss


    @tf.function
    def distributed_training(inputs) :
        output, per_replicas_losses = strategy.run(train_step, args=(inputs,))
        per_replicas_losses = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replicas_losses, axis=None)
        return per_replicas_losses, output





    for epoch in range(epochs2):
        sparse_tensorized_data = strategy.experimental_distribute_dataset(tensorized_data)
        iterator = iter(sparse_tensorized_data)

        num, total_losses = 0, 0
        for num_, batch in enumerate(iterator):
            per_loss, _ = distributed_training(batch)
            num += 1
            total_losses += per_loss
            print(f'per_batch_loss:{per_loss} epoch:{epoch} batch_index:{num_+1}')
        train_loss = total_losses / num

        print(f'Epoch {epoch + 1}/{epochs2 + 1}, Loss: {train_loss.numpy()}')
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Epoch {epoch + 1}, Batch Losses: {train_loss.numpy()}\n")

        if (epoch + 1) % save_interval == 0 :
            sentence = "the brain's surface pulses emits a soft, ethereal light, drawing seekers of knowledge ever closer to its enigmatic depths"
            generate_image_from_text(sentence, text2image_model, vae, width, height, time_embedding_dim, save_path,
                                     gross_magnitude, epoch + 1)
            text2image_model.save_weights(f'models/UnetSD{epoch + 1}')

    print('''
        -----------------------------
        ---Stage 2 Is Terminated-----  
        -----------------------------

    ''')





def main(mode="restart"):
    latent_dim = 1024 


    def load_state():
        try:
            model_ = VAE(latent_dim, width, height, channel)
            model_.load_weights('./temporary_checkpoints/InsuranceModel')
            with open("./temporary_checkpoints/last_latent_vector.pkl", "rb") as f:
                x = pickle.load(f)
                x1, x2 = x[0], x[1]
            return model_, x1, x2                        
        except Exception as e:
            print(f"Error encountered during reactivating repository: {e}")
            return None



    if mode == 'restart':
        model, vocab_size, magnitude = main_stage1(latent_dim)
        main_stage2(model, vocab_size, magnitude, latent_dim)
    elif mode == 'recover':
        model, vocab_size, magnitude = load_state()
        main_stage2(model, vocab_size, magnitude, latent_dim)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main stages of the program.")
    parser.add_argument("mode", type=str, choices=["restart", "recover"],
                        help="Mode to run the program. 'train' for training mode and 'recover' for recovery mode.")
    args = parser.parse_args()
    main(args.mode)