import os, argparse
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
'''
Note: the relationship between GPUs and SIZE is complicated, 
meaning that larger size will guarantee the lower storing usage of GPUS, 
but increase the rate of occupation of GPU in the meantime 
'''                             
channel = 3
assert BATCH_SIZE >= 1; channel == 3
CHANNEL = channel
GLOBAL_BATCH_SIZE = BATCH_SIZE * tf.distribute.MirroredStrategy().num_replicas_in_sync




def configuration() :
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./samples', exist_ok=True)
    os.makedirs('./StageI', exist_ok=True)





class CharCnnRnn(tf.keras.Model):
    def __init__(self, optimizer, condition=True):
        super(CharCnnRnn, self).__init__()

        self.rnn_dim = 512
        self.rnn_num_steps = 18
        self.use_maxpool3 = condition
        self.optimizer = optimizer

        self.conv1 = layers.Conv1D(384, 4, activation=tf.nn.relu)
        self.maxpool1 = layers.MaxPooling1D(pool_size=3, strides=3)

        self.conv2 = layers.Conv1D(512, 4, activation=tf.nn.relu)
        self.maxpool2 = layers.MaxPooling1D(pool_size=3, strides=3)

        self.conv3 = layers.Conv1D(self.rnn_dim, 4, activation=tf.nn.relu)
        if self.use_maxpool3:
            self.maxpool3 = layers.MaxPooling1D(pool_size=3, strides=2)

        self.rnn = layers.GRU(self.rnn_dim)
        self.emb_proj = layers.Dense(64*64*3)

    def call(self, txt):
        out = self.conv1(txt)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        if self.use_maxpool3:
            out = self.maxpool3(out)

        out = tf.transpose(out, perm=[0, 2, 1])
        out = self.rnn(out)

        out = self.emb_proj(out)

        return out
    
    
    def sje_loss(self, text_embeds, image_embeds, margin=0.2, reduction=tf.keras.losses.Reduction.SUM):
        text_norm = tf.nn.l2_normalize(text_embeds, axis=-1)
        image_norm = tf.nn.l2_normalize(image_embeds, axis=-1)
        similarity = tf.matmul(text_norm, image_norm, transpose_b=True)

        positive_pairs = tf.linalg.diag_part(similarity)
        num_pairs = tf.shape(positive_pairs)[0]

        def compute_loss(i):
            loss_i = 0.0
            for j in range(num_pairs):
                if i != j:
                    loss_i += tf.maximum(0.0, margin - positive_pairs[i] + similarity[i, j])
                    loss_i += tf.maximum(0.0, margin - positive_pairs[i] + similarity[j, i])
            return loss_i / tf.cast(num_pairs - 1, tf.float32)

        losses = tf.map_fn(compute_loss, tf.range(num_pairs), dtype=tf.float32)
        if reduction == tf.keras.losses.Reduction.SUM :
            sje_loss = tf.reduce_mean(losses) 
        else :
            sje_loss = losses
        correct_text_to_image = tf.reduce_sum(tf.cast(tf.argmax(similarity, axis=1, output_type=tf.int32) == tf.range(num_pairs, dtype=tf.int32), tf.float32))
        correct_image_to_text = tf.reduce_sum(tf.cast(tf.argmax(similarity, axis=0, output_type=tf.int32) == tf.range(num_pairs, dtype=tf.int32), tf.float32))

        accuracy = (correct_text_to_image + correct_image_to_text) / (2 * tf.cast(num_pairs, tf.float32))

        return sje_loss, accuracy



    
    def train_step(self, texts, images, symmetric=True):
        with tf.GradientTape() as tape:
            text_embeds = self(texts, training=True)
            loss_1, acc_1 = self.sje_loss(images, text_embeds, reduction=tf.keras.losses.Reduction.NONE)
            loss_2, acc_2 = 0, 0
            acc = acc_1 + acc_2
            if symmetric :
                loss_2, acc_2 = self.sje_loss(text_embeds, images, reduction=tf.keras.losses.Reduction.NONE)
                acc = (acc_1 + acc_2) / 2
            loss = loss_1 + loss_2

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, acc





def load_dataset(description_file, image_directory, batch_size, height, width, max_len):
    df = pd.read_csv(description_file)
    descriptions = [desc.replace('"', '') for desc in df['description']]
    image_paths = [f"{image_directory}/{image_id}" for image_id in df['image_id']]

    def str_to_labelvec(string, max_str_len):
        string = string.lower()
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
        alpha_to_num = {k: v + 1 for k, v in zip(alphabet, range(len(alphabet)))}
        labels = tf.zeros(max_str_len, dtype=tf.int32)
        max_i = min(max_str_len, len(string))
        for i in range(max_i):
            char_index = alpha_to_num.get(string[i], alpha_to_num[' '])
            labels = tf.tensor_scatter_nd_update(labels, [[i]], [char_index])    
        return labels, alpha_to_num


    def labelvec_to_onehot(labels):
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        one_hot = tf.one_hot(labels, depth=71)    
        one_hot = one_hot[:, 1:]   
        one_hot = tf.transpose(one_hot, perm=[1, 0])    
        return one_hot



    def preparation_txt(string, max_str_len):
        labels, tokenizer = str_to_labelvec(string, max_str_len)
        one_hot = labelvec_to_onehot(labels)
        return one_hot, tokenizer

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

    def preprocess_text(descriptions, max_len):
            txt_tensors = []
            for sentence in descriptions :
                tensor, tokenizer = preparation_txt(sentence, max_len)
                txt_tensors.append(tensor)
            text_dataset = tf.data.Dataset.from_tensor_slices(txt_tensors)
            return text_dataset, tokenizer


        
    image_dataset_I = tf.data.Dataset.from_tensor_slices(image_paths).map(preprocess_image_I)
    image_dataset_II = tf.data.Dataset.from_tensor_slices(image_paths).map(preprocess_image_II)
    text_dataset, tokenizer = preprocess_text(descriptions, max_len)

    dataset = tf.data.Dataset.zip((image_dataset_I, image_dataset_II ,text_dataset))
    dataset = dataset.shuffle(buffer_size=max(len(df)+1, 1024), reshuffle_each_iteration=True).batch(batch_size)
    return dataset



def main_stage1(max_len, path, recover=False) :
    print('''
        -----------------------------
        ---Stage 1 Is Initialized-----  
        -----------------------------

    ''')
    configuration()
    coversion_log_path = './log/CharCnnRnn.log'
    epochs_stage = 5000
    csv_path = 'descriptions.csv'
    images_path = './images'
    save_interval = 150

    with strategy.scope() :
        dataset = load_dataset(csv_path, images_path, GLOBAL_BATCH_SIZE, height, width, max_len)
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=2650,
            decay_rate=0.96,
            staircase=True
        )

        
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
        '''
        Warning:
        if the previous version is implemented under tf(2.11)(not include 2.11)
        the restoring line should be modified like this tf.keras.optimizers.legacy.RMSprop
        '''
        char = CharCnnRnn(optimizer)
        char.compile(optimizer=optimizer)
        if recover :
            char.load_weights(f'models/{path}')



    print(f'Number of available GPUs: {strategy.num_replicas_in_sync}')




    @tf.function
    def train_step_stage1(batch) :
        images, _, text = batch
        images = tf.reshape(images, [BATCH_SIZE, -1])
        print(images.shape, text.shape)
        loss, acc = char.train_step(text, images)
        return loss, acc


    @tf.function
    def distributed_train_stage1(datum) :
        loss, acc = strategy.run(train_step_stage1, args=(datum,))
        loss = strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        acc = strategy.reduce(tf.distribute.ReduceOp.SUM, acc, axis=None)
        return loss, acc



    for epoch in range(epochs_stage) :
        sparse_tensorized_data = strategy.experimental_distribute_dataset(dataset)
        iterator = iter(sparse_tensorized_data)

        num, loss, acc = 0, 0, 0
        for num_, batch in enumerate(iterator):
            loss_, acc_ = distributed_train_stage1(batch)
            num += 1
            loss += loss_
            acc += acc_
            print(f'per_batch_loss:{loss_} per_batch_accuracy:{acc_} epoch:{epoch + 1} batch_index:{num_+1}')
        loss = loss / num
        acc = acc / num


        print(f'Epoch {epoch + 1}/{epochs_stage}, Loss:{loss.numpy()}, Accuracy:{acc.numpy()}')
        with open(coversion_log_path, 'a') as log_file:
            log_file.write(f'Epoch {epoch + 1}/{epochs_stage}, Loss:{loss.numpy()}, Accuracy: {acc.numpy()}\n')

        if (epoch + 1) % save_interval == 0 or epoch == epochs_stage - 1:
            char.save_weights(f'models/CharCNNRnn{epoch + 1}')

    print('''
        -----------------------------
        ---Stage 1 Is Terminated-----  
        -----------------------------

    ''')

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='define checkpoints and recover the previous training')
    parser.add_argument('-path', type=str, nargs='?', default=None, help='Path to the directory or file')
    parser.add_argument('--flag', action='store_true', help='A boolean flag')
    args = parser.parse_args()
    flag = args.flag
    module_path = args.path
    main_stage1(256, module_path, flag)