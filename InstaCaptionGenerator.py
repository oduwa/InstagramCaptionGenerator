# -*- coding: utf-8 -*-
import math
import os, re, random
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import cv2
import pickle as pkl
import tensorflow.python.platform
from collections import Counter
from nltk.tokenize import TweetTokenizer
import tables


model_path = './models/tensorflow_insta'
model_path_transfer = './models/tf_final_insta'
feature_path_vgg = './data/insta_features_vgg2.h5'
annotation_path = './data/insta_captions2.npy'
vgg_path = './data/vgg16-20160129.tfmodel'
idx2word_path = 'data/insta_ixtoword2_thresh2.npy'
logdir = "models/tensorflow_insta/logs/t2_lrate_4e-2"

### Hyperparameters ###
WORD_COUNT_THRESHOLD_FOR_VOCAB = 2
dim_embed = 512 # size of image embedding
dim_hidden = 512 # number of neurons in RNN
dim_in = 4096 # flattened CNN image feature size
batch_size = 512
momentum = 0.9
n_epochs = 150
eval_epoch_interval = 1

### Helper Functions ###
def removePunctuation(text):
    '''
    Removes punctuation, changes to lower case and strips leading and trailing
    spaces.

    Args:
        text (str): Input string.

    Returns:
        (str): The cleaned up string.
    '''
    a=0
    while(a==0):
        if(text[0]==' '):
            text=text[1:]
        else:
            a=1
    while(a==1):
        if(text[-1]==' '):
            text=text[0:-1]
        else:
            a=0
    text=text.lower()
    return re.sub('[^#@.0-9a-zA-Z\\U\\u\\ ]', '', text.encode('unicode_escape')) # includes hashtags and @

def emoji_tokenize(text):
    emoji_pattern = re.compile(
        u"(\ud83d[\ude00-\ude4f])|"  # emoticons
        u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
        u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
        u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
        u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
        "+", flags=re.UNICODE)
    reg_split = emoji_pattern.split(text)
    new_text = ''
    for component in reg_split:
        if(component):
            new_text = new_text + component + ' '
    new_text = removePunctuation(new_text)

    t = TweetTokenizer()
    return t.tokenize(new_text)


def crop_image(x, target_height=224, target_width=224, as_float=True):
    image = cv2.imread(x)
    if as_float:
        image = image.astype(np.float32)

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))

def load_image(path, model_image_size=224):
    img = crop_image(path, target_height=model_image_size, target_width=model_image_size)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    #img = img[None, ...]
    return img

def save_numpy_array(arr, save_path):
    f = tables.open_file(save_path, mode='a')
    f.root.data.append(arr)
    f.close()

def load_numpy_array(save_path):
    f = tables.open_file(save_path, mode='a')
    arr = f.root.data.read()
    f.close()
    return arr

def load_dataset():
    x = load_numpy_array(feature_path_vgg)
    print x.shape

    y = np.load(annotation_path)
    # for cap in y:
    #     print cap
    #     print "\n"
    print len(y)

def test_append():
    x = ["hello boy", "hello girl"]
    x = np.array(x)

    f = tables.open_file("x", mode='w')
    atom = tables.StringAtom(itemsize=255)
    array_c = f.create_earray(f.root, 'data', atom, (0,))
    f.close()

    save_numpy_array(x, "x")
    save_numpy_array(np.array(["pikachu."]), "x")
    print load_numpy_array("x")

    # f = tables.open_file("x", mode='a')
    # #f.root.data.append(np.random.rand(100,4096))
    # f.root.data.append(x)
    # f.close()



    # f = tables.open_file("x", mode='r')
    # print(f.root.data.shape)
    # print f.root.data.read()
    # f.close()
    #
    # x = ["hello man", "hello woman"]
    # x = np.array(x)
    #
    # f = tables.open_file("x", mode='a')
    # f.root.data.append(x)
    # f.close()
    #
    # f = tables.open_file("x", mode='r')
    # print(f.root.data.shape)
    # print f.root.data.read()
    # f.close()

def parse_vgg_data():
    annotations = np.load(annotation_path)
    return load_numpy_array(feature_path_vgg), annotations

def buildWordVocab(caption_iterator, word_count_threshold=WORD_COUNT_THRESHOLD_FOR_VOCAB):
    print('BUILDING WORD VOCAB WITH WORDS WITH COUNT >= THRESHOLD OF %d' % (word_count_threshold, ))
    word_counts = {}
    ncaps = 0
    for cap in caption_iterator:
      ncaps += 1
      # Preprocess caption text
      cap = cap.decode("utf-8")
      cap = re.sub('@[A-Za-z0-9_-]*', '<USER_MENTION>', cap)
      cap = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                             '<URL_HERE>', cap)
      for w in emoji_tokenize(cap):
        word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('%d WORDS INCLUDED IN VOCAB OUT OF FOUND %d' % (len(vocab), len(word_counts)))

    ixtoword = {}
    ixtoword[0] = '.'
    wordtoix = {}
    wordtoix['#START#'] = 0
    ix = 1
    for w in vocab:
      wordtoix[w] = ix
      ixtoword[ix] = w
      ix += 1

    word_counts['.'] = ncaps
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector)
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)
    return wordtoix, ixtoword, bias_init_vector.astype(np.float32)

class InstaCaptionGeneratorTrain():
    def __init__(self, dim_in, dim_embed, dim_hidden, batch_size, n_lstm_steps, n_words, init_b=None):
        self.dim_in = dim_in
        self.dim_embed = dim_embed
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_words = n_words

        # declare the variables to be used for our word embeddings
        self.word_embedding = tf.Variable(tf.random_uniform([self.n_words, self.dim_embed], -0.1, 0.1), name='word_embedding')
        self.embedding_bias = tf.Variable(tf.zeros([dim_embed]), name='embedding_bias')

        # declare the RNN we use for caption generation
        #self.lstm = tf.contrib.rnn.BasicLSTMCell(dim_hidden)
        # 2-layer LSTM, each layer has n_hidden units.
        self.lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LayerNormBasicLSTMCell(dim_hidden,layer_norm=True), tf.contrib.rnn.LayerNormBasicLSTMCell(dim_hidden,layer_norm=True)])

        # declare the variables to be used to embed the image feature embedding to the word embedding space
        # (i.e. layer for going from CNN image representation to RNN which is kind of the first layer)
        self.img_embedding = tf.Variable(tf.random_uniform([dim_in, dim_hidden], -0.1, 0.1), name='img_embedding')
        self.img_embedding_bias = tf.Variable(tf.zeros([dim_hidden]), name='img_embedding_bias')

        # declare the variables to go from an LSTM output to a word encoding output
        # (i.e. output layer)
        self.word_encoding = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='word_encoding')
        if init_b is not None:
            self.word_encoding_bias = tf.Variable(init_b, name='word_encoding_bias')
        else:
            self.word_encoding_bias = tf.Variable(tf.zeros([n_words]), name='word_encoding_bias')

        # Flag to know whether running training or inference
        self.is_train = tf.placeholder(tf.bool, name="is_train")


    def build_model(self):
        # declaring the placeholders for our extracted image feature vectors, our caption, and our mask
        # (mask describes how long our caption is with an array of 0/1 values of length `maxlen`
        img = tf.placeholder(tf.float32, [self.batch_size, self.dim_in])
        caption_placeholder = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps])
        mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])

        # first layer: taking the image feature to the RNN embedding space
        image_embedding = tf.matmul(img, self.img_embedding) + self.img_embedding_bias

        # setting initial state of our RNN
        state = self.lstm.zero_state(self.batch_size, dtype=tf.float32)

        total_loss = 0.0
        with tf.variable_scope("RNN"):
            for i in range(self.n_lstm_steps):
                if i > 0:
                    # if this isn't the first iteration of our RNN we need to get the word embedding corresponding
                    # to the (i-1)th word in our caption.
                    # This is because the first layer takes as input, the image feature and embeds it. The next layer
                    # then takes the first word in the caption and the following layers continue on.
                    current_embedding = tf.nn.embedding_lookup(self.word_embedding,
                                                               caption_placeholder[:, i - 1]) + self.embedding_bias

                    # allows us to reuse the LSTM tensor variable on each iteration
                    tf.get_variable_scope().reuse_variables()

                    out, state = self.lstm(current_embedding, state)

                    # get the one-hot representation of the next word in our caption
                    labels = tf.expand_dims(caption_placeholder[:, i], 1)
                    ix_range = tf.range(0, self.batch_size, 1)
                    ixs = tf.expand_dims(ix_range, 1)
                    concat = tf.concat([ixs, labels], 1)
                    onehot = tf.sparse_to_dense(
                        concat, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

                    # perform a softmax classification to generate the next word in the caption.
                    # This is basically applying an RNN timestep layer to get its output.
                    logit = tf.matmul(out, self.word_encoding) + self.word_encoding_bias
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=onehot)

                    # Apply mask to ensure caption length
                    cross_entropy = cross_entropy * mask[:, i]

                    loss = tf.reduce_sum(cross_entropy)
                    total_loss += loss
                else:
                    # if this is the first iteration of our LSTM we utilize the embedded image as our input
                    current_embedding = image_embedding

                    out, state = self.lstm(current_embedding, state)

            total_loss = total_loss / tf.reduce_sum(mask[:, 1:])
            return total_loss, img, caption_placeholder, mask


    def build_generator(self, maxlen, batchsize=1, isTrain=True):
        # same setup as `build_model` function
        img = tf.placeholder(tf.float32, [batchsize, self.dim_in])
        image_embedding = tf.matmul(img, self.img_embedding) + self.img_embedding_bias
        state = self.lstm.zero_state(batchsize, dtype=tf.float32)

        # list to hold the words of our generated captions
        all_words = []

        with tf.variable_scope("RNN", reuse=isTrain):
            # First pass in image embedding
            output, state = self.lstm(image_embedding, state)

            # set the first word to the embedding of the start token ([0]) for the future iterations
            previous_word = tf.nn.embedding_lookup(self.word_embedding, [0]) + self.embedding_bias

            for i in range(maxlen):
                tf.get_variable_scope().reuse_variables()

                out, state = self.lstm(previous_word, state)
                # get  maximum probability word and it's encoding from the output of the RNN
                logit = tf.matmul(out, self.word_encoding) + self.word_encoding_bias
                best_word = tf.argmax(logit, 1)

                # get the embedding of the best_word to use as input to the next iteration of our LSTM
                previous_word = tf.nn.embedding_lookup(self.word_embedding, best_word)
                previous_word += self.embedding_bias

                all_words.append(best_word)

        return img, all_words


def train(lrate=0.075, continue_training=False, transfer=True):
    tf.reset_default_graph()

    feats, captions = parse_vgg_data()
    # feats = feats[:1000]
    # captions = captions[:1000]
    wordtoix, ixtoword, init_b = buildWordVocab(captions)
    np.save(idx2word_path, ixtoword)

    # Create indices to use to select images from dataset and shuffle
    index = (np.arange(len(feats)).astype(int))
    np.random.shuffle(index)

    n_words = len(wordtoix)
    maxlen = np.max([x for x in map(lambda x: len(x.split(' ')), captions)])
    caption_generator = InstaCaptionGeneratorTrain(dim_in, dim_hidden, dim_embed, batch_size, maxlen + 2, n_words, init_b)
    loss, image, sentence, mask = caption_generator.build_model()

    # Training loop variables
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(lrate, global_step, (int(len(index) / batch_size)), 0.99)
    learning_rate_const = tf.train.exponential_decay(lrate, global_step, (int(len(index) / batch_size)), 1)
    #learning_rate = tf.train.exponential_decay(learning_rate, global_step, 200, 2.0)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
        optimizer_const = tf.train.AdamOptimizer(lrate).minimize(loss, global_step=global_step)


    saver = tf.train.Saver(max_to_keep=100)

    # Tensorboard Summaries
    #tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('loss', loss)

    merged_summary = tf.summary.merge_all()

    with tf.Session() as sess:
        # Tensorboard
        file_writer = tf.summary.FileWriter(logdir, sess.graph)

        tf.global_variables_initializer().run()

        if continue_training:
            if not transfer:
                if True:#not tf.train.latest_checkpoint(model_path):
                    saver.restore(sess, model_path+"/model_c-30")
                else:
                    saver.restore(sess, tf.train.latest_checkpoint(model_path))

            else:
                # saver.restore(sess, tf.train.latest_checkpoint(model_path_transfer))
                saver.restore(sess, model_path + "/model_a")

        losses = []
        losses_2 = []
        steps = 0
        log_step = 0
        previously_saved_loss = 5.0
        use_adaptive_lrate = False
        try:
            for epoch in range(n_epochs):
                for start, end in zip(range(0, len(index), batch_size), range(batch_size, len(index), batch_size)):
                    # Get image features, captions and indices for captions
                    current_feats = feats[index[start:end]]
                    current_captions = captions[index[start:end]]
                    current_caption_ind = [x for x in map(
                        lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix],
                        current_captions)]

                    # matrify caption indices and get mask
                    current_caption_matrix = tf.keras.preprocessing.sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen + 1)
                    current_caption_matrix = np.hstack(
                        [np.full((len(current_caption_matrix), 1), 0), current_caption_matrix])
                    current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
                    nonzeros = np.array([x for x in map(lambda x: (x != 0).sum() + 2, current_caption_matrix)])
                    for ind, row in enumerate(current_mask_matrix):
                        row[:nonzeros[ind]] = 1

                    # Train
                    if not use_adaptive_lrate:
                        sess.run(optimizer_const, feed_dict={
                            image: current_feats.astype(np.float32),
                            sentence: current_caption_matrix.astype(np.int32),
                            mask: current_mask_matrix.astype(np.float32)
                        })
                    else:
                        sess.run(optimizer, feed_dict={
                            image: current_feats.astype(np.float32),
                            sentence: current_caption_matrix.astype(np.int32),
                            mask: current_mask_matrix.astype(np.float32)
                        })
                    # sess.run(optimizer, feed_dict={
                    #     image: current_feats.astype(np.float32),
                    #     sentence: current_caption_matrix.astype(np.int32),
                    #     mask: current_mask_matrix.astype(np.float32)
                    # })

                    # steps += 1
                    #
                    # if steps % 50 == 0:
                    #     log_step += 1
                    #     summary = sess.run(merged_summary, feed_dict={
                    #         image: current_feats.astype(np.float32),
                    #         sentence: current_caption_matrix.astype(np.int32),
                    #         mask: current_mask_matrix.astype(np.float32)
                    #     })
                    #     file_writer.add_summary(summary, global_step=log_step)

                if epoch % eval_epoch_interval == 0:
                    loss_value = sess.run(loss, feed_dict={
                        image: current_feats.astype(np.float32),
                        sentence: current_caption_matrix.astype(np.int32),
                        mask: current_mask_matrix.astype(np.float32)
                    })
                    # summary = sess.run(merged_summary, feed_dict={
                    #     image: current_feats.astype(np.float32),
                    #     sentence: current_caption_matrix.astype(np.int32),
                    #     mask: current_mask_matrix.astype(np.float32)
                    # })
                    losses.append(int(loss_value))
                    losses_2.append(loss_value)

                    # file_writer.add_summary(summary, global_step=epoch)
                    print("Current Cost: ", loss_value, "\t Epoch {}/{}".format(epoch, n_epochs))

                    # Test and generate text with model 3 times
                    for i in range(3):
                        rnd_idx = random.sample(index, 1)
                        X = feats[rnd_idx]
                        Y = captions[rnd_idx]
                        max_gen_len = 15
                        test_im, generated_words = caption_generator.build_generator(maxlen=max_gen_len, batchsize=1)
                        generated_word_index = sess.run(generated_words, feed_dict={test_im: X})
                        generated_word_index = np.hstack(generated_word_index)
                        generated_words = [ixtoword[x] for x in generated_word_index]
                        #generated_words = [w for w in generated_words if w != "."]
                        generated_sentence = ' '.join(generated_words)
                        print("ACTUAL: {}\nPREDICTED: {}\n".format(Y, generated_sentence))

                    # if len(losses) >= 10 and all(elem == losses[-1] for elem in losses[:-10:-1]):
                    #     use_adaptive_lrate = True

                if epoch % 5 == 0 and loss_value <= previously_saved_loss and epoch >= 20:
                    print("Saving the model from epoch: ", epoch)
                    saver.save(sess, os.path.join(model_path, 'model_d'), global_step=epoch)
                    #saver.save(sess, os.path.join(model_path, 'model_c'))
                    previously_saved_loss = loss_value
        except KeyboardInterrupt:
            answer = raw_input("Keyboard Interrupt\n(S)ave model or (I)gnore?\n")
            if answer.lower() == 's':
                saver.save(sess, os.path.join(model_path, 'model_c2'))


def run_training_loop():
    try:
        train(.001,False,False) #train from scratch
        #train(.001,True,True)    #continue training from pretrained weights @epoch500
        #train(.001,True,False)  #train from previously saved weights
    except KeyboardInterrupt:
        print('Exiting Training')

def generate_caption_for_image_with_path(image_path, max_gen_len=15):
    if not os.path.exists(idx2word_path):
        print ('You must run a training loop for at least one epoch first.')
    else:
        tf.reset_default_graph()

        # Load pretrained VGG16
        with open(vgg_path, 'rb') as f:
            fileContent = f.read()
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fileContent)

        # Placeholder for VGG16's expected input
        images = tf.placeholder("float32", [1, 224, 224, 3])
        tf.import_graph_def(graph_def, input_map={"images": images})

        ixtoword = np.load(idx2word_path, allow_pickle=True).tolist()
        n_words = len(ixtoword)

        graph = tf.get_default_graph()
        sess = tf.InteractiveSession(graph=graph)
        caption_generator = InstaCaptionGeneratorTrain(dim_in, dim_hidden, dim_embed, 1, max_gen_len + 2, n_words)
        graph = tf.get_default_graph()
        image, generated_words = caption_generator.build_generator(maxlen=max_gen_len, isTrain=False)

        im = load_image(image_path,224)
        im = np.array([im])
        feat = sess.run(graph.get_tensor_by_name("import/Relu_1:0"), feed_dict={images: im})

        saver = tf.train.Saver()

        # Load model graph
        # saved_path = tf.train.latest_checkpoint(model_path)
        # saver.restore(sess, saved_path)
        saver.restore(sess, model_path + "/model_d")

        generated_word_index = sess.run(generated_words, feed_dict={image: feat})
        generated_word_index = np.hstack(generated_word_index)
        generated_words = [ixtoword[x] for x in generated_word_index]
        generated_words = [w for w in generated_words if w != "."]

        generated_sentence = ' '.join(generated_words)
        print(generated_sentence)

if __name__ == "__main__":
    generate_caption_for_image_with_path("data/images/custom_h.jpg", max_gen_len=15)


    #create_vgg_dataset("/Users/Odie/Downloads/instagram-scraper-master/instagram_scraper/foodposts")
    #load_dataset()
    #test_append()

    # x = np.load(annotation_path)
    # x = np.append(x, np.array(["pikachu"]))
    # print x[-1]

    #print emoji_tokenize("Happy Halloween 🙈👀🙈 #Halloween #selfie".decode("utf-8"))

    # feats, caps = parse_vgg_data()
    # buildWordVocab(caps)

    #run_training_loop()
