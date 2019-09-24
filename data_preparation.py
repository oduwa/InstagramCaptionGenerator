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
import tqdm


model_path = './models/tensorflow_insta'
model_path_transfer = './models/tf_final_insta'
feature_path_vgg = './data/insta_features_vgg2.h5'
annotation_path = './data/insta_captions2.npy'
vgg_path = './data/vgg16-20160129.tfmodel'
idx2word_path = 'data/insta_ixtoword_thresh1.npy'
logdir = "models/tensorflow_insta/logs/t2_lrate_4e-2"

### Hyperparameters ###
WORD_COUNT_THRESHOLD_FOR_VOCAB = 2
dim_embed = 512 # size of image embedding
dim_hidden = 512 # number of neurons in RNN
dim_in = 4096 # flattened CNN image feature size
batch_size = 128
momentum = 0.9
n_epochs = 500
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
    return re.sub('[^#@0-9a-zA-Z\\U\\u\\ ]', '', text.encode('unicode_escape')) # includes hashtags and @

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

def create_vgg_dataset(image_directory):
    # initialize the data and labels
    filenames = []
    batch_size = 200

    # Get captions and total number of images in folder
    print("COLLECTING FILENAMES")
    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg"):
            filenames.append(filename)
    print("COLLECTED {} FILES".format(len(filenames)))

    # Load VGG16 graph
    print("LOADING VGG16")
    with open(vgg_path, 'rb') as f:
        fileContent = f.read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fileContent)

    # Placeholder for VGG16's expected input
    input = tf.placeholder("float32", [batch_size, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={"images": input})
    graph = tf.get_default_graph()

    # Create HDF5 file used for serializing image arrays
    feat_file = tables.open_file(feature_path_vgg, mode='w')
    atom = tables.Float32Atom()
    array_c = feat_file.create_earray(feat_file.root, 'data', atom, (0, dim_in))
    feat_file.close()

    with tf.Session(graph=graph) as sess:
        N = len(filenames)
        n_processed = 0
        start = 0
        stop = batch_size
        progress = tqdm.tqdm(total=N)
        captions = []
        print("PROCESSING IMAGES...")
        while n_processed < N:
            # Load a batch of images
            images = []

            for filename in filenames[start:stop]:
                images.append(load_image(os.path.join(image_directory, filename), 224))
                captions.append(filename[:-4])

            # Prepare for numpy
            images = np.array(images, dtype="float")

            # Apply VGG
            if(images.shape[0] == batch_size):
                feats = sess.run(graph.get_tensor_by_name("import/Relu_1:0"), feed_dict={input: images})
                save_numpy_array(feats, feature_path_vgg)
            else:
                # abandon remainder
                remainder = len(captions) % batch_size
                captions = captions[:len(captions)-remainder]
                break

            # Update batch processing vars
            n_processed += batch_size
            start += batch_size
            stop += batch_size
            progress.update(batch_size)
        progress.close()
        print("COMPLETED IMAGE FEATURE EXTRACTION")

        # Serialize captions to numpy file
        print("SAVING CAPTIONS")
        np.save(annotation_path, captions, allow_pickle=True)


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

def buildWordVocab(caption_iterator, word_count_threshold):
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
    #print word_counts
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


if __name__ == "__main__":
    #create_vgg_dataset("/Users/Odie/Downloads/instagram-scraper-master/instagram_scraper/posts2")

    feats = load_numpy_array(feature_path_vgg)
    captions = np.load(annotation_path)

    print(feats.shape)
    print(captions.shape)

    # answer = raw_input("Keyboard Interrupt\n(S)ave model or (I)gnore?\n")
    # if answer.lower() == 's':
    #     print answer

    # c = 0
    # for cap in captions:
    #     if "tuned" in emoji_tokenize(cap.decode("utf-8")):
    #         c+=1
    # print c
    # print(len(captions))

    # c = 0
    # filenames = []
    # for filename in os.listdir("/Users/Odie/Downloads/instagram-scraper-master/instagram_scraper/posts2"):
    #     if filename.endswith(".jpg"):
    #         filenames.append(filename)
    #         c+=1
    # print c
    #
    # captions = []
    # for f in filenames:
    #     cap = f[:-4]
    #     captions.append(cap)
    # print(len(captions))
    # captions = captions[:len(captions) - 200]
    # print(len(captions))
    # c = 0
    # for cap in captions:
    #     if "tuned" in emoji_tokenize(cap.decode("utf-8")):
    #         c += 1
    # print c






    wordtoix, ixtoword, _ = buildWordVocab(captions, 5)
