# -*- coding: utf-8 -*-
import langdetect
import re
import os, shutil
from nltk.tokenize import TweetTokenizer

def remove_emoji(inputString):
    emoji_pattern = re.compile(
        u"(\ud83d[\ude00-\ude4f])|"  # emoticons
        u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
        u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
        u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
        u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
        "+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', inputString)

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

def tokenize(text):
    '''
    Custom function to split a chunk of text into tokens.
    Splits text chunk with " " and handles emoji unicode if present

    @param text text to be tokenized
    @return a list of string tokens
    '''
    tokens = []

    # Split by space
    components = text.split(" ")
    for comp in components:
        # Add to tokens list if no emoji regexed. If emoji regexed add each match separately
        matches = re.findall('(U\d+.{4})', comp, re.DOTALL)
        if(len(matches) < 1):
            tokens.append(comp.replace(' ', ''))
        else:
            for m in matches:
                tokens.append(m.replace(' ', ''))

    return tokens

def remove_non_english_posts(image_directory):
    for filename in os.listdir(image_directory):
        # Check if image
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_directory, filename)
            print(filename + "\n\n")

            # Check if in english
            caption = filename[:-4]
            caption = caption.decode("UTF-8")
            try:
                #if langdetect.detect(remove_emoji(caption)) != "en" or "_NOCAP" in caption:
                if "_NOCAP" in caption:
                    # delete if not english
                    print("DELETING.." + "\n\n")
                    try:
                        os.remove(image_path)
                    except OSError:
                        shutil.rmtree(image_path)
            except langdetect.lang_detect_exception.LangDetectException:
                continue

        else:
            # delete if not image
            try:
                path = os.path.join(image_directory, filename)
                os.remove(path)
            except OSError:
                path = os.path.join(image_directory, filename)
                shutil.rmtree(path)








# x = "When @larahubel and her 👶 send you post-book completion 🍪💕#caturday"
# print(emoji_tokenize(unicode(x.decode('utf-8'))))
# x = x.decode("UTF-8")
# print remove_emoji(x)
# print(langdetect.detect(remove_emoji(x)))

remove_non_english_posts("/Users/Odie/Downloads/instagram-scraper-master/instagram_scraper/posts")