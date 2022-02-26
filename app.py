from flask import Flask, render_template, request


import numpy as np
import pandas as pd
import cv2
import os
from glob import glob
import pathlib
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from tqdm import tqdm

from keras.applications import resnet
incept_model = resnet.ResNet50(include_top=False, weights='imagenet',input_shape=(224,224,3),pooling='avg')
print("resnet loaded")

vocab = np.load('vocab.npy', allow_pickle=True)
vocab = vocab[()]
#print(vocab.dtype)
inv_vocab = {v:k for k,v in vocab.items()}
embedding_size = 128
max_len = 37
vocab_size = len(vocab)

image_model = Sequential()
#add a layer of embedding_size as the number of units for the image vector
image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))



language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))



conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)

# model.load_weights("../input/model_weights.h5")
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.load_weights('mine_model_weights.h5')
print("model loaded")


app =Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] =1
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after',methods=['GET', 'POST'])

def after():
    global model, vocab, inv_vocab, incept_model
    
    file = request.files['file1']
    file.save("static/file.jpg")
    
    img = cv2.imread('static/file.jpg')
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, (224,224,))
    img = np.resize(img,(1,224,224,3))
    
    features = incept_model.predict(img).reshape(1,2048)
    
    text_in = ['startofseq']
    final = ''
    
    count=0
    
    while tqdm(count<20):
        count+=1
        
        encoded = []
        #caption = ''
        for i in text_in:
            encoded.append(vocab[i])

        encoded = [encoded]

        padded = pad_sequences(encoded, padding='post', truncating='post', maxlen=37)


        sampled_index = np.argmax(model.predict([features, padded]))

        sampled_word = inv_vocab[sampled_index]

        final = final + ' ' + sampled_word
            
        if sampled_word == 'endofseq':
            break

        text_in.append(sampled_word)
        
        
        
    
    print("getting captions")
    
    return render_template('predict.html',final = final)
    

if __name__ == '__main__':
    app.run(debug=True)
