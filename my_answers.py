import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
from string import punctuation
import re
import string


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    end=len(series)-window_size
    for i in range(0,end,1): 
        X2= i +window_size
        X.append(series[i:X2]) 
        y.append(series[X2:X2+1])
        


    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    print ('--- the input X will look like ----')
    print (X)

    print ('--- the associated output y will look like ----')
    print (y)
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    #based on keras LSTM documentation. 
    hidden_units = 5
    model = Sequential()
  #  model.add(Embedding(vocabulary, hidden_units, input_length=window_size))
    model.add(LSTM(hidden_units,input_shape=(window_size,1) ))

    model.add (Dense(1),activation ='linear')
    #model.compile(loss='mean_squared_error',optimizer='adam')    
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):

    # As you can see I tried a lot here. I got a couple things to work but used a regular expression as it's much cleaner.
    # used https://regex101.com/ to test my regular expressions out on some copied and pasted text! 



   # chars_to_keep = ['!', ',', '.', ':', ';', '?']
   # punct = list(punctuation)
   # punct_set = set(punct)
   # puncter_set=set(chars_to_keep)

    #chars_to_remove=punct_set.difference(puncter_set)
    #chars_to_remove= str(chars_to_remove)
    #chars_to_remove=chars_to_remove.replace(',','')
    #chars_to_remove=chars_to_remove.replace("'",'')
    #chars_to_remove=chars_to_remove.replace(" ","")
    #punct =punct.remove('!')
   # punct = re.sub("!|,|.|:|;|?",'',punctuation)
    #whitelist = set(chars_to_keep)
    #punct = re.sub('[!,.:;?',punctuation)
   # punct = ''.join(filter(whitelist.__contains__, punct))
    #chars_to_keep =''.join(chars_to_keep)
    #re.sub(chars_to_keep, "", )
   # punct_list= list(punctuation)
   # punct_set= set(punct_list)
   # char_set= set(chars_to_keep)
   
    text= re.sub(r'[^a-z!,.:;? ]', ' ', text)

    return text 


    #punct_set = punct_set.difference(char_set)

    #punct = list(punct_set)
    #punct= str(punct_list)

   # chars_to_remove = punctuation.strip(chars_to_keep)
#take this we're given and subtract it out of the punctuation set. 
    #punct=''.join(punct.split())
    #print(chars_to_remove)
   # print(punctuation)
    #print(punct)
    #remove the whitespace so it's a cohesive string like the one we got ( otherwise we'll delete the spaces!)
    #text = ''.join([c for c in text if c not in chars_to_remove])
    #remove the things in that string. 
    #return text
    #return text without the characters we don't want. 


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []


# Heavily borrowed from the previous windowing function we built earlier. However the nice thing here is we don't need to shape the outputs nearly as much because they are lists
# Lists are the native data structure in python!!!
   
    end = len(text) - window_size
    for i in range(0, end, step_size):
        inputs.append(text[i : i + window_size])
        outputs.append(text[i + window_size])
    return inputs,outputs





    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):

    # This is heavily borrowed from the previous network we built. The notable changes are 
    # the input shape, the dense layer being set to number of chars vs 1 and using softmax to select the next character! 
    # also the LSTM having 200 units. 
    model = Sequential()

  #  model.add(Embedding(vocabulary, hidden_units, input_length=window_size))
    model.add(LSTM(200,input_shape=(window_size,num_chars )))

    model.add (Dense(num_chars,activation ='softmax'))
    #model.compile(loss='mean_squared_error',optimizer='adam')    
    return model
