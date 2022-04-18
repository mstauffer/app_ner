import os
import streamlit as st
import pickle as pkl
import numpy as np
import nltk
from tensorflow import keras
from nltk.tokenize import RegexpTokenizer, word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

def set_cpu_only():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def call_nltk_punkt():
    nltk.download('punkt')

def passito_passito(texto, max_len):
    # se keep_punctuation == True
    texto_tokenizado = word_tokenize(texto)
    # caso contrário
    # tokenizador = Regexptokenizer('\w+')
    # texto_tokenizado = tokenizador.tokenize(texto)
    trechos = []
    if len(texto_tokenizado) > max_len:
        num_listas = len(texto_tokenizado) // max_len
        idx_tokens_restantes = num_listas*max_len
        inicio = 0
        for num in range(num_listas):
            fim = (num+1)*max_len
            lista = texto_tokenizado[inicio:fim]
            inicio = (num+1)*max_len
            trechos.append(lista)
            if num == num_listas-1:
                lista = texto_tokenizado[idx_tokens_restantes:]
                trechos.append(lista)
    else:
        trechos.append(texto_tokenizado)
    return trechos

def get_preds(texto, modelo, max_len, word2idx, idx2tag):
    trechos = passito_passito(texto, max_len)
    preds_final = []
    texto_tok_final = []

    for trecho in trechos:
        print(f'Qtd de tokens no texto dado: {len(trecho)}')
        index_textos = []
        for word in trecho:
            if word in word2idx.keys():
                index_textos.append(word2idx[word])
            else:
                index_textos.append(word2idx['UNK'])
        index_textos = pad_sequences(maxlen=max_len, sequences=[index_textos], padding='post', value=len(word2idx)-1)
        index_textos = np.array(index_textos)
        # triggered tf.function retracing
        # pred = modelo.predict(index_textos)
        pred = modelo(index_textos)
        pred = np.argmax(pred, axis=-1)
        for token, rotulo in zip(trecho, pred[0]):
            texto_tok_final.append(token)
            preds_final.append(idx2tag[rotulo])

    return texto_tok_final, preds_final

def carregar_pickle(path):
    with open(path, 'rb') as f:
        objeto_pickle = pkl.load(f)
    
    return objeto_pickle

if 'model' not in st.session_state:
    st.session_state['model'] = keras.models.load_model('model.h5')

if 'tag2idx' not in st.session_state:
    st.session_state['tag2idx'] = carregar_pickle('tag2idx.pkl')

if 'idx2tag' not in st.session_state:
    st.session_state['idx2tag'] = {k: v for v, k in st.session_state['tag2idx'].items()}

if 'word2idx' not in st.session_state:
    st.session_state['word2idx'] = carregar_pickle('word2idx.pkl')

if 'punkt' not in st.session_state:
    st.session_state['punkt'] = call_nltk_punkt()

st.title("Detector de entidades nomeadas")

txt = st.text_area('Insira um ato para encontrar entidades')

exemplo_tok, preds = get_preds(txt, st.session_state['model'],
                               128, st.session_state['word2idx'],
                               st.session_state['idx2tag'])
for token, tag in zip(exemplo_tok, preds):
    st.write(f'{token}\t {tag}')
