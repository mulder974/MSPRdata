from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import database
import pickle
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd  # To play sound in the notebook

from matplotlib.pyplot import specgram

def create_waveplot(data, sr, emotion):
    fig = plt.figure(figsize=(15, 5))
    plt.title('Waveplot for audio with {} emotion'.format(emotion), size=15)
    librosa.display.waveshow(data, sr=sampling_rate)
    return fig

def create_spectrogram(voice, sr, e):
    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    fig = plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    return fig



angry_file = "data/Actor_01/03-01-05-01-01-01-01.wav"
happy_file = "data/Actor_01/03-01-03-01-01-01-01.wav"
neutral_file = "data/Actor_01/03-01-01-01-01-01-01.wav"


st.title("Data Visualisation")


st.write("Count of emotions :")
data = database.query_all()

data_dic = {"emotion": []}

for tupl in data:
    data_dic["emotion"].append(tupl[1])

df_emotion = pd.DataFrame(data=data_dic)

df_emotion = pd.DataFrame(data=data_dic).reset_index()

df_emotion_count = df_emotion.groupby("emotion").count().reset_index()
df_emotion_count = df_emotion_count.rename(columns={'index': 'count'})


# Bar_chart_emotion
st.bar_chart(data = df_emotion_count, x="emotion", y = "count")

st.markdown("***")


# Figures Angry
st.write(" Visualisation of an angry voice :")
data, sampling_rate = librosa.load(angry_file)
fig_angry = create_waveplot(data, sampling_rate, "angry")
st.pyplot(fig_angry)
fig_2_angry = create_spectrogram(data, sampling_rate, "angry")
st.pyplot(fig_2_angry)
st.audio(angry_file)
st.markdown("***")

# Figures Happy
st.write(" Visualisation of an happy voice :")
data, sampling_rate = librosa.load(happy_file)
fig_happy = create_waveplot(data, sampling_rate, "happy")
st.pyplot(fig_happy)
fig2_happy = create_spectrogram(data, sampling_rate, "happy")
st.pyplot(fig2_happy)
st.audio(happy_file)
st.markdown("***")

# Figures Neutral
st.write(" Visualisation of a neutral voice :")
data, sampling_rate = librosa.load(neutral_file)
fig_neutral = create_waveplot(data, sampling_rate, "neutral")
st.pyplot(fig_neutral)
fig2_neutral = create_spectrogram(data, sampling_rate, "neutral")
st.pyplot(fig2_neutral)

st.audio(neutral_file)




