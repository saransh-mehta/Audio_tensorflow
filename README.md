# Audio
built a CNN to recognize 30 keywords from tensorflow's speech command dataset

The tensorflow speech command detection dataset has around 65,000 audio samples of one second each. There are total 30 command keywords, viz. ‘on’ , ‘off’, ‘left, ‘right’, ‘stop’, ‘bed’ etc.
So, in the project CNN has been used to classify accurately the audios in these 30 classes.  First Mel-Cepstral Coefficient were extracted, 40 coefficients with a hop length of 512 with sampling rate at 16000 with librosa. Thus each audio now becomes a numpy array of (40, 32, 1) . Thus each audio is now transformed in a 40x32 image. Thus the images are now fed into the CNN model to classify into the required class. Here log power through spectrogram is also calculated for the test set, so that a threshold power in decibels is set around -45 db. Audios having power in log scale below than this are apparently noise or silence. Hence, the system can distinguish between mere silence and audio. CNN’s give quite promising result on this because they are tolerant to noises in audio.
Tensorflow along with librosa in python have been used.
http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz.
