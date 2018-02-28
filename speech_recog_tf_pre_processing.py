# In this file, we will be extracting tensorflow speech command dataset
# the dataset has 30 classes and over 65000 audios, we also have a testfilelist
# and validation file list, we will use them to create our test set
import os
import librosa

DATA_PATH = os.path.abspath('speech_command_data_tf')
DATA_DIR = os.path.join(DATA_PATH, 'data')
SR = 16000  # sampling rate for reading files
print('data paths set')
# making a dictionary of all classes
classes = os.listdir(DATA_DIR)
classDicti = {}

for i in range(len(classes)):
	classDicti[i] = classes[i]

# reading all the files along with their names

audios = []
filenames = []

for eachClass in classes:
	filePath = os.path.join(DATA_DIR, eachClass)
	for file in os.listdir(filePath):
		y, sr = librosa.load(os.path.join(filePath, file), sr = SR)
		audios.append(y)
		filenames.append(eachClass + '/' + file)
print('reading of files done with librosa')
# now here we have a problem that even after sampling at 16000 rate of 1 sec long audio
# we don't get 16000 sample length, hence we will pad those audios having less than 16000 samples with zeros
audios = np.array(audios)
print(len(audios))
for i in range(len(audios)):
    #print(xTrain.shape)
    if audios[i].shape[0] != SR:
        zeros = np.zeros(SR - (audios[i].shape[0]))
        audios[i] = np.concatenate((audios[i],zeros))
        #or we can use np.hstack() also
        print(audios[i].shape)