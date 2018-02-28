import os
import pickle
import math
import librosa.feature

DATA_PATH = os.path.abspath('speech_command_data_tf')
DATA_DIR = os.path.join(DATA_PATH, 'data')
SR = 16000  # sampling rate for reading files
MFCC_NUM = 40
HOP_LENGTH = 512
NFFT =2048 # window length for n- fast fourier transform
FRAME_COUNT = math.ceil(SR / HOP_LENGTH) 
print('data paths set')

# we will now load our saved test and train data
with open(os.path.join(DATA_PATH, 'extracted_data.pickle'), 'rb') as f:
	xTrain, yTrain, xTest, yTest = pickle.load(f)
print('loaded train and test data')

# now we need to convert the xTrain and xTest data to mfcc feature so that we can feed them in 
# CNN model later
xTrainNew = []
for i in range(len(xTrain)):
    mfcc = librosa.feature.mfcc(y = xTrain[i], sr = SR, n_mfcc = MFCC_NUM, hop_length = HOP_LENGTH)
    #normalizing the mfcc
    mfcc_mean = mfcc - np.mean(mfcc, axis = 0)
    xTrainNew.append(mfcc_mean)

xTrainNew = np.array(xTrainNew).reshape((len(xTrainNew), MFCC_NUM, FRAME_COUNT, 1))
print('mfcc feature extracted for xTrain')

# now we will save these extracted feature as a pickle dump again
with open(os.path.join(DATA_PATH, 'xTrain_mfcc.pickle'), 'wb') as f:
	pickle.dump([xTrainNew, yTrain], f)

# now we need to do the same with test
# but the differrence will be that this time we will also calculate the average power of each audio
# in DB, so that at the time of prediction, we can reject files that have silence
# mostly cut off will be set around -45 db

xTestNew = []
for i in range(len(xTest)):
    mfcc = librosa.feature.mfcc(y = xTest[i], sr = SR, n_mfcc = MFCC_NUM, hop_length = HOP_LENGTH)
    #normalizing the mfcc
    mfcc_mean = mfcc - np.mean(mfcc, axis = 0)
    xTestNew.append(mfcc_mean)

    # now we will calculate power in db

    power = np.abs(librosa.core.stft(xTest[i], n_fft = NFFT))
    powerDb = librosa.core.power_to_db(power, ref = 1.0)
	xTestPowerDb.append(powerDb)
xTest = np.array(xTest)
xTestPowerDb = np.array(xTestPowerDb)
print('mfcc feature and power calculated for xTest')

with open(os.path.join(DATA_PATH, 'xTest_mfcc_powerDb.pickle'), 'wb') as f:
	pickle.dump([xTest, xTestPowerDb, yTest], f)
print('dumped for test')