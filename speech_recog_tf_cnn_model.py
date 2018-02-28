# In this file, we will be making our CNN to feed extracted MFCC features
# we will load mfcc features from pickle dump of train set to train the model 
# and then save the model
import tensorflow as tf
import numpy as np

DATA_PATH = os.path.abspath('speech_command_data_tf')
DATA_DIR = os.path.join(DATA_PATH, 'data')
SR = 16000  # sampling rate for reading files
CLASS_NUM = len( os.listdir(DATA_DIR) )
MFCC_NUM = 40
HOP_LENGTH = 512
FRAME_COUNT = math.ceil(SR / HOP_LENGTH) 

SEED = 2
np.random.seed(SEED)
# load data into memory
with open(os.path.join(DATA_PATH, 'xTrain_mfcc.pickle'), 'rb') as f:
	xTrain, yTrain = pickle.load(f)

with open(os.path.join(DATA_PATH, 'xTest_mfcc_powerDb.pickle'), 'rb') as f:
	xTest, xTestPoweDb, yTest = pickle.load(f)

print('loaded data into memory')
# now we will convert labels for both train and test into one hot

def one_hot_encoder(labelsList, classes):
	n = len(labelsList)
	out = np.zeros((n, classes))
	out[range(n), labelsList] = 1
	return out

yTrainOneHot = one_hot_encoder(yTrain, classes = CLASS_NUM)
yTestOneHot = one_hot_encoder(yTest, classes = CLASS_NUM)
print('converted labels to one hot')

# now we will make a function that will create batches of data for the given
# batch size that will be used for training
def get_next_batch(batchSize, dataX, dataY):
	# this fn will take out batches of batchSize from training data
	indexes = list(range(len(dataX)))
	np.random.shuffle(indexes)
	batch = indexes[:batchSize]
	# now the trick is to convert the words into their respective integer through
	# wordIndexMap and then feed into Rnn
	X = [ dataX[i] for i in batch]
	Y = [ dataY[i] for i in batch]
	return X, Y

# now here is an issue with tensorflow, In the convolution filter, we can't directly pass
# a list like [3, 3, 1, 32] because it considers it a list which has rank 1, but it requires rank 4
#input, hence we need to create a variable first of the required shape
def create_filter(shape):
	filters = tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
	return filters

# now we can start building our model

# hyper Parameters
BATCH_SIZE = 512
EPOCHS = 1000
LOG_DIR = os.path.join(DATA_PATH, 'tmp')
DROP_OUT = 0.5

with tf.name_scope('placeholders') as scope:

	# here we know the input shape is (40, 32, 1), i.e (mfcc_count, frameCnt, 1)
	x = tf.placeholder(shape = [None, MFCC_NUM, FRAME_COUNT, 1], name = 'input', dtype = tf.float32)
	y = tf.placeholder(shape = [None, CLASS_NUM], name = 'output', dtype = tf.float32)

with tf.name_scope('cnn') as scope:

	conv1 = tf.nn.relu(tf.nn.conv2d(x, filter = create_filter([3, 3, 1, 32]), strides = [1, 1, 1, 1],
		padding = 'SAME', name = 'conv1'))
	# here we have defined our first layer with a convolution window of 3x3 and 32 feature maps
	# the 1 in between shows that initially we are having only 1 feature map (tht is the mono channel of image)
	# after this image size will be (40 ,32, 32)
	maxPool1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME',
		name = 'maxPool1')
	# after this the image size will be reduced to 20x16x32
	conv2 = tf.nn.relu(tf.nn.conv2d(maxPool1, filter = create_filter([3, 3, 32, 64]), strides = [1, 1, 1, 1],
	 padding = 'SAME', name = 'conv3'))
	# after this, 20x16x64
	maxPool2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME',
		name = 'maxPool2')
	# after this 10x8x64
	conv3 = tf.nn.relu(tf.nn.conv2d(maxPool2, filter = create_filter([3, 3, 64, 128]), strides = [1, 1, 1, 1],
	 padding = 'SAME', name = 'conv5'))
	# after this 10x8x128
	maxPool3 = tf.nn.max_pool(conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME',
		name = 'maxPool3')
	# now the image has reduced to 5x4x128 after this maxpooling
	# here we can see that we have reduced the height, width dimensions of the image, but increased 
	# the number of features of the image, hence making a balance between the number of neurons.
	flatten = tf.reshape(maxPool3, (-1, 5 * 4 * 128))
	# here we have unrolled the whole structure of image into one dimensional tensors so that
	# we can connect it to dense layers

with tf.name_scope('dense') as scope:

	dense1 = tf.nn.relu(tf.layers.dense(flatten, units = 1024, name = 'dense1'))
	# thus here the neuron count in our model is 1024*6*6*128
	keep_prob = tf.placeholder(tf.float32)
	# the dropout ratio has to be a placeholder which will be fed value at training like x
	dropOut1 = tf.nn.dropout(dense1, keep_prob = keep_prob, name = 'drop1')
	# probability that an element is kept is keep_prob
	dense2 = tf.nn.relu(tf.layers.dense(dropOut1, units = 512, name = 'dense2'))
	dropOut2 = tf.nn.dropout(dense2, keep_prob = keep_prob, name = 'drop2')

with tf.name_scope('out_layer') as scope:

	finalOutput = tf.nn.softmax(tf.layers.dense(dropOut2, units = CLASS_NUM, name = 'Final_layer_out'))

with tf.name_scope('train') as scope:

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = finalOutput, labels = y))
	tf.summary.histogram('loss', loss)

	optimizer = tf.train.AdamOptimizer()
	train = optimizer.minimize(loss)

with tf.name_scope('accuracy') as accuracy:

	correctPrediction = tf.equal(tf.argmax(finalOutput, axis = 1), tf.argmax(y, axis = 1))
	accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32)) * 100

init = tf.global_variables_initializer()
saver = tf.Saver()
merge = tf.summary.merge_all()

with tf.Session() as sess:

	sess.run(init)
	trainWriter = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
	testWriter = tf.summary.FileWriter(LOG_DIR + '/test', sess.graph)

	for i in range(EPOCHS):
		batchX, batchY = data.get_next_batch(BATCH_SIZE, trainX, trainY)
		summary, _ = sess.run(train, feed_dict = {x : batchX, y : batchY, keep_prob : DROP_OUT})
		saver.save(sess, os.path.join(MODEL_DIR, 'model_1'), global_step = 100)

		if i % 100 == 0:
			# calculating train accuracy
			acc, lossTmp = sess.run([accuracy, loss], feed_dict = {x : batchX, y : batchY, keep_prob : DROP_OUT})
			print('Iter: '+str(i)+' Minibatch_Loss: '+"{:.6f}".format(lossTmp)+' Train_acc: '+"{:.5f}".format(acc))

	for i in range(5):
		# calculating test accuracy
		testBatchX, testBatchY = data.get_next_batch(BATCH_SIZE, Xtest, Ytest)
		testAccuracy = sess.run(accuracy, feed_dict = {x : testBatchX, y : testBatchY, keep_prob : DROP_OUT})
		print('test accuracy : ', testAccuracy)

