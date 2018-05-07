from fstring import fstring
import tensorflow as tf
import tflearn
import numpy as np
import re
#from model import TagSpace
from sklearn.utils import shuffle
from reader import load_csv, VocabDict, load_txt

'''
parse
'''

tf.app.flags.DEFINE_integer('num_epochs', 50, 'number of epochs to train')
tf.app.flags.DEFINE_integer('batch_size', 20, 'batch size to train in one step')
tf.app.flags.DEFINE_integer('labels', 5838, 'number of label classes')
tf.app.flags.DEFINE_integer('word_pad_length', 60, 'word pad length for training')
tf.app.flags.DEFINE_float('learn_rate', 1e-2, 'learn rate for training optimization')
tf.app.flags.DEFINE_boolean('shuffle', True, 'shuffle data FLAG')

FLAGS = tf.app.flags.FLAGS

num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
tag_size = FLAGS.labels
word_pad_length = FLAGS.word_pad_length
learn_rate = FLAGS.learn_rate
lr_decr = (learn_rate - (1e-9))/num_epochs

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)
def token_parse(iterator):
  for value in iterator:
    return TOKENIZER_RE.findall(value)

tokenizer = tflearn.data_utils.VocabularyProcessor(word_pad_length, tokenizer_fn=lambda tokens: [token_parse(x) for x in tokens])
label_dict = VocabDict()

def string_parser(arr, fit):
  if fit == False:
    return list(tokenizer.transform(arr))
  else:
    return list(tokenizer.fit_transform(arr))

l = word_pad_length
tN = tag_size
N=100000
d=10
K=5
H=1000
m=0.05
lr = tf.placeholder('float32', shape=[1], name='lr')
doc = tf.placeholder('float32', shape=[None, l], name='doc')
tag_flag = tf.placeholder('float32', shape=[None, tN], name='tag_flag')

doc_embed = tflearn.embedding(doc, input_dim=N, output_dim=d)
lt_embed = tf.Variable(tf.random_normal([tN, d], stddev=0.1))
net = tflearn.conv_1d(doc_embed, H, K, activation='tanh')
net = tflearn.max_pool_1d(net, K)
net = tflearn.tanh(net)
logit = tflearn.fully_connected(net, d, activation=None)

zero_vector = tf.zeros(shape=(1,1), dtype=tf.float32)

logit = tf.expand_dims(logit, 1)
logit_set = tf.concat([logit for i in range(tN)], axis=1)

tag_flag_ex = tf.expand_dims(tag_flag, 2)
tg = tf.concat([tag_flag_ex for i in range(d)], axis=2)

tag_logit = tf.reduce_sum(tf.multiply(logit_set, tf.multiply(tf.ones_like(tg), lt_embed)), axis=2)

positive_logit = tf.reduce_sum(tf.multiply(logit_set, tf.multiply(tg, lt_embed)), axis=2)
random_sample = tf.random_uniform([batch_size],minval=0,maxval=1,dtype=tf.float32)
#f_test = tf.not_equal(positive_logit, zero_vector)
#f_test1 = tf.boolean_mask(positive_logit[0], f_test[0])
#f_test2 = tf.boolean_mask(positive_logit[1], f_test[1])
#f_test3 = tf.expand_dims(tf.boolean_mask(positive_logit[0], f_test[0])[tf.cast(tf.floor(tf.multiply(random_sample[0],tf.cast(tf.shape(tf.boolean_mask(positive_logit[0], f_test[0])), dtype=tf.float32))), dtype=tf.int32)[0]], -1)
#f_positive = tf.map_fn(lambda x: (tf.boolean_mask(x[0], x[1]), True), (positive_logit[:1], tf.not_equal(positive_logit[:1], zero_vector)))
f_positive = tf.map_fn(lambda x: (tf.expand_dims(tf.boolean_mask(x[0], x[1])[tf.cast(tf.floor(tf.multiply(x[2],tf.cast(tf.shape(tf.boolean_mask(x[0], x[1])), dtype=tf.float32))), dtype=tf.int32)[0]], -1), True, x[2]), (positive_logit, tf.not_equal(positive_logit, zero_vector), random_sample))
positive = tf.reduce_min(f_positive[0], axis=1)

tag_flag_ex = tf.expand_dims(1-tag_flag, 2)
tg = tf.concat([tag_flag_ex for i in range(d)], axis=2)
negative_logit = tf.reduce_sum(tf.multiply(logit_set, tf.multiply(tg, lt_embed)), axis=2)
random_sample_negative = tf.random_uniform([batch_size],minval=0,maxval=1,dtype=tf.float32)

f_negative = tf.map_fn(lambda x: (tf.expand_dims(tf.boolean_mask(x[0], x[1])[tf.cast(tf.floor(tf.multiply(x[2],tf.cast(tf.shape(tf.boolean_mask(x[0], x[1])), dtype=tf.float32))), dtype=tf.int32)[0]], -1), True, x[2]), (negative_logit, tf.not_equal(negative_logit, zero_vector), random_sample_negative))
#f_negative = tf.map_fn(lambda x: (tf.boolean_mask(x[0], x[1]), True), (negative_logit, tf.not_equal(negative_logit, zero_vector)))
negative = tf.reduce_max(f_negative[0], axis=1)

f_loss = tf.reduce_mean(tf.reduce_max([tf.reduce_min([tf.expand_dims(m - positive + negative,1), tf.expand_dims(tf.fill([tf.shape(doc)[0]], 10e7),1)], axis=0), tf.zeros([tf.shape(doc)[0], 1])], axis=0))
params = tf.trainable_variables()

opt = tf.train.AdamOptimizer(learning_rate=lr[0])
gradients = tf.gradients(f_loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
op = opt.apply_gradients(zip(clipped_gradients, params))
train_opts = [op, f_loss, logit, lt_embed, f_positive[0][0], f_negative[0][0]]
test_opts = [tag_logit]

with tf.Session() as sess:
  #with tf.device('/cpu:0'):
  sess.run(tf.global_variables_initializer())

  words, tags = load_txt("./data/toutiao_data.txt", target_dict=label_dict)
  ret_dict = {}
  for k in label_dict.dict:
    ret_dict[label_dict.dict[k]] = k
  print len(ret_dict), len(tags[0])
  #words, tags = load_csv('./data/ag_news_csv/train.csv', target_columns=[0], columns_to_ignore=[1], target_dict=label_dict)
  if FLAGS.shuffle == True:
    words, tags = shuffle(words, tags)

  words = string_parser(words, fit=True)
  word_input = tflearn.data_utils.pad_sequences(words, maxlen=word_pad_length)
  total = len(word_input)
  step_print = int((total/batch_size) / 13)
  global_step = 0

  print 'start training'
  for epoch_num in range(num_epochs):
    epoch_loss = 0
    step_loss = 0
    for i in range(int(total/batch_size)):
      batch_input, batch_tags = (word_input[i*batch_size:(i+1)*batch_size], tags[i*batch_size:(i+1)*batch_size])
      #batch_tags = np.array(batch_tags)
      result = sess.run(train_opts, feed_dict={'doc:0': batch_input, 'tag_flag:0': batch_tags, 'lr:0': [learn_rate]})
      #result = sess.run(train_opts, feed_dict={'doc:0': batch_input, 'tag_flag:0': batch_tags, 'lr:0': [lr]})
      step_loss += result[1]
      epoch_loss += result[1]
      if i % step_print == 0:
        print 'step_log: (epoch: %d, step: %d, global_step: %d, Loss:%f)' % (epoch_num, i, global_step, float(step_loss)/step_print)
        step_loss = 0
      global_step += 1
    print 'epoch_log: (epoch: %d, global_step: %d), Loss:%f)' % (epoch_num, global_step, float(epoch_loss)/(total/batch_size))
    learn_rate -= lr_decr

  #words, tags = load_csv('./data/ag_news_csv/test.csv', target_columns=[0], columns_to_ignore=[1], target_dict=label_dict)
  #words = string_parser(words, fit=True)
  #word_input = tflearn.data_utils.pad_sequences(words, maxlen=word_pad_length)
  #total = len(word_input)
  rs = 0.
  fw = open("result", "w+")
  for i in range(int(total/batch_size)):
    batch_input, batch_tags = (word_input[i*batch_size:(i+1)*batch_size], tags[i*batch_size:(i+1)*batch_size])
    result = sess.run(test_opts, feed_dict={'doc:0': batch_input, 'tag_flag:0': np.ones_like(batch_tags)})
    arr = result[0]
    for j in range(len(batch_tags)):
      #rs+=np.sum(np.argmax(arr[j]) == np.argmax(batch_tags[j]))
      batch_tags_tmp = np.array(batch_tags[j])
      y = np.where(batch_tags_tmp==1.0)[0]
      n = len(y)
      h = arr[j].argsort()[len(arr[j]) - n:]
      #print y, h
      if np.argmax(arr[j]) in y:
        rs += 1
      ret_real = []
      ret_predict = []
      for ty in list(y):
        ret_real.append(ret_dict[ty])
      for th in list(h):
        ret_predict.append(ret_dict[th])
      fw.write(",".join(ret_real) + "\t" + ",".join(ret_predict) + "\n")
  print 'Test accuracy: %f' % (float(rs)/total)
  fw.close()
  sess.close()

