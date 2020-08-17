from preprocessor import *
from model.JoVA import JoVA
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import argparse

# hinge mode
using_hinge= 1  # set 1 to incorporate ranking loss.

current_time = time.time()


parser = argparse.ArgumentParser(description='JoVA')
parser.add_argument('--beta_value', type=float, default=0.01) #0.01 for both Movie-Lens and Pinterest and 0.001 for Yelp.
parser.add_argument('--train_epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int,default=1500)
parser.add_argument('--using_hinge', type=int, default=using_hinge)
parser.add_argument('--base_lr', type=float, default=0.003)
parser.add_argument('--decay_epoch_step', type=int, default=50,help="decay the learning rate for each n epochs")




args = parser.parse_args()
tf.set_random_seed(1000)
np.random.seed(1000)


data_name = ml1m
train_R, test_R = data_name.test()

num_users=train_R.shape[0]
num_items=train_R.shape[1]


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    JoVA = JoVA(sess,args,
                      num_users,num_items,
                    train_R,  test_R)
    JoVA.run()
