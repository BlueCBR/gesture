# Ergebnisse von Azure Auto ML:
# MinMaxScaler, LightGBM
# StandardScalerWrapper, LightGBM
# StandardScalerWrapper, XGBoostClassifier
# RobustScaler, KNN  -- nearest neighbour
# StandardScalerWrapper, RandomForest
# StandardScalerWrapper, KNN

import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# ---------------- boosted tree error ?? -----------
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0],True)
# https://github.com/tensorflow/tensorflow/issues/34118

plain_train = pd.read_csv("samples.csv")

# plain_train.head()

plain_features_pd = plain_train.copy()
# print ('Unique Labels:')
# print (plain_features_pd['sampleType'].unique())  # should be 1 and 2 
plain_features_pd = plain_features_pd.sample(frac=1).reset_index(drop=True)  # shuffle rows

plain_labels = plain_features_pd.pop('sampleType')
# print (plain_labels)
# plain_features = np.array(plain_features_pd)
# print (plain_features_pd)

# plain_labels.replace(2, 0)   # sample has 1 and 2 -> change to 0 and 1 further down. Is considered 2 labels -> n-class = 3 !

# normalize input values:
plain_features_pd =((plain_features_pd-plain_features_pd.min())/(plain_features_pd.max()-plain_features_pd.min()))

# sampledata[0] training 
# sampledata[1] test / eval
sdata = np.split(plain_features_pd, [900], axis=0)
slabel = np.split(plain_labels, [900], axis=0)

# print ('sdata[0]')
# print (sdata[0])
# print ('slabel[0]')
# print (slabel[0])

# das wäre ohne shuffle :
# print ("1. Reihe Anfang acc -1077,-909,-2256,  gy -129,256,-175,   acc -1034,-858,-1649,   gy -53,317,-108")
# print ("1. Reihe Ende   Ax198 -812 -470 -1188    gx198  221 -241 137       AX199 -884 -767 -1946    gx199 167 -268 147")

feature_columns = []
for feature_name in plain_features_pd.columns:
  feature_columns.append(tf.feature_column.numeric_column(feature_name,
                                                          dtype=tf.float32))


# Use entire batch since this is such a small dataset.
# NUM_EXAMPLES = len(slabel[0])

def make_input_fn(X, y, n_epochs=None):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    # For training, cycle thru dataset as many times as need (n_epochs=None).    
    dataset = dataset.repeat(n_epochs)
    # In memory training doesn't use batching.
    dataset = dataset.batch(len(y)) #  NUM_EXAMPLES)
    return dataset
  return input_fn

# Training and evaluation input functions.
train_input_fn = make_input_fn(sdata[0], slabel[0])
eval_input_fn = make_input_fn(sdata[1], slabel[1], n_epochs=1)

# print ('---------------------------------  logistic regression model  --------------------------------')
# linear_est = tf.estimator.LinearClassifier(feature_columns, n_classes=3)
# print ('.. train .. ')
# Train model.
# linear_est.train(train_input_fn, max_steps=100)

# Evaluation.
# result = linear_est.evaluate(eval_input_fn)
# clear_output()
# print(pd.Series(result))
# accuracy          1.000000
# average_loss      0.000509
# loss              0.000509
# global_step     100.000000


print ('----------------------------------  Boosted Trees model  -------------------------------')
n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns,
                                          n_batches_per_layer=n_batches,
                                          n_classes=3)
                                          # n_trees = 50)
                                          # train_in_memory=True)
#                                          # only for n-class < 2 -> center_bias=True )
print ('.. train .. ')

# The model will stop training once the specified number of trees is built, not 
# based on the number of steps.
est.train(train_input_fn, max_steps=200)

# Eval.
result = est.evaluate(eval_input_fn)
print(pd.Series(result))

# accuracy          1.000000
# average_loss      0.274033
# loss              0.274033
# global_step     200.000000

