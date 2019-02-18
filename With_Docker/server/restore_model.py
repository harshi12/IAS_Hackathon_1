#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
eps = np.finfo(float).eps


# In[2]:


# load data
#filename = input("Enter the name of test file")
test_data = pd.read_csv("test.csv")


# In[3]:


test_data


# In[4]:


# Feature Engineering
from sklearn.preprocessing import Imputer

def nan_padding(data, columns):
    for column in columns:
        imputer=Imputer()
        data[column]=imputer.fit_transform(data[column].values.reshape(-1,1))
    return data


nan_columns = ["Age", "SibSp", "Parch"]

test_data = nan_padding(test_data, nan_columns)


# In[5]:


#save PassengerId for evaluation
test_passenger_id=test_data["PassengerId"]


# In[6]:


def drop_not_concerned(data, columns):
    return data.drop(columns, axis=1)

not_concerned_columns = ["PassengerId","Name", "Ticket", "Fare", "Cabin", "Embarked"]
test_data = drop_not_concerned(test_data, not_concerned_columns)


# In[7]:


test_data.head()


# In[8]:


def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data


dummy_columns = ["Pclass"]
test_data=dummy_data(test_data, dummy_columns)


# In[9]:


test_data.head()


# In[10]:


from sklearn.preprocessing import LabelEncoder
def sex_to_int(data):
    le = LabelEncoder()
    le.fit(["male","female"])
    data["Sex"]=le.transform(data["Sex"]) 
    return data

test_data = sex_to_int(test_data)
test_data.head()


# In[11]:


from sklearn.preprocessing import MinMaxScaler

def normalize_age(data):
    scaler = MinMaxScaler()
    data["Age"] = scaler.fit_transform(data["Age"].values.reshape(-1,1))
    return data
test_data = normalize_age(test_data)
test_data.head()


# In[12]:


test_data.shape


# In[13]:


from collections import namedtuple

def build_neural_network(hidden_units=10):
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None, test_data.shape[1]])
    labels = tf.placeholder(tf.float32, shape=[None, 1])
    learning_rate = tf.placeholder(tf.float32)
    is_training=tf.Variable(True,dtype=tf.bool)
    
    initializer = tf.contrib.layers.xavier_initializer()
    fc = tf.layers.dense(inputs, hidden_units, activation=None,kernel_initializer=initializer)
    fc=tf.layers.batch_normalization(fc, training=is_training)
    fc=tf.nn.relu(fc)
    
    logits = tf.layers.dense(fc, 1, activation=None)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    cost = tf.reduce_mean(cross_entropy)
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.nn.sigmoid(logits)
    correct_pred = tf.equal(tf.round(predicted), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Export the nodes 
    export_nodes = ['inputs', 'labels', 'learning_rate','is_training', 'logits',
                    'cost', 'optimizer', 'predicted', 'accuracy']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph


# In[14]:



model = build_neural_network()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('titanic.ckpt.meta')

    saver.restore(sess,"titanic.ckpt")
    feed={
        model.inputs:test_data,
        model.is_training:False
    }
    test_predict=sess.run(model.predicted,feed_dict=feed)
    
test_predict[:10]


# In[15]:


from sklearn.preprocessing import Binarizer
binarizer=Binarizer(0.5)
test_predict_result=binarizer.fit_transform(test_predict)
test_predict_result=test_predict_result.astype(np.int32)
test_predict_result[:10]


# In[16]:


passenger_id=test_passenger_id.copy()
evaluation=passenger_id.to_frame()
evaluation["Survived"]=test_predict_result
evaluation


# In[17]:


evaluation.to_csv("evaluation_submission.csv",index=False)


# In[18]:


actualDF = pd.read_csv(r"gender_submission.csv")


# In[19]:


def validate_test(predicted, actual):
    true_positive = 0
    true_negative = 0 
    false_negative = 0
    false_positive = 0
    for i in range(0,actualDF.shape[0]):
        value = evaluation.iloc[i][-1]
        if value == actualDF.iloc[i][-1]:
            if value == 1:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if value == 1:
                false_positive += 1
            else:
                false_negative += 1
    
    return true_positive, true_negative , false_negative, false_positive


# In[20]:


def accuracy(true_positive , true_negative , false_negative, false_positive):
    return ((true_positive + true_negative)*100)/(true_positive + true_negative + false_positive + false_negative + eps)


# In[21]:


def recall(true_positive , false_negative):
    return true_positive*100/(true_positive +  false_negative+ eps)


# In[22]:


def precision(true_positive , false_positive):
    return true_positive*100/(true_positive +  false_positive + eps)


# In[23]:


def f1score(recall , prescision):
    return 2/(1/(float(recall)+eps)+1/(float(prescision)+eps))


# In[24]:


true_positive, true_negative , false_negative, false_positive = validate_test(evaluation, actualDF)


# In[25]:


accuracy = accuracy(true_positive , true_negative , false_negative, false_positive)
print("Accuracy: ",accuracy)

# In[26]:


recall = recall(true_positive , false_negative)
print("Recall: ",recall)


# In[27]:


prescision = precision(true_positive , false_positive)
print("Precision: ",prescision)


# In[28]:


f1score = f1score(recall , prescision)
print("f1score: ",f1score)

# In[ ]:





# In[ ]:




