from __future__ import print_function, division
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import os, random
from scipy.misc import imread
import ast
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

from utils import helpers

def prepare_data(dataset_dir):
    train_input=[]
    train_output=[]
    val_input=[]
    val_output=[]
    test_input=[]
    test_output=[]
    for file in os.listdir(dataset_dir + "/train"):
        cwd = os.getcwd()
        train_input.append(cwd + "/" + dataset_dir + "/train/" + file)
    for file in os.listdir(dataset_dir + "/train_labels"):
        cwd = os.getcwd()
        train_output.append(cwd + "/" + dataset_dir + "/train_labels/" + file)
    for file in os.listdir(dataset_dir + "/val"):
        cwd = os.getcwd()
        val_input.append(cwd + "/" + dataset_dir + "/val/" + file)
    for file in os.listdir(dataset_dir + "/val_labels"):
        cwd = os.getcwd()
        val_output.append(cwd + "/" + dataset_dir + "/val_labels/" + file)
    for file in os.listdir(dataset_dir + "/test"):
        cwd = os.getcwd()
        test_input.append(cwd + "/" + dataset_dir + "/test/" + file)
    for file in os.listdir(dataset_dir + "/test_labels"):
        cwd = os.getcwd()
        test_output.append(cwd + "/" + dataset_dir + "/test_labels/" + file)
    train_input.sort(),train_output.sort(), val_input.sort(), val_output.sort(), test_input.sort(), test_output.sort()
    return train_input,train_output, val_input, val_output, test_input, test_output

def load_image(path):
    return cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)


def filepath_to_name(name):
    file = os.path.basename(name)
    file = os.path.splitext(file_name)[0]
    return file_name


def LOG(X, f=None):
    stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)



def total_params():
    total= 0
    for var in tf.trainable_variables():
        shape = variable.get_shape()
        variable = 1
        for dim in shape:
            variable *= dim.value
        total += variable
    print("This model has %d trainable parameters"% (total))


def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def crop(image, label, crop_height, crop_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1]-crop_width)
        y = random.randint(0, image.shape[0]-crop_height)
        
        if len(label.shape) == 3:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width, :]
        else:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width]
    else:
        raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (crop_height, crop_width, image.shape[0], image.shape[1]))


def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies


def compute_mean_iou(pred, label):

    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


    mean_iou = np.mean(I / U)
    return mean_iou


def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, class_accuracies, prec, rec, f1, iou

    
