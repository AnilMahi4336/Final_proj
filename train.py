from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess
import matplotlib
matplotlib.use('Agg')
from utils import utils, tool,model_builder
import matplotlib.pyplot as plt

def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

var = argparse.ArgumentParser()
var.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
var.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
var.add_argument('--checkpoint_step', type=int, default=5, help='How often to save checkpoints (epochs)')
var.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
var.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
var.add_argument('--continue_training', type=str_to_bool, default=False, help='Whether to continue training from a checkpoint')
var.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
var.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
var.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
var.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
var.add_argument('--num_val_images', type=int, default=20, help='The number of images to used for validations')
var.add_argument('--model', type=str, default="Model", help='The model you are using. See model_builder.py for supported models')
var.add_argument('--frontend', type=str, default="ResNet101", help='The frontend you are using. See frontend_builder.py for supported models')
args = var.parse_args()


def data_augmentation(input_image, output_image):
    
    return utils.crop(input_image, output_image, args.crop_height, args.crop_width)

class_names_list, label_values = tool.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)



net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network, init_fn = model_builder.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=True)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))
I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
U[index] = float(np.sum(np.logical_or(label_i, pred_i)))
mean_iou = np.mean(I / U)
loss=loss+mean_iou
opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.total_params()

if init_fn is not None:
    init_fn(sess)

model_checkpoint_name = "checkpoints/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
if args.continue_training:
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)

print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)


print("\n***** Begin training *****")
print("Dataset ::", args.dataset)
print("Model ::", args.model)
print("Crop Height ::", args.crop_height)
print("Crop Width ::", args.crop_width)
print("Num Epochs ::", args.num_epochs)
print("Batch Size ::", args.batch_size)
print("Num Classes ::", num_classes)


avg_loss_per_epoch = []
avg_scores_per_epoch = []
avg_iou_per_epoch = []


val_indices = []
num_vals = min(args.num_val_images, len(val_input_names))


random.seed(17)
val_indices=random.sample(range(0,len(val_input_names)),num_vals)

for epoch in range(args.epoch_start_i, args.num_epochs):

    current_losses = []

    cnt=0

    
    id_list = np.random.permutation(len(train_input_names))

    num_iters = int(np.floor(len(id_list) / args.batch_size))
    st = time.time()
    epoch_st=time.time()
    for i in range(num_iters):

        input_image_batch = []
        output_image_batch = []

        for j in range(args.batch_size):
            index = i*args.batch_size + j
            id = id_list[index]
            input_image = utils.load_image(train_input_names[id])
            output_image = utils.load_image(train_output_names[id])

            with tf.device('/cpu:0'):
                input_image, output_image = data_augmentation(input_image, output_image)

                input_image = np.float32(input_image) / 255.0
                output_image = np.float32(tool.one_hot_it(label=output_image, label_values=label_values))

                input_image_batch.append(np.expand_dims(input_image, axis=0))
                output_image_batch.append(np.expand_dims(output_image, axis=0))

        if args.batch_size == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))
        
        _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
        current_losses.append(current)
        cnt = cnt + args.batch_size
        if cnt % 20 == 0:
            string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
            utils.LOG(string_print)
            st = time.time()

    mean_loss = np.mean(current_losses)
    avg_loss_per_epoch.append(mean_loss)

    if not os.path.isdir("%s/%04d"%("checkpoints",epoch)):
        os.makedirs("%s/%04d"%("checkpoints",epoch))

    print("Saving latest checkpoint")
    saver.save(sess,model_checkpoint_name)

    if val_indices != 0 and epoch % args.checkpoint_step == 0:
        print("Saving checkpoint for this epoch")
        saver.save(sess,"%s/%04d/model.ckpt"%("checkpoints",epoch))


    if epoch % args.validation_step == 0:
        print("Performing validation")
        target=open("%s/%04d/val_scores.csv"%("checkpoints",epoch),'w')
        target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))


        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []

        for ind in val_indices:

            input_image = np.expand_dims(np.float32(utils.load_image(val_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
            gt = utils.load_image(val_output_names[ind])[:args.crop_height, :args.crop_width]
            gt = tool.reverse_one_hot(tool.one_hot_it(gt, label_values))


            output_image = sess.run(network,feed_dict={net_input:input_image})


            output_image = np.array(output_image[0,:,:,:])
            output_image = tool.reverse_one_hot(output_image)
            out_vis_image = tool.colour_code_segmentation(output_image, label_values)

            accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)

            file_name = utils.filepath_to_name(val_input_names[ind])
            target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
            for item in class_accuracies:
                target.write(", %f"%(item))
            target.write("\n")

            scores_list.append(accuracy)
            class_scores_list.append(class_accuracies)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            iou_list.append(iou)

            gt = tool.colour_code_segmentation(gt, label_values)

            file_name = os.path.basename(val_input_names[ind])
            file_name = os.path.splitext(file_name)[0]
            cv2.imwrite("%s/%04d/%s_pred.png"%("checkpoints",epoch, file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
            cv2.imwrite("%s/%04d/%s_gt.png"%("checkpoints",epoch, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))


        target.close()

        avg_score = np.mean(scores_list)
        class_avg_scores = np.mean(class_scores_list, axis=0)
        avg_scores_per_epoch.append(avg_score)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)
        avg_iou_per_epoch.append(avg_iou)

        print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
        print("Average per class validation accuracies for epoch # %04d:"% (epoch))
        for index, item in enumerate(class_avg_scores):
            print("%s = %f" % (class_names_list[index], item))
        print("Validation precision = ", avg_precision)
        print("Validation recall = ", avg_recall)
        print("Validation F1 score = ", avg_f1)
        print("Validation IoU score = ", avg_iou)

    epoch_time=time.time()-epoch_st
    remain_time=epoch_time*(args.num_epochs-1-epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s!=0:
        train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Remaining training time : Training completed.\n"
    utils.LOG(train_time)
    scores_list = []


    fig1, ax1 = plt.subplots(figsize=(11, 8))

    ax1.plot(range(epoch+1), avg_scores_per_epoch)
    ax1.set_title("Average validation accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. accuracy")


    plt.savefig('accuracy_vs_epochs.png')

    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(11, 8))

    ax2.plot(range(epoch+1), avg_loss_per_epoch)
    ax2.set_title("Average loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")

    plt.savefig('loss_vs_epochs.png')

    plt.clf()

    fig3, ax3 = plt.subplots(figsize=(11, 8))

    ax3.plot(range(epoch+1), avg_iou_per_epoch)
    ax3.set_title("Average IoU vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current IoU")

    plt.savefig('iou_vs_epochs.png')



