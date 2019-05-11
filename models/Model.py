import tensorflow as tf
from tensorflow.contrib import slim
from utils import resnet_v2,frontend_builder
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
import os, sys
label_size=[2048,2048]
pooling_type = "MAX"
def UP(inputs,scale):
    return tf.image.resize_nearest_neighbor(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])

def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic conv transpose block for Encoder-Decoder UP
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d_transpose(net, n_filters, kernel_size=[3, 3], stride=[scale, scale], activation_fn=None)
    return net

def ConvBlock(inputs, n_filters, kernel_size=[3, 3], strides=1):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = slim.conv2d(inputs, n_filters, kernel_size, stride=[strides, strides], activation_fn=None, normalizer_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    return net
def UP_by_shape(inputs, feature_map_shape):
    return tf.image.resize_bilinear(inputs, size=feature_map_shape)
  
def ARM(inputs, n_filters):

    # Global average pooling
    net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = slim.batch_norm(net, fused=True)
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    return net
def UP_by_scale(inputs, scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])

def FFM(input_1, input_2, n_filters):
    inputs = tf.concat([input_1, input_2], axis=-1)
    inputs = ConvBlock(inputs, n_filters=n_filters, kernel_size=[3, 3])

    # Global average pooling
    net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
    
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.sigmoid(net)
   
   
    net = tf.multiply(inputs, net)

    net = tf.add(inputs, net)

    return net

def relu(x, name='relu6'):
    return tf.nn.relu6(x, name)


def batch_norm(x, momentum=0.9, epsilon=1e-5, train=True, name='bn'):
    return tf.layers.batch_normalization(x,
                      momentum=momentum,
                      epsilon=epsilon,
                      scale=True,
                      training=train,
                      name=name)


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d', bias=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv
def conv2d_block(input, out_dim, k, s, is_train, name):
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv2d(input, out_dim, k, k, s, s, name='conv2d')
        net = batch_norm(net, train=is_train, name='bn')
        net = relu(net)
        return net


def conv_1x1(input, output_dim, name, bias=False):
    with tf.name_scope(name):
        return conv2d(input, output_dim, 1,1,1,1, stddev=0.02, name=name, bias=bias)

def pwise_block(input, output_dim, is_train, name, bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        out=conv_1x1(input, output_dim, bias=bias, name='pwb')
        out=batch_norm(out, train=is_train, name='bn')
        out=relu(out)
        return out


def dwise_conv(input, k_h=3, k_w=3, channel_multiplier= 1, strides=[1,1,1,1],
               padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
    with tf.variable_scope(name):
        in_channel=input.get_shape().as_list()[-1]
        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                        regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None,name=None,data_format=None)
        if bias:
            biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def res_block(input, expansion_ratio, output_dim, stride, is_train, name, bias=False, shortcut=True):
    with tf.name_scope(name), tf.variable_scope(name):
        # pw
        bottleneck_dim=round(expansion_ratio*input.get_shape().as_list()[-1])
        net = conv_1x1(input, bottleneck_dim, name='pw', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_bn')
        net = relu(net)
        # dw
        net = dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
        net = batch_norm(net, train=is_train, name='dw_bn')
        net = relu(net)
        # pw & linear
        net = conv_1x1(net, output_dim, name='pw_linear', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_linear_bn')

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            in_dim=int(input.get_shape().as_list()[-1])
            if in_dim != output_dim:
                ins=conv_1x1(input, output_dim, name='ex_dim')
                net=ins+net
            else:
                net=input+net

        return net


def separable_conv(input, k_size, output_dim, stride, pad='SAME', channel_multiplier=1, name='sep_conv', bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        in_channel = input.get_shape().as_list()[-1]
        dwise_filter = tf.get_variable('dw', [k_size, k_size, in_channel, channel_multiplier],
                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                  initializer=tf.truncated_normal_initializer(stddev=0.02))

        pwise_filter = tf.get_variable('pw', [1, 1, in_channel*channel_multiplier, output_dim],
                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        strides = [1,stride, stride,1]

        conv=tf.nn.separable_conv2d(input,dwise_filter,pwise_filter,strides,padding=pad, name=name)
        if bias:
            biases = tf.get_variable('bias', [output_dim],initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv


def global_avg(x):
    with tf.name_scope('global_avg'):
        net=tf.layers.average_pooling2d(x, 2, 2)
        return net


def flatten(x):
    #flattened=tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
    return tf.contrib.layers.flatten(x)


def pad2d(inputs, pad=(0, 0), mode='CONSTANT'):
    paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
    net = tf.pad(inputs, paddings, mode=mode)
    return net

def mobilenetv2(inputs, num_classes, is_train=True, reuse=False):
    exp = 4  # expansion ratio
    with tf.variable_scope('mobilenetv2'):
        net = conv2d_block(inputs, 32, 3, 1, is_train, name='conv1_1')  # size/2
        
        net = res_block(net, 1, 16, 1, is_train, name='res2_1')
        
        net = res_block(net, exp, 64, 2, is_train, name='res3_1')  # size/4
       
        
        net = res_block(net, exp, 32, 1, is_train, name='res4_1')  # size/8
        net = res_block(net, exp, 64, 1, is_train, name='res4_2',shortcut=True)
        net = res_block(net, exp, 64, 2, is_train, name='res4_3',shortcut=True)

        
        net = res_block(net, exp, 128, 1, is_train, name='res5_1',shortcut=True)
        net = res_block(net, exp, 128, 2, is_train, name='res5_1_1',shortcut=True)
        
        net = pwise_block(net, 256, is_train, name='conv9_1')
        
        #inputs = tf.concat([k ,m], axis=-1)
        #logits = flatten(conv_1x1(net, num_classes, name='logits'))
        
        return k

def prelu(x, scope='anil', decoder=False):
    
    alpha= tf.get_variable(scope + 'alpha', x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alpha * (x - abs(x)) * 0.5
    return pos + neg
def initial_block(inputs, is_training=True, scope='initial_block'):
  
    net_conv = slim.conv2d(inputs, 13, [3,3], stride=2, activation_fn=None, scope=scope+'_conv')
    net_conv = slim.batch_norm(net_conv, is_training=is_training, fused=True, scope=scope+'_batchnorm')
    net_conv = prelu(net_conv, scope=scope+'_prelu')

    #Max pool branch
    net_pool = slim.max_pool2d(inputs, [2,2], stride=2, scope=scope+'_max_pool')

    #Concatenated output - does it matter max pool comes first or conv comes first? probably not.
    net_concatenated = tf.concat([net_conv, net_pool], axis=3, name=scope+'_concat')
    return net_concatenated
scope='anil'
def InterpBlock(net, level, feature_map_shape, pooling_type):
    
    # Compute the kernel and stride sizes according to how large the final feature map will be
    # When the kernel size and strides are equal, then we can compute the final feature map size
    # by simply dividing the current size by the kernel or stride size
    # The final feature map sizes are 1x1, 2x2, 3x3, and 6x6. We round to the closest integer
    kernel_size = [int(np.round(float(feature_map_shape[0]) / float(level))), int(np.round(float(feature_map_shape[1]) / float(level)))]
    stride_size = kernel_size
    print('4')
    print(tf.shape(net))
    #net = slim.pool(net, kernel_size, stride=stride_size, pooling_type='MAX')
    net = slim.conv2d(net, 512, [1, 1], activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    net = UP_by_shape(net, feature_map_shape)
    print('5')
    print(tf.shape(net))
    
    return net

def spatial_dropout(x, p, seed, scope, is_training=True):

    if is_training:
        keep_prob = 1.0 - p
        input_shape = x.get_shape().as_list()
        noise_shape = tf.constant(value=[1, 1, 1, input_shape[3]])
        output = tf.nn.dropout(x, keep_prob, noise_shape, seed=seed, name=scope)

        return output

    return x
output_depth=128
def PyramidPoolingModule(inputs, feature_map_shape, pooling_type):
    """
    Build the Pyramid Pooling Module.
    """

    interp_block1 = InterpBlock(inputs, 1, feature_map_shape, pooling_type)
    interp_block2 = InterpBlock(inputs, 2, feature_map_shape, pooling_type)
    interp_block3 = InterpBlock(inputs, 3, feature_map_shape, pooling_type)
    interp_block6 = InterpBlock(inputs, 6, feature_map_shape, pooling_type)
    
    res = tf.concat([interp_block6, interp_block3, interp_block2,interp_block1],axis=-1)
    
    return res

def CFFBlock(F1, F2, num_classes):
    F1_big = UP_by_scale(F1, scale=2)
    F1_out = slim.conv2d(F1, num_classes, [1, 1], activation_fn=None)

    F1_big = slim.conv2d(F1_big, 512, [3, 3], rate=2, activation_fn=None)
    F1_big = slim.batch_norm(F1_big, fused=True)

    F2_proj = slim.conv2d(F2, 512, [1, 1], rate=1, activation_fn=None)
    F2_proj = slim.batch_norm(F2_proj, fused=True)

    F2_out = tf.add((F1_big, F2_proj),y=0)
    F2_out = tf.nn.relu(F2_out)

    return F1_out, F2_out
def build_model(inputs, num_classes, preset_model='BiSeNet', frontend="ResNet152", weight_decay=1e-5, is_training=True, pretrained_dir="models"):
    
    ### Context path
    logits, end_points, frontend_scope, init_fn  = frontend_builder.build_frontend(inputs, frontend, pretrained_dir=pretrained_dir, is_training=is_training)
    spatial_net = ConvBlock(inputs, n_filters=64, kernel_size=[3, 3], strides=2)
    spatial_net = ConvBlock(spatial_net, n_filters=128, kernel_size=[3, 3], strides=2)
    spatial_net = ConvBlock(spatial_net, n_filters=256, kernel_size=[3, 3], strides=2)
    #spatial_net = ConvBlock(inputs, n_filters=64, kernel_size=[3, 3], scale=2)
    #spatial_net = ConvBlock(spatial_net, n_filters=128, kernel_size=[3, 3], scale=2)
    #spatial_net = ConvUpscaleBlock(spatial_net, n_filters=256, kernel_size=[3, 3], scale=2)
    #spatial_net = global_avg(spatial_net)
    #spatial_net = global_avg(spatial_net)
    #spatial_net=mobilenetv2(inputs, num_classes, is_train=is_training)
    #net=end_points['pool2']
    #net = global_avg(net)
    #j = tf.concat([spatial_net ,net], axis=-1)
    #net_4 = ARM(end_points['pool4'], n_filters=512)
    #net = conv_transpose_block(inputs, 512)
    #net = initial_block(inputs, scope='initial_block_1')
    
    #j = tf.concat([spatial_net ,net], axis=-1)

    #net_main= spatial_net
    #net = slim.conv2d(net, 4, [3,3], rate=2, scope=scope+'_dilated_conv2')
    #net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm2')
    #net = prelu(net, scope=scope+'_prelu2')
    inputs_4 = tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*0.25,  tf.shape(inputs)[2]*0.25])   
    inputs_2 = tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*0.5,  tf.shape(inputs)[1]*0.5])
    #with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
     # logits_32, end_points_32 = resnet_v2.resnet_v2_101(inputs_4, is_training=is_training, scope='resnet_v2_102')
     # logits_16, end_points_16 = resnet_v2.resnet_v2_101(inputs_2, is_training=is_training, scope='resnet_v2_103')
     #logits_8, end_points_8 = resnet_v2.resnet_v2_101(inputs, is_training=is_training, scope='resnet_v2_104')
    #feature_map_shape = [int(x / 32.0) for x in label_size]
    #print(tf.shape(feature_map_shape))
    #feature_map_shape_1 = [int(x / 32.0) for x in label_size]
    #print(tf.shape(feature_map_shape_1))
    #block_32 = PyramidPoolingModule(end_points_32['pool3'], feature_map_shape=feature_map_shape, pooling_type=pooling_type)
    #block_16 = PyramidPoolingModule(end_points_16['pool3'], feature_map_shape=feature_map_shape, pooling_type=pooling_type)
    #out_16, block_16_1 = CFFBlock(block_32, end_points_16['pool3'],32)
    #out_8, block_8 = CFFBlock(block_16, end_points_8['pool3'],32) 
    spatial_net_1 = ConvBlock(inputs_2, n_filters=64, kernel_size=[3, 3], strides=2)
    spatial_net_1 = ConvBlock(spatial_net_1, n_filters=128, kernel_size=[3, 3], strides=2)
    spatial_net_1 = ConvBlock(spatial_net_1, n_filters=256, kernel_size=[3, 3], strides=2)
    spatial_net_1 = initial_block(spatial_net_1, scope='initial_block_1')
    spatial_net_2= ConvBlock(inputs, n_filters=64, kernel_size=[3, 3], strides=2)
    spatial_net_2 = ConvBlock(spatial_net_2, n_filters=128, kernel_size=[3, 3], strides=2)
    spatial_net_2= ConvBlock(spatial_net_2, n_filters=256, kernel_size=[3, 3], strides=2)
    spatial_net_2 = initial_block(spatial_net_2, scope='initial_block_2')
    #spatial_net_2 = initial_block(spatial_net_2, scope='initial_block_3')
    
    net = slim.conv2d(spatial_net_2, output_depth, [1,1], scope=scope+'_conv3')
    net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm3')
    net = prelu(net, scope=scope+'_prelu3')

    net = spatial_dropout(net, p=0.01, seed=0, scope=scope+'_spatial_dropout')
    #net = prelu(net, scope=scope+'_prelu4')
    
    #net = global_avg(net)        
    #net = tf.add(spatial_net_2, net, name=scope+'_add_dilated')
    spatial_net_2 = prelu(net, scope=scope+'_last_prelu')
    spatial_net_2 = global_avg(spatial_net_2)       
    #net = global_avg(net)
    #net = bottleneck(net, output_depth=128, filter_size=3, dilated=True, dilation_rate=2, scope='bottleneck'+str(i)+'_2')
    #net = bottleneck(net, output_depth=128, filter_size=5, asymmetric=True, scope='bottleneck'+str(i)+'_3')
    #k=ENet(inputs,num_classes,batch_size=1,num_initial_blocks=1,stage_two_repeat=2,skip_connections=True,reuse=None,is_training=True,scope='ENet')
    net= tf.concat([spatial_net,spatial_net_1,spatial_net_2],axis=-1)
      
    net_5 = ARM(end_points['pool5'], n_filters=2048)

    global_channels = tf.reduce_mean(net_5, [1, 2], keep_dims=True)
    net_5_scaled = tf.multiply(global_channels, net_5)

    ### Combining the paths
    #net_4 = UP(net_4, scale=2)
    net_5_scaled = UP(net_5_scaled, scale=4)
    #net_5_scaled=global_avg(net_5_scaled)
    #context_net = tf.concat([net_4, net_5_scaled], axis=-1)
    
    net = FFM(input_1=net, input_2=net_5_scaled, n_filters=num_classes)


    ### Final upscaling and finish
    net = UP(net, scale=8)
    
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')

    return net, init_fn

