# -*- coding: utf-8 -*-
import tensorflow as tf


def get_scope(scope):
    try:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    except:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)


def init_weight(shape):
    initializer = tf.truncated_normal_initializer(stddev=0.02)
    return tf.get_variable('weights', shape=shape, initializer=initializer)


def init_bias(shape):
    initializer = tf.constant_initializer(0.0)
    #initializer = tf.truncated_normal_initializer(stddev=0.02)
    return tf.get_variable('bias', initializer=initializer, shape=shape, dtype=tf.float32)


def leaky_relu(in_tensor, alpha=0.2):
    return tf.maximum(in_tensor, alpha * in_tensor)


def fully_connected(input_tensor, output_size, name, activation=tf.nn.relu, batch_norm=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weights = init_weight((input_tensor.shape[1].value, output_size))
        output = tf.matmul(input_tensor, weights)
        if batch_norm:
            output = batchnorm_layer(output)
        else:
            bias = init_bias((output_size, ))
            output += bias
        if activation is not None:
            output = activation(output)
    return output

def build_generator(z_prior, h_sizes):
    prev_size = z_prior.shape[1].value
    output = z_prior
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
        for i, size in enumerate(h_sizes):
            if i != len(h_sizes)-1:
                output = fully_connected(output, size, 'gen' + str(i), batch_norm=False)
            else:
                output = fully_connected(output, size,
                                                   'gen' + str(i), activation=tf.nn.tanh, batch_norm=False)
            prev_size = size
        out_shape = (-1, 1, 32, 32)
        output = tf.reshape(output, out_shape)
    return output, get_scope(scope)


def batchnorm_layer(in_tensor, center=True, scale=True, is_training=True):
    output = tf.contrib.layers.batch_norm(in_tensor, decay=0.9, center=center, scale=scale,
                                          is_training=is_training, fused=True,
                                          data_format='NCHW')
    return output


def bias_layer(in_tensor, shape):
    bias = init_bias(shape)
    output = tf.nn.bias_add(in_tensor, bias, data_format='NCHW')
    return output, bias


def deconv_layer(in_tensor, out_size, filters, kernel_size, name, strides=(1, 1, 1, 1),
                 batch_norm=True, padding='SAME', activation=leaky_relu, train=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W_conv = init_weight((kernel_size[0], kernel_size[1], filters, in_tensor.shape[1].value))
        output = tf.nn.conv2d_transpose(in_tensor, W_conv,
                                        output_shape=(in_tensor.shape[0].value,
                                                      filters,
                                                      out_size[0],
                                                      out_size[1]),
                                        strides=strides, padding=padding, data_format='NCHW')
        if batch_norm:
            output = batchnorm_layer(output, is_training=train)
        else:
            output, bias = bias_layer(output, (filters, ))
        if activation is not None:
            output = activation(output)
    return output


def conv_layer(in_tensor, filters, kernel_size, name, strides=(1, 1, 1, 1), batch_norm=True,
               padding='SAME', activation=leaky_relu, train=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W_conv = init_weight((kernel_size[0], kernel_size[1], in_tensor.shape[1].value, filters))
        output = tf.nn.conv2d(in_tensor, W_conv, strides, padding=padding, data_format='NCHW')
        if batch_norm:
            output = batchnorm_layer(output, name, is_training=train)
        else:
            output, bias = bias_layer(output, (filters, ))
        if activation is not None:
            output = activation(output)
    return output


def generator_32(in_tensor, gray_scale=False, train=True):
    last_channels = 1 if gray_scale else 3
    return build_deconv_gen(in_tensor, filters=(512, 256, 128, last_channels), strides=(2, 2, 2), train=train)

def generator_64(in_tensor, gray_scale=False, train=True):
    last_channels = 1 if gray_scale else 3
    return build_deconv_gen(in_tensor, filters=(1024, 512, 256, 128, last_channels), strides=(2, 2, 2, 2), train=train)

def build_deconv_gen(in_tensor, size=(4, 4), filters=(1024, 512, 256, 128, 1),
                     strides=(2, 2, 2, 2), kernel_size=(5, 5), train=True):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
        width, height = size
        output = fully_connected(in_tensor, width*height*filters[0], 'fc', activation=leaky_relu)
        output = tf.reshape(output, (in_tensor.shape[0].value, filters[0], size[0], size[1]))
        for idx, (filter_amt, stride) in enumerate(zip(filters[1:], strides)):
            width *= stride
            height *= stride
            kwargs = {'batch_norm': True, 'activation': leaky_relu}
            if idx+1 == len(strides):
                kwargs = {'batch_norm': False, 'activation': tf.tanh}
            output = deconv_layer(output, (width, height), filter_amt,
                                kernel_size, 'deconv' + str(idx),
                                strides=(1, 1, stride, stride), train=train,
                                **kwargs)
    return output, get_scope(scope)

def disc_32(in_tensor):
    return build_conv_discriminator(
            in_tensor, filters=(64, 128, 256), strides=(2, 2, 2), global_pooling=False)

def discriminator(images, reuse=False):
    """
    Create the discriminator network
    """
    alpha = 0.2

    with tf.variable_scope('discriminator', reuse=reuse):
        # using 4 layer network as in DCGAN Paper

        # Conv 1
        conv1 = tf.layers.conv2d(images, 128, 5, 2, 'SAME', data_format='channels_first', name='conv0')
        lrelu1 = tf.maximum(alpha * conv1, conv1)

        # Conv 2
        conv2 = tf.layers.conv2d(lrelu1, 256, 5, 2, 'SAME', data_format='channels_first', name='conv1')
        batch_norm2 = tf.layers.batch_normalization(conv2, training=True, axis=1)
        lrelu2 = tf.maximum(alpha * batch_norm2, batch_norm2)

        # Conv 3
        conv3 = tf.layers.conv2d(lrelu2, 512, 5, 2, 'SAME', data_format='channels_first', name='conv2')
        batch_norm3 = tf.layers.batch_normalization(conv3, training=True, axis=1)
        lrelu3 = tf.maximum(alpha * batch_norm3, batch_norm3)

        # Conv 3
        #conv4 = tf.layers.conv2d(lrelu3, 1024, 5, 2, 'SAME', data_format='channels_first', name='conv3')
        #batch_norm4 = tf.layers.batch_normalization(conv3, training=True, axis=1)
        #lrelu4 = tf.maximum(alpha * batch_norm4, batch_norm4)

        # Flatten
        flat = tf.reshape(lrelu3, (-1, 8*8*1024))

        # Logits
        logits = tf.layers.dense(flat, 1)

        # Output
        out = tf.sigmoid(logits)

        return out, logits

def build_conv_discriminator(in_tensor, filters=(128, 256, 512, 1024), strides=(2, 2, 2, 2),
                             kernel_size=(5, 5), global_pooling=True):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
        output = in_tensor
        for idx, (filter_amt, stride) in enumerate(zip(filters, strides)):
            output = conv_layer(output, filter_amt, kernel_size, 'conv' + str(idx),
                                strides=(1, 1, stride, stride), batch_norm=idx!=0, activation=leaky_relu)
        if global_pooling:
            output = tf.reduce_mean(output, axis=[2, 3])
        else:
            output = tf.reshape(output, [-1, 8*8*filters[-1]])
            output = fully_connected(output, output_size=1,
                                     name='fc', activation=None)
    return tf.nn.sigmoid(output), output, get_scope(scope)


def build_discriminator(in_tensor, keep_prob):
    h_sizes = [300, 150, 1]
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
        output = in_tensor
        output = tf.reshape(output, [output.shape[0], -1])
        prev_size = output.shape[1]
        for i, size in enumerate(h_sizes):
           if i != len(h_sizes)-1:
               output = fully_connected(output, size, 'disc' + str(i), batch_norm=False)
               output = tf.nn.dropout(output, keep_prob)
           else:
               output = fully_connected(output, size, 'disc' + str(i), activation=tf.nn.sigmoid, batch_norm=False)
           prev_size = size

    return output, get_scope(scope)
