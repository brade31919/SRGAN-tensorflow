from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.ops import *
import collections
import os
import math
import scipy.misc as sic
import numpy as np


# Define the dataloader
def data_loader(FLAGS):
    with tf.device('/cpu:0'):
        # Define the returned data batches
        Data = collections.namedtuple('Data', 'paths_LR, paths_HR, inputs, targets, image_count, steps_per_epoch')

        #Check the input directory
        if (FLAGS.input_dir_LR == 'None') or (FLAGS.input_dir_HR == 'None'):
            raise ValueError('Input directory is not provided')

        if (not os.path.exists(FLAGS.input_dir_LR)) or (not os.path.exists(FLAGS.input_dir_HR)):
            raise ValueError('Input directory not found')

        image_list_LR = os.listdir(FLAGS.input_dir_LR)
        image_list_LR = [_ for _ in image_list_LR if _.endswith('.jpg')]
        if len(image_list_LR)==0:
            raise Exception('No png files in the input directory')

        image_list_LR_temp = sorted(image_list_LR)
        image_list_LR = [os.path.join(FLAGS.input_dir_LR, _) for _ in image_list_LR_temp]
        image_list_HR = [os.path.join(FLAGS.input_dir_HR, _) for _ in image_list_LR_temp]

        image_list_LR_tensor = tf.convert_to_tensor(image_list_LR, dtype=tf.string)
        image_list_HR_tensor = tf.convert_to_tensor(image_list_HR, dtype=tf.string)

        with tf.variable_scope('load_image'):
            # define the image list queue
            # image_list_LR_queue = tf.train.string_input_producer(image_list_LR, shuffle=False, capacity=FLAGS.name_queue_capacity)
            # image_list_HR_queue = tf.train.string_input_producer(image_list_HR, shuffle=False, capacity=FLAGS.name_queue_capacity)
            #print('[Queue] image list queue use shuffle: %s'%(FLAGS.mode == 'Train'))
            output = tf.train.slice_input_producer([image_list_LR_tensor, image_list_HR_tensor],
                                                   shuffle=False, capacity=FLAGS.name_queue_capacity)

            # Reading and decode the images
            reader = tf.WholeFileReader(name='image_reader')
            image_LR = tf.read_file(output[0])
            image_HR = tf.read_file(output[1])
            input_image_LR = tf.image.decode_jpeg(image_LR, channels=3)
            input_image_HR = tf.image.decode_jpeg(image_HR, channels=3)
            input_image_LR = tf.image.convert_image_dtype(input_image_LR, dtype=tf.float32)
            input_image_HR = tf.image.convert_image_dtype(input_image_HR, dtype=tf.float32)

            assertion = tf.assert_equal(tf.shape(input_image_LR)[2], 3, message="image does not have 3 channels")
            with tf.control_dependencies([assertion]):
                input_image_LR = tf.identity(input_image_LR)
                input_image_HR = tf.identity(input_image_HR)

            # Normalize the low resolution image to [0, 1], high resolution to [-1, 1]
            a_image = preprocessLR(input_image_LR)
            b_image = preprocess(input_image_HR)

            inputs, targets = [a_image, b_image]

        # The data augmentation part
        with tf.name_scope('data_preprocessing'):
            with tf.name_scope('random_crop'):
                # Check whether perform crop
                if (FLAGS.random_crop is True) and FLAGS.mode == 'train':
                    print('[Config] Use random crop')
                    # Set the shape of the input image. the target will have 4X size
                    input_size = tf.shape(inputs)
                    target_size = tf.shape(targets)
                    offset_w =  tf.cast(tf.floor(tf.random_uniform([], 0, tf.cast(input_size[1], tf.float32) - FLAGS.crop_size)),
                                       dtype=tf.int32)
                    offset_h = tf.cast(tf.floor(tf.random_uniform([], 0 , tf.cast(input_size[0], tf.float32) - FLAGS.crop_size)),
                                       dtype=tf.int32)

                    if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
                        inputs = tf.image.crop_to_bounding_box(inputs, offset_h, offset_w, FLAGS.crop_size,
                                                               FLAGS.crop_size)
                        targets = tf.image.crop_to_bounding_box(targets, offset_h*4, offset_w*4, FLAGS.crop_size*4,
                                                                FLAGS.crop_size*4)
                    elif FLAGS.task == 'denoise':
                        inputs = tf.image.crop_to_bounding_box(inputs, offset_h, offset_w, FLAGS.crop_size,
                                                               FLAGS.crop_size)
                        targets = tf.image.crop_to_bounding_box(targets, offset_h, offset_w,
                                                                FLAGS.crop_size, FLAGS.crop_size)
                # Do not perform crop
                else:
                    inputs = tf.identity(inputs)
                    targets = tf.identity(targets)

            with tf.variable_scope('random_flip'):
                # Check for random flip:
                if (FLAGS.flip is True) and (FLAGS.mode == 'train'):
                    print('[Config] Use random flip')
                    # Produce the decision of random flip
                    decision = tf.random_uniform([], 0, 1, dtype=tf.float32)

                    input_images = random_flip(inputs, decision)
                    target_images = random_flip(targets, decision)
                else:
                    input_images = tf.identity(inputs)
                    target_images = tf.identity(targets)

            if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
                input_images.set_shape([FLAGS.crop_size, FLAGS.crop_size, 3])
                target_images.set_shape([FLAGS.crop_size*4, FLAGS.crop_size*4, 3])
            elif FLAGS.task == 'denoise':
                input_images.set_shape([FLAGS.crop_size, FLAGS.crop_size, 3])
                target_images.set_shape([FLAGS.crop_size, FLAGS.crop_size, 3])

        if FLAGS.mode == 'train':
            paths_LR_batch, paths_HR_batch, inputs_batch, targets_batch = tf.train.shuffle_batch([output[0], output[1], input_images, target_images],
                                            batch_size=FLAGS.batch_size, capacity=FLAGS.image_queue_capacity+4*FLAGS.batch_size,
                                            min_after_dequeue=FLAGS.image_queue_capacity, num_threads=FLAGS.queue_thread)
        else:
            paths_LR_batch, paths_HR_batch, inputs_batch, targets_batch = tf.train.batch([output[0], output[1], input_images, target_images],
                                            batch_size=FLAGS.batch_size, num_threads=FLAGS.queue_thread, allow_smaller_final_batch=True)

        steps_per_epoch = int(math.ceil(len(image_list_LR) / FLAGS.batch_size))
        if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
            inputs_batch.set_shape([FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 3])
            targets_batch.set_shape([FLAGS.batch_size, FLAGS.crop_size*4, FLAGS.crop_size*4, 3])
        elif FLAGS.task == 'denoise':
            inputs_batch.set_shape([FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 3])
            targets_batch.set_shape([FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 3])
    return Data(
        paths_LR=paths_LR_batch,
        paths_HR=paths_HR_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        image_count=len(image_list_LR),
        steps_per_epoch=steps_per_epoch
    )


# The test data loader. Allow input image with different size
def test_data_loader(FLAGS):
    # Get the image name list
    if (FLAGS.input_dir_LR == 'None') or (FLAGS.input_dir_HR == 'None'):
        raise ValueError('Input directory is not provided')

    if (not os.path.exists(FLAGS.input_dir_LR)) or (not os.path.exists(FLAGS.input_dir_HR)):
        raise ValueError('Input directory not found')

    image_list_LR_temp = os.listdir(FLAGS.input_dir_LR)
    image_list_LR = [os.path.join(FLAGS.input_dir_LR, _) for _ in image_list_LR_temp if _.split('.')[-1] == 'jpg']
    image_list_HR = [os.path.join(FLAGS.input_dir_HR, _) for _ in image_list_LR_temp if _.split('.')[-1] == 'jpg']

    # Read in and preprocess the images
    def preprocess_test(name, mode):
        im = sic.imread(name, mode="RGB").astype(np.float32)
        # check grayscale image
        if im.shape[-1] != 3:
            h, w = im.shape
            temp = np.empty((h, w, 3), dtype=np.uint8)
            temp[:, :, :] = im[:, :, np.newaxis]
            im = temp.copy()
        if mode == 'LR':
            im = im / np.max(im)
        elif mode == 'HR':
            im = im / np.max(im)
            im = im * 2 - 1

        return im

    image_LR = [preprocess_test(_, 'LR') for _ in image_list_LR]
    image_HR = [preprocess_test(_, 'HR') for _ in image_list_HR]

    # Push path and image into a list
    Data = collections.namedtuple('Data', 'paths_LR, paths_HR, inputs, targets')

    return Data(
        paths_LR = image_list_LR,
        paths_HR = image_list_HR,
        inputs = image_LR,
        targets = image_HR
    )


# The inference data loader. Allow input image with different size
def inference_data_loader(FLAGS):
    # Get the image name list
    if (FLAGS.input_dir_LR == 'None'):
        raise ValueError('Input directory is not provided')

    if not os.path.exists(FLAGS.input_dir_LR):
        raise ValueError('Input directory not found')

    image_list_LR_temp = os.listdir(FLAGS.input_dir_LR)
    image_list_LR = [os.path.join(FLAGS.input_dir_LR, _) for _ in image_list_LR_temp if _.split('.')[-1] == 'jpg']

    # Read in and preprocess the images
    def preprocess_test(name):
        im = sic.imread(name, mode="RGB").astype(np.float32)
        # check grayscale image
        if im.shape[-1] != 3:
            h, w = im.shape
            temp = np.empty((h, w, 3), dtype=np.uint8)
            temp[:, :, :] = im[:, :, np.newaxis]
            im = temp.copy()
        im = im / np.max(im)

        return im

    image_LR = [preprocess_test(_) for _ in image_list_LR]

    # Push path and image into a list
    Data = collections.namedtuple('Data', 'paths_LR, inputs')

    return Data(
        paths_LR=image_list_LR,
        inputs=image_LR
    )


# Definition of the generator
def generator(gen_inputs, gen_output_channels, reuse=False, FLAGS=None):
    # Check the flag
    if FLAGS is None:
        raise  ValueError('No FLAGS is provided for generator')

    # The Bx residual blocks
    def residual_block(inputs, output_channel, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, 3, output_channel, stride, use_bias=False, scope='conv_1',norm=FLAGS.w_norm)
            net = batchnorm(net, FLAGS.is_training)
            net = prelu_tf(net)
            net = conv2(net, 3, output_channel, stride, use_bias=False, scope='conv_2',norm=FLAGS.w_norm)
            net = batchnorm(net, FLAGS.is_training)
            net = net + inputs

        return net


    with tf.variable_scope('generator_unit', reuse=reuse):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = conv2(gen_inputs, 9, 64, 1, scope='conv',norm=FLAGS.w_norm)
            net = prelu_tf(net)

        stage1_output = net

        # The residual block parts
        for i in range(1, FLAGS.num_resblock+1 , 1):
            name_scope = 'resblock_%d'%(i)
            net = residual_block(net, 64, 1, name_scope)

        with tf.variable_scope('resblock_output'):
            net = conv2(net, 3, 64, 1, use_bias=False, scope='conv',norm=FLAGS.w_norm)
            net = batchnorm(net, FLAGS.is_training)
        
        if FLAGS.attention:
            net = attention(net,64,reuse=reuse,FLAGS)
            
        net = net + stage1_output

        with tf.variable_scope('subpixelconv_stage1'):
            net = conv2(net, 3, 256, 1, scope='conv',norm=FLAGS.w_norm)
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('subpixelconv_stage2'):
            net = conv2(net, 3, 256, 1, scope='conv',norm=FLAGS.w_norm)
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('output_stage'):
            net = conv2(net, 9, gen_output_channels, 1, scope='conv',norm=FLAGS.w_norm)

    return net


# Definition of the discriminator
def discriminator(dis_inputs, FLAGS=None, reuse=False):
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator')

    # Define the discriminator block
    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, kernel_size, output_channel, stride, use_bias=False, scope='conv1',norm=FLAGS.w_norm)
            net = batchnorm(net, FLAGS.is_training)
            net = lrelu(net, 0.2)

        return net

    with tf.device('/gpu:0'):
        with tf.variable_scope('discriminator_unit',reuse=reuse):
            # The input layer
            with tf.variable_scope('input_stage'):
                net = conv2(dis_inputs, 3, 64, 1, scope='conv',norm=FLAGS.w_norm)
                net = lrelu(net, 0.2)

            # The discriminator block part
            # block 1
            net = discriminator_block(net, 64, 3, 2, 'disblock_1')

            # block 2
            net = discriminator_block(net, 128, 3, 1, 'disblock_2')

            # block 3
            net = discriminator_block(net, 128, 3, 2, 'disblock_3')

            # block 4
            net = discriminator_block(net, 256, 3, 1, 'disblock_4')
            
            if FLAGS.attention:
                net = attention(net,256,reuse=reuse,FLAGS)
            
            # block 5
            net = discriminator_block(net, 256, 3, 2, 'disblock_5')

            # block 6
            net = discriminator_block(net, 512, 3, 1, 'disblock_6')

            # block_7
            net = discriminator_block(net, 512, 3, 2, 'disblock_7')

            # The dense layer 1
            with tf.variable_scope('dense_layer_1'):
                net = slim.flatten(net)
                net = denselayer(net, 1024,norm=FLAGS.w_norm)
                net = lrelu(net, 0.2)

            # The dense layer 2
            with tf.variable_scope('dense_layer_2'):
                net = denselayer(net, 1,norm=FLAGS.w_norm)
                net = tf.nn.sigmoid(net)

    return net

def attention(x, ch, scope='attention', reuse=False,FLAGS=None):
        with tf.variable_scope(scope, reuse=reuse):
            f = conv2(x, ch // 8, kernel=1, stride=1, scope='f_conv',norm=FLAGS.w_norm) # [bs, h, w, c']
            g = conv2(x, ch // 8, kernel=1, stride=1, scope='g_conv',norm=FLAGS.w_norm) # [bs, h, w, c']
            h = conv2(x, ch, kernel=1, stride=1, scope='h_conv',norm=FLAGS.w_norm) # [bs, h, w, c]

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

            beta = tf.nn.softmax(s, axis=-1)  # attention map

            o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
            x = gamma * o + x

        return x

def VGG19_slim(input, type, reuse, scope):
    # Define the feature to extract according to the type of perceptual
    if type == 'VGG54':
        target_layer =  'vgg_19/conv5/conv5_4'
    elif type == 'VGG22':
        target_layer = 'vgg_19/conv2/conv2_2'
    else:
        raise NotImplementedError('Unknown perceptual type')
    _, output = vgg_19(input, is_training=False, reuse=reuse)
    output = output[target_layer]

    return output


# Define the whole network architecture
def SRGAN(inputs, targets, FLAGS,  devices = ['/gpu:%d'%i for i in range(8)]):
    # Define the container of the parameter
    Network = collections.namedtuple('Network', 'discrim_real_output, discrim_fake_output, discrim_loss, \
        discrim_grads_and_vars, adversarial_loss, content_loss, gen_grads_and_vars, gen_output, train, global_step, \
        learning_rate')

    #generator tower lists
    tower_grads = []
    tower_outputs = []
    #discriminator tower lists
    tower_grads_d = []
    tower_outputs_real_d = []
    tower_outputs_fake_d = []
    tower_discriminator_global = []

    with tf.device('/gpu:0'):
        split_inputs = tf.split(inputs, len(devices), axis=0)
        split_targets = tf.split(targets, len(devices), axis=0)
        # Define the learning rate and global step
        with tf.variable_scope('get_learning_rate_and_global_step'):
            global_step = tf.contrib.framework.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate,
                                                       staircase=FLAGS.stair)
            incr_global_step = tf.assign(global_step, global_step + 1)


        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i, (inputs, targets, dev) in enumerate(zip(split_inputs, split_targets, devices)):        
                    with tf.device(dev):
                        with tf.name_scope('tower%d'%i):
                            # Build the generator part
                            with tf.variable_scope('generator'):
                                output_channel = targets.get_shape().as_list()[-1]
                                gen_output = generator(inputs, output_channel, reuse=False, FLAGS=FLAGS)
                                gen_output.set_shape([FLAGS.batch_size/len(devices) , FLAGS.crop_size*4, FLAGS.crop_size*4, 3])
                                
                                tower_outputs.append(gen_output)

                            # Build the fake discriminator
                            with tf.name_scope('fake_discriminator'):
                                with tf.variable_scope('discriminator', reuse=False):
                                    discrim_fake_output = discriminator(gen_output, FLAGS=FLAGS)
                                    tower_outputs_fake_d.append(discrim_fake_output)

                            # Build the real discriminator
                            with tf.name_scope('real_discriminator'):
                                with tf.variable_scope('discriminator', reuse=True):
                                    discrim_real_output = discriminator(targets, FLAGS=FLAGS)
                                    tower_outputs_real_d.append(discrim_real_output)

                            # Use the VGG54 feature
                            if FLAGS.perceptual_mode == 'VGG54':
                                with tf.name_scope('vgg19_1') as scope:
                                    extracted_feature_gen = VGG19_slim(gen_output, FLAGS.perceptual_mode, reuse=False, scope=scope)
                                with tf.name_scope('vgg19_2') as scope:
                                    extracted_feature_target = VGG19_slim(targets, FLAGS.perceptual_mode, reuse=True, scope=scope)

                            # Use the VGG22 feature
                            elif FLAGS.perceptual_mode == 'VGG22':
                                with tf.name_scope('vgg19_1') as scope:
                                    extracted_feature_gen = VGG19_slim(gen_output, FLAGS.perceptual_mode, reuse=False, scope=scope)
                                with tf.name_scope('vgg19_2') as scope:
                                    extracted_feature_target = VGG19_slim(targets, FLAGS.perceptual_mode, reuse=True, scope=scope)

                            # Use MSE loss directly
                            elif FLAGS.perceptual_mode == 'MSE':
                                extracted_feature_gen = gen_output
                                extracted_feature_target = targets

                            else:
                                raise NotImplementedError('Unknown perceptual type!!')

                            # Calculating the generator loss
                            with tf.variable_scope('generator_loss'):
                                # Content loss
                                with tf.variable_scope('content_loss'):
                                    # Compute the euclidean distance between the two features
                                    diff = extracted_feature_gen - extracted_feature_target
                                    if FLAGS.perceptual_mode == 'MSE':
                                        content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))
                                    else:
                                        content_loss = FLAGS.vgg_scaling*tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))

                                with tf.variable_scope('adversarial_loss'):
                                    adversarial_loss = tf.reduce_mean(-tf.log(discrim_fake_output + FLAGS.EPS))

                                gen_loss = content_loss + (FLAGS.ratio)*adversarial_loss
                                print(adversarial_loss.get_shape())
                                print(content_loss.get_shape())

                            # Calculating the discriminator loss
                            with tf.variable_scope('discriminator_loss'):
                                discrim_fake_loss = tf.log(1 - discrim_fake_output + FLAGS.EPS)
                                discrim_real_loss = tf.log(discrim_real_output + FLAGS.EPS)

                                discrim_loss = tf.reduce_mean(-(discrim_fake_loss + discrim_real_loss))

                            # Define the learning rate and global step
                            with tf.variable_scope('get_learning_rate_and_global_step'):
                                global_step = tf.contrib.framework.get_or_create_global_step()
                                learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate, staircase=FLAGS.stair)
                                incr_global_step = tf.assign(global_step, global_step + 1)

                            #scope.reuse_variables()
                            with tf.variable_scope('dicriminator_train',reuse=tf.AUTO_REUSE):
                                discrim_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                                discrim_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
                                discrim_grads_and_vars = discrim_optimizer.compute_gradients(discrim_loss, discrim_tvars)
                                #discrim_train = discrim_optimizer.apply_gradients(discrim_grads_and_vars)
                                #scope.reuse_variables()
                            tower_grads_d.append(discrim_grads_and_vars)
                                
                            scope.reuse_variables()
                            with tf.variable_scope('generator_train'):
                                # Need to wait discriminator to perform train step
                                with tf.control_dependencies( tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                                    gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                                    gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
                                    gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
                                    #gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars)
                            tower_grads.append(gen_grads_and_vars)

    #[ToDo] If we do not use moving average on loss??
    exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss = exp_averager.apply([discrim_loss, adversarial_loss, content_loss])
    
    
    #discriminator aggregation
    avg_grads_d = average_gradients(tower_grads_d)
    discrim_train = discrim_optimizer.apply_gradients(avg_grads_d)
    
    all_outputs_real_d = tf.concat(tower_outputs_real_d, axis=0)
    all_outputs_fake_d = tf.concat(tower_outputs_fake_d, axis=0)
    
    with tf.control_dependencies([discrim_train]):
        #generator aggregation
        avg_grads = average_gradients(tower_grads)
        gen_train = gen_optimizer.apply_gradients(avg_grads)

        all_outputs_g = tf.concat(tower_outputs, axis=0)
    
    
    return Network(
        discrim_real_output = all_outputs_real_d,
        discrim_fake_output = all_outputs_fake_d,
        discrim_loss = exp_averager.average(discrim_loss),
        discrim_grads_and_vars = discrim_grads_and_vars,
        adversarial_loss = exp_averager.average(adversarial_loss),
        content_loss = exp_averager.average(content_loss),
        gen_grads_and_vars = gen_grads_and_vars,
        gen_output = all_outputs_g,
        train = tf.group(update_loss, incr_global_step, gen_train),
        global_step = global_step,
        learning_rate = learning_rate
    )


def SRResnet(inputs, targets, FLAGS, devices = ['/gpu:%d'%i for i in range(8)]):
    # Define the container of the parameter
    Network = collections.namedtuple('Network', 'content_loss, gen_grads_and_vars, gen_output, train, global_step, \
            learning_rate')
    tower_grads = []
    tower_outputs = []
    with tf.device('/gpu:0'):
        split_inputs = tf.split(inputs, len(devices), axis=0)
        split_targets = tf.split(targets, len(devices), axis=0)
        # Define the learning rate and global step
        with tf.variable_scope('get_learning_rate_and_global_step'):
            global_step = tf.contrib.framework.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate,
                                                       staircase=FLAGS.stair)
            incr_global_step = tf.assign(global_step, global_step + 1)


        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i, (inputs, targets, dev) in enumerate(zip(split_inputs, split_targets, devices)):        
                    with tf.device(dev):
                        with tf.name_scope('tower%d'%i):
                            # Build the generator part
                            with tf.variable_scope('generator'):
                                output_channel = targets.get_shape().as_list()[-1]
                                gen_output = generator(inputs, output_channel, reuse=False, FLAGS=FLAGS)
                                gen_output.set_shape([FLAGS.batch_size/len(devices), FLAGS.crop_size * 4, FLAGS.crop_size * 4, 3])
                                tower_outputs.append(gen_output)

                            # Use the VGG54 feature
                            if FLAGS.perceptual_mode == 'VGG54':
                                with tf.name_scope('vgg19_1') as scope:
                                    extracted_feature_gen = VGG19_slim(gen_output, FLAGS.perceptual_mode, reuse=False, scope=scope)
                                with tf.name_scope('vgg19_2') as scope:
                                    extracted_feature_target = VGG19_slim(targets, FLAGS.perceptual_mode, reuse=True, scope=scope)

                            elif FLAGS.perceptual_mode == 'VGG22':
                                with tf.name_scope('vgg19_1') as scope:
                                    extracted_feature_gen = VGG19_slim(gen_output, FLAGS.perceptual_mode, reuse=False, scope=scope)
                                with tf.name_scope('vgg19_2') as scope:
                                    extracted_feature_target = VGG19_slim(targets, FLAGS.perceptual_mode, reuse=True, scope=scope)

                            elif FLAGS.perceptual_mode == 'MSE':
                                extracted_feature_gen = gen_output
                                extracted_feature_target = targets

                            else:
                                raise NotImplementedError('Unknown perceptual type')

                            # Calculating the generator loss
                            with tf.variable_scope('generator_loss'):
                                # Content loss
                                with tf.variable_scope('content_loss'):
                                    # Compute the euclidean distance between the two features
                                    # check=tf.equal(extracted_feature_gen, extracted_feature_target)
                                    diff = extracted_feature_gen - extracted_feature_target
                                    if FLAGS.perceptual_mode == 'MSE':
                                        content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))
                                    else:
                                        content_loss = FLAGS.vgg_scaling * tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))

                                gen_loss = content_loss

                            scope.reuse_variables()
                            with tf.variable_scope('generator_train'):
                                # Need to wait discriminator to perform train step
                                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                                    gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                                    gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
                                    gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, gen_tvars) 
                            tower_grads.append(gen_grads_and_vars)
        # [ToDo] If we do not use moving average on loss??
        exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
        update_loss = exp_averager.apply([content_loss])

        avg_grads = average_gradients(tower_grads)
        gen_train = gen_optimizer.apply_gradients(avg_grads)

        all_outputs = tf.concat(tower_outputs, axis=0)
        return Network(
        content_loss=exp_averager.average(content_loss),
        gen_grads_and_vars=gen_grads_and_vars,
        gen_output=all_outputs,
        train=tf.group(update_loss, incr_global_step, gen_train),
        global_step=global_step,
        learning_rate=learning_rate
    )


def save_images(fetches, FLAGS, step=None):
    image_dir = os.path.join(FLAGS.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    in_path = fetches['path_LR']
    name, _ = os.path.splitext(os.path.basename(str(in_path)))
    fileset = {"name": name, "step": step}

    if FLAGS.mode == 'inference':
        kind = "outputs"
        filename = name + ".jpg"
        if step is not None:
            filename = "%08d-%s" % (step, filename)
        fileset[kind] = filename
        out_path = os.path.join(image_dir, filename)
        contents = fetches[kind][0]
        with open(out_path, "wb") as f:
            f.write(contents)
        filesets.append(fileset)
    else:
        psnr = fetches['psnr']
        ssim = fetches['SSIM']
        for kind in ["inputs", "outputs", "targets"]:
            if kind == "outputs":
                filename = name + "-" + kind + "(PSNR: " +str(psnr)+" and SSIM: "+str(ssim)+ ").jpg" 
            else:
                filename = name + "-" + kind + ".jpg"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][0]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads












