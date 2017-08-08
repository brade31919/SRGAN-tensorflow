from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from lib.model import data_loader, generator, SRGAN, test_data_loader, inference_data_loader, save_images, SRResnet
from lib.ops import *
import math
import time
import numpy as np

Flags = tf.app.flags

# The system parameter
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint')
Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')
Flags.DEFINE_string('mode', 'train', 'The mode of the model train, test.')
Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint')
Flags.DEFINE_boolean('pre_trained_model', False, 'If set True, the weight will be loaded but the global_step will still '
                                                 'be 0. If set False, you are going to continue the training. That is, '
                                                 'the global_step will be initiallized from the checkpoint, too')
Flags.DEFINE_string('pre_trained_model_type', 'SRResnet', 'The type of pretrained model (SRGAN or SRResnet)')
Flags.DEFINE_boolean('is_training', True, 'Training => True, Testing => False')
Flags.DEFINE_string('vgg_ckpt', './vgg19/vgg_19.ckpt', 'path to checkpoint file for the vgg19')
Flags.DEFINE_string('task', None, 'The task: SRGAN, SRResnet')
# The data preparing operation
Flags.DEFINE_integer('batch_size', 16, 'Batch size of the input batch')
Flags.DEFINE_string('input_dir_LR', None, 'The directory of the input resolution input data')
Flags.DEFINE_string('input_dir_HR', None, 'The directory of the high resolution input data')
Flags.DEFINE_boolean('flip', True, 'Whether random flip data augmentation is applied')
Flags.DEFINE_boolean('random_crop', True, 'Whether perform the random crop')
Flags.DEFINE_integer('crop_size', 24, 'The crop size of the training image')
Flags.DEFINE_integer('name_queue_capacity', 2048, 'The capacity of the filename queue (suggest large to ensure'
                                                  'enough random shuffle.')
Flags.DEFINE_integer('image_queue_capacity', 2048, 'The capacity of the image queue (suggest large to ensure'
                                                   'enough random shuffle')
Flags.DEFINE_integer('queue_thread', 10, 'The threads of the queue (More threads can speedup the training process.')
# Generator configuration
Flags.DEFINE_integer('num_resblock', 16, 'How many residual blocks are there in the generator')
# The content loss parameter
Flags.DEFINE_string('perceptual_mode', 'VGG54', 'The type of feature used in perceptual loss')
Flags.DEFINE_float('EPS', 1e-12, 'The eps added to prevent nan')
Flags.DEFINE_float('ratio', 0.001, 'The ratio between content loss and adversarial loss')
Flags.DEFINE_float('vgg_scaling', 0.0061, 'The scaling factor for the perceptual loss if using vgg perceptual loss')
# The training parameters
Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
Flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
Flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving images')


FLAGS = Flags.FLAGS

# Print the configuration of the model
print_configuration_op(FLAGS)

# Check the output_dir is given
if FLAGS.output_dir is None:
    raise ValueError('The output directory is needed')

# Check the output directory to save the checkpoint
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

# Check the summary directory to save the event
if not os.path.exists(FLAGS.summary_dir):
    os.mkdir(FLAGS.summary_dir)

# The testing mode
if FLAGS.mode == 'test':
    # Check the checkpoint
    if FLAGS.checkpoint is None:
        raise ValueError('The checkpoint file is needed to performing the test.')

    # In the testing time, no flip and crop is needed
    if FLAGS.flip == True:
        FLAGS.flip = False

    if FLAGS.crop_size is not None:
        FLAGS.crop_size = None

    # Declare the test data reader
    test_data = test_data_loader(FLAGS)

    inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='inputs_raw')
    targets_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='targets_raw')
    path_LR = tf.placeholder(tf.string, shape=[], name='path_LR')
    path_HR = tf.placeholder(tf.string, shape=[], name='path_HR')

    with tf.variable_scope('generator'):
        if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
            gen_output = generator(inputs_raw, 3, reuse=False, FLAGS=FLAGS)
        else:
            raise NotImplementedError('Unknown task!!')

    print('Finish building the network')

    with tf.name_scope('convert_image'):
        # Deprocess the images outputed from the model
        inputs = deprocessLR(inputs_raw)
        targets = deprocess(targets_raw)
        outputs = deprocess(gen_output)

        # Convert back to uint8
        converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
        converted_targets = tf.image.convert_image_dtype(targets, dtype=tf.uint8, saturate=True)
        converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

    with tf.name_scope('encode_image'):
        save_fetch = {
            "path_LR": path_LR,
            "path_HR": path_HR,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name='input_pngs'),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name='output_pngs'),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name='target_pngs')
        }

    # Define the weight initiallizer (In inference time, we only need to restore the weight of the generator)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    weight_initiallizer = tf.train.Saver(var_list)

    # Define the initialization operation
    init_op = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Load the pretrained model
        print('Loading weights from the pre-trained model')
        weight_initiallizer.restore(sess, FLAGS.checkpoint)

        max_iter = len(test_data.inputs)
        print('Evaluation starts!!')
        for i in range(max_iter):
            input_im = np.array([test_data.inputs[i]]).astype(np.float32)
            target_im = np.array([test_data.targets[i]]).astype(np.float32)
            path_lr = test_data.paths_LR[i]
            path_hr = test_data.paths_HR[i]
            results = sess.run(save_fetch, feed_dict={inputs_raw: input_im, targets_raw: target_im,
                                                      path_LR: path_lr, path_HR: path_hr})
            filesets = save_images(results, FLAGS)
            for i, f in enumerate(filesets):
                print('evaluate image', f['name'])


# the inference mode (just perform super resolution on the input image)
elif FLAGS.mode == 'inference':
    # Check the checkpoint
    if FLAGS.checkpoint is None:
        raise ValueError('The checkpoint file is needed to performing the test.')

    # In the testing time, no flip and crop is needed
    if FLAGS.flip == True:
        FLAGS.flip = False

    if FLAGS.crop_size is not None:
        FLAGS.crop_size = None

    # Declare the test data reader
    inference_data = inference_data_loader(FLAGS)

    inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='inputs_raw')
    path_LR = tf.placeholder(tf.string, shape=[], name='path_LR')

    with tf.variable_scope('generator'):
        if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
            gen_output = generator(inputs_raw, 3, reuse=False, FLAGS=FLAGS)
        else:
            raise NotImplementedError('Unknown task!!')

    print('Finish building the network')

    with tf.name_scope('convert_image'):
        # Deprocess the images outputed from the model
        inputs = deprocessLR(inputs_raw)
        outputs = deprocess(gen_output)

        # Convert back to uint8
        converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
        converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

    with tf.name_scope('encode_image'):
        save_fetch = {
            "path_LR": path_LR,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name='input_pngs'),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name='output_pngs')
        }

    # Define the weight initiallizer (In inference time, we only need to restore the weight of the generator)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    weight_initiallizer = tf.train.Saver(var_list)

    # Define the initialization operation
    init_op = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Load the pretrained model
        print('Loading weights from the pre-trained model')
        weight_initiallizer.restore(sess, FLAGS.checkpoint)

        max_iter = len(inference_data.inputs)
        print('Evaluation starts!!')
        for i in range(max_iter):
            input_im = np.array([inference_data.inputs[i]]).astype(np.float32)
            path_lr = inference_data.paths_LR[i]
            results = sess.run(save_fetch, feed_dict={inputs_raw: input_im, path_LR: path_lr})
            filesets = save_images(results, FLAGS)
            for i, f in enumerate(filesets):
                print('evaluate image', f['name'])


# The training mode
elif FLAGS.mode == 'train':
    # Load data for training and testing
    # ToDo Add online downscaling
    data = data_loader(FLAGS)
    print('Data count = %d' % (data.image_count))

    # Connect to the network
    if FLAGS.task == 'SRGAN':
        Net = SRGAN(data.inputs, data.targets, FLAGS)
    elif FLAGS.task =='SRResnet':
        Net = SRResnet(data.inputs, data.targets, FLAGS)
    else:
        raise NotImplementedError('Unknown task type')

    print('Finish building the network!!!')

    # Convert the images output from the network
    with tf.name_scope('convert_image'):
        # Deprocess the images outputed from the model
        inputs = deprocessLR(data.inputs)
        targets = deprocess(data.targets)
        outputs = deprocess(Net.gen_output)

        # Convert back to uint8
        converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
        converted_targets = tf.image.convert_image_dtype(targets, dtype=tf.uint8, saturate=True)
        converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

    # Compute PSNR
    with tf.name_scope("compute_psnr"):
        psnr = compute_psnr(converted_targets, converted_outputs)

    # Add image summaries
    with tf.name_scope('inputs_summary'):
        tf.summary.image('input_summary', converted_inputs)

    with tf.name_scope('targets_summary'):
        tf.summary.image('target_summary', converted_targets)

    with tf.name_scope('outputs_summary'):
        tf.summary.image('outputs_summary', converted_outputs)

    # Add scalar summary
    if FLAGS.task == 'SRGAN':
        tf.summary.scalar('discriminator_loss', Net.discrim_loss)
        tf.summary.scalar('adversarial_loss', Net.adversarial_loss)
        tf.summary.scalar('content_loss', Net.content_loss)
        tf.summary.scalar('generator_loss', Net.content_loss + FLAGS.ratio*Net.adversarial_loss)
        tf.summary.scalar('PSNR', psnr)
        tf.summary.scalar('learning_rate', Net.learning_rate)
    elif FLAGS.task == 'SRResnet':
        tf.summary.scalar('content_loss', Net.content_loss)
        tf.summary.scalar('generator_loss', Net.content_loss)
        tf.summary.scalar('PSNR', psnr)
        tf.summary.scalar('learning_rate', Net.learning_rate)


    # Define the saver and weight initiallizer
    saver = tf.train.Saver(max_to_keep=10)

    # The variable list
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # Here if we restore the weight from the SRResnet the var_list2 do not need to contain the discriminator weights
    # On contrary, if you initial your weight from other SRGAN checkpoint, var_list2 need to contain discriminator
    # weights.
    if FLAGS.task == 'SRGAN':
        if FLAGS.pre_trained_model_type == 'SRGAN':
            var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator') + \
                      tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        elif FLAGS.pre_trained_model_type == 'SRResnet':
            var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        else:
            raise ValueError('Unknown pre_trained model type!!')
    elif FLAGS.task == 'SRResnet':
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    weight_initiallizer = tf.train.Saver(var_list2)

    # When using MSE loss, no need to restore the vgg net
    if not FLAGS.perceptual_mode == 'MSE':
        vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
        vgg_restore = tf.train.Saver(vgg_var_list)

    # Start the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Use superviser to coordinate all queue and summary writer
    sv = tf.train.Supervisor(logdir=FLAGS.summary_dir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=config) as sess:
        if (FLAGS.checkpoint is not None) and (FLAGS.pre_trained_model is False):
            print('Loading model from the checkpoint...')
            checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
            saver.restore(sess, checkpoint)

        elif (FLAGS.checkpoint is not None) and (FLAGS.pre_trained_model is True):
            print('Loading weights from the pre-trained model')
            weight_initiallizer.restore(sess, FLAGS.checkpoint)

        if not FLAGS.perceptual_mode == 'MSE':
            vgg_restore.restore(sess, FLAGS.vgg_ckpt)
            print('VGG19 restored successfully!!')

        # Performing the training
        if FLAGS.max_epoch is None:
            if FLAGS.max_iter is None:
                raise ValueError('one of max_epoch or max_iter should be provided')
            else:
                max_iter = FLAGS.max_iter
        else:
            max_iter = FLAGS.max_epoch * data.steps_per_epoch

        print('Optimization starts!!!')
        start = time.time()
        for step in range(max_iter):
            fetches = {
                "train": Net.train,
                "global_step": sv.global_step,
            }

            if ((step+1) % FLAGS.display_freq) == 0:
                if FLAGS.task == 'SRGAN':
                    fetches["discrim_loss"] = Net.discrim_loss
                    fetches["adversarial_loss"] = Net.adversarial_loss
                    fetches["content_loss"] = Net.content_loss
                    fetches["PSNR"] = psnr
                    fetches["learning_rate"] = Net.learning_rate
                    fetches["global_step"] = Net.global_step
                elif FLAGS.task == 'SRResnet':
                    fetches["content_loss"] = Net.content_loss
                    fetches["PSNR"] = psnr
                    fetches["learning_rate"] = Net.learning_rate
                    fetches["global_step"] = Net.global_step

            if ((step+1) % FLAGS.summary_freq) == 0:
                fetches["summary"] = sv.summary_op

            results = sess.run(fetches)

            if ((step + 1) % FLAGS.summary_freq) == 0:
                print('Recording summary!!')
                sv.summary_writer.add_summary(results['summary'], results['global_step'])

            if ((step + 1) % FLAGS.display_freq) == 0:
                train_epoch = math.ceil(results["global_step"] / data.steps_per_epoch)
                train_step = (results["global_step"] - 1) % data.steps_per_epoch + 1
                rate = (step + 1) * FLAGS.batch_size / (time.time() - start)
                remaining = (max_iter - step) * FLAGS.batch_size / rate
                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                if FLAGS.task == 'SRGAN':
                    print("global_step", results["global_step"])
                    print("PSNR", results["PSNR"])
                    print("discrim_loss", results["discrim_loss"])
                    print("adversarial_loss", results["adversarial_loss"])
                    print("content_loss", results["content_loss"])
                    print("learning_rate", results['learning_rate'])
                elif FLAGS.task == 'SRResnet':
                    print("global_step", results["global_step"])
                    print("PSNR", results["PSNR"])
                    print("content_loss", results["content_loss"])
                    print("learning_rate", results['learning_rate'])

            if ((step +1) % FLAGS.save_freq) == 0:
                print('Save the checkpoint')
                saver.save(sess, os.path.join(FLAGS.output_dir, 'model'), global_step=sv.global_step)

        print('Optimization done!!!!!!!!!!!!')








