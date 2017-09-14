import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    print("Loading VGG model and weights for the encoder")
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    print("Take out layers and keep_prob for skip layers")
    graph = tf.get_default_graph()

    print("Get all the tensors by name")
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # We are adding layers after vgg layers
    print("Build Skip layers using conv2d_transpose for the decoder")
    kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01)

    print("1x1 convolution on vgg_layer7")
    layer7_conv = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,
                                   padding='same',
                                   kernel_initializer=kernel_initializer,
                                   strides=(1,1))

    print("Upsample 1")
    up1 = tf.layers.conv2d_transpose(layer7_conv, num_classes, 4,
                                     padding='same',
                                     kernel_initializer=kernel_initializer,
                                     strides=(2,2))

    print("1x1 convolution on vgg_layer4")
    layer4_conv = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,
                                   padding='same',
                                   kernel_initializer=kernel_initializer,
                                   strides=(1,1))

    print("Skip vgg_layer4")
    skip1 = tf.add(layer4_conv, up1)

    print("Upsample 2")
    up2 = tf.layers.conv2d_transpose(skip1, num_classes, 4,
                                     padding='same',
                                     kernel_initializer=kernel_initializer,
                                     strides=(2,2))

    print("1x1 convolution on vgg_layer3")
    layer3_conv = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,
                                   padding='same',
                                   kernel_initializer=kernel_initializer,
                                   strides=(1,1))
    print("Skip vgg_layer3")
    skip2 = tf.add(layer3_conv, up2)

    print("Upsample 3")
    input6 = tf.layers.conv2d_transpose(skip2, num_classes, 16,
                                        padding='same',
                                        kernel_initializer=kernel_initializer,
                                        strides=(8,8))
    return input6
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    print("Reshape logits and labels")
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    print("Cross entropy loss")
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    print("Training optimizer")
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param image_input: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    print("training..\n")
    index = 0
    KEEP_PROBABILITY = np.float32(0.75)
    LEARN_RATE = np.float32(0.0001)

    for epoch in range(epochs):
        print("EPOCH {} ...".format(epoch))
        # shuffle the batches? nope, done in the get_batches_fn function
        for image, label in get_batches_fn(batch_size):
            index += 1
            # do training
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={image_input: image,
                                          correct_label: label,
                                          keep_prob: KEEP_PROBABILITY,
                                          learning_rate: LEARN_RATE
                                          })

            print("Iteration among the batch:", '%04d | ' % (index), "cost =", "{:.9f}".format(loss))
        print("Epoch:", '%04d | ' % (epoch+1), "cost =", "{:.9f}".format(loss))
    pass
tests.test_train_nn(train_nn)


def run():
    print("starting run()..\n")
    num_classes = 2
    image_shape = (160, 576)
    epochs = 20
    batch_size = 17 # 289 is divisible by 17, easy way to batch up evenly

    data_dir = './data'
    runs_dir = './runs'
    print("testing kitti dataset..\n")
    tests.test_for_kitti_dataset(data_dir)

    print("download vgg..\n")
    # Download pre-trained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    print("start tf session..\n")
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        print("Create function to get batches")
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        print("Build Encoder portion of NN using load_vgg")
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        print("Create decoder and FCN using layers function")
        last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        print("Create TF Placeholder for correct_label")
        # TODO: CAN I MAKE THE BATCH_SIZE PORTION OF THIS A VARIABLE? THAT WAY IT DOES IT WITH WHATEVER IS AVAILABLE?
        # only two channels because the correct stuff is just using pink and not pink.
        correct_label = tf.placeholder(tf.float32, shape=(batch_size, image_shape[0], image_shape[1], 2), name='correct_label')

        print("Create TF Placeholder for learning_rate")
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        print("Create an optimization function that will be used to train the neural network")
        logits, train_op, cross_entropy_loss = optimize(last_layer, correct_label, learning_rate,
                                                        num_classes)

        sess.run(tf.global_variables_initializer())
        print("Train NN using the train_nn function")
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob, learning_rate)

        print("Save inference data from trained model and run NN on the test directory")
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits,
                                      keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
