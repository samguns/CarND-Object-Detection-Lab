import tensorflow as tf
import numpy as np



def vanilla_conv_block(x, kernel_size, output_channels):
    """
    Vanilla Conv -> Batch Norm -> ReLU
    """
    x = tf.layers.conv2d(
        x, output_channels, kernel_size, (2, 2), padding='SAME')
    x = tf.layers.batch_normalization(x)
    return tf.nn.relu(x)


def mobilenet_conv_block(x, kernel_size, output_channels):
    """
    Depthwise Conv -> Batch Norm -> ReLU -> Pointwise Conv -> Batch Norm -> ReLU
    """
    input_channel_dim = x.get_shape().as_list()[-1]
    W = tf.Variable(tf.truncated_normal((kernel_size, kernel_size, input_channel_dim, 1)))
    x = tf.nn.depthwise_conv2d(x, W, (1, 2, 2, 1), padding='SAME')
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, output_channels, (1, 1), padding='SAME')
    x = tf.layers.batch_normalization(x)
    return tf.nn.relu(x)


def compare_parameters():
    INPUT_CHANNELS = 32
    OUTPUT_CHANNELS = 512
    KERNEL_SIZE = 3
    IMG_HEIGHT = 256
    IMG_WIDTH = 256

    with tf.Session(graph=tf.Graph()) as sess:
        # input
        x = tf.constant(np.random.randn(1, IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS), dtype=tf.float32)

        with tf.variable_scope('vanilla'):
            vanilla_conv = vanilla_conv_block(x, KERNEL_SIZE, OUTPUT_CHANNELS)

        with tf.variable_scope('mobile'):
            mobilenet_conv = mobilenet_conv_block(x, KERNEL_SIZE, OUTPUT_CHANNELS)

        vanilla_params = [
            (v.name, np.prod(v.get_shape().as_list()))
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vanilla')
        ]

        mobile_params = [
            (v.name, np.prod(v.get_shape().as_list()))
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'mobile')
        ]

        print("VANILLA CONV BLOCK")
        total_vanilla_params = sum([p[1] for p in vanilla_params])
        for p in vanilla_params:
            print("Variable {0}: number of params = {1}".format(p[0], p[1]))
        print("Total number of params =", total_vanilla_params)
        print()

        print("MOBILENET CONV BLOCK")
        total_mobile_params = sum([p[1] for p in mobile_params])
        for p in mobile_params:
            print("Variable {0}: number of params = {1}".format(p[0], p[1]))
        print("Total number of params =", total_mobile_params)
        print()

        print("{0:.3f}x parameter reduction".format(total_vanilla_params /
                                                    total_mobile_params))


if __name__ == "__main__":
    compare_parameters()