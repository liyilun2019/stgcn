# @Time     : Jan. 12, 2019 19:01
# @Author   : Veritas YIN
# @FileName : base_model.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from models.layers import *
from os.path import join as pjoin
import tensorflow as tf


def build_model(inputs, n_his, Ks, Kt, blocks, keep_prob,n_pred=8):
    '''
    Build the base model.
    :param inputs: placeholder.
    :param n_his: int, size of historical records for training.
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param blocks: list, channel configs of st_conv blocks.
    :param keep_prob: placeholder.
    '''
    x = inputs[:, 0:n_his, :, :]
    print(f"inputs:{inputs.shape}")
    print(f"x.shape={x.shape}")

    # Ko>0: kernel size of temporal convolution in the output layer.
    Ko = n_his
    # ST-Block
    for i, channels in enumerate(blocks):
        x = st_conv_block(x, Ks, Kt, channels, i, keep_prob, act_func='GLU')
        Ko -= 2 * (Kt - 1)
        print(f"x.shape={x.shape}")
    print(f"Ko={Ko}")
    Ko-=n_pred-1
    print(f"Ko'={Ko}")
    # Output Layer
    if Ko > 1:
        y = output_layer(x, Ko, 'output_layer')
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    y_=inputs[:,n_his:n_his+n_pred,:,:];
    print(f"y.shape={y.shape}")
    print(f"y_.shape={y_.shape}")

    tf.add_to_collection(name='copy_loss',
                         value=tf.nn.l2_loss(inputs[:, n_his - 1:n_his, :, :] - inputs[:, n_his:n_his + 1, :, :]))
    train_loss = tf.nn.l2_loss(y - y_)
    single_pred = y
    tf.add_to_collection(name='y_pred', value=single_pred)
    return train_loss, single_pred


def model_save(sess, global_steps, model_name, save_path='./output/models/'):
    '''
    Save the checkpoint of trained model.
    :param sess: tf.Session().
    :param global_steps: tensor, record the global step of training in epochs.
    :param model_name: str, the name of saved model.
    :param save_path: str, the path of saved model.
    :return:
    '''
    saver = tf.train.Saver(max_to_keep=3)
    prefix_path = saver.save(sess, pjoin(save_path, model_name), global_step=global_steps)
    print(f'<< Saving model to {prefix_path} ...')
