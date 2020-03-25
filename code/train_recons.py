# Train the DenseFuse Net

from __future__ import print_function

import scipy.io as scio
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import attention
import decoder
from ssim_loss_function import SSIM_LOSS
from densefuse_net import DenseFuseNet
from utils import get_train_images

STYLE_LAYERS  = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

TRAINING_IMAGE_SHAPE = (256, 256, 1) # (height, width, color_channels)
TRAINING_IMAGE_SHAPE_OR = (256, 256, 1) # (height, width, color_channels)

LEARNING_RATE = 1e-4
LEARNING_RATE_2 = 1e-4

EPSILON = 1e-5



def train_recons(original_imgs_path, validatioin_imgs_path, save_path, model_pre_path, ssim_weight, EPOCHES_set, BATCH_SIZE, debug=False, logging_period=1):
    if debug:
        from datetime import datetime
        start_time = datetime.now()
    EPOCHS = EPOCHES_set
    print("EPOCHES   : ", EPOCHS)             #EPOCHS = 4           遍历整个数据集的次数，训练网络一共要执行n*4次
    print("BATCH_SIZE: ", BATCH_SIZE)         #BATCH_SIZE = 2       每个Batch有2个样本，共n/2个Batch，每处理两个样本模型权重就更新

    num_val = len(validatioin_imgs_path)        #测试集样本个数
    num_imgs = len(original_imgs_path)          #训练集样本个数
    # num_imgs = 100
    original_imgs_path = original_imgs_path[:num_imgs]                          #迷惑行为，自己赋给自己
    mod = num_imgs % BATCH_SIZE                 #Batch个数

    print('Train images number %d.\n' % num_imgs)
    print('Train images samples %s.\n' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]                          #original_imags_path 数组移除最后两个

    # get the traing image shape
    #训练图像的长宽及通道数    255，255，1
    HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
    INPUT_SHAPE = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)                         #定义元组，意义不明

    HEIGHT_OR, WIDTH_OR, CHANNELS_OR = TRAINING_IMAGE_SHAPE_OR
    INPUT_SHAPE_OR = (BATCH_SIZE, HEIGHT_OR, WIDTH_OR, CHANNELS_OR)             #OR是什么意思，意义不明

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        original = tf.placeholder(tf.float32, shape=INPUT_SHAPE_OR, name='original')
        #神经网络构建graph的时候在模型中的占位，只分配必要的内存，运行模型时通过feed_dict()向占位符喂入数据
        #第一个参数，数据类型，常用tf.float32,tf.float64
        #第二个参数，数据形状，矩阵形状，图像的长宽及通道数
        #第三个参数，名称
        #返回Tensor类型
        source = original                                               #迷惑行为，意义不明

        print('source  :', source.shape)
        print('original:', original.shape)

        # create the deepfuse net (encoder and decoder)
        #创建深度学习网络
        dfn = DenseFuseNet(model_pre_path)                              #这里的model_pre_path是自己设置的模型参数，默认是None，若不为None则起始训练的参数为设置的文件
        generated_img = dfn.transform_recons(source)                    #输出图像
        print('generate:', generated_img.shape)

        #########################################################################################
        # COST FUNCTION 部分
        ssim_loss_value = SSIM_LOSS(original, generated_img)                #计算SSIM
        pixel_loss = tf.reduce_sum(tf.square(original - generated_img))
        pixel_loss = pixel_loss/(BATCH_SIZE*HEIGHT*WIDTH)                   #计算pixel loss
        ssim_loss = 1 - ssim_loss_value                                     #SSIM loss数值

        loss = ssim_weight*ssim_loss + pixel_loss                           #整体loss
        #train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)     #自适应矩估计（梯度下降的一种方法）
        train_op = tf.train.AdamOptimizer(LEARNING_RATE_2).minimize(loss)  # 自适应矩估计（梯度下降的一种方法）
        ##########################################################################################

        sess.run(tf.global_variables_initializer())

        # saver = tf.train.Saver()
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        # ** Start Training **
        step = 0
        count_loss = 0
        n_batches = int(len(original_imgs_path) // BATCH_SIZE)
        val_batches = int(len(validatioin_imgs_path) // BATCH_SIZE)

        if debug:
            elapsed_time = datetime.now() - start_time
            print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
            print('Now begin to train the model...\n')
            start_time = datetime.now()

        Loss_all = [i for i in range(EPOCHS * n_batches)]
        Loss_ssim = [i for i in range(EPOCHS * n_batches)]
        Loss_pixel = [i for i in range(EPOCHS * n_batches)]
        Val_ssim_data = [i for i in range(EPOCHS * n_batches)]
        Val_pixel_data = [i for i in range(EPOCHS * n_batches)]
        for epoch in range(EPOCHS):

            np.random.shuffle(original_imgs_path)

            for batch in range(n_batches):
                # retrive a batch of content and style images

                original_path = original_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
                original_batch = get_train_images(original_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                original_batch = original_batch.reshape([BATCH_SIZE, 256, 256, 1])

                # print('original_batch shape final:', original_batch.shape)

                # run the training step
                sess.run(train_op, feed_dict={original: original_batch})
                step += 1
                if debug:
                    is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)

                    if is_last_step or step % logging_period == 0:
                        elapsed_time = datetime.now() - start_time
                        _ssim_loss, _loss, _p_loss = sess.run([ssim_loss, loss, pixel_loss], feed_dict={original: original_batch})
                        Loss_all[count_loss] = _loss
                        Loss_ssim[count_loss] = _ssim_loss
                        Loss_pixel[count_loss] = _p_loss
                        print('epoch: %d/%d, step: %d,  total loss: %s, elapsed time: %s' % (epoch, EPOCHS, step, _loss, elapsed_time))
                        print('p_loss: %s, ssim_loss: %s ,w_ssim_loss: %s ' % (_p_loss, _ssim_loss, ssim_weight * _ssim_loss))

                        # calculate the accuracy rate for 1000 images, every 100 steps
                        val_ssim_acc = 0
                        val_pixel_acc = 0
                        np.random.shuffle(validatioin_imgs_path)
                        val_start_time = datetime.now()
                        for v in range(val_batches):
                            val_original_path = validatioin_imgs_path[v * BATCH_SIZE:(v * BATCH_SIZE + BATCH_SIZE)]
                            val_original_batch = get_train_images(val_original_path, crop_height=HEIGHT, crop_width=WIDTH,flag=False)
                            val_original_batch = val_original_batch.reshape([BATCH_SIZE, 256, 256, 1])
                            val_ssim, val_pixel = sess.run([ssim_loss, pixel_loss], feed_dict={original: val_original_batch})
                            val_ssim_acc = val_ssim_acc + (1 - val_ssim)
                            val_pixel_acc = val_pixel_acc + val_pixel
                        Val_ssim_data[count_loss] = val_ssim_acc/val_batches
                        Val_pixel_data[count_loss] = val_pixel_acc / val_batches
                        val_es_time = datetime.now() - val_start_time
                        print('validation value, SSIM: %s, Pixel: %s, elapsed time: %s' % (val_ssim_acc/val_batches, val_pixel_acc / val_batches, val_es_time))
                        print('------------------------------------------------------------------------------')
                        count_loss += 1


        # ** Done Training & Save the model **
        saver.save(sess, save_path)
#----------------------------------------------------------------------------------------------------------------
        loss_data = Loss_all[:count_loss]
        scio.savemat('/data/ljy/paper_again/19-11-20-final/loss/DeepDenseLossData' + str(ssim_weight) + '.mat',
                     {'loss': loss_data})

        loss_ssim_data = Loss_ssim[:count_loss]
        scio.savemat('/data/ljy/paper_again/19-11-20-final/loss/DeepDenseLossSSIMData' + str(
            ssim_weight) + '.mat', {'loss_ssim': loss_ssim_data})

        loss_pixel_data = Loss_pixel[:count_loss]
        scio.savemat('/data/ljy/paper_again/19-11-20-final/loss/DeepDenseLossPixelData.mat' + str(
            ssim_weight) + '', {'loss_pixel': loss_pixel_data})

        validation_ssim_data = Val_ssim_data[:count_loss]
        scio.savemat('/data/ljy/paper_again/19-11-20-final/val/Validation_ssim_Data.mat' + str(
            ssim_weight) + '', {'val_ssim': validation_ssim_data})

        validation_pixel_data = Val_pixel_data[:count_loss]
        scio.savemat('/data/ljy/paper_again/19-11-20-final/val/Validation_pixel_Data.mat' + str(
            ssim_weight) + '', {'val_pixel': validation_pixel_data})
#----------------------------------------------------------------------------------------------------
        if debug:
            elapsed_time = datetime.now() - start_time
            print('Done training! Elapsed time: %s' % elapsed_time)
            print('Model is saved to: %s' % save_path)

def train_recons_a(original_imgs_path, validatioin_imgs_path, save_path_a, model_pre_path_a, ssim_weight_a, EPOCHES_set, BATCH_SIZE,MODEL_SAVE_PATHS, debug=False, logging_period=1):
    if debug:
        from datetime import datetime
        start_time = datetime.now()
    EPOCHS = EPOCHES_set
    print("EPOCHES   : ", EPOCHS)  # EPOCHS = 4           遍历整个数据集的次数，训练网络一共要执行n*4次
    print("BATCH_SIZE: ", BATCH_SIZE)  # BATCH_SIZE = 2       每个Batch有2个样本，共n/2个Batch，每处理两个样本模型权重就更新

    num_val = len(validatioin_imgs_path)  # 测试集样本个数
    num_imgs = len(original_imgs_path)  # 训练集样本个数
    # num_imgs = 100
    original_imgs_path = original_imgs_path[:num_imgs]  # 迷惑行为，自己赋给自己
    mod = num_imgs % BATCH_SIZE  # Batch个数

    print('Train images number %d.\n' % num_imgs)
    print('Train images samples %s.\n' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]  # original_imags_path 数组移除最后两个

    # get the traing image shape
    # 训练图像的长宽及通道数    255，255，1
    HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
    INPUT_SHAPE = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)  # 定义元组，意义不明

    HEIGHT_OR, WIDTH_OR, CHANNELS_OR = TRAINING_IMAGE_SHAPE_OR
    INPUT_SHAPE_OR = (BATCH_SIZE, HEIGHT_OR, WIDTH_OR, CHANNELS_OR)  # OR是什么意思，意义不明

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        original = tf.placeholder(tf.float32, shape=INPUT_SHAPE_OR, name='original')
        attention_map = tf.placeholder(tf.float32, shape=INPUT_SHAPE_OR, name='attention')
        # 神经网络构建graph的时候在模型中的占位，只分配必要的内存，运行模型时通过feed_dict()向占位符喂入数据
        # 第一个参数，数据类型，常用tf.float32,tf.float64
        # 第二个参数，数据形状，矩阵形状，图像的长宽及通道数
        # 第三个参数，名称
        # 返回Tensor类型
        source = original  # 迷惑行为，意义不明

        print('source  :', source.shape)
        print('original:', original.shape)

        # create the deepfuse net (encoder and decoder)
        # 创建深度学习网络
        model_pre_path=MODEL_SAVE_PATHS
        dfn = DenseFuseNet(model_pre_path)  # 这里的model_pre_path是自己设置的模型参数，默认是None，若不为None则起始训练的参数为设置的文件

        atn = attention.Attention(None)
        enc, enc_res_block, enc_block, enc_block2 = dfn.transform_encoder(source)
        weight_map=atn.get_attention(attention_map)
        enc_res_block_6_a= tf.multiply(enc_res_block[6],weight_map)
        enc_res_block_9_a=tf.multiply(enc_res_block[9],weight_map)
        feature = enc_res_block[0]
        mix_indices = (1, 2, 3)
        for i in mix_indices:
            feature = tf.concat([feature, enc_res_block[i]], 3)
        t_decode=feature+0.1*enc_res_block_6_a+0.1*enc_res_block_9_a
        generated_img = dfn.transform_decoder(t_decode,enc_block,enc_block2)
        print('generate:', generated_img.shape)
        ssim_loss_value = SSIM_LOSS(original, generated_img)  # 计算SSIM
        pixel_loss = tf.reduce_sum(tf.square(original - generated_img))
        pixel_loss = pixel_loss / (BATCH_SIZE * HEIGHT * WIDTH)  # 计算pixel loss
        ssim_loss = 1 - ssim_loss_value  # SSIM loss数值

        loss = ssim_weight_a * ssim_loss + pixel_loss  # 整体loss
        # train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)     #自适应矩估计（梯度下降的一种方法）
        train_op = tf.train.AdamOptimizer(LEARNING_RATE_2).minimize(loss,var_list=atn.weights)  # 自适应矩估计（梯度下降的一种方法）
        sess.run(tf.global_variables_initializer())

        # saver = tf.train.Saver()
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        # ** Start Training **
        step = 0
        count_loss = 0
        n_batches = int(len(original_imgs_path) // BATCH_SIZE)
        val_batches = int(len(validatioin_imgs_path) // BATCH_SIZE)

        if debug:
            elapsed_time = datetime.now() - start_time
            print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
            print('Now begin to train the model...\n')
            start_time = datetime.now()

        Loss_all = [i for i in range(EPOCHS * n_batches)]
        Loss_ssim = [i for i in range(EPOCHS * n_batches)]
        Loss_pixel = [i for i in range(EPOCHS * n_batches)]
        Val_ssim_data = [i for i in range(EPOCHS * n_batches)]
        Val_pixel_data = [i for i in range(EPOCHS * n_batches)]
        for epoch in range(EPOCHS):

            np.random.shuffle(original_imgs_path)

            for batch in range(n_batches):
                # retrive a batch of content and style images

                original_path = original_imgs_path[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE)]
                original_batch = get_train_images(original_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                original_batch = original_batch.reshape([BATCH_SIZE, 256, 256, 1])

                # print('original_batch shape final:', original_batch.shape)
                # -----------------------------------------------
                imag = sess.run(original, feed_dict={original: original_batch})
                guideFilter_imgs = np.zeros(INPUT_SHAPE_OR)
                for i in range(BATCH_SIZE):
                    input = np.squeeze(imag[i])
                    out = atn.Grad(input)
                    out = np.expand_dims(out, axis=-1)
                    out[out < 0] = 0
                    guideFilter_imgs[i] = out
                # ----------------------------------------------
                # run the training step
                sess.run(train_op, feed_dict={original: original_batch, attention_map:guideFilter_imgs })
                step += 1
                if debug:
                    is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)

                    if is_last_step or step % logging_period == 0:
                        elapsed_time = datetime.now() - start_time
                        _ssim_loss, _loss, _p_loss = sess.run([ssim_loss, loss, pixel_loss],
                                                              feed_dict={original: original_batch, attention_map: guideFilter_imgs})
                        Loss_all[count_loss] = _loss
                        Loss_ssim[count_loss] = _ssim_loss
                        Loss_pixel[count_loss] = _p_loss
                        print('epoch: %d/%d, step: %d,  total loss: %s, elapsed time: %s' % (
                        epoch, EPOCHS, step, _loss, elapsed_time))
                        print('p_loss: %s, ssim_loss: %s ,w_ssim_loss: %s ' % (
                        _p_loss, _ssim_loss, ssim_weight_a * _ssim_loss))

                        # calculate the accuracy rate for 1000 images, every 100 steps
                        val_ssim_acc = 0
                        val_pixel_acc = 0
                        np.random.shuffle(validatioin_imgs_path)
                        val_start_time = datetime.now()
                        for v in range(val_batches):
                            val_original_path = validatioin_imgs_path[v * BATCH_SIZE:(v * BATCH_SIZE + BATCH_SIZE)]
                            val_original_batch = get_train_images(val_original_path, crop_height=HEIGHT,
                                                                  crop_width=WIDTH, flag=False)
                            val_original_batch = val_original_batch.reshape([BATCH_SIZE, 256, 256, 1])
                            val_ssim, val_pixel = sess.run([ssim_loss, pixel_loss],
                                                           feed_dict={original: val_original_batch, attention_map: guideFilter_imgs})
                            val_ssim_acc = val_ssim_acc + (1 - val_ssim)
                            val_pixel_acc = val_pixel_acc + val_pixel
                        Val_ssim_data[count_loss] = val_ssim_acc / val_batches
                        Val_pixel_data[count_loss] = val_pixel_acc / val_batches
                        val_es_time = datetime.now() - val_start_time
                        print('validation value, SSIM: %s, Pixel: %s, elapsed time: %s' % (
                        val_ssim_acc / val_batches, val_pixel_acc / val_batches, val_es_time))
                        print('------------------------------------------------------------------------------')
                        count_loss += 1

        # ** Done Training & Save the model **
        saver.save(sess, save_path_a)
        # ----------------------------------------------------------------------------------------------------------------
        loss_data = Loss_all[:count_loss]
        scio.savemat('/data/ljy/paper_again/19-11-20-final/model_a/loss/DeepDenseLossData' + str(ssim_weight_a) + '.mat',
                     {'loss': loss_data})

        loss_ssim_data = Loss_ssim[:count_loss]
        scio.savemat('/data/ljy/paper_again/19-11-20-final/model_a/loss/DeepDenseLossSSIMData' + str(
            ssim_weight_a) + '.mat', {'loss_ssim': loss_ssim_data})

        loss_pixel_data = Loss_pixel[:count_loss]
        scio.savemat('/data/ljy/paper_again/19-11-20-final/model_a/loss/DeepDenseLossPixelData.mat' + str(
            ssim_weight_a) + '', {'loss_pixel': loss_pixel_data})

        validation_ssim_data = Val_ssim_data[:count_loss]
        scio.savemat('/data/ljy/paper_again/19-11-20-final/model_a/val/Validation_ssim_Data.mat' + str(
            ssim_weight_a) + '', {'val_ssim': validation_ssim_data})

        validation_pixel_data = Val_pixel_data[:count_loss]
        scio.savemat('/data/ljy/paper_again/19-11-20-final/model_a/val/Validation_pixel_Data.mat' + str(
            ssim_weight_a) + '', {'val_pixel': validation_pixel_data})
        # ----------------------------------------------------------------------------------------------------
        if debug:
            elapsed_time = datetime.now() - start_time
            print('Done training! Elapsed time: %s' % elapsed_time)
            print('Model is saved to: %s' % save_path_a)
