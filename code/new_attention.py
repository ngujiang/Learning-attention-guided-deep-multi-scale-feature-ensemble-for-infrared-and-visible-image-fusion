from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from bilinear_sampler import *

monodepth_parameters = namedtuple('parameters',
                                  'encoder, '
                                  'height, width, '
                                  'batch_size, '
                                  'num_threads, '
                                  'num_epochs, '
                                  'do_stereo, '
                                  'wrap_mode, '
                                  'use_deconv, '
                                  'alpha_image_loss, '
                                  'disp_gradient_loss_weight, '
                                  'lr_loss_weight, '
                                  'full_summary')


class MonodepthModel(object):
	"""monodepth model"""

	def __init__(self, params, mode, left, right, reuse_variables=None, model_index=0):
		self.params = params
		self.mode = mode
		self.left = left
		self.right = right
		self.model_collection = ['model_' + str(model_index)]

		self.reuse_variables = reuse_variables

		self.build_model()
		self.build_outputs()

		if self.mode == 'test':
			return

		self.build_losses()
		self.build_summaries()

	def gradient_x(self, img):
		gx = img[:, :, :-1, :] - img[:, :, 1:, :]
		return gx

	def gradient_y(self, img):
		gy = img[:, :-1, :, :] - img[:, 1:, :, :]
		return gy

	def upsample_nn(self, x, ratio):
		s = tf.shape(x)
		h = s[1]
		w = s[2]
		return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

	def scale_pyramid(self, img, num_scales):
		scaled_imgs = [img]
		s = tf.shape(img)
		h = s[1]
		w = s[2]
		for i in range(num_scales - 1):
			ratio = 2 ** (i + 1)
			nh = h // ratio
			nw = w // ratio
			scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
		return scaled_imgs

	def generate_image_left(self, img, disp):
		return bilinear_sampler_1d_h(img, -disp)

	def generate_image_right(self, img, disp):
		return bilinear_sampler_1d_h(img, disp)

	def SSIM(self, x, y):
		C1 = 0.01 ** 2
		C2 = 0.03 ** 2

		mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
		mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

		sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
		sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
		sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

		SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
		SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

		SSIM = SSIM_n / SSIM_d

		return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

	def get_disparity_smoothness(self, disp, pyramid):
		disp_gradients_x = [self.gradient_x(d) for d in disp]
		disp_gradients_y = [self.gradient_y(d) for d in disp]

		image_gradients_x = [self.gradient_x(img) for img in pyramid]
		image_gradients_y = [self.gradient_y(img) for img in pyramid]

		weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
		weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

		smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
		smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
		return smoothness_x + smoothness_y

	def get_disp(self, x):
		disp = 0.3 * self.conv(x, 2, 3, 1, tf.nn.sigmoid)
		return disp

	def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
		p = np.floor((kernel_size - 1) / 2).astype(np.int32)
		p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
		return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

	def conv_dia(self, x, num_out_layers, kernel_size, stride, rate_dia, activation_fn=tf.nn.elu):
		# p = np.floor((kernel_size - 1) / 2).astype(np.int32)
		# p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
		stride = 1
		return slim.conv2d(x, num_out_layers, kernel_size, stride, 'SAME', activation_fn=activation_fn, rate=rate_dia)

	def conv_block(self, x, num_out_layers, kernel_size):
		conv1 = self.conv(x, num_out_layers, kernel_size, 1)
		conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
		return conv2

	def maxpool(self, x, kernel_size):
		p = np.floor((kernel_size - 1) / 2).astype(np.int32)
		p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
		return slim.max_pool2d(p_x, kernel_size)

	##########################################################################CBAM-Net
	def cbam_module(self, inputs, out_dim, reduction_ratio=0.5, name=""):
		with tf.variable_scope("cbam_" + name, reuse=tf.AUTO_REUSE):  ##tf.AUTO_REUSE
			batch_size, hidden_num = inputs.get_shape().as_list()[0], inputs.get_shape().as_list()[3]
			# batch_size  = inputs.get_shape().as_list()[0]
			# hidden_num = out_dim
			# print('=====================================')
			# print(inputs.shape)
			# print(hidden_num)

			maxpool_channel = tf.reduce_max(tf.reduce_max(inputs, axis=1, keep_dims=True), axis=2, keep_dims=True)
			avgpool_channel = tf.reduce_mean(tf.reduce_mean(inputs, axis=1, keep_dims=True), axis=2, keep_dims=True)

			# print('----------------------------------')
			# print(maxpool_channel.shape)
			# print(avgpool_channel.shape)

			# 上面全局池化结果为batsize * 1 * 1 * channel，它这个拉平输入到全连接层
			# 这个拉平，它会保留batsize，所以结果是[batsize,channel]
			maxpool_channel = tf.layers.Flatten()(maxpool_channel)
			avgpool_channel = tf.layers.Flatten()(avgpool_channel)

			mlp_1_max = tf.layers.dense(inputs=maxpool_channel, units=int(hidden_num * reduction_ratio), name="mlp_1",
			                            reuse=None, activation=tf.nn.elu)  ####relu
			mlp_2_max = tf.layers.dense(inputs=mlp_1_max, units=hidden_num, name="mlp_2", reuse=None)
			mlp_2_max = tf.reshape(mlp_2_max, [batch_size, 1, 1, hidden_num])

			mlp_1_avg = tf.layers.dense(inputs=avgpool_channel, units=int(hidden_num * reduction_ratio), name="mlp_1",
			                            reuse=True, activation=tf.nn.elu)
			mlp_2_avg = tf.layers.dense(inputs=mlp_1_avg, units=hidden_num, name="mlp_2", reuse=True)
			mlp_2_avg = tf.reshape(mlp_2_avg, [batch_size, 1, 1, hidden_num])

			channel_attention = tf.nn.sigmoid(mlp_2_max + mlp_2_avg)
			channel_refined_feature = inputs * channel_attention

			maxpool_spatial = tf.reduce_max(inputs, axis=3, keep_dims=True)
			avgpool_spatial = tf.reduce_mean(inputs, axis=3, keep_dims=True)
			max_avg_pool_spatial = tf.concat([maxpool_spatial, avgpool_spatial], axis=3)
			conv_layer = tf.layers.conv2d(inputs=max_avg_pool_spatial, filters=1, kernel_size=(7, 7), padding="same",
			                              activation=None)
			spatial_attention = tf.nn.sigmoid(conv_layer)

			refined_feature = channel_refined_feature * spatial_attention
		# print(refined_feature.shape)

		return refined_feature

	##########################################################################
	def resconv(self, x, num_layers, stride, attention=False):
		do_proj = tf.shape(x)[3] != num_layers or stride == 2
		shortcut = []
		# print(num_layers)
		conv1 = self.conv(x, num_layers, 1, 1)
		# if num_layers == 128:
		#     print(conv1)
		if stride == 2 and num_layers != 64:
			rate_dia = np.int(num_layers / 64)
			conv2 = self.conv_dia(conv1, num_layers, 3, stride, rate_dia)

		else:
			conv2 = self.conv(conv1, num_layers, 3, stride)
		conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
		if attention == True:  ##CBAM-Net layer
			# reduction_ratio = 16 ########
			# print(num_layers)
			if num_layers == 64:
				conv3 = self.cbam_module(conv3, out_dim=num_layers, reduction_ratio=0.5, name='64')
			if num_layers == 128:
				conv3 = self.cbam_module(conv3, out_dim=num_layers, reduction_ratio=0.5, name='128')
			if num_layers == 256:
				conv3 = self.cbam_module(conv3, out_dim=num_layers, reduction_ratio=0.5, name='256')
			if num_layers == 512:
				conv3 = self.cbam_module(conv3, out_dim=num_layers, reduction_ratio=0.5, name='512')

		if do_proj:
			if stride == 2 and num_layers != 64:
				rate_dia = np.int(num_layers / 64)

				shortcut = self.conv_dia(x, 4 * num_layers, 1, stride, rate_dia)
			# print('====ok')
			# print(shortcut)
			# print(conv2)
			else:
				shortcut = self.conv(x, 4 * num_layers, 1, stride, None)  ###### print('=====ok')
		else:
			shortcut = x
		return tf.nn.elu(conv3 + shortcut)

	def resblock(self, x, num_layers, num_blocks, attention=False):
		out = x
		for i in range(num_blocks - 1):
			out = self.resconv(out, num_layers, 1, attention)
		# print('++++++++++++++++++++++++++++++++++')
		out = self.resconv(out, num_layers, 2, attention)
		return out

	def upconv(self, x, num_out_layers, kernel_size, scale):
		upsample = self.upsample_nn(x, scale)
		conv = self.conv(upsample, num_out_layers, kernel_size, 1)
		return conv

	def deconv(self, x, num_out_layers, kernel_size, scale):
		p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
		conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
		return conv[:, 3:-1, 3:-1, :]

	def build_vgg(self):
		# set convenience functions
		conv = self.conv
		if self.params.use_deconv:
			upconv = self.deconv
		else:
			upconv = self.upconv

		with tf.variable_scope('encoder'):
			conv1 = self.conv_block(self.model_input, 32, 7)  # H/2
			conv2 = self.conv_block(conv1, 64, 5)  # H/4
			conv3 = self.conv_block(conv2, 128, 3)  # H/8
			conv4 = self.conv_block(conv3, 256, 3)  # H/16
			conv5 = self.conv_block(conv4, 512, 3)  # H/32
			conv6 = self.conv_block(conv5, 512, 3)  # H/64
			conv7 = self.conv_block(conv6, 512, 3)  # H/128

		with tf.variable_scope('skips'):
			skip1 = conv1
			skip2 = conv2
			skip3 = conv3
			skip4 = conv4
			skip5 = conv5
			skip6 = conv6

		with tf.variable_scope('decoder'):
			upconv7 = upconv(conv7, 512, 3, 2)  # H/64
			concat7 = tf.concat([upconv7, skip6], 3)
			iconv7 = conv(concat7, 512, 3, 1)

			upconv6 = upconv(iconv7, 512, 3, 2)  # H/32
			concat6 = tf.concat([upconv6, skip5], 3)
			iconv6 = conv(concat6, 512, 3, 1)

			upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
			concat5 = tf.concat([upconv5, skip4], 3)
			iconv5 = conv(concat5, 256, 3, 1)

			upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
			concat4 = tf.concat([upconv4, skip3], 3)
			iconv4 = conv(concat4, 128, 3, 1)
			self.disp4 = self.get_disp(iconv4)
			udisp4 = self.upsample_nn(self.disp4, 2)

			upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
			concat3 = tf.concat([upconv3, skip2, udisp4], 3)
			iconv3 = conv(concat3, 64, 3, 1)
			self.disp3 = self.get_disp(iconv3)
			udisp3 = self.upsample_nn(self.disp3, 2)

			upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
			concat2 = tf.concat([upconv2, skip1, udisp3], 3)
			iconv2 = conv(concat2, 32, 3, 1)
			self.disp2 = self.get_disp(iconv2)
			udisp2 = self.upsample_nn(self.disp2, 2)

			upconv1 = upconv(iconv2, 16, 3, 2)  # H
			concat1 = tf.concat([upconv1, udisp2], 3)
			iconv1 = conv(concat1, 16, 3, 1)
			self.disp1 = self.get_disp(iconv1)

	def build_resnet50(self):
		# set convenience functions
		conv = self.conv
		if self.params.use_deconv:
			upconv = self.deconv
		else:
			upconv = self.upconv

		with tf.variable_scope('encoder'):
			conv1 = conv(self.model_input, 64, 7, 2)  # H/2  -   64D
			pool1 = self.maxpool(conv1, 3)  # H/4  -   64D
			conv2 = self.resblock(pool1, 64, 3, True)  # H/8  -  256D
			conv3 = self.resblock(conv2, 128, 4, True)  # H/16 -  512D ##True
			conv4 = self.resblock(conv3, 256, 6, True)  # H/32 - 1024D
			conv5 = self.resblock(conv4, 512, 3, True)  # H/64 - 2048D

		with tf.variable_scope('skips'):
			skip1 = conv1
			skip2 = pool1
			skip3 = conv2
			skip4 = conv3
			skip5 = conv4

		# DECODING
		with tf.variable_scope('decoder'):

			# upconv5 = upconv(iconv6, 256, 3, 2) #H/1
			# print("ok")
			# print(conv5)
			# print(conv4)
			# print(conv3)
			# print(conv2)
			# print(skip2)
			# concat5 = tf.concat([conv5, skip5, skip4], 3)
			# iconv5  = conv(concat5,   256, 3, 1)
			#
			# upconv4 = upconv(iconv5,  128, 3, 2) #H/8
			concat4 = tf.concat([conv5, skip5, skip4, skip3], 3)
			iconv4 = conv(concat4, 128, 3, 1)
			self.disp4 = self.get_disp(iconv4)
			udisp4 = self.upsample_nn(self.disp4, 2)
			############
			self.disp4_up = self.upsample_nn(self.disp4, 8)
			# upconv4_4 = upconv(iconv4, 128, 3, 8)
			upconv4_4 = self.upsample_nn(iconv4, 8)

			upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
			concat3 = tf.concat([upconv3, skip2, udisp4], 3)
			iconv3 = conv(concat3, 64, 3, 1)
			self.disp3 = self.get_disp(iconv3)
			udisp3 = self.upsample_nn(self.disp3, 2)
			#############
			self.disp3_up = self.upsample_nn(self.disp3, 4)
			# upconv3_3 = upconv(iconv3, 64, 3, 4)
			upconv3_3 = self.upsample_nn(iconv3, 4)

			upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
			concat2 = tf.concat([upconv2, skip1, udisp3], 3)
			iconv2 = conv(concat2, 32, 3, 1)
			self.disp2 = self.get_disp(iconv2)
			udisp2 = self.upsample_nn(self.disp2, 2)
			###########
			self.disp2_up = self.upsample_nn(self.disp2, 2)
			# upconv2_2 = upconv(iconv2, 32, 3, 2)
			upconv2_2 = self.upsample_nn(iconv2, 2)

			# upconv1 = upconv(iconv2,  16, 3, 2) #H
			# concat1 = tf.concat([upconv1, udisp2], 3)
			# iconv1  = conv(concat1,   16, 3, 1)
			# self.disp1 = self.get_disp(iconv1)
			print('===================ok')
			# print(upconv2_2)
			# print(upconv3_3)
			# print(upconv4_4)
			upconv2_3_4 = conv(tf.concat([upconv4_4, upconv3_3, upconv2_2], 3), 16, 3, 1)
			# upconv2_3_4 = conv(tf.concat([ upconv3_3, upconv3_3], 3), 16, 3, 1)
			# upconv2_3_4 = upconv2_2

			upconv1 = upconv(iconv2, 16, 3, 2)  # H
			concat1 = tf.concat([upconv1, udisp2, upconv2_3_4], 3)
			# concat1 = tf.concat([upconv1, udisp2], 3)
			# iconv1  = conv(concat1,   16, 3, 1)
			iconv1 = conv(concat1, 32, 3, 1)
			iconv1 = conv(iconv1, 32, 3, 1)
			iconv1 = conv(iconv1, 16, 3, 1)
			self.disp1 = self.get_disp(iconv1)

	def build_model(self):
		with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
			with tf.variable_scope('model', reuse=self.reuse_variables):

				self.left_pyramid = self.scale_pyramid(self.left, 4)
				if self.mode == 'train':
					self.right_pyramid = self.scale_pyramid(self.right, 4)

				if self.params.do_stereo:
					self.model_input = tf.concat([self.left, self.right], 3)
				else:
					self.model_input = self.left

				# build model
				if self.params.encoder == 'vgg':
					self.build_vgg()
				elif self.params.encoder == 'resnet50':
					self.build_resnet50()
				else:
					return None

	def build_outputs(self):
		# STORE DISPARITIES
		with tf.variable_scope('disparities'):
			self.disp_est = [self.disp1, self.disp2, self.disp3, self.disp4]
			self.disp_left_est = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.disp_est]
			self.disp_right_est = [tf.expand_dims(d[:, :, :, 1], 3) for d in self.disp_est]

		# self.left_est  = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i])  for i in range(4)]
		# self.l1_left = [tf.abs( self.left_est[i] - self.left_pyramid[i]) for i in range(4)]

		if self.mode == 'test':
			return

		# GENERATE IMAGES
		with tf.variable_scope('images'):
			self.left_est = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i]) for i in range(4)]
			self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]

		# LR CONSISTENCY
		with tf.variable_scope('left-right'):
			self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i]) for i in
			                           range(4)]
			self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in
			                           range(4)]

		# DISPARITY SMOOTHNESS
		with tf.variable_scope('smoothness'):
			self.disp_left_smoothness = self.get_disparity_smoothness(self.disp_left_est, self.left_pyramid)
			self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_pyramid)

	def build_losses(self):
		with tf.variable_scope('losses', reuse=self.reuse_variables):
			# IMAGE RECONSTRUCTION
			# L1
			self.l1_left = [tf.abs(self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
			self.l1_reconstruction_loss_left = [tf.reduce_mean(l) for l in self.l1_left]
			self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
			self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

			# SSIM
			self.ssim_left = [self.SSIM(self.left_est[i], self.left_pyramid[i]) for i in range(4)]
			self.ssim_loss_left = [tf.reduce_mean(s) for s in self.ssim_left]
			self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
			self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

			# WEIGTHED SUM
			self.image_loss_right = [
				self.params.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.params.alpha_image_loss) *
				self.l1_reconstruction_loss_right[i] for i in range(4)]
			self.image_loss_left = [
				self.params.alpha_image_loss * self.ssim_loss_left[i] + (1 - self.params.alpha_image_loss) *
				self.l1_reconstruction_loss_left[i] for i in range(4)]
			self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

			# DISPARITY SMOOTHNESS
			self.disp_left_loss = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i])) / 2 ** i for i in range(4)]
			self.disp_right_loss = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
			self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)

			# LR CONSISTENCY
			self.lr_left_loss = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.disp_left_est[i])) for i in
			                     range(4)]
			self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in
			                      range(4)]
			self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)

			# TOTAL LOSS
			self.total_loss = self.image_loss + self.params.disp_gradient_loss_weight * self.disp_gradient_loss + self.params.lr_loss_weight * self.lr_loss

	def build_summaries(self):
		# SUMMARIES
		with tf.device('/cpu:0'):
			for i in range(4):
				tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_right[i],
				                  collections=self.model_collection)
				tf.summary.scalar('l1_loss_' + str(i),
				                  self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i],
				                  collections=self.model_collection)
				tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i],
				                  collections=self.model_collection)
				tf.summary.scalar('disp_gradient_loss_' + str(i), self.disp_left_loss[i] + self.disp_right_loss[i],
				                  collections=self.model_collection)
				tf.summary.scalar('lr_loss_' + str(i), self.lr_left_loss[i] + self.lr_right_loss[i],
				                  collections=self.model_collection)
				tf.summary.image('disp_left_est_' + str(i), self.disp_left_est[i], max_outputs=4,
				                 collections=self.model_collection)
				tf.summary.image('disp_right_est_' + str(i), self.disp_right_est[i], max_outputs=4,
				                 collections=self.model_collection)

				if self.params.full_summary:
					tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4,
					                 collections=self.model_collection)
					tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4,
					                 collections=self.model_collection)
					tf.summary.image('ssim_left_' + str(i), self.ssim_left[i], max_outputs=4,
					                 collections=self.model_collection)
					tf.summary.image('ssim_right_' + str(i), self.ssim_right[i], max_outputs=4,
					                 collections=self.model_collection)
					tf.summary.image('l1_left_' + str(i), self.l1_left[i], max_outputs=4,
					                 collections=self.model_collection)
					tf.summary.image('l1_right_' + str(i), self.l1_right[i], max_outputs=4,
					                 collections=self.model_collection)

			if self.params.full_summary:
				tf.summary.image('left', self.left, max_outputs=4, collections=self.model_collection)
				tf.summary.image('right', self.right, max_outputs=4, collections=self.model_collection)
