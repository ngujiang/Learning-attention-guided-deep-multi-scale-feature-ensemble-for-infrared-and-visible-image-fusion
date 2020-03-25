# Use a trained DenseFuse Net to generate fused images

import tensorflow as tf
import numpy as np
import attention
from datetime import datetime

from fusion_l1norm import L1_norm, L1_norm_attention
from densefuse_net import DenseFuseNet
from utils import get_images, save_images, get_train_images, get_train_images_rgb
from fusion_addition import Strategy
import cv2
import attention
TRAINING_IMAGE_SHAPE = (256, 256, 1) # (height, width, color_channels)
TRAINING_IMAGE_SHAPE_OR = (256, 256, 1) # (height, width, color_channels)

def guideFilter(I, p, winSize, eps):

    mean_I = cv2.blur(I, winSize)      # I的均值平滑
    mean_p = cv2.blur(p, winSize)      # p的均值平滑

    mean_II = cv2.blur(I * I, winSize) # I*I的均值平滑
    mean_Ip = cv2.blur(I * p, winSize) # I*p的均值平滑

    var_I = mean_II - mean_I * mean_I  # 方差
    cov_Ip = mean_Ip - mean_I * mean_p # 协方差

    a = cov_Ip / (var_I + eps + 0.0000001)         # 相关因子a
    b = mean_p - a * mean_I            # 相关因子b

    mean_a = cv2.blur(a, winSize)      # 对a进行均值平滑
    mean_b = cv2.blur(b, winSize)      # 对b进行均值平滑

    q = mean_a * I + mean_b

    return q

def RollingGuidance(I, sigma_s, sigma_r, iteration):
	sigma_s = (sigma_s, sigma_s)
	out = cv2.GaussianBlur(I, sigma_s, 0)
	for i in range(iteration):
		out = guideFilter(out, I, sigma_s, sigma_r*sigma_r)

	return out

def Grad(I1):
    G1=[]
    L1=[]
    G1.append(I1)
    sigma_s = 3
    sigma_r = [0.5, 0.5, 0.5, 0.5]
    iteration = [4, 4, 4, 4]
    indice=(1,2,3)
    for i in indice:
        G1.append(RollingGuidance(G1[i-1],sigma_s,sigma_r[i-1],iteration[i-1]))
        L1.append(G1[i-1]-G1[i])
        sigma_s = 3 * sigma_s
    sigma_s = (3, 3)
    G1.append(cv2.GaussianBlur(G1[3], sigma_s, 0))
    L1.append(G1[3]-G1[4])
    L1.append(G1[4])
    grad = L1[0]
    return grad

def generate(infrared_path, visible_path, model_path, model_pre_path,model_path_a,model_pre_path_a ,ssim_weight, index, IS_VIDEO, IS_RGB, type='addition', output_path=None):

	if IS_VIDEO:
		print('video_addition')
		_handler_video(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, output_path=output_path)
	else:
		if IS_RGB:
			print('RGB - addition')
			_handler_rgb(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index,
			         output_path=output_path)

			print('RGB - l1')
			_handler_rgb_l1(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index,
			             output_path=output_path)
		else:
			_handler_mix_a(infrared_path, visible_path, model_path, model_pre_path,model_path_a,model_pre_path_a, ssim_weight, index,
					 output_path=output_path)
			#if type == 'addition':
			#print('addition')
			#_handler(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index, output_path=output_path)
			#elif type == 'l1':
			#print('l1')
			#_handler_l1(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index, output_path=output_path)


def _handler(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	ir_img = get_train_images(ir_path, flag=False)
	vis_img = get_train_images(vis_path, flag=False)
	# ir_img = get_train_images_rgb(ir_path, flag=False)
	# vis_img = get_train_images_rgb(vis_path, flag=False)
	dimension = ir_img.shape

	ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])

	ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	vis_img = np.transpose(vis_img, (0, 2, 1, 3))

	print('img shape final:', ir_img.shape)

	with tf.Graph().as_default(), tf.Session() as sess:
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		output_image = dfn.transform_addition(infrared_field, visible_field)
		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		output = sess.run(output_image, feed_dict={infrared_field: ir_img, visible_field: vis_img})

		save_images(ir_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_addition_'+str(ssim_weight))


def _handler_l1(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	ir_img = get_train_images(ir_path, flag=False)
	vis_img = get_train_images(vis_path, flag=False)
	dimension = ir_img.shape

	ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])

	ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	vis_img = np.transpose(vis_img, (0, 2, 1, 3))

	print('img shape final:', ir_img.shape)

	with tf.Graph().as_default(), tf.Session() as sess:

		# build the dataflow graph
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		enc_ir,enc_ir_res_block,enc_ir_res_block1,enc_ir_res_block2 = dfn.transform_encoder(infrared_field)
		enc_vis,enc_vis_res_block,enc_vis_res_block1,enc_vis_res_block2 = dfn.transform_encoder(visible_field)

		target = tf.placeholder(
		    tf.float32, shape=enc_ir.shape, name='target')
		block1 = tf.placeholder(
			tf.float32, shape=enc_ir_res_block1.shape, name='block1')
		block2= tf.placeholder(
			tf.float32, shape=enc_ir_res_block2.shape, name='block2')
		output_image = dfn.transform_decoder(target,block1,block2)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		enc_ir_temp, enc_vis_temp,ir_block1,ir_block2,vis_block1,vis_block2 = sess.run([enc_ir, enc_vis,enc_ir_res_block1,enc_ir_res_block2 ,enc_vis_res_block1,enc_vis_res_block2], feed_dict={infrared_field: ir_img, visible_field: vis_img})
		feature = L1_norm(enc_ir_temp, enc_vis_temp)
		t_block1=L1_norm(ir_block1,vis_block1)

		t_block2 = L1_norm(ir_block2, vis_block2)
		output = sess.run(output_image, feed_dict={target: feature ,block1:t_block1,block2:t_block2})
		save_images(ir_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_l1norm_'+str(ssim_weight))


def _handler_video(ir_path, vis_path, model_path, model_pre_path, ssim_weight, output_path=None):
	infrared = ir_path[0]
	img = get_train_images(infrared, flag=False)
	img = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])
	img = np.transpose(img, (0, 2, 1, 3))
	print('img shape final:', img.shape)
	num_imgs = len(ir_path)

	with tf.Graph().as_default(), tf.Session() as sess:
		# build the dataflow graph
		infrared_field = tf.placeholder(
			tf.float32, shape=img.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=img.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		output_image = dfn.transform_addition(infrared_field, visible_field)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		##################GET IMAGES###################################################################################
		start_time = datetime.now()
		for i in range(num_imgs):
			print('image number:', i)
			infrared = ir_path[i]
			visible = vis_path[i]

			ir_img = get_train_images(infrared, flag=False)
			vis_img = get_train_images(visible, flag=False)
			dimension = ir_img.shape

			ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
			vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])

			ir_img = np.transpose(ir_img, (0, 2, 1, 3))
			vis_img = np.transpose(vis_img, (0, 2, 1, 3))

			################FEED########################################
			output = sess.run(output_image, feed_dict={infrared_field: ir_img, visible_field: vis_img})
			save_images(infrared, output, output_path,
			            prefix='fused' + str(i), suffix='_addition_' + str(ssim_weight))
			######################################################################################################
		elapsed_time = datetime.now() - start_time
		print('Dense block video==> elapsed time: %s' % (elapsed_time))


def _handler_rgb(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	# ir_img = get_train_images(ir_path, flag=False)
	# vis_img = get_train_images(vis_path, flag=False)
	ir_img = get_train_images_rgb(ir_path, flag=False)
	vis_img = get_train_images_rgb(vis_path, flag=False)
	dimension = ir_img.shape

	ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])

	#ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	#vis_img = np.transpose(vis_img, (0, 2, 1, 3))

	ir_img1 = ir_img[:, :, :, 0]
	ir_img1 = ir_img1.reshape([1, dimension[0], dimension[1], 1])
	ir_img2 = ir_img[:, :, :, 1]
	ir_img2 = ir_img2.reshape([1, dimension[0], dimension[1], 1])
	ir_img3 = ir_img[:, :, :, 2]
	ir_img3 = ir_img3.reshape([1, dimension[0], dimension[1], 1])

	vis_img1 = vis_img[:, :, :, 0]
	vis_img1 = vis_img1.reshape([1, dimension[0], dimension[1], 1])
	vis_img2 = vis_img[:, :, :, 1]
	vis_img2 = vis_img2.reshape([1, dimension[0], dimension[1], 1])
	vis_img3 = vis_img[:, :, :, 2]
	vis_img3 = vis_img3.reshape([1, dimension[0], dimension[1], 1])

	print('img shape final:', ir_img1.shape)

	with tf.Graph().as_default(), tf.Session() as sess:
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_img1.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_img1.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		output_image = dfn.transform_addition(infrared_field, visible_field)
		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		output1 = sess.run(output_image, feed_dict={infrared_field: ir_img1, visible_field: vis_img1})
		output2 = sess.run(output_image, feed_dict={infrared_field: ir_img2, visible_field: vis_img2})
		output3 = sess.run(output_image, feed_dict={infrared_field: ir_img3, visible_field: vis_img3})

		output1 = output1.reshape([1, dimension[0], dimension[1]])
		output2 = output2.reshape([1, dimension[0], dimension[1]])
		output3 = output3.reshape([1, dimension[0], dimension[1]])

		output = np.stack((output1, output2, output3), axis=-1)
		#output = np.transpose(output, (0, 2, 1, 3))
		save_images(ir_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_addition_'+str(ssim_weight))


def _handler_rgb_l1(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	# ir_img = get_train_images(ir_path, flag=False)
	# vis_img = get_train_images(vis_path, flag=False)
	ir_img = get_train_images_rgb(ir_path, flag=False)
	vis_img = get_train_images_rgb(vis_path, flag=False)
	dimension = ir_img.shape

	ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])

	#ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	#vis_img = np.transpose(vis_img, (0, 2, 1, 3))

	ir_img1 = ir_img[:, :, :, 0]
	ir_img1 = ir_img1.reshape([1, dimension[0], dimension[1], 1])
	ir_img2 = ir_img[:, :, :, 1]
	ir_img2 = ir_img2.reshape([1, dimension[0], dimension[1], 1])
	ir_img3 = ir_img[:, :, :, 2]
	ir_img3 = ir_img3.reshape([1, dimension[0], dimension[1], 1])

	vis_img1 = vis_img[:, :, :, 0]
	vis_img1 = vis_img1.reshape([1, dimension[0], dimension[1], 1])
	vis_img2 = vis_img[:, :, :, 1]
	vis_img2 = vis_img2.reshape([1, dimension[0], dimension[1], 1])
	vis_img3 = vis_img[:, :, :, 2]
	vis_img3 = vis_img3.reshape([1, dimension[0], dimension[1], 1])

	print('img shape final:', ir_img1.shape)

	with tf.Graph().as_default(), tf.Session() as sess:
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_img1.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_img1.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		enc_ir,enc_ir_res_block = dfn.transform_encoder(infrared_field)
		enc_vis,enc_vis_res_block = dfn.transform_encoder(visible_field)

		target = tf.placeholder(
			tf.float32, shape=enc_ir.shape, name='target')

		output_image = dfn.transform_decoder(target)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		enc_ir_temp, enc_vis_temp = sess.run([enc_ir, enc_vis], feed_dict={infrared_field: ir_img1, visible_field: vis_img1})
		feature = L1_norm(enc_ir_temp, enc_vis_temp)
		output1 = sess.run(output_image, feed_dict={target: feature})

		enc_ir_temp, enc_vis_temp = sess.run([enc_ir, enc_vis], feed_dict={infrared_field: ir_img2, visible_field: vis_img2})
		feature = L1_norm(enc_ir_temp, enc_vis_temp)
		output2 = sess.run(output_image, feed_dict={target: feature})

		enc_ir_temp, enc_vis_temp = sess.run([enc_ir, enc_vis], feed_dict={infrared_field: ir_img3, visible_field: vis_img3})
		feature = L1_norm(enc_ir_temp, enc_vis_temp)
		output3 = sess.run(output_image, feed_dict={target: feature})

		output1 = output1.reshape([1, dimension[0], dimension[1]])
		output2 = output2.reshape([1, dimension[0], dimension[1]])
		output3 = output3.reshape([1, dimension[0], dimension[1]])

		output = np.stack((output1, output2, output3), axis=-1)
		#output = np.transpose(output, (0, 2, 1, 3))
		save_images(ir_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_l1norm_'+str(ssim_weight))
def _handler_mix(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	mix_block=[]
	ir_img = get_train_images(ir_path, flag=False)
	vis_img = get_train_images(vis_path, flag=False)
	dimension = ir_img.shape
	ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	vis_img = np.transpose(vis_img, (0, 2, 1, 3))

	print('img shape final:', ir_img.shape)
	with tf.Graph().as_default(), tf.Session() as sess:
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=vis_img.shape, name='style')

		# -----------------------------------------------

		dfn = DenseFuseNet(model_pre_path)

		#sess.run(tf.global_variables_initializer())

		enc_ir,enc_ir_res_block ,enc_ir_block,enc_ir_block2= dfn.transform_encoder(infrared_field)
		enc_vis,enc_vis_res_block,enc_vis_block,enc_vis_block2 = dfn.transform_encoder(visible_field)

		result = tf.placeholder(
		    tf.float32, shape=enc_ir.shape, name='target')



		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		enc_ir_temp, enc_ir_res_block_temp, enc_ir_block_temp, enc_ir_block2_temp = sess.run(
			[enc_ir, enc_ir_res_block, enc_ir_block, enc_ir_block2], feed_dict={infrared_field: ir_img})
		enc_vis_temp, enc_vis_res_block_temp, enc_vis_block_temp, enc_vis_block2_temp = sess.run(
			[enc_vis, enc_vis_res_block, enc_vis_block, enc_vis_block2], feed_dict={visible_field: vis_img})

		block = L1_norm(enc_ir_block_temp, enc_vis_block_temp)
		block2=L1_norm(enc_ir_block2_temp,enc_vis_block2_temp)

		first_first = L1_norm(enc_ir_res_block_temp[0], enc_vis_res_block_temp[0])
		first_second = Strategy(enc_ir_res_block_temp[1], enc_vis_res_block_temp[1])
		#first_third = L1_norm_attention(enc_ir_res_block_temp[2],feation_ir, enc_vis_res_block_temp[2],feation_vis)
		#first_four = L1_norm_attention(enc_ir_res_block_temp[3],feation_ir, enc_vis_res_block_temp[3],feation_vis)
		first_third=L1_norm(enc_ir_res_block_temp[2],enc_vis_res_block_temp[2])
		first_four=Strategy(enc_ir_res_block_temp[3],enc_vis_res_block_temp[3])
		first_first = tf.concat([first_first, tf.to_int32(first_second, name='ToInt')],3)
		first_first = tf.concat([first_first, tf.to_int32(first_third, name='ToInt')],3)
		first_first = tf.concat([first_first, first_four],3)

		first = first_first

		second = L1_norm(enc_ir_res_block_temp[6], enc_vis_res_block_temp[6])
		third = L1_norm(enc_ir_res_block_temp[9], enc_vis_res_block_temp[9])

		feature = 1 * first + 0.1 * second + 0.1 * third

		#---------------------------------------------------------
		# block=Strategy(enc_ir_block_temp,enc_vis_block_temp)
		# block2=L1_norm(enc_ir_block2_temp,enc_vis_block2_temp)
		#---------------------------------------------------------

		feature = feature.eval()

		output_image = dfn.transform_decoder(result, block, block2)

		# output = dfn.transform_decoder(feature)
		# print(type(feature))
		# output = sess.run(output_image, feed_dict={result: feature,enc_res_block:block,enc_res_block2:block2})
		output = sess.run(output_image, feed_dict={result: feature})

		save_images(ir_path, output, output_path,
					prefix='fused' + str(index), suffix='_mix_' + str(ssim_weight))
def _get_attention(ir_path,vis_path,model_path_a,model_pre_path_a):
	ir_img = get_train_images(ir_path, flag=False)
	vis_img = get_train_images(vis_path, flag=False)
	dimension = ir_img.shape
	ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	vis_img = np.transpose(vis_img, (0, 2, 1, 3))
	g1 = tf.Graph()  # 加载到Session 1的graph

	sess1 = tf.Session(graph=g1)  # Session1

	with sess1.as_default():
		with g1.as_default(), tf.Session() as sess:
			infrared_field = tf.placeholder(
				tf.float32, shape=ir_img.shape, name='content')
			visible_field = tf.placeholder(
				tf.float32, shape=vis_img.shape, name='style')
			edge_ir = tf.placeholder(tf.float32, shape=ir_img.shape, name='attention')
			edge_vis = tf.placeholder(tf.float32, shape=ir_img.shape, name='attention')

			# -----------------------------------------------
			image_ir = sess.run(infrared_field, feed_dict={infrared_field: ir_img})
			image_vis = sess.run(visible_field, feed_dict={visible_field: vis_img})

			p_vis = image_vis[0]
			p_ir = image_ir[0]

			p_vis = np.squeeze(p_vis)  # 降维
			p_ir = np.squeeze(p_ir)

			guideFilter_img_vis = Grad(p_vis)
			guideFilter_img_ir = Grad(p_ir)

			guideFilter_img_vis[guideFilter_img_vis < 0] = 0
			guideFilter_img_ir[guideFilter_img_ir < 0] = 0
			guideFilter_img_vis = np.expand_dims(guideFilter_img_vis, axis=-1)
			guideFilter_img_ir = np.expand_dims(guideFilter_img_ir, axis=-1)
			guideFilter_img_vis = np.expand_dims(guideFilter_img_vis, axis=0)
			guideFilter_img_ir = np.expand_dims(guideFilter_img_ir, axis=0)

			a = attention.Attention(model_pre_path_a)
			saver = tf.train.Saver()
			saver.restore(sess, model_path_a)

			feature_a=a.get_attention(edge_ir)
			feature_b=a.get_attention(edge_vis)


			edge_ir_temp = sess.run([feature_a], feed_dict={edge_ir: guideFilter_img_ir})
			edge_vis_temp = sess.run([feature_b], feed_dict={edge_vis: guideFilter_img_vis})
			'''feature_a = a.get_attention(edge_ir_temp)
			feature_b = a.get_attention(edge_vis_temp)'''

			return  edge_ir_temp,edge_vis_temp


def _handler_mix_a(ir_path, vis_path, model_path, model_pre_path,model_path_a,model_pre_path_a, ssim_weight, index, output_path=None):
	ir_img = get_train_images(ir_path, flag=False)
	vis_img = get_train_images(vis_path, flag=False)
	dimension = ir_img.shape
	ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	vis_img = np.transpose(vis_img, (0, 2, 1, 3))


	g2 = tf.Graph()  # 加载到Session 2的graph

	sess2 = tf.Session(graph=g2)  # Session2

	with sess2.as_default():  # 1
		with g2.as_default(),tf.Session() as sess:
			infrared_field = tf.placeholder(
				tf.float32, shape=ir_img.shape, name='content')
			visible_field = tf.placeholder(
				tf.float32, shape=vis_img.shape, name='style')

			dfn = DenseFuseNet(model_pre_path)

			# sess.run(tf.global_variables_initializer())

			enc_ir, enc_ir_res_block, enc_ir_block, enc_ir_block2 = dfn.transform_encoder(infrared_field)
			enc_vis, enc_vis_res_block, enc_vis_block, enc_vis_block2 = dfn.transform_encoder(visible_field)

			result = tf.placeholder(
				tf.float32, shape=enc_ir.shape, name='target')

			saver = tf.train.Saver()
			saver.restore(sess, model_path)
			print("______000________")
			feature_a,feature_b=_get_attention(ir_path,vis_path,model_path_a,model_pre_path_a)
			print("______111________")
			print(feature_a[0].shape)

			enc_ir_temp, enc_ir_res_block_temp, enc_ir_block_temp, enc_ir_block2_temp = sess.run(
				[enc_ir, enc_ir_res_block, enc_ir_block, enc_ir_block2], feed_dict={infrared_field: ir_img})
			print("______222________")
			enc_vis_temp, enc_vis_res_block_temp, enc_vis_block_temp, enc_vis_block2_temp = sess.run(
				[enc_vis, enc_vis_res_block, enc_vis_block, enc_vis_block2], feed_dict={visible_field: vis_img})
			print("______333________")
			# ----------------------------------------------------------------------------------------------------------
			# ----------------------------------------------------------------------------------------------------------


			#----------------------------------跳跃部分-----------------------------------------------------------------
			block = Strategy(enc_ir_block_temp, enc_vis_block_temp) * 0
			block2 = Strategy(enc_ir_block2_temp, enc_vis_block2_temp) * 0
			#block = L1_norm_attention(enc_ir_block_temp, feature_a, enc_vis_block_temp, feature_b)
			#block2 = L1_norm_attention(enc_ir_block2_temp, feature_a, enc_vis_block2_temp, feature_b)
			# ----------------------------------------------------------------------------------------------------------

			first_first = Strategy(enc_ir_res_block_temp[0], enc_vis_res_block_temp[0])
			#first_first = L1_norm_attention(enc_ir_res_block_temp[0],feature_a, enc_vis_res_block_temp[0],feature_b)
			first_second = Strategy(enc_ir_res_block_temp[1], enc_vis_res_block_temp[1])
			#first_second = L1_norm_attention(enc_ir_res_block_temp[1],feature_a, enc_vis_res_block_temp[1],feature_b)
			first_third = Strategy(enc_ir_res_block_temp[2], enc_vis_res_block_temp[2])
			#first_third = L1_norm_attention(enc_ir_res_block_temp[2], feature_a, enc_vis_res_block_temp[2], feature_b)
			#first_third = L1_norm(enc_ir_res_block_temp[2], enc_vis_res_block_temp[2]) * 0
			first_four = Strategy(enc_ir_res_block_temp[3], enc_vis_res_block_temp[3])
			#first_four = L1_norm_attention(enc_ir_res_block_temp[3], feature_a, enc_vis_res_block_temp[3], feature_b)
			#first_four = L1_norm(enc_ir_res_block_temp[3],  enc_vis_res_block_temp[3])
			first_first = tf.concat([first_first, tf.to_int32(first_second, name='ToInt')], 3)
			first_first = tf.concat([first_first, tf.to_int32(first_third, name='ToInt')], 3)
			first_first = tf.concat([first_first, first_four], 3)
			print("______444________")

			first = first_first

			# -------------------------------------空洞卷积部分---------------------------------------------------------
			#second = L1_norm_attention(enc_ir_res_block_temp[6],feature_a, enc_vis_res_block_temp[6],feature_b)
			print (enc_ir_res_block_temp[6].shape)
			second = L1_norm(enc_ir_res_block_temp[6], enc_vis_res_block_temp[6])
			print("______4545________")
			third = Strategy(enc_ir_res_block_temp[9], enc_vis_res_block_temp[9])
			print("______555________")
			# ----------------------------------------------------------------------------------------------------------


			# ----------------------------------------------------------------------------------------------------------
			# ----------------------------------------------------------------------------------------------------------



			# -------------------------------------空洞卷积部分---------------------------------------------------------
			feature = 1 * first + 0.1 * second + 0.1 * third
			# ----------------------------------------------------------------------------------------------------------
			print ("51515151")

			# ---------------------------------------------------------
			# block=Strategy(enc_ir_block_temp,enc_vis_block_temp)
			# block2=L1_norm(enc_ir_block2_temp,enc_vis_block2_temp)
			# ---------------------------------------------------------

			feature = feature.eval()


			print ("52525252")

			# --------------将特征图压成单通道----------------------------------
			#feature_map_vis_out = sess.run(tf.reduce_sum(feature_a[0], 3, keep_dims=True))
			#feature_map_ir_out = sess.run(tf.reduce_sum(feature_b[0],3, keep_dims=True))
			# ------------------------------------------------------------------
			print (result.shape)
			print ("5555555")
			output_image = dfn.transform_decoder(result, block, block2)
			print("______666________")
			# output = dfn.transform_decoder(feature)
			# print(type(feature))
			# output = sess.run(output_image, feed_dict={result: feature,enc_res_block:block,enc_res_block2:block2})

			output = sess.run(output_image, feed_dict={result: feature})
			print (output_image.shape)
			print("______777________")
			save_images(ir_path, output, output_path,
			            prefix='' + str(index), suffix='-1')
						#prefix = '' + str(index), suffix = '-4' + str(ssim_weight))
			#save_images(ir_path, feature_map_vis_out, output_path,
			#            prefix='fused' + str(index), suffix='vis' + str(ssim_weight))
			#save_images(ir_path, feature_map_ir_out, output_path,
			#            prefix='fused' + str(index), suffix='ir' + str(ssim_weight))



