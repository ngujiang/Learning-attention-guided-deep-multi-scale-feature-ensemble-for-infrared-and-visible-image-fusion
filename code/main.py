# Demo - train the DenseFuse network & use it to generate an image

from __future__ import print_function

import time

from train_recons import train_recons,train_recons_a
from generate import generate
from utils import list_images
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# True for training phase
IS_TRAINING = False
IS_TRAINING_A = False
# True for video sequences(frames)
IS_VIDEO = False
# True for RGB images
is_RGB = False

BATCH_SIZE = 32
EPOCHES = 4

SSIM_WEIGHTS = [1, 10, 100, 1000]
SSIM_WEIGHTS_A=[1, 10, 100]

#-------------------------------------------------------------------------------------------------
MODEL_SAVE_PATHS = [
	'/data/ljy/paper_again/19-11-20-final/model/densefuse_model_bs2_epoch4_all_weight_1e0.ckpt',
    '/data/ljy/paper_again/19-11-20-final/model/densefuse_model_bs2_epoch4_all_weight_1e1.ckpt',
    '/data/ljy/paper_again/19-11-20-final/model/densefuse_model_bs2_epoch4_all_weight_1e2.ckpt',
    '/data/ljy/paper_again/19-11-20-final/model/densefuse_model_bs2_epoch4_all_weight_1e3.ckpt',
]
MODEL_SAVE_PATHS_A = [
	'/data/ljy/paper_again/19-11-20-final/model_a/densefuse_model_bs2_epoch4_all_weight_1e0.ckpt',
	'/data/ljy/paper_again/19-11-20-final/model_a/densefuse_model_bs2_epoch4_all_weight_1e1.ckpt',
	'/data/ljy/paper_again/19-11-20-final/model_a/densefuse_model_bs2_epoch4_all_weight_1e2.ckpt',
]
#ckpt文件用于保存tensorflow的模型
#-----------------------------------------------------------------------------------------------
# MODEL_SAVE_PATH = './models/deepfuse_dense_model_bs4_epoch2_relu_pLoss_noconv_test.ckpt'
# model_pre_path  = './models/deepfuse_dense_model_bs2_epoch2_relu_pLoss_noconv_NEW.ckpt'

# In testing process, 'model_pre_path' is set to None
# The "model_pre_path" in "main.py" is just a pre-train model and not necessary for training and testing. 
# It is set as None when you want to train your own model. 
# If you already train a model, you can set it as your model for initialize weights.
model_pre_path = None
model_pre_path_a = None
def main():

	if IS_TRAINING:
#-------------------------------------------------------------------------------------------------------
		original_imgs_path = list_images('/data/ljy/train_mix/mix_256/')
		validatioin_imgs_path = list_images('/data/ljy/修改专用/imagefusion_densefuse-master/validation/validation/')
#---------------------------------------------------------------------------------------------------------
		for ssim_weight, model_save_path in zip(SSIM_WEIGHTS, MODEL_SAVE_PATHS):
			print('\nBegin to train the network ...\n')
			train_recons(original_imgs_path, validatioin_imgs_path, model_save_path, model_pre_path, ssim_weight, EPOCHES, BATCH_SIZE, debug=True)

			print('\nSuccessfully! Done training...\n')
	#====================================================================================================
	elif IS_TRAINING_A:
		original_imgs_path = list_images('/data/ljy/train_mix/mix_256/')
		validatioin_imgs_path = list_images('/data/ljy/修改专用/imagefusion_densefuse-master/validation/validation/')
		for ssim_weight_a, model_save_path_a in zip(SSIM_WEIGHTS_A, MODEL_SAVE_PATHS_A):
			print('\nBegin to train the attention network ...\n')
			train_recons_a(original_imgs_path, validatioin_imgs_path, model_save_path_a, model_pre_path_a, ssim_weight_a,EPOCHES, BATCH_SIZE,MODEL_SAVE_PATHS[0], debug=True)
			print('\nSuccessfully! Done training...\n')




	#================================================================================================
	else:
		if IS_VIDEO:
			ssim_weight = SSIM_WEIGHTS[0]
			model_path = MODEL_SAVE_PATHS[0]

			IR_path = list_images('video/1_IR/')
			VIS_path = list_images('video/1_VIS/')
			output_save_path = 'video/fused'+ str(ssim_weight) +'/'
			generate(IR_path, VIS_path, model_path, model_pre_path,
			         ssim_weight, 0, IS_VIDEO, 'addition', output_path=output_save_path)
		else:
			ssim_weight = SSIM_WEIGHTS[1]
			model_path = MODEL_SAVE_PATHS[1]
			model_path_a=MODEL_SAVE_PATHS_A[1]
			print('\nBegin to generate pictures ...\n')
			# path = 'images/IV_images/'
			path = '/data/ljy/IV_images/'
			for i in range(20):
				#if i != 1 :
				#	continue
				index = i + 1
				infrared = path + 'IR' + str(index) + '.png'
				visible = path + 'VIS' + str(index) + '.png'

				# RGB images
				#infrared = path + 'lytro-' + str(index) + '-A.jpg'
				#visible = path + 'lytro-' + str(index) + '-B.jpg'

				# choose fusion layer
				#fusion_type = 'addition'
				fusion_type = 'l1'
				# for ssim_weight, model_path in zip(SSIM_WEIGHTS, MODEL_SAVE_PATHS):
				# 	output_save_path = 'outputs'
                #
				# 	generate(infrared, visible, model_path, model_pre_path,
				# 	         ssim_weight, index, IS_VIDEO, is_RGB, type = fusion_type, output_path = output_save_path)

				output_save_path = '/data/ljy/paper_again/19-11-20-final/attention/'
				generate(infrared, visible, model_path, model_pre_path,model_path_a,model_pre_path_a,
						 ssim_weight, index, IS_VIDEO, is_RGB, type = fusion_type, output_path = output_save_path)


if __name__ == '__main__':
    main()

