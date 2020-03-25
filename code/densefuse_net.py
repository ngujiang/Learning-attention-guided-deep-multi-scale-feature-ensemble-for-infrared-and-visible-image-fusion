# DenseFuse Network
# Encoder -> Addition/L1-norm -> Decoder

import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from fusion_addition import Strategy

class DenseFuseNet(object):

    def __init__(self, model_pre_path):
        print("------------------------------------")
        print(model_pre_path)
        self.encoder = Encoder(model_pre_path)
        self.decoder = Decoder(model_pre_path)

    def transform_addition(self, img1, img2):
        # encode image
        enc_1, enc_1_res_block,enc_1_block,enc_1_block2 = self.encoder.encode(img1)
        enc_2, enc_2_res_block ,enc_2_block,enc_2_block2= self.encoder.encode(img2)
        target_features = Strategy(enc_1, enc_2)
        # target_features = enc_c
        self.target_features = target_features
        print('target_features:', target_features.shape)
        # decode target features back to image
        generated_img = self.decoder.decode(target_features,enc_1_block,enc_1_block2)
        return generated_img
#------------------------------------------------------------------------
    #不涉及融合层的图像encoder decoder
    def transform_recons(self, img):
        # encode image

        enc, enc_res_block ,block,block2= self.encoder.encode(img)

        target_features = enc
        self.target_features = target_features
        generated_img = self.decoder.decode(target_features,block,block2)
        return generated_img

#-----------------------------------------------------------------------------
    def transform_encoder(self, img):
        # encode image
        enc, enc_res_block,block,block2 = self.encoder.encode(img)
        return enc, enc_res_block,block,block2

    def transform_decoder(self, feature,block,block2):
        # decode image
        generated_img = self.decoder.decode(feature,block,block2)
        return generated_img
