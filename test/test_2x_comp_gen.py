'''
Created on 3May,2017

@author: yizuo
'''
from PIL import Image
import numpy as np
import tensorflow as tf
import guided_srgan_layer_blocks as gslb

up_factor=2

#read test HR inten img
color_art = Image.open("/media/kenny/Data/test_data/noise_free/middlebury_bicubic_LR_test_pairs/dolls/color_dolls_croped.png")#####for noise-free midd2006
#color_art = Image.open("/media/kenny/Data/test_data/noisy/table2/_gth/laundry_color.png")####mid 2006
#color_art = Image.open("/media/kenny/Data/test_data/noisy/iccv15_test_imgs/gth/venus/color_crop.png")####mid 2001
inten_art =color_art.convert("L")
np_inten=np.asarray(inten_art)
val_inten=np_inten.astype(np.float32)/255.

#read test HR dep img
gth_dep_art=Image.open("/media/kenny/Data/test_data/noise_free/middlebury_bicubic_LR_test_pairs/dolls/dolls_croped.png")#####for noise-free midd2006
#gth_dep_art=Image.open("/media/kenny/Data/test_data/noisy/table2/_gth/laundry_big.png")###mid 2006
#gth_dep_art=Image.open("/media/kenny/Data/test_data/noisy/iccv15_test_imgs/gth/venus/depth_crop.png")###mid 2001
np_gth_dep=np.asarray(gth_dep_art)
val_gth_dep=np_gth_dep.astype(np.float32)/255.

#read test LR dep img
LR_dep_art=Image.open("/media/kenny/Data/test_data/noise_free/middlebury_bicubic_LR_test_pairs/dolls/dolls2x_bicubic.png")#####for noise-free midd2006
#LR_dep_art=Image.open("/media/kenny/Data/test_data/noisy/table2/_input/laundry_big/depth_1_n.png")###mid 2006
#LR_dep_art=Image.open("/media/kenny/Data/test_data/noisy/iccv15_test_imgs/input/venus/2_x_dep.png")###mid 2001
np_LR_dep=np.asarray(LR_dep_art)
val_LR_dep=np_LR_dep.astype(np.float32)/255.

height=val_inten.shape[0]
width=val_inten.shape[1]
LR_height=height/up_factor
LR_width=width/up_factor
val_inten=val_inten.reshape((1,height,width,1))
val_gth_dep=val_gth_dep.reshape(1,height,width,1)
val_LR_dep=val_LR_dep.reshape(1,LR_height,LR_width,1)

#setting input size and training data addr
phase_train=tf.placeholder(tf.bool)
HR_patch_size=[height,width]
HR_batch_dims=(1,height,width,1)
LR_batch_dims=(1,LR_height,LR_width,1)

#setting input placeholders
HR_depth_batch_input=tf.placeholder(tf.float32,HR_batch_dims)
HR_inten_batch_input=tf.placeholder(tf.float32,HR_batch_dims)
LR_depth_batch_input=tf.placeholder(tf.float32,LR_batch_dims)
coar_inter_dep_batch=tf.image.resize_images(LR_depth_batch_input,tf.constant(HR_patch_size,dtype=tf.int32),tf.image.ResizeMethod.BICUBIC)

#gen_network construction
with tf.variable_scope("gen_inten"):
    guided_ten=gslb.inten_feature_extraction_unit(HR_inten_batch_input,phase_train=phase_train)
with tf.variable_scope("gen_dep"):
    dep_ten=gslb.LR_dep_feature_extraction_uint(LR_depth_batch_input,phase_train=phase_train)
    dep_ten=gslb.LR_dep_upsampling_unit(dep_ten, phase_train=phase_train)
with tf.variable_scope("gen_2x_last4_layers"):
    gen_ten=gslb.LR_dep_fusion_simple_unit(dep_ten, guided_ten,phase_train=phase_train)
    gen_ten=gslb.LR_recon_unit(gen_ten, coar_inter_dep_batch,phase_train=phase_train)

#define loss for gen
loss=tf.reduce_mean(tf.squared_difference(gen_ten,HR_depth_batch_input))
saver_full=tf.train.Saver()

#begin comp_gen testing
with tf.Session() as sess:
    model_path="/media/kenny/Data/trained_models/residual_dense_models/noise_free/2x/full_model2/2x_nf_full_model.ckpt-32"
    saver_full.restore(sess, model_path)
    ten_fets=sess.run([loss,gen_ten],feed_dict={HR_inten_batch_input:val_inten,HR_depth_batch_input:val_gth_dep,LR_depth_batch_input:val_LR_dep,phase_train:False})
    final_array=ten_fets[1]*255.0+0.5
    final_array[final_array>255]=255.0
    final_array[final_array<0]=0.0
    final_array=final_array.astype(np.uint8).reshape((height,width))
    result_img=Image.fromarray(final_array)
    #result_img.show()
    result_img.save("/media/kenny/Data/trained_models/residual_dense_models/noise_free/2x/results/dolls2x_result.png")
    ######################computing rmse
    final_array=final_array.astype(np.double)
    np_gth_dep=np_gth_dep.astype(np.double)
    #print(np.sqrt(((final_array-np_gth_dep)**2).mean()))
    #print((np.absolute(final_array-np_gth_dep)).mean())
    print("below are evaluated on 1080*1320")
    print(np.sqrt(((final_array[0:1080,0:1320]-np_gth_dep[0:1080,0:1320])**2).mean()))
    print((np.absolute(final_array[0:1080,0:1320]-np_gth_dep[0:1080,0:1320])).mean())
    #######################
