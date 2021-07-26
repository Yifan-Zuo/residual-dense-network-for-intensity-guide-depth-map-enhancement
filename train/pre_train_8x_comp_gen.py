'''
Created on 3May,2017

@author: yizuo
'''
import tensorflow as tf
import guided_srgan_layer_blocks as gslb
import h5py

up_factor=8
batch_sz=32
height=128
width=128

#setting input size and training data addr
phase_train=tf.placeholder(tf.bool)
epo_range=50
train_h5F_addr="/media/kenny/Data/training_data/gdsr_train_data/8x_data/shuffle_version/8x_training_data.h5"
total_pat=220416
LR_height=height/up_factor
LR_width=width/up_factor
batch_total=total_pat/batch_sz
HR_patch_size=[height,width]
HR_batch_dims=(batch_sz,height,width,1)
LR_batch_dims=(batch_sz,LR_height,LR_width,1)

#setting input placeholders
HR_depth_batch_input=tf.placeholder(tf.float32,HR_batch_dims)
HR_inten_batch_input=tf.placeholder(tf.float32,HR_batch_dims)
LR_depth_batch_input=tf.placeholder(tf.float32,LR_batch_dims)
coar_inter_dep_batch=tf.image.resize_images(LR_depth_batch_input,tf.constant(HR_patch_size,dtype=tf.int32),tf.image.ResizeMethod.BICUBIC)

#gen_network construction
with tf.variable_scope("gen_inten"):
    guided_ten=gslb.inten_feature_extraction_unit(HR_inten_batch_input,phase_train=phase_train)
with tf.variable_scope("gen_down1_inten"):
    guided_4xto2x_gen=gslb.inten_downsample_simple_unit(guided_ten, phase_train=phase_train)
with tf.variable_scope("gen_down2_inten"):
    guided_8xto4x_gen=gslb.inten_downsample_simple_unit(guided_4xto2x_gen, phase_train=phase_train)
with tf.variable_scope("gen_dep"):
    dep_ten=gslb.LR_dep_feature_extraction_uint(LR_depth_batch_input,phase_train=phase_train)
    dep_ten=gslb.LR_dep_upsampling_unit(dep_ten, phase_train=phase_train)
with tf.variable_scope("gen_up3_dep"):
    dep_ten=gslb.LR_dep_fusion_simple_unit(dep_ten, guided_8xto4x_gen, phase_train=phase_train)
    dep_ten=gslb.LR_dep_upsampling_unit(dep_ten, phase_train=phase_train)
with tf.variable_scope("gen_up2_dep"):
    dep_ten=gslb.LR_dep_fusion_simple_unit(dep_ten, guided_4xto2x_gen, phase_train=phase_train)
    dep_ten=gslb.LR_dep_upsampling_unit(dep_ten, phase_train=phase_train)
with tf.variable_scope("gen_8x_last4_layers"):
    gen_ten=gslb.LR_dep_fusion_simple_unit(dep_ten, guided_ten,phase_train=phase_train)
    gen_ten=gslb.LR_recon_unit(gen_ten, coar_inter_dep_batch,phase_train=phase_train)

#define loss for gen
#loss=tf.reduce_mean(tf.squared_difference(gen_ten,HR_depth_batch_input))
loss=tf.reduce_mean(tf.abs(gen_ten-HR_depth_batch_input))
#loss=tf.reduce_mean(tf.sqrt(tf.squared_difference(gen_ten,HR_depth_batch_input)+1e-3))
train_op_small = tf.train.AdamOptimizer(1e-5).minimize(loss)
train_op_large = tf.train.AdamOptimizer(1e-4).minimize(loss)

#initial net and saver object
#var_list=tf.trainable_variables()
#gen_reuse_vars = [v for v in var_list if (v.name.startswith("gen_dep")) or (v.name.startswith("gen_inten")) or (v.name.startswith("gen_down1_inten")) or (v.name.startswith("gen_up2_dep"))]
#saving_var_list=gen_reuse_vars+[v for v in var_list if (v.name.startswith("gen_down2_inten")) or (v.name.startswith("gen_up3_dep"))]
#saver = tf.train.Saver(saving_var_list,max_to_keep=35)
saver_full=tf.train.Saver(max_to_keep=120)
#gen_reuse_saver=tf.train.Saver(gen_reuse_vars)
model_ind=0
init_op=tf.global_variables_initializer()

#begin comp_gen training
with h5py.File(train_h5F_addr,"r") as train_file:
    with tf.Session() as sess:
        sess.run(init_op)
        #gen_reuse_saver.restore(sess, "/home/yifan/eclipse4.6_workspace/guided_srgan_multi_skip_version/train/4x_pretrain_comp_gen/std5_version/full_model1/4x_comp_gen_model_full.ckpt-15")
        #saver_full.restore(sess, "/media/kenny/Data/trained_models/residual_dense_models/noise_free/8x/full_model1/8x_nf_full_model.ckpt-45")
        for epo in range(epo_range):
            if epo<25:
                train_op=train_op_large
            else:
                train_op=train_op_small
            for ind in range(batch_total):
                gen_pat_ind_range=range(ind*batch_sz,(ind+1)*batch_sz,1)
                gen_inten_bat,gen_gth_dep_bat,gen_LR_dep_bat=gslb.reading_data(train_file, gen_pat_ind_range, HR_batch_dims, LR_batch_dims)
                sess.run(train_op,feed_dict={HR_inten_batch_input:gen_inten_bat,HR_depth_batch_input:gen_gth_dep_bat,LR_depth_batch_input:gen_LR_dep_bat,phase_train:True})
                if (ind+1)%861==0:
                    train_mse_loss=loss.eval(feed_dict={HR_inten_batch_input:gen_inten_bat,HR_depth_batch_input:gen_gth_dep_bat,LR_depth_batch_input:gen_LR_dep_bat,phase_train:True})
                    print("step %d, training loss %g"%(ind, train_mse_loss))
                if (ind+1)%3444==0:
                    save_path=saver_full.save(sess,"/media/kenny/Data/trained_models/residual_dense_models/noise_free/8x/retrain_l1loss/full_model1/8x_nf_full_model.ckpt",global_step=model_ind)
                    print("Full Model saved in file: %s" % save_path)
                    model_ind=model_ind+1
            #save_path=saver.save(sess, "/home/yifan/eclipse4.6_workspace/guided_srgan_multi_skip_version/train/8x_pretrain_comp_gen/std5_version/reuse_part1/8x_comp_gen_model_part.ckpt",global_step=epo) 
            #print("Reusing Model saved in file: %s" % save_path)
            #save_path=saver_full.save(sess,"/media/yf/Data/yf/trained_models/guided_depth_up/noise_free/8x/full_model1/8x_nf_full_model.ckpt",global_step=epo)
            #print("Full Model saved in file: %s" % save_path)
