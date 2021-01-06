'''
Created on 6Apr.,2017

@author: yifanzuo
'''
import tensorflow as tf
#define sub-functions for creating variables
#ksize,w_shape,b_shape,strides are all list type, this is very important
def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(w_shape,std_dev=0.02,name=None):
    initial_W = tf.truncated_normal(w_shape, stddev=std_dev)
    if name is None:
        return tf.Variable(initial_W,name="conv_weight",dtype=tf.float32)
    else:
        return tf.get_variable(name,dtype=tf.float32,initializer=initial_W)
    
def bias_variable(b_shape,name=None):
    initial_B = tf.constant(0.0, shape=b_shape)
    if name is None:
        return tf.Variable(initial_B,name="conv_bias",dtype=tf.float32)
    else:
        return tf.get_variable(name,dtype=tf.float32,initializer=initial_B)
    
def Prelu(input_tensor,name=None):
    initial_a=tf.constant(0.25, shape=[input_tensor.get_shape().as_list()[3]])
    if name is None:
        alphas=tf.Variable(initial_a,name="prelu_alpha",dtype=tf.float32)
    else:
        alphas=tf.get_variable(name,dtype=tf.float32,initializer=initial_a)   
    pos = tf.nn.relu(input_tensor)
    neg = alphas * (input_tensor - abs(input_tensor)) * 0.5
    return pos+neg

#define batch normalization function

def batch_norm(x, n_out, phase_train,name=None):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    if name is None:
        beta = tf.Variable(tf.constant(0.0, shape=n_out),name="bn_beta",dtype=tf.float32)
        gamma = tf.Variable(tf.random_normal(n_out, 1.0, 0.02),name="bn_gamma",dtype=tf.float32)
    else:
        beta=tf.get_variable(name[0],dtype=tf.float32,initializer=tf.constant(0.0, shape=n_out))
        gamma =tf.get_variable(name[1],dtype=tf.float32,initializer=tf.random_normal(n_out, 1.0, 0.02))
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(phase_train,mean_var_with_update,lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    return normed

#define sub-function for conv layer with batch normalization
def conv_Prelu_bn_block(input_tensor,w_shape,strides,skip_Prelu,skip_bn,phase_train):
    W=weight_variable(w_shape)
    conv_ten = tf.nn.conv2d(input_tensor, W,strides, padding='SAME')
    if skip_bn:
        B=bias_variable([w_shape[3]])
        if skip_Prelu:
            return conv_ten + B
        else:
            return Prelu(conv_ten + B)
    else:
        bn_ten=batch_norm(conv_ten,[w_shape[3]],phase_train)
        if skip_Prelu:
            return bn_ten
        else:
            return Prelu(bn_ten)
        
#define sub-function for deconv layer with batch normalization
def deconv_Prelu_bn_block(input_tensor,w_shape,out_put_shape,strides,skip_Prelu,skip_bn,phase_train): 
    W=weight_variable(w_shape) 
    deconv_ten= tf.nn.conv2d_transpose(input_tensor,W,out_put_shape,strides,padding="SAME")
    if skip_bn:
        B=bias_variable([w_shape[2]])
        if skip_Prelu:
            return deconv_ten + B
        else:
            return Prelu(deconv_ten + B)
    else:
        bn_ten=batch_norm(deconv_ten,[w_shape[2]],phase_train)
        if skip_Prelu:
            return bn_ten
        else:
            return Prelu(bn_ten)
        
#from this on is the construction blocks of gen network
#define sub-functions for the first intensity layer block (the first three layers)
def inten_feature_extraction_unit(block_ten,shape_list=[[7,7,1,64],[7,7,64,32]],strides=[1,1,1,1],skip_Prelu_list=[False,False],skip_bn_list=[True,True],phase_train=tf.constant(True,tf.bool)):
    for index in range(len(skip_Prelu_list)):
        block_ten=conv_Prelu_bn_block(block_ten, shape_list[index], strides, skip_Prelu_list[index], skip_bn_list[index], phase_train)
    inten_kz3=conv_Prelu_bn_block(block_ten, [5,5,32,32], strides, False, True, phase_train)
    return inten_kz3

def inten_downsample_simple_unit(input_tensor,shape=[5,5,32,32],strides=[1,1,1,1],skip_Prelu=False,skip_bn=True,phase_train=tf.constant(True,tf.bool)):
    block_ten_kz3= conv_Prelu_bn_block(input_tensor, shape, strides, skip_Prelu, skip_bn, phase_train)
    block_ten_kz3=max_pool_3x3(block_ten_kz3)
    return block_ten_kz3

def inten_downsample_typical_unit(input_tensor,shape=[[7,7,32,32],[5,5,32,32],[3,3,32,32]],strides=[1,1,1,1],skip_Prelu=False,skip_bn=False,phase_train=tf.constant(True,tf.bool)):
    block_ten_kz7= conv_Prelu_bn_block(input_tensor[0], shape[0], strides, skip_Prelu, skip_bn, phase_train)
    block_ten_kz5= conv_Prelu_bn_block(input_tensor[1], shape[1], strides, skip_Prelu, skip_bn, phase_train)
    block_ten_kz3= conv_Prelu_bn_block(input_tensor[2], shape[2], strides, skip_Prelu, skip_bn, phase_train)
    block_ten_kz7=max_pool_3x3(block_ten_kz7)
    block_ten_kz5=max_pool_3x3(block_ten_kz5)
    block_ten_kz3=max_pool_3x3(block_ten_kz3)
    return (block_ten_kz7,block_ten_kz5,block_ten_kz3)

def LR_dep_feature_extraction_uint(block_ten,shape_list=[[5,5,1,64],[5,5,64,32]],strides=[1,1,1,1],skip_Prelu_list=[False,False],skip_bn_list=[True,True],phase_train=tf.constant(True,tf.bool)):
    for index in range(len(skip_Prelu_list)):
        block_ten=conv_Prelu_bn_block(block_ten, shape_list[index], strides, skip_Prelu_list[index], skip_bn_list[index], phase_train)
    return block_ten

def LR_dep_upsampling_unit(input_tensor,shape=[5,5,32,32],strides=[1,2,2,1],skip_Prelu=False,skip_bn=True,phase_train=tf.constant(True,tf.bool)):
    input_shape=input_tensor.get_shape().as_list()
    out_shape_list=[input_shape[0],2*input_shape[1],2*input_shape[2],shape[2]]
    block_ten=deconv_Prelu_bn_block(input_tensor, shape,tf.convert_to_tensor(out_shape_list), strides, skip_Prelu, skip_bn, phase_train)
    return block_ten

def RDB_block(in_ten,conv_num=6,kz=5,ch_num=32,stride=[1,1,1,1],skip_Prelu=False,skip_bn=True,phase_train=tf.constant(True,tf.bool)):
    for ind in range(conv_num-1):
        in_ch=(ind+1)*ch_num
        mid_ten=conv_Prelu_bn_block(in_ten, [kz,kz,in_ch,ch_num], stride, skip_Prelu,skip_bn,phase_train)
        in_ten=tf.concat([in_ten,mid_ten],3)
    mid_ten=conv_Prelu_bn_block(in_ten, [1,1,in_ch+ch_num,ch_num], stride, skip_Prelu=True,skip_bn=skip_bn,phase_train=phase_train)
    return mid_ten

def LR_dep_fusion_simple_unit(input_dep_ten,input_inten_ten,shape_list=[5,5,64,32],strides=[1,1,1,1],skip_Prelu_list=False,skip_bn_list=True,phase_train=tf.constant(True,tf.bool)):
    concat_input=tf.concat([input_dep_ten,input_inten_ten],3)
    fusion_ten=conv_Prelu_bn_block(concat_input, shape_list, strides, skip_Prelu_list, skip_bn_list, phase_train)
    fusion_final=RDB_block(fusion_ten,6,5,32,[1,1,1,1],skip_Prelu_list,skip_bn_list,phase_train)
    return fusion_final+input_dep_ten

def LR_recon_unit(block_ten,input_coarse_ten,shape_list=[[5,5,32,32],[5,5,32,1]],strides=[1,1,1,1],skip_Prelu_list=[False,True],skip_bn_list=[True,True],phase_train=tf.constant(True,tf.bool)):
    for index in range(len(skip_Prelu_list)):
        block_ten=conv_Prelu_bn_block(block_ten, shape_list[index], strides, skip_Prelu_list[index], skip_bn_list[index], phase_train)
    block_ten=block_ten+input_coarse_ten
    return block_ten

#from this on is construction blocks of disc network
#define sub-function for 1conv followed by 1leaky relu
def disc_layer_block(input_tensor,w_shape,strides,skip_Lrelu=False,skip_bn=False,phase_train=tf.constant(True,tf.bool),Name=[None,None,None,None],scope="disc",reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        W=weight_variable(w_shape,name=Name[0])
        conv_ten = tf.nn.conv2d(input_tensor, W,strides, padding='SAME')
    if skip_bn:
        with tf.variable_scope(scope,reuse=reuse):
            B=bias_variable([w_shape[3]],name=Name[1])
            if skip_Lrelu:
                return conv_ten + B
            else:
                return leaky_relu(conv_ten + B)
    else:
        bn_ten=batch_norm_disc(conv_ten,[w_shape[3]],phase_train,name=Name[2:4],scope=scope,reuse=reuse)
        with tf.variable_scope(scope,reuse=reuse):
            if skip_Lrelu:
                return bn_ten
            else:
                return leaky_relu(bn_ten)

def disc_fc_units(input_ten,keep_prob=0.5,scope="disc",fc_shape=[1024,1],reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        input_shape=input_ten.get_shape().as_list()
        input_size=input_shape[1]*input_shape[2]*input_shape[3]
        wfc_1=weight_variable([input_size,fc_shape[0]], name="fc1w")
        bfc_1=bias_variable([fc_shape[0]],name="fc1b")
        input_ten_flat=tf.reshape(input_ten,[input_shape[0],input_size])
        fc1_ten=leaky_relu(tf.matmul(input_ten_flat, wfc_1) + bfc_1)
        fc1_ten = tf.nn.dropout(fc1_ten, keep_prob)
        wfc_2=weight_variable([fc_shape[0],fc_shape[1]], name="fc2w")
        bfc_2=bias_variable([fc_shape[1]],name="fc2b")
        fc2_ten=tf.matmul(fc1_ten, wfc_2) + bfc_2
    return fc2_ten

def disc_fc_wgan(input_ten,scope="disc",reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        input_shape=input_ten.get_shape().as_list()
        input_size=input_shape[1]*input_shape[2]*input_shape[3]
        wfc_1=weight_variable([input_size,1], name="fc1w")
        bfc_1=bias_variable([1],name="fc1b")
        input_ten_flat=tf.reshape(input_ten,[input_shape[0],input_size])
        fc1_ten=tf.matmul(input_ten_flat, wfc_1) + bfc_1
    return fc1_ten

#define function for reading data from hdf5
def reading_data(train_file,pat_ind_range,HR_batch_dims,LR_batch_dims):
    inten_bat=train_file['inten_patch'][pat_ind_range,:,:]
    inten_bat=inten_bat.reshape(HR_batch_dims)
    gth_dep_bat=train_file['depth_patch'][pat_ind_range,:,:]
    gth_dep_bat=gth_dep_bat.reshape(HR_batch_dims)
    #LR_dep_bat=train_file['LR_noisy_depth_std5'][pat_ind_range,:,:]
    LR_dep_bat=train_file['LR_depth_patch'][pat_ind_range,:,:]
    LR_dep_bat=LR_dep_bat.reshape(LR_batch_dims) 
    return inten_bat,gth_dep_bat,LR_dep_bat

def batch_norm_disc(x, n_out, phase_train,name=None,scope="disc",reuse=False):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope,reuse=reuse):
        if name is None:
            beta = tf.Variable(tf.constant(0.0, shape=n_out),name="bn_beta",dtype=tf.float32)
            gamma = tf.Variable(tf.random_normal(n_out, 1.0, 0.02),name="bn_gamma",dtype=tf.float32)
        else:
            beta=tf.get_variable(name[0],dtype=tf.float32,initializer=tf.constant(0.0, shape=n_out))
            gamma =tf.get_variable(name[1],dtype=tf.float32,initializer=tf.random_normal(n_out, 1.0, 0.02))
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(phase_train,mean_var_with_update,lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    return normed






