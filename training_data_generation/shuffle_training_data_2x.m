function [  ] = shuffle_training_data_2x( )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
clc;
clear;
%%%%%%%%%%%%%%% changing part for different up-sampling factors
down_fac=2;
h5_des_name='/media/kenny/Data/training_data/gdsr_train_data/2x_data/shuffle_version/2x_training_data.h5';
h5_src_addr_pre='./2x_data/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_pat=220500;
shuf_train=randperm(train_pat);
valid_pat=24500;
shuf_valid=randperm(valid_pat)+train_pat;
shuf_ind=[shuf_train,shuf_valid];
total_pat=245000;
HR_pat_dims=[128,128];
LR_pat_dims=HR_pat_dims./down_fac;
read_bat=1000;
bat_num=total_pat/read_bat;
h5create(h5_des_name,'/depth_patch',[HR_pat_dims,total_pat],'Datatype','single');
h5create(h5_des_name,'/inten_patch',[HR_pat_dims,total_pat],'Datatype','single');
h5create(h5_des_name,'/LR_depth_patch',[LR_pat_dims,total_pat],'Datatype','single');
h5create(h5_des_name,'/LR_noisy_depth_std5',[LR_pat_dims,total_pat],'Datatype','single');
h5create(h5_des_name,'/LR_noisy_depth_std10',[LR_pat_dims,total_pat],'Datatype','single');
h5create(h5_des_name,'/LR_noisy_depth_std15',[LR_pat_dims,total_pat],'Datatype','single');
h5create(h5_des_name,'/LR_noisy_depth_std20',[LR_pat_dims,total_pat],'Datatype','single');
HRdep_data_buffer=zeros([HR_pat_dims,read_bat]);
HRinten_data_buffer=zeros([HR_pat_dims,read_bat]);
LRdep_data_buffer=zeros([LR_pat_dims,read_bat]);
LRdepstd5_data_buffer=zeros([LR_pat_dims,read_bat]);
LRdepstd10_data_buffer=zeros([LR_pat_dims,read_bat]);
LRdepstd15_data_buffer=zeros([LR_pat_dims,read_bat]);
LRdepstd20_data_buffer=zeros([LR_pat_dims,read_bat]);
for bat_index=1:bat_num
    for pat_index=1:read_bat
        tmp_ind=shuf_ind(pat_index+(bat_index-1)*read_bat);
        HRdep_data_buffer(:,:,pat_index)=h5read([h5_src_addr_pre,'depth_patches.h5'],'/depth_patch',[1,1,tmp_ind],[HR_pat_dims,1]);
        HRinten_data_buffer(:,:,pat_index)=h5read([h5_src_addr_pre,'inten_patches.h5'],'/inten_patch',[1,1,tmp_ind],[HR_pat_dims,1]);
        LRdep_data_buffer(:,:,pat_index)=h5read([h5_src_addr_pre,'LR_depth_patches.h5'],'/LR_depth_patch',[1,1,tmp_ind],[LR_pat_dims,1]);
        LRdepstd5_data_buffer(:,:,pat_index)=h5read([h5_src_addr_pre,'LR_noisy_depth_stdvar5.h5'],'/LR_noisy_depth_std5',[1,1,tmp_ind],[LR_pat_dims,1]);
        LRdepstd10_data_buffer(:,:,pat_index)=h5read([h5_src_addr_pre,'LR_noisy_depth_stdvar10.h5'],'/LR_noisy_depth_std10',[1,1,tmp_ind],[LR_pat_dims,1]);
        LRdepstd15_data_buffer(:,:,pat_index)=h5read([h5_src_addr_pre,'LR_noisy_depth_stdvar15.h5'],'/LR_noisy_depth_std15',[1,1,tmp_ind],[LR_pat_dims,1]);
        LRdepstd20_data_buffer(:,:,pat_index)=h5read([h5_src_addr_pre,'LR_noisy_depth_stdvar20.h5'],'/LR_noisy_depth_std20',[1,1,tmp_ind],[LR_pat_dims,1]);
    end
    w_start=(bat_index-1)*read_bat+1;
    h5write(h5_des_name,'/depth_patch',single(HRdep_data_buffer),[1,1,w_start],size(HRdep_data_buffer));
    h5write(h5_des_name,'/inten_patch',single(HRinten_data_buffer),[1,1,w_start],size(HRinten_data_buffer));
    h5write(h5_des_name,'/LR_depth_patch',single(LRdep_data_buffer),[1,1,w_start],size(LRdep_data_buffer));
    h5write(h5_des_name,'/LR_noisy_depth_std5',single(LRdepstd5_data_buffer),[1,1,w_start],size(LRdepstd5_data_buffer));
    h5write(h5_des_name,'/LR_noisy_depth_std10',single(LRdepstd10_data_buffer),[1,1,w_start],size(LRdepstd10_data_buffer));
    h5write(h5_des_name,'/LR_noisy_depth_std15',single(LRdepstd15_data_buffer),[1,1,w_start],size(LRdepstd15_data_buffer));
    h5write(h5_des_name,'/LR_noisy_depth_std20',single(LRdepstd20_data_buffer),[1,1,w_start],size(LRdepstd20_data_buffer));
    disp('batch ok! ')
end
end