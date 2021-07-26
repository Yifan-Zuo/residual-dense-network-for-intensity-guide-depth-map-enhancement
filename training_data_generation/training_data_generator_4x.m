function [  ] = training_data_generator_4x( )
% stride is a 4-element vector, the size of stride in the adjacent patches for [2x,4x,8x,16x], full_size is a 4
% element vector according to [2x, 4x, 8x, 16x].
% stride=[64,64,64,64];
% HR_patch_size=[128,128,128,128];
% LR_patch_size=[64,32,16,8];
% patch_total_onedim=[7,7,7,7];
% h5_des_name=['./2x_data/', './4x_data/', './8x_data/', './16x_data/'];
% down_factor=[0.5,0.25,0.125,1/16];

clc;
clear;
allnames=struct2cell(dir('../gdsr_dataset/*.h5'));
[~,total_h5imgs]=size(allnames);
full_width=512;
full_height=512;
stride=64;
HR_patch_size=128;
LR_patch_size=32;
patch_total_onedim=7;
down_factor=0.25;
h5_src_name='../gdsr_dataset/';
h5_des_name='./4x_data/';
batch_total_one_img=patch_total_onedim^2;
batch_total=batch_total_one_img*total_h5imgs;
h5create([h5_des_name,'depth_patches.h5'],'/depth_patch',[HR_patch_size,HR_patch_size,batch_total],'ChunkSize',[HR_patch_size,HR_patch_size,1]);
h5create([h5_des_name,'inten_patches.h5'],'/inten_patch',[HR_patch_size,HR_patch_size,batch_total],'ChunkSize',[HR_patch_size,HR_patch_size,1]);
h5create([h5_des_name,'LR_depth_patches.h5'],'/LR_depth_patch',[LR_patch_size,LR_patch_size,batch_total],'ChunkSize',[LR_patch_size,LR_patch_size,1]);
h5create([h5_des_name,'LR_noisy_depth_stdvar5.h5'],'/LR_noisy_depth_std5',[LR_patch_size,LR_patch_size,batch_total],'ChunkSize',[LR_patch_size,LR_patch_size,1]);
h5create([h5_des_name,'LR_noisy_depth_stdvar10.h5'],'/LR_noisy_depth_std10',[LR_patch_size,LR_patch_size,batch_total],'ChunkSize',[LR_patch_size,LR_patch_size,1]);
h5create([h5_des_name,'LR_noisy_depth_stdvar15.h5'],'/LR_noisy_depth_std15',[LR_patch_size,LR_patch_size,batch_total],'ChunkSize',[LR_patch_size,LR_patch_size,1]);
h5create([h5_des_name,'LR_noisy_depth_stdvar20.h5'],'/LR_noisy_depth_std20',[LR_patch_size,LR_patch_size,batch_total],'ChunkSize',[LR_patch_size,LR_patch_size,1]);
reshaped_color=single(zeros(full_height,full_width,3));
depth_patch=single(zeros(HR_patch_size,HR_patch_size,batch_total_one_img));
inten_patch=single(zeros(HR_patch_size,HR_patch_size,batch_total_one_img));
LR_depth_patch=single(zeros(LR_patch_size,LR_patch_size,batch_total_one_img));
noisy_LR_depth_patch_std5=single(zeros(LR_patch_size,LR_patch_size,batch_total_one_img));
noisy_LR_depth_patch_std10=single(zeros(LR_patch_size,LR_patch_size,batch_total_one_img));
noisy_LR_depth_patch_std15=single(zeros(LR_patch_size,LR_patch_size,batch_total_one_img));
noisy_LR_depth_patch_std20=single(zeros(LR_patch_size,LR_patch_size,batch_total_one_img));
for index=1:total_h5imgs
    sub_img_index=1;
    h5_name=allnames{1,index};
    color_img=h5read([h5_src_name,h5_name],'/rgb');
    reshaped_color(:,:,1)=color_img(1,:,:);
    reshaped_color(:,:,2)=color_img(2,:,:);
    reshaped_color(:,:,3)=color_img(3,:,:);
    inten_img=rgb2gray(reshaped_color);
    depth_img=h5read([h5_src_name,h5_name],'/depth');
    depth_img=depth_img./max(max(depth_img));    
    for patch_index_h=1:stride:full_height
        patch_h_up_ran=patch_index_h+HR_patch_size-1;
        if patch_h_up_ran>full_height
            continue;
        end
        patch_h_range=patch_index_h:patch_h_up_ran;
        for patch_index_w=1:stride:full_width
            patch_w_up_ran=patch_index_w+HR_patch_size-1;
            if patch_w_up_ran>full_width
                continue;
            end
            patch_w_range=patch_index_w:patch_w_up_ran;
            depth_patch(:,:,sub_img_index)=depth_img(patch_h_range,patch_w_range);
            inten_patch(:,:,sub_img_index)=inten_img(patch_h_range,patch_w_range);
            LR_depth_subimg=imresize(depth_img(patch_h_range,patch_w_range),down_factor,'bicubic');
            LR_depth_patch(:,:,sub_img_index)=LR_depth_subimg;
            noisy_std5_LR_depth_subimg=imnoise(LR_depth_subimg,'gaussian',0,25/65025);
            noisy_LR_depth_patch_std5(:,:,sub_img_index)=noisy_std5_LR_depth_subimg;
            noisy_std10_LR_depth_subimg=imnoise(LR_depth_subimg,'gaussian',0,100/65025);
            noisy_LR_depth_patch_std10(:,:,sub_img_index)=noisy_std10_LR_depth_subimg;            
            noisy_std15_LR_depth_subimg=imnoise(LR_depth_subimg,'gaussian',0,225/65025);
            noisy_LR_depth_patch_std15(:,:,sub_img_index)=noisy_std15_LR_depth_subimg;            
            noisy_std20_LR_depth_subimg=imnoise(LR_depth_subimg,'gaussian',0,400/65025);
            noisy_LR_depth_patch_std20(:,:,sub_img_index)=noisy_std20_LR_depth_subimg;            
            sub_img_index=sub_img_index+1;
        end
    end
    batch_range_start=(index-1)*batch_total_one_img+1;
    h5write([h5_des_name,'depth_patches.h5'],'/depth_patch',depth_patch,[1,1,batch_range_start],size(depth_patch));
    h5write([h5_des_name,'inten_patches.h5'],'/inten_patch',inten_patch,[1,1,batch_range_start],size(inten_patch));
    h5write([h5_des_name,'LR_depth_patches.h5'],'/LR_depth_patch',LR_depth_patch,[1,1,batch_range_start],size(LR_depth_patch));
    h5write([h5_des_name,'LR_noisy_depth_stdvar5.h5'],'/LR_noisy_depth_std5',noisy_LR_depth_patch_std5,[1,1,batch_range_start],size(noisy_LR_depth_patch_std5));
    h5write([h5_des_name,'LR_noisy_depth_stdvar10.h5'],'/LR_noisy_depth_std10',noisy_LR_depth_patch_std10,[1,1,batch_range_start],size(noisy_LR_depth_patch_std10));
    h5write([h5_des_name,'LR_noisy_depth_stdvar15.h5'],'/LR_noisy_depth_std15',noisy_LR_depth_patch_std15,[1,1,batch_range_start],size(noisy_LR_depth_patch_std15));
    h5write([h5_des_name,'LR_noisy_depth_stdvar20.h5'],'/LR_noisy_depth_std20',noisy_LR_depth_patch_std20,[1,1,batch_range_start],size(noisy_LR_depth_patch_std20));
end
end