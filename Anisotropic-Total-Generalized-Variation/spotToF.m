
clear all;
close all;
clc;
% read high-res rgb and depth
disp_gt = double(imread('test_imgs/bkf_sparse_depth_0.png'));%depth
disp_gt (disp_gt == 0 ) = 3000;
disp_gt = disp_gt/10;
gray_gt = im2double(rgb2gray(imread('test_imgs/RGB_0.png')));%rgb

ours = disp_gt;
smin = min(disp_gt(:));
smax = max(disp_gt(:));
fprintf('smin is %d,smax is %d\n',smin,smax);
    
[M, N] = size(disp_gt);

weights = zeros(M,N);
weights(ours < 300 ) = 1;
%weights(ours == 0.0) = 32001;


% normalize input depth map 
d_min = min(ours(ours>0));
d_max = max(ours(ours>0));
fprintf('d_min is %d,d_max is %d\n',d_min,d_max); 

ours_norm = (ours-d_min)/(d_max-d_min);
ours_norm(ours_norm < 0) = 0;
ours_norm(ours_norm > 1) = 1;
%disp(ours_norm);
csvwrite("test_imgs/ours_norm.csv", ours_norm);
%% tgv l2
factor = 1;

timestep_lambda = 1;

%tgv_alpha = [17 1.2];
%tgv_alpha = [5 0.02];
tgv_alpha = [3.95 0.279];

tensor_ab = [9 0.85];

lambda_tgvl2 = 40;

maxits = 150;
disp(' ---- ');
%check = round(maxits/10);
check = round(maxits/100);
% check = maxits+10;
     
upsampling_result_norm = upsamplingTensorTGVL2(ours_norm, ours_norm, ...
weights.*lambda_tgvl2, gray_gt, tensor_ab, tgv_alpha./factor, timestep_lambda, maxits, ...
check, 0.1, 1);

%csvwrite("test_imgs/upsampling_result_norm.csv", upsampling_result_norm);

upsampling_result = (upsampling_result_norm*(d_max-d_min)+d_min);

upsampling_result_show = uint8(upsampling_result_norm*255);

csvwrite("test_imgs/spot_result_1000_alph3.csv", upsampling_result*10);

%upsampling_result = uint8(upsampling_result_norm*255);

%disp(upsampling_result)

%csvwrite("test_imgs/upsampling_result.csv", upsampling_result);

mse_ours = sum(sum(sqrt((double(disp_gt)-double(upsampling_result)).^2))) / numel(disp_gt);

figure;
subplot(211); imshow(disp_gt,[smin smax]); title(sprintf('groundtruth %dx%d',size(disp_gt)));
subplot(212); imshow(upsampling_result_show,[smin smax]); title('ours');

colormap(jet);
impixelinfo;
drawnow;


fprintf(1,'-- OURS: \n');
disp(['mse         =  ' num2str(mse_ours)]);
