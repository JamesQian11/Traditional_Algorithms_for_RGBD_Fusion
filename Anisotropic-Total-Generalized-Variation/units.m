clear all;
close all;
clc;
row=304;  
col=228;

fin = ("img_gray_1.csv"); % range (0,1200)
I=csvread(fin);
disp(I);
K= imshow(I,[]);
    


 
