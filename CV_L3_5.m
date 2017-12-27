pkg load image;
image = imread('vinay_official.jpg');
hsize = 1;
sigma = 1;
%for sigma = 1:3:10
h = fspecial('gaussian', hsize, sigma);
%surf(h);
%pause;
%imagesc(h);
image_noisy = imfilter(image, h);
imshow(image_noisy);
%pause(5);
%end