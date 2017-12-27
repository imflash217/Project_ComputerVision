%% function definitions
addpath(pwd);   % needed to add function definitions in the same file

function result = select_gdir(gmag, gdir, mag_min, angle_low, angle_high)
  % find and return pixels that fall within the desired mag, angle range
  result = (gmag >= mag_min) & (gdir > angle_low) & (gdir < angle_high);
endfunction

%%%%%%%%%%%%%%%%%%%%

pkg load image;

img = double(imread('origgon_web.png'))/255.;
img = rgb2gray(img);
figure(1);
imshow(img);

% compute x, y gradients
[gx gy] = imgradientxy(img, 'sobel'); %note: gx, gy are not normalized

% obtain gradient magnitude and direction
[gmag gdir] = imgradient(gx, gy);
figure(2);
imshow(gmag / (4*sqrt(2))); % mag = sqrt(gx^2 + gy^2), so [0, (4*sqrt(2))]
figure(3);
imshow((gdir + 180.0) / 360.0); % angle in degrees [-180, 180]

% find pixels with desired gradient direction
my_grad = select_gdir(gmag, gdir, 1, 30, 60);
figure(4);
imshow(my_grad);

