pkg load image;

img1 = rgb2gray(imread('vinay_official.jpg'));
img2 = rgb2gray(imread('youngVinay_crop.jpg'));

img1_edged = edge(img1, "Canny");
img2_edged = edge(img2, "Canny");

figure(1);
imshow(img1);
figure(2);
imshow(img2);

figure(3);
imshow(img1_edged);
figure(4);
imshow(img2_edged);
