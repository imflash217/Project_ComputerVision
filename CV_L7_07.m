%%%%%%%%%%%%%%%%%%%

%% FUNCTION DEFINITIONS

addpath(pwd);   % needed to add function definitions in the same file

%%  Method-1: Shift left & right and show difference
function result =  edge_methodShift(iImage)
	%% converting the input Image to grayScale
	iImage_gray = rgb2gray(iImage);
	%% smoothing the image
	h = fspecial('gaussian', [11, 11], 4);
	% figure, surf(h), title('h : smoothing filter');
	iImage_smoothed = imfilter(iImage_gray, h);
	% figure, imshow(iImage_smoothed), title('smoothed via Gaussian filter');
	
	%% creating the left and right image
	iImage_L = iImage_smoothed;
	iImage_L(:, [1:(end-1)]) = iImage_L(:, [2:(end)]);
	iImage_R = iImage_smoothed;
	iImage_R(:, [2:(end)]) = iImage_L(:, [1:(end-1)]);
	
	iImage_LR_diff = double(iImage_R) - double(iImage_L);
	
	figure, imshow(iImage_LR_diff), title('Left-Right Difference');
	result = iImage_LR_diff;
	
endfunction

%% Canny Edge detector
function result = edge_methodCanny(iImage)
	%% converting the input Image to grayScale
	iImage_gray = rgb2gray(iImage);
	%% smoothing the image
	h = fspecial('gaussian', [11, 11], 4);
	% figure, surf(h), title('h : smoothing filter');
	iImage_smoothed = imfilter(iImage_gray, h);
	% figure, imshow(iImage_smoothed), title('smoothed via Gaussian filter');
	
	cannyEdges = edge(iImage_smoothed, 'canny');
	figure, imshow(cannyEdges), title('canny edged over smoothed image');
	
	result = cannyEdges;
endfunction


%% Laplassian-of-Gaussian 
function result = edge_methodLOG(iImage)
	%% converting the input Image to grayScale
	iImage_gray = rgb2gray(iImage);
	%% smoothing the image
	h = fspecial('gaussian', [11, 11], 4);
	% figure, surf(h), title('h : smoothing filter');
	iImage_smoothed = imfilter(iImage_gray, h);
	% figure, imshow(iImage_smoothed), title('smoothed via Gaussian filter');
	
	%% laplassian of Gaussian
	logEdges = edge(iImage_gray, 'log');
	figure, imshow(logEdges), title('laplassian of gaussian over NON-smoothed image');
	
	logEdges_smoothed = edge(iImage_smoothed, 'log');
	figure, imshow(logEdges_smoothed), title('laplassian of gaussian over SMOOTHED image');
	
	result = logEdges;

endfunction



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pkg load image;

edge_methodShift(imread('vinay_official.jpg'));
edge_methodCanny(imread('vinay_official.jpg'));
edge_methodLOG(imread('vinay_official.jpg'));













