clc; clear; close all;
 
data_train_path = 'D:\MatlabFolder\bin\data_latih_res';
 
%%%%%%%%%  finding number of training images in the data path specified as argument  %%%%%%%%%%
filenames = dir(fullfile(data_train_path, '*.JPG'));
total_images = numel(filenames);
 
%%%%%%%%%%%%%%%%%%%%%%%%%%  creating the image matrix X  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = [];
for i = 1 : total_images
    str = fullfile(data_train_path, filenames(i).name);
    img = imread(str);
    img = rgb2gray(img);
    [r,c] = size(img);
    temp = reshape(img',r*c,1);  %% Reshaping 2D images into 1D image vectors
    %%% here img' is used because reshape(A,M,N) function reads the matrix A columnwise
    %%% where as an image matrix is constructed with first N pixels as first row,next N in second row so on
    X = [X temp];                %% X,the image matrix with columnsgetting added for each image
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Now we calculate m, A and eigenfaces.The descriptions are below :
%
%          m           -    (MxN)x1  Mean of the training images
%          A           -    (MxN)xP  Matrix of image vectors after each vector getting subtracted from the mean vector m
%     eigenfaces       -    (MxN)xP' P' Eigenvectors of Covariance matrix (C) of training database X
%                                    where P' is the number of eigenvalues of C that best represent the feature set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%%%%% calculating mean image vector %%%%%
 
m = mean(X,2); % Computing the average face image m = (1/P)*sum(Xj's)    (j = 1 : P)
imgcount = size(X,2);
 
%%%%%%%%  calculating A matrix, i.e. after subtraction of all image vectors from the mean image vector %%%%%%
 
A = [];
for i=1 : imgcount
    temp = double(X(:,i)) - m;
    A = [A temp];
end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALCULATION OF EIGENFACES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  we know that for a MxN matrix, the maximum number of non-zero eigenvalues that its covariance matrix can have
%%%  is min[M-1,N-1]. As the number of dimensions (pixels) of each image vector is very high compared to number of
%%%  test images here, so number of non-zero eigenvalues of C will be maximum P-1 (P being the number of test images)
%%%  if we calculate eigenvalues & eigenvectors of C = A*A' , then it will be very time consuming as well as memory.
%%%  so we calculate eigenvalues & eigenvectors of L = A'*A , whose eigenvectors will be linearly related to eigenvectors of C.
%%%  these eigenvectors being calculated from non-zero eigenvalues of C, will represent the best feature sets.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
L = A' * A;
[V,D] = eig(L);  %% V : eigenvector matrix  D : eigenvalue matrix
 
%%%% again we use Kaiser's rule here to find how many Principal Components (eigenvectors) to be taken
%%%% if corresponding eigenvalue is greater than 1, then the eigenvector will be chosen for creating eigenface
 
L_eig_vec = [];
for i = 1 : size(V,2)
    if( D(i,i) > 1 )
        L_eig_vec = [L_eig_vec V(:,i)];
    end
end
 
%%% finally the eigenfaces %%%
eigenfaces = A * L_eig_vec;

%In this part of recognition, we compare two faces by projecting the images into facespace and
% measuring the Euclidean distance between them.
%
%            recogimg           -   the recognized image name
%             testimg           -   the path of test image
%                m              -   mean image vector
%                A              -   mean subtracted image vector matrix
%           eigenfaces          -   eigenfaces that are calculated from eigenface function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%%%%%%% finding the projection of each image vector on the facespace (where the eigenfaces are the co-ordinates or dimensions) %%%%%
 
projectimg = [];  % projected image vector matrix
for i = 1 : size(eigenfaces,2)
    temp = eigenfaces' * A(:,i);
    projectimg = [projectimg temp];
end


%%%%% extracting PCA features of the train image %%%%%
train_target = zeros(total_images,1);
num_img = 15;
num_face = 2;
 
for i = 1 : num_face
    train_target((i-1)*num_img+1:i*num_img) = i;
end
 
class_index = zeros(total_images,1);
 
for n = 1 : total_images
    str = fullfile(data_train_path, filenames(n).name);
    test_image = imread(str);
    test_image = rgb2gray(test_image);
    [r,c] = size(test_image);
    temp = reshape(test_image',r*c,1); % creating (MxN)x1 image vector from the 2D image
    temp = double(temp)-m; % mean subtracted vector
    projtestimg = eigenfaces'*temp; % projection of test image onto the facespace
     
    %%%%% calculating & comparing the euclidian distance of all projected trained images from the projected test image %%%%%
     
    euclide_dist = [ ];
    for i=1 : size(eigenfaces,2)
        temp = (norm(projtestimg-projectimg(:,i)))^2;
        euclide_dist = [euclide_dist temp];
    end
     
    [euclide_dist_min,recognized_index] = min(euclide_dist);
    class_index(n) = train_target(recognized_index);
end
 
[~,b] = find(class_index==train_target);
train_accuracy = sum(b)/total_images*100



%%%%% extracting PCA features of the test image %%%%%
data_test_path = 'D:\MatlabFolder\bin\data_uji_res';
 
%%%%%%%%%  finding number of training images in the data path specified as argument  %%%%%%%%%%
filenames = dir(fullfile(data_test_path, '*.jpg'));
total_images = numel(filenames);
class_index = zeros(total_images,1);
 
for n = 1 : total_images
    str = fullfile(data_test_path, filenames(n).name);
    test_image = imread(str);
    test_image = rgb2gray(test_image);
    [r,c] = size(test_image);
    temp = reshape(test_image',r*c,1); % creating (MxN)x1 image vector from the 2D image
    temp = double(temp)-m; % mean subtracted vector
    projtestimg = eigenfaces'*temp; % projection of test image onto the facespace
     
    %%%%% calculating & comparing the euclidian distance of all projected trained images from the projected test image %%%%%
     
    euclide_dist = [ ];
    for i=1 : size(eigenfaces,2)
        temp = (norm(projtestimg-projectimg(:,i)))^2;
        euclide_dist = [euclide_dist temp];
    end
     
    [euclide_dist_min,recognized_index] = min(euclide_dist);
    class_index(n) = train_target(recognized_index);
end
 
test_target = zeros(total_images,1);
num_img = 2;
num_face = 2;
 
for i = 1 : num_face
    test_target((i-1)*num_img+1:i*num_img) = i;
end
 
[~,b] = find(class_index==test_target);
test_accuracy = sum(b)/total_images*100
save('pca','projectimg','eigenfaces','train_target','m')

% clc; clear; close all;
 
load pca
 
datapath = 'D:\MatlabFolder\bin\data_latih_res';
testimg = 'D:\MatlabFolder\bin\data_uji_res\b18.jpg';
 
filenames = dir(fullfile(datapath, '*.jpg'));
 
%%%%% extracting PCA features of the test image %%%%%
test_image = imread(testimg);
test_image = rgb2gray(test_image);
[r,c] = size(test_image);
temp = reshape(test_image',r*c,1); % creating (MxN)x1 image vector from the 2D image
temp = double(temp)-m; % mean subtracted vector
projtestimg = eigenfaces'*temp; % projection of test image onto the facespace
 
%%%%% calculating & comparing the euclidian distance of all projected trained images from the projected test image %%%%%
 
euclide_dist = [ ];
for i=1 : size(eigenfaces,2)
    temp = (norm(projtestimg-projectimg(:,i)))^2;
    euclide_dist = [euclide_dist temp];
end
 
[euclide_dist_min,recognized_index] = min(euclide_dist);
recognized_img = fullfile(datapath, filenames(recognized_index).name);
test_image = imread(testimg);
 
figure,
subplot(1,2,1),imshow(test_image),title('Test Image');
subplot(1,2,2),imshow(recognized_img),title('Recognized Image');
