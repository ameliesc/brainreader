%% DEMO_SPARSE_CODING

% This is a demo of the neural-coding toolbox using sparse coding (https://github.com/ccnlab/neural-coding).

%% Add neural-coding toolbox to search path, cleanup and seed random number generator

addpath(genpath(pwd));

close all; clear all; clc;

sd = 1; rng(sd); 

%% Load data

features           = h5read('featuremap_train_conv3_3.h5', '/data');% from the vim-1 data set.
test_feat = h5read('featuremaps_test_conv3_3.h5', '/data');
% size(response, 1) is the number of stimuli.
% size(response, 2) is the number of voxels.
% For example:
EstimatedResponses = load('EstimatedResponses.mat');                       % from the vim-1 data set.
ROI                = 1;
training_response  = EstimatedResponses.dataTrnS1(ROI == EstimatedResponses.roiS1 & all(isfinite(EstimatedResponses.dataTrnS1), 2) & all(isfinite(EstimatedResponses.dataValS1), 2), :)';
test_response      = EstimatedResponses.dataValS1(ROI == EstimatedResponses.roiS1 & all(isfinite(EstimatedResponses.dataTrnS1), 2) & all(isfinite(EstimatedResponses.dataValS1), 2), :)';




%% Define response model

% All of the following fields of the response_model structure should be defined:
% name, K and lambda_number.

response_model = [];

% name can be 'kernel_ridge', 'elastic_net' or 'Lasso'.
response_model.name = 'kernel_ridge';

% fold_number is the number of k-fold cross-validation folds.
% fold_number is required for kernel_ridge, elastic_net and Lasso models.
% For example:
response_model.fold_number = 3;

% lambda_number is the number of complexity parameters that control the amount of regularization.
% lambda_number is required for kernel_ridge, elastic_net and Lasso models.
% For example:
response_model.lambda_number = 10;

%% Train response model

% If you do not want to use the Parallel Computing Toolbox then:
% change the parfor in line 25 of response_models/kernel_ridge/kernel_ride.m to for for the kernel_ridge model
% change the true in line 11 of wrappers/train_response_model.m to false for the elastic_net model
% change the true in line 19 of wrapper/train_response_model.m to false for the Lasso model
%responses_weights = struct('conv1_1',[], 'conv1_2',[], 'conv2_1',[], 'conv2_2',[],'conv3_1', [], 'conv3_2',[],  'conv3_3',[] , 'conv3_4',[], 'conv4_1',[], 'conv4_2',[], 'conv4_3',[],  'conv4_4', [],'conv5_1',[],'conv5_2', [], 'conv5_3', [], 'conv5_4',[],  'fc6', [], 'fc7',[], 'fc8',[]);
%responses_prediction = struct('conv1_1',[], 'conv1_2',[], 'conv2_1',[], 'conv2_2',[],'conv3_1', [], 'conv3_2',[],  'conv3_3',[] , 'conv3_4',[], 'conv4_1',[], 'conv4_2',[], 'conv4_3',[],  'conv4_4', [],'conv5_1',[],'conv5_2', [], 'conv5_3', [], 'conv5_4',[],  'fc6', [], 'fc7',[], 'fc8',[]);
responses_weights = zeros(size(features')) ;
b = size(training_response);
for i = 1:b(1)
    feat_train = features;
    feat_test = test_feat;
    %for k = 1:1750
     %   name = fields{i};
      %  training_feature = features.(fields{i});
     %   feat_train(k,:) = training_feature;
    %end
    %for l = 1:120
     %   test_feature = test_feat.(fields{i});
      %  feat_test(l,:) = test_feature;
    %end
    response_model = train_response_model(response_model, feat_train, training_response(:,i));
    responses_weights(i,:) = response_model.Beta_hat';
%% Simulate response model

    training_response_hat = simulate_response_model(response_model, feat_train);
    %responses_prediction = training_response_hat;
    test_response_hat     = simulate_response_model(response_model, feat_test);
    
    cost = (sum(test_response - test_response_hat) .^2);
    disp(['regression cost:' num2str(cost) ])
end
%% Analyze encoding performance





