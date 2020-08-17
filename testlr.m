clc;clear;

%% initializing data
%{
M = readmatrix('titanic.csv');
features = M(:,[2 5:8]);
features = zscore(features); % normalizing variables
targets = M(:,1);
features = [ones(length(features),1) features]; %prepend a 1 for bias
%}

%{
M = readmatrix('iris.txt');
M(1:50,5) = 0;
M(51:100,5) = 1;
M(101:150,5) = 1;
features = M(:,1:4);
%features = zscore(features);
targets = M(:,5);
%}
load('diabetes.mat');
M = diabetes1;
disp(size(M));
features = M(:, 1:8);
targets = M(:,9);
features = zscore(features); % normalizing variables
features = [ones(length(features),1) features]; 
weights = zeros(size(features,2),1)+.05; % init weights to be zero
lr = 0.01;

%% predictions
predictions = sigmoid(features*weights);

%{
%% loss function
% calculating average loss per sample
m = size(features,1);
loss = -(1/m)*sum(targets.*log(predictions)+(1-targets).*log(1-predictions));
%% batch gradient
%The mini-batch gradient is the average of the individual gradients
m=size(features,1);
predictions = sigmoid(features*weights);
loss = -(1/m)*sum(targets.*log(predictions)+(1-targets).*log(1-predictions)); % beware overflow
batch_gradient = (features'*(sigmoid(features*weights)-targets))/m;
%weights = weights - lr*batch_gradient; % enact momentum here, or regularization coefficient

%% stochastic gradient vs batch vs minibatch
% they are all similar, can be calculated and updated by feeding proper
% parameters into equation
stochastic_gradient = (features(1,:)'*(sigmoid(features(1,:)*weights)-targets(1)))/1;
%weights = weights - lr*stochastic_gradient;

mini_batch_gradient = (features(1:10,:)'*(sigmoid(features(1:10,:)*weights)-targets(1:10)))/10;
%weights = weights - lr*mini_batch_gradient;
% we can randomize the samples we take for mini-batch gradient to get a
% more generalized descent
%}
%% performing 10 epochs of stochastic gradient descent

for i = 1:5000
    disp(['Epoch: ' num2str(i)]);
    m = size(features,1);
    lr = .01;
    for j = 1:m
        stochastic_gradient = (features(j,:)'*(sigmoid(features(j,:)*weights)-targets(j)));
        weights = weights - lr*stochastic_gradient;
    end
    loss_history(i) = -(1/m)*sum(targets.*log(sigmoid(features*weights)) ...
                        +(1-targets).*log(1-sigmoid(features*weights)));
end
loss_history(end)
plot(loss_history)
disp(loss_history(end));
figure(1)
plot(loss_history)
figure(2)
predictions = sigmoid(features*weights)'
plot(predictions, '.')
predictions(predictions >= 0.5) = 1;
predictions(predictions < 0.5) = 0;
predictions;

%% performing 10 epochs of batch gradient descent
%{
weights = zeros(size(features,2),1)+.05;
loss_history = [];
for i = 1:10000
    %disp(['Epoch: ' num2str(i)]);
    m = size(features,1);
    lr = .01;
    batch_gradient = (features'*(sigmoid(features*weights)-targets))/m;
    weights = weights - lr*batch_gradient;
    loss_history(i) = -(1/m)*sum(targets.*log(sigmoid(features*weights)) ...
                        +(1-targets).*log(1-sigmoid(features*weights)));
end
loss_history(end)
plot(loss_history)
disp(loss_history(end));
figure(1)
plot(loss_history)
figure(2)
predictions = sigmoid(features*weights)'
plot(predictions, '.')
predictions(predictions >= 0.5) = 1;
predictions(predictions < 0.5) = 0;
predictions;
%}

count = 0;
for i = 1:768
    if predictions(i) ~= targets(i)
        count = count + 1;
    end
end
disp(count);
disp(count/768);
