% Calculates the gradient of the objective function with respect to the dictionary psi, 
% updates the dictionary, and recalculates the sparse codes

function x = sgd_dict(initialization_data_filename, gradient_filename, prev_psi_update, cur_iter)

x = cur_iter;

setenv('MKL_NUM_THREADS','1')
setenv('MKL_SERIAL','YES')
setenv('MKL_DYNAMIC','NO')

addpath('~/spams-matlab/test_release');
addpath('~/spams-matlab/src_release');
addpath('~/spams-matlab/build');
addpath('/cis/home/divya/SDSDL');


load(initialization_data_filename);
load(gradient_filename);

if cur_iter ~= 1
    load(prev_psi_update, 'Uhat_new', 'psi');
    psi_init = psi;
    Uhats = Uhat_new;
else
    psi_init = initialization_data.psi_init;
    Uhats = initialization_data.Uhat; % pre-learned sparse codes
end    
    

w_init = initialization_data.w_init;

Dhats = initialization_data.Dhat; % training data
%Uhats = Uhat_prelearned;
Dhat_test = initialization_data.Dhat_test; % testing data
labels = initialization_data.labels;
idxs = initialization_data.idx; % indices of the downsampled training data
window_size = initialization_data.window_size;
G_train = initialization_data.G;
no_frames_per_trial_train = initialization_data.no_frames;
G_test = initialization_data.G_test;
no_frames_per_trial_test = initialization_data.no_frames_test;
delay = initialization_data.delay;
lambda_lasso = initialization_data.lambda_lasso;


half_win = (window_size-1)/2;

y_train_pred = y_star;
y_train_lst = y_gt;

%disp(size(y_train_pred));

% concatenate y_train_pred and y_train_lst sequences into one long sequence
y_train = []; % ground truth labels
y_pred = []; % predicted labels from the sccrf layer

[~, num_seq] = size(y_train_pred);
%disp(num_seq);

for i=1:num_seq
    y_train = [y_train, y_train_lst{i}];
    y_pred = [y_pred, y_train_pred{i}];
end

%disp(size(y_pred));

%y_train = labels;
% c = setdiff(y_train - labels)

%if isequal(y_train, labels)
    %disp('yes y_train == labels')
%end


%disp(size(y_train))

classes = w_init.labels; % class labels
no_classes = numel(classes);
classes_idx = 1:no_classes;

for i=1:no_classes
    labels(labels==classes(i))=classes_idx(i);
    y_train(y_train==classes(i))=classes_idx(i);
    y_pred(y_pred==classes(i))=classes_idx(i);
end

% isequal(y_train, labels)

if window_size == 1
    Uhat = Uhats;
    label = labels;
    Uhat_idx = 1:numel(label);
else
    % 'Uhat' is formed by padding zeros (half the window size) at the beginning and end of each trial in 'Uhats':
    Uhat = zeros(size(Uhats,1),half_win);
    Dhat = zeros(size(Dhats,1),half_win);
    
    % 'label' is formed by padding zeros (half the window size) at the beginning and end of each trial in 'labels':
    label = zeros(1,half_win);
    y_pred_temp = zeros(1, half_win);
    y_train_temp = zeros(1, half_win);

    Uhat_idx = [];
    for i=1:numel(no_frames_per_trial_train)
        Uhat = [Uhat,Uhats(:,1+(sum(no_frames_per_trial_train(1:i-1))-(i-1)*delay):sum(no_frames_per_trial_train(1:i))-i*delay),zeros(size(Uhats,1),half_win)];
        Dhat = [Dhat,Dhats(:,1+(sum(no_frames_per_trial_train(1:i-1))-(i-1)*delay):sum(no_frames_per_trial_train(1:i))-i*delay),zeros(size(Dhats,1),half_win)];
        
        label = [label,labels(:,1+(sum(no_frames_per_trial_train(1:i-1))-(i-1)*delay):sum(no_frames_per_trial_train(1:i))-i*delay),zeros(1,half_win)];
        y_pred_temp = [y_pred_temp,y_pred(:,1+(sum(no_frames_per_trial_train(1:i-1))-(i-1)*delay):sum(no_frames_per_trial_train(1:i))-i*delay),zeros(1,half_win)];
        y_train_temp = [y_train_temp,y_train(:,1+(sum(no_frames_per_trial_train(1:i-1))-(i-1)*delay):sum(no_frames_per_trial_train(1:i))-i*delay),zeros(1,half_win)];
        % the indices in Uhat that correspond to actual sparse codes in Uhats:
        Uhat_idx = [Uhat_idx,1+(sum(no_frames_per_trial_train(1:i-1))-(i-1)*delay)+(i*half_win):(sum(no_frames_per_trial_train(1:i))-i*delay)+(i*half_win)];
    end
end


y_train = y_train_temp;
y_pred = y_pred_temp;

%disp('new sizes');

%disp(size(y_train));
%disp(size(y_pred));

w = w_init.w;
% w = zeros(m,no_classes);

psi = psi_init;
[psi_n,psi_m] = size(psi);

%disp(size(psi))
%disp(size(w_init.w)) % 400, 10


Uhat_copy = Uhat;

param.K=psi_m;  % learns a dictionary with 100 elements
param.lambda=lambda_lasso;
param.numThreads=-1; % number of threads
param.verbose=false;
param.iter=1000;

% learning rates for the classifier and dictionary:
 alpha_w = 10^-2;
 alpha_psi = 10^-2;

% find the subgradient w.r.t. the dictionary psi

%disp(alpha_psi);
%disp(size(Uhat_idx));

sum_tempPsi = zeros(psi_n,psi_m);

 %Uhat_idx = Uhat_idx(1:6000);
 %Uhat_idx = datasample(Uhat_idx, 6000,'Replace',false);
 %disp(size(Uhat_idx));

 disp(['Finding the gradient with respect to the dictionary psi across ' num2str(numel(Uhat_idx)) 'training examples..']);
 
 parfor t_par = 1:length(Uhat_idx)
    %if mod(t, 1000) == 0
    %    disp(t);
    %end
    t = Uhat_idx(t_par);
    gradPsi_temp = gradient_calculation(t, Uhat_copy, half_win, w, y_train, y_pred, psi_n, psi_m, window_size, Dhat, psi)
    sum_tempPsi = sum_tempPsi + gradPsi_temp;
    
end
gradPsi = sum_tempPsi/numel(Uhat_idx);

%save('gradPsi', 'gradPsi');

%load('/cis/home/divya/LCTM/gradPsi.mat');

psi = psi - alpha_psi*(gradPsi);
psi = normc(psi);

% recompute sparse codes
Uhat_new = mexLasso(Dhats,psi,param);
Uhat_test_new = mexLasso(Dhat_test, psi, param);

% compute feature 2
% sumpooling and duplicate sparse codes
[X_train, y_train] = compute_feature2(Uhat_new, window_size, G_train, no_frames_per_trial_train, 1);
[X_test, y_test] = compute_feature2(Uhat_test_new, window_size, G_test, no_frames_per_trial_test, 1);

X_train = X_train/window_size;
X_test = X_test/window_size;

outfile = strcat('/cis/home/divya/LCTM/joint_learning/output-iter', int2str(cur_iter)); 

save (outfile, 'X_train', 'y_train', 'X_test', 'y_test', 'psi', 'Uhat_new');


