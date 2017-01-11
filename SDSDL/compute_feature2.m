function [features,labels] = compute_feature(Uhat,window_size,G,no_frames,delay)

[m,n] = size(G);

% alter labels sequence length using delay
for i=1:m
    for j=1:n
        if ~isempty(G{i,j})
%         labels = [labels,G{i,j}(1:end-1)];
        G{i,j} = G{i,j}(1:end-delay);

        end
    end
end

labels = G;

% temporal window for sum-pooling
template = ones(1,window_size);

%% sum-pooling of sparse codes (after duplicating the sparse codes to positive and negative components)

Uhat_pos = +full(Uhat>0);
Uhat_neg = +full(Uhat<0);

% duplicating the sparse codes to positive and negative components
Uhat_pos = full(Uhat_pos.*Uhat);

Uhat_neg = full(Uhat_neg.*Uhat); % negative components keep their sign (FEATURE III.B)

hist_columns_pos = [];
hist_columns_neg = [];
for i=1:numel(no_frames)
    hist_columns_pos = [hist_columns_pos,conv2(Uhat_pos(:,1+(sum(no_frames(1:i-1))-(i-1)*delay):sum(no_frames(1:i))-i*delay),template,'same')];
    hist_columns_neg = [hist_columns_neg,conv2(Uhat_neg(:,1+(sum(no_frames(1:i-1))-(i-1)*delay):sum(no_frames(1:i))-i*delay),template,'same')];
end

hist_columns_signs = [hist_columns_pos;hist_columns_neg];

features = hist_columns_signs;

