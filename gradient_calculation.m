% Separate function to calculate the gradient of one training example with respect to the dictionary. Called in a parfor loop in sgd_dict_par.m
% Returns the gradient of the objective function with respect to one training example specified by index t

function gradPsi_temp = gradient_calculation(t, Uhat_copy, half_win, w, y_train, y_pred, psi_n, psi_m, window_size, Dhat, psi)

[~, Uhat_temp] = sumpool_dup(Uhat_copy(:, t-half_win:t+half_win));
diff_feature = -w(:,y_train(t)) + w(:,y_pred(t));
gradPsi_temp = zeros(psi_n,psi_m);
for j=1:window_size
    [diff_psi,activeset] = diffDic(Dhat(:,t-half_win-1+j),psi,Uhat_temp(:,j),diff_feature);
     gradPsi_temp(:,activeset) = gradPsi_temp(:,activeset) + diff_psi;
end
gradPsi_temp = gradPsi_temp/window_size;



