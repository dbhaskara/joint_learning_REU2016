%%%%% After joint learning the dictionary and classifier, extract the sparse codes of the JIGSAWS data and save them in the folder sparse-z

in_path_prefix = '~/SDSDL/pca-sdl/';
out_path_prefix = '~/SDSDL/sparse-z/';

% task: {task name, pca/lasso parameters LOUO, pca/lasso paratmeters LOSO, window size LOUO, window size LOSO}
task = {    {'Suturing', 'n-35_m-200_lambda-lasso-0.1_delay-1/', 'n-35_m-100_lambda-lasso-0.1_delay-1/', 71, 81}, 
            {'Needle_Passing', 'n-35_m-150_lambda-lasso-0.1_delay-1/', 'n-35_m-100_lambda-lasso-0.05_delay-1/', 51, 61},
            {'Knot_Tying', 'n-35_m-150_lambda-lasso-0.05_delay-1/', 'n-35_m-150_lambda-lasso-0.05_delay-1/', 81, 71}};

for setup = {'LOUO', 'LOSO'} 
    for t=1:3 % loop over the 3 tasks
        if char(setup) == 'LOUO'
            in_path = strcat(in_path_prefix, setup, '/', task{t}{1}, '/', task{t}{2});
            leaveout = {'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'}; % users
            window_size = task{t}{4};
        else
            in_path = strcat(in_path_prefix, setup, '/', task{t}{1}, '/', task{t}{3});
            leaveout = {'1', '2', '3', '4', '5'}; % supertrials
            window_size = task{t}{5};
        end    
        disp(in_path);
        for l = leaveout
            
            % needle passing user G doesn't exist
            if (t == 2) && (char(l) == 'G') 
                continue; 
            end

            path = strcat(in_path, setup, '-', l, '.mat');
            disp(path);
            load(char(path));

            [X_train, y_train] = compute_feature2(Uhat, window_size, G_train, no_frames_per_trial_train, 1);
            [X_test, y_test] = compute_feature2(Uhat_test, window_size, G_test, no_frames_per_trial_test, 1);
           
            X_train = X_train/window_size;
            X_test = X_test/window_size;

            filename = char(strcat(setup, '-', l));

            disp(filename);

            save (char(strcat(out_path_prefix, '/', setup, '/', task{t}{1}, '/', filename)), 'X_train', 'y_train')
            save (char(strcat(out_path_prefix, '/', setup, '/', task{t}{1}, '/', filename, '-test')), 'X_test', 'y_test');

        end 
    end
end

