# Segmentation and Classification of Surgical Gestures using Joint Learning of Spatial and Temporal Information
MATLAB R2015a and Python 3.4 (Anaconda 3.4)
Each experiment was run on the JHU foo.cis.jhu.edu server.

## Instructions to run the joint learning algorithm
1. Set up JIGSAWS. You can download it from here: http://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/

2. Set up the SDSDL code [by Shahin Sefati]. Follow these steps:
  1. Download the SDSDL code and add this folder's contents to joint_learning_REU2016/SDSDL/
  2. Download the Matlab interface of SPAMS from here: http://spams-devel.gforge.inria.fr/downloads.html
    1. I used SPAMS v2.5.
  3. In the scripts in iv - viii , you have to modify the address for the JIGSAWS dataset and the SPAMS mex files to point to your local files otherwise you get errors.
  4. Run "script_pca_sdl_LOSO.m" and "script_pca_sdl_LOUO.m" to:
    * read the raw data, 
    * standardize the data (subtract the mean and divide by the standard deviation),
    * apply PCA, 
    * run sparse dictionary learning,
    * save results (to "pca-sdl" folder)
    * These two scripts call other .m files including: compute_pca_sdl_LOSO.m, compute_pca_sdl_LOUO.m, read_data.m, standardize_data.m, pca_identification.m, testdata_pca_sparsecoding.m
  5. Run "script_svm_LOSO.m" and "script_svm_LOUO.m" to:
    * load data from previous step
    * train linear svm using stochastic gradient descent
    * save results (to "lin-svm" folder)
    * These two scripts call other .m files including: compute_svm_LOSO.m, compute_svm_LOUO.m, sgd_linSVM.m, linSVM_predict.m, downsampleTrainingData.m
  6. Run "script_supervised_DL_LOSO.m: and "script_supervised_DL_LOUO.m" to:
    * load the initialization data for supervised dictionary learning and save (to "initialization" folder)
    * run supervised dictionary learning to jointly learn the classifier and the dictionary
    * save results (to "supervised" folder)
    * These two scripts call other .m files including: load_pca_sdl_linsvm_save_LOSO.m, load_pca_sdl_linsvm_save_LOUO.m, compute_supervised_DL_LOSO.m, compute_supervised_DL_LOUO.m, sgd_supervisedDL_linSVM.m, compute_feature.m, diffDic.m, sumpool_dup.m, downsampleTrainingData.m 
  7. Run "script_viterbi_LOSO.m" and "script_viterbi_LOUO.m" to:
    * load the PCA of the data, dictionary and sparse codes, and jointly learned classifier and dictionary (load data from "pca-sdl" and "supervised" folders)
    * compute frame by frame predictions after applying Viterbi algorithm
    * save the final predictions (save to "final-results" folder)
    * These two scripts call other .m files including: compute_feature.m, compute_label.m, compute_viterbi_LOSO.m, compute_viterbi_LOUO.m, tranistion_probs.m, viterbi_path_log.m, saveAllResults.m
  8. Run "compute_micro_accuracy.m" to:
    * load results from "final-results" folder
    * compute the micro averages

3. Setup the LCTM code by running ``` python setup.py build ```
4. Extract the sparse codes into the sparse-z/ directory 
  1. Under SDSDL, create a new folder titled sparse-z. 
  2. Edit the in_path_prefix and out_path_prefix variables in extract_sparse.m
  3. Run extract_sparse.m to populate the sparse-z directory.
  
5. Run joint learning (finally!)
  1. Check paths to SPAMS and SDSDL in sgd_dict_par.m
  2. Check the paths to the training data, testing data, and the initialization data in ```joint_learning.py```
  3. Specify the task (Suturing, Knot_Tying, Needle_Passing) in the path, the experimental setup (LOSO or LOUO), and which user/supertrial is being left out. Specify the parameters ```n_psi_updates```, ```n_wp_iter```, ```skip_chain```, ```c```.
  4. Create a directory with the name ```joint_learning``` in the same directory as the ```joint_learning.py``` script
  5. Run the ```joint_learning.py``` script
  6. The objective function plot and the test accuracy plot will be saved into the joint_learning directory. 
  
  
  


