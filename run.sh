#! /bin/tcsh -f

# set INPUT_DATA_CSV="./ml_practice/Vp_small_var0_desc_norm.csv"
# set INPUT_VALUES_TXT="./ml_practice/Vp_small_norm_values.txt"
# set INPUT_DATA_CSV="./ml_practice/BHL_small_var0_desc_norm.csv"
# set INPUT_VALUES_TXT="./ml_practice/BHL_small_norm_values.txt"
set INPUT_DATA_CSV="./ml_practice/Fp_small_var0_desc_norm.csv"
set INPUT_VALUES_TXT="./ml_practice/Fp_small_norm_values.txt"
set LAMBDA="0.1"

python3 svm.py "${INPUT_DATA_CSV}" "${INPUT_VALUES_TXT}" "${LAMBDA}"
