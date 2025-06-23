# #!/bin/bash
#cd eval
#case $1 in
#    "normal")
#        python3 pnv_evaluate.py --config ../config/config_test_1.txt --model_config ../models/minkloc3dv2.txt --weights ../weights/$2
#        ;;
#    "ensemble")
#        python3 pnv_evaluate_mahalanhobis.py --config ../config/config_test_1.txt --model_config ../models/minkloc3dv2.txt --weights ../weights/ensemble_weights/ --n_ensemble $2
#        ;;
#    "median")
#        python3 pnv_evaluate_median.py --config ../config/config_test_2.txt --model_config ../models/minkloc3dv2.txt --weights ../weights/ensemble_weights_ts/ --n_ensemble $2
#        ;;
#    "kl")
#        python3 pnv_evaluate_kl.py --config ../config/config_test_1.txt --model_config ../models/minkloc3dv2.txt --weights ../weights/ensemble_weights/ --n_ensemble $2
#        ;;
#    "val_var")
#        python3 pnv_evaluate_variance.py --config ../config/config_test_2.txt --model_config ../models/minkloc3dv2.txt --weights ../weights/ensemble_weights_ts/ --n_ensemble $2
#        ;;
#    "comparison")
#        python3 pnv_evaluate_variance_comparison.py --config ../config/config_test_1.txt --model_config ../models/minkloc3dv2.txt --weights ../weights/ensemble_weights/ --n_ensemble $2
#        ;;
#esac

python3 pnv_evaluate_der.py --config config/config_test_3.txt --model_config models/minkloc3dv2.txt --weights weights/model_MinkLocEvd_20240609_2106_final.pth
