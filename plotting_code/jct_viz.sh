expt=/media/drive0/sgupta/output/mp3d/operators_invmodel_lstm_models_sn5/10_6,12,18_77000__32_4_-1_9,12,15,18_30,-20_80,-40_RN5N__40.4,10,16_10,15,20_30_8_30,-10_60,-30_F_1500/
run=68000

#expt=/media/drive0/sgupta/output/mp3d/operators_invmodel_lstm_models_sn5/20_6,12,18_100000__32_4_-1_9,12,15,18_30,-20_80,-40_RN5N/
#run=20000
#PT=mxop1
PT=entropy_th
#entropy_th
#two_way
CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. python plotting_code/jct_viz.py \
  --expt_args $expt\
  --run_no $run\
  --test_type area4\
  --path_length 60\
  --plot_type $PT
#  --save_img h0.99
#'h0.8'
