eps='1,7,13,18,28,44,47,54'

CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. python plotting_code/sr_consistency.py \
  --basepath /media/drive0/sgupta/output/mp3d/operators_invmodel_lstm_models_sn5/ \
  --test_type area4\
  --expt_name 10_6,12,18_77000__32_4_-1_9,12,15,18_30,-20_80,-40_RN5N__40.4,10,16_10,15,20_30_8_30,-10_60,-30_F_1500\
  --run_name 68000_100_1 \
  --num_ops 4\
  --eps $eps\
  --suffix 'sel'

CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. python plotting_code/sr_consistency.py \
  --basepath /media/drive0/sgupta/output/mp3d/operators_invmodel_lstm_models_sn5/ \
  --test_type area4\
  --expt_name 10_6,12,18_77000__32_4_-1_9,12,15,18_30,-20_80,-40_RN5N__40.4,10,16_10,15,20_30_8_30,-10_60,-30_F_1500\
  --run_name 68000_100_1 \
  --num_ops 4\
  --eps $eps\
  --legend 'no'\
  --suffix 'sel'
