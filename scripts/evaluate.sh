export num_orients=1
export unroll_length=80
export num_inits=5
export num_runs=100 
export randor=1

export expt_name=10_6,12,18_77000__32_4_-1_9,12,15,18_30,-20_80,-40_RN5N_train1_40.4,10,16_10,15,20_30_8_30,-10_60,-30_F_1500

CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. python pytorch_code/evaluate.py \
  --logdir_prefix output/mp3d/ \
  --expt_name $expt_name \
  --test_env area4 \
  --snapshot 000 \
  --num_operators 4\
  --stable_mdt True \
  --randor $randor \
  --unroll_length $unroll_length --num_inits $num_inits --num_orients $num_orients --num_runs $num_runs
