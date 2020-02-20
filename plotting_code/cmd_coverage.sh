CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. python plotting_code/stitched.py \
  --basepath /media/drive0/sgupta/output/mp3d/operators_invmodel_lstm_models_sn5/ \
  --test_type area4\
  --expt_name 10_6,12,18_77000__32_4_-1_9,12,15,18_30,-20_80,-40_RN5N__40.4,10,16_10,15,20_30_8_30,-10_60,-30_F_1500\
  --run_name n0100_inits05_or01_unroll408_rinit1.0000068000_009_009 \
  --color 'b-' \
  --label 'Ours'&  


bsmethod=stat
CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. python plotting_code/stitched.py \
  --basepath output/mp3d/operators_invmodel_lstm_models_sn5/ \
  --test_type area4 \
  --expt_name $bsmethod\
  --run_name n0100_inits04_or05_unroll080.$bsmethod \
  --color 'k-'\
  --label 'Forward Bias Policy' &


bsmethod=random
CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. python plotting_code/stitched.py \
  --basepath output/mp3d/operators_invmodel_lstm_models_sn5/ \
  --test_type area4\
  --expt_name $bsmethod\
  --run_name n0100_inits04_or05_unroll080.$bsmethod \
  --color 'c-'\
  --label 'Random' &

bsmethod=collav
CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. python plotting_code/stitched.py \
  --basepath output/mp3d/operators_invmodel_lstm_models_sn5/ \
  --test_type area4\
  --expt_name $bsmethod\
  --run_name n0100_inits04_or05_unroll080.$bsmethod \
  --color 'r-'\
  --label 'Always Forward, Rotate on Collision' &

bsmethod=outputs-curiosity-v5-Curiosity-bs8_sz8_o12_0_16_n16_n0x05_10_10_1_200_dense2_traininv.l9_fc_lr1en3_bnfix1-v0-a2c_resnet18_rnn_bnfreeze
CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. python plotting_code/stitched.py \
  --basepath /media/drive0/sgupta/rl-explr-results/ \
  --test_type area4\
  --expt_name $bsmethod\
  --run_name n0100_inits04_or05_unroll080.0001024320_000_999 \
  --color 'g-'\
  --label 'Skills from Curiosity' &

bsmethod=outputs-diayn-v3-DIAYN-bs8_sz8_o12_0_16_n16_n0x05_10_10_1_10_dense2_traininv.l9_fc_lr1en3_frz0_n4-v0-a2c_resnet18_rnn_ent1.00e-01
CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. python plotting_code/stitched.py \
  --basepath /media/drive0/sgupta/rl-explr-results/ \
  --test_type area4\
  --expt_name $bsmethod\
  --run_name n0100_inits04_or05_unroll080.0001024080_009_009 \
  --color 'm-'\
  --label 'Skills from Diversity'

