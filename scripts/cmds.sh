#CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. python pytorch_code/operator_invmodel_lstm_forward.py --expt_args _8,10,12,16_80000__32__-1_9,12,15,18_30,-20_50,-50_RN5N --num_workers 12
NUMOPS="4"
PATHLEN="10"
CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. python pytorch_code/train_vmsr.py \
  --expt_args "$PATHLEN"_6,12,18_77000__32_"$NUMOPS"_-1_9,12,15,18_30,-20_80,-40_RN5N_train1_40.4,10,16_10,15,20_30_8_30,-10_60,-30_F_1500 \
  --inv_base /media/drive0/sgupta/output/mp3d/invmodels/ \
  --num_workers 0
