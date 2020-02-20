NSTARTS="1500"
STEPS="30"
CUDA_VISIBLE_DEVICES='1' PYOPENGL_PLATFORM=egl PYTHONPATH=. python pytorch_code/train_inverse_model.py \
  --num_workers 4\
  --expt_args 4,10,16_10,15,20_"$STEPS"_8_30,-10_60,-30_F_$NSTARTS
