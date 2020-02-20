## vmsr for POINTGAOL task with SPARSE rewards
CUDA_VISIBLE_DEVICES='0' PYOPENGL_PLATFORM=egl PYTHONPATH='.' python  rl/pytorch-a2c-ppo-acktr/main.py --env-name GoToPos-bs4_sz8_o12_50_60_16_n0x05_10_10_1_60_sparse_area4-v0 --gamma 0.99 --num-processes 1 --log-dir-prefix output/rl/final-runs-v2/ --log-dir-suffix _rnn_Nav --recurrent-policy --hidden-size 256 --log-interval 1 --init-policy ours_hr_45K_ms --num-steps 6 --policy-type Nav --meta_steps 10 --meta_actions 4 --seed 0 --base-model-folder output/mp3d/ 

## vmsr for POINTGAOL task with DENSE rewards
CUDA_VISIBLE_DEVICES='0' PYOPENGL_PLATFORM=egl PYTHONPATH='.' python  rl/pytorch-a2c-ppo-acktr/main.py --env-name GoToPos-bs4_sz8_o12_50_60_16_n0x05_10_10_1_60_dense2_area4-v0 --gamma 0.99 --num-processes 1 --log-dir-prefix output/rl/final-runs-v2/ --log-dir-suffix _rnn_Nav --recurrent-policy --hidden-size 256 --log-interval 1 --init-policy ours_hr_45K_ms --num-steps 6 --policy-type Nav --meta_steps 10 --meta_actions 4 --seed 0 --base-model-folder output/mp3d/ 

## vmsr for POINTGAOL task with SPARSE rewards
CUDA_VISIBLE_DEVICES='0' PYOPENGL_PLATFORM=egl PYTHONPATH='.' python  rl/pytorch-a2c-ppo-acktr/main.py --env-name SemanticTask-bs4_sz8_o12_80_100_16_n0x05_10_10_1_100_sparse_area4-v0 --gamma 0.99 --num-processes 1 --log-dir-prefix output/rl/final-runs-v2/ --log-dir-suffix _rnn_Nav --recurrent-policy --hidden-size 128 --log-interval 1 --init-policy ours_hr_45K_ms --num-steps 10 --policy-type Nav --meta_steps 10 --meta_actions 4 --seed 0 --base-model-folder output/mp3d/ 

## vmsr for POINTGAOL task with DENSE rewards
CUDA_VISIBLE_DEVICES='0' PYOPENGL_PLATFORM=egl PYTHONPATH='.' python  rl/pytorch-a2c-ppo-acktr/main.py --env-name SemanticTask-bs4_sz8_o12_80_100_16_n0x05_10_10_1_100_dense2_area4-v0 --gamma 0.99 --num-processes 1 --log-dir-prefix output/rl/final-runs-v2/ --log-dir-suffix _rnn_Nav --recurrent-policy --hidden-size 128 --log-interval 1 --init-policy ours_hr_45K_ms --num-steps 10 --policy-type Nav --meta_steps 10 --meta_actions 4 --seed 0 --base-model-folder output/mp3d/ 
