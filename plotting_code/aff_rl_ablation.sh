ignore='ours_hr_45K,ours_hr_3M,ours_hr_90K,scratch,diayn,scratch_hr,imnet_hr,ours_hr_45K_ms'
basefolder='/media/drive0/sgupta/output/rl/final-runs-combined/'
legend='Y'
ytext='Y'
savestr='aff_ablation'

python plotting_code/plot_rl.py --basepath "$basefolder"GoToPos-bs4_sz8_o12_50_60_16_n0x05_10_10_1_60_sparse_area4-v0 --savepath plotting_code/plots/"$savestr"/ --ylim -3.5,1 --ignore $ignore --title "Point Navigation Task with Sparse Reward" \
  --legend $legend --ytext $ytext --window 8000 --xlim 15
