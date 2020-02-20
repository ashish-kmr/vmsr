basefolder='/media/drive0/sgupta/output/rl/final-runs-combined/'

include='ours_inv_imnet,ours_hr_45K_ms'
python plotting_code/plot_rl.py --basepath "$basefolder"GoToPos-bs4_sz8_o12_50_60_16_n0x05_10_10_1_60_sparse_area4-v0 --savepath plotting_code/plots/invfeats_hrl/ --ylim -5.5,1 --include $include --title "Area Goal with Sparse Rewards" \
  --legend Y --ytext Y --xtext Y

