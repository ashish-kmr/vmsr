basefolder='/media/drive0/sgupta/output/rl/final-runs-combined/'

include='scratch_hr,imnet_hr,diayn,ours_hr_45K_ms'
python plotting_code/plot_rl.py --basepath "$basefolder"SemanticTask-bs4_sz8_o12_80_100_16_n0x05_10_10_1_100_sparse_area4-v0 --savepath plotting_code/plots/"$savestr"_hrl/ --ylim -5.5,1 --include $include --title "Area Goal with Sparse Rewards" \
  --legend $legend --ytext $ytext --xtext $xtext

python plotting_code/plot_rl.py --basepath "$basefolder"SemanticTask-bs4_sz8_o12_80_100_16_n0x05_10_10_1_100_dense2_area4-v0 --savepath plotting_code/plots/"$savestr"_hrl/ --ylim -4.0,1.0 --include $include --title "Area Goal with Dense Rewards" \
  --legend $legend --ytext $ytext --xtext $xtext

include='scratch,scratch_imnet,curiosity,ours_1op_ms'
python plotting_code/plot_rl.py --basepath "$basefolder"SemanticTask-bs4_sz8_o12_80_100_16_n0x05_10_10_1_100_sparse_area4-v0 --savepath plotting_code/plots/"$savestr"_oneop/ --ylim -5.5,1 --include $include --title "Area Goal with Sparse Rewards" \
  --legend $legend --ytext $ytext --xtext $xtext

python plotting_code/plot_rl.py --basepath "$basefolder"SemanticTask-bs4_sz8_o12_80_100_16_n0x05_10_10_1_100_dense2_area4-v0 --savepath plotting_code/plots/"$savestr"_oneop/ --ylim -4.0,1.0 --include $include --title "Area Goal with Dense Rewards" \
  --legend $legend --ytext $ytext --xtext $xtext

#include='ours_hr_45K_ms,diayn,imnet_hr,scratch_hr,scratch'
include='scratch_hr,imnet_hr,diayn,ours_hr_45K_ms'
python plotting_code/plot_rl.py --basepath "$basefolder"GoToPos-bs4_sz8_o12_50_60_16_n0x05_10_10_1_60_dense2_area4-v0 --savepath plotting_code/plots/"$savestr"_hrl/ --ylim -3,1 --include $include --title "Point Navigation with Dense Reward" \
  --legend $legend --ytext $ytext --xtext $xtext

python plotting_code/plot_rl.py --basepath "$basefolder"GoToPos-bs4_sz8_o12_50_60_16_n0x05_10_10_1_60_sparse_area4-v0 --savepath plotting_code/plots/"$savestr"_hrl/ --ylim -3.5,1 --include $include --title "Point Navigation with Sparse Reward" \
  --legend $legend --ytext $ytext --xtext $xtext
