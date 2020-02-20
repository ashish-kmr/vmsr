echo 'ns'
area='area3'

for xlab in xyes xno
do
  for ylab in yyes yno
  do
savepth='plotting_code/plots/online_data_ablations/'
#python plotting_code/plot_metrics.py --basepath ablation_files/numops.txt \
#  --prefix /media/drive0/sgupta/output/mp3d/operators_invmodel_lstm_models_sn5/ \
#  --xtype nsubroutines --xname 'Num Subroutines'\
#  --savepath $savepth\
#  --area $area \
#  --xlabel $xlab\
#  --ylabel $ylab
#
#echo 'is'
#python plotting_code/plot_metrics.py --basepath ablation_files/interaction_samples.txt \
#  --prefix /media/drive0/sgupta/output/mp3d/operators_invmodel_lstm_models_sn5/ \
#  --xtype interaction_samples --xname 'Interaction Samples (Visual Diversity)'\
#  --savepath $savepth\
#  --area $area\
#  --xlabel $xlab\
#  --ylabel $ylab
#
#echo 'pl'
#python plotting_code/plot_metrics.py --basepath ablation_files/pathlen.txt \
#  --prefix /media/drive0/sgupta/output/mp3d/operators_invmodel_lstm_models_sn5/ \
#  --xtype pathlen --xname 'Path Length'\
#  --savepath $savepth\
#  --area $area\
#  --xlabel $xlab\
#  --ylabel $ylab
#
#echo 'ss'
#python plotting_code/plot_metrics.py --basepath ablation_files/ss_samples.txt \
#  --prefix /media/drive0/sgupta/output/mp3d/operators_invmodel_lstm_models_sn5/ \
#  --xtype sssteps --xname 'Interaction Samples (Episode Length)'\
#  --savepath $savepth\
#  --area $area\
#  --xlabel $xlab\
#  --ylabel $ylab

#echo 'online data'
#python plotting_code/plot_metrics.py --basepath ablation_files/data_comparison.txt \
#  --prefix /media/drive0/sgupta/output/mp3d/operators_invmodel_lstm_models_sn5/ \
#  --xtype onlinedata --xname 'Num Videos ( x 1K)'\
#  --savepath $savepth\
#  --area $area\
#  --xlabel $xlab\
#  --ylabel $ylab

#echo gt_comparison
#python plotting_code/plot_metrics.py --basepath ablation_files/gt_comparison.txt \
#  --prefix /media/drive0/sgupta/output/mp3d/operators_invmodel_lstm_models_sn5/ \
#  --xtype gtdata --xname 'Num Videos ( x 1K)'\
#  --savepath $savepth\
#  --area $area\
#  --xlabel $xlab\
#  --ylabel $ylab

echo combined_comparison
python plotting_code/plot_metrics.py --basepath ablation_files/gt_comparison.txt,ablation_files/data_comparison.txt \
  --prefix /media/drive0/sgupta/output/mp3d/operators_invmodel_lstm_models_sn5/ \
  --xtype combined_gt_inv --xname 'Num Videos ( x 1K)'\
  --savepath $savepth\
  --area $area\
  --xlabel $xlab\
  --ylabel $ylab



done
done
