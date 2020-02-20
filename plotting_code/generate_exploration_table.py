import os
import numpy as np
import json
base1 = '/media/drive0/sgupta/output/mp3d/operators_invmodel_lstm_models_sn5/' 
base2 = '/media/drive0/sgupta/rl-explr-results/' 

runstr = 'n0100_inits05_or01_unroll408_rinit1'


base = [base1, base1, base1, base2, base2, base1]
append = [\
        ('random','random/area4/' + runstr + '.random'), \
        ('stat','stat/area4/' + runstr + '.stat_0.660'),\
        ('collav','collav/area4/' + runstr + '.collav'), \
        ('diayn','outputs-diayn-v3-DIAYN-bs8_sz8_o12_0_16_n16_n0x05_10_10_1_10_dense2_traininv'+\
        '.l9_fc_lr1en3_frz0_n4-v0-a2c_resnet18_rnn_ent1.00e-01/area4/' + runstr + '.0001024080_009_009'), \
        ('curiosity','outputs-curiosity-v5-Curiosity-bs8_sz8_o12_0_16_n16_n0x05_10_10_1_200_dense2_traininv'+\
        '.l9_fc_lr1en3_bnfix1-v0-a2c_resnet18_rnn_bnfreeze/area4/' + runstr + '.0001024320_000_999'),\
        ('ours','10_6,12,18_77000__32_4_-1_9,12,15,18_30,-20_80,-40_RN5N__40.4,10,16_10,15,20_30_8_30,-10_60,-30_F_1500/area4/' \
        + runstr + '.0000068000_009_009'), \
        ]

mdt_list = []
max_dist = []
collision_rate = []
median_dist = []
max_max_dist = []

for i in range(len(append)):
    data_pth = os.path.join(base[i], append[i][1] + '.json')
    data_f = open(data_pth)
    data = json.load(data_f)
    mdt_list.append(data['mdt_new'])
    max_dist.append(data['max_dist'])
    max_max_dist.append(np.max(data['max_dist_list']))
    median_dist.append(data['max_dist_median'])
    collision_rate.append(data['collisions']*1.0/data['act_dist'][3])

print(runstr)
print('Method', 'ADT', 'Max_Dist','Median_Max_Dist', 'Max_Max_Dist', 'Collision_Rate')
for i in range(len(append)):
    print(append[i][0],np.round(mdt_list[i],2),np.round(max_dist[i],2),np.round(median_dist[i],2), \
            np.round(max_max_dist[i],2), np.round(collision_rate[i],2))
