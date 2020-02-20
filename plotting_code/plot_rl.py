#import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt
from absl import flags, app
from src import utils
import matplotlib

FLAGS = flags.FLAGS
flags.DEFINE_string('basepath','','base directory for the experiment')
flags.DEFINE_string('savepath','','base directory for the experiment')
flags.DEFINE_string('window', '8000', 'x axis info')
flags.DEFINE_string('ylim', '-1,1', 'x axis info')
flags.DEFINE_string('xlim', 'None', 'x axis info')
flags.DEFINE_string('include', '', 'Values to ignore')
flags.DEFINE_string('title', '', 'Plot title')
flags.DEFINE_string('legend', 'Y', 'show legend')
flags.DEFINE_string('ytext', 'Y', 'y axis label')
flags.DEFINE_string('xtext', 'Y', 'y axis label')

color_dict_i = {'diayn':'purple', 'scratch_hr':'red', 'ours_hr_3M':'cyan', 'curiosity':'orange', 'scratch':'red', \
        'ours_hr_45K_ms':'blue', 'ours_hr_90K':'dodgerblue', 'imnet_hr':'green', \
        'ours_1op':'skyblue', 'ours_hr_45K_ms_imnet':'dodgerblue', 'ours_hr_45K_ms__3_max': 'blue',\
        'ours_hr_45K_ms__3_prob':'k', 'ours_1op_ms':'blue', 'scratch_imnet':'green', 'ours_inv_imnet': 'purple'} 
#label_dict = {'diayn':'HRL (DIAYN)', 'scratch_hr':'HRL (Random Init)', \
#        'ours_hr_3M':'cyan', 'curiosity':'RL (Curiosity)', 'scratch':'RL (Random Init)', \
#        'ours_hr_45K_ms':'HRL (Our 4 Subroutines)', 'ours_hr_90K':'dodgerblue', \
#        'imnet_hr':'HRL (ImageNet Init)', 'ours_1op':'RL (Our 1 Subroutine)', 'ours_hr_45K_ms_imnet':'ImageNet Affordance',
#        'ours_hr_45K_ms__3_max': 'Retrained Affordance', 'ours_hr_45K_ms__3_prob':'Retrained Affordance (max)',
#        'ours_1op_ms':'RL (Our 1 Subroutine)'} 
label_dict = {'diayn':'DIAYN', 'scratch_hr':'Random Init', \
        'ours_hr_3M':'cyan', 'curiosity':'Curiosity', 'scratch':'Random Init', \
        'ours_hr_45K_ms':'VMSR (4 SubRs) [Ours]', 'ours_hr_90K':'dodgerblue', \
        'imnet_hr':'ImageNet Init', 'ours_1op':'Our 1 Subroutine', 'ours_hr_45K_ms_imnet':'ImageNet Affordance',
        'ours_hr_45K_ms__3_max': 'Retrained Affordance', 'ours_hr_45K_ms__3_prob':'Retrained Affordance (max)',
        'ours_1op_ms':'VMSR (1 SubR) [Ours]', 'scratch_imnet':'ImageNet Init', 'ours_inv_imnet': 'Inverse Features'} 
color_dict = {}
for k in color_dict_i.keys():
    color_dict[label_dict[k]] = color_dict_i[k]

#base = '/media/drive0/sgupta/output/rl/final-runs-v1/SemanticTask-bs4_sz8_o12_80_100_16_n0x05_10_10_1_100_dense2_area4-v0'


def find_events(pth):
    curr_n = ''
    for fname in os.listdir(pth):
        match_str = 'events.out.tfevents'
        if fname[0:len(match_str)] == match_str: 
            if os.path.getsize(pth+'/'+fname)>200 or curr_n =='':
                curr_n = fname
    return curr_n

def pivot_data(x, y, tag, xpivots, window):
    xval = xpivots
    y_accm = [[] for itr in range(len(xpivots))]
    for n,itr in enumerate(x):
        idx = int(round(itr/window))
        if idx >= len(y_accm): break
        y_accm[idx].append(y[n])
    x_, y_ = [], []
    for n, itr in enumerate(y_accm):
        if len(itr)>0:
            x_.append(xval[n])
            y_.append(np.mean(itr))
    return x_, y_, tag[0:len(x_)]

def extract_info(pth, mname):
    x, y, tag = [], [], []
    for e in tf.train.summary_iterator(pth):
        for v in e.summary.value:
            if v.tag == 'episode_reward':
                x.append(e.step)
                y.append(v.simple_value)
                tag.append(label_dict[mname])
    return x, y, tag

def smoothed_estimate(data_x, window = 1):
    nlim = len(data_x)
    #window = max(1, int(alpha*nlim))

    for itr in range(nlim):
        init_i = max(itr - window//2, 0)
        end_i = min(itr + window//2 + 1, nlim)
        data_x[itr] = np.mean(data_x[init_i:end_i])
    return data_x

def normalize_y(y):
    min_y = np.min(y)
    max_y = np.max(y)
    return (y - min_y)/(max_y - min_y)

def flip_order(labels):
    target = FLAGS.include.split(',')
    orderlist = [labels.index(label_dict[tgt]) for tgt in target]
    return orderlist
    
def make_plot(ev_dict, xlabel, ylabel, xlim):
    ev_dict['x'] = np.array(ev_dict['x'])
    ev_dict['y'] = np.array(ev_dict['y'])
    ev_dict['y'] = normalize_y(ev_dict['y'])
    sns.lineplot(x='x', y='y', hue = 'tag', data = ev_dict, \
            ci = 'sd', sort = True, palette = color_dict)
    ylim = [float(itr) for itr in FLAGS.ylim.split(',')]
    #plt.ylim(ylim)
    plt.ylim((-0.1,1.1))
    plt.xlim(xlim)
    if FLAGS.xtext == 'Y':
        plt.xlabel(xlabel, fontsize = 25)
    if FLAGS.ytext == 'Y':
        plt.ylabel(ylabel, fontsize = 25)
    plt.title(FLAGS.title, fontsize = 25)
    ax = plt.gca()
    ax.tick_params(labelsize = 15)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    empty_string_labels = ['']*len(labels)
    if FLAGS.ytext == 'N':
        ax.set_yticklabels(empty_string_labels)
    ax.legend(fontsize = 30)#, loc = 4)
    handles,labels = ax.get_legend_handles_labels()
    forder = flip_order(labels)
    handles_n = [handles[idx_i] for idx_i in forder]
    labels_n = [labels[idx_i] for idx_i in forder]
    ax.legend(handles_n,labels_n, fontsize = 20)
    if FLAGS.legend == 'N':
        ax.get_legend().remove()
    utils.mkdir_if_missing(FLAGS.savepath)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(os.path.join(FLAGS.savepath, FLAGS.basepath.split('/')[-1] + '.pdf'), bbox_inches = 'tight', pad_inches = 0)
    plt.clf()

def accumulate_methods(base, window, xpivots):
    #ymin = float(FLAGS.ylim.split(',')[0])
    #ymax = float(FLAGS.ylim.split(',')[1])
    methods = os.listdir(base)
    suffix = 'a2c_rnn_Nav'
    ev_dict = {'x':[], 'y':[], 'tag':[]}
    for method_i in methods:
        mname = '_'.join(method_i.split('_')[:-1])
        if mname not in FLAGS.include.split(','): continue
        events_file = find_events(os.path.join(base, method_i, suffix))
        pth_event = os.path.join(base, method_i, suffix, events_file)
        x, y, tag = extract_info(pth_event, mname)
        x, y, tag = pivot_data(x, y, tag, xpivots, window)
        #sub_sampling_r = 20
        #x = x[0::sub_sampling_r]; y = y[0::sub_sampling_r]; tag = tag[0::sub_sampling_r]
        y = smoothed_estimate(y, 10)
        #y = list((np.array(y) - ymin)/(ymax-ymin))
        print("{0} \t {1:05d}".format(method_i, len(x)))
        ev_dict['x'] += x; ev_dict['y'] += y; ev_dict['tag'] += tag
    return ev_dict

def worker():
    print(FLAGS.basepath.split('/')[-1])
    window = int(FLAGS.window)
    xpivots = range(0,2500000,window)
    ev_dict = accumulate_methods(FLAGS.basepath, window, xpivots)
    xlabel = '# Env Interactions (x 100K)'
    ylabel = 'Episode Reward'
    ev_dict['x'] = np.array(ev_dict['x'])/100000.0
    flags_xlim = FLAGS.xlim
    if flags_xlim == 'None':
        flags_xlim = np.max(ev_dict['x'])
    xlim = (-1,int(flags_xlim))
    make_plot(ev_dict, xlabel, ylabel, xlim)


def main(_):
    worker()

if __name__ == '__main__':
    app.run(main)
