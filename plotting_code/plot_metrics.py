from src import utils
#from pytorch_code.compute_metrics import compute_metrics_i
from absl import flags, app
import json
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

FLAGS = flags.FLAGS
flags.DEFINE_string('basepath','','base directory for the experiment')
flags.DEFINE_string('prefix','output/mp3d/operators_invmodel_lstm_models_sn5/','prefix to experiment folders')
flags.DEFINE_string('savepath','','base directory for the experiment')
flags.DEFINE_string('xname', None, 'x axis info')
flags.DEFINE_string('yname', None, 'x axis info')
flags.DEFINE_string('xtype', None, 'x axis info')
flags.DEFINE_string('area', 'area4', 'x axis info')
flags.DEFINE_string('xlabel', 'xyes', 'x axis info')
flags.DEFINE_string('ylabel', 'yyes', 'x axis info')


def _interaction_samples(folder_pth, ns, tag = None):
    jf = open(folder_pth)
    data = json.load(jf)
    nruns = data['num_runs']
    # new_mdt_list; max_dist_list
    ns_l = [ns for i in range(nruns)]
    max_dist_l = data['max_dist_list']
    adt_l = data['new_mdt_list']
    tag_l = [tag for i in range(nruns)]
    return {'x':ns_l, 'x_c':[ns], 'ADT': adt_l, \
            'Max Dist': max_dist_l, 'ADT Old': [data['mdt']],\
            #'Collisions': [1.0*data['collisions']/(data['act_dist'][3]*1.0)], \
            'Collisions': data['coll_frac_list'], \
            'Tag_c': [tag], 'Tag': tag_l}

def compress_dict(listofd):
    dkeys = listofd[0].keys()
    dict_ ={}
    for k in dkeys:
        acc_val = []
        for dct in listofd: acc_val += dct[k]
        dict_[k] = np.array(acc_val)
    return dict_

def gen_pth(l, addstr):
    fpth = l.split(':')[1]
    snap = int(l.split(':')[2])
    #pth = os.path.join(fpth, FLAGS.area, 'n0100_inits05_or01_unroll408_rinit1.{:010d}'.format(snap) + addstr)
    pth = os.path.join(fpth, FLAGS.area, 'n0100_inits05_or01_unroll274_rinit1.{:010d}'.format(snap) + addstr)
    return pth 

def process_files(basepath):
    addstr_base = '_009_009'
    addstr = addstr_base + '.json'
    #pth = os.path.join(fpth, 'area4', 'n0100_inits04_or05_unroll080.{:010d}_010_003.json'.format(snap))
    if FLAGS.xtype == 'interaction_samples':
        lines = []
        for l in open(basepath):  
            if l[0] =='#': continue
            ns = int(l.split(':')[0])#/1000
            pth = gen_pth(l, addstr)
            lines.append(_interaction_samples(os.path.join(FLAGS.prefix, pth), ns))
        cdict = compress_dict(lines)
        cdict['xaxis'] = FLAGS.xname
        return cdict

    if FLAGS.xtype == 'nsubroutines':
        lines = []
        for l in open(basepath):  
            if l[0] =='#': continue
            ns = int(l.split(':')[0])
            pth = gen_pth(l, addstr)
            lines.append(_interaction_samples(os.path.join(FLAGS.prefix, pth), ns, tag = 'With Affordance Model'))
            #lines.append(_interaction_samples(os.path.join(FLAGS.prefix, pth), ns, tag = None))
            #pfix_list = ['random', 'affth']
            #tag_list = ['Randomly Sampling Subroutines', 'With Affordance Model Thresholded']
            #pfix_list = ['random', 'max_random', 'max_affth_0.001','affth_0.001','max']
            #tag_list = ['random', 'max_random', 'max_affth_0.001','affth_0.001','max']
            pfix_list = ['random']
            tag_list = ['random']
            for pfix, tag in zip(pfix_list, tag_list):
                pth = gen_pth(l, addstr_base + '_' + pfix + '.json')
                lines.append(_interaction_samples(os.path.join(FLAGS.prefix, pth), ns, tag = tag))
        cdict = compress_dict(lines)
        cdict['xaxis'] = FLAGS.xname
        return cdict

    if FLAGS.xtype == 'pathlen':
        lines = []
        for l in open(basepath):  
            if l[0] =='#': continue
            ns = int(l.split(':')[0])
            pth = gen_pth(l, addstr)
            lines.append(_interaction_samples(os.path.join(FLAGS.prefix, pth), ns))
        cdict = compress_dict(lines)
        cdict['xaxis'] = FLAGS.xname
        return cdict

    if FLAGS.xtype == 'sssteps':
        lines = []
        for l in open(basepath):  
            if l[0] =='#': continue
            ns = int(l.split(':')[0])
            pth = gen_pth(l, addstr)
            lines.append(_interaction_samples(os.path.join(FLAGS.prefix, pth), ns))
        cdict = compress_dict(lines)
        cdict['xaxis'] = FLAGS.xname
        return cdict

    if FLAGS.xtype in ['onlinedata', 'gtdata', 'combined_gt_inv']:
        lines = []
        for l in open(basepath):  
            if l[0] =='#': continue
            ns = float(l.split(':')[0])
            pth = gen_pth(l, addstr)
            lines.append(_interaction_samples(os.path.join(FLAGS.prefix, pth), ns))
        cdict = compress_dict(lines)
        cdict['xaxis'] = FLAGS.xname
        return cdict

def plot_lines(lines, yname, ylim, yaxis, logflag, xname, hue):
    sns.lineplot(x = xname, y = yname, data = lines, marker = 'o', hue = hue)
    if logflag: plt.xscale("log")
    ax = plt.gca()
    ax.tick_params(labelsize = 15)
    if FLAGS.ylabel == 'yyes':
        plt.ylabel(yname, fontsize = 27)
    if FLAGS.xlabel == 'xyes':
        plt.xlabel(lines['xaxis'], fontsize = 20)
    plt.ylim(ylim)
   
def plot_and_save(metric, ylim, lines, savepath, logflag, xname = 'x', hue = 'Tag', legend = False, lname = None):
    plot_lines(lines, metric, ylim, metric, logflag, xname, hue)
    if legend:
        plt.legend(lname)
    plt.savefig(\
            os.path.join(savepath, FLAGS.xtype + '_' + '_'.join(metric.split(' ')) + '.pdf'), \
            bbox_inches = 'tight')


def plot_all_lines(lines, savepath, plot_metric, legend, lname):
    #plt.subplot(3,1,1)
    #locs, labels = plt.xticks()    
    #plt.xticks(locs, [])
    logflag = False
    if FLAGS.xtype in ['interaction_samples', 'sssteps']: logflag = True
    if plot_metric == 'ADT':
        plot_and_save('ADT', (0,12), lines, savepath, logflag, legend = legend, lname = lname)
    elif plot_metric == "ADT Old":
        plot_and_save('ADT Old', (0,1), lines, savepath, logflag, xname = 'x_c', hue = 'Tag_c', legend = legend, lname = lname)
    elif plot_metric == "Max Dist":
        plot_and_save('Max Dist', (5,20), lines, savepath, logflag, legend = legend, lname = lname)
    elif plot_metric == "Collisions":
        plot_and_save('Collisions', (0,0.8), lines, savepath, logflag, xname = 'x', hue = 'Tag', legend = legend, lname = lname)

    #plt.title('Ablations')

def worker():
    legend = len( FLAGS.basepath.split(',')) > 1
    lname = [basepath.split('/')[-1].split('.')[0] for basepath in FLAGS.basepath.split(',')]
    for plot_metric in ['ADT', 'ADT Old', 'Max Dist', 'Collisions']:
        for basepath in FLAGS.basepath.split(','):
            lines = process_files(basepath) 
            save_base = os.path.join(FLAGS.savepath, FLAGS.xlabel + '_' + FLAGS.ylabel)
            utils.mkdir_if_missing(save_base)
            plot_all_lines(lines, save_base, plot_metric, legend, lname = lname)
        plt.close()

def main(_):
    worker()

if __name__ == '__main__':
    app.run(main)
