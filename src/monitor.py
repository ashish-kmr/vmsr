import os, time, platform, psutil, sys
import logging

def get_stats():
  C = 2.**30.
  cpu_stats = psutil.cpu_times_percent()
  mem_stats = psutil.virtual_memory()
  load = os.getloadavg()
  
  name = ['user_cpu', 'sys_cpu', 'idle_cpu', 'io_cpu', 'load_1', 'load_2', 'load_3', 'mem_prct']
  vals = [cpu_stats.user, cpu_stats.system, cpu_stats.idle, cpu_stats.iowait, load[0], load[1], load[2], mem_stats.percent]

  name += ['mem_t', 'mem_f', 'mem_u', 'mem_buf']
  vals += [mem_stats.total/C, mem_stats.free/C, mem_stats.used/C, mem_stats.buffers/C]
  return name, vals

def monitor_(tf_event=False, logdir=None):
  node = platform.node()
  if tf_event:
    import tensorflow as tf
    from tfcode import tf_utils
    writer = tf.summary.FileWriter(logdir)

  while True:
    time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    name, vals = get_stats()

    if tf_event:
      metric_summary = tf.summary.Summary()
      for n, v in zip(name, vals):
        tf_utils.add_value_to_summary(metric_summary, 'sysstats/' + n, v)
      writer.add_summary(metric_summary)

    str_ = '{:s}, {:s}. '.format(time_str, node)
    for i in range(8):
      str_ += '{:>8s}: {:5.2f},   '.format(name[i], vals[i])
    logging.error('%s', str_)
    
    str_ = '{:s}, {:s}. '.format(time_str, node)
    for i in range(8, 12):
      str_ += '{:>7s}: {:5.2f}G,   '.format(name[i], vals[i])
    logging.error('%s', str_)
    
    # logging.error('%s, %s,   u %5.2f,   s %5.2f,   i %5.2f,   io %5.2f,   mem %5.2f,   load %s', 
    #   time_str, platform.node(), cpu_stats.user, cpu_stats.system, cpu_stats.idle, cpu_stats.iowait, mem_stats.percent, str(load))
    # logging.error('%s, %s,   total %5.2fG,   free %5.2fG,   used %5.2fG,   buffers %5.2fG',
    #   time_str, node, mem_stats.total/C, mem_stats.free/C, mem_stats.used/C, mem_stats.buffers/C)
    # logging.error('%s, %s,   %s', time_str, node, str(mem_stats))
    time.sleep(60)

def monitor():
  while True:
    time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    logging.error('%s, %s, %s', time_str, platform.node(), str(os.getloadavg()))
    time.sleep(60)

if __name__ == '__main__':
  print sys.argv
  if len(sys.argv) == 2:
    monitor_(True, sys.argv[1])
  else:
    monitor_()
