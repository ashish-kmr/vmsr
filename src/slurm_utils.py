from __future__ import print_function
import os, sys, math, logging

def get_():
  return(time.strftime("%Y-%m-%d %H:%M"))

def simple_slurm_tf(job_name, log_dir, code_dir, mem, hh, commands,
  nodes=1, ppn=1, q_name='savio', venv=None, env_vars=[], ngpus=0):
  if code_dir is None:
    code_dir = os.getcwd()

  file_name = os.path.join(log_dir, 'run.slurm')
  with open(file_name,'wt') as f:
    print('#!/bin/sh', file=f)
    print('#SBATCH -J ' + job_name, file=f)
    print('#SBATCH --no-requeue ', file=f)
    print('#SBATCH -D ' + code_dir, file=f);
    
    err_file = os.path.join(log_dir, 'slurm.log.err')
    print('#SBATCH -e {:s}'.format(err_file), file=f);
    out_file = os.path.join(log_dir, 'slurm.log.out')
    print('#SBATCH -o {:s}'.format(out_file), file=f);

    print('#SBATCH -p {:s}'.format(q_name), file=f);
    print('#SBATCH -n {:d}'.format(ppn), file=f);
    print('#SBATCH -N {:d}'.format(nodes), file=f)
    
    if ngpus > 0: print('#SBATCH --gres=gpu:{:d}'.format(ngpus), file=f)
    
    # print('#SBATCH -N ' num2str(nodes) ':' num2str(ppn), file=f);
    # print('#SBATCH --tasks-per-node=' num2str(ppn), file=f);
    # print('#SBATCH --ntasks=' num2str(nodes), file=f);
    if (mem>0):
      print('#SBATCH --mem={:d}'.format(int(mem*1024)), file=f)
    if (hh>0):
      MM = math.floor((hh-math.floor(hh))*60);
      print('#SBATCH -t {:d}:{:02d}:00'.format(int(math.floor(hh)), int(MM)), file=f)

    # print('SLURM_ID=$SLURM_ARRAY_TASK_ID', file=f);
    # print('NUM_JOBS={:d}'.format(num_jobs), file=f);
    # print('echo Running job number $SLURM_ID', file=f);
 
    print('echo Running on host `hostname`', file=f);
    print('echo Time is `date`', file=f);
    print('echo Directory is `pwd`', file=f);
    print('', file=f);
    
    # Source virtual env.
    if venv is not None: 
      print('source {:s}'.format(venv), file=f) 
      print('', file=f);
      print('module list', file=f)
      print('', file=f);

    for i, env_var in enumerate(env_vars):
      if type(env_var[1]) == str:
        print('export {:s}="{:s}"'.format(env_var[0], env_var[1]), file=f)
      elif type(env_var[1]) == int:
        print('export {:s}={:d}'.format(env_var[0], env_var[1]), file=f)
      else:
        logging.error('Unknown type for env_vars[%d].', i)
        assert(False)
    
    for i, command in enumerate(commands):
      print(command + ' 1>>{:s}-{:04d} 2>&1 &'.format(
        os.path.join(log_dir, 'log.log'), i), file=f)
      print('sleep 20', file=f)
    print('wait', file=f);
  
  cmd = 'sbatch {:s} '.format(file_name)
  print('{:s}'.format(cmd))

def simple_slurm(job_name, log_dir, code_dir, mem, hh, num_jobs, command, 
  jpn=20, nodes=1, ppn=1, q_name='savio'):
  if code_dir is None:
    code_dir = os.getcwd()

  file_name = os.path.join(log_dir, job_name + '.slurm')
  with open(file_name,'wt') as f:
    print('#!/bin/sh', file=f)
    print('#SBATCH -J ' + job_name, file=f)
    print('#SBATCH --no-requeue ', file=f)
    print('#SBATCH -D ' + code_dir, file=f);
    print('#SBATCH -e ' + '/dev/null', file=f);
    print('#SBATCH -o ' + '/dev/null', file=f);
    print('#SBATCH -p ' + q_name, file=f);
    print('#SBATCH -n ' + '{:d}'.format(jpn), file=f);
    # print('#SBATCH -B ' num2str(nodes) ':' num2str(ppn), file=f);
    # print('#SBATCH --tasks-per-node=' num2str(ppn), file=f);
    # print('#SBATCH --ntasks=' num2str(nodes), file=f);
    if (mem>0):
      print('#SBATCH --mem={:d}'.format(int(mem*1024)), file=f)
    if (hh>0):
      MM = math.floor((hh-math.floor(hh))*60);
      print('#SBATCH -t {:d}:{:02d}:00'.format(int(math.floor(hh)), int(MM)), file=f)

    print('SLURM_ID=$SLURM_ARRAY_TASK_ID', file=f);
    print('NUM_JOBS={:d}'.format(num_jobs), file=f);
    print('echo Running job number $SLURM_ID', file=f);
 
    print('echo Running on host `hostname`', file=f);
    print('echo Time is `date`', file=f);
    print('echo Directory is `pwd`', file=f);
    print('', file=f);

    print('for (( c=0; c<$SLURM_JOB_CPUS_PER_NODE; c++))', file=f);
    print('do', file=f);
    print('  ID=`printf %03d $((SLURM_ID+c))`', file=f);
    print('  echo Launching $ID', file=f);
    print('  ' + command + ' 1>{:s}-$ID 2>&1 &'.format(os.path.join(log_dir, 'log.log')), file=f)
    print('done', file=f);
    print('wait', file=f);
  
  for i in range(0, num_jobs, jpn):
    cmd = 'sbatch --array {:d} {:s} '.format(i, file_name)
    print('{:s}'.format(cmd))

# def job_parallel(job_name, log_dir, nodes, mem, hh, fn_name, args, q_name='savio', 
#   ppn=1, jpn=20, code_dir=None):
# 
#   if code_dir is None:
#     code_dir = os.getcwd()
