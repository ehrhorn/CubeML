import time
import json
import subprocess
from gpuinfo import GPUInfo
from pathlib import Path


def get_project_root():
    """Finds absolute path to project root - useful for running code on different machines.
    
    Returns:
        str -- path to project root
    """    
    
    # * Find project root
    current_dir_splitted = str(Path.cwd()).split('/')
    i = 0
    while current_dir_splitted[i] != 'CubeML':
        i += 1
    return '/'.join(current_dir_splitted[:i+1]) 

# * IDEA
# * A script that checks how many of the GPUs that are idle. Whenever one GPU is idle,
# * it is attempted to start an experiment on it.
# ! PROBLEM: We dont have same environment.
# * IMPLEMENTATION
# * run a while loop -> check if available and run an experiment if yes -> sleep for x seconds.
# * A script is run by running a shell command ('nohup python -u <COMMAND> --gpu' or '<COMMAND> --gpu') given by experiment file. The flag --gpu 1 (or 0) is added.

def get_time():
    """Reports the current local time

    Reports time in the format required for the name of the .txt-files holding the run-commands.
    
    Returns:
        str -- Current time
    """    
    return time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())

def find_oldest_experiment():
    cwd = Path.cwd()
    path = Path(str(Path(get_project_root()).parent)+'/experiments/')
    exps = sorted([str(exp) for exp in path.iterdir() if exp.suffix=='.pickle'])
    
    try:
        exp_str = exps[0]
    # * If no experiment is there, pass None back.
    except IndexError:
        exp_str = None

    return exp_str

def launch_and_remove_experiment(execute_file, gpu_id=None):

    # * Load the JSON-file containing the command
    with open(execute_file, 'r') as f:
        d = json.load(f)
        # command = [line for line in f][0]+' --gpu %s'%(str(gpu_id))
    
    # * Prepare the command
    if d['nohup_dest']:
        # command = ['nohup', 'python -u', d['command']+' --gpu %s'%(str(gpu_id)), '> %s'%(d['nohup_dest']), '&disown']
        command = ['nohup', d['command']+' --gpu %s'%(str(gpu_id)), '> %s'%(d['nohup_dest']), '&disown']
        command = ['nohup', d['command']+' --gpu %s'%(str(gpu_id)), '> %s'%(d['nohup_dest']), '&disown']

    else:
        # command = ['nohup', 'python -u', d['command']+' --gpu %s'%(str(gpu_id)), '&disown']
        command = ['nohup', d['command']+' --gpu %s'%(str(gpu_id)), '&disown']
        command = 'nohup '+d['command']+' --gpu %s'%(str(gpu_id))#+' &disown'
    
    # * Call it from terminal
    print(command)
    subprocess.call(command, shell=True)

    # * Delete the execute file
    Path(execute_file).unlink()

if __name__ == '__main__':

    # * Keep script running in background indefinitely
    while True:
        
        # * Check available GPUs
        gpus_available = GPUInfo.check_empty()
        if gpus_available:
            
            # * For each available GPU, attempt to run an experiment on it.
            for gpu_id in gpus_available:
                exp_file = find_oldest_experiment()

                # * An experiment is only launched if it exists 
                if exp_file:
                    launch_and_remove_experiment(exp_file, gpu_id=gpu_id)
        
        # * Check for new experiments every N seconds
        time.sleep(5)