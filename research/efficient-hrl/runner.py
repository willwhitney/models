import os
import sys
import itertools

dry_run = '--dry-run' in sys.argv
clear = '--clear' in sys.argv

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")
if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")
code_dir = '/private/home/willwhitney/code'

# basename = "antmaze_gpu"
# grids = [
#     {
#         "env": ["ant_maze"],
#         "seed": range(8),
#     },
# ]

# basename = "pusher_z7"
# grids = [
#     {
#         "env": ["Pusher-v2"],
#         "seed": range(8),
#     },
# ]

basename = "reacher_zerosamp_noise001_metanoise1"
grids = [
    {
        "env": ["Reacher-v2"],
        "seed": range(8),
    },
]


jobs = []
for grid in grids:
    individual_options = [[{key: value} for value in values]
                          for key, values in grid.items()]
    product_options = list(itertools.product(*individual_options))
    jobs += [{k: v for d in option_set for k, v in d.items()}
             for option_set in product_options]

if dry_run:
    print("NOT starting {} jobs:".format(len(jobs)))
else:
    print("Starting {} jobs:".format(len(jobs)))

all_keys = set().union(*[g.keys() for g in grids])
merged = {k: set() for k in all_keys}
for grid in grids:
    for key in all_keys:
        grid_key_value = grid[key] if key in grid else ["<<NONE>>"]
        merged[key] = merged[key].union(grid_key_value)
varying_keys = {key for key in merged if len(merged[key]) > 1}

excluded_flags = {}

for job in jobs:
    jobname = basename
    for flag in job:
        # construct the job's name
        if flag in varying_keys:
            jobname = jobname + "_" + flag + str(job[flag])
    
    # flagstring = "{} hiro_repr {} base_uvf suite".format(jobname, job['env'])


    slurm_script_path = 'slurm_scripts/' + jobname + '.slurm'
    slurm_script_dir = os.path.dirname(slurm_script_path)
    os.makedirs(slurm_script_dir, exist_ok=True)

    slurm_log_dir = 'slurm_logs/' + jobname 
    os.makedirs(os.path.dirname(slurm_log_dir), exist_ok=True)

    true_source_dir = code_dir + '/models/research/efficient-hrl' 
    job_source_dir = code_dir + '/hiro-clones/' + jobname
    try:
        os.makedirs(job_source_dir)
        os.system('cp -R ./* ' + job_source_dir)
        # os.system('cp -R reacher_family ' + job_source_dir)
    except FileExistsError:
        # with the 'clear' flag, we're starting fresh
        # overwrite the code that's already here
        if clear:
            print("Overwriting existing files.")
            os.system('cp -R ./* ' + job_source_dir)

    train_command = "python {}/scripts/local_train.py {} hiro_repr {} base_uvf suite".format(
            job_source_dir, jobname, job['env'])
    eval_command = "python {}/scripts/local_eval.py {} hiro_repr {} base_uvf suite".format(
            job_source_dir, jobname, job['env'])

    job_start_command = "sbatch " + slurm_script_path

    print(train_command)
    with open(slurm_script_path, 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write("#SBATCH --job-name" + "=" + jobname + "\n")
        slurmfile.write("#SBATCH --open-mode=append\n")
        slurmfile.write("#SBATCH --output=slurm_logs/" +
                        jobname + ".out\n")
        slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
        slurmfile.write("#SBATCH --export=ALL\n")
        slurmfile.write("#SBATCH --signal=USR1@600\n")
        slurmfile.write("#SBATCH --time=0-12\n")
        # slurmfile.write("#SBATCH --time=2-00\n")
        # slurmfile.write("#SBATCH -p dev\n")
        # slurmfile.write("#SBATCH -p uninterrupted,dev\n")
        # slurmfile.write("#SBATCH -p uninterrupted\n")
        slurmfile.write("#SBATCH -p dev,uninterrupted,priority\n")
        slurmfile.write("#SBATCH --comment='contract end 4/24'\n")
        slurmfile.write("#SBATCH -N 1\n")
        slurmfile.write("#SBATCH --mem=32gb\n")


        slurmfile.write("#SBATCH --cpus-per-task=8\n")
        slurmfile.write("#SBATCH --gres=gpu:1\n")
        slurmfile.write("#SBATCH --ntasks=1\n")

        # slurmfile.write("#SBATCH -c 40\n")
        # slurmfile.write("#SBATCH --constraint=pascal\n")

        slurmfile.write("cd " + true_source_dir + '\n')
        slurmfile.write("{} &\n".format(train_command))
        slurmfile.write("env CUDA_VISIBLE_DEVICES='' {} &\n".format(eval_command))
        slurmfile.write("wait\n")
        slurmfile.write("\n")

    if not dry_run:
        os.system(job_start_command + " &")
