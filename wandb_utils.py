import os
import shutil
import wandb

api = wandb.Api()
entity = 'cv-exp'
project = 'Cross-Domain-GCD'
old_split_ids = ['split_90390', 'split_43794', 'split_31963', 'split_4856']


def setup_conda_env(API_KEY: str):
    """sets the correct wandb user on activating conda environment"""
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        print("CONDA_PREFIX is not set. Please make sure you are running this inside a Conda environment.")
        return

    activate_dir = os.path.join(conda_prefix, 'etc/conda/activate.d')
    deactivate_dir = os.path.join(conda_prefix, 'etc/conda/deactivate.d')

    # Create directories if they don't exist
    os.makedirs(activate_dir, exist_ok=True)
    os.makedirs(deactivate_dir, exist_ok=True)

    # Create and write to the activate script
    activate_script_path = os.path.join(activate_dir, 'env_vars.sh')
    with open(activate_script_path, 'w') as activate_script:
        activate_script.write('#!/bin/sh\n')
        activate_script.write(f'export WANDB_API_KEY={API_KEY}\n')
        activate_script.write(f'WANDB_ENTITY={entity}\n')
        activate_script.write(f'WANDB_PROJECT={project}\n')

    # Create and write to the deactivate script
    deactivate_script_path = os.path.join(deactivate_dir, 'env_vars.sh')
    with open(deactivate_script_path, 'w') as deactivate_script:
        deactivate_script.write('#!/bin/sh\n')
        deactivate_script.write('unset WANDB_API_KEY\n')
        deactivate_script.write('unset WANDB_ENTITY\n')
        deactivate_script.write('unset WANDB_PROJECT\n')

    print(f"Setup completed successfully. Activate script path: {activate_script_path}, Deactivate script path: {deactivate_script_path}")


def delete_wandb_artifacts(run_id):
    run = api.run(f"{entity}/{project}/{run_id}")
    extension = ['.tar', '.pkl', '.pt', '.pth']
    files = run.files()
    for file in files:
        for ext in extension:
            if file.name.endswith(ext):
                file.delete()


def delete_local_outputs(output_folder, dry_run=True):
    runs = api.runs(f"{entity}/{project}")
    run_started = []
    for run in runs:
        if run.config.get('run_started') is not None:
            run_started.append(run.config.get('run_started'))

    for root, dirs, _ in os.walk(output_folder, topdown=False):
        for directory in dirs:
            if not any(run_start in directory for run_start in run_started):
                # if not any(split_id in directory for split_id in old_split_ids):
                print(f'deleting run {directory}..')
                if not dry_run:
                    shutil.rmtree(os.path.join(root, directory))


if __name__ == '__main__':
    # setup_conda_env('')
    delete_local_outputs('outputs', dry_run=False)