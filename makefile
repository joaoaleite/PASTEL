extract_signals:
	sbatch run.slurm {dataset} {model_size}

extract_signals_all:
	sbatch run.slurm politifact 7
	sbatch run.slurm politifact 13
	sbatch run.slurm politifact 70
	sbatch run.slurm gossipcop 7
	sbatch run.slurm gossipcop 13
	sbatch run.slurm gossipcop 70
	sbatch run.slurm celebritydataset 7
	sbatch run.slurm celebritydataset 13
	sbatch run.slurm celebritydataset 70
	sbatch run.slurm fakenewsdataset 7
	sbatch run.slurm fakenewsdataset 13
	sbatch run.slurm fakenewsdataset 70

train:
	sbatch train.slurm {dataset} {model_size} 0
	sbatch train.slurm {dataset} {model_size} 1
	sbatch train.slurm {dataset} {model_size} 2
	sbatch train.slurm {dataset} {model_size} 3
	sbatch train.slurm {dataset} {model_size} 4
	sbatch train.slurm {dataset} {model_size} 5
	sbatch train.slurm {dataset} {model_size} 6
	sbatch train.slurm {dataset} {model_size} 7
	sbatch train.slurm {dataset} {model_size} 8
	sbatch train.slurm {dataset} {model_size} 9