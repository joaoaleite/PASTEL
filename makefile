extract_signals:
	sbatch run.slurm $(dataset) 7
	sbatch run.slurm $(dataset) 13
	sbatch run.slurm $(dataset) 70

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
	sbatch train.slurm $(dataset) 7 0
	sbatch train.slurm $(dataset) 13 0
	sbatch train.slurm $(dataset) 70 0
	sbatch train.slurm $(dataset) 7 1
	sbatch train.slurm $(dataset) 13 1
	sbatch train.slurm $(dataset) 70 1
	sbatch train.slurm $(dataset) 7 2
	sbatch train.slurm $(dataset) 13 2
	sbatch train.slurm $(dataset) 70 2
	sbatch train.slurm $(dataset) 7 3
	sbatch train.slurm $(dataset) 13 3
	sbatch train.slurm $(dataset) 70 3
	sbatch train.slurm $(dataset) 7 4
	sbatch train.slurm $(dataset) 13 4
	sbatch train.slurm $(dataset) 70 4
	sbatch train.slurm $(dataset) 7 5
	sbatch train.slurm $(dataset) 13 5
	sbatch train.slurm $(dataset) 70 5
	sbatch train.slurm $(dataset) 7 6
	sbatch train.slurm $(dataset) 13 6
	sbatch train.slurm $(dataset) 70 6
	sbatch train.slurm $(dataset) 7 7
	sbatch train.slurm $(dataset) 13 7
	sbatch train.slurm $(dataset) 70 7
	sbatch train.slurm $(dataset) 7 8
	sbatch train.slurm $(dataset) 13 8
	sbatch train.slurm $(dataset) 70 8
	sbatch train.slurm $(dataset) 7 9
	sbatch train.slurm $(dataset) 13 9
	sbatch train.slurm $(dataset) 70 9
