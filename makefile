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
	sbatch train.slurm $(dataset) 7 0 $(training_method)
	sbatch train.slurm $(dataset) 13 0 $(training_method)
	sbatch train.slurm $(dataset) 70 0 $(training_method)
	sbatch train.slurm $(dataset) 7 1 $(training_method)
	sbatch train.slurm $(dataset) 13 1 $(training_method)
	sbatch train.slurm $(dataset) 70 1 $(training_method)
	sbatch train.slurm $(dataset) 7 2 $(training_method)
	sbatch train.slurm $(dataset) 13 2 $(training_method)
	sbatch train.slurm $(dataset) 70 2 $(training_method)
	sbatch train.slurm $(dataset) 7 3 $(training_method)
	sbatch train.slurm $(dataset) 13 3 $(training_method)
	sbatch train.slurm $(dataset) 70 3 $(training_method)
	sbatch train.slurm $(dataset) 7 4 $(training_method)
	sbatch train.slurm $(dataset) 13 4 $(training_method)
	sbatch train.slurm $(dataset) 70 4 $(training_method)
	sbatch train.slurm $(dataset) 7 5 $(training_method)
	sbatch train.slurm $(dataset) 13 5 $(training_method)
	sbatch train.slurm $(dataset) 70 5 $(training_method)
	sbatch train.slurm $(dataset) 7 6 $(training_method)
	sbatch train.slurm $(dataset) 13 6 $(training_method)
	sbatch train.slurm $(dataset) 70 6 $(training_method)
	sbatch train.slurm $(dataset) 7 7 $(training_method)
	sbatch train.slurm $(dataset) 13 7 $(training_method)
	sbatch train.slurm $(dataset) 70 7 $(training_method)
	sbatch train.slurm $(dataset) 7 8 $(training_method)
	sbatch train.slurm $(dataset) 13 8 $(training_method)
	sbatch train.slurm $(dataset) 70 8 $(training_method)
	sbatch train.slurm $(dataset) 7 9 $(training_method)
	sbatch train.slurm $(dataset) 13 9 $(training_method)
	sbatch train.slurm $(dataset) 70 9 $(training_method)

learning_curve:
	sbatch learning_curve.slurm $(dataset) 70 0
	sbatch learning_curve.slurm $(dataset) 70 1
	sbatch learning_curve.slurm $(dataset) 70 2
	sbatch learning_curve.slurm $(dataset) 70 3
	sbatch learning_curve.slurm $(dataset) 70 4
	sbatch learning_curve.slurm $(dataset) 70 5
	sbatch learning_curve.slurm $(dataset) 70 6
	sbatch learning_curve.slurm $(dataset) 70 7
	sbatch learning_curve.slurm $(dataset) 70 8
	sbatch learning_curve.slurm $(dataset) 70 9