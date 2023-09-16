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
	sbatch slurm_jobs/train_$(device).slurm $(dataset) 70 0 $(training_method)
	sbatch slurm_jobs/train_$(device).slurm $(dataset) 70 1 $(training_method)
	sbatch slurm_jobs/train_$(device).slurm $(dataset) 70 2 $(training_method)
	sbatch slurm_jobs/train_$(device).slurm $(dataset) 70 3 $(training_method)
	sbatch slurm_jobs/train_$(device).slurm $(dataset) 70 4 $(training_method)
	sbatch slurm_jobs/train_$(device).slurm $(dataset) 70 5 $(training_method)
	sbatch slurm_jobs/train_$(device).slurm $(dataset) 70 6 $(training_method)
	sbatch slurm_jobs/train_$(device).slurm $(dataset) 70 7 $(training_method)
	sbatch slurm_jobs/train_$(device).slurm $(dataset) 70 8 $(training_method)
	sbatch slurm_jobs/train_$(device).slurm $(dataset) 70 9 $(training_method)

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

finetune:
	sbatch slurm_jobs/finetune.slurm $(dataset) 70 0
	sbatch slurm_jobs/finetune.slurm $(dataset) 70 1
	sbatch slurm_jobs/finetune.slurm $(dataset) 70 2
	sbatch slurm_jobs/finetune.slurm $(dataset) 70 3
	sbatch slurm_jobs/finetune.slurm $(dataset) 70 4
	sbatch slurm_jobs/finetune.slurm $(dataset) 70 5
	sbatch slurm_jobs/finetune.slurm $(dataset) 70 6
	sbatch slurm_jobs/finetune.slurm $(dataset) 70 7
	sbatch slurm_jobs/finetune.slurm $(dataset) 70 8
	sbatch slurm_jobs/finetune.slurm $(dataset) 70 9
