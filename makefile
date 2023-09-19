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

cross-dataset:
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 0
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 1
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 2
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 3
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 4
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 5
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 6
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 7
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 8
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 9

	sbatch slurm_jobs/crossdataset.slurm politifact celebritydataset ws 0
	sbatch slurm_jobs/crossdataset.slurm politifact celebritydataset ws 1
	sbatch slurm_jobs/crossdataset.slurm politifact celebritydataset ws 2
	sbatch slurm_jobs/crossdataset.slurm politifact celebritydataset ws 3
	sbatch slurm_jobs/crossdataset.slurm politifact celebritydataset ws 4
	sbatch slurm_jobs/crossdataset.slurm politifact celebritydataset ws 5
	sbatch slurm_jobs/crossdataset.slurm politifact celebritydataset ws 6
	sbatch slurm_jobs/crossdataset.slurm politifact celebritydataset ws 7
	sbatch slurm_jobs/crossdataset.slurm politifact celebritydataset ws 8
	sbatch slurm_jobs/crossdataset.slurm politifact celebritydataset ws 9

	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 0
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 1
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 2
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 3
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 4
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 5
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 6
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 7
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 8
	sbatch slurm_jobs/crossdataset.slurm politifact gossipcop ws 9

	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset gossipcop ws 0
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset gossipcop ws 1
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset gossipcop ws 2
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset gossipcop ws 3
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset gossipcop ws 4
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset gossipcop ws 5
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset gossipcop ws 6
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset gossipcop ws 7
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset gossipcop ws 8
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset gossipcop ws 9

	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset politifact ws 0
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset politifact ws 1
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset politifact ws 2
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset politifact ws 3
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset politifact ws 4
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset politifact ws 5
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset politifact ws 6
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset politifact ws 7
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset politifact ws 8
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset politifact ws 9

	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset celebritydataset ws 0
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset celebritydataset ws 1
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset celebritydataset ws 2
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset celebritydataset ws 3
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset celebritydataset ws 4
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset celebritydataset ws 5
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset celebritydataset ws 6
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset celebritydataset ws 7
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset celebritydataset ws 8
	sbatch slurm_jobs/crossdataset.slurm fakenewsdataset celebritydataset ws 9

	sbatch slurm_jobs/crossdataset.slurm gossipcop politifact ws 0
	sbatch slurm_jobs/crossdataset.slurm gossipcop politifact ws 1
	sbatch slurm_jobs/crossdataset.slurm gossipcop politifact ws 2
	sbatch slurm_jobs/crossdataset.slurm gossipcop politifact ws 3
	sbatch slurm_jobs/crossdataset.slurm gossipcop politifact ws 4
	sbatch slurm_jobs/crossdataset.slurm gossipcop politifact ws 5
	sbatch slurm_jobs/crossdataset.slurm gossipcop politifact ws 6
	sbatch slurm_jobs/crossdataset.slurm gossipcop politifact ws 7
	sbatch slurm_jobs/crossdataset.slurm gossipcop politifact ws 8
	sbatch slurm_jobs/crossdataset.slurm gossipcop politifact ws 9

	sbatch slurm_jobs/crossdataset.slurm gossipcop fakenewsdataset ws 0
	sbatch slurm_jobs/crossdataset.slurm gossipcop fakenewsdataset ws 1
	sbatch slurm_jobs/crossdataset.slurm gossipcop fakenewsdataset ws 2
	sbatch slurm_jobs/crossdataset.slurm gossipcop fakenewsdataset ws 3
	sbatch slurm_jobs/crossdataset.slurm gossipcop fakenewsdataset ws 4
	sbatch slurm_jobs/crossdataset.slurm gossipcop fakenewsdataset ws 5
	sbatch slurm_jobs/crossdataset.slurm gossipcop fakenewsdataset ws 6
	sbatch slurm_jobs/crossdataset.slurm gossipcop fakenewsdataset ws 7
	sbatch slurm_jobs/crossdataset.slurm gossipcop fakenewsdataset ws 8
	sbatch slurm_jobs/crossdataset.slurm gossipcop fakenewsdataset ws 9

	sbatch slurm_jobs/crossdataset.slurm gossipcop celebritydataset ws 0
	sbatch slurm_jobs/crossdataset.slurm gossipcop celebritydataset ws 1
	sbatch slurm_jobs/crossdataset.slurm gossipcop celebritydataset ws 2
	sbatch slurm_jobs/crossdataset.slurm gossipcop celebritydataset ws 3
	sbatch slurm_jobs/crossdataset.slurm gossipcop celebritydataset ws 4
	sbatch slurm_jobs/crossdataset.slurm gossipcop celebritydataset ws 5
	sbatch slurm_jobs/crossdataset.slurm gossipcop celebritydataset ws 6
	sbatch slurm_jobs/crossdataset.slurm gossipcop celebritydataset ws 7
	sbatch slurm_jobs/crossdataset.slurm gossipcop celebritydataset ws 8
	sbatch slurm_jobs/crossdataset.slurm gossipcop celebritydataset ws 9

	sbatch slurm_jobs/crossdataset.slurm celebritydataset politifact ws 0
	sbatch slurm_jobs/crossdataset.slurm celebritydataset politifact ws 1
	sbatch slurm_jobs/crossdataset.slurm celebritydataset politifact ws 2
	sbatch slurm_jobs/crossdataset.slurm celebritydataset politifact ws 3
	sbatch slurm_jobs/crossdataset.slurm celebritydataset politifact ws 4
	sbatch slurm_jobs/crossdataset.slurm celebritydataset politifact ws 5
	sbatch slurm_jobs/crossdataset.slurm celebritydataset politifact ws 6
	sbatch slurm_jobs/crossdataset.slurm celebritydataset politifact ws 7
	sbatch slurm_jobs/crossdataset.slurm celebritydataset politifact ws 8
	sbatch slurm_jobs/crossdataset.slurm celebritydataset politifact ws 9

	sbatch slurm_jobs/crossdataset.slurm celebritydataset fakenewsdataset ws 0
	sbatch slurm_jobs/crossdataset.slurm celebritydataset fakenewsdataset ws 1
	sbatch slurm_jobs/crossdataset.slurm celebritydataset fakenewsdataset ws 2
	sbatch slurm_jobs/crossdataset.slurm celebritydataset fakenewsdataset ws 3
	sbatch slurm_jobs/crossdataset.slurm celebritydataset fakenewsdataset ws 4
	sbatch slurm_jobs/crossdataset.slurm celebritydataset fakenewsdataset ws 5
	sbatch slurm_jobs/crossdataset.slurm celebritydataset fakenewsdataset ws 6
	sbatch slurm_jobs/crossdataset.slurm celebritydataset fakenewsdataset ws 7
	sbatch slurm_jobs/crossdataset.slurm celebritydataset fakenewsdataset ws 8
	sbatch slurm_jobs/crossdataset.slurm celebritydataset fakenewsdataset ws 9

	sbatch slurm_jobs/crossdataset.slurm celebritydataset gossipcop ws 0
	sbatch slurm_jobs/crossdataset.slurm celebritydataset gossipcop ws 1
	sbatch slurm_jobs/crossdataset.slurm celebritydataset gossipcop ws 2
	sbatch slurm_jobs/crossdataset.slurm celebritydataset gossipcop ws 3
	sbatch slurm_jobs/crossdataset.slurm celebritydataset gossipcop ws 4
	sbatch slurm_jobs/crossdataset.slurm celebritydataset gossipcop ws 5
	sbatch slurm_jobs/crossdataset.slurm celebritydataset gossipcop ws 6
	sbatch slurm_jobs/crossdataset.slurm celebritydataset gossipcop ws 7
	sbatch slurm_jobs/crossdataset.slurm celebritydataset gossipcop ws 8
	sbatch slurm_jobs/crossdataset.slurm celebritydataset gossipcop ws 9

	
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
