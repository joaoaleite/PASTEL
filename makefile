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
	sbatch run.slurm $(dataset) 7 0
 	sbatch run.slurm $(dataset) 13 0
	sbatch run.slurm $(dataset) 70 0
	sbatch run.slurm $(dataset) 7 1
 	sbatch run.slurm $(dataset) 13 1
	sbatch run.slurm $(dataset) 70 1
	sbatch run.slurm $(dataset) 7 2
 	sbatch run.slurm $(dataset) 13 2
	sbatch run.slurm $(dataset) 70 2
	sbatch run.slurm $(dataset) 7 3
 	sbatch run.slurm $(dataset) 13 3
	sbatch run.slurm $(dataset) 70 3
	sbatch run.slurm $(dataset) 7 4
 	sbatch run.slurm $(dataset) 13 4
	sbatch run.slurm $(dataset) 70 4
	sbatch run.slurm $(dataset) 7 5
 	sbatch run.slurm $(dataset) 13 5
	sbatch run.slurm $(dataset) 70 5
	sbatch run.slurm $(dataset) 7 6
 	sbatch run.slurm $(dataset) 13 6
	sbatch run.slurm $(dataset) 70 6
	sbatch run.slurm $(dataset) 7 7
 	sbatch run.slurm $(dataset) 13 7
	sbatch run.slurm $(dataset) 70 7
	sbatch run.slurm $(dataset) 7 8
 	sbatch run.slurm $(dataset) 13 8
	sbatch run.slurm $(dataset) 70 8
	sbatch run.slurm $(dataset) 7 9
 	sbatch run.slurm $(dataset) 13 9
	sbatch run.slurm $(dataset) 70 9
	

train-all:
	sbatch run.slurm politifact 7 0
 	sbatch run.slurm politifact 13 0
	sbatch run.slurm politifact 70 0
	sbatch run.slurm politifact 7 1
 	sbatch run.slurm politifact 13 1
	sbatch run.slurm politifact 70 1
	sbatch run.slurm politifact 7 2
 	sbatch run.slurm politifact 13 2
	sbatch run.slurm politifact 70 2
	sbatch run.slurm politifact 7 3
 	sbatch run.slurm politifact 13 3
	sbatch run.slurm politifact 70 3
	sbatch run.slurm politifact 7 4
 	sbatch run.slurm politifact 13 4
	sbatch run.slurm politifact 70 4
	sbatch run.slurm politifact 7 5
 	sbatch run.slurm politifact 13 5
	sbatch run.slurm politifact 70 5
	sbatch run.slurm politifact 7 6
 	sbatch run.slurm politifact 13 6
	sbatch run.slurm politifact 70 6
	sbatch run.slurm politifact 7 7
 	sbatch run.slurm politifact 13 7
	sbatch run.slurm politifact 70 7
	sbatch run.slurm politifact 7 8
 	sbatch run.slurm politifact 13 8
	sbatch run.slurm politifact 70 8
	sbatch run.slurm politifact 7 9
 	sbatch run.slurm politifact 13 9
	sbatch run.slurm politifact 70 9

	sbatch run.slurm gossipcop 7 0
 	sbatch run.slurm gossipcop 13 0
	sbatch run.slurm gossipcop 70 0
	sbatch run.slurm gossipcop 7 1
 	sbatch run.slurm gossipcop 13 1
	sbatch run.slurm gossipcop 70 1
	sbatch run.slurm gossipcop 7 2
 	sbatch run.slurm gossipcop 13 2
	sbatch run.slurm gossipcop 70 2
	sbatch run.slurm gossipcop 7 3
 	sbatch run.slurm gossipcop 13 3
	sbatch run.slurm gossipcop 70 3
	sbatch run.slurm gossipcop 7 4
 	sbatch run.slurm gossipcop 13 4
	sbatch run.slurm gossipcop 70 4
	sbatch run.slurm gossipcop 7 5
 	sbatch run.slurm gossipcop 13 5
	sbatch run.slurm gossipcop 70 5
	sbatch run.slurm gossipcop 7 6
 	sbatch run.slurm gossipcop 13 6
	sbatch run.slurm gossipcop 70 6
	sbatch run.slurm gossipcop 7 7
 	sbatch run.slurm gossipcop 13 7
	sbatch run.slurm gossipcop 70 7
	sbatch run.slurm gossipcop 7 8
 	sbatch run.slurm gossipcop 13 8
	sbatch run.slurm gossipcop 70 8
	sbatch run.slurm gossipcop 7 9
 	sbatch run.slurm gossipcop 13 9
	sbatch run.slurm gossipcop 70 9

	sbatch run.slurm celebritydataset 7 0
 	sbatch run.slurm celebritydataset 13 0
	sbatch run.slurm celebritydataset 70 0
	sbatch run.slurm celebritydataset 7 1
 	sbatch run.slurm celebritydataset 13 1
	sbatch run.slurm celebritydataset 70 1
	sbatch run.slurm celebritydataset 7 2
 	sbatch run.slurm celebritydataset 13 2
	sbatch run.slurm celebritydataset 70 2
	sbatch run.slurm celebritydataset 7 3
 	sbatch run.slurm celebritydataset 13 3
	sbatch run.slurm celebritydataset 70 3
	sbatch run.slurm celebritydataset 7 4
 	sbatch run.slurm celebritydataset 13 4
	sbatch run.slurm celebritydataset 70 4
	sbatch run.slurm celebritydataset 7 5
 	sbatch run.slurm celebritydataset 13 5
	sbatch run.slurm celebritydataset 70 5
	sbatch run.slurm celebritydataset 7 6
 	sbatch run.slurm celebritydataset 13 6
	sbatch run.slurm celebritydataset 70 6
	sbatch run.slurm celebritydataset 7 7
 	sbatch run.slurm celebritydataset 13 7
	sbatch run.slurm celebritydataset 70 7
	sbatch run.slurm celebritydataset 7 8
 	sbatch run.slurm celebritydataset 13 8
	sbatch run.slurm celebritydataset 70 8
	sbatch run.slurm celebritydataset 7 9
 	sbatch run.slurm celebritydataset 13 9
	sbatch run.slurm celebritydataset 70 9

	sbatch run.slurm fakenewsdataset 7 0
 	sbatch run.slurm fakenewsdataset 13 0
	sbatch run.slurm fakenewsdataset 70 0
	sbatch run.slurm fakenewsdataset 7 1
 	sbatch run.slurm fakenewsdataset 13 1
	sbatch run.slurm fakenewsdataset 70 1
	sbatch run.slurm fakenewsdataset 7 2
 	sbatch run.slurm fakenewsdataset 13 2
	sbatch run.slurm fakenewsdataset 70 2
	sbatch run.slurm fakenewsdataset 7 3
 	sbatch run.slurm fakenewsdataset 13 3
	sbatch run.slurm fakenewsdataset 70 3
	sbatch run.slurm fakenewsdataset 7 4
 	sbatch run.slurm fakenewsdataset 13 4
	sbatch run.slurm fakenewsdataset 70 4
	sbatch run.slurm fakenewsdataset 7 5
 	sbatch run.slurm fakenewsdataset 13 5
	sbatch run.slurm fakenewsdataset 70 5
	sbatch run.slurm fakenewsdataset 7 6
 	sbatch run.slurm fakenewsdataset 13 6
	sbatch run.slurm fakenewsdataset 70 6
	sbatch run.slurm fakenewsdataset 7 7
 	sbatch run.slurm fakenewsdataset 13 7
	sbatch run.slurm fakenewsdataset 70 7
	sbatch run.slurm fakenewsdataset 7 8
 	sbatch run.slurm fakenewsdataset 13 8
	sbatch run.slurm fakenewsdataset 70 8
	sbatch run.slurm fakenewsdataset 7 9
 	sbatch run.slurm fakenewsdataset 13 9
	sbatch run.slurm fakenewsdataset 70 9