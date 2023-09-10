politifact:
	sbatch run.slurm politifact 7
	sbatch run.slurm politifact 13
	sbatch run.slurm politifact 70

gossipcop:
	sbatch run.slurm gossipcop 7
	sbatch run.slurm gossipcop 13
	sbatch run.slurm gossipcop 70

celebritydataset:
	sbatch run.slurm celebritydataset 7
	sbatch run.slurm celebritydataset 13
	sbatch run.slurm celebritydataset 70

fakenewsdataset:
	sbatch run.slurm fakenewsdataset 7
	sbatch run.slurm fakenewsdataset 13
	sbatch run.slurm fakenewsdataset 70

process-all:
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

consolidate-politifact:
