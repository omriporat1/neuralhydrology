#!/bin/bash -l

#SBATCH --job-name=basin_rain
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --account=efratmorin

# Safer defaults; defer -u until after env activation
set -e -o pipefail

# Activate your conda env
# Temporarily disable nounset because some activation scripts reference unset vars (e.g., GDAL_DATA)
set +u
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/efratmorin/omripo/condaenvs/neuralhydrology

# Re‑enable nounset for the rest of the script
set -u



# Avoid thread oversubscription inside each worker
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export GDAL_NUM_THREADS=1

# Optional: flush Python stdout/stderr immediately
export PYTHONUNBUFFERED=1

# Paths on the cluster filesystem (final locations)
BASINS=/sci/labs/efratmorin/omripo/PhD/Data/Caravan/Caravan_winter/shapefiles/il/il_basin_shapes.shp
NETCDF_ROOT=/sci/labs/efratmorin/omripo/PhD/Data/IMS/Data_by_station/Data_by_station_formatted/output/            # folder with yearly subfolders containing rain_grid.nc
OUT_DIR=/sci/labs/efratmorin/omripo/PhD/Data/IMS/Data_by_station/Data_by_station_formatted/output/basin_average_rain/       # final output folder
LOG_DIR="$OUT_DIR"/logs
LOG_FILE="$OUT_DIR"/run.log

mkdir -p "$OUT_DIR" "$LOG_DIR"

/usr/bin/time -v python -u /sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology/Experiments/Calculate_mean_basin_rain_from_gauges/calculate_basin_average_rain/src/extract_basins_avg_rain_from_netcdf.py \
  --basins "$BASINS" \
  --netcdf "$NETCDF_ROOT" \
  --out_dir "$OUT_DIR" \
  --log_dir "$LOG_DIR" \
  --log_file "$LOG_FILE" \
  --chunk_size 4320
#  --parallel \
#  --workers "$SLURM_CPUS_PER_TASK"