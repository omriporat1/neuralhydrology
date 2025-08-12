#!/bin/bash -l

#SBATCH --job-name=basin_rain
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=95
#SBATCH --mem=512G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --account=efratmorin

set -euo pipefail

# Activate conda
set +u
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/efratmorin/omripo/condaenvs/neuralhydrology
set -u

# Avoid thread oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export GDAL_NUM_THREADS=1

# Safer I/O
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

# Final locations
SRC_BASINS=/sci/labs/efratmorin/omripo/PhD/Data/Caravan/Caravan_winter/shapefiles/il/il_basin_shapes.shp
SRC_NETCDF_ROOT=/sci/labs/efratmorin/omripo/PhD/Data/IMS/Data_by_station/Data_by_station_formatted/output/
OUT_DIR=/sci/labs/efratmorin/omripo/PhD/Data/IMS/Data_by_station/Data_by_station_formatted/output/basin_average_rain/

# Stage inputs to node-local scratch
SCRATCH="${SLURM_TMPDIR:-/tmp}/basin_rain_${SLURM_JOB_ID:-$$}"
mkdir -p "$SCRATCH/years" "$SCRATCH/shp" "$OUT_DIR"

# Copy shapefile bundle
base="${SRC_BASINS%.shp}"
for ext in shp shx dbf prj cpg; do
  [ -f "${base}.${ext}" ] && cp -f "${base}.${ext}" "$SCRATCH/shp/" || true
done
BASINS="$SCRATCH/shp/$(basename "$SRC_BASINS")"

# Copy yearly rain_grid.nc (handles year_YYYY layout)
# Example: <root>/year_2022/rain_grid.nc -> $SCRATCH/years/year_2022/rain_grid.nc
while IFS= read -r -d '' nc; do
  yr_dir="$(basename "$(dirname "$nc")")"
  mkdir -p "$SCRATCH/years/$yr_dir"
  cp -f "$nc" "$SCRATCH/years/$yr_dir/rain_grid.nc"
done < <(find "$SRC_NETCDF_ROOT" -mindepth 2 -maxdepth 2 -type f -name 'rain_grid.nc' -print0)

NETCDF_ROOT="$SCRATCH/years"

# Logs on local disk (mirrored to stdout by Python)
LOG_DIR="$SCRATCH/logs"
LOG_FILE="$LOG_DIR/run.log"
mkdir -p "$LOG_DIR"

echo "Node: $(hostname)"
echo "Scratch: $SCRATCH"
echo "Local log: $LOG_FILE"
echo "OUT_DIR: $OUT_DIR"

# Run (use all allocated CPUs)
/usr/bin/time -v python -u /sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology/Experiments/Calculate_mean_basin_rain_from_gauges/calculate_basin_average_rain/src/extract_basins_avg_rain_from_netcdf.py \
  --basins "$BASINS" \
  --netcdf "$NETCDF_ROOT" \
  --out_dir "$OUT_DIR" \
  --log_dir "$LOG_DIR" \
  --log_file "$LOG_FILE" \
  --chunk_size 2000 \
  --log_every 2000 \
  --time_block 500 \
  --parallel \
  --workers "${SLURM_CPUS_PER_TASK}"

# Copy logs back to final location
mkdir -p "$OUT_DIR/logs"
cp -f "$LOG_FILE" "$OUT_DIR/run.log" 2>/dev/null || true
cp -f "$LOG_DIR"/*.log "$OUT_DIR/logs/" 2>/dev/null || true