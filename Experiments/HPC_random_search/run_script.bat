@echo off
echo Running validation hydrographs script...
cd /d "c:\PhD\Python\neuralhydrology\Experiments\HPC_random_search"
python Create_validation_hydrographs_fixed.py
echo Script completed.
pause
