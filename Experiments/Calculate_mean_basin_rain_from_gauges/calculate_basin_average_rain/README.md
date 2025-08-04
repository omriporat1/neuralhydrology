# Basin Average Rainfall Calculation

This project computes basin-average rainfall time series from point rain gauge data using Inverse Distance Weighting (IDW) interpolation. The main script implements various functions to handle data processing, quality control, interpolation, and visualization.

## Project Structure

```
calculate_basin_average_rain
├── src
│   ├── calculate_basin_average_rain.py  # Main script for calculating basin-average rainfall
│   └── utils
│       └── interpolation.py               # Utility functions for interpolation
├── requirements.txt                       # Required Python packages
└── README.md                              # Project documentation
```

## Functionality

The main script (`calculate_basin_average_rain.py`) includes the following functions:

- **fill_missing()**: Handles missing rainfall data by filling single missing values and excluding gauges with multiple consecutive missing values.
- **quality_check()**: Performs quality control on the rainfall data, flagging out-of-range values and generating a QC log.
- **interpolate_idw()**: Implements Inverse Distance Weighting (IDW) interpolation to estimate rainfall across the basin grid.
- **compute_basin_avg()**: Computes the average rainfall for each basin using zonal statistics.
- **summarize_statistics()**: Calculates and saves summary statistics for basin rainfall.
- **plot_rain_map()**: Generates visualizations of rainfall distribution for a given timestep.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd calculate_basin_average_rain
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage Example

To run the main script and compute basin-average rainfall, execute the following command:

```
python src/calculate_basin_average_rain.py
```

Ensure that your input data is correctly formatted and located in the specified directories before running the script.

## License

This project is licensed under the MIT License. See the LICENSE file for details.