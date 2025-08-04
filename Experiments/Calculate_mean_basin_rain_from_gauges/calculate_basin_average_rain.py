# Filter date range
    if date_range:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        # Make start/end UTC if the data is UTC-aware
        if hasattr(gauges_data['datetime'].dt, 'tz') and gauges_data['datetime'].dt.tz is not None:
            tz = gauges_data['datetime'].dt.tz
            if start.tzinfo is None:
                start = start.tz_localize(tz)
            else:
                start = start.tz_convert(tz)
            if end.tzinfo is None:
                end = end.tz_localize(tz)
            else:
                end = end.tz_convert(tz)
        mask = (gauges_data['datetime'] >= start) & (gauges_data['datetime'] <= end)
        gauges_data = gauges_data[mask]

    for i, t in enumerate(times):
        frame = gauges_data[gauges_data['datetime'] == t]
        if frame.empty:
            continue
        # Robust location column detection
        location_cols = None
        for candidate in [['ITM_X', 'ITM_Y'], ['X', 'Y'], ['EASTING', 'NORTHING']]:
            if all(col in frame.columns for col in candidate):
                location_cols = candidate
                break
        if location_cols is None:
            raise KeyError("Rain gauge location columns not found. Expected one of ['ITM_X', 'ITM_Y'], ['X', 'Y'], ['EASTING', 'NORTHING'] in the input data.")
        coords = frame[location_cols].values
        values = frame['rain'].values if 'rain' in frame.columns else frame.iloc[:, 3].values
        # IDW interpolation
        dists = np.sqrt(((grid_points[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
        with np.errstate(divide='ignore'):  # Ignore division by zero
            weights = 1 / np.power(dists, power)
            weights[dists > max_radius] = 0
            weights[dists == 0] = 1e12  # Large weight for exact matches
        grid_vals = np.nansum(weights * values, axis=1) / np.nansum(weights, axis=1)
        grid_array[i] = grid_vals.reshape(grid_shape)
