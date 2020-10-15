Configuration Arguments
=======================

This page provides a list of possible configuration arguments.
Check out the file `examples/config.yml.example <https://github.com/neuralhydrology/neuralhydrology/blob/master/examples/config.yml.example>`__ for an example of how a config file could look like.

General experiment configurations
---------------------------------

-  ``experiment_name``: Defines the name of your experiment that will be
   used as a folder name (+ date-time string), as well as the name in
   TensorBoard.

-  ``run_dir``: Full or relative path to where the run directory is
   stored (if empty runs are stored in ${current\_working\_dir}/runs/)

-  ``train_basin_file``: Full or relative path to a text file containing
   the training basins (use data set basin id, one id per line).
-  ``validation_basin_file``: Full or relative path to a text file
   containing the validation basins (use data set basin id, one id per
   line).
-  ``test_basin_file``: Full or relative path to a text file containing
   the training basins (use data set basin id, one id per line).

-  ``train_start_date``: Start date of the training period (first day of
   discharge) in the format ``DD/MM/YYYY``. Can also be a list of dates
   to specify multiple training periods. If a list is specified, ``train_end_date``
   must also be a list of equal length. Corresponding pairs of start and
   end date denote the different periods.
-  ``train_end_date``: End date of the training period (last day of
   discharge) in the format ``DD/MM/YYYY``. Can also be a list of dates.
   If a list is specified, also ``train_start_date`` must be a list with
   an equal length.
-  ``validation_start_date``: Start date of the validation period (first
   day of discharge) in the format ``DD/MM/YYYY``. Can also be 
   a list of dates (similar to train period specifications).
-  ``validation_end_date``: End date of the validation period (last day
   of discharge) in the format ``DD/MM/YYYY``. Can also be 
   a list of dates (similar to train period specifications).
-  ``test_start_date``: Start date of the test period (first day of
   discharge) in the format ``DD/MM/YYYY``. Can also be 
   a list of dates (similar to train period specifications).
-  ``test_end_date``: End date of the validation period (last day of
   discharge) in the format ``DD/MM/YYYY``. Can also be 
   a list of dates (similar to train period specifications).

-  ``seed``: Fixed random seed. If empty, a random seed is generated for
   this run.

-  ``device``: Which device to use in format of ``cuda:0``, ``cuda:1``,
   etc, for GPUs or ``cpu``

Validation settings
-------------------

-  ``validate_every``: Integer that specifies in which interval a
   validation is performed. If empty, no validation is done during
   training.

-  ``validate_n_random_basins``: Integer that specifies how many random
   basins to use per validation. Values larger *n_basins* are clipped
   to *n_basins*.

-  ``metrics``: List of metrics to calculate during validation/testing.
   See
   `codebase.evaluation.metrics <https://neuralhydrology.readthedocs.io/en/latest/api/neuralhydrology.evaluation.metrics.html>`__
   for a list of available metrics.

-  ``save_validation_results``: True/False, if True, stores the
   validation results to disk as a pickle file. Otherwise they are only
   used for TensorBoard

General model configuration
---------------------------

-  ``model``: Defines the core of the model that will be used. Names
   have to match the values in `this
   function <https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/modelzoo/__init__.py#L14>`__,
   e.g., [``cudalstm``, ``ealstm``, ``embcudalstm``, ``mtslstm``]

-  ``head``: The prediction head that is used on top of the output of
   the core model. Currently supported is ``regression``.
   Make sure to pass the necessary options depending on your
   choice of the head (see below).

-  ``hidden_size``: Hidden size of the core model. In the case of an
   LSTM, this reflects the number of LSTM states.

-  ``initial_forget_bias``: Initial value of the forget gate bias.

-  ``output_dropout``: Dropout applied to the output of the LSTM

Regression head
~~~~~~~~~~~~~~~

Can be ignored if ``head != 'regression'``

-  ``output_activation``: Which activation to use on the output
   neuron(s) of the linear layer. Currently supported are ``linear``,
   ``relu``, ``softplus``. If empty, ``linear`` is used.


Multi-timescale training settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These are used if ``model == mtslstm``.

-  ``transfer_mtslstm_states``: Specifies if and how hidden and cell
   states are transferred from lower to higher frequencies. This
   configuration should be a dictionary with keys ``h`` (hidden state)
   and ``c`` (cell state). Possible values are
   ``[None, linear, identity]``. If ``transfer_mtslstm_states`` is not
   provided or empty, the default is linear transfer.

-  ``shared_mtslstm``: If False, will use a distinct LSTM with
   individual weights for each timescale. If True, will use a single
   LSTM for all timescales and use one-hot-encoding to identify the
   current input timescale. In both cases, ``transfer_mtslstm_states``
   can be used to configure hidden and cell state transfer.

ODE-LSTM settings
~~~~~~~~~~~~~~~~~

These are used if ``model == odelstm``.

-  ``ode_method``: Method to use to solve the ODE. One of
   ``[euler, rk4, heun]``.

-  ``ode_num_unfolds``: Number of iterations to break each ODE solving
   step into.

-  ``ode_random_freq_lower_bound``: Lowest frequency that will be used
   to randomly aggregate the first slice of the input sequence. See the
   documentation of the ODELSTM class for more details on the frequency
   randomization.

Embedding network settings
--------------------------

These settings apply to small fully connected networks that are used in
various places, such as the embedding network for static features in the
``embcudalstm`` model or as an optional extended input gate network in 
the ``ealstm`` model. For all other models, these settings can be ignored.
If specified, but the ``cudalstm`` model is selected, the code will print a 
warning.

-  ``embedding_hiddens``: List of integers that define the number of
   neurons per layer in the fully connected network. The last number is
   the number of output neurons.

-  ``embedding_activation``: activation function of embedding network
   (currently only ``tanh`` is supported). The activation function is
   not applied to the output neurons, which always have a linear
   activation function. A activation function for the output neurons has
   to be applied in the main model class.

-  ``embedding_dropout``: Dropout rate applied to the embedding network

Training settings
-----------------

-  ``optimizer``: Specify which optimizer to use. Currently supported
   is Adam (standard). New optimizers can be added
   `here <https://neuralhydrology.readthedocs.io/en/latest/api/neuralhydrology.training.html#neuralhydrology.training.get_optimizer>`__.

-  ``loss``: Which loss to use. Currently supported are ``MSE``,
   ``NSE``, ``WeightedNSE``, ``RMSE``. New losses can be added
   `here <https://neuralhydrology.readthedocs.io/en/latest/api/neuralhydrology.training.loss.html>`__.
   The ``WeightedNSE`` is especially for multi-target 
   settings. Use ``target_loss_weights`` to specify per-target
   weights.

-  ``target_loss_weights``: Only necessary if ``loss == WeightedNSE``. A list 
   of float values specifying the per-target loss weight. The order of the 
   weights corresponds to the order of the ``target_variables``.

-  ``regularization``: List of optional regularization terms. Currently
   supported is ``tie_frequencies``, which couples the predictions of
   all frequencies via an MSE term. New regularizations can be added
   `here <https://neuralhydrology.readthedocs.io/en/latest/api/neuralhydrology.training.regularization.html>`__.

-  ``learning_rate``: Learning rate. Can be either a single number (for
   a constant learning rate) or a dictionary. If it is a dictionary, the
   keys must be integer that reflect the epochs at which the learning
   rate is changed to the corresponding value. The key ``0`` defines the
   initial learning rate.

-  ``batch_size``: Mini-batch size used for training.

-  ``epochs``: Number of training epochs

-  ``use_frequencies``: Defines the time step frequencies to use (daily,
   hourly, ...). Use `pandas frequency
   strings <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases>`__
   to define frequencies. Note: The strings need to include values,
   e.g., '1D' instead of 'D'. If used, ``predict_last_n`` and
   ``seq_length`` must be dictionaries.

-  ``no_loss_frequencies``: Subset of frequencies from
   ``use_frequencies`` that are "evaluation-only", i.e., the model will
   get input and produce output in the frequencies listed here, but they
   will not be considered in the calculation of loss and regularization
   terms.

-  ``seq_length``: Length of the input sequence. If ``use_frequencies``
   is used, this needs to be a dictionary mapping each frequency to a
   sequence length, else an int.

-  ``predict_last_n``: Defines which time steps are used to calculate
   the loss, counted backwards. Can't be larger than ``seq_length``.
   Sequence-to-one would be ``predict_last_n: 1`` and
   sequence-to-sequence (with e.g. a sequence length of 365)
   ``predict_last_n: 365``. If ``use_frequencies`` is used, this needs
   to be a dictionary mapping each frequency to a
   predict\_last\_n-value, else an int.

-  ``target_noise_std``: Defines the standard deviation of gaussian
   noise which is added to the labels during training. Set to zero or
   leave empty to *not* add noise.

-  ``clip_gradient_norm``: If a value, clips norm of gradients to that
   specific value during training. Leave empty for not clipping.

-  ``num_workers``: Number of (parallel) threads used in the data
   loader.

-  ``save_weights_every``: Interval, in which the weights of the model
   are stored to disk. ``1`` means to store the weights after each
   epoch, which is the default if not otherwise specified.

Logger settings
---------------

-  ``log_interval``: Interval at which the training loss is logged, 
   by default 10.
-  ``log_tensorboard``: True/False. If True, writes logging results into
   TensorBoard file. The default, if not specified, is True.

-  ``log_n_figures``: If a (integer) value greater than 0, saves the
   predictions as plots of that n specific (random) basins during
   validations.

Data settings
-------------

-  ``dataset``: Defines which data set will be used. Currently supported
   are ``camels_us`` (CAMELS data set by Newman et al.), ``CAMELS_GB``
   (the GB version of CAMELS by Coxon et al.), ``CAMELS_CL`` (the CL
   version of CAMELS by Alvarez-Garreton et al.), and 
   ``hourly_camels_us`` (hourly data for 516 CAMELS basins).

-  ``data_dir``: Full or relative path to the root directory of the data set.

-  ``train_data_file``: If not empty, uses the pickled file at this path
   as the training data. Can be used to not create the same data set
   multiple times, which saves disk space and time. If empty, creates
   new data set and optionally stores the data in the run directory (if
   ``save_train_data`` is True).

-  ``cache_validation_data``: True/False. If True, caches validation data 
   in memory for the time of training, which does speed up the overall
   training time. By default True, since even larger datasets are usually
   just a few GB in memory, which most modern machines can handle.

-  ``dynamic_inputs``: List of variables to use as time series inputs.
   Names must match the exact names as defined in the data set. Note: In
   case of multiple input forcing products, you have to append the
   forcing product behind each variable. E.g., 'prcp(mm/day)' of the
   daymet product is 'prcp(mm/day)_daymet'. When training on multiple
   frequencies (cf. ``use_frequencies``), it is possible to define
   dynamic inputs for each frequency individually. To do so,
   ``dynamic_inputs`` must be a dict mapping each frequency to a list of
   variables. E.g., to use precipitation from daymet for daily and from
   nldas-hourly for hourly predictions:

   ::

       dynamic_inputs:
         1D:
           - prcp(mm/day)_daymet
         1H:
           - total_precipitation_nldas_hourly

-  ``target_variables``: List of the target variable(s). Names must match
   the exact names as defined in the data set.

-  ``clip_targets_to_zero``: Optional list of target variables to clip to
   zero during evaluation.

-  ``duplicate_features``: Can be used to duplicate time series features
   (e.g., for different normalizations). Can be either a str, list or dictionary
   (mapping from strings to ints). If string, duplicates the corresponding
   feature once. If list, duplicates all features in that list once. Use
   a dictionary to specify the exact number of duplicates you like.
   To each duplicated feature, we append ``_copyN``, where `N` is counter
   starting at 1.

-  ``lagged_features``: Can be used to add a lagged copy of another
   feature to the list of available input/output features. Has to be a
   dictionary mapping from strings to int, where the string specifies the
   feature name and the int the number of lagged time steps. Those values
   can be positive or negative (see
   `pandas shift <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html>`__
   for details). We append ``_shiftN`` to each lagged feature, where `N`
   is the shift count.

-  ``custom_normalization``: Has to be a dictionary, mapping from
   time series feature names to ``centering`` and/or ``scaling``. Using
   this argument allows to overwrite the default zero mean, unit
   variance normalization per feature. Supported options for
   ``centering`` are 'None' or 'none', 'mean', 'median' and min.
   None/none sets the centering parameter to 0.0, mean to the feature
   mean, median to the feature median, and min to the feature
   minimum, respectively. Supported options for `scaling` are
   'None' or 'none', 'std', 'minmax'. None/none sets the scaling
   parameter to 1.0, std to the feature standard deviation and
   minmax to the feature max minus the feature min. The combination
   of centering: min and scaling: minmax results in min/max
   feature scaling to the range [0,1].

-  ``additional_feature_files``: Path to a pickle file (or list of paths
   for multiple files), containing a dictionary with each key
   corresponding to one basin id and the value is a date-time indexed
   pandas DataFrame. Allows the option to add any arbitrary data that is
   not included in the standard data sets. **Convention**: If a column
   is used as static input, the value to use for specific sample should
   be in same row (datetime) as the target discharge value.

-  ``static_inputs``: Columns of the DataFrame loaded with the
   ``additional_feature_files`` that should be used as static feature.
   Names must match the column names in the DataFrame. Leave empty to
   not use any additional static feature.

-  ``use_basin_id_encoding``: True/False. If True, creates a
   basin-one-hot encoding as a(n) (additional) static feature vector for
   each sample.

-  ``hydroatlas_attributes``: Which HydroATLAS attributes to use. Leave
   empty if none should be used. Names must match the exact names as
   defined in the data set.

CAMELS US specific
~~~~~~~~~~~~~~~~~~

Can be ignored if ``dataset not in ['camels_us', 'hourly_camels_us']``

-  ``forcings``: Can be either a string or a list of strings that
   correspond to forcing products in the camels data set. Also supports
   ``maurer_extended``, ``nldas_extended``, and (for
   ``hourly_camels_us``) ``nldas_hourly``.

-  ``camels_attributes``: Which CAMELS attributes to use. Leave empty if
   none should be used. Names must match the exact names as defined in
   the data set.
