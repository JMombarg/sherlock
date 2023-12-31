# Sherlock

Scheme using Hidden markov models for Establishing Reiteratively a List Of Candidate period-spacings with liKelihood (SHERLOCK)

SHERLOCK is an interactive Python script that is used to automatically search of period-spacing patterns in a frequency list of a gravity-mode pulsator. It relies on a grid of pulsation models to predict to most probable period peaks in the Fourier spectrum that are part of the period-spacing pattern.

This repository is an open-source package, GNU-licensed. See GNU License in LICENSE.

## Installation
To ensure you have all required packages, an install with Poetry is adviced. Once you have Poetry installed (see instructions on [their website](https://python-poetry.org/docs/)), clone this github repo and run `poetry install` in the top-level of the directory. With `Poetry`, you can run `pytest` to check whether all tests are passing and the expected results are generated.

## Running the code
Before running, set the following parameters in `HMM_PSP_interactive.py`.
Azimuthal order to search for. m = -1 retrograde, m = 0 zonal, m = 1 prograde.  
`m = 1`  


Maximum number of skipped modes in the pattern before terminating.  
`max_skip = 2`


Maximum number of modes towards smaller periods before terminating.  
`max_modes_smaller_periods = 35`


Maximum number of modes towards larger periods before terminating.  
`max_modes_larger_periods  = 35`

Work directory, no trailing slash.  
`WORK_DIR = ...`  

Whether to use the entire grid for the calculations of the search window and expected value for the next period, instead of a selection of models based on the previous period spacing. Setting this to True might help if the period-spacing is complicated (e.g. mode trapping).
`always_use_entire_grid = False`

To run `SHERLOCK`, run the Python script `HMM_PSP_interactive.py`. You should see a figure with three panels:
1. mode periods vs. amplitude
2. the found period spacing pattern (initially empty)
3. the total probability for each selected period in the pattern (initiall empty)

The first step is to select three *consecutive* radial orders. Best practise in to select high amplitude peaks. You can select periods by clicking on them, and unselect by clicking again (zoom in if you select multiple periods at the same time). Once you are confident about your choice, press the `search` buttom to start the automated search. You should now see the results in the figure. In the top panel, the green shaded areas indicate the search window, the green dashed line the expected period, and the selected period in red. A grey shaded region means, no period was found. In the bottom panel, the initial manually selected periods are shown in red and always have a probability of 1.

If you are satisfied with the result, you can save it by pressing the `save` button. Otherwise, use the `reset` button to start again.



## Contents

1. `auxiliary`: Script for generating your own input file based on a grid of GYRE models. (Optional)
2. `docs`: Source to generate the documentation webpages.
3. `example_input_data`: An example of a frequency list from Van Beeck et al. (2021, A&A, 655, A59) used as input.
4. `example_expected_output`: Expected output for the frequency list provided in `example_input_data`.
5. `grids`: Input files containing the periods of radial orders -1 to -100, and m = -1, 0, 1 (l=1, dipole modes).
6. `sherlock`: The SHERLOCK python package.
7. `paper`: Paper placing the package in scientific context.
8. `tests`: pytest for the package.
9. `LICENSE`: GNU general public license.
10. `poetry.lock`: List of dependencies and their exact versions.
11. `pyproject.toml`: Installation configuration file.

### Author
Developed by Joey Mombarg
```
joey.mombarg[at]irap.omp.eu
Institut de Recherche en Astropysique et Planétology (IRAP)
Toulouse, France
```
