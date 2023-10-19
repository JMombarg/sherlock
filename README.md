# Sherlock

Scheme using Hidden markov models for Establishing Reiteratively a List Of Candidate period-spacings with liKelihood (SHERLOCK)

This repository is an open-source package, GNU-licensed. See GNU License in LICENSE.

## Documentation
To ensure you have all required packages, an install with Poetry is adviced. Once you have Poetry installed (see instructions on their website), clone this github repo and run `poetry install` in the top-level of the directory.

Before running, set the following parameters in `HMM_PSP_interactive.py`.
Azimuthal order to search for. m = -1 retrograde, m = 0 zonal, m = 1 prograde.
`m = 1`
Maximum number of skipped modes in the pattern before terminating.
`max_skip = 2`
Maximum number of modes towards smaller periods before terminating.
`max_modes_smaller_periods = 35`
Maximum number of modes towards larger periods before terminating.
`max_modes_larger_periods  = 35`


To run `SHERLOCK`, run the Python script `HMM_PSP_interactive.py`. You should see a figure with three panels:
1. mode periods vs. amplitude
2. the found period spacing pattern (initially empty)
3. the total probability for each selected period in the pattern (initiall empty)

The first step is to select three *consecutive* radial orders. Best practise in to select high amplitude peaks. You can select periods by clicking on them, and unselect by clicking again (zoom in if you select multiple periods at the same time). Once you are confident about your choice, press the `search` buttom to start the automated search. You should now see the results in the figure. In the top panel, the green shaded areas indicate the search window, the green dashed line the expected period, and the selected period in red. A grey shaded region means, no period was found.

If you are satisfied about the result, you can save it by pressing the `save` button. Otherwise, use the `reset` button to start again.



## Contents

1. `docs`: Source to generate the documentation webpages
2. `example_setup`: An example setup
3. `sherlock`: The SHERLOCK python package
4. `paper`: Paper placing the package in scientific context.
5. `tests`: pytest for the package
6. `LICENSE`: GNU general public license
7. `poetry.lock`: List of dependencies and their exact versions.
8. `pyproject.toml`: Installation configuration file.

### Author
Developed by Joey Mombarg
```
joey9mombarg[at]gmail.com
Institut de Recherche en Astropysique et Planétology
Toulouse, France
```
