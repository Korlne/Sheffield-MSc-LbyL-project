# Full work of MSc Light-by-Light Scattering Phenomena in Ultra-Peripheral Collisions at the ATLAS Experiment
(Code and plots only, no AOD files)

The mainly used code were:

**lbyl_common.py**   (Helper function for cutflow)

**process_cutflow.py**   (Using lbyl_common process the cutflow, include a use of multi-processing to improve the efficiency)

**plots.py**   (Normalisation and ATLAS style plot)

**root_plots.py**   (Another ATLAS style plot, but root input)

**bkg_root_process.py** (Processing the normalisation of all bkgs, sigs, systematics)

**significance_analysis_root4.py**   (Statistic analysis, include sliding window algorithm, significance calculation, exclusion limit, discovery reach, plots.)

**crosssection2.py**   (calculation of cross section)
