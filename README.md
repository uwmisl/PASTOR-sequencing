# PASTOR-sequencing
### Code files:
Extracting segments: ``translocation_finder.py`` \
Segmentation of PASTORs into VRs and YY dips, and featurization of VRs: ``YetAnotherYYSegmenter.ipynb`` \
Sequence -> signal model: ``squiggler.py`` \
Bayesian-based segmentation algorithm: ``chunkation.py`` \
ClpX stepping analysis: ``clpx_stepping_analysis.ipynb`` \
Random Forest training, as used in rereads_simulation.ipynb: ``humpsClassifier.py`` \
CNN training: ``cnn_training.py`` \
Reread evaluation: ``rereads_simulation.ipynb`` \
Barcode space decoding accuracy evaluation: ``encoding_decoding.ipynb`` \
Deamidation analysis:  ``n_to_d_analysis.ipynb``

### Data files:
##### All .json data files can be read into a Pandas dataframe
Example raw .fast5 file (for PASTOR-AVLIM): ``DESKTOP_CHF4GRO_20221220_FAV72770_MN40387_sequencing_run_12_20_22_run04_a.fast5`` \
Segmented raw and processed P1-P4 signals: ``yy_mutants.json`` \
Ordering of channels in the DTW-distance features, as created in YetAnotherYYSegmenter.ipynb and used in humpsClassifier:``channels_arr.npy``\
Segmented raw and processed PASTOR signals: ``pretty_segments_df.json`` \
Segmented raw and processed PASTOR-VGDNY signals in deamidation catalyzing conditions: ``n_to_d_segments_df.json``\
Manually labeled YY dips for ClpX stepping analysis: ``traces_df_sy.json`` \
Reread simulation results, as created in rereads_simulation.ipynb and used in encoding_decoding.ipynb: ``rereads_acc.npy`` \
Barcode accuracy evaluation results, as created in encoding_decoding.ipynb: all contents in ``barcode_results`` \
Segmented raw folded domain signals:
* Amyloid Beta 15: ``segments_df_beta_15.json``
* Amyloid Beta 42: ``segments_df_beta_42.json``
* Titin: ``segments_df_titin_vp15.json``
* dTitin: ``segments_df_titin_vp15ee.json``

Segmented raw folded domain signals with the second (N-terminal) half of the PASTOR context:
* Amyloid Beta 15: ``second_beta_15_segs_df.json``
* Amyloid Beta 42: ``second_beta_42_segs_df.json``
* Titin: ``second_titin_segs_df.json``
* dTitin: ``second_titin_ee_segs_df.json``

### Environment:
1) Install miniconda: https://docs.anaconda.com/free/miniconda/miniconda-install/
2) Run ``conda env create -f environment.yml`` Should take ~20 minutes
Code has only been tested on versions specified in the yml file and on MacOS

### Demo:
Expected results of files should match (barring variability from randomness) data seen in rereads_acc.npy, channels_arr.npy, pretty_segments_df.json, the images created and saved within .ipynb files, and the results seen in the manuscript. All code should take <5 min to run, unless otherwise specified in the comments (e.g. pairwise DTW comparison YetAnotherYYSegmenter.ipynb, reread simulation).
