# PASTOR-sequencing
### Code files:
Sequence -> signal model: ``squiggler.py`` \
Random Forest training, as used in rereads_simulation.ipynb: ``humpsClassifier.py`` \
CNN training: ``cnn_training.py`` \
Reread evaluation: ``rereads_simulation.ipynb`` \
Barcode space decoding accuracy evaluation: ``encoding_decoding.ipynb`` 

### Data files:
##### All .json data files can be read into a Pandas dataframe
Example raw .fast5 file (for PASTOR-AVLIM): ``DESKTOP_CHF4GRO_20221220_FAV72770_MN40387_sequencing_run_12_20_22_run04_a.fast5`` \
Segmented raw and processed P1-P4 signals: ``yy_mutants.json`` \
Segmented raw and processed PASTOR signals: ``pretty_segments_df.json`` \
Segmented raw and processed PASTOR-VGDNY signals in deamidation catalyzing conditions: ``n_to_d_segments_df.json``\
Reread simulation results, as created in rereads_simulation.ipynb and used in encoding_decoding.ipynb: ``rereads_acc.npy`` \
Barcode accuracy evaluation results, as created in encoding_decoding.ipynb: all contents in ``barcode_results`` \
Segmented raw folded domain signals:
* Amyloid Beta 15: ``segments_df_beta_15.json``
* Amyloid Beta 42: ``segments_df_beta_42.json``
* Titin: ``segments_df_titin_vp15.json``
* dTitin: ``segments_df_titin_vp15ee.json``

Segmented raw folded domain signals with the second (N-terminal) half of the PASTOR context:
* Amyloid Beta 15: ``segments_df_beta_15.json``
* Amyloid Beta 42: ``segments_df_beta_42.json``
* Titin: ``segments_df_titin_vp15.json``
* dTitin: ``segments_df_titin_vp15ee.json``
