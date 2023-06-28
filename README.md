# PASTOR-sequencing
### Code files:
Sequence -> signal model: ``squiggler.py`` \
Random Forest training: ``humpsClassifier.py`` \
CNN training: ``cnn_training.py`` \
Reread evaluation: ``rereads_simulation.ipynb`` \
Barcode space decoding accuracy evaluation: ``encoding_decoding.ipynb`` 

### Data files:
Example raw .fast5 file (for PASTOR-AVLIM): ``DESKTOP_CHF4GRO_20221220_FAV72770_MN40387_sequencing_run_12_20_22_run04_a.fast5`` \
Processed PASTOR signals: ``pretty_segments_df.json`` \
Processed PASTOR-VGDNY signals in deamidation catalyzing conditions: ``n_to_d_segments_df.json``\
Reread simulation results, as created in rereads_simulation.ipynb and used in encoding_decoding.ipynb: ``rereads_acc.npy`` \
Processed folded domain signals:
* Amyloid Beta 15: ``segments_df_beta_15.json``
* Amyloid Beta 42: ``segments_df_beta_42.json``
* Titin: ``segments_df_titin_vp15.json``
* dTitin: ``segments_df_titin_vp15ee.json``

Processed folded domain signals with the second (N-terminal) half of the PASTOR context:
* Amyloid Beta 15: ``segments_df_beta_15.json``
* Amyloid Beta 42: ``segments_df_beta_42.json``
* Titin: ``segments_df_titin_vp15.json``
* dTitin: ``segments_df_titin_vp15ee.json``
