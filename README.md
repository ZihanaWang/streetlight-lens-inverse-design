# Streetlight lens inverse design

Data and code for the paper "Machine Learning-Based Inverse Design of Streetlight Lenses" by Zihan Wang, Rakesh Murugesan, Dupeng Li and Xihua Wang (submitted to Advanced Photonics Research).

## Data

data.xlsx has the lens designs simulated in SolidWorks and LightTools.

- CleanedData has the 496 designs used for the forward model.
- ErrorFree has the 369 designs that passed the LightTools merit function. The inverse model uses the 366 of them that also meet all three luminance thresholds.
- The first sheet is the original list of the 500 sampled designs.

R1, R2 and H are in mm and the spline angle is in degrees. Average luminance is in cd/m², and the two uniformities have no unit.

historical/pred40.xlsx has the 40 random targets from the LightTools check. The paper uses the first 30.

assets/ has the two SolidWorks renderings shown in Fig. 5.

## Running

```
pip install -r requirements.txt
python pipeline_v2.py        # about 3 h on 2 cores, writes to results_v2/
python export_evidence.py
python figures.py all        # figures 4 to 9
python flowchart.py          # figure 3, needs playwright and chromium
```

The stages of pipeline_v2.py can also be run one at a time (fwd, cv, cvmerge, invsel, inv, sens, reach, lt30). Finished stages are skipped, so set RESULTS_DIR to a new folder to start over.

results_v2/ and figures_v2/ already have the results and figures used in the paper.

## Which file is which

- Table 3 and Fig. 6 come from forward_test.json and forward_test_predictions.csv.
- Table 4 comes from cv_fold*.csv and nested_cv_*.csv.
- Table 5 and Fig. 8 come from inverse.json and inverse_in_member*.csv / inverse_ho_member*.csv.
- Fig. 9 comes from inverse_surrogate_predictions.csv.
- Fig. 7 comes from sensitivity.json and nonuniqueness.csv.
- The composite scores and splits behind Fig. 5 are in split_assignments.csv.
- Section 4.5 uses reachability.json and lighttools_check_designs.csv.
- All splits and seeds are in split_assignments.csv and seeds.json.

## Notes

We reran everything from scratch in a new folder and got the same results up to round-off (rerun_from_scratch/). TensorFlow training is not bit-reproducible across machines, so the last digit can differ.

Tested with Python 3.13. Package versions are in requirements.txt.
