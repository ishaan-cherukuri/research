# Open questions and data limitations — NI: Reports build

Logged per the build spec. Nothing below was worked around by inventing a number.

## Resolved during the build

**Table 2 did not match the cross-validation output.**
`results/spec_v3_harmonized/table3_cindex.csv` was written on 2026-08-20 09:06,
after `cv_results.csv` was produced on 2026-08-19 16:17, and eleven of its
ninety-one cells disagree with the run. All eleven are in the two BSC-only rows.
The XGBoost and RSF entries for the global-slope row were transposed, and the
Cox and parametric entries for both BSC-only rows came from an earlier,
unharmonized run. The table has been regenerated directly from `cv_results.csv`
as `table3_cindex_regen.csv` and the manuscript now uses it. Consequences:

- BSC baseline row: 0.472 / 0.444 / 0.520 / 0.494 / 0.510 / 0.528 / 0.532,
  not 0.532 / 0.502 / 0.520 / 0.5199 / 0.510 / 0.528 / 0.513.
- BSC global slope row: 0.540 / 0.584 / 0.443 / 0.391 / 0.422 / 0.428 / 0.425,
  not 0.584 / 0.540 / 0.542 / 0.537 / 0.552 / 0.520 / 0.581.
- The prose claim "global BSC slopes reached 0.584 at best" survives, but 0.584
  is the Random Survival Forest value, not the XGBoost AFT value.
- The prose claim "thirteen of the 91 combinations fell below 0.5" was already
  correct against `cv_results.csv`; the old table showed only five.

**BSC feature count.** The manuscript said 35 summary features per scan and the
feature-category table summed to 33. The per-scan file carries 33
(`bsc_simple_features_merged.csv`, 38 columns less 5 identifier columns). Both
places now read 33.

**Citations.** `hafeez2024deep` and `baytas2024predicting` are present in
`references.bib` but are not cited anywhere in the manuscript, so the spec's
concern about them no longer applies. The claims still carrying citations were
checked: Mitchell and Shiri-Feshki 2009 is a meta-analysis of 41 inception
cohorts reporting an annual conversion rate in the 5-10% range, and Ansart et
al. 2021 reviewed 234 experiments from 111 articles and found T1-weighted MRI
did not significantly improve prediction. Both match the sentences they support.

## Limitations that must stay stated in the manuscript

**Death ascertainment is split across two incompatible tables.** ADNI records
death in TREATDIS (withdrawal reason, exact form date, covering ADNI1, ADNIGO
and ADNI2) and in ADVERSE (`AEHDTHDT`, covering ADNI3 and ADNI4). The two do not
overlap for any cohort subject. `AEHDTHDT` is de-identified to a year, so 25 of
the 61 deaths in the cohort carry a year rather than a date and were placed at
the midpoint of that year. The withdrawal reason code for death is 2 in the
ADNI1 data dictionary and 1 in every later one, and is mapped per protocol.

**Deaths are recorded after administrative censoring.** For the 28 subjects who
died without a dementia diagnosis, the death date falls a median of 0.95 years
after the last clinical visit. The competing-risks analysis therefore extends
follow-up to the death date. Whether a subject converted between the last visit
and death is unobservable, and no assumption about it is defensible from these
data.

**Death ascertainment is almost certainly incomplete.** Only 61 of 417 subjects
carry any death record. In a cohort of mean age 76.9 followed for several years
the true number is higher; deaths occurring after a subject left the study are
not captured by either table. The competing-risks analysis is therefore a lower
bound on the competing-event burden, which is stated in the manuscript.

**Residualizing on scanner identity uses a reduced encoding.** The LongComBat
batch variable has 120 levels across 417 subjects, so residualizing 124 regional
features on it would remove variance mechanically rather than informatively. The
scanner design matrix pools sites contributing fewer than ten subjects, giving
12 retained site indicators plus vendor and field strength. This measures coarse
acquisition structure, not the full batch structure, and the manuscript says so.

## Results that did not come out the way the spec anticipated

**Partialling image quality out of the regional-BSC outcome association does not
shrink it.** The spec expected the outcome association to move toward zero once
quality was controlled. The out-of-fold regional-BSC risk score gives a hazard
ratio of 1.52 per SD raw and 1.57 per SD adjusted for the signal-to-noise and
brain-to-background slopes. Reported as measured.

**Residualizing regional BSC on scanner identity costs little discrimination.**
0.641 before, 0.630 after, a drop of 0.011. The features entering that model
have already been through longitudinal ComBat, which is the most likely reason.
Reported as measured, and not presented as support for the strong claim.

**Within subject, BSC also tracks brain parenchymal fraction.** The coefficient
is -0.32 per SD, so boundary sharpness rises as parenchymal fraction falls. This
is a biological association and it is reported, with the note that its sign runs
against a reading of BSC as a measure of tissue integrity.

## Not attempted

The normalized-BSC variant (BSC divided by a local image-sharpness estimate)
described as optional in the spec was not built. It requires recomputing BSC
from the images rather than working from the frozen per-scan features, which the
spec rules out. It stays in the manuscript as future work.
