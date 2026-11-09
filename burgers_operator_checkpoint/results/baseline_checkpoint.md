# Burgers solution operator: baseline checkpoint

## Scope and data
- Governing PDE: viscous Burgers equation, periodic [0,1), viscosity 0.01, T=0.25.
- Simulated pairs: 1200 (800 train, 200 validation, 200 locked test), 64 sensors each.
- Training subset sizes: [50, 100, 200, 400, 800]; repeated seeds: [4020, 4021, 4022, 4023, 4024].
- **Test policy:** the locked test file was never read during baseline model selection.

## Validation findings (not final test results)
- Persistence (predict u(x,T)=u(x,0)) relative L2 error: 0.70893.
- At N=800, PCR mean validation relative L2 across five subsets: 0.31875.
- At N=800, ridge-PCR mean validation relative L2: 0.31875.
- Ridge improvement against PCR at N=800: -0.00% (positive favors ridge).
- PCR improves by 55.0% over persistence at N=800.
- PCR error decreases by 13.9% between N=50 and N=800.
- For the N=800, seed-4020 PCR fit: input rank 8,
  output rank 8, and output-PCA projection floor
  0.0748 on validation.
- Baseline analysis runtime: 1.4 seconds.

| Training functions N | PCR mean +/- seed SD | ridge-PCR mean +/- seed SD |
| ---: | ---: | ---: |
| 50 | 0.3701 +/- 0.0113 | 0.3698 +/- 0.0117 |
| 100 | 0.3447 +/- 0.0058 | 0.3447 +/- 0.0058 |
| 200 | 0.3258 +/- 0.0022 | 0.3258 +/- 0.0022 |
| 400 | 0.3198 +/- 0.0015 | 0.3198 +/- 0.0015 |
| 800 | 0.3187 +/- 0.0000 | 0.3187 +/- 0.0000 |

See `validation_summary.csv` for all N, means, and seed-to-seed standard
deviations. The error bars in `figures/validation_learning_curves.png`
show **one standard deviation across five training subsets**, not 95% CIs.

## Interpretation and cautions
- PCR and ridge-PCR are both linear mappings between learned PCA coefficients.
- Input functions lie in a 10-dimensional Fourier subspace; therefore the
  input PCA search is restricted to 2, 4, 6, 8, or 10 components.
- Nonlinear Burgers evolution can generate output modes beyond those 10 inputs;
  output rank is separately tuned among 4, 8, 12, 16, 24, and 32 components.
- Nearly identical ridge/PCR errors suggest the dominant limitation is the
  **linear input-to-output mapping** rather than overfitting in this setting.
  The PCA projection floor is below total PCR prediction error, consistent
  with a gap that improved linear fitting alone may not close.
- Validation error decreases noticeably up to 200 functions but levels off
  between 400 and 800. DeepONet will test whether a nonlinear operator
  approximation can reduce this apparent error floor.
- At N=800, all five subset seeds contain the same complete training pool.
  Their near-zero SD is a design consequence, **not** evidence of zero
  uncertainty. At smaller N the SD measures sensitivity to subsampling
  this fixed pool, not variation across independently simulated datasets.
- Model selection and reported performance both use the validation set; scores
  are exploratory and can be optimistic. Only the planned final evaluation
  on untouched test functions can support generalization claims.
- No DeepONet was trained in this checkpoint; that is the next comparison.

## Next steps
1. Share the data format and training/validation splits with Sudarshan.
2. Train DeepONet on the same five nested subset sizes and evaluate on validation.
3. Freeze architectures, PCA ranks, penalties, and other tuning decisions.
4. Evaluate all frozen models on the locked test functions once, near Nov 30.
5. Compute paired function-level bootstrap intervals on *test* differences.
