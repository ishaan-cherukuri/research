# Evaluation Strategies for Limited Sample Size (456 subjects)

## The Problem
With only 456 subjects (96 events), we need to maximize data usage while getting reliable performance estimates.

## ✅ RECOMMENDED APPROACH

### **Use Stratified Cross-Validation for Evaluation**
- **What**: 5-fold CV using all 456 subjects
- **Performance**: Correlation = 0.855 ± 0.036, AUC = 0.961 ± 0.025
- **Why**: 
  - Uses ALL data (no waste)
  - Balanced event distribution
  - Robust, unbiased performance estimate
  - Lower variance than single train/test split

### **Train Final Model on ALL Data**
- **What**: Train on all 456 subjects for deployment
- **Why**:
  - Maximum information = best predictions
  - Report CV results as expected performance
  - Standard practice in ML with limited data

## Comparison of Strategies

| Strategy | Train Size | Test Size | Pros | Cons | When to Use |
|----------|-----------|-----------|------|------|-------------|
| **Stratified 5-Fold CV** | 365 per fold | 91 per fold | ✓ Uses all data<br>✓ Balanced events<br>✓ Robust estimate | ⚠ Computationally expensive | **Always** (for evaluation) |
| **Hold-out Split** | 364 (77 events) | 92 (19 events) | ✓ Simple<br>✓ Fast | ✗ Wastes 20% data<br>✗ High variance | Quick experiments only |
| **Train on All Data** | 456 (96 events) | 0 | ✓ Best model<br>✓ Max information | ✗ No test performance | **Always** (for deployment) |

## Our Results

### Stratified Cross-Validation (Primary Metric)
```
Correlation:  0.855 ± 0.036  ✓ Excellent time prediction
AUC:          0.961 ± 0.025  ✓ Excellent risk discrimination  
RMSE:         99.5 ± 11.1 years
MAE:          69.6 ± 8.1 years
```

### Hold-out Test (Single Split)
```
Correlation:  0.834  ← More variable (depends on random split)
AUC:          0.951  ← Similar to CV
RMSE:         90.3 years
```

## Best Practice Workflow

```
1. Run stratified 5-fold CV
   → Get performance: 0.855 correlation, 0.961 AUC
   
2. Train final model on ALL 456 subjects
   → Save as deployment model
   
3. Report CV results as expected performance
   → "5-fold CV: AUC = 0.961 ± 0.025"
   
4. Use final model for predictions
   → Best possible model (uses all data)
```

## Why NOT Use Simple Train/Test Split?

With only 96 events:
- **20% split** = only ~19 events in test set
- High variance in performance estimate
- Wastes 92 subjects that could improve the model
- Performance depends heavily on which subjects end up in test set

## Advanced Options (if needed)

### Nested Cross-Validation
For hyperparameter tuning with unbiased evaluation:
- Outer loop: 5-fold for evaluation
- Inner loop: 5-fold for hyperparameter selection
- More robust but computationally expensive

### Leave-One-Out CV
- Ultimate data efficiency (train on 455, test on 1)
- Computationally expensive (456 models)
- High variance for survival metrics

### Time-Based Validation
If you want to test temporal generalization:
- Train on subjects enrolled 2005-2010
- Test on subjects enrolled 2011-2015
- Requires sufficient data in each period

## Files Generated

```
survival_models/
├── final_model_all_data.json  ← Use this for predictions
├── scaler.pkl                  ← Feature scaling
├── config.json                 ← Model hyperparameters
├── cv_results.json            ← CV performance (report this)
├── cv_results.png             ← Visualization
└── test_predictions.png       ← Hold-out results
```

## Summary

**For your 456-subject dataset:**
1. ✅ **Report**: Stratified 5-fold CV results (0.961 AUC)
2. ✅ **Deploy**: Model trained on all 456 subjects
3. ❌ **Don't**: Waste data with simple 80/20 split

This approach is standard in medical ML with limited samples and gives you the best of both worlds: reliable performance estimates AND the best possible model.
