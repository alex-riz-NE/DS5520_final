# DS5520_final

# Unsupervised Fraud Detection for Job Postings

This project applies unsupervised learning techniques to identify potentially fraudulent job postings. The pipeline combines text vectorization, representation learning, density-based clustering, and ensemble anomaly scoring to flag suspicious listings without relying on labeled training data.

This project was completed as part of a course assignment and is designed to be reproducible and easy to run.

---

## Model

The model follows a multi-stage unsupervised approach:

1. **Text Vectorization**  
   Job descriptions are converted into numerical features using TF-IDF. The dimensionality of the TF-IDF representation is treated as a tunable hyperparameter.

2. **Representation Learning**  
   Two reconstruction-based methods are used:
   - **PCA reconstruction**, which provides a linear baseline
   - **Autoencoder reconstruction**, which learns a non-linear latent representation of the data

   Reconstruction errors from these methods contribute to the anomaly scoring process.

3. **Clustering and Density Estimation**  
   HDBSCAN is applied to identify dense regions in the learned feature space. Observations that do not belong to any stable cluster are treated as potential outliers.

4. **Ensemble Anomaly Scoring**  
   Reconstruction errors and HDBSCAN outlier scores are combined into an ensemble score. A percentile-based threshold is applied to control the final anomaly rate.

---

## Project Structure

- Final model outputs and summaries are stored in `results/final_v1/`
- Hyperparameter tuning results are saved to `results/tuning_summary.csv`
- All figures generated during analysis are saved in the `fig/` directory

---

## Results

All outputs from the final model run are written to disk rather than displayed inline.

- Numeric results, including anomaly scores and cluster statistics, are saved in:


## Running the project

To create a model with hypertuned parameters:

- make tune

To run the final model: 
- First look at the tuning_summary.csv located in the results file. Pick a set of  hyper parameters that have mid-range TF-IDF, moderate HDBSCAN settings, and around a 5% anomaly rate since that is the portion of fraud in the data. This makes sure that there are enough clusters, but not too many

- make run

## Create Graphs and Charts

- make plots