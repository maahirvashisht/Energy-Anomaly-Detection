# EAI Lab – Viva Questions & Answers (with results from the scripts)

## Experiment 1 – Measures of Central Tendency (exp1.py)
**Q: What are MOCT?** Mean, median, mode – single values that describe the centre of data.
**Q: Mean vs median vs mode?** Mean = sum/n (uses every value); median = middle value of sorted data; mode = most frequent value.
**Q: Which model/technique is used here? Accuracy?** None – descriptive statistics only. There is no model, so no accuracy.
**Q: Effect of an outlier?** Mean 14.0 → 30.91 after adding 200; median 14.5 → 15.0; mode stays 15. Mean is not robust, median/mode are.
**Q: Why is mode odd for continuous data?** Every value is usually unique, so mode is meaningless (we got 30.4, the first value). Bin the data first (histogram peak).
**Q: axis=0 vs axis=1 in a 2D array?** axis=0 → per column; axis=1 → per row.
**Q: Does df.mode() return one value?** No – it returns all modes (a DataFrame), hence `.iloc[0]`.
**Q: Which measure for salary data?** Median – salary is skewed with outliers.

## Experiment 2 – MOCT vs distribution shape, transformations (exp2.py)
**Q: Relationship in symmetric / right / left skew?** Symmetric: mean=median=mode (50.03, 50.03, 49.96). Right skew: mode<median<mean (4.9<8.4<10.0). Left skew: mean<median<mode (90.0<91.6<94.5).
**Q: Why does the mean follow the tail?** It uses actual values, so extreme tail values pull it; median/mode depend on position/frequency.
**Q: Empirical relation?** Mean − Mode ≈ 3 (Mean − Median) for moderately skewed data.
**Q: Effect of shift (x+c)?** Mean, median, mode all shift by +c.
**Q: Effect of scale (a·x)?** All three multiply by a.
**Q: Mixed (a·x+b)?** new MOCT = a·old MOCT + b (verified True in the script).
**Q: Model/accuracy?** None – descriptive analysis.

## Experiment 3 – Measures of spread, quartiles (exp3.py)
**Q: Define range, variance, SD.** Range = max−min; variance = Σ(x−x̄)²/(n−1) (sample); SD = √variance (same units as data).
**Q: Why n−1?** Bessel's correction – gives an unbiased estimate of population variance.
**Q: Effect of outlier?** Range 12 → 90, SD 3.48 → 23.56, variance 12.08 → 554.9. All highly sensitive; IQR is robust.
**Q: Effect of Xnew = c·Xold + d?** Range & SD × |c|, variance × c²; the shift d has no effect on spread.
**Q: What is IQR and outlier rule?** IQR = Q3 − Q1 (here 42). Outliers lie outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR].
**Q: Q1, Q2, Q3 meaning?** 25th, 50th (median), 75th percentiles; each quartile holds 25 % of entries (10 of 40 each).
**Q: Boxplot relation?** Box = Q1–Q3, line = median, whiskers = fences.

## Experiment 4(a) – Proximity measures (exp4a.py)
**Q: Similarity vs dissimilarity?** Similarity ∈ [0,1], high = alike; dissimilarity = 0 for identical, usually 1 − similarity.
**Q: Nominal formula?** d = (p − m)/p; p = attributes, m = matches (0.667 in the example).
**Q: Symmetric vs asymmetric binary?** Symmetric: 0 and 1 equally important (d = (r+s)/(q+r+s+t)). Asymmetric: 1 is rare/important (d = (r+s)/(q+r+s), Jaccard) – negative matches t ignored (e.g. disease tests).
**Q: Manhattan vs Euclidean?** Manhattan = Σ|xi−yi| (=11), Euclidean = √Σ(xi−yi)² (=6.708). Both are Minkowski (h=1, h=2).
**Q: Ordinal handling?** Replace value by rank r, normalise z = (r−1)/(M−1), then use a numeric distance.
**Q: Cosine similarity?** (x·y)/(‖x‖‖y‖); measures angle, not magnitude – [1,2,3] and [2,4,6] give 1.0. Used for text/documents.
**Q: Why normalise numeric attributes?** So that large-scale attributes don't dominate.
**Q: Model/accuracy?** None – distance functions; they are building blocks for KNN / clustering.

## Experiment 4(b) – Mixed-type dissimilarity (exp4b.py)
**Q: Formula?** d(i,j) = Σf δij(f)·dij(f) / Σf δij(f).
**Q: What is δ?** Indicator: 0 if a value is missing or (asymmetric binary and both 0); else 1.
**Q: Why divide by Σδ?** So each attribute contributes on a [0,1] scale and averaged only over valid attributes.
**Q: Properties of the matrix?** Symmetric, zero diagonal, values in [0,1]. Closest pair here: O2–O5 (0.063); farthest: O3–O4 (0.971).
**Q: Where used?** KNN, clustering (e.g. K-Medoids/hierarchical) on mixed data (Gower-style distance).

## Experiment 5 – Sampling (exp5.py)
**Q: Random / cluster / stratified sampling?** Random: every unit has equal chance. Cluster: randomly pick whole groups. Stratified: split into homogeneous strata and sample from each.
**Q: Sample size formula?** n0 = Z²·p(1−p)/e² (370 for N=10,000, 95 % confidence, 5 % error) with finite population correction.
**Q: Cluster sample size?** SRS size × design effect (1.5) → 555 people ≈ 12 clusters of 50.
**Q: Proportional allocation?** n_h = n·N_h/N → [185, 111, 74].
**Q: When prefer which?** Stratified – diverse population, subgroup representation; cluster – cheap/geographically spread; random – homogeneous population.
**Q: How was the best sample chosen?** Goodness = |mean err| + |median err| + |std err| vs the population; Sample 3 was best (error 0.306).
**Q: Sampling vs non-sampling error?** Sampling error is chance difference between sample and population; it shrinks as n grows (∝ 1/√n).
**Q: Model/accuracy?** None – "accuracy" here means closeness of sample statistics to population parameters.

## Experiment 6 – K-Means (exp6.py)
**Q: Which model? Learning type?** K-Means – unsupervised, centroid-based, partitional clustering.
**Q: Algorithm steps?** 1) choose k, init centroids 2) assign each point to nearest centroid (Euclidean) 3) recompute centroids as means 4) repeat until centroids stop moving.
**Q: Objective?** Minimise SSE (within-cluster sum of squares): Σ‖x − μc‖². Converged to 78.85 on Iris in 6 iterations.
**Q: What is the accuracy?** K-Means has no labels, so accuracy is evaluated only because Iris has true classes: **89.33 %** after the best cluster→class mapping. Internal metric: **silhouette = 0.553**. Setosa is perfectly separated (50/50); versicolor and virginica overlap (cluster sizes 50/62/38).
**Q: How to choose k?** Elbow method (SSE vs k: 681 → 152 → 78.9 → 57 → … bend at k=3), silhouette score.
**Q: Disadvantages?** Need k beforehand; sensitive to init (local minima → run many restarts / K-Means++); sensitive to outliers and scale; assumes spherical, similar-sized clusters.
**Q: Why scale features?** Euclidean distance is dominated by large-range features.
**Q: Complexity?** O(n·k·i·d).
**Q: K-Means vs K-Medoids?** Medoids are actual data points → robust to outliers.
**Q: Does it always converge?** Yes (SSE never increases), but possibly to a local, not global, optimum.
