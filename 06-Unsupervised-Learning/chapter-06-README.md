# Chapter 6 — Unsupervised Learning

## What this chapter covers

Learning from data **without labels** — no right answers, no teacher. The algorithm finds structure on its own.

---

## The Big Idea

Every algorithm in Chapters 1–5 had labels. You gave the model houses with known prices. Emails with known spam tags. The model learned by comparing guesses to the right answers.

**Unsupervised learning has no labels.**

You throw raw data at the algorithm and say: *"Find the structure yourself."*

Two types of structure we look for:

**Clustering** — find natural groups. Similar things should end up together.

**Dimensionality Reduction** — compress many features into fewer while keeping the most important information.

---

## The Real World Thread

**Meena's FreshMart** — a supermarket chain with 6 branches across Kochi. 500 customers, loyalty card data, no labels. Goal: find natural customer segments for targeted marketing.

Same dataset runs through 6.1, 6.2, and 6.3 so you can directly compare what each algorithm finds and how they differ.

---

## Algorithms Covered

| Notebook | Algorithm | Core idea |
|---|---|---|
| 6.1 | K-Means Clustering | Assign points to nearest centroid, update centroids, repeat until convergence |
| 6.2 | Hierarchical Clustering | Build a tree of merges (dendrogram) — cut anywhere to get any K |
| 6.3 | DBSCAN | Find clusters by density — any shape, explicit outlier detection |
| 6.4 | PCA | Compress features into fewer dimensions keeping maximum variance |

---

## Key Concepts Introduced

**Inertia / WCSS** — sum of squared distances from each point to its centroid. K-Means minimises this.

**Elbow Method** — plot inertia vs K. The bend = good K.

**Silhouette Score** — measures how well each point fits its own cluster vs the nearest other cluster. Range -1 to +1, higher is better.

**Dendrogram** — tree diagram showing every merge in Hierarchical Clustering. Height = distance at merge. Cut at biggest gap for natural K.

**Linkage** — how distance between clusters is measured. Ward (minimise inertia increase) is the default and best for most data.

**Core / Border / Noise points** — DBSCAN's three point types. Noise points (label = -1) are genuine outliers — never forced into any cluster.

**ε and min_samples** — DBSCAN's two parameters. ε = neighbourhood radius. min_samples = minimum neighbours to be a core point.

**Principal Components** — PCA's output. Each component is a direction of maximum variance in the original feature space.

**Explained Variance Ratio** — how much of the original information each PCA component retains.

---

## Evaluation (Unsupervised is different)

No ground truth labels → no accuracy, no RMSE. Instead:

| Method | Used for | What it tells you |
|---|---|---|
| Elbow Method | K-Means | Diminishing returns on inertia as K increases |
| Silhouette Score | K-Means, Hierarchical, DBSCAN | Cohesion vs separation — higher is better |
| Dendrogram gap | Hierarchical | Natural number of clusters from merge distances |
| K-distance plot | DBSCAN | Find the right ε from nearest neighbour distances |
| Explained variance | PCA | How much information is kept after compression |

---

## Algorithm Comparison

| | K-Means | Hierarchical | DBSCAN | PCA |
|---|---|---|---|---|
| **Type** | Clustering | Clustering | Clustering | Dimensionality reduction |
| **Specify K upfront?** | Yes | No | No | Yes (n_components) |
| **Any cluster shape?** | No — round only | Ward=No, Single=yes | Yes | N/A |
| **Outlier detection?** | No | No | Yes — label -1 | N/A |
| **Scales to large data?** | Yes | No — O(n²) | Yes | Yes |
| **Reproducible?** | No — random init | Yes | Yes | Yes |
| **Best for** | Large data, round clusters | Small data, exploring structure | Any shape + outliers | Visualisation, preprocessing |

---

## Algorithm Selection Guide

| Situation | Use |
|---|---|
| Large dataset, roughly round clusters | K-Means |
| Don't know K, want to explore structure | Hierarchical |
| Clusters are irregular shapes | DBSCAN |
| Have outliers you want explicitly flagged | DBSCAN |
| Dataset < 5,000 rows, know K | Hierarchical (better quality, deterministic) |
| Need to visualise high-dimensional data | PCA → scatter plot |
| Too many features, want to reduce before clustering | PCA first, then cluster |

---

## Important Rules

**Always scale before clustering** — K-Means, Hierarchical, and DBSCAN are all distance-based. Large-scale features dominate distance without StandardScaler.

**Cluster labels are arbitrary** — K-Means cluster 0 is not better than cluster 2. Meaning comes from analysing what each cluster looks like (feature means per cluster).

**Unsupervised evaluation is subjective** — silhouette score guides you, but final interpretation always requires domain knowledge.

---

## Chapter Project

**FreshMart Customer Segmentation** — find natural customer groups from loyalty card data (income, spending score, visit frequency, basket size). Use K-Means for segmentation, DBSCAN to flag unusual customers, PCA to visualise. Deliver targeted marketing recommendations for each segment.

---

## Prerequisites

- Chapter 2 (Data Skills — scaling, EDA)
- Chapter 3 (Evaluation — silhouette score builds on the metrics mindset)

## Environment

```
Python 3.11.9 | ml-env virtual environment
sklearn, numpy, pandas, matplotlib, scipy
```
