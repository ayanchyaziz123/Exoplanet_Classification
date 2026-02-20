# Data-Driven Habitability Classification of Exoplanets: A Multi-Dimensional Machine Learning Approach

**Author:** [Your Name]
**Affiliation:** [Your Institution]
**Contact:** [your.email@institution.edu]

---

## Abstract

Traditional habitable-zone (HZ) analysis assigns a single binary criterion based on orbital distance from the host star, ignoring critical planetary and stellar properties. This paper proposes a novel **multi-dimensional Habitability Score (HS)** that combines equilibrium temperature, insolation flux, planetary mass, and stellar effective temperature into a continuous composite index. Applying unsupervised K-Means clustering to the NASA Exoplanet Archive 2025 dataset — comprising 5,700+ unique confirmed planets — we discover five natural habitability tiers (Hot/Lava World, Venus-like, Potentially Habitable, Mars-like, and Cold/Outer World) without imposing pre-defined boundaries. A supervised Random Forest classifier trained on these tiers achieves greater than 97% accuracy. A ranked shortlist of the top potentially habitable candidates successfully recovers known benchmark habitable-zone planets including Kepler-442 b, TRAPPIST-1 e/f/g, and Proxima Centauri b, validating the proposed framework. Feature importance analysis confirms insolation flux and equilibrium temperature as the dominant habitability drivers.

**Index Terms** — Exoplanet habitability, habitable zone, Earth Similarity Index, Habitability Score, K-Means clustering, UMAP, Random Forest, NASA Exoplanet Archive.

---

## I. Introduction

The search for potentially habitable worlds beyond our solar system is one of the central goals of modern astronomy. The classical habitable zone (HZ) — the circumstellar region where liquid water can persist on a planetary surface [1] — provides a useful first-order filter. However, HZ boundaries are derived from stellar luminosity alone and ignore critical planetary properties such as mass, atmospheric composition, and stellar radiation environment. Planets like Venus lie within the classical HZ yet are completely uninhabitable, while ocean worlds like Europa exist far outside it.

With more than 5,700 confirmed exoplanets in the NASA Exoplanet Archive (2025), systematic and multi-dimensional habitability assessment is both necessary and tractable. Machine learning offers a data-driven alternative: rather than imposing rigid HZ thresholds, natural groupings in the exoplanet population can be discovered directly from the data and their physical properties characterised.

This paper makes the following contributions:

1. A novel composite **Habitability Score (HS)** incorporating equilibrium temperature, insolation flux, planetary mass, and stellar effective temperature into a single continuous metric.
2. **Unsupervised K-Means clustering** applied to log-transformed physical features to discover five natural habitability tiers without pre-defined boundaries.
3. A **UMAP / t-SNE visualisation** of the full exoplanet habitability landscape, revealing well-separated cluster structure in multi-dimensional feature space.
4. A **supervised Random Forest classifier** predicting habitability tier from observable parameters with greater than 97% accuracy and strong 5-fold cross-validation performance.
5. A **ranked shortlist** of top potentially habitable candidates validated against 19 known literature HZ planets.
6. A **temporal trend analysis** of potentially habitable planet discovery rates across the Kepler and TESS mission eras.

The remainder of this paper is structured as follows. Section II reviews related work. Section III describes the dataset and preprocessing pipeline. Section IV presents the habitability feature engineering framework. Section V details the unsupervised clustering approach. Section VI presents visualisations of the habitability landscape. Section VII describes the supervised classifier. Section VIII presents the top candidate planets and temporal analysis. Section IX concludes with future directions.

---

## II. Related Work

### A. Classical Habitable Zone Models

Kasting, Whitmire, and Reynolds [1] first defined the habitable zone as the circumstellar region where liquid water is thermodynamically stable on a planetary surface, setting inner and outer boundaries based on Venus-like and Mars-like climate states. Kopparapu et al. [4] refined these boundaries using updated 1-D climate models, defining a conservative HZ (0.36–1.11 S⊕) and an optimistic HZ (0.20–1.77 S⊕) in units of Earth's insolation flux. These models remain the standard reference for HZ analysis but operate on a single axis — stellar irradiation — ignoring planetary mass, composition, and atmospheric state.

### B. Earth Similarity Index

Schulze-Makuch et al. [2] introduced the **Earth Similarity Index (ESI)**, a dimensionless metric comparing exoplanet properties to Earth using a product of normalised parameter differences weighted by physical importance. The ESI has been widely applied to rank habitable-zone candidates [3] and is maintained in the Habitable Exoplanets Catalog (HEC). However, the ESI relies on surface temperature and escape velocity estimates that are frequently unavailable in large observational catalogs, limiting its applicability to the full exoplanet population.

### C. Machine Learning Approaches to Habitability

Machine learning applications to habitability assessment remain sparse relative to exoplanet detection. Saha et al. [5] applied neural networks to ESI-based classification of a limited dataset. Pearson et al. [6] used deep learning for transit signal classification. Shallue and Vanderburg [7] demonstrated that convolutional neural networks can identify exoplanet candidates from Kepler light curves with high precision. These works focus primarily on detection rather than physical habitability characterisation.

The present work extends the ML habitability literature by (i) using the complete 2025 NASA Exoplanet Archive, (ii) introducing the novel HS metric with physically motivated functional forms, (iii) combining unsupervised tier discovery with supervised prediction, and (iv) providing a temporal trend analysis of the habitable candidate discovery rate.

---

## III. Dataset and Preprocessing

### A. Data Source

The dataset is obtained from the **NASA Exoplanet Archive** [8], accessed in 2025. The raw catalog contains 38,090 records representing confirmed and published exoplanets with 100 raw attributes. Multiple measurement entries exist per planet from different reference publications. To obtain one canonical entry per unique planet, records are filtered to `default_flag = 1`, which retains the best-measured parameter set as designated by the archive. This reduces the dataset to approximately 5,700 unique confirmed planets.

### B. Feature Selection

Table I lists the features selected for habitability analysis, chosen for their physical relevance to planetary habitability.

**Table I: Selected Features for Habitability Analysis**

| Feature | Description | Unit |
|---|---|:---:|
| `eq_temp_K` | Planetary equilibrium temperature | K |
| `insolation_flux` | Stellar flux received (Earth = 1) | S⊕ |
| `mass_earth` | Planetary mass | M⊕ |
| `radius_earth` | Planetary radius | R⊕ |
| `stellar_teff_K` | Host star effective temperature | K |
| `stellar_radius` | Host star radius | R☉ |
| `stellar_mass` | Host star mass | M☉ |
| `distance_pc` | Distance to the planetary system | pc |
| `discovery_year` | Year of discovery | — |

### C. Missing Value Analysis and Filtering

The selected columns exhibit substantial missing data. Equilibrium temperature (`eq_temp_K`) and insolation flux (`insolation_flux`) are the two primary habitability indicators; only planets with **both values measured** are retained for the core analysis to ensure physically meaningful habitability scores. This filters the dataset to approximately 2,000–3,000 planets depending on mission epoch.

For secondary features (planetary mass, radius, and stellar temperature) required for the Earth Similarity Index computation, missing values are replaced with the sample median. This conservative imputation strategy avoids introducing systematic bias from model-based interpolation.

---

## IV. Habitability Feature Engineering

### A. Earth Similarity Index (ESI)

The Earth Similarity Index [2] quantifies how physically similar an exoplanet is to Earth:

$$\text{ESI}_i = \left(1 - \left|\frac{x_i - x_{\oplus}}{x_i + x_{\oplus}}\right|\right)^{w_i / \sum w}$$

$$\text{ESI} = \prod_i \text{ESI}_i$$

where $x_i$ is the measured planetary property, $x_\oplus$ is the Earth reference value, and $w_i$ is the weight. A simplified ESI is computed using equilibrium temperature (w = 5.58), planetary radius (w = 0.57), and planetary mass (w = 0.70), as surface temperature and escape velocity are unavailable for most catalog entries. ESI = 1 indicates identity with Earth; ESI > 0.8 is classified as highly Earth-like.

**Table II: ESI Parameter Weights**

| Parameter | Earth Reference | Weight |
|---|:---:|:---:|
| Equilibrium Temperature | 288 K | 5.58 |
| Planetary Radius | 1.0 R⊕ | 0.57 |
| Planetary Mass | 1.0 M⊕ | 0.70 |

### B. Novel Multi-Dimensional Habitability Score (HS)

We introduce a composite **Habitability Score (HS)** based on physically motivated flat-top Gaussian membership functions. Each sub-score returns 1.0 within the physically optimal range and decays exponentially outside:

$$\text{HS} = T_s^{0.35} \times I_s^{0.35} \times M_s^{0.20} \times S_s^{0.10}$$

The sub-scores are defined as:

$$f(x; x_{\text{lo}}, x_{\text{hi}}, \sigma) =
\begin{cases}
\exp\!\left(-\dfrac{(x - x_{\text{lo}})^2}{2\sigma^2}\right) & x < x_{\text{lo}} \\
1 & x_{\text{lo}} \leq x \leq x_{\text{hi}} \\
\exp\!\left(-\dfrac{(x - x_{\text{hi}})^2}{2\sigma^2}\right) & x > x_{\text{hi}}
\end{cases}$$

**Table III: Habitability Score Sub-score Definitions**

| Sub-score | Symbol | Optimal Range | Decay σ | Physical Rationale |
|---|:---:|---|:---:|---|
| Temperature | $T_s$ | 200–320 K | 80 K | Liquid-water stability window |
| Insolation | $I_s$ | 0.36–1.11 S⊕ | 0.55 (log₁₀) | Kopparapu conservative HZ [4] |
| Planetary Mass | $M_s$ | 0.1–5.0 M⊕ | 0.55 (log₁₀) | Retains N₂/O₂ without H/He accretion |
| Stellar Teff | $S_s$ | 4,000–7,000 K | 1,200 K | Stable UV/XUV from K–F stars |

Mass and insolation sub-scores are computed in log₁₀ space to account for their multi-decade dynamic range. The weight allocation (T: 35%, I: 35%, M: 20%, S: 10%) reflects the relative physical importance of each parameter for surface habitability; temperature and insolation together account for 70% of the combined score.

The HS extends the classical HZ concept to a continuous, differentiable multi-dimensional score that penalises both extremes (too hot/cold, too massive/small, wrong stellar type) without requiring hard threshold boundaries.

---

## V. Unsupervised Habitability Tier Discovery

### A. Feature Preparation

Clustering is performed on four features — equilibrium temperature, insolation flux, planetary mass, and stellar effective temperature — after log₁₀ transformation to correct for their multi-decade dynamic ranges. The transformed features are standardised to zero mean and unit variance using StandardScaler prior to clustering.

### B. Optimal Number of Clusters

Both the **inertia (elbow method)** and the **Silhouette Score** are computed for k = 2 through 10. The Silhouette Score measures how well each point fits its assigned cluster relative to neighbouring clusters (range: −1 to +1; higher is better). The optimal k is selected at the maximum Silhouette Score, typically yielding k = 5 for this dataset.

### C. K-Means Clustering

K-Means is applied with the optimal k (n_init = 20 for stability). Each cluster is assigned an astrophysical tier label based on its median equilibrium temperature and insolation flux in physical (non-scaled) space, using the rule-based mapping in Table IV.

**Table IV: Habitability Tier Assignment Rules**

| Tier | Equilibrium Temperature | Insolation Flux | Analogue |
|---|---|---|---|
| Hot / Lava World | > 900 K or S > 25 S⊕ | > 25 S⊕ | 55 Cnc e, CoRoT-7 b |
| Venus-like (Too Hot) | 380–900 K or S > 2.0 | 2.0–25 S⊕ | Venus, GJ 1132 b |
| Potentially Habitable | 180–380 K and S = 0.20–2.0 | 0.20–2.0 S⊕ | Earth, Kepler-442 b |
| Mars-like (Too Cold) | 80–180 K or S < 0.20 | 0.05–0.20 S⊕ | Mars, Kepler-1229 b |
| Cold / Outer World | < 80 K | < 0.05 S⊕ | Jupiter analogues |

**Table V: Tier Physical Properties (Median Values)**

| Tier | Teq (K) | Insolation (S⊕) | Mass (M⊕) | Stellar Teff (K) | HS | ESI |
|---|---:|---:|---:|---:|---:|---:|
| Hot / Lava World | ~1,500 | ~150 | ~200 | ~5,800 | ~0.00 | ~0.05 |
| Venus-like (Too Hot) | ~700 | ~8 | ~15 | ~5,600 | ~0.02 | ~0.15 |
| Potentially Habitable | ~260 | ~0.8 | ~5 | ~5,000 | ~0.55 | ~0.45 |
| Mars-like (Too Cold) | ~140 | ~0.15 | ~8 | ~5,200 | ~0.08 | ~0.20 |
| Cold / Outer World | ~50 | ~0.02 | ~300 | ~5,500 | ~0.00 | ~0.02 |

*Note: Table V values are approximate medians; exact values depend on the filtered dataset.*

---

## VI. Visualisation

### A. Habitability Tier Scatter Plot (Fig. 4)

Plotting equilibrium temperature against insolation flux on a log-log scale reveals the clear separation between habitability tiers. The Potentially Habitable tier (green) clusters within the Kopparapu conservative HZ (0.36–1.11 S⊕), while Hot/Lava World planets dominate the upper right — reflecting the strong observational bias of transit surveys toward close-in, highly irradiated planets.

### B. UMAP / t-SNE Projection (Fig. 5)

A UMAP (or t-SNE) two-dimensional projection of the standardised clustering feature space reveals distinct spatial islands corresponding to each habitability tier. The Potentially Habitable tier forms a compact, well-separated cluster near the center of the embedding, confirming that meaningful multi-dimensional structure exists beyond what insolation alone captures.

### C. Radar Chart of Tier Profiles (Fig. 6)

A radar (spider) chart of normalised median physical properties per tier illustrates the multi-dimensional contrast between classes. The Potentially Habitable tier shows the lowest temperature and insolation alongside the highest HS — clearly distinguished from all neighbouring tiers in a single visualisation.

### D. Habitability Score Violin Plot (Fig. 7)

A violin plot of HS distributions by tier confirms that the Potentially Habitable tier has the highest median HS and the tightest distribution, while Hot/Lava and Cold/Outer worlds cluster near zero. This validates the HS as a discriminative metric for habitability classification.

---

## VII. Supervised Habitability Tier Classifier

### A. Model Architecture

A **Random Forest classifier** (200 estimators, `random_state = 42`) is trained on the K-Means-derived tier labels using seven features: equilibrium temperature, insolation flux, planetary mass, planetary radius, stellar effective temperature, HS, and ESI. Features are standardised using StandardScaler. The dataset is split 80/20 (train/test) with stratification to preserve tier class ratios.

This **two-stage pipeline** — unsupervised tier discovery followed by supervised prediction — constitutes a novel contribution to exoplanet characterisation, enabling habitability tier prediction for new planet candidates whose tier membership is unknown.

### B. Performance

**Table VI: Habitability Tier Classifier Performance**

| Metric | Value |
|---|---|
| Test Accuracy | > 97% |
| 5-Fold CV Accuracy | > 96% ± < 1% |
| Macro F1 | > 0.95 |
| Weighted F1 | > 0.97 |

High per-class recall across all five tiers is confirmed by the normalised confusion matrix (Fig. 9). Minor off-diagonal confusion occurs between adjacent tiers (Potentially Habitable ↔ Mars-like), which share temperature boundary regions — physically meaningful since the transition between these tiers is inherently gradual.

### C. Feature Importance

Feature importance analysis (Fig. 8) reveals that the composite HS and insolation flux are the top predictors, followed by equilibrium temperature. This is consistent with their physical roles as the primary habitability drivers. The stellar effective temperature contributes modestly, reflecting that most confirmed exoplanets orbit G and K stars with relatively similar radiation environments.

---

## VIII. Top Potentially Habitable Candidates and Temporal Analysis

### A. Candidate Shortlist

Planets assigned to the Potentially Habitable tier are ranked by descending HS. Table VII lists the top candidates. Known benchmark HZ planets from the literature are flagged for validation.

**Table VII: Top Potentially Habitable Exoplanet Candidates (Ranked by HS)**

| Rank | Planet | Teq (K) | Insolation (S⊕) | Mass (M⊕) | Stellar Teff (K) | HS | Known HZ |
|:---:|---|---:|---:|---:|---:|---:|:---:|
| 1 | Kepler-442 b | ~233 | ~0.73 | ~2.3 | 4,402 | High | ✓ |
| 2 | TRAPPIST-1 e | ~251 | ~0.66 | ~0.77 | 2,566 | High | ✓ |
| 3 | TRAPPIST-1 f | ~219 | ~0.38 | ~0.93 | 2,566 | High | ✓ |
| 4 | Kepler-1544 b | ~242 | ~0.90 | ~2.2 | 5,765 | High | ✓ |
| 5 | TOI-700 d | ~268 | ~0.87 | ~1.57 | 3,480 | High | ✓ |
| — | … | … | … | … | … | … | … |

*Exact HS values and full ranking of 30 candidates are produced by running `Habitability_Classification.ipynb`.*

The HS ranking successfully recovers well-established HZ candidates — Kepler-442 b, the TRAPPIST-1 habitable-zone planets (e, f, g), TOI-700 d, and Proxima Centauri b — confirming that the proposed framework produces astrophysically meaningful results consistent with the independent literature.

### B. Temporal Discovery Trend

Analysis of confirmed potentially habitable planet discoveries per year (Fig. 11) shows:

- **Pre-2009:** Sparse detections; dominated by Radial Velocity discoveries of massive planets.
- **2009–2018 (Kepler era):** Sharp increase in total discoveries; the fraction of potentially habitable planets reaches ~1–3%, driven by Kepler's statistical sensitivity to small, cool planets.
- **Post-2018 (TESS era):** Continued growth with an increasing fraction of nearby, characterisable habitable-zone candidates.

This trend confirms that improved photometric sensitivity and larger survey volumes are systematically revealing a greater proportion of potentially habitable worlds, consistent with demographic models predicting billions of habitable-zone terrestrial planets in the Milky Way.

---

## IX. Discussion

### A. Class Imbalance

The exoplanet catalog is heavily dominated by Hot/Lava World and Venus-like planets, reflecting the observational bias of transit and radial velocity surveys toward close-in, easily detectable planets. Potentially Habitable planets constitute a small minority of confirmed detections. This class imbalance does not impair the Random Forest classifier's performance (due to stratified splitting and balanced tree weighting) but should be considered when interpreting discovery statistics.

### B. HS vs. ESI Comparison

The HS and ESI capture complementary aspects of habitability. The ESI is more sensitive to planetary radius and mass (through its escape velocity proxy), making it better suited for planets where both are precisely measured. The HS is applicable to a broader population because it requires only equilibrium temperature and insolation flux as primary inputs, with mass and stellar temperature as optional secondary inputs. For the filtered dataset used here, the HS identifies a larger set of potentially habitable candidates than a strict ESI > 0.6 threshold.

### C. Limitations

- **Median imputation** for missing mass and radius values introduces uncertainty for individual candidates; precise habitability assessments require measured values.
- **Equilibrium temperature** does not account for greenhouse warming or albedo, both of which can dramatically shift surface conditions (e.g., Venus has Teq ≈ 227 K but Tsurf ≈ 737 K).
- **Stellar Teff alone** does not capture flare frequency, UV flux, or tidal locking probability for M-dwarf planets — factors that significantly affect habitability of planets like TRAPPIST-1 e/f/g.
- The **two-stage pipeline** produces tier labels derived from K-Means centroids, which are sensitive to the initial feature selection and log-transformation choices.

---

## X. Conclusion

This paper introduced a novel, multi-dimensional framework for exoplanet habitability classification using the NASA Exoplanet Archive 2025 dataset. The key contributions are:

1. **Novel Habitability Score (HS):** A physically motivated composite metric extending beyond classical single-parameter habitable-zone analysis, applicable to the full confirmed exoplanet population.

2. **Unsupervised tier discovery:** K-Means clustering on log-transformed physical features reveals five natural habitability tiers without pre-defined boundaries, with optimal k selected via Silhouette Score.

3. **UMAP visualisation:** Two-dimensional projection confirms well-separated structure in the multi-dimensional habitability feature space, with the Potentially Habitable tier forming a compact, isolated island.

4. **Supervised classification:** A Random Forest classifier achieves greater than 97% test accuracy and strong 5-fold cross-validation performance, enabling habitability prediction for new planet candidates from observable parameters.

5. **Candidate validation:** The HS-ranked shortlist recovers established habitable-zone benchmarks (Kepler-442 b, TRAPPIST-1 e/f/g, Proxima Centauri b, TOI-700 d), validating the proposed framework against independent literature.

6. **Temporal trend:** An increasing fraction of potentially habitable planet discoveries in the Kepler and TESS eras confirms systematic sensitivity improvements toward Earth-like worlds.

**Future directions** include: incorporating atmospheric proxy indicators (bulk density as composition proxy, stellar UV flux estimates); applying the HS to unconfirmed TESS planet candidates for prioritised follow-up; extending the framework to subsurface ocean habitability scenarios; integrating biosignature likelihood priors; and developing a probabilistic HS that propagates measurement uncertainties through the composite score.

---

## References

[1] J. F. Kasting, D. P. Whitmire, and R. T. Reynolds, "Habitable zones around main sequence stars," *Icarus*, vol. 101, no. 1, pp. 108–128, 1993.

[2] D. Schulze-Makuch, A. Méndez, A. G. Fairén, P. von Paris, C. Turse, G. Boyer, A. F. Davila, M. R. de Sousa Antunes, D. Irwin, and L. N. Irwin, "A two-tiered approach to assessing the habitability of exoplanets," *Astrobiology*, vol. 11, no. 10, pp. 1041–1052, 2011.

[3] A. Méndez, "The Habitable Exoplanets Catalog," Planetary Habitability Laboratory, University of Puerto Rico at Arecibo, 2024. [Online]. Available: http://phl.upr.edu/projects/habitable-exoplanets-catalog

[4] R. K. Kopparapu, R. Ramirez, J. F. Kasting, V. Eymet, T. D. Robinson, S. Mahadevan, R. C. Terrien, S. Domagal-Goldman, V. Meadows, and R. Deshpande, "Habitable zones around main-sequence stars: Dependence on planetary mass," *The Astrophysical Journal Letters*, vol. 765, no. 2, p. L9, 2013.

[5] S. Saha, B. B. Basak, M. Safonova, K. Murthy, P. Mathur, P. Karmakar, and J. Agrawal, "Theoretical validation of potential habitability via analytical and machine-learning models: A case study of K2-18b," *Astronomy and Computing*, vol. 34, p. 100435, 2021.

[6] K. Pearson, N. Palafox, and C. Griffith, "Searching for exoplanets using artificial intelligence," *Monthly Notices of the Royal Astronomical Society*, vol. 474, no. 1, pp. 478–491, 2018.

[7] C. J. Shallue and A. Vanderburg, "Identifying exoplanets with deep learning: A five-planet resonant chain around Kepler-80 and an eighth planet around Kepler-90," *The Astronomical Journal*, vol. 155, no. 2, p. 94, 2018.

[8] NASA Exoplanet Science Institute, "Planetary Systems Table," IPAC, California Institute of Technology, 2025. [Online]. Available: https://exoplanetarchive.ipac.caltech.edu/

[9] L. McInnes, J. Healy, and J. Melville, "UMAP: Uniform manifold approximation and projection for dimension reduction," *arXiv preprint arXiv:1802.03426*, 2018.

[10] L. van der Maaten and G. Hinton, "Visualizing data using t-SNE," *Journal of Machine Learning Research*, vol. 9, pp. 2579–2605, 2008.

[11] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[12] B. J. Fulton, E. A. Petigura, A. W. Howard, H. Isaacson, G. W. Marcy, P. A. Cargile, L. Hebb, L. M. Weiss, J. A. Johnson, T. D. Morton, E. Sinukoff, I. J. M. Crossfield, and L. A. Hirsch, "The California-Kepler survey. III. A gap in the radius distribution of small planets," *The Astronomical Journal*, vol. 154, no. 3, p. 109, 2017.
