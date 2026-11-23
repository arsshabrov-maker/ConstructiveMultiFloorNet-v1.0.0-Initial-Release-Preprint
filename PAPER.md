# ConstructiveMultiFloorNet: Progressive Beam-Search Neural Growth with Raw-Feature Passthrough and Error-Reweighted Floors

**Author:** Arsenii Shabrov  
**Date:** November 2026  

## Abstract
Classical Group Method of Data Handling (GMDH) algorithms construct polynomial networks layer by layer but suffer from combinatorial explosion, loss of raw input signals in deep layers, and uniform sample weighting. We introduce **ConstructiveMultiFloorNet**, a PyTorch constructive neural boosting framework combining four mechanisms: (1) **beam-search micro-network selection**, (2) **persistent raw-feature passthrough** at every new floor, (3) **inter-floor error reweighting**, which amplifies loss penalties on misclassified samples so deeper floors specialize on hard boundary cases without corrupting target labels with residual noise, and (4) **multi-depth leader ensembling** across all floors (`1-floor` to `N-floor` sub-networks). Evaluated on the real-world **Spaceship Titanic** dataset (`8,693` samples), progressive floor construction improves test ROC AUC monotonically (`0.8439 → 0.8589 → 0.8720 → 0.8783`), and inter-floor error reweighting boosts clean test accuracy from `79.07%` to **`79.58%`** (`+0.0036` AUC), rivaling full-width MLPs (`79.85%`) and LightGBM (`79.99%`).

## 1. Progressive Floor Construction & Raw-Feature Passthrough
At Floor $f = 1$, candidate micro-networks evaluate single raw features as well as beam-sampled pairs and triples. At Floors $f \ge 2$, candidates combine **prior-floor leaders directly with original raw features** (e.g., `[Raw(FoodCourt), Raw(Spa), F2_M5]`), preventing deep-layer signal degradation.

## 2. Inter-Floor Error Reweighting
Instead of training deeper floors on noisy target residuals $y - \hat{p}$, each floor trains on clean binary labels $y \in \{0, 1\}$ with dynamically updated sample weights:
$$w_i^{(f+1)} \propto 1 + \gamma \, |y_i - \hat{p}_i^{(f)}|^{1.5}$$
where $\gamma = 2.5$ focuses subsequent beam searches onto hard misclassified examples while keeping target supervision clean.

## 3. Experimental Verification (*Spaceship Titanic*)
* **Progressive Floor Growth (AUC-selected):** Floor 1 (`0.8439`) → Floor 2 (`0.8589`) → Floor 3 (`0.8720`) → Final 24-Model Multi-Floor Ensemble (**`0.8783` Test AUC**, **`79.35%` Acc**).
* **Ablation — Inter-Floor Error Reweighting:** Increases clean test accuracy from `79.07%` (`0.8709` AUC) to **`79.58%`** (`0.8745` AUC).
