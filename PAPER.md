# ConstructiveMultiFloorNet: Progressive Beam-Search Neural Growth with Raw-Feature Passthrough and Multi-Depth Ensembling

**Author:** Arsenii Shabrov  
**Date:** November 2026  

## Abstract
Classical Group Method of Data Handling (GMDH) algorithms construct polynomial networks layer by layer but suffer from combinatorial explosion and information loss because deeper layers only observe outputs from the immediately preceding layer. We present **ConstructiveMultiFloorNet**, a modern PyTorch constructive neural architecture that overcomes both limitations through three mechanisms: (1) **beam-search candidate generation** over compact micro-networks, (2) **persistent raw-feature passthrough**, allowing deep floors to directly combine prior representations with original raw features, and (3) **multi-depth leader boosting**, which ensembles the Top-$K$ winning architectures from every depth floor (1-floor through $N$-floor models) rather than discarding intermediate depths. Evaluated on the real-world **Spaceship Titanic** dataset (`8,693` samples), test ROC AUC improves monotonically at every constructed floor (`0.8439 → 0.8589 → 0.8720`), and the final multi-floor ensemble of 24 diverse sub-networks reaches **`0.8783` Test AUC (`79.35%` Accuracy)**, closely matching full-width MLPs (`0.8798`) and LightGBM (`0.8802`).

## 1. Progressive Floor Construction
At Floor $f = 1$, candidate micro-networks evaluate individual raw features as well as beam-sampled pairs and triples. The Top-$K$ ($K=8$) performers on a held-out validation split are retained, and their normalized logits $\tanh(0.5 z)$ are appended to a growing feature pool.

At Floors $f \ge 2$, candidate architectures are sampled from combinations linking **prior-floor leaders directly with original raw features** alongside multi-leader interactions. This prevents deep-layer signal degradation and enables interpretable cross-depth feature discovery (e.g., `[Raw(FoodCourt), Raw(Spa), F2_M5]`).

## 2. Multi-Depth Leader Ensembling
Instead of keeping only the deepest floor, all $N \times K$ winning micro-networks across all floors—spanning shallow single-feature detectors to 3-floor composite sub-networks—are combined via an $L_2$-regularized meta-booster calibrated on the validation split.

## 3. Experimental Results (Spaceship Titanic)
* **Floor 1 (Top-8 shallow micro-nets):** `0.8439` Test AUC | `74.47%` Test Acc
* **Floor 2 (Top-8 2-floor micro-nets):** `0.8589` Test AUC | `77.97%` Test Acc
* **Floor 3 (Top-8 3-floor micro-nets with raw passthrough):** `0.8720` Test AUC | `79.02%` Test Acc
* **Final Multi-Floor Ensemble (all 24 leaders):** **`0.8783` Test AUC** | **`79.35%` Test Acc**
* **Standard 2-Layer MLP Baseline:** `0.8798` Test AUC | `79.85%` Test Acc
* **LightGBM Gradient Boosting Baseline:** `0.8802` Test AUC | `79.99%` Test Acc
