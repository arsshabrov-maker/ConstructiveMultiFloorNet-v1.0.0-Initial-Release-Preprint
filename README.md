# ConstructiveMultiFloorNet: Progressive Neural Growth with Raw-Feature Passthrough & Error-Reweighted Floors

**ConstructiveMultiFloorNet** grows a multi-depth neural ensemble floor by floor using **beam-search architecture selection**, **persistent raw-feature passthrough**, **inter-floor error reweighting**, and **multi-depth leader ensembling**.

---

## 💡 Core Architectural Innovations

1. **Beam-Search Micro-Network Growth:** Evaluates a focused beam of compact micro-networks (`MicroConstructiveNet`) at each depth floor and promotes the Top-$K$ winners.
2. **Direct Raw-Feature Passthrough:** Every new floor combines prior-floor winners **directly with raw input features** (e.g., discovering `[Raw(FoodCourt), Raw(Spa), F2_M5]`), preventing deep-layer information loss.
3. **Inter-Floor Error Reweighting:** After each floor, sample weights are amplified on misclassified instances ($w_i \propto 1 + \gamma |y_i - \hat{p}_i|^{1.5}$), forcing deeper floors to discover specialized feature combinations for hard boundary cases while preserving clean target labels.
4. **Multi-Depth Ensemble (All Floors Combined):** Aggregates all 24 winning micro-networks across Floors 1, 2, and 3 into a single meta-booster.

---

## 📊 Experimental Results (Real-World *Spaceship Titanic* Dataset)

Evaluated on a strict 3-way split (`Train: 4,563` | `Val: 1,956` | `Clean Test: 2,174` samples):

### 1. Progressive Floor-by-Floor Growth

| Stage / Model | Top Discovered Structure | Test ROC AUC ↑ | Test Accuracy ↑ |
| :--- | :--- | :---: | :---: |
| **Constructive Floor 1 (Top-8)** | `[Raw(HomePlanet), Raw(RoomService), Raw(VRDeck)]` | `0.8439` | `74.47%` |
| **Constructive Floor 2 (Top-8)** | `[F1_M1, F1_M2]` | `0.8589` | `77.97%` |
| **Constructive Floor 3 (Top-8)** | `[Raw(FoodCourt), Raw(Spa), F2_M5]` | `0.8720` | `79.02%` |
| **Final Multi-Floor Ensemble (24 Nets)** | **All 1-, 2-, and 3-Floor Leaders Combined** | **`0.8783`** | **`79.35%`** |
| Classical 2-Layer MLP (Baseline) | Dense `10 -> 64 -> 64 -> 1` | `0.8798` | `79.85%` |
| LightGBM (Gradient Boosting Baseline) | 300 Boosted Decision Trees | `0.8802` | `79.99%` |

### 2. Ablation: Inter-Floor Error Reweighting

| Configuration | Clean Test ROC AUC ↑ | Clean Test Accuracy ↑ |
| :--- | :---: | :---: |
| ConstructiveMultiFloorNet (Equal Sample Weights) | `0.8709` | `79.07%` |
| **ConstructiveMultiFloorNet (+ Inter-Floor Error Reweighting)** | **`0.8745`** | **`79.58%`** |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python benchmark_constructive.py
```
