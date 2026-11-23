# ConstructiveMultiFloorNet: Progressive Neural Growth with Raw-Feature Passthrough

**ConstructiveMultiFloorNet** grows a neural ensemble floor by floor using **beam-search architecture selection**, **persistent raw-feature passthrough**, and **multi-depth leader boosting**.

---

## 💡 Core Architectural Innovations

1. **Beam-Search Micro-Network Growth:** Instead of brute-forcing all $2^d - 1$ feature subsets, each floor evaluates a focused beam of compact micro-networks (`MicroConstructiveNet`) and promotes the Top-$K$ winners on validation.
2. **Direct Raw-Feature Passthrough:** Unlike classical GMDH where deeper layers only see previous layer outputs, every new floor in `ConstructiveMultiFloorNet` explicitly combines prior-floor winners **directly with raw input features** (e.g., discovering `[Raw(FoodCourt), Raw(Spa), F2_M5]`).
3. **Multi-Depth Ensemble (All Floors Combined):** Rather than discarding shallower models, the final meta-booster aggregates all 24 winning micro-networks across Floors 1, 2, and 3—combining diverse depths and receptive fields into a single prediction.

---

## 📊 Experimental Results (Real-World *Spaceship Titanic* Dataset)

Evaluated on a strict 3-way split (`Train: 4,563` | `Val: 1,956` | `Clean Test: 2,174` samples):

| Stage / Model | Top Discovered Structure | Test ROC AUC ↑ | Test Accuracy ↑ |
| :--- | :--- | :---: | :---: |
| **Constructive Floor 1 (Top-8)** | `[Raw(HomePlanet), Raw(RoomService), Raw(VRDeck)]` | `0.8439` | `74.47%` |
| **Constructive Floor 2 (Top-8)** | `[F1_M1, F1_M2]` | `0.8589` | `77.97%` |
| **Constructive Floor 3 (Top-8)** | `[Raw(FoodCourt), Raw(Spa), F2_M5]` | `0.8720` | `79.02%` |
| **Final Multi-Floor Ensemble (24 Nets)** | **All 1-, 2-, and 3-Floor Leaders Combined** | **`0.8783`** | **`79.35%`** |
| Classical 2-Layer MLP (Baseline) | Dense `10 -> 64 -> 64 -> 1` | `0.8798` | `79.85%` |
| LightGBM (Gradient Boosting Baseline) | 300 Boosted Decision Trees | `0.8802` | `79.99%` |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python benchmark_constructive.py
```
