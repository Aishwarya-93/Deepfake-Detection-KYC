# Dataset Selection

## Primary Dataset

### DFDC (DeepFake Detection Challenge)

**Reason:**
- Large-scale dataset
- Realistic deepfake videos
- Suitable for Video KYC scenarios
- Diverse actors and recording conditions
- Various lighting environments

---

## Secondary Dataset

### DeeperForensics-1.0

**Reason:**
- Used for cross-dataset evaluation
- Tests model generalization
- Includes challenging real-world perturbations

---

## Research Strategy

Train on **DFDC**

↓

Evaluate on **DFDC**

↓

Test on **DeeperForensics-1.0**

---

## Why this combination?

Training on DFDC allows the model to learn from a large and diverse set of deepfake videos.

Testing on DeeperForensics-1.0 helps evaluate whether the model can generalize to a completely different dataset instead of memorizing patterns from only one dataset.