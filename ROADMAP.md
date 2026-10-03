# Roadmap

Legend: [x] done, [ ] to do

## Phase 0: Environment and project skeleton
- [x] Create repository and folder structure
- [ ] README, ROADMAP, requirements

## Phase 1: Exploratory data analysis
- [ ] Inspect dataset folder structure and file formats
- [ ] Analyze disaster types and damage class distribution
- [ ] Visualize image pairs and label masks

## Phase 2: Data preprocessing
- [ ] Parse labels into segmentation masks
- [ ] Train / validation split
- [ ] PyTorch Dataset and DataLoader
- [ ] Class weighting for imbalanced damage classes

## Phase 3: Baseline model
- [ ] Siamese U-Net (shared pretrained encoder)

## Phase 4: Loss and training loop
- [ ] Combined loss (weighted cross-entropy + Dice)
- [ ] Training and validation loops
- [ ] Overfitting test on a small subset

## Phase 5: Full training and evaluation
- [ ] Train on the full xBD dataset
- [ ] Evaluate with per-class F1 and overall score

## Phase 6: Improvement and reporting
- [ ] Fine-tune on the earthquake subset
- [ ] Fine-tune on a Turkey earthquake dataset
- [ ] Final report