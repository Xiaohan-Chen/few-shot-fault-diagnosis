## Few-shot learning for bearing fault diagnosis
A few shot learning repository for bearing fault diagnosis.

## :clipboard: To Do
- [x] Siamese Networks
- [x] Prototypical Networks
- [ ] Matching Networks
- [ ] Relation Networks
- [x] MoCo v2

## :package: Packages
- Python 3.9.12
- Numpy 1.23.1
- torchvision 0.13.0
- Pytorch 1.12.0
- tqdm 4.46.0

## :pouch: Datasets
- CWRU

## :tv: Results

**Siamese Networks**
Training samples: 200 * 10
Test samples: 100 * 10
Results: CNN1D 83%

**Prototypical Networks**
Training samples: 800 * 10
Test samples: 200 * 10
Results CNN1D:
|                       | 1     | 2     | 3     | 4     | 5     | avg    |
|-----------------------|-------|-------|-------|-------|-------|--------|
| Prototypical          | 81.07 | 81.90 | 83.05 | 82.27 | 82.75 | 82.208 |
| Momentum Prototypical | 84.40 | 83.82 | 84.25 | 84.20 | 84.55 | 84.244 |