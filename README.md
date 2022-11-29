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

```bash
python3 Prototypical.py --n_train 800 --lr_scheduler 'step' --showstep 5
python3 moprototypical.py --n_train 800 --m 0.99 --lr_scheduler 'step' --showstep 5 --lr 1e-3 --max_epoch 500
```
|                       | 1     | 2     | 3     | 4     | 5     | avg    |
|-----------------------|-------|-------|-------|-------|-------|--------|
| Prototypical          | 81.77 | 85.05 | 83.20 | 83.60 | 81.55 | 83.034 |
| Momentum ProNet split4| 86.07 | 86.97 | 85.55 | 85.32 | 85.62 | 85.906 |
| Momentum ProNet split1| 84.77 | 84.50 | 86.85 | 85.45 | 85.60 | 85.434 |

Prototypical: with or without split batch normalization
n_train = 300 * 10
1: 80.25, 80.00, 80.15, 79.92, 80.05, 79.17, 78.55, 79.50, 79.10, 79.70 => 79.639
4: 78.55, 78.42, 78.90, 79.50, 80.02, 78.50, 79.52, 78.82, 77.65, 78.55 => 78.843

Cross machine fine-grained:
source_dataname: CWRU
target_dataname: PU
s_load: 3
t_load: 2
s_label_set: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
t_label_set: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Accuracy: 42.32%

source_dataname: CWRU
target_dataname: PU
s_load: 3
t_load: 2
s_label_set: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
t_label_set: [0, 1, 2, 3, 4]
Accuracy: 56.75%