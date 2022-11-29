# weaksup-subset-selection

## Overview
Implementation of the subset selection method from our NeurIPS 2022 paper ["Training Subset Selection for Weak Supervision"](https://arxiv.org/abs/2206.02914)

This is a simple method to improve the performance of weak supervision pipelines by carefully selecting a subset of the weakly labeled data.
This procedure prunes mislabeled training examples and thus improves the end model performance.

## Citation
If you find this method useful, please cite our paper:
```bibtex
@article{lang2022training,
  title={Training Subset Selection for Weak Supervision},
  author={Lang, Hunter and Vijayaraghavan, Aravindan and Sontag, David},
  journal={arXiv preprint arXiv:2206.02914},
  year={2022}
}
```

## Tutorial: using subset selection
To get started, set up:
 - The [WRENCH](https://github.com/JieyuZ2/wrench) weak supervision library

In a usual two-stage weak supervision pipeline, using [WRENCH](https://github.com/JieyuZ2/wrench), we first load the data and extract features:
```python
#### Load dataset
dataset_path = 'path_to_wrench_data/'
data = 'semeval'
train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data,
    extract_feature=True,
    extract_fn='bert', # extract bert embedding
    model_name='bert-base-cased',
    cache_name='bert'
)
```

Then train the label model (majority vote doesn't actually train anything, but the others do):
```python
#### Run label model: majority vote
label_model = MajorityVoting()
label_model.fit(
    dataset_train=train_data,
)
acc = label_model.test(test_data, 'acc')
logger.info(f'label model test acc: {acc}')
```

Then filter out the uncovered data and train the end model:
```python
#### Filter out uncovered training data
train_data = train_data.get_covered_subset()
pseudolabels = label_model.predict(train_data)

#### Run end model: MLP
model = EndClassifierModel(
    batch_size=128,
    test_batch_size=512,
    n_steps=10000,
    backbone='MLP',
    optimizer='Adam',
    optimizer_lr=1e-2,
    optimizer_weight_decay=0.0,
)
model.fit(
    dataset_train=train_data,
    y_train=pseudolabels,
    dataset_valid=valid_data,
    evaluation_step=10,
    metric='acc',
    patience=100,
    device=device
)
acc = model.test(test_data, 'acc')
logger.info(f'end model (MLP) test acc: {acc}')
```

To use subset selection, we add two lines to the final block:
```python
#### Filter out uncovered training data
train_data = train_data.get_covered_subset()
pseudolabels = label_model.predict(train_data)

#### subset selection
from cutstat import get_cutstat_subset
train_data, pseudolabels = get_cutstat_subset(train_data, train_data.features, pseudolabels, coverage=0.8, K=20)

...
```

`get_cut_statistic_subset` takes a torch.Tensor of features (shape `N x d`) and a tensor of the pseudolabels for each example (shape `N`).
The optional arguments are the coverage fraction (what percent of the covered data to keep in the subset) and a nearest-neighbor parameter `K`.

## Example
`examples/run_two_stage_pipeline_cutstat.py` compares using subset selection to the status quo of using all the covered data.

```
$ python examples/run_two_stage_pipeline_cutstat.py
2022-11-29 15:24:43 - end model (MLP) test acc: 0.8383333333333334
2022-11-29 15:24:46 - end model (MLP) test acc *with cutstat*: 0.905
```
