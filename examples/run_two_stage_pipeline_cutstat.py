import logging
import torch
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.labelmodel import MajorityVoting
from wrench.endmodel import EndClassifierModel

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device='cuda'

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

#### Run label model: majority vote
label_model = MajorityVoting()
label_model.fit(
    dataset_train=train_data,
)
acc = label_model.test(test_data, 'acc')
logger.info(f'label model test acc: {acc}')


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

from cutstat import get_cutstat_subset
train_data, pseudolabels = get_cutstat_subset(train_data,
                                              train_data.features,
                                              pseudolabels,
                                              coverage=0.8,
                                              K=20)

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
logger.info(f'end model (MLP) test acc *with cutstat*: {acc}')
