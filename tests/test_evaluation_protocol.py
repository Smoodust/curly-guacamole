from dataclasses import replace

import pandas as pd
import pytest


def sample_rows():
    return pd.DataFrame([dict(chng_img_path=f'{i}.jpg', orgl_img_path=None, gt_path=f'{i}.png',
                              group_id=str(i), domain='plain', generator='none', mask_area=0. if i % 2 else .2,
                              is_negative=bool(i % 2), broken=False, stem=str(i), q_kind='unit') for i in range(60)])


def test_manifest_keeps_linked_originals_together_and_detects_edits(tmp_path):
    from src.eval.protocol import EvaluationProtocol

    rows = sample_rows()
    originals = pd.DataFrame([dict(chng_img_path='source.jpg', orgl_img_path='source.jpg', gt_path=None,
                                   group_ids=['0', '1'], group_id='0', pixel_hash='abc', domain='plain',
                                   generator='original', target_kind='original_zero', is_negative=True,
                                   mask_area=0., broken=False, stem='source')])
    pairs = pd.DataFrame([dict(group_id=g, pixel_hash='abc', orgl_img_path='source.jpg') for g in ['0', '1']])
    protocol = EvaluationProtocol.create(tmp_path / 'protocol', rows, originals, pairs)
    all_rows = pd.concat([protocol.rows(role, include_originals=True) for role in ('train', 'development', 'holdout')])
    linked = all_rows[all_rows.chng_img_path.isin(['0.jpg', '1.jpg', 'source.jpg'])]
    assert linked.role.nunique() == 1
    assert linked.group_id.nunique() == 1
    assert set(all_rows.role) == {'train', 'development', 'holdout'}
    assert not protocol.rows('train', include_originals=False).target_kind.eq('original_zero').any()
    assert protocol.rows('development').target_kind.eq('original_zero').sum() <= 1
    p = tmp_path / 'protocol' / 'samples.parquet'
    edited = pd.read_parquet(p)
    edited.loc[0, 'role'] = 'holdout'
    edited.to_parquet(p, index=False)
    with pytest.raises(ValueError, match='hash'):
        EvaluationProtocol.load(tmp_path / 'protocol')


def test_frozen_validation_uses_requested_thresholds_not_best():
    import torch

    from src.config import load_experiment_config
    from src.inference.predict import ThresholdConfig
    from src.training.builders import build_amp
    from src.training.validation import validate

    class Model(torch.nn.Module):
        def forward(self, image, fmap=None):
            return {'logits': torch.full_like(image[:, :1], .4), 'cls_logits': torch.ones(len(image), 1)}

    config = load_experiment_config('configs/baseline.yaml')
    config = replace(config, train=replace(config.train, device='cpu', amp='off'),
                     loss=replace(config.loss, aux_weight=0),
                     eval=replace(config.eval, mask_thresholds=(.5,), cls_thresholds=(0.,), min_areas=(0.,)))
    batch = dict(image=torch.zeros(2, 3, 8, 8), mask=torch.cat([torch.ones(1, 1, 8, 8), torch.zeros(1, 1, 8, 8)]))
    batch['original_mask'] = [m[0].bool() for m in batch['mask']]
    result = validate(Model(), [batch], build_amp(config.train), config, torch.device('cpu'),
                      thresholds=ThresholdConfig(.75, 0., 0.))
    assert result.tuned.mask_threshold == .75
    assert result.tuned.dice_pos == 0
    assert result.tuned.fpr_neg == 0


def test_protocol_config_routes_training_only_to_train_and_development(tmp_path):
    from src.config import ExperimentConfig, load_experiment_config
    from src.eval.protocol import EvaluationProtocol
    from src.training.engine import ExperimentRunner

    protocol = EvaluationProtocol.create(tmp_path / 'p', sample_rows(), pd.DataFrame(), pd.DataFrame())
    raw = load_experiment_config('configs/baseline.yaml').to_dict()
    raw['dataset']['protocol_path'] = str(protocol.path)
    config = ExperimentConfig.from_dict(raw)
    train, dev = ExperimentRunner(config)._split_data()
    assert set(train.role) == {'train'}
    assert set(dev.role) == {'development'}
    assert not set(train.group_id) & set(dev.group_id)
    assert ExperimentConfig.from_dict(config.to_dict()) == config


def test_holdout_rejects_legacy_or_wrong_training_provenance(tmp_path):
    from src.eval.protocol import EvaluationProtocol

    protocol = EvaluationProtocol.create(tmp_path / 'p', sample_rows(), pd.DataFrame(), pd.DataFrame())
    with pytest.raises(ValueError, match='protocol'):
        protocol.verify_run({})
    with pytest.raises(ValueError, match='training'):
        protocol.verify_run({'protocol_digest': protocol.digest, 'training_rows_digest': 'wrong',
                             'development_rows_digest': 'wrong'})
