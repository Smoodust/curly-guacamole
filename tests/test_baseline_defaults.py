import io

import numpy as np
import pytest
from PIL import Image

from src.config import ExperimentConfig


def test_yaml_scientific_notation_is_loaded_as_numbers(tmp_path):
    from src.config import load_experiment_config

    path = tmp_path / 'experiment.yaml'
    path.write_text('run_name: scientific\n'
                    'train: {encoder_lr: 1e-4, weight_decay: 1e-4}\n'
                    'loss: {dice_weight: 1e-1}\n'
                    'augmentation: {full_frame_probability: 5e-1}\n'
                    'eval: {min_areas: [0, 1e-4]}\n', encoding='utf-8')
    cfg = load_experiment_config(path)
    assert cfg.train.encoder_lr == 1e-4
    assert cfg.train.weight_decay == 1e-4
    assert cfg.loss.dice_weight == .1
    assert cfg.augmentation.full_frame_probability == .5
    assert cfg.eval.min_areas == (0., 1e-4)


def test_minimal_recipe_has_complete_emcad_protocol():
    cfg = ExperimentConfig.from_dict({'run_name': 'trial'})
    assert cfg.model.encoder == 'pvt_v2_b2'
    assert cfg.dataset.image_size == 640
    assert cfg.dataset.protocol_path
    assert cfg.train.epochs == 6
    assert cfg.augmentation.full_frame_probability == .5
    assert cfg.augmentation.foreground_crop_probability == .5
    assert cfg.augmentation.final_full_frame_epochs == 2
    for section, removed in ((cfg.dataset, ('fold', 'n_folds', 'train_originals', 'jpeg_qtable_order')),
                             (cfg.eval, ('resolution',)), (cfg.model, ('decoder_name', 'decoder_channels'))):
        assert all(not hasattr(section, key) for key in removed)


@pytest.mark.parametrize('section,key,value', [
    ('dataset', 'train_originals', False), ('dataset', 'jpeg_qtable_order', 'legacy_zigzag'),
    ('dataset', 'fold', 0), ('eval', 'resolution', 'resized'), ('augmentation', 'full_frame', True),
])
def test_old_protocol_switches_are_rejected(section, key, value):
    with pytest.raises(ValueError, match=key):
        ExperimentConfig.from_dict({'run_name': 'trial', section: {key: value}})


def test_default_dct_preserves_pillow_frequency_order():
    from src.forensic.jpeg import luma_qtable

    q = np.arange(1, 65).reshape(8, 8)
    buf = io.BytesIO()
    Image.new('RGB', (16, 16)).save(buf, format='JPEG', qtables=[q.ravel().tolist()])
    np.testing.assert_array_equal(luma_qtable(buf.getvalue()), q)


def test_snapshots_are_versioned_and_old_runs_are_not_silently_reinterpreted():
    from src.inference.submission import InferenceConfig

    cfg = ExperimentConfig.from_dict({'run_name': 'trial'})
    for snapshot in (cfg.to_dict(), cfg.to_flat_dict()):
        assert snapshot['pipeline_version'] == 'jpeg640_v1'
        assert InferenceConfig.from_snapshot(snapshot).model == cfg.model
        del snapshot['pipeline_version']
        with pytest.raises(ValueError, match='pipeline'):
            InferenceConfig.from_snapshot(snapshot)


def test_protocol_always_includes_train_originals():
    from src.eval.protocol import EvaluationProtocol
    cfg = ExperimentConfig.from_dict({'run_name': 'trial'})
    protocol = EvaluationProtocol.load(cfg.dataset.protocol_path)
    assert protocol.rows('train').target_kind.eq('original_zero').any()
    assert protocol.provenance()['training_originals'] is True


def test_protocol_cannot_be_disabled():
    with pytest.raises(ValueError, match='protocol_path'):
        ExperimentConfig.from_dict({'run_name': 'trial', 'dataset': {'protocol_path': None}})
