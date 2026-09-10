import pytest

from src.training.ablation import ArmResult, LossAblation

ARM_CONFIGS = [
    "configs/loss_l0_baseline.yaml",
    "configs/loss_l1_pos_weight.yaml",
    "configs/loss_l2_focal.yaml",
    "configs/loss_l3_focal_tversky.yaml",
    "configs/loss_l4_aic_surrogate.yaml",
    "configs/loss_l5_aic_harmonic.yaml",
]


def test_every_arm_differs_from_the_control_only_by_its_loss():
    from dataclasses import replace

    ablation = LossAblation(ARM_CONFIGS)
    control = ablation.configs[0]

    assert control.loss.name == "bce_dice"
    for config in ablation.configs[1:]:
        assert config.loss.name != control.loss.name
        assert replace(config, loss=control.loss,
                       paths=control.paths) == replace(control, paths=control.paths)


def test_arms_keep_the_baseline_protocol_on_a_halved_budget():
    from src.config import load_experiment_config

    baseline = load_experiment_config("configs/baseline_protocol_originals.yaml")
    for config in LossAblation(ARM_CONFIGS).configs:
        assert config.model == baseline.model
        assert config.augmentation == baseline.augmentation
        assert config.dataset == baseline.dataset
        assert config.eval == baseline.eval
        assert config.seed == baseline.seed
        assert config.train.epoch_size == baseline.train.epoch_size // 2
        assert config.train.epochs == baseline.train.epochs


def test_arms_write_to_separate_run_directories():
    ablation = LossAblation(ARM_CONFIGS)
    names = [config.paths.run_name for config in ablation.configs]

    assert len(set(names)) == len(names)
    assert ablation.arm_names == ["l0_baseline", "l1_pos_weight", "l2_focal",
                                  "l3_focal_tversky", "l4_aic_surrogate", "l5_aic_harmonic"]


def test_arms_train_on_crops_first_and_only_on_full_frames_at_the_end():
    """The arms inherit the protocol schedule, not just its data split.

    An arm that quietly lost the final full-frame phase would be tuned on a crop
    distribution and then evaluated on whole originals.
    """
    from src.data.augmentation.pipeline import AugmentationPipeline

    for config in LossAblation(ARM_CONFIGS).configs:
        assert config.dataset.protocol_path is not None
        assert config.dataset.train_originals
        final = config.augmentation.final_full_frame_epochs
        assert 0 < final < config.train.epochs

        pipeline = AugmentationPipeline(config.augmentation, total_epochs=config.train.epochs)
        probabilities = []
        for epoch in range(config.train.epochs):
            pipeline.set_epoch(epoch)
            probabilities.append(pipeline.full_frame_probability)

        assert all(p < 1.0 for p in probabilities[:-final]), "crops must reach the model first"
        assert all(p == 1.0 for p in probabilities[-final:]), "the run must end on whole frames only"


def test_colliding_run_names_fail_before_anything_is_overwritten():
    with pytest.raises(ValueError, match="share a run_name"):
        LossAblation(["configs/loss_l0_baseline.yaml", "configs/loss_l0_baseline.yaml"])


def test_data_path_override_reaches_every_arm(tmp_path):
    for config in LossAblation(ARM_CONFIGS, data_path=tmp_path).configs:
        assert config.paths.data_path == tmp_path.resolve()


def test_comparison_ranks_by_aic_and_reports_the_gap_to_the_control():
    configs = LossAblation(ARM_CONFIGS).configs
    results = [
        ArmResult("l0_baseline", configs[0],
                  {"best_aic": 0.70, "samples": 96000,
                   "best": {"dice_pos": 0.75, "fpr_neg": 0.1}}),
        ArmResult("l4_aic_surrogate", configs[4],
                  {"best_aic": 0.74, "samples": 96000,
                   "best": {"dice_pos": 0.80, "fpr_neg": 0.12}}),
        ArmResult("l2_focal", configs[2], {}, error="RuntimeError: out of memory"),
    ]

    table = LossAblation.compare(results)

    assert list(table["arm"]) == ["l4_aic_surrogate", "l0_baseline", "l2_focal"]
    assert table.loc[0, "delta_vs_control"] == pytest.approx(0.04)
    assert table.loc[0, "loss"] == "aic_surrogate"
    assert table.loc[2, "error"].startswith("RuntimeError")
    assert not results[2].completed


def test_a_failed_arm_does_not_stop_the_series(monkeypatch, tmp_path):
    from dataclasses import replace

    import src.training.ablation as ablation_module

    ablation = LossAblation(ARM_CONFIGS[:3])
    ablation.configs = [replace(config, paths=replace(config.paths, runs_path=tmp_path))
                        for config in ablation.configs]
    calls = []

    def fake_run(config):
        calls.append(config.paths.run_name)
        if config.loss.name == "balanced_bce_dice":
            raise RuntimeError("CUDA out of memory")
        return type("R", (), {"summary": {"best_aic": 0.5, "samples": 1, "best": {}}})()

    monkeypatch.setattr(ablation_module, "run_experiment", fake_run)
    results = ablation.run()

    assert len(calls) == 3
    assert [result.completed for result in results] == [True, False, True]
    assert "CUDA out of memory" in results[1].error


def test_finished_arms_are_read_from_disk_instead_of_retrained(monkeypatch, tmp_path):
    from dataclasses import replace

    import src.training.ablation as ablation_module
    from src.training.runs import Run

    ablation = LossAblation(ARM_CONFIGS[:1])
    ablation.configs = [replace(config, paths=replace(config.paths, runs_path=tmp_path))
                        for config in ablation.configs]
    run = Run.create(tmp_path, ablation.configs[0].paths.run_name, resume=False)
    run.save_summary({"best_aic": 0.66, "samples": 96000, "best": {"dice_pos": 0.7}})
    run.close()

    monkeypatch.setattr(ablation_module, "run_experiment",
                        lambda config: pytest.fail("a finished arm must not be retrained"))
    results = ablation.run()

    assert results[0].summary["best_aic"] == 0.66
    assert LossAblation.compare(results).loc[0, "aic"] == 0.66
