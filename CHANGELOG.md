# Changelog

## [Unreleased]

### Fixed

- Fixed a number of issues with the unit tests to ensure they all pass.
- Renamed `examples/test_with_yml_config..py` to `examples/test_with_yml_config.py` to fix an `ImportError`.
- Fixed an `AttributeError` in `examples/test_with_yml_config.py::test_simple_fusion` by passing a mock config object to `AttentionFusion`.
- Fixed a `NameError` in `examples/test_with_yml_config.py::test_simple_fusion` by reintroducing the `hidden_size` variable.
- Fixed an `AssertionError` in `multitask_bert/tests/test_multitask_bert.py::test_trainer_init_logger` by removing redundant `SummaryWriter` and `mlflow` imports inside `_init_logger` in `multitask_bert/training/trainer.py`.
- Fixed an `AssertionError` in `multitask_bert/tests/test_multitask_bert.py::test_trainer_close` by correctly mocking MLflow.
- Fixed a `TypeError` and `RuntimeError` in `multitask_bert/tests/test_multitask_bert.py::test_trainer_train` by correctly mocking the `DataLoader` and `loss` tensor.
- Fixed a `ValueError` in `multitask_bert/tests/test_multitask_bert.py::test_trainer_evaluate` by correctly mocking `_create_dataloaders` and `tqdm`.
