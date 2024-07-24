from dataclasses import dataclass


@dataclass
class DirConfig:
    data_dir: str
    processed_dir: str
    output_dir: str


@dataclass
class DatasetConfig:
    _target_: str
    data_dir: str
    train_num: int
    valid_num: int
    test_num: int
    batch_size: int
    num_workers: int
    pin_memory: bool


@dataclass
class OptimizerConfig:
    lr: float
    weight_decay: float
    num_warmup_steps: int


@dataclass
class SchedulerConfig:
    mode: str
    factor: float
    patience: int


@dataclass
class LossConfig:
    loss_name: str
    pos_weight: float | None


@dataclass
class ModelConfig:
    _target_: str
    model_name: str
    backbone_name: str
    pretrained: bool
    in_channels: int
    out_channels: int
    use_batchnorm: bool
    dropout: float
    loss_config: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig


@dataclass
class EarlyStoppingConfig:
    _target_: str
    monitor: str
    min_delta: float
    patience: int
    mode: str
    strict: bool
    check_finite: bool
    stopping_threshold: float
    divergence_threshold: float
    check_on_train_epoch_end: bool


@dataclass
class ModelcheckpointConfig:
    _target_: str
    dirpath: str
    filename: str
    monitor: str
    verbose: bool
    save_last: bool
    save_top_k: int
    mode: str
    auto_insert_metric_name: bool
    every_n_val_epochs: int
    train_time_interval: int
    every_n_epochs: int
    save_on_train_epoch_end: bool


@dataclass
class ModelSummaryConfig:
    _target_: str
    max_depth: int


@dataclass
class RichProgressbar:
    _target_: str


@dataclass
class Callbacks:
    early_stopping: EarlyStoppingConfig
    modelcheckpoint: ModelcheckpointConfig
    model_summary: ModelSummaryConfig
    rich_progressbar: RichProgressbar


@dataclass
class LoggerConfig:
    _target_: str


@dataclass
class TrainerConfig:
    epochs: int
    accelerator: str
    use_amp: bool
    debug: bool
    gradient_clip_val: float
    accumulate_grad_batches: int
    monitor: str
    monitor_mode: str
    check_val_every_n_epoch: int


@dataclass
class TrainConfig:
    exp_name: str
    seed: int
    ckpt_path: str
    dir: DirConfig
    model: ModelConfig
    dataset: DatasetConfig
    trainer: TrainerConfig
    callbacks: Callbacks
    logger: LoggerConfig
