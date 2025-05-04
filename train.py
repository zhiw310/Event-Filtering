# ==============================================================================
# SAST: Scene Adaptive Sparse Transformer for Event-based Object Detection
# Copyright (c) 2023 The SAST Authors.
# Licensed under The MIT License.
# Written by Yansong Peng.
# Modified from RVT.
# ==============================================================================

import os
import time

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy
# from pytorch_lightning.profilers import PyTorchProfiler

from callbacks.custom import get_ckpt_callback, get_viz_callback
from callbacks.gradflow import GradFlowLogCallback
from config.modifier import dynamically_modify_train_config
from data.utils.types import DatasetSamplingMode, DataType
from loggers.utils import get_wandb_logger, get_ckpt_path
from modules.utils.fetch import fetch_data_module, fetch_model_module
from pytorch_lightning.loggers import CSVLogger
import torch._dynamo.config
torch._dynamo.config.verbose=True
import sys
sys.setrecursionlimit(100000)

class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar
    
# class DetailedTimingCallback(pl.Callback):
#     def __init__(self, num_batches_to_profile=5):
#         super().__init__()
#         self.num_batches_to_profile = num_batches_to_profile
#         self.timing_stats = {
#             'data_loading': [],
#             'forward': [],
#             'backward': [], 
#             'optimizer': [],
#             'total': []
#         }
#         self.current_batch = 0
#         self.profiling_done = False
#         # 初始化所有时间变量，避免属性未定义错误
#         self.last_batch_end_time = time.time()
#         self.batch_start_time = self.last_batch_end_time
#         self.forward_start_time = self.last_batch_end_time
#         self.backward_start_time = self.last_batch_end_time
#         self.optimizer_start_time = self.last_batch_end_time
        
#     def on_train_epoch_start(self, trainer, pl_module):
#         # 重置计数器和计时器
#         self.current_batch = 0
#         self.last_batch_end_time = time.time()
        
#     def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
#         if self.profiling_done:
#             return
            
#         # 计算数据加载时间（从上一批次结束到这一批次开始）
#         now = time.time()
#         data_loading_time = now - self.last_batch_end_time
        
#         # 开始记录这个批次
#         if self.current_batch < self.num_batches_to_profile:
#             self.timing_stats['data_loading'].append(data_loading_time)
#             self.batch_start_time = now
#             self.forward_start_time = now
        
#         self.current_batch += 1
            
#     def on_before_backward(self, trainer, pl_module, loss):
#         if self.profiling_done or self.current_batch > self.num_batches_to_profile:
#             return
            
#         # 前向传播结束，反向传播开始
#         now = time.time()
#         forward_time = now - self.forward_start_time
#         self.timing_stats['forward'].append(forward_time)
#         self.backward_start_time = now
            
#     def on_after_backward(self, trainer, pl_module):
#         if self.profiling_done or self.current_batch > self.num_batches_to_profile:
#             return
            
#         # 反向传播结束
#         now = time.time()
#         backward_time = now - self.backward_start_time
#         self.timing_stats['backward'].append(backward_time)
#         self.optimizer_start_time = now
            
#     def on_before_zero_grad(self, trainer, pl_module, optimizer):
#         if self.profiling_done or self.current_batch > self.num_batches_to_profile:
#             return
        
#         try:    
#             # 优化器步骤结束
#             now = time.time()
#             # 安全获取时间，防止缺少属性
#             optimizer_time = now - getattr(self, 'optimizer_start_time', self.batch_start_time)
#             self.timing_stats['optimizer'].append(optimizer_time)
            
#             # 整个批次的总时间
#             total_time = now - self.batch_start_time
#             self.timing_stats['total'].append(total_time)
            
#             # 记录批次结束时间，用于下一个批次的数据加载时间计算
#             self.last_batch_end_time = now
            
#             # 检查是否已经完成了足够的批次分析
#             if len(self.timing_stats['total']) >= self.num_batches_to_profile:
#                 self._print_statistics(trainer)
#                 self.profiling_done = True
#         except Exception as e:
#             print(f"计时回调错误: {e}")
    
#     def _print_statistics(self, trainer):
#         # 计算平均值、最小值、最大值
#         stats = {}
#         for phase, times in self.timing_stats.items():
#             if not times:
#                 continue
#             avg = sum(times) / len(times)
#             min_time = min(times)
#             max_time = max(times)
#             stats[phase] = {
#                 'avg': avg,
#                 'min': min_time,
#                 'max': max_time,
#                 'percentage': avg / stats.get('total', {'avg': 1.0})['avg'] * 100 if phase != 'total' else 100.0
#             }
        
#         # 打印详细统计信息
#         print("\n" + "="*50)
#         print("训练性能分析 (秒)")
#         print("="*50)
#         print(f"分析了 {len(self.timing_stats.get('total', []))} 个批次")
#         print("-"*50)
#         print(f"{'阶段':<15} {'平均时间':<10} {'最小时间':<10} {'最大时间':<10} {'占比 (%)':<10}")
#         print("-"*50)
        
#         for phase in ['data_loading', 'forward', 'backward', 'optimizer', 'total']:
#             if phase in stats:
#                 s = stats[phase]
#                 print(f"{phase:<15} {s['avg']:<10.4f} {s['min']:<10.4f} {s['max']:<10.4f} {s['percentage']:<10.1f}")
        
#         print("="*50)
#         # 安全获取批次大小
#         try:
#             batch_size = trainer.train_dataloader.loaders.batch_size
#         except:
#             batch_size = 2  # 默认值，如果无法获取
#             print("无法获取确切批次大小，使用默认值2")
            
#         print(f"训练吞吐量: {1.0/stats['total']['avg']:.2f} 批次/秒")
#         print(f"每批次样本数: {batch_size}")
#         print(f"估计吞吐量: {batch_size/stats['total']['avg']:.2f} 样本/秒")
#         print("="*50 + "\n")
        
#         # GPU利用率分析
#         print("GPU利用率分析:")
#         compute_time = 0
#         for phase in ['forward', 'backward', 'optimizer']:
#             if phase in stats:
#                 compute_time += stats[phase]['avg']
                
#         data_time = stats.get('data_loading', {'avg': 0})['avg']
        
#         if compute_time < data_time:
#             print("⚠️ 数据加载是瓶颈 - 考虑增加num_workers或优化数据预处理")
#         else:
#             print("💡 计算是瓶颈 - GPU正在高效利用")
            
#             # 分析计算瓶颈
#             if 'forward' in stats and 'backward' in stats:
#                 if stats['forward']['avg'] > stats['backward']['avg']:
#                     print("  - 前向传播占用时间最多，考虑优化模型架构或启用更高效的运算符")
#                 else:
#                     print("  - 反向传播占用时间最多，这是正常现象")
                
#         # 建议
#         print("\n优化建议:")
#         if data_time > 0.1 * stats.get('total', {'avg': 1.0})['avg']:
#             print("1. 增加DataLoader的num_workers (当前可能较低)")
#             print("2. 考虑使用更快的数据存储或预处理方法")
            
#         if 'optimizer' in stats and stats['optimizer']['avg'] > 0.2 * stats.get('total', {'avg': 1.0})['avg']:
#             print("3. 考虑使用更高效的优化器或启用融合优化器")
            
#         print("4. 检查是否可以使用更大的批次大小提高吞吐量")
#         print("5. 确保模型充分利用GPU计算能力（如检查是否有CPU瓶颈）")

@hydra.main(config_path='config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # ---------------------
    # Reproducibility
    # ---------------------
    dataset_train_sampling = config.dataset.train.sampling
    assert dataset_train_sampling in iter(DatasetSamplingMode)
    disable_seed_everything = dataset_train_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.MIXED)
    if disable_seed_everything:
        print('Disabling PL seed everything because of unresolved issues with shuffling during training on streaming '
              'datasets')
    seed = config.reproduce.seed_everything
    if seed is not None and not disable_seed_everything:
        assert isinstance(seed, int)
        print(f'USING pl.seed_everything WITH {seed=}')
        pl.seed_everything(seed=seed, workers=True)

    # ---------------------
    # DDP
    # ---------------------
    gpu_config = config.hardware.gpus
    gpus = OmegaConf.to_container(gpu_config) if OmegaConf.is_config(gpu_config) else gpu_config
    gpus = gpus if isinstance(gpus, list) else [gpus]
    distributed_backend = config.hardware.dist_backend
    assert distributed_backend in ('nccl', 'gloo'), f'{distributed_backend=}'
    strategy = DDPStrategy(process_group_backend=distributed_backend,
                           find_unused_parameters=False,
                           gradient_as_bucket_view=True) if len(gpus) > 1 else None

    # ---------------------
    # Data
    # ---------------------
    data_module = fetch_data_module(config=config)
            
    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    logger = get_wandb_logger(config)
    # logger = CSVLogger(save_dir='./logs/', name='experiment_name')
    ckpt_path = None
    if config.wandb.artifact_name is not None:
        ckpt_path = get_ckpt_path(logger, wandb_config=config.wandb)

    # ---------------------
    # Model
    # ---------------------
    module = fetch_model_module(config=config)
    if ckpt_path is not None and config.wandb.wandb.resume_only_weights:
        print('Resuming only the weights instead of the full training state')
        module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config}, strict=True)
        ckpt_path = None

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = list()
    
    callbacks.append(TQDMProgressBar(refresh_rate=100))
    callbacks.append(get_ckpt_callback(config))
    callbacks.append(GradFlowLogCallback(config.logging.train.log_model_every_n_steps))
    if config.training.lr_scheduler.use:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    if config.logging.train.high_dim.enable or config.logging.validation.high_dim.enable:
        viz_callback = get_viz_callback(config=config)
        callbacks.append(viz_callback)
    callbacks.append(ModelSummary(max_depth=2))

    logger.watch(model=module, log='all', log_freq=config.logging.train.log_model_every_n_steps, log_graph=True)

    # # 创建详细计时回调，分析前5个批次
    # timing_callback = DetailedTimingCallback(num_batches_to_profile=5)
    # callbacks.append(timing_callback)

    # ---------------------
    # Training
    # ---------------------

    val_check_interval = config.validation.val_check_interval
    check_val_every_n_epoch = None
    assert val_check_interval is None or check_val_every_n_epoch is None

    # profiler = PyTorchProfiler(dirpath=".", filename="profile")

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        enable_checkpointing=True,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        default_root_dir='./output/',
        devices=gpus,
        gradient_clip_val=config.training.gradient_clip_val,
        gradient_clip_algorithm='value',
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.validation.limit_val_batches,
        logger=logger,
        log_every_n_steps=config.logging.train.log_every_n_steps,
        plugins=None,
        precision=config.training.precision,
        max_epochs=config.training.max_epochs,
        max_steps=config.training.max_steps,
        strategy=strategy,
        sync_batchnorm=False if strategy is None else True,
        move_metrics_to_cpu=False,
        benchmark=config.reproduce.benchmark,
        deterministic=config.reproduce.deterministic_flag,
        auto_scale_batch_size=False,
        # profiler=profiler
    )

    trainer.fit(model=module, ckpt_path=ckpt_path, datamodule=data_module)


if __name__ == '__main__':
    main()
