"""
NaN 损失诊断脚本

用于检测训练过程中损失值变为 NaN 的根本原因。
检查输入数据、模型输出、损失计算、梯度等各个环节。
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Any, List
import argparse
import logging
from datetime import datetime

# 导入模型和工具
from multimodal_jscc import MultimodalJSCC
from losses import MultimodalLoss
from data_loader import MultimodalDataLoader
from config import TrainingConfig
from utils import seed_torch, logger_configuration, makedirs, load_manifest


class NaNChecker:
    """NaN 检查器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.issues = []
    
    def check_tensor(self, tensor: torch.Tensor, name: str, check_grad: bool = False) -> bool:
        """
        检查张量是否包含 NaN 或 Inf
        
        Args:
            tensor: 要检查的张量
            name: 张量名称
            check_grad: 是否检查梯度
            
        Returns:
            bool: True 表示正常，False 表示有问题
        """
        if tensor is None:
            self.logger.warning(f"  {name}: None")
            return True
        
        is_valid = True
        
        # 检查 NaN
        if torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            nan_ratio = nan_count / tensor.numel()
            self.issues.append(f"{name} 包含 {nan_count} 个 NaN ({nan_ratio*100:.2f}%)")
            self.logger.error(f"  ❌ {name}: 包含 NaN ({nan_count}/{tensor.numel()})")
            is_valid = False
        else:
            self.logger.info(f"  ✓ {name}: 无 NaN")
        
        # 检查 Inf
        if torch.isinf(tensor).any():
            inf_count = torch.isinf(tensor).sum().item()
            inf_ratio = inf_count / tensor.numel()
            self.issues.append(f"{name} 包含 {inf_count} 个 Inf ({inf_ratio*100:.2f}%)")
            self.logger.error(f"  ❌ {name}: 包含 Inf ({inf_count}/{tensor.numel()})")
            is_valid = False
        else:
            self.logger.info(f"  ✓ {name}: 无 Inf")
        
        # 检查数值范围
        if tensor.numel() > 0:
            # 【修复】创建一个浮点类型的副本用于统计计算，避免 LongTensor 调用 mean() 失败
            tensor_float = tensor.float()
            
            min_val = tensor_float.min().item()
            max_val = tensor_float.max().item()
            mean_val = tensor_float.mean().item()
            std_val = tensor_float.std().item()
            
            # 检查异常大的值
            if abs(max_val) > 1e6 or abs(min_val) > 1e6:
                self.issues.append(f"{name} 包含异常大的值: min={min_val:.2e}, max={max_val:.2e}")
                self.logger.warning(f"  ⚠ {name}: 数值范围异常 [min={min_val:.2e}, max={max_val:.2e}]")
            
            # 检查异常小的值
            if abs(mean_val) < 1e-10 and abs(std_val) < 1e-10:
                self.logger.warning(f"  ⚠ {name}: 数值过小 [mean={mean_val:.2e}, std={std_val:.2e}]")
            
            self.logger.info(f"    {name} 统计: shape={tuple(tensor.shape)}, "
                           f"min={min_val:.4f}, max={max_val:.4f}, "
                           f"mean={mean_val:.4f}, std={std_val:.4f}")
        
        # 检查梯度
        if check_grad and tensor.requires_grad:
            if tensor.grad is not None:
                if torch.isnan(tensor.grad).any():
                    nan_grad_count = torch.isnan(tensor.grad).sum().item()
                    self.issues.append(f"{name}.grad 包含 {nan_grad_count} 个 NaN")
                    self.logger.error(f"  ❌ {name}.grad: 包含 NaN ({nan_grad_count})")
                    is_valid = False
                elif torch.isinf(tensor.grad).any():
                    inf_grad_count = torch.isinf(tensor.grad).sum().item()
                    self.issues.append(f"{name}.grad 包含 {inf_grad_count} 个 Inf")
                    self.logger.error(f"  ❌ {name}.grad: 包含 Inf ({inf_grad_count})")
                    is_valid = False
                else:
                    grad_norm = tensor.grad.norm().item()
                    self.logger.info(f"  ✓ {name}.grad: 正常, norm={grad_norm:.4f}")
        
        return is_valid
    
    def check_model_weights(self, model: nn.Module) -> bool:
        """检查模型权重"""
        self.logger.info("\n" + "="*80)
        self.logger.info("检查模型权重")
        self.logger.info("="*80)
        
        is_valid = True
        nan_params = []
        inf_params = []
        large_params = []
        
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
                is_valid = False
            if torch.isinf(param).any():
                inf_params.append(name)
                is_valid = False
            if param.abs().max() > 1e6:
                large_params.append(name)
        
        if nan_params:
            self.logger.error(f"  ❌ 包含 NaN 的参数: {len(nan_params)} 个")
            for name in nan_params[:10]:  # 只显示前10个
                self.logger.error(f"    - {name}")
            if len(nan_params) > 10:
                self.logger.error(f"    ... 还有 {len(nan_params)-10} 个")
            self.issues.append(f"模型权重包含 NaN: {len(nan_params)} 个参数")
        else:
            self.logger.info(f"  ✓ 所有参数无 NaN")
        
        if inf_params:
            self.logger.error(f"  ❌ 包含 Inf 的参数: {len(inf_params)} 个")
            for name in inf_params[:10]:
                self.logger.error(f"    - {name}")
            if len(inf_params) > 10:
                self.logger.error(f"    ... 还有 {len(inf_params)-10} 个")
            self.issues.append(f"模型权重包含 Inf: {len(inf_params)} 个参数")
        else:
            self.logger.info(f"  ✓ 所有参数无 Inf")
        
        if large_params:
            self.logger.warning(f"  ⚠ 数值过大的参数: {len(large_params)} 个")
            for name in large_params[:5]:
                param = dict(model.named_parameters())[name]
                self.logger.warning(f"    - {name}: max={param.abs().max().item():.2e}")
        
        return is_valid
    
    def check_model_outputs(self, model: nn.Module, batch: Dict[str, Any], device: torch.device) -> bool:
        """检查模型各层输出"""
        self.logger.info("\n" + "="*80)
        self.logger.info("检查模型前向传播")
        self.logger.info("="*80)
        
        model.eval()
        is_valid = True
        
        # 提取输入
        inputs = batch['inputs']
        text_input = inputs.get('text_input', None)
        image_input = inputs.get('image_input', None)
        video_input = inputs.get('video_input', None)
        attention_mask = batch.get('attention_mask', None)
        
        # 移动到设备
        if text_input is not None:
            text_input = text_input.to(device)
            attention_mask = attention_mask.to(device) if attention_mask is not None else None
        if image_input is not None:
            image_input = image_input.to(device)
        if video_input is not None:
            video_input = video_input.to(device)
        
        # 检查输入
        self.logger.info("\n检查输入数据:")
        if text_input is not None:
            is_valid &= self.check_tensor(text_input, "text_input")
        if image_input is not None:
            is_valid &= self.check_tensor(image_input, "image_input")
        if video_input is not None:
            is_valid &= self.check_tensor(video_input, "video_input")
        if attention_mask is not None:
            is_valid &= self.check_tensor(attention_mask, "attention_mask")
        
        # 使用钩子函数捕获中间输出
        intermediate_outputs = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    for i, out in enumerate(output):
                        intermediate_outputs[f"{name}_output_{i}"] = out
                else:
                    intermediate_outputs[name] = output
            return hook
        
        # 注册钩子
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm, nn.BatchNorm2d)):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        try:
            # 前向传播
            with torch.no_grad():
                results = model(
                    text_input=text_input,
                    image_input=image_input,
                    video_input=video_input,
                    text_attention_mask=attention_mask,
                    snr_db=10.0
                )
            
            # 检查模型输出
            self.logger.info("\n检查模型输出:")
            for key, value in results.items():
                if isinstance(value, torch.Tensor):
                    is_valid &= self.check_tensor(value, f"results[{key}]")
            
            # 检查中间输出（只检查前20个，避免输出过多）
            self.logger.info("\n检查中间层输出 (前20个):")
            for i, (name, value) in enumerate(intermediate_outputs.items()):
                if i >= 20:
                    break
                if isinstance(value, torch.Tensor):
                    is_valid &= self.check_tensor(value, f"intermediate[{name}]")
        
        except Exception as e:
            self.logger.error(f"  ❌ 前向传播出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            is_valid = False
        
        finally:
            # 移除钩子
            for hook in hooks:
                hook.remove()
        
        return is_valid
    
    def check_loss_computation(
        self, 
        model: nn.Module, 
        loss_fn: nn.Module, 
        batch: Dict[str, Any], 
        device: torch.device
    ) -> bool:
        """检查损失计算"""
        self.logger.info("\n" + "="*80)
        self.logger.info("检查损失计算")
        self.logger.info("="*80)
        
        model.train()
        is_valid = True
        
        # 提取输入和目标
        inputs = batch['inputs']
        targets = batch['targets']
        attention_mask = batch.get('attention_mask', None)
        
        text_input = inputs.get('text_input', None)
        image_input = inputs.get('image_input', None)
        video_input = inputs.get('video_input', None)
        
        # 移动到设备
        if text_input is not None:
            text_input = text_input.to(device)
            attention_mask = attention_mask.to(device) if attention_mask is not None else None
        if image_input is not None:
            image_input = image_input.to(device)
        if video_input is not None:
            video_input = video_input.to(device)
        
        device_targets = {}
        for key, value in targets.items():
            if value is not None:
                device_targets[key] = value.to(device)
        
        try:
            # 前向传播
            results = model(
                text_input=text_input,
                image_input=image_input,
                video_input=video_input,
                text_attention_mask=attention_mask,
                snr_db=10.0
            )
            
            # 检查预测结果
            self.logger.info("\n检查预测结果:")
            for key, value in results.items():
                if isinstance(value, torch.Tensor) and key.endswith('_decoded'):
                    is_valid &= self.check_tensor(value, f"prediction[{key}]")
            
            # 检查目标
            self.logger.info("\n检查目标数据:")
            for key, value in device_targets.items():
                is_valid &= self.check_tensor(value, f"target[{key}]")
            
            # 计算损失
            self.logger.info("\n计算损失:")
            loss_dict = loss_fn(
                predictions=results,
                targets=device_targets,
                attention_mask=attention_mask
            )
            
            # 检查损失值
            self.logger.info("\n检查损失值:")
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    is_valid &= self.check_tensor(value, f"loss[{key}]")
                    if key == 'total_loss':
                        loss_value = value.item()
                        self.logger.info(f"  总损失: {loss_value:.6f}")
            
            # 反向传播
            self.logger.info("\n执行反向传播:")
            total_loss = loss_dict['total_loss']
            total_loss.backward()
            
            # 检查梯度
            self.logger.info("\n检查梯度:")
            grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms[name] = grad_norm
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        is_valid &= self.check_tensor(param.grad, f"grad[{name}]", check_grad=False)
            
            # 统计梯度信息
            if grad_norms:
                max_grad_norm = max(grad_norms.values())
                mean_grad_norm = np.mean(list(grad_norms.values()))
                self.logger.info(f"  最大梯度范数: {max_grad_norm:.4f}")
                self.logger.info(f"  平均梯度范数: {mean_grad_norm:.4f}")
                
                if max_grad_norm > 100:
                    self.issues.append(f"梯度爆炸: 最大梯度范数={max_grad_norm:.2f}")
                    self.logger.warning(f"  ⚠ 检测到可能的梯度爆炸")
        
        except Exception as e:
            self.logger.error(f"  ❌ 损失计算出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            is_valid = False
        
        return is_valid
    
    def check_channel_module(self, model: nn.Module, device: torch.device) -> bool:
        """检查信道模块"""
        self.logger.info("\n" + "="*80)
        self.logger.info("检查信道模块")
        self.logger.info("="*80)
        
        is_valid = True
        
        if not hasattr(model, 'channel'):
            self.logger.warning("  模型没有 channel 属性")
            return True
        
        channel = model.channel
        
        # 测试不同输入
        test_inputs = [
            torch.randn(2, 10, 256, device=device),  # 文本特征
            torch.randn(2, 100, 256, device=device),  # 图像特征
            torch.randn(2, 5, 256, 16, 16, device=device),  # 视频特征
        ]
        
        for i, test_input in enumerate(test_inputs):
            try:
                output = channel(test_input)
                is_valid &= self.check_tensor(output, f"channel_output[{i}]")
                
                # 检查功率归一化
                input_power = torch.mean(test_input ** 2).item()
                output_power = torch.mean(output ** 2).item()
                self.logger.info(f"  输入功率: {input_power:.6f}, 输出功率: {output_power:.6f}")
                
            except Exception as e:
                self.logger.error(f"  ❌ 信道模块测试失败: {e}")
                is_valid = False
        
        return is_valid
    
    def check_loss_function(self, loss_fn: nn.Module, device: torch.device) -> bool:
        """检查损失函数"""
        self.logger.info("\n" + "="*80)
        self.logger.info("检查损失函数")
        self.logger.info("="*80)
        
        is_valid = True
        
        # 测试文本损失
        try:
            text_pred = torch.randn(2, 10, 10000, device=device)
            text_target = torch.randint(0, 10000, (2, 10), device=device)
            text_loss = loss_fn.text_loss_fn(text_pred, text_target)
            is_valid &= self.check_tensor(text_loss[0], "text_loss")
        except Exception as e:
            self.logger.error(f"  ❌ 文本损失测试失败: {e}")
            is_valid = False
        
        # 测试图像损失
        try:
            img_pred = torch.rand(2, 3, 128, 128, device=device)
            img_target = torch.rand(2, 3, 128, 128, device=device)
            img_loss = loss_fn.image_loss_fn(img_pred, img_target)
            is_valid &= self.check_tensor(img_loss[0], "image_loss")
        except Exception as e:
            self.logger.error(f"  ❌ 图像损失测试失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            is_valid = False
        
        # 测试视频损失
        try:
            vid_pred = torch.rand(2, 5, 3, 128, 128, device=device)
            vid_target = torch.rand(2, 5, 3, 128, 128, device=device)
            vid_loss = loss_fn.video_loss_fn(vid_pred, vid_target)
            is_valid &= self.check_tensor(vid_loss[0], "video_loss")
        except Exception as e:
            self.logger.error(f"  ❌ 视频损失测试失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            is_valid = False
        
        return is_valid
    
    def print_summary(self):
        """打印诊断摘要"""
        self.logger.info("\n" + "="*80)
        self.logger.info("诊断摘要")
        self.logger.info("="*80)
        
        if not self.issues:
            self.logger.info("  ✓ 未发现明显问题")
        else:
            self.logger.error(f"  发现 {len(self.issues)} 个问题:")
            for i, issue in enumerate(self.issues, 1):
                self.logger.error(f"    {i}. {issue}")


def diagnose(
    data_dir: str,
    batch_size: int = 4,
    device: Optional[torch.device] = None
):
    """执行完整诊断"""
    
    # 设置设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建日志
    log_dir = './diagnosis_logs'
    makedirs(log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'diagnosis_{timestamp}.log')
    
    logger = logging.getLogger('diagnosis')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    logger.info("="*80)
    logger.info("NaN 损失诊断工具")
    logger.info("="*80)
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"批次大小: {batch_size}")
    logger.info(f"设备: {device}")
    logger.info(f"日志文件: {log_file}")
    
    # 创建检查器
    checker = NaNChecker(logger)
    
    # 设置随机种子
    seed_torch(42)
    
    # 创建配置
    config = TrainingConfig()
    config.data_dir = data_dir
    config.batch_size = batch_size
    config.device = device
    
    # 设置manifest路径
    config.train_manifest = os.path.join(config.data_dir, 'train_manifest.json')
    
    # 加载数据清单
    logger.info("\n加载数据清单...")
    train_data_list = load_manifest(config.train_manifest)
    if not train_data_list:
        logger.error(f"训练数据清单为空或文件不存在: {config.train_manifest}")
        return
    
    logger.info(f"训练样本数: {len(train_data_list)}")
    
    # 创建模型
    logger.info("\n创建模型...")
    from multimodal_jscc import MultimodalJSCC
    model = MultimodalJSCC(
        vocab_size=config.vocab_size,
        text_embed_dim=config.text_embed_dim,
        text_num_heads=config.text_num_heads,
        text_num_layers=config.text_num_layers,
        text_output_dim=config.text_output_dim,
        img_size=config.img_size,
        patch_size=config.patch_size,
        img_embed_dims=config.img_embed_dims,
        img_depths=config.img_depths,
        img_num_heads=config.img_num_heads,
        img_output_dim=config.img_output_dim,
        video_hidden_dim=config.video_hidden_dim,
        video_num_frames=config.video_num_frames,
        video_use_optical_flow=config.video_use_optical_flow,
        video_use_convlstm=config.video_use_convlstm,
        video_output_dim=config.video_output_dim,
        channel_type=config.channel_type,
        snr_db=config.snr_db
    )
    model = model.to(device)
    
    # 检查模型权重
    checker.check_model_weights(model)
    
    # 创建损失函数
    logger.info("\n创建损失函数...")
    from losses import MultimodalLoss
    loss_fn = MultimodalLoss(
        text_weight=config.text_weight,
        image_weight=config.image_weight,
        video_weight=config.video_weight,
        reconstruction_weight=config.reconstruction_weight,
        perceptual_weight=config.perceptual_weight,
        temporal_weight=config.temporal_weight,
        data_range=1.0
    )
    loss_fn = loss_fn.to(device)
    
    # 检查损失函数
    checker.check_loss_function(loss_fn, device)
    
    # 检查信道模块
    checker.check_channel_module(model, device)
    
    # 创建数据加载器
    logger.info("\n创建数据加载器...")
    from data_loader import MultimodalDataLoader
    data_loader_manager = MultimodalDataLoader(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=0,  # 使用0避免多进程问题
        shuffle=False,
        image_size=config.image_size,
        max_text_length=config.max_text_length,
        max_video_frames=config.max_video_frames
    )
    
    train_dataset = data_loader_manager.create_dataset(train_data_list)
    train_loader = data_loader_manager.create_dataloader(train_dataset, shuffle=False)
    
    # 获取一个批次
    logger.info("\n获取测试批次...")
    try:
        batch = next(iter(train_loader))
        logger.info("  成功获取批次")
    except Exception as e:
        logger.error(f"  ❌ 获取批次失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # 检查模型输出
    checker.check_model_outputs(model, batch, device)
    
    # 检查损失计算
    checker.check_loss_computation(model, loss_fn, batch, device)
    
    # 打印摘要
    checker.print_summary()
    
    logger.info(f"\n诊断完成！详细日志保存在: {log_file}")


def main():
    parser = argparse.ArgumentParser(description='NaN 损失诊断工具')
    parser.add_argument('--data-dir', type=str, required=True, help='数据目录')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    args = parser.parse_args()
    
    diagnose(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()

