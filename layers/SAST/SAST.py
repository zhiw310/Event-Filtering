# ==============================================================================
# 修改 SAST.py 文件
# ==============================================================================

from functools import partial
from typing import Optional, Tuple, List

import math
import torch
from omegaconf import DictConfig
from torch import nn

from .layers import DropPath
from .layers import get_act_layer, get_norm_layer
from .layers import to_2tuple
from .ops import window_partition, window_reverse, grid_partition, \
                 grid_reverse, LayerScale, MLP


class AdaptiveThresholdLearner(nn.Module):
    def __init__(self, dim, init_bounce=1e-3, num_stages=4):
        super().__init__()
        # 为不同stage创建不同的基础阈值参数
        # 不同stage使用不同的初始缩放因子 - 从浅到深，逐渐降低缩放因子
        # 这样深层网络的阈值更高，选择的token更少，剪枝更激进
        initial_scales = [5.0, 2.5, 1.0, 1.0]
        self.stage_scales = nn.Parameter(torch.tensor(initial_scales[:num_stages]))
        # self.base_bounces = nn.Parameter(torch.ones(num_stages) * init_bounce)
        self.base_bounces = init_bounce
        # 在初始化函数中添加
        #self.bias = nn.Parameter(torch.tensor(0.5))  # 初始化为0.5
        self.num_stages = num_stages

        # 场景感知网络
        self.scene_encoder = nn.Sequential(
            nn.Linear(dim, dim//4),
            nn.ReLU(inplace=True),
            nn.Linear(dim//4, 1)
        )
        
        # 事件密度编码器
        self.density_encoder = nn.Sequential(
            nn.Linear(20, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )
        
        # 为不同stage创建不同的温度参数 - 深层网络使用更高的温度，软化阈值
        temperature_values = [0.2, 0.25, 0.3, 0.35]
        self.temperatures = nn.Parameter(torch.tensor(temperature_values[:num_stages]))
        
    def forward(self, x, r, stage_id=0):
        # x: 输入特征 [B, H, W, C] 或 [B*N, H*W, C]
        # r: 事件非零比例 [B, 20]
        # stage_id: 当前stage的ID，默认为0
        
        # 确保stage_id在有效范围内
        stage_id = min(stage_id, self.num_stages - 1)
        
        # 获取特征均值作为场景表示
        if x.dim() == 4:
            scene_feat = x.mean(dim=(1, 2))  # [B, C]
        else:
            scene_feat = x.mean(dim=1)  # [B*N, C]
        
        # 计算场景因子 - 使用tanh而不是sigmoid，允许正负值
        # 范围为[-1, 1] * 0.9 = [-0.9, 0.9]
        scene_factor = torch.tanh(self.scene_encoder(scene_feat)) * 0.45 - 0.1
        
        # 计算密度因子 - 将r的通道维度平均
        #r_mean = r.mean(dim=1, keepdim=True)  # [B, 1]
        # 允许正负值，范围为[-0.9, 0.9]
        density_factor = torch.tanh(self.density_encoder(r)) * 0.4
        
        # 计算自适应阈值，使用当前stage的base_bounce
        base_bounce = self.base_bounces
        stage_scale = self.stage_scales[stage_id].sigmoid() * 0.201 + 0.800
        
        
        # 加入stage_scale因子，允许不同stage有不同权重
        adaptive_bounce = base_bounce * (1.0 + scene_factor + density_factor) * stage_scale
        
        # 返回当前stage的温度参数
        temperature = self.temperatures[stage_id]
        
        return adaptive_bounce, temperature
    
    def compute_soft_mask(self, scores, d, stage_id=0, adaptive_bounce=None, temperature=None):
        """计算软掩码，而不是二元选择"""
        # scores: 评分张量 [B, N] 或 [B*N, H*W]
        # d: 期望选择的比例 (float)
        # stage_id: 当前stage的ID
        
        if adaptive_bounce is None:
            # 获取当前stage的base_bounce
            adaptive_bounce = self.base_bounces[stage_id]
            
        if temperature is None:
            # 获取当前stage的temperature
            temperature = self.temperatures[stage_id]
            
        # 计算阈值
        # 确保阈值与scores形状兼容，可能需要广播
        if isinstance(adaptive_bounce, float) or isinstance(adaptive_bounce, int) or adaptive_bounce.numel() == 1:
            # 如果是标量，直接计算
            threshold = d / (1 + adaptive_bounce)
        else:
            # 如果是张量，尝试广播到scores的形状
            # 先将adaptive_bounce调整为[B,1]形状，以便广播
            if adaptive_bounce.dim() > 1:
                adaptive_bounce = adaptive_bounce.view(-1)
            threshold = d / (1 + adaptive_bounce.unsqueeze(-1))
        
        # 使用sigmoid函数创建软掩码，temperature控制平滑程度
        # 当分数远高于阈值时，掩码值接近1；远低于阈值时，掩码值接近0
        soft_mask = torch.sigmoid((scores - threshold) / temperature)
        
        return soft_mask


class SAST_block(nn.Module):
    ''' SAST block contains two SAST layers '''

    def __init__(
            self,
            dim: int,
            attention_cfg: DictConfig,
            first_block: bool=False,
            stage_id: int=0,  # 添加stage_id参数
    ):
        super().__init__()
        norm_eps = attention_cfg.get('norm_eps', 1e-5)
        partition_size = attention_cfg.partition_size
        dim_head = attention_cfg.get('dim_head', 32)
        attention_bias = attention_cfg.get('attention_bias', True)
        mlp_act_string = attention_cfg.mlp_activation
        mlp_bias = attention_cfg.get('mlp_bias', True)
        mlp_expand_ratio = attention_cfg.get('mlp_ratio', 4)

        drop_path = attention_cfg.get('drop_path', 0.0)
        drop_mlp = attention_cfg.get('drop_mlp', 0.0)
        ls_init_value = attention_cfg.get('ls_init_value', 1e-5)
        
        if isinstance(partition_size, int):
            partition_size = to_2tuple(partition_size)
        else:
            partition_size = tuple(partition_size)
            assert len(partition_size) == 2
        self.partition_size = partition_size

        norm_layer = partial(get_norm_layer('layernorm'), eps=norm_eps)

        mlp_act_layer = get_act_layer(mlp_act_string)

        sub_layer_params = (ls_init_value, drop_path, mlp_expand_ratio, mlp_act_layer, mlp_bias, drop_mlp)
        
        self_attn_module = MS_WSA
        self.enable_CB = attention_cfg.get('enable_CB', False)

        self.win_attn = self_attn_module(dim,
                                         dim_head=dim_head,
                                         bias=attention_bias,
                                         sub_layer_params=sub_layer_params,
                                         norms=[norm_layer(dim), norm_layer(dim)])

        self.grid_attn = self_attn_module(dim,
                                          dim_head=dim_head,
                                          bias=attention_bias,
                                          sub_layer_params=sub_layer_params,
                                          norms=[norm_layer(dim), norm_layer(dim)])
        
        if first_block:
            self.to_scores = nn.Linear(dim, dim)
            self.to_controls = PositiveLinear(20, dim, bias=False)
            torch.nn.init.constant_(self.to_controls.weight, 1)
            self.act = nn.ReLU()
            
            # 添加自适应阈值学习器
            init_bounce = attention_cfg.get('BOUNCE', 1e-3)
            
            # 获取阈值学习器的学习率缩放因子，默认为0.1，即比全局学习率小10倍
            threshold_lr_scale = attention_cfg.get('threshold_lr_scale', 0.01)
            
            # 获取stage数量，从配置文件中获取或使用默认值4
            num_stages = attention_cfg.get('num_stages', 4)
            
            self.threshold_learner = AdaptiveThresholdLearner(dim, init_bounce, num_stages)
            
            # 为阈值学习器的参数注册学习率缩放因子
            for param in self.threshold_learner.parameters():
                param.register_hook(lambda grad: grad * threshold_lr_scale)

        self.amp_value = attention_cfg.get('AMP', 2e-4)
        self.bounce_value = attention_cfg.get('BOUNCE', 1e-3)
        self.first_block = first_block
        self.B, self.N, self.dim = None, None, dim
        self.stage_id = stage_id  # 存储stage_id

    def window_selection(self, scores: torch.Tensor, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, h, w = self.B, self.N, self.partition_size[0], self.partition_size[1]
        temp = h * w
        norm_window = (torch.norm(scores, dim=[2, 3], p=1) / temp).softmax(-1) 
        
        # 获取自适应阈值和温度参数，传递stage_id
        adaptive_bounce, temperature = self.threshold_learner(scores, r, self.stage_id)
        
        # 将adaptive_bounce变成标量用于索引选择
        adaptive_bounce_scalar = adaptive_bounce.mean()
        
        # 计算窗口级别的软掩码分数
        d = 1/N
        threshold = d / (1 + adaptive_bounce_scalar)
        
        # 计算标准化的窗口分数
        flat_norm_window = norm_window.view(B, N)
        
        # 应用sigmoid函数获取软掩码
        window_soft_mask_values = torch.sigmoid((flat_norm_window - threshold) / temperature)
        
        # 硬选择用于前向传播
        with torch.no_grad():
            index_window = get_score_index_2d21d(flat_norm_window, 1/N, adaptive_bounce_scalar)
        # 添加保护措施
        if len(index_window) == 0:
            print("警告: 没有窗口被选择，强制选择最高分窗口")
            # 选择每个批次中得分最高的窗口
            _, top_indices = torch.max(flat_norm_window, dim=1)
            # 构建1D索引
            batch_indices = torch.arange(B, device=scores.device)
            index_window = batch_indices * N + top_indices
        # 返回硬选择的索引、软掩码值和自适应阈值
        return index_window, window_soft_mask_values, adaptive_bounce
    
    def token_selection(self, scores: torch.Tensor, index_window: torch.Tensor, adaptive_bounce: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, h, w = self.B, self.N, self.partition_size[0], self.partition_size[1]
        temp = 1
        norm_token = (torch.norm(scores, dim=[3], p=1) / temp).view(B * N, -1)[index_window].softmax(-1)
        
        # 获取温度参数
        temperature = self.threshold_learner.temperatures[self.stage_id]
        
        # 将adaptive_bounce变成标量，使用平均值
        adaptive_bounce_scalar = adaptive_bounce.mean().item()
        
        # 计算token的软掩码
        with torch.no_grad():
            index_token, asy_index_partition, K = get_score_index_with_padding(norm_token, 1 / (h * w), adaptive_bounce_scalar)
        
        # 计算token级别的软掩码分数
        d = 1 / (h * w)
        threshold = d / (1 + adaptive_bounce_scalar)
        
        # 创建一个与XX张量大小相同的全零掩码
        # 这样掩码就与asy_index张量兼容
        batch_size = B * N
        hw = h * w
        token_soft_mask_full = torch.zeros(batch_size * hw, device=scores.device)
        
        # 计算选定索引处的sigmoid掩码值
        # 确保只计算真正选中的索引位置
        # 使用asy_index_partition，而不是整个norm_token
        selected_scores = torch.gather(norm_token.flatten(), 0, asy_index_partition)
        token_values_flat = torch.sigmoid((selected_scores - threshold) / temperature)
        
        # 只为选定的token设置值
        token_soft_mask_full[asy_index_partition] = token_values_flat
        
        return index_token, asy_index_partition, K, token_soft_mask_full
        
    def _partition_attn(self, x: torch.Tensor, pos_emb: torch.Tensor, r: torch.Tensor, index_list: List) -> Tuple[torch.Tensor, ...]:
        index_count = 0
        self.B = x.shape[0]
        img_size = x.shape[1:3]
        self.N = img_size[0] * img_size[1] // (self.partition_size[0] * self.partition_size[1])

        ''' First SAST Layer '''
        x = x + pos_emb(x)
        x = window_partition(x, self.partition_size).view(self.B, self.N, -1, self.dim) 
        if self.first_block:
            # Scoring Module* 
            scale = self.to_controls(r + 1e-6)[:, None, None, :]  
            scores = self.act(self.to_scores(x)) 

            # STP Weighting
            weight = scale.sigmoid() * scores.sigmoid() 
            x = (weight * x).view(self.B * self.N, -1, self.dim) # Weight x use sigmoid scores 

            # Selection Module 
            scale = self.amp_value / scale
            scale[scale==torch.inf] = 0
            scores = scale * scores
            
            # 使用自适应阈值和软掩码
            index_window, window_soft_mask, adaptive_bounce = self.window_selection(scores, r)
            index_token, asy_index, K, token_soft_mask = self.token_selection(scores, index_window, adaptive_bounce)
            padding_index = index_token[torch.isin(index_token, asy_index, assume_unique=True, invert=True)] # Get padding index
            index_list1 = [index_window, index_token, padding_index, asy_index, K, window_soft_mask, token_soft_mask] # Buffer index list for reusing
        else:
            # Reuse index list
            x = x.view(self.B * self.N, -1, self.dim)
            index_list1, index_list2 = index_list
            index_window, index_token, padding_index, asy_index, K, window_soft_mask, token_soft_mask = index_list1
        M = len(index_window)
        
        if len(index_token):
            # MS-WSA (Masked Sparse Window Self-Attention) 
            x = self.win_attn(x, index_window, index_token, padding_index, asy_index, M, self.B, 
                             self.enable_CB, window_soft_mask, token_soft_mask)
        x = window_reverse(x, self.partition_size, (img_size[0], img_size[1]))
        
        index_count += len(asy_index) // self.B

        ''' Second SAST Layer '''
        if self.first_block:
            # Reuse scores* for the second SAST layer
            scores = window_reverse(scores.view_as(x), self.partition_size, (img_size[0], img_size[1]))
            scores = grid_partition(scores, self.partition_size).view(self.B, self.N, -1, self.dim)

            # Selection Module 
            index_window, window_soft_mask, adaptive_bounce = self.window_selection(scores, r)
            index_token, asy_index, K, token_soft_mask = self.token_selection(scores, index_window, adaptive_bounce)
            padding_index = index_token[torch.isin(index_token, asy_index, assume_unique=True, invert=True)]
            index_list2 = [index_window, index_token, padding_index, asy_index, K, window_soft_mask, token_soft_mask]
        else:
            index_window, index_token, padding_index, asy_index, K, window_soft_mask, token_soft_mask = index_list2
        x = x.view(self.B, img_size[0], img_size[1], self.dim)
        x = grid_partition(x, self.partition_size).view(self.B * self.N, -1, self.dim)
        
        M = len(index_window)
        if len(index_token): 
            # MS-WSA (Masked Sparse Window Self-Attention) 
            x = self.grid_attn(x, index_window, index_token, padding_index, asy_index, M, self.B, 
                              self.enable_CB, window_soft_mask, token_soft_mask)
        x = grid_reverse(x, self.partition_size, (img_size[0], img_size[1]))
        index_count += len(asy_index) // self.B
        return x, index_count, [index_list1, index_list2]

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, r: torch.Tensor, index_list: List, stage_idx: int = None) -> Tuple[torch.Tensor, ...]:
        # 如果传入了stage_idx，则使用传入的值
        original_stage_id = self.stage_id
        if stage_idx is not None:
            self.stage_id = stage_idx
        
        try:
            x, index_count, index_list = self._partition_attn(x, pos_emb, r, index_list)
            return x, index_count, index_list
        finally:
            # 恢复原始stage_id
            self.stage_id = original_stage_id
    

class MS_WSA(nn.Module):
    ''' Masked Sparse Window (multi-head) Self-Attention (MS-WSA) '''
    ''' Channels-last (B, ..., C) '''

    def __init__(
            self,
            dim: int,
            dim_head: int = 32,
            bias: bool = True,
            sub_layer_params: Optional[List[nn.Module]] = None,
            norms: nn.Module = None,):
        super().__init__()
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.norm1 =norms[0]

        ls_init_value, drop_path, mlp_expand_ratio, mlp_act_layer, mlp_bias, drop_mlp = sub_layer_params
        self.ls1 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop1 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norms[1]
        self.mlp = MLP(dim=dim, channel_last=True, expansion_ratio=mlp_expand_ratio,
                       act_layer=mlp_act_layer, bias=mlp_bias, drop_prob=drop_mlp)
        self.ls2 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop2 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.sub_layers = nn.ModuleList([self.ls1, self.drop1, self.norm2, self.mlp, self.ls2, self.drop2])
            
        self.eps = 1e-6
        

    def forward(self, x: torch.Tensor, index_window: torch.Tensor, 
                index_token: torch.Tensor, padding_index: torch.Tensor, 
                asy_index: torch.Tensor, M: int, B: torch.Tensor, enable_CB: bool,
                window_soft_mask: torch.Tensor = None, token_soft_mask: torch.Tensor = None) -> torch.Tensor:
        
        N, C = x.shape[0], x.shape[-1]
        restore_shape = x.shape
        x = x.view(N, -1, C)
        x = self.norm1(x)  
        if len(index_token) == 0: # No selected tokens
            return x.view(*restore_shape)
        
        # Gather selected tokens, X and XX are used to store the original tokens and selected windows.
        X = x.clone() 
        x = x[index_window].view(-1, C) 
        XX = x.clone() 
        x[asy_index] = self.norm2(x[asy_index])  
        shortcut = x[asy_index]  
        x = x[index_token].view(M, -1, C)  

        # Attention
        q, k, v = self.qkv(x).view(M, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Column masking, the padded positions are masked.
        attn_map = torch.zeros((XX.shape[0], q.shape[2], self.num_heads), device=x.device, dtype=attn.dtype) 
        attn_map[index_token] = attn.transpose(1, 3).reshape(-1, q.shape[2], self.num_heads) 
        attn_map[padding_index] = -1e4 
        attn = attn_map[index_token].view(M, -1, q.shape[2], self.num_heads).transpose(1, 3) 

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2)
        x = self.proj(x.reshape(M, -1, C)).to(XX.dtype)

        XX[index_token] = x.view(-1, C)
        x = XX[asy_index] 

        # 应用软掩码，保证梯度传递
        if token_soft_mask is not None:
            # 使用软掩码创建一个残差连接的形式
            # 当软掩码为1时，完全使用新特征；为0时，完全保留原始特征
            original_x = shortcut
            # 直接使用token_soft_mask，不需要索引asy_index
            # 取对应位置的掩码值
            token_mask_values = token_soft_mask[asy_index].unsqueeze(-1)
            x = original_x * (1 - token_mask_values) + x * token_mask_values

        x = shortcut + self.drop1(self.ls1(x))
        shortcut = x
        x = self.mlp(x).to(X.dtype)

        # Context Broadcasting operation
        if enable_CB: 
            temp_X, temp_XX = torch.zeros_like(X), torch.zeros_like(XX)
            temp_XX[asy_index] = x
            temp_X[index_window] = temp_XX.view(M, -1, C)
            temp_X = temp_X.view(B, -1, C)
            temp_X = (0.5 * temp_X + (1 - 0.5) * temp_X.mean(dim=1, keepdim=True)).view(*restore_shape)
            x = temp_X[index_window].view(-1, C)[asy_index]

        x = shortcut + self.drop2(self.ls2(x))

        # 同样应用软掩码到最终输出
        if window_soft_mask is not None and token_soft_mask is not None:
            # 计算综合掩码，结合window和token的软掩码
            # 创建一个与XX张量相同大小的全零掩码
            window_mask_expanded = window_soft_mask.new_zeros(XX.shape[0])
            
            # 获取与index_window对应的window_soft_mask值
            # 从原始形状[B,N]中提取正确的值
            # 注意：B是作为参数传入的批次大小
            N = window_soft_mask.shape[1]
            idx_batch = index_window // N  # 获取batch索引
            idx_window = index_window % N  # 获取窗口索引
            selected_mask_values = window_soft_mask[idx_batch, idx_window]
            
            # 将选定的掩码值分配给对应的窗口索引
            window_mask_expanded[index_window] = selected_mask_values
            
            # 计算综合掩码
            combined_mask = window_mask_expanded[asy_index].unsqueeze(-1) * token_soft_mask[asy_index].unsqueeze(-1)
            original_x = X[index_window].view(-1, C)[asy_index]
            x = original_x * (1 - combined_mask) + x * combined_mask

        # Scatter selected tokens back to the original position.
        XX[asy_index] = x.view(-1, C)
        XX[padding_index] = X[index_window].view(-1, C)[padding_index]
        X[index_window] = XX.view(M, -1, C) 
        x = X.view(*restore_shape) 
        return x


def get_score_index_2d21d(x: torch.Tensor, d: float, b: float) -> torch.Tensor:
    '''2D window index selection'''
    if x.shape[0] == 1:
        # Batch size 1 is a special case because torch.nonzero returns a 1D tensor already.
        return torch.nonzero(x >= d / (1 + b))[:, 1]
    # The selected window indices (asychronous indices).
    gt = x >= d / (1 + b)
    index_2d = torch.nonzero(gt)
    index_1d = index_2d[:, 0] * x.shape[-1] + index_2d[:, 1]
    return index_1d


def get_score_index_with_padding(x: torch.Tensor, d: float, b: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''2D token index selection (w/ and w/o paddings)'''
    if torch.isnan(x).any():
        print("警告: 输入张量包含NaN值")
    gt = x >= d / (1 + b)
    K = torch.sum(gt, dim=1)
    
    # 安全检查：确保每行至少选择一个token
    if K.sum() == 0:
        print(f"警告所有元素都小于阈值{d / (1 + b)}")
        # 如果没有token满足条件，则为每行选择得分最高的token
        _, top_indices_fallback = torch.topk(x, k=1, dim=1)
        K = torch.ones_like(K)
        # 更新gt，确保每行有一个为True
        batch_indices = torch.arange(x.size(0), device=x.device)
        gt = torch.zeros_like(gt, dtype=torch.bool)
        gt[batch_indices, top_indices_fallback.squeeze(1)] = True
    
    # 获取最大K值，确保不为0
    max_k = max(K.max().item(), 1)
    
    # The top-k indices are idealized padded token indices.
    top_indices = torch.topk(x, k=max_k, dim=1, largest=True, sorted=False)[1]
    # Adding offsets to the top-k indices.
    arange = torch.arange(0, x.shape[0] * x.shape[1], x.shape[1], device=x.device).view(-1, 1)
    # The actual selected token indices (asychronous indices).
    index_2d = torch.nonzero(gt)
    index_1d = index_2d[:, 0] * x.shape[-1] + index_2d[:, 1]
    return (top_indices + arange).view(-1), index_1d, K


class PositiveLinear(nn.Module):
    ''' Linear layer with positive weights'''
    def __init__(self, in_features, out_features, bias=True):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Apply exponential function to ensure weights are positive
        positive_weights = torch.exp(self.weight)
        return nn.functional.linear(input, positive_weights, self.bias)