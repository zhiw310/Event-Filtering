"""
Event Signal Filtering via Probability Flux Estimation

Implementation inspired by the paper:
"Event Signal Filtering via Probability Flux Estimation"

This module aims to filter event signals by estimating the probability flux at threshold boundaries
of the underlying irradiance diffusion process, which can be used to enhance signal fidelity
by mitigating internal randomness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EventDensityFlowFilter(nn.Module):
    """
    Event Density Flow Filter (EDFilter) module that models event correlation by
    estimating continuous probability flux from discrete input events.
    
    This is a simplified version inspired by the original paper, adapted to work
    as a preprocessing module for event-based object detection.
    """
    def __init__(self, channels, reduction_ratio=8, kernel_size=3, temporal_window=10, gamma=0.5):
        """
        Args:
            channels: Number of input channels
            reduction_ratio: Channel reduction ratio for the attention mechanism
            kernel_size: Spatial kernel size for local pattern extraction
            temporal_window: Temporal window size for event correlation
            gamma: Gamma parameter for scaled sigmoid activation
        """
        super(EventDensityFlowFilter, self).__init__()
        
        self.channels = channels
        self.gamma = gamma
        self.temporal_window = temporal_window
        
        # Spatial feature extraction
        self.spatial_conv = nn.Conv2d(
            channels, channels, 
            kernel_size=kernel_size, 
            padding=kernel_size//2, 
            groups=channels,
            bias=False
        )
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_channels = max(8, channels // reduction_ratio)
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=True)
        
        # Non-local event correlation
        self.estimate_flux = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        
        self.estimate_reliability = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _channel_attention(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        
        out = avg_out + max_out
        return torch.sigmoid(out)
    
    def _compute_event_density(self, x):
        """
        Estimate the event density flow based on the input tensor
        """
        # Extract spatial features
        spatial_features = self.spatial_conv(x)
        
        # Apply channel attention
        channel_weights = self._channel_attention(spatial_features)
        spatial_features = spatial_features * channel_weights
        
        # Estimate probability flux
        flux = self.estimate_flux(spatial_features)
        reliability = self.estimate_reliability(spatial_features)
        
        return flux, reliability
    
    def forward(self, x):
        """
        Args:
            x: Input event representation tensor [B, C, H, W]
        
        Returns:
            Filtered event representation tensor [B, C, H, W]
        """
        # Compute event density flow
        flux, reliability = self._compute_event_density(x)
        
        # Apply the filtering operation
        # We use a residual connection to learn the perturbations/noise
        x_filtered = x + (flux * reliability * self.gamma)
        
        # Scale with sigmoid to maintain range
        scaled_factor = 2.0 * torch.sigmoid(x_filtered) - 1.0
        
        # Apply the scaling only to non-zero values to preserve sparsity
        mask = (x != 0).float()
        x_out = x * (1.0 + scaled_factor * mask)
        
        return x_out


class EventFilterModule(nn.Module):
    """
    A module that applies event filtering to enhance signal fidelity
    """
    def __init__(self, input_channels=20, filter_type='density_flow'):
        super(EventFilterModule, self).__init__()
        
        self.filter_type = filter_type
        
        if filter_type == 'density_flow':
            self.filter = EventDensityFlowFilter(
                channels=input_channels,
                reduction_ratio=4,
                kernel_size=3,
                temporal_window=10,
                gamma=0.5
            )
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
    
    def forward(self, x):
        """
        Apply event filtering to the input tensor
        
        Args:
            x: Input event representation tensor [B, C, H, W]
        
        Returns:
            Filtered event representation tensor [B, C, H, W]
        """
        return self.filter(x) 