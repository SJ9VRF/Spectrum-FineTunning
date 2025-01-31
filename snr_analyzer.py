import torch
import torch.nn as nn

class SNRAnalyzer:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def get_snr(self, layer):
        """
        Calculate the signal-to-noise ratio for a given layer.
        Considers any layer with trainable parameters.
        """
        if hasattr(layer, 'weight') and layer.weight is not None:
            weights = layer.weight.data
            signal = torch.mean(weights).item()
            noise = torch.std(weights).item()
            return signal / noise if noise != 0 else 0
        return 0  # Default SNR for layers without weights or with non-trainable weights

    def get_high_snr_layers(self):
        """
        Identify and return the names of layers with SNR above the defined threshold.
        Extensively checks across all common layer types that contain trainable parameters.
        """
        high_snr_layers = []
        for name, layer in self.model.named_modules():
            # Extend to include all layer types typically having trainable parameters
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.BatchNorm1d, 
                                  nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding, nn.GRU, nn.LSTM)):
                if hasattr(layer, 'weight') and layer.weight is not None:
                    snr = self.get_snr(layer)
                    if snr > self.threshold:
                        high_snr_layers.append((name, snr))
        return high_snr_layers
