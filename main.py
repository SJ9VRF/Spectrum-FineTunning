from transformers import AutoModel
from snr_analyzer import SNRAnalyzer
from spectrum_trainer import SpectrumTrainer

# Load model
model = AutoModel.from_pretrained("bert-base-uncased")

# SNR Analysis
snr_analyzer = SNRAnalyzer(model)
high_snr_layers = snr_analyzer.get_high_snr_layers()

# Setup Trainer
trainer = SpectrumTrainer(model, high_snr_layers, None, None)
trainer.train()
