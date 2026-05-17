#!/usr/bin/env python3
# ============================================================================
# 02c_train_vae_WORKING.py - VAE Training with All Fixes
# ============================================================================

import scanpy as sc
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

for d in ['results', 'figures', 'supplementary']:
    os.makedirs(d, exist_ok=True)

print("="*60)
print("MODEL 3/3: Variational Autoencoder Training")
print("="*60)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("[1/6] Loading data...")
input_file = "results/DL_INPUT_3000_HVG.tsv"
full_df = pd.read_table(input_file, index_col=0)

metadata_columns = ['patient_ID', 'treatment_phase', 'anatomical_location', 'PFI_category_12_months']
metadata_columns = [col for col in metadata_columns if col in full_df.columns]

obs = full_df[metadata_columns].copy()
X_df = full_df.drop(columns=metadata_columns)

if 'cell_barcode' in X_df.columns:
    X_df = X_df.drop(columns=['cell_barcode'])

X = X_df.values.astype(np.float32)
print(f"  Loaded: {X.shape[0]:,} cells × {X.shape[1]:,} genes")

# ============================================================================
# STEP 2: Normalize
# ============================================================================
print("\n[2/6] Standardizing data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.FloatTensor(X_scaled)
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# ============================================================================
# STEP 3: Define VAE
# ============================================================================
print("\n[3/6] Building VAE...")

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def get_latent(self, x):
        mu, _ = self.encode(x)
        return mu

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(input_dim=3000, latent_dim=20).to(device)
print(f"  Device: {device}")

# ============================================================================
# STEP 4: Train
# ============================================================================
print("\n[4/6] Training...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

n_epochs = 100
losses = []
best_loss = float('inf')
patience_counter = 0
max_patience = 10

for epoch in range(n_epochs):
    epoch_loss = 0
    for batch in dataloader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader) / X.shape[0]
    losses.append(avg_loss)
    scheduler.step(avg_loss)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), "results/vae_best.pth")
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    if patience_counter >= max_patience:
        print(f"  Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(torch.load("results/vae_best.pth", weights_only=True))

# ============================================================================
# STEP 5: Extract Latent Space (FIXED)
# ============================================================================
print("\n[5/6] Extracting latent representations...")

model.eval()
with torch.no_grad():
    latent = model.get_latent(X_tensor.to(device)).cpu().numpy()

adata = sc.AnnData(X=X, obs=obs)
adata.var_names = X_df.columns
adata.obs_names = full_df.index
adata.obsm["X_VAE"] = latent

# ============================================================================
# STEP 6: Save (WITH ALL FIXES)
# ============================================================================
print("\n[6/6] Saving...")

# FIX 1: anndata string array
import anndata
anndata.settings.allow_write_nullable_strings = True
for col in adata.obs.columns:
    if adata.obs[col].dtype == 'string':
        adata.obs[col] = adata.obs[col].astype('object')

adata.write("results/ovarian_VAE_atlas.h5ad")
torch.save(model.state_dict(), "results/VAE_Weights.pth")

# Training plot
plt.figure(figsize=(10, 5))
plt.plot(losses, color='#4DAF4A', linewidth=2)
plt.xlabel('Epoch', fontsize=12, fontfamily='Arial')
plt.ylabel('VAE Loss (ELBO)', fontsize=12, fontfamily='Arial')
plt.title('VAE Training Convergence', fontsize=14, fontweight='bold', fontfamily='Arial')
plt.grid(True, alpha=0.3)
plt.savefig("figures/VAE_Training_Diagnostics.png", dpi=600, bbox_inches='tight')
plt.close()

summary = {
    'model': 'VAE',
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_cells': int(adata.n_obs),
    'n_genes': int(adata.n_vars),
    'n_latent': 20,
    'final_loss': float(losses[-1]),
    'best_loss': float(best_loss),
    'n_epochs_trained': len(losses),
    'latent_variance': np.var(latent, axis=0).tolist()
}
with open("results/VAE_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("✅ VAE COMPLETE")
print("="*60)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n📁 Output files:")
print("  ✓ results/ovarian_VAE_atlas.h5ad")
print("  ✓ results/VAE_Weights.pth")
print("  ✓ results/VAE_summary.json")
print("  ✓ figures/VAE_Training_Diagnostics.png")
print("\n🔜 Next: Run benchmarking pipeline")
print("   python 04_benchmarking_stability_final.py\n")