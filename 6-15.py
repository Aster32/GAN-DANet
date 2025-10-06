import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import RelationshipLearner, Downsampler, Discriminator1, FlexibleUpsamplingModule, AttentionModule, weights_init_normal, SSIM, TVLoss
from datasets import CustomDataset, load_data
import torch.nn.functional as F
from utils import plot_results
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch.nn as nn
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import savgol_filter
#import cv2

# Gaussian Filter
def smooth_data_gaussian(data, sigma=2):
    return gaussian_filter(data, sigma=sigma)

# Median Filter
def smooth_data_median(data, size=3):
    return median_filter(data, size=size)

# Bilateral Filter
#def smooth_data_bilateral(data, d=9, sigma_color=75, sigma_space=75):
   # data_uint8 = np.uint8(data)
    #smoothed_data = cv2.bilateralFilter(data_uint8, d, sigma_color, sigma_space)
    #return smoothed_data

# Savitzky-Golay Filter
def smooth_data_savitzky_golay(data, window_length=5, polyorder=2):
    return savgol_filter(data, window_length, polyorder)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batchsizes=8

# Load and prepare data
lr_grace_05, lr_grace_025, hr_aux, grace_scaler_05, grace_scaler_025, aux_scalers = load_data()
print(np.min(hr_aux))

# Apply data smoothing to hr_aux
hr_aux = smooth_data_gaussian(hr_aux, sigma=3)

# Split data into training and testing sets
train_lr_grace_05, test_lr_grace_05, train_lr_grace_025, test_lr_grace_025, train_hr_aux, test_hr_aux = train_test_split(
    lr_grace_05, lr_grace_025, hr_aux, test_size=0.2, random_state=42)

# Create datasets and dataloaders
eventual_dataset = CustomDataset(lr_grace_05, lr_grace_025, hr_aux)
train_dataset = CustomDataset(train_lr_grace_05, train_lr_grace_025, train_hr_aux)
test_dataset = CustomDataset(test_lr_grace_05, test_lr_grace_025, test_hr_aux)

train_loader = DataLoader(train_dataset, batch_size=batchsizes)
test_loader = DataLoader(test_dataset, batch_size=batchsizes)
eventual_loader = DataLoader(eventual_dataset, batch_size=batchsizes)

# Initialize models
downsampler = Downsampler(input_channels=39, output_size=(45, 22)).to(device)
relationship_learner = RelationshipLearner(input_channels=40).to(device)
discriminator = Discriminator1(input_channels=1).to(device)
upsampling_module = FlexibleUpsamplingModule().to(device)
attention_module = AttentionModule(input_channels=39,output_channels=39).to(device)

# Initialize weights
relationship_learner.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
upsampling_module.apply(weights_init_normal)
attention_module.apply(weights_init_normal)

# Optimizers
optimizer_RL = optim.Adam(relationship_learner.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
optimizer_U = optim.Adam(upsampling_module.parameters(), lr=0.0002)
optimizer_A = optim.Adam(attention_module.parameters(), lr=0.0002)

# Learning Rate Schedulers
scheduler_RL = ReduceLROnPlateau(optimizer_RL, mode='min', factor=0.5, patience=5, verbose=True)
scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5, verbose=True)
scheduler_U = ReduceLROnPlateau(optimizer_U, mode='min', factor=0.5, patience=5, verbose=True)
scheduler_A = ReduceLROnPlateau(optimizer_A, mode='min', factor=0.5, patience=5, verbose=True)

# Loss functions
adversarial_loss = torch.nn.BCEWithLogitsLoss()
pixelwise_loss = torch.nn.MSELoss()
ssim_loss = SSIM(window_size=11, size_average=True).to(device)
tv_loss = TVLoss(weight=1e-5).to(device)

# Training loop
epochs = 20
for epoch in range(epochs):
    epoch_loss_G = 0
    epoch_loss_D = 0
    relationship_learner.train()
    for lr_grace_05, lr_grace_025, hr_aux in train_loader:
        lr_grace = F.interpolate(lr_grace_05, scale_factor=0.5, mode='bicubic', align_corners=False)
        lr_grace, hr_aux = lr_grace.to(device), hr_aux.to(device)
        lr_grace_025 = lr_grace_025.to(device)
        
        # Combine lr_grace and downsampled hr_aux
        downsampled_aux = F.interpolate(hr_aux, scale_factor=0.25, mode='bicubic', align_corners=False)
        attention_weights = attention_module(downsampled_aux)
        combined_input = torch.cat([lr_grace, downsampled_aux * attention_weights], dim=1)

        # Learn relationship features
        relationship_features = relationship_learner(combined_input)

        # Generate HR result using improved upsampling module
        hr_generated = upsampling_module(relationship_features)

        # Discriminator training
        optimizer_D.zero_grad()
        real_output = discriminator(lr_grace_025)
        fake_output = discriminator(hr_generated.detach())
        real_labels = torch.ones_like(real_output, device=device)
        fake_labels = torch.zeros_like(fake_output, device=device)

        loss_D_real = adversarial_loss(real_output, real_labels)
        loss_D_fake = adversarial_loss(fake_output, fake_labels)
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # Generator training (RelationshipLearner and ImprovedUpsamplingModule)
        optimizer_RL.zero_grad()
        optimizer_U.zero_grad()
        fake_output = discriminator(hr_generated)
        loss_G_adv = adversarial_loss(fake_output, real_labels)
        loss_G_pixel = pixelwise_loss(hr_generated, lr_grace_025)
        loss_G_ssim = 1 - ssim_loss(hr_generated, lr_grace_025)
        loss_G_tv = tv_loss(hr_generated)
        loss_G = loss_G_adv + loss_G_pixel + loss_G_ssim + loss_G_tv
        loss_G.backward()
        optimizer_RL.step()
        optimizer_U.step()

        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()

    # Update the schedulers at the end of the epoch
    scheduler_RL.step(epoch_loss_G)
    scheduler_D.step(epoch_loss_D)
    scheduler_U.step(epoch_loss_G)
    scheduler_A.step(epoch_loss_G)

    print(f'Epoch [{epoch+1}/{epochs}], Loss D: {epoch_loss_D/len(train_loader)}, Loss G: {epoch_loss_G/len(train_loader)}')

    # Plot results periodically
    if epoch % 50 == 0:
        plot_results(lr_grace[0,0].cpu(), hr_generated[0,0].cpu(), lr_grace_025[0,0].cpu())

# Evaluation
relationship_learner.eval()
upsampling_module.eval()
attention_module.eval()
with torch.no_grad():
    test_loss_G = 0
    preds = []
    trues = []
    for lr_grace_05, lr_grace_025, hr_aux in test_loader:
        lr_grace_05, lr_grace_025, hr_aux = lr_grace_05.to(device), lr_grace_025.to(device), hr_aux.to(device)
        lr_grace = F.interpolate(lr_grace_05, scale_factor=0.5, mode='bicubic', align_corners=False)
        
        # Combine lr_grace and downsampled hr_aux
        downsampled_aux = F.interpolate(hr_aux, scale_factor=0.25, mode='bicubic', align_corners=False)
        attention_weights = attention_module(downsampled_aux)
        combined_input = torch.cat([lr_grace, downsampled_aux * attention_weights], dim=1)

        # Learn relationship features
        relationship_features = relationship_learner(combined_input)

        # Generate HR result using improved upsampling module
        hr_generated = upsampling_module(relationship_features)

        # Upsample lr_grace to create the ground truth for hr_generated
        hr_grace_upsampled = lr_grace_025

        # Compute loss
        loss_G_pixel = pixelwise_loss(hr_generated, hr_grace_upsampled)
        test_loss_G += loss_G_pixel.item()

        # Save predictions and true values for metrics calculation
        preds.append(hr_generated.cpu().numpy())
        trues.append(hr_grace_upsampled.cpu().numpy())

    # Compute evaluation metrics
    preds = np.concatenate(preds, axis=0).reshape(-1)
    trues = np.concatenate(trues, axis=0).reshape(-1)

    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)

    print(f"Test MSE: {mse}, Test MAE: {mae}, Test R2: {r2}")

with torch.no_grad():
    for lr_grace_05, lr_grace_025, hr_aux in test_loader:
        lr_grace_05, lr_grace_025, hr_aux = lr_grace_05.to(device), lr_grace_025.to(device), hr_aux.to(device)
        
        # Combine lr_grace and downsampled hr_aux
        attention_weights = attention_module(hr_aux)
        combined_input = torch.cat([lr_grace_025, hr_aux * attention_weights], dim=1)

        # Learn relationship features
        relationship_features = relationship_learner(combined_input)

        # Generate HR result using improved upsampling module
        hr_generated = upsampling_module(relationship_features)

        plot_results(hr_aux[0,-1].cpu(), hr_generated[0,0].cpu(), lr_grace_025[0,0].cpu(), True)

# Save the pretrained model
torch.save(relationship_learner.state_dict(), 'relationship_learner.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
torch.save(upsampling_module.state_dict(), 'upsampling_module.pth')
torch.save(attention_module.state_dict(), 'attention_module.pth')