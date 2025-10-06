# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 22:23:00 2024

@author: 17689
"""
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import OriginalRelationshipLearner, Discriminator1, FlexibleUpsamplingModule, weights_init_normal, SSIM, TVLoss, PerceptualLoss
from datasets import CustomDataset, load_data
import torch.nn.functional as F
from utils import plot_results
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pandas as pd
from taylorDiagram import TaylorDiagram
from torchvision import models

class ModelTrainer:
    def __init__(self, epochs, batch_size, relationship_learner, relationship_output_channels, smoothing_method=None, attention=None, senet=None):
        self.epochs = epochs
        self.batch_size = batch_size
        #self.relationship_learner = relationship_learner
        self.relationship_output_channels = relationship_output_channels
        self.smoothing_method = smoothing_method
        self.attention = attention
        self.senet = senet
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load and prepare data
        self.lr_grace_05, self.lr_grace_025, self.hr_aux, self.grace_scaler_05, self.grace_scaler_025, self.aux_scalers = load_data()
        
        # Apply data smoothing to hr_aux if smoothing_method is specified
        if self.smoothing_method:
            self.hr_aux = self.smoothing_method(self.hr_aux)
        else:
            self.hr_aux = self.hr_aux
        
        # Split data into training and testing sets
        self.train_lr_grace_05, self.test_lr_grace_05, self.train_lr_grace_025, self.test_lr_grace_025, self.train_hr_aux, self.test_hr_aux = train_test_split(
            self.lr_grace_05, self.lr_grace_025, self.hr_aux, test_size=0.2, random_state=42)
        
        # Create datasets and dataloaders
        self.train_dataset = CustomDataset(self.train_lr_grace_05, self.train_lr_grace_025, self.train_hr_aux)
        self.test_dataset = CustomDataset(self.test_lr_grace_05, self.test_lr_grace_025, self.test_hr_aux)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
        
        # Initialize models
        #self.relationship_learner = self.relationship_learner.to(self.device)
        self.discriminator = Discriminator1().to(self.device)
        self.upsampling_module = FlexibleUpsamplingModule(input_channels=self.hr_aux.shape[-1]+1,attention_type=self.attention).to(self.device)
        self.flag=self.attention
        self.attention=None
        # Initialize optional modules
        if self.attention:
            self.attention_module = self.attention.to(self.device)
        else:
            self.attention_module = None
            
        if self.senet:
            self.senet_module = self.senet.to(self.device)
        else:
            self.senet_module = None
        
        # Initialize weights
        #self.relationship_learner.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
        self.upsampling_module.apply(weights_init_normal)
        if self.attention_module:
            self.attention_module.apply(weights_init_normal)
        if self.senet_module:
            self.senet_module.apply(weights_init_normal)
        
        # Optimizers
        #self.optimizer_RL = optim.Adam(self.relationship_learner.parameters(), lr=0.0002)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        self.optimizer_U = optim.Adam(self.upsampling_module.parameters(), lr=0.0002)
        if self.attention_module:
            self.optimizer_A = optim.Adam(self.attention_module.parameters(), lr=0.0002)
        if self.senet_module:
            self.optimizer_SE = optim.Adam(self.senet_module.parameters(), lr=0.0002)
        
        # Learning Rate Schedulers
        #self.scheduler_RL = ReduceLROnPlateau(self.optimizer_RL, mode='min', factor=0.5, patience=5, verbose=True)
        self.scheduler_D = ReduceLROnPlateau(self.optimizer_D, mode='min', factor=0.5, patience=5, verbose=True)
        self.scheduler_U = ReduceLROnPlateau(self.optimizer_U, mode='min', factor=0.5, patience=5, verbose=True)
        if self.attention_module:
            self.scheduler_A = ReduceLROnPlateau(self.optimizer_A, mode='min', factor=0.5, patience=5, verbose=True)
        if self.senet_module:
            self.scheduler_SE = ReduceLROnPlateau(self.optimizer_SE, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Loss functions
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        self.pixelwise_loss = torch.nn.MSELoss()
        self.ssim_loss = SSIM(window_size=11, size_average=True).to(self.device)
        self.tv_loss = TVLoss(weight=1e-5).to(self.device)
        self.perceptual_loss = PerceptualLoss(use_gpu=torch.cuda.is_available())
        #self.perceptual_loss = PerceptualLoss([1, 6, 11, 20], use_gpu=torch.cuda.is_available())
    def smooth_data_gaussian(self, data, sigma=2):
        return gaussian_filter(data, sigma=sigma)

    def smooth_data_median(self, data, size=3):
        return median_filter(data, size=size)

    def smooth_data_savitzky_golay(self, data, window_length=5, polyorder=2):
        return savgol_filter(data, window_length, polyorder)

    def train(self):
        train_losses_G = []
        train_losses_D = []
        for epoch in range(self.epochs):
            epoch_loss_G = 0
            epoch_loss_D = 0
            #self.relationship_learner.train()
            for lr_grace_05, lr_grace_025, hr_aux in self.train_loader:
                lr_grace = F.interpolate(lr_grace_05, scale_factor=0.5, mode='bicubic', align_corners=False)
                lr_grace, hr_aux = lr_grace.to(self.device), hr_aux.to(self.device)
                lr_grace_025 = lr_grace_025.to(self.device)
                
                # Combine lr_grace and downsampled hr_aux
                downsampled_aux = F.interpolate(hr_aux, scale_factor=0.25, mode='bicubic', align_corners=False)
                combined_input = torch.cat([lr_grace, downsampled_aux], dim=1)
                # Learn relationship features
                #relationship_features = self.relationship_learner(combined_input)
                relationship_features = combined_input
                # Apply attention or SENet if exists
                if self.attention_module:
                    relationship_features = self.attention_module(relationship_features)
                elif self.senet_module:
                    relationship_features = self.senet_module(relationship_features)

                # Generate HR result using improved upsampling module
                hr_generated = self.upsampling_module(relationship_features)
                # Discriminator training
                self.optimizer_D.zero_grad()
                real_output = self.discriminator(lr_grace_025)
                fake_output = self.discriminator(hr_generated.detach())
                real_labels = torch.ones_like(real_output, device=self.device)
                fake_labels = torch.zeros_like(fake_output, device=self.device)

                loss_D_real = self.adversarial_loss(real_output, real_labels)
                loss_D_fake = self.adversarial_loss(fake_output, fake_labels)
                loss_D = (loss_D_real + loss_D_fake) / 2
                loss_D.backward()
                self.optimizer_D.step()

                # Generator training (RelationshipLearner and ImprovedUpsamplingModule)
                #self.optimizer_RL.zero_grad()
                self.optimizer_U.zero_grad()
                fake_output = self.discriminator(hr_generated)
                loss_G_adv = self.adversarial_loss(fake_output, real_labels)
                loss_G_pixel = self.pixelwise_loss(hr_generated, lr_grace_025)
                loss_G_ssim = 1 - self.ssim_loss(hr_generated, lr_grace_025)
                loss_G_tv = self.tv_loss(hr_generated)
                loss_G_perceptual = self.perceptual_loss(hr_generated, lr_grace_025)
                loss_G = loss_G_adv + loss_G_pixel+loss_G_perceptual#+ loss_G_pixel#+ loss_G_ssim # + loss_G_pixel # #+ loss_G_tv+ loss_G_ssim
                loss_G.backward()
                #self.optimizer_RL.step()
                self.optimizer_U.step()

                epoch_loss_G += loss_G.item()
                epoch_loss_D += loss_D.item()

            # Update the schedulers at the end of the epoch
            #self.scheduler_RL.step(epoch_loss_G)
            self.scheduler_D.step(epoch_loss_D)
            self.scheduler_U.step(epoch_loss_G)
            if self.attention_module:
                self.scheduler_A.step(epoch_loss_G)
            if self.senet_module:
                self.scheduler_SE.step(epoch_loss_G)

            train_losses_G.append(epoch_loss_G / len(self.train_loader))
            train_losses_D.append(epoch_loss_D / len(self.train_loader))

            #print(f'Epoch [{epoch+1}/{self.epochs}], Loss D: {epoch_loss_D/len(self.train_loader):.4f}, Loss G: {epoch_loss_G/len(self.train_loader):.4f}')

        return train_losses_G, train_losses_D

    def evaluate(self):
       # self.relationship_learner.eval()
        self.upsampling_module.eval()
        if self.attention_module:
            self.attention_module.eval()
        if self.senet_module:
            self.senet_module.eval()
        with torch.no_grad():
            preds = []
            trues = []
            bs=0
            for lr_grace_05, lr_grace_025, hr_aux in self.test_loader:
                
                bs=bs+1
                if bs==1 :
                    lr_grace_05, lr_grace_025, hr_aux = lr_grace_05.to(self.device), lr_grace_025.to(self.device), hr_aux.to(self.device)
                    
                    # Combine lr_grace and downsampled hr_aux
                    combined_input = torch.cat([lr_grace_025, hr_aux], dim=1)

                    # Learn relationship features
                    #relationship_features = self.relationship_learner(combined_input)
                    relationship_features = combined_input
                    # Apply attention or SENet if exists
                    if self.attention_module:
                        relationship_features = self.attention_module(relationship_features)
                    elif self.senet_module:
                        relationship_features = self.senet_module(relationship_features)

                    # Generate HR result using improved upsampling module
                    hr_generated = self.upsampling_module(relationship_features)

                    plot_results(lr_grace_05[0,0].cpu(), hr_generated[0,0].cpu(), lr_grace_025[0,0].cpu(), True)
                # Save predictions and true values for metrics calculation
                lr_grace_05, lr_grace_025, hr_aux = lr_grace_05.to(self.device), lr_grace_025.to(self.device), hr_aux.to(self.device)
                lr_grace = F.interpolate(lr_grace_05, scale_factor=0.5, mode='bicubic', align_corners=False)
                
                # Combine lr_grace and downsampled hr_aux
                downsampled_aux = F.interpolate(hr_aux, scale_factor=0.25, mode='bicubic', align_corners=False)
                combined_input = torch.cat([lr_grace, downsampled_aux], dim=1)

                # Learn relationship features
                #relationship_features = self.relationship_learner(combined_input)
                relationship_features = combined_input
                # Apply attention or SENet if exists
                if self.attention_module:
                    relationship_features = self.attention_module(relationship_features)
                elif self.senet_module:
                    relationship_features = self.senet_module(relationship_features)

                # Generate HR result using improved upsampling module
                hr_generated = self.upsampling_module(relationship_features)

                # Upsample lr_grace to create the ground truth for hr_generated
                hr_grace_upsampled = lr_grace_025
                preds.append(hr_generated.cpu().numpy())
                trues.append(hr_grace_upsampled.cpu().numpy())

            # Compute evaluation metrics
            preds = np.concatenate(preds, axis=0).reshape(-1)
            trues = np.concatenate(trues, axis=0).reshape(-1)

            mse = mean_squared_error(trues, preds)
            mae = mean_absolute_error(trues, preds)
            r2 = r2_score(trues, preds)

            print(f"Test MSE: {mse}, Test MAE: {mae}, Test R²: {r2}")

        return preds, trues, r2

# Set parameters
epochs = 100
batch_size = 16

# Define smoothing method
smoothing_method = ModelTrainer(epochs, batch_size, OriginalRelationshipLearner(40), 1024).smooth_data_gaussian
smoothing_method = None
# Define modules
#attention_module = AttentionModule(input_channels=40, output_channels=40)
#senet_module = SqueezeExcitation(input_channels=40, reduction_ratio=8)

# Train the baseline model without any additional module
model1 = ModelTrainer(epochs=epochs, batch_size=batch_size, relationship_learner=OriginalRelationshipLearner(40), relationship_output_channels=1024, smoothing_method=smoothing_method)
train_losses_G1, train_losses_D1 = model1.train()
preds1, trues1, r2_1 = model1.evaluate()
# Release GPU memory
torch.cuda.empty_cache()
# Train the model with Attention
model2 = ModelTrainer(epochs=epochs, batch_size=batch_size, relationship_learner=OriginalRelationshipLearner(40), relationship_output_channels=1024, smoothing_method=smoothing_method, attention='senet')
train_losses_G2, train_losses_D2 = model2.train()
preds2, trues2, r2_2 = model2.evaluate()
# Release GPU memory
torch.cuda.empty_cache()
# Train the model with SEnet
model3 = ModelTrainer(epochs=epochs, batch_size=batch_size, relationship_learner=OriginalRelationshipLearner(40), relationship_output_channels=1024, smoothing_method=smoothing_method, attention='simple')
train_losses_G3, train_losses_D3 = model3.train()
preds3, trues3, r2_3 = model3.evaluate()
# Release GPU memory
torch.cuda.empty_cache()
model4 = ModelTrainer(epochs=epochs, batch_size=batch_size, relationship_learner=OriginalRelationshipLearner(40), relationship_output_channels=1024, smoothing_method=smoothing_method, attention='cbam')
train_losses_G4, train_losses_D4 = model4.train()
preds4, trues4, r2_4 = model4.evaluate()
torch.cuda.empty_cache()
model5 = ModelTrainer(epochs=epochs, batch_size=batch_size, relationship_learner=OriginalRelationshipLearner(40), relationship_output_channels=1024, smoothing_method=smoothing_method, attention='nonlocal')
train_losses_G5, train_losses_D5 = model5.train()
preds5, trues5, r2_5 = model5.evaluate()
torch.cuda.empty_cache()
model6 = ModelTrainer(epochs=epochs, batch_size=batch_size, relationship_learner=OriginalRelationshipLearner(40), relationship_output_channels=1024, smoothing_method=smoothing_method, attention='selfattention')
train_losses_G6, train_losses_D6 = model6.train()
preds6, trues6, r2_6 = model6.evaluate()
torch.cuda.empty_cache()
# Plot training losses for each model
plt.figure(figsize=(12, 6))
plt.plot(train_losses_G1, label='Generator Loss - Baseline')
plt.plot(train_losses_D1, label='Discriminator Loss - Baseline')
plt.plot(train_losses_G2, label='Generator Loss - With SEnet')
plt.plot(train_losses_D2, label='Discriminator Loss - With SEnet')
plt.plot(train_losses_G3, label='Generator Loss - With None')
plt.plot(train_losses_D3, label='Discriminator Loss - With None')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Losses')
plt.show()

# Prepare data for Taylor diagram
ref = pd.Series(trues1, name='Reference')
samples = pd.DataFrame({
    'Baseline': preds1,
    'With SEnet': preds2,
    'With Simple': preds3,
    'With CBAM': preds4,
    'With nonlocal': preds5,
    'With selfattention': preds6,
})

# Compute standard deviations and correlations
stddev = samples.std(axis=0)
corrcoef = samples.corrwith(ref)

# Create Taylor diagram for evaluation
fig = plt.figure(figsize=(10, 10))
dia = TaylorDiagram(ref.std(), fig=fig, rect=111, label="Reference")

colors = plt.matplotlib.cm.jet(np.linspace(0, 1, len(samples.columns)))

# Add models to Taylor diagram
for i, (stddev, corrcoef) in enumerate(zip(stddev.values, corrcoef.values)):
    dia.add_sample(stddev, corrcoef,
                   marker='o', ms=10, ls='',
                   mfc=colors[i], mec=colors[i],
                   label=samples.columns[i])

# Add grid and contours
dia.add_grid()
contours = dia.add_contours(levels=5, colors='0.5')
plt.clabel(contours, inline=1, fontsize=10, fmt='%.2f')

# Add legend
fig.legend(dia.samplePoints,
           [p.get_label() for p in dia.samplePoints],
           numpoints=1, prop=dict(size='small'), loc='upper right')
fig.suptitle("Taylor Diagram", size='x-large')
plt.show()

# Save the pretrained models
#torch.save(model1.relationship_learner.state_dict(), 'model1_relationship_learner.pth')
#torch.save(model2.relationship_learner.state_dict(), 'model2_relationship_learner.pth')
#torch.save(model3.relationship_learner.state_dict(), 'model3_relationship_learner.pth')
torch.save(model1.discriminator.state_dict(), 'model1_discriminator.pth')
torch.save(model2.discriminator.state_dict(), 'model2_discriminator.pth')
torch.save(model3.discriminator.state_dict(), 'model3_discriminator.pth')
torch.save(model1.upsampling_module.state_dict(), 'model1_upsampling_module.pth')
torch.save(model2.upsampling_module.state_dict(), 'model2_upsampling_module.pth')
torch.save(model3.upsampling_module.state_dict(), 'model3_upsampling_module.pth')
if model2.attention_module:
    torch.save(model2.attention_module.state_dict(), 'model2_attention_module.pth')
if model3.senet_module:
    torch.save(model3.senet_module.state_dict(), 'model3_senet_module.pth')
