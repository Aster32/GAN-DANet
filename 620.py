# Assuming all required modules from model.py are imported
# Include the necessary import statements and setup from the provided code snippet

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import RelationshipLearner, Downsampler, Discriminator1, ImprovedUpsamplingModule, AttentionModule, weights_init_normal
from datasets import CustomDataset, load_data
import torch.nn.functional as F
from utils import plot_results, evaluate_metrics
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and prepare data
lr_grace_05, lr_grace_025, hr_aux, grace_scaler_05, grace_scaler_025, aux_scalers = load_data()
print(np.min(hr_aux))

# Split data into training and testing sets
train_lr_grace_05, test_lr_grace_05, train_lr_grace_025, test_lr_grace_025, train_hr_aux, test_hr_aux = train_test_split(
    lr_grace_05, lr_grace_025, hr_aux, test_size=0.2, random_state=42)

# Create datasets and dataloaders
train_dataset = CustomDataset(train_lr_grace_05, train_lr_grace_025, train_hr_aux)  # Enable augmentation for training
test_dataset = CustomDataset(test_lr_grace_05, test_lr_grace_025, test_hr_aux)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# Initialize models
downsampler = Downsampler(input_channels=39, output_size=(90, 44)).to(device)
relationship_learner = RelationshipLearner(input_channels=40).to(device)
discriminator = Discriminator1(input_channels=1).to(device)
upsampling_module = ImprovedUpsamplingModule(input_channels=40, scale_factor=4).to(device)  # Upsampling module to 0.1 degree
attention_module = AttentionModule(input_channels=39).to(device)

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

# Training loop
epochs = 50
for epoch in range(epochs):
    epoch_loss_G = 0
    epoch_loss_D = 0
    relationship_learner.train()
    for lr_grace_05, lr_grace_025, hr_aux in train_loader:
        lr_grace_05, lr_grace_025, hr_aux = lr_grace_05.to(device), lr_grace_025.to(device), hr_aux.to(device)
        hr_aux=F.interpolate(hr_aux, scale_factor=2.5, mode='bicubic', align_corners=False)
        # Downsample auxiliary data
        downsampled_aux = downsampler(hr_aux)
        attention_weights = attention_module(downsampled_aux)
        combined_input = torch.cat([lr_grace_05, downsampled_aux * attention_weights], dim=1)

        # Learn relationship features
        relationship_features = relationship_learner(combined_input)

        # Generate HR result using improved upsampling module
        hr_generated = upsampling_module(relationship_features)

        # Use lr_grace_025 as the ground truth for hr_generated
        hr_grace_upsampled = lr_grace_025

        # Discriminator training
        optimizer_D.zero_grad()
        real_output = discriminator(hr_grace_upsampled)
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
        loss_G_pixel = pixelwise_loss(hr_generated, hr_grace_upsampled)
        loss_G = loss_G_adv + loss_G_pixel
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
    if epoch % 10 == 0:
        plot_results(lr_grace_05[0,0].cpu(), hr_generated[0,0].cpu(), hr_grace_upsampled[0,0].cpu())

# Evaluation
relationship_learner.eval()
with torch.no_grad():
    test_loss_G = 0
    preds = []
    trues = []
    for lr_grace_05, lr_grace_025, hr_aux in test_loader:
        lr_grace_05, lr_grace_025, hr_aux = lr_grace_05.to(device), lr_grace_025.to(device), hr_aux.to(device)
        
        # Downsample auxiliary data
        downsampled_aux = downsampler(hr_aux)
        attention_weights = attention_module(downsampled_aux)
        combined_input = torch.cat([lr_grace_05, downsampled_aux * attention_weights], dim=1)

        # Learn relationship features
        relationship_features = relationship_learner(combined_input)

        # Generate HR result using improved upsampling module
        hr_generated = upsampling_module(relationship_features)

        # Use lr_grace_025 as the ground truth for hr_generated
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

# Save the pretrained model
torch.save(relationship_learner.state_dict(), 'relationship_learner.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
torch.save(upsampling_module.state_dict(), 'upsampling_module.pth')
torch.save(attention_module.state_dict(), 'attention_module.pth')
