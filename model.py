import torch
import torch.nn as nn
import torch.nn.functional as F
class OriginalRelationshipLearner(nn.Module):
    def __init__(self, input_channels):
        super(OriginalRelationshipLearner, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x
class Downsampler(nn.Module):
    def __init__(self, input_channels, output_size):
        super(Downsampler, self).__init__()
        self.output_size = output_size
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, input_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = F.interpolate(x, size=self.output_size, mode='bicubic', align_corners=False)
        return x
class SRGAN_d(nn.Module):
    def __init__(self, dim=64):
        super(SRGAN_d, self).__init__()
        self.conv1 = nn.Conv2d(1, dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(dim * 2)
        self.conv3 = nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(dim * 4)
        self.conv4 = nn.Conv2d(dim * 4, dim * 8, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(dim * 8)
        self.conv5 = nn.Conv2d(dim * 8, dim * 16, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(dim * 16)
        self.conv6 = nn.Conv2d(dim * 16, dim * 32, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(dim * 32)
        self.conv7 = nn.Conv2d(dim * 32, dim * 16, kernel_size=1, stride=1, padding=0)
        self.bn6 = nn.BatchNorm2d(dim * 16)
        self.conv8 = nn.Conv2d(dim * 16, dim * 8, kernel_size=1, stride=1, padding=0)
        self.bn7 = nn.BatchNorm2d(dim * 8)
        self.conv9 = nn.Conv2d(dim * 8, dim * 2, kernel_size=1, stride=1, padding=0)
        self.bn8 = nn.BatchNorm2d(dim * 2)
        self.conv10 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(dim * 2)
        self.conv11 = nn.Conv2d(dim * 2, dim * 8, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(dim * 8)
        self.add = nn.Sequential()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(dim * 8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn1(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv6(x)), 0.2)
        x = F.leaky_relu(self.bn6(self.conv7(x)), 0.2)
        x = F.leaky_relu(self.bn7(self.conv8(x)), 0.2)
        temp = x
        x = F.leaky_relu(self.bn8(self.conv9(x)), 0.2)
        x = F.leaky_relu(self.bn9(self.conv10(x)), 0.2)
        x = F.leaky_relu(self.bn10(self.conv11(x)), 0.2)
        x = x + temp
        x = self.global_avg_pool(x)  # Apply global average pooling
        x = torch.flatten(x, 1)  # Flatten the tensor, keeping the batch dimension
        x = self.dense(x)  # Fully connected layer
        return x
from torchvision import models

class PerceptualLoss(torch.nn.Module):
    def __init__(self, feature_layers=[1, 6, 11, 20], use_gpu=True):
        super(PerceptualLoss, self).__init__()
        self.feature_layers = feature_layers
        vgg = models.vgg19(pretrained=True).features
        self.vgg = torch.nn.Sequential(*list(vgg)[:max(feature_layers)+1]).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        if use_gpu:
            self.vgg = self.vgg.cuda()

    def forward(self, x, y):
        x_vgg, y_vgg = x, y
        if x_vgg.shape[1] == 1:
            x_vgg = x_vgg.repeat(1, 3, 1, 1)
        if y_vgg.shape[1] == 1:
            y_vgg = y_vgg.repeat(1, 3, 1, 1)
        
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            x_vgg = layer(x_vgg)
            y_vgg = layer(y_vgg)
            if i in self.feature_layers:
                loss += torch.nn.functional.l1_loss(x_vgg, y_vgg)
        return loss
class Discriminator1(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator1, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        # Initialize the linear layer without specifying the in_features
        self.fc1 = nn.Linear(0, 1024)  # Placeholder value for in_features
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x), 0.2)
        x = F.relu(self.conv2(x), 0.2)
        x = F.relu(self.conv3(x), 0.2)
        x = F.relu(self.conv4(x), 0.2)
        # Calculate the size of the flattened features dynamically
        batch_size = x.size(0)
        num_features = torch.prod(torch.tensor(x.size()[1:])).item()  # Total features from the conv layers
        if self.fc1.in_features == 0:  # Adjust in_features if not set
            self.fc1 = nn.Linear(num_features, 1024).to(x.device)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x), 0.2)
        x = self.fc2(x)
        return x
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class DenseLayer(nn.Module):
    def __init__(self, input_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_channels, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, input_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([DenseLayer(input_channels + i * growth_rate, growth_rate) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(self.relu(self.bn(x)))



class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = max(1, input_channels // reduction_ratio)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, reduced_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, input_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        squeeze = self.global_avg_pool(x)
        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)
        return x * excitation

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = SqueezeExcitation(channels, reduction_ratio)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.channel_attention(x)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_attention_input = torch.cat([max_out, avg_out], dim=1)
        spatial_attention = self.spatial_attention(spatial_attention_input)
        return x * spatial_attention

class SimpleAttention(nn.Module):
    def __init__(self, input_channels):
        super(SimpleAttention, self).__init__()
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define necessary modules: DenseBlock, TransitionLayer, and DANetAttention

class DenseBlock(nn.Module):
    """
    A dense block as used in DenseNet architectures.
    """
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )
        return layer

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, 1))
            features.append(new_feat)
        return torch.cat(features, 1)

class TransitionLayer(nn.Module):
    """
    A transition layer that reduces the number of features.
    """
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.layer(x)

class DANetAttention(nn.Module):
    """
    Dual Attention Network module.
    """
    def __init__(self, in_channels):
        super(DANetAttention, self).__init__()
        self.position_attention = PAM_Module(in_channels)
        self.channel_attention = CAM_Module(in_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        pa_feat = self.position_attention(x)
        ca_feat = self.channel_attention(x)
        feat = torch.cat([pa_feat, ca_feat], dim=1)
        feat = self.conv(feat)
        return feat

class PAM_Module(nn.Module):
    """
    Position Attention Module.
    """
    def __init__(self, in_channels):
        super(PAM_Module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key   = self.key_conv(x).view(batch_size, -1, height * width)
        energy     = torch.bmm(proj_query, proj_key)
        attention  = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, height * width)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return out

class CAM_Module(nn.Module):
    """
    Channel Attention Module.
    """
    def __init__(self, in_channels):
        super(CAM_Module, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, height, width = x.size()
        proj_query = x.view(batch_size, C, -1)
        proj_key   = x.view(batch_size, C, -1).permute(0, 2, 1)
        energy     = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention  = F.softmax(energy_new, dim=-1)
        proj_value = x.view(batch_size, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return out

class FlexibleUpsamplingModule(nn.Module):
    """
    A neural network module for flexible upsampling tasks with optional attention mechanisms.
    """
    def __init__(self, input_channels=40, growth_rate=24, num_blocks=3, num_layers_per_block=4, attention_type='danet'):
        super(FlexibleUpsamplingModule, self).__init__()
        self.attention_type = attention_type
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.growth_rate = growth_rate
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block
        num_features = 64  # Initial number of features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Build Dense Blocks, Transition Layers, and Attention Modules
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        self.attention_modules = nn.ModuleList()
        self.feature_channels = []  # To store number of channels at each stage

        for i in range(num_blocks):
            # Dense Block
            dense_block = DenseBlock(num_layers_per_block, num_features, growth_rate)
            self.dense_blocks.append(dense_block)
            num_features = num_features + num_layers_per_block * growth_rate  # Update num_features after Dense Block

            # Attention Module
            if self.attention_type is not None:
                attention_module = DANetAttention(num_features)
                self.attention_modules.append(attention_module)
            else:
                self.attention_modules.append(None)

            # Record feature channels after attention
            self.feature_channels.append(num_features)

            # Transition Layer (except after the last block)
            if i != num_blocks - 1:
                transition_layer = TransitionLayer(num_features, num_features // 2)
                self.transition_layers.append(transition_layer)
                num_features = num_features // 2  # Update num_features after Transition Layer

        # Define channel adjustment layers for skip connections
        self.channel_adjust_layers = nn.ModuleList()
        for in_channels in reversed(self.feature_channels):
            adjust_layer = nn.Conv2d(in_channels, 64, kernel_size=1, bias=False)
            self.channel_adjust_layers.append(adjust_layer)

        # Upsampling Layers (Using nn.Upsample instead of PixelShuffle)
        self.upsample_layers = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
        )

        # Final Convolution
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.initial_conv(x)
        previous_features = []

        for idx, (dense_block, attention_module) in enumerate(zip(self.dense_blocks, self.attention_modules)):
            x = dense_block(x)
            # Apply attention after each dense block
            if attention_module is not None:
                x = attention_module(x)
            previous_features.append(x)
            if idx < len(self.transition_layers):
                x = self.transition_layers[idx](x)

        # Upsampling Path
        x = self.upsample_layers(x)

        # Adjust previous_features before adding
        for adjust_layer, feature in zip(self.channel_adjust_layers, reversed(previous_features)):
            # Interpolate feature to match x's spatial size
            feature = F.interpolate(feature, size=x.size()[2:], mode='bilinear', align_corners=False)
            # Adjust the number of channels to match x
            feature = adjust_layer(feature)
            x = x + feature

        x = self.final_conv(x)
        return x



'''
class FlexibleUpsamplingModule(nn.Module):
    def __init__(self, input_channels=40, growth_rate=32, num_blocks=2, num_layers_per_block=4, attention_type='cbam'):
        super(FlexibleUpsamplingModule, self).__init__()
        self.attention_type = attention_type
        self.initial_conv = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)
        dense_blocks = []
        transition_layers = []
        num_features = 256
        for i in range(num_blocks):
            dense_blocks.append(DenseBlock(num_layers_per_block, num_features, growth_rate))
            num_features += growth_rate * num_layers_per_block
            if i != num_blocks - 1:
                transition_layers.append(TransitionLayer(num_features, num_features // 2))
                num_features //= 2
        self.dense_blocks = nn.ModuleList(dense_blocks)
        self.transition_layers = nn.ModuleList(transition_layers)

        self.attentions = nn.ModuleDict({
            'senet': SqueezeExcitation(num_features),
            'cbam': CBAMBlock(num_features),
            'simple': SimpleAttention(num_features),
            'nonlocal': NonLocalBlock(num_features),
            'selfattention': SelfAttention(num_features)
        })

        self.final_upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            #nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            #nn.PixelShuffle(2),
            #nn.ConvTranspose2d(num_features, num_features, kernel_size=4, stride=2, padding=1),
        )
        self.final_conv = nn.Conv2d(num_features, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.initial_conv(x)
        previous_features = []
        for dense_block, transition_layer in zip(self.dense_blocks, self.transition_layers + [None]):
            x = dense_block(x)
            previous_features.append(x)
            if transition_layer is not None:
                x = transition_layer(x)

        if self.attention_type is not None:
            x = self.attentions[self.attention_type](x)
        x = self.final_upsample(x)
        x = self.final_conv(x)
        return x
'''
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // reduction_ratio
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, c, h, w = x.size()

        # Reduce the channel dimension first
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # Efficient matrix multiplication
        f = torch.bmm(theta_x.permute(0, 2, 1), phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.bmm(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, h, w)
        
        W_y = self.W(y)
        z = W_y + x

        return z

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        value = self.value(x).view(batch_size, -1, width * height)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out
# Assuming SimpleAttention is already defined somewhere in the code
# For testing, initialize the model and run a forward pass with a random tensor
if __name__ == "__main__":
    input_tensor = torch.randn(8, 40, 45, 22)  # Example input tensor
    model = FlexibleUpsamplingModule(input_channels=40, attention_type='cbam')
    output_tensor = model(input_tensor)
    print("Output size:", output_tensor.size()) 

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

'''
def weights_init_normal(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Parameter):
        nn.init.constant_(m, 0)  # Initialize nn.Parameter with constant value
    elif isinstance(m, nn.AdaptiveAvgPool2d) or isinstance(m, nn.PixelShuffle):
        pass  # These layers don't have weights to initialize
    elif isinstance(m, AttentionModule):
        weights_init_normal(m.query_conv)
        weights_init_normal(m.key_conv)
        weights_init_normal(m.value_conv)
        nn.init.constant_(m.gamma, 0)  # Initialize gamma parameter in AttentionModule
    elif isinstance(m, SqueezeExcitation):
        weights_init_normal(m.fc1)
        weights_init_normal(m.fc2)
'''

class TVLoss(nn.Module):
    def __init__(self, weight=1):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.tensor([torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            self.window = window
        
        window = window.to(img1.device)

        self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

# Channel Attention Module (unchanged)
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(channels // reduction_ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        try:
            b, c, h, w = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            y = self.sigmoid(y)
            return x * y
        except Exception as e:
            raise ValueError(f"Error in ChannelAttention forward pass: {e}")

# Window Attention Module (unchanged)
class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super(WindowAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = coords.reshape(2, -1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        try:
            B_, N, C = x.shape  # x is of shape (B*num_windows, N, C)
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # (3, B_, num_heads, N, head_dim)
            q, k, v = qkv.unbind(0)  # Each: (B_, num_heads, N, head_dim)

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # (B_, num_heads, N, N)

            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                N, N, -1
            )  # (N, N, num_heads)
            relative_position_bias = (
                relative_position_bias.permute(2, 0, 1).contiguous()
            )  # (num_heads, N, N)
            attn = attn + relative_position_bias.unsqueeze(0)
            attn = attn.softmax(dim=-1)

            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            return x
        except Exception as e:
            raise ValueError(f"Error in WindowAttention forward pass: {e}")

# Feed-Forward Network (FFN) for Transformer-like block
class FeedForward(nn.Module):
    """
    A simple MLP-like feed-forward block to enhance representation capabilities.
    Typical in transformer-based architectures (SwinIR, ViT).
    """
    def __init__(self, channels, expansion_factor=4, drop=0.0):
        super(FeedForward, self).__init__()
        hidden_dim = channels * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # x: (B*num_windows, N, C)
        return self.net(x)

# Hybrid Attention Block (HAB) enhanced for Super-Resolution
class HAB(nn.Module):
    def __init__(self, channels, window_size, num_heads, residual_scale=0.1, ffn_expansion=4):
        super(HAB, self).__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(channels)
        self.channel_attention = ChannelAttention(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.window_attention = WindowAttention(channels, num_heads, window_size)
        # Additional FFN after window attention
        self.norm3 = nn.LayerNorm(channels)
        self.ffn = FeedForward(channels, expansion_factor=ffn_expansion)

        # Residual scaling factor to stabilize training
        self.residual_scale = residual_scale

    def forward(self, x):
        try:
            # Channel Attention Branch
            residual_ca = x
            x_ca = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
            x_ca = self.norm1(x_ca)
            x_ca = x_ca.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            x_ca = self.channel_attention(x_ca)
            x = x_ca * self.residual_scale + residual_ca

            # Window Attention + FFN Branch
            residual_wa = x
            x_wa = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
            x_wa = self.norm2(x_wa)
            B, H, W, C = x_wa.shape

            # Handle window partitioning
            pad_h = (self.window_size - H % self.window_size) % self.window_size
            pad_w = (self.window_size - W % self.window_size) % self.window_size
            if pad_h > 0 or pad_w > 0:
                x_wa = F.pad(x_wa, (0, 0, 0, pad_w, 0, pad_h))
            Hp = H + pad_h
            Wp = W + pad_w

            x_wa = x_wa.view(
                B,
                Hp // self.window_size,
                self.window_size,
                Wp // self.window_size,
                self.window_size,
                C,
            )
            x_wa = x_wa.permute(0, 1, 3, 2, 4, 5).contiguous()
            x_wa = x_wa.view(-1, self.window_size * self.window_size, C)

            # Window Attention
            x_wa = self.window_attention(x_wa)

            # FFN
            x_wa = self.norm3(x_wa)
            x_wa = x_wa + self.ffn(x_wa) * self.residual_scale

            # Merge windows
            x_wa = x_wa.view(
                B,
                Hp // self.window_size,
                Wp // self.window_size,
                self.window_size,
                self.window_size,
                C,
            )
            x_wa = x_wa.permute(0, 1, 3, 2, 4, 5).contiguous()
            x_wa = x_wa.view(B, Hp, Wp, C)

            if pad_h > 0 or pad_w > 0:
                x_wa = x_wa[:, :H, :W, :].contiguous()

            x_wa = x_wa.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            x = x_wa * self.residual_scale + residual_wa
            return x
        except Exception as e:
            raise ValueError(f"Error in HAB forward pass: {e}")

# Residual Hybrid Attention Group (RHAG) (unchanged)
class RHAG(nn.Module):
    def __init__(self, channels, num_habs, window_size, num_heads, residual_scale=0.1):
        super(RHAG, self).__init__()
        self.habs = nn.ModuleList(
            [HAB(channels, window_size, num_heads, residual_scale=residual_scale) for _ in range(num_habs)]
        )
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        try:
            residual = x
            for hab in self.habs:
                x = hab(x)
            x = self.conv(x)
            x = x + residual
            return x
        except Exception as e:
            raise ValueError(f"Error in RHAG forward pass: {e}")

# Full HAT Network with Enhancements
class HAT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=1,
        channels=64,
        num_groups=4,
        num_habs=6,
        window_size=8,
        num_heads=8,
        upscale_factor=4,
        device=None,
    ):
        super(HAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.upscale_factor = upscale_factor
        self.device = device or torch.device('cpu')

        self.entry = nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=1, padding=1
        ).to(self.device)
        self.groups = nn.ModuleList(
            [
                RHAG(channels, num_habs, window_size, num_heads, residual_scale=0.1)
                for _ in range(num_groups)
            ]
        ).to(self.device)
        self.conv_after_body = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1
        ).to(self.device)
        self.upsample = self._make_upsample_layer()
        self.exit = nn.Conv2d(
            channels, out_channels, kernel_size=3, stride=1, padding=1
        ).to(self.device)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            ).to(self.device)
        else:
            self.residual_conv = nn.Identity()

    def _make_upsample_layer(self):
        layers = []
        num_upsamples = int(self.upscale_factor / 2)
        for _ in range(num_upsamples):
            layers += [
                nn.Conv2d(
                    self.channels, self.channels * 4, kernel_size=3, stride=1, padding=1
                ),
                nn.PixelShuffle(2),
            ]
        return nn.Sequential(*layers).to(self.device)

    def forward(self, x):
        try:
            x = x.to(self.device)
            B, C, H, W = x.shape

            # Initial residual connection
            residual = F.interpolate(
                x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False
            )
            residual = self.residual_conv(residual)

            x = self.entry(x)
            res = x.clone()

            for group in self.groups:
                x = group(x)

            x = self.conv_after_body(x)
            x = x + res

            x = self.upsample(x)
            x = self.exit(x)

            # Ensure shapes match before adding residual
            assert x.shape == residual.shape, (
                f"Shape mismatch: x.shape={x.shape}, residual.shape={residual.shape}"
            )

            x = x + residual
            return x
        except Exception as e:
            raise ValueError(f"Error in HAT forward pass: {e}")
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class LPIPSNet(nn.Module):
    """
    LPIPS network for single or three-channel input,
    adapted initialization for lin layers as identity for Conv2d.
    """
    def __init__(self, net='vgg', requires_grad=False, single_channel=False, normalize_inputs=True):
        super(LPIPSNet, self).__init__()
        self.single_channel = single_channel
        self.normalize_inputs = normalize_inputs

        # Mean/Std for normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        # If single_channel, map (B,1,H,W) to (B,3,H,W)
        if self.single_channel:
            self.input_convert = nn.Conv2d(1, 3, kernel_size=1, bias=False)
            with torch.no_grad():
                # Initialize to replicate the single channel to all three channels equally
                self.input_convert.weight.fill_(1.0/3.0)
        else:
            self.input_convert = nn.Identity()

        # Load VGG16 features
        vgg_pretrained = models.vgg16(pretrained=True).features
        # LPIPS uses layers after conv1_2, conv2_2, conv3_3, conv4_3, conv5_3
        self.slice1 = nn.Sequential(*[vgg_pretrained[i] for i in range(0,4)])   # relu1_2
        self.slice2 = nn.Sequential(*[vgg_pretrained[i] for i in range(4,9)])   # relu2_2
        self.slice3 = nn.Sequential(*[vgg_pretrained[i] for i in range(9,16)])  # relu3_3
        self.slice4 = nn.Sequential(*[vgg_pretrained[i] for i in range(16,23)]) # relu4_3
        self.slice5 = nn.Sequential(*[vgg_pretrained[i] for i in range(23,30)]) # relu5_3

        # Channels at each stage: [64, 128, 256, 512, 512]
        channels_list = [64, 128, 256, 512, 512]
        # Learned linear layers for weighting feature differences
        self.lins = nn.ModuleList([nn.Conv2d(ch, ch, kernel_size=1, bias=False) for ch in channels_list])

        # If not requires grad, freeze parameters
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        # Initialize conv weights as identity: out_ch=ch, in_ch=ch, kernel=1
        # We'll set lin.weight[i,i,0,0]=1.0 and others=0
        for lin, ch in zip(self.lins, channels_list):
            with torch.no_grad():
                lin.weight.zero_()
                for i in range(ch):
                    lin.weight[i, i, 0, 0] = 1.0

    def load_lpips_weights(self, weight_path):
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = self.mean.device

        weights = torch.load(weight_path, map_location=device)
        own_state = self.state_dict()

        # official LPIPS weight keys like 'lin0.weight', 'lin0.bias', etc.
        for k,v in weights.items():
            if k in own_state:
                own_state[k].copy_(v)

    def forward_once(self, x):
        # Extract multi-level features
        h0 = self.slice1(x)
        h1 = self.slice2(h0)
        h2 = self.slice3(h1)
        h3 = self.slice4(h2)
        h4 = self.slice5(h3)
        return [h0,h1,h2,h3,h4]

    def forward(self, x, y):
        # Convert single-channel to 3-channel if needed
        if self.single_channel:
            x = self.input_convert(x)
            y = self.input_convert(y)

        # Normalize if requested
        if self.normalize_inputs:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std

        feats_x = self.forward_once(x)
        feats_y = self.forward_once(y)

        layer_diff_list = []
        for i in range(len(feats_x)):
            x_f = feats_x[i]
            y_f = feats_y[i]
            x_norm = F.normalize(x_f, p=2, dim=1)
            y_norm = F.normalize(y_f, p=2, dim=1)

            diff = (x_norm - y_norm) ** 2
            diff = self.lins[i](diff)  # shape: (B, C, H, W)

            # First average spatially
            diff_spatial = diff.mean([2,3])  # shape: (B, C)

            # Then average over channels
            diff_scalar = diff_spatial.mean(1)  # shape: (B,)

            layer_diff_list.append(diff_scalar)

        # Stack results from all layers: shape (num_layers, B)
        dist_all = torch.stack(layer_diff_list, dim=0)

        # Sum over layers
        dist_per_image = dist_all.sum(dim=0)  # shape (B,)

        # Average over batch
        dist = dist_per_image.mean()  # scalar

        return dist