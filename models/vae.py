import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Bloc Residual : Permet au gradient de circuler sans perte d'information.
    Essentiel pour préserver les détails fins comme le texte et les bordures de fenêtres.
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.leaky_relu(x + self.conv_block(x), 0.2)

class VAE(nn.Module):
    def __init__(self, latent_dim=2048, img_size=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # ---------------------------------------------------------
        # ENCODER : Compresse l'image 128x128 en vecteur conceptuel
        # ---------------------------------------------------------
        self.encoder = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(64),
            
            # 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(128),
            
            # 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(256),
            
            # 16x16 -> 8x8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(512)
        )
        
        # Taille après les 4 convolutions : 512 filtres * 8px * 8px
        self.flat_size = 512 * (img_size // 16) * (img_size // 16)
        
        # Couches de goulot d'étranglement (Bottleneck)
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
        # ---------------------------------------------------------
        # DECODER : Reconstruit l'image à partir du vecteur
        # ---------------------------------------------------------
        self.fc_decode = nn.Linear(latent_dim, self.flat_size)
        
        # On utilise Upsample + Conv au lieu de ConvTranspose pour éviter le quadrillage
        self.decoder_layers = nn.ModuleList([
            # 8x8 -> 16x16
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(256)
            ),
            # 16x16 -> 32x32
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(128)
            ),
            # 32x32 -> 64x64
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(64)
            ),
            # 64x64 -> 128x128
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(64, 3, kernel_size=3, padding=1),
                nn.Sigmoid() # Pour ramener les pixels entre 0 et 1
            )
        ])

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        # Redimensionnement vers la forme 512x8x8
        h = h.view(h.size(0), 512, self.img_size // 16, self.img_size // 16)
        
        x = h
        for layer in self.decoder_layers:
            x = layer(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss_function(recon, x, mu, logvar, beta, edge_weight, chroma_weight):
    """
    LOSS TITAN v3.5 - 'The Text Seer'
    Intègre un filtre Laplacien pour forcer la reconstruction des caractères.
    """
    
    # 1. Reconstruction Globale (Boostée pour le contraste)
    # L1 est plus 'pointilleux' que MSE pour le texte.
    recon_loss = F.l1_loss(recon, x, reduction='mean') * 20.0
    
    # 2. PERTE DE LAPLACIEN (Détection des micro-détails/Texte)
    # Ce filtre mathématique isole uniquement les changements brusques de pixels
    def get_laplacian(img):
        # Noyau Laplacien standard (3x3)
        kernel = torch.tensor([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=torch.float32).to(img.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        return F.conv2d(img, kernel, padding=1, groups=3)

    lap_recon = get_laplacian(recon)
    lap_real = get_laplacian(x)
    # On punit sévèrement si les 'bords' des lettres ne correspondent pas
    lap_loss = F.l1_loss(lap_recon, lap_real, reduction='mean') * 800.0
    
    # 3. EDGE LOSS (Gradients de fenêtres)
    v_recon = torch.abs(recon[:, :, :, :-1] - recon[:, :, :, 1:])
    v_real = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    h_recon = torch.abs(recon[:, :, :-1, :] - recon[:, :, 1:, :])
    h_real = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    edge_loss = F.l1_loss(v_recon, v_real) + F.l1_loss(h_recon, h_real)

    # 4. CHROMA & KL (Stabilité et Organisation)
    chroma_loss = F.l1_loss(recon[:, :, 1:, :] - recon[:, :, :-1, :], 
                           x[:, :, 1:, :] - x[:, :, :-1, :])
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Total avec pondération Titan
    return recon_loss + \
           (edge_weight * edge_loss) + \
           lap_loss + \
           (chroma_weight * chroma_loss) + \
           (beta * kl_loss)