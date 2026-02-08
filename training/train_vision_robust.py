"""
=============================================================================
TITAN ENGINE v3.1 - VISION TRAINING SYSTEM (ULTRA ROBUST & THREADED)
=============================================================================
Conçu pour l'entraînement haute performance sur GPU AMD (ROCm) avec 12 Go VRAM.
Optimisé pour le traitement de datasets massifs (21 Go+) par segmentation 
de mémoire et combustion directe en VRAM.

Dernière mise à jour : Intégration du Multi-Threading I/O et Vectorisation NumPy.
=============================================================================
"""

import os
import sys
import shutil
import gc
import glob
import json
import signal
import tempfile
import uuid
import atexit
import time
import logging
import traceback
import threading
import queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# --- VÉRIFICATION DES DÉPENDANCES CRITIQUES ---
try:
    import psutil
except ImportError:
    print(" >>> [AUTO-INSTALL] psutil manquant. Installation en cours...")
    os.system(f"{sys.executable} -m pip install psutil")
    import psutil

try:
    import numpy as np
    import torch
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision.utils import save_image
except ImportError as e:
    print(f" >>> [ERREUR] Dépendances manquantes (numpy, torch, torchvision).")
    print(f"Détails : {e}")
    sys.exit(1)

# =============================================================================
# --- CONFIGURATION DU MATÉRIEL (AMD / ROCm) ---
# =============================================================================
# Ces paramètres forcent le comportement du driver pour éviter les fuites de 
# mémoire et les instabilités de cache MIOpen sur les cartes AMD.
# -----------------------------------------------------------------------------
os.environ['MIOPEN_DISABLE_CACHE'] = '1'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['MIOPEN_FIND_MODE'] = '1'

# Injection du chemin projet pour les imports locaux
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import du modèle VAE synchronisé
try:
    from models.vae import VAE, vae_loss_function
except ImportError:
    print(" >>> [ERREUR] Impossible de trouver models/vae.py. Vérifiez votre structure de dossiers.")
    sys.exit(1)

# =============================================================================
# --- TABLEAU DE BORD DES TUNING KNOBS (GLOBAL CONFIG) ---
# =============================================================================
# Pilote ton entraînement ici.
# -----------------------------------------------------------------------------
# 1. Hyperparamètres de Vision
LEARNING_RATE      = 8e-5      # Vitesse d'apprentissage adaptative
BATCH_SIZE         = 64       # Densité de combustion (Batch massif)
LATENT_DIM         = 2048      # Capacité du goulot d'étranglement (Cerveau)
CROP_SIZE          = 128       # Résolution de l'analyse (Patchs)
EPOCHS             = 100       # Cycles de vie maximum du projet

# 2. Paramètres Overdrive (Gestion CPU / RAM)
NUM_THREADS         = 6        # Threads CPU pour le chargement (Ryzen 7)
MAX_IMAGES_PER_POOL = 12000     # Nombre d'images max stockées en RAM
MIN_SYS_RAM_GB      = 4.0       # Seuil de panique RAM système
FILES_PER_LOAD      = 16        # Nombre de fichiers .npz traités par cycle de thread

# 3. Tuning de la Perte (The Multi-Anchor System)
EDGE_WEIGHT        = 500.0      # Poids de la netteté EPOCH < 25: 100; > 25: 500
CHROMA_WEIGHT      = 30.0       # Poids de la fidélité des couleurs
BETA               = 0.00000001    # Poids de la régularisation KL (Cerveau)

# 4. Monitoring et Visuels
VISUALIZE_EVERY_N_POOLS  = 2   # Snapshots visuels tous les X cycles
CHECKPOINT_EVERY_N_POOLS = 1   # Sauvegarde des poids tous les X cycles
# -----------------------------------------------------------------------------

# --- DÉFINITION DES CHEMINS SYSTÈME ---
CHECKPOINT_DIR = os.path.join(project_root, "checkpoints")
RESULTS_DIR = os.path.join(project_root, "results")
DATA_DIR = os.path.join(project_root, "data")
LOG_FILE = os.path.join(CHECKPOINT_DIR, f"titan_train_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
STATE_FILE = os.path.join(CHECKPOINT_DIR, "training_state.json")

# Création des infrastructures physiques
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- CONFIGURATION DU LOGGING INDUSTRIEL ---
logger = logging.getLogger("TITAN")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# =============================================================================
# --- CLASSES ET UTILITAIRES DE SÉCURITÉ ---
# =============================================================================

class InMemoryDataset(Dataset):
    """Encapsulation des données résidentes en RAM pour PyTorch."""
    def __init__(self, tensor_data):
        self.data = tensor_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def get_vram_usage():
    """Mesure chirurgicale de l'occupation mémoire du GPU en Go."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0

def cleanup_memory():
    """Libération forcée des ressources mémoire (Python & CUDA)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def check_panic_switches():
    """Moniteur de sécurité système en temps réel."""
    mem = psutil.virtual_memory()
    avail_sys_gb = mem.available / (1024**3)
    
    if avail_sys_gb < MIN_SYS_RAM_GB:
        logger.critical(f"PANIC RAM SYSTÈME : Seulement {avail_sys_gb:.2f} Go restants. Arrêt de sécurité.")
        return True
    
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        if peak_vram > 11.6:
            logger.critical(f"PANIC VRAM : Pic de consommation à {peak_vram:.2f} Go détecté.")
            return True
            
    return False

# Enregistre le nettoyage automatique
atexit.register(cleanup_memory)

# =============================================================================
# --- DATA LOADING ENGINE (THREADED & VECTORIZED) ---
# =============================================================================

def load_single_file(fp):
    """
    Travailleur CPU Unitaire : Décompresse, Croppe et Normalise un fichier .npz.
    Utilise NumPy pour relâcher le GIL.
    """
    try:
        with np.load(fp) as data:
            if 'images' not in data:
                return None
            raw_images = data['images']
            
            n, h, w, c = raw_images.shape
            if h < CROP_SIZE or w < CROP_SIZE:
                return None
            
            # Stratégie de data-augmentation (Crops multiples vectorisés)
            # On prend un crop aléatoire par image présente dans le fichier
            y = np.random.randint(0, h - CROP_SIZE, size=n)
            x = np.random.randint(0, w - CROP_SIZE, size=n)
            
            # Allocation d'un bloc contigu pour la rapidité
            patches = np.zeros((n, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
            for i in range(n):
                patches[i] = raw_images[i, y[i]:y[i]+CROP_SIZE, x[i]:x[i]+CROP_SIZE]
            
            # Conversion BGR -> RGB et retour
            return patches[:, :, :, ::-1].copy()
    except Exception as e:
        return None

def background_loader_manager(all_files, start_ptr, pool_queue, stop_event):
    """
    Gère le thread principal de chargement qui distribue le travail 
    aux sous-threads (ThreadPoolExecutor).
    """
    current_ptr = start_ptr
    total_files = len(all_files)
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        while not stop_event.is_set():
            vram_resident_pool = []
            current_pool_size = 0
            
            while current_pool_size < MAX_IMAGES_PER_POOL and not stop_event.is_set():
                batch_end_ptr = min(current_ptr + FILES_PER_LOAD, total_files)
                files_to_open = all_files[current_ptr : batch_end_ptr]
                
                if not files_to_open:
                    current_ptr = 0  # Rebouclage infini sur le dataset
                    continue
                
                # Distribution parallèle sur les threads
                results = list(executor.map(load_single_file, files_to_open))
                
                for res in results:
                    if res is not None:
                        # Conversion NumPy -> Tenseur CPU (Float32)
                        # On normalise (0-1) et on permute les canaux (NHWC -> NCHW)
                        cpu_tensor = torch.from_numpy(
                            res.astype(np.float32) / 255.0
                        ).permute(0, 3, 1, 2)
                        
                        vram_resident_pool.append(cpu_tensor)
                        current_pool_size += res.shape[0]
                
                current_ptr = batch_end_ptr

            if vram_resident_pool and not stop_event.is_set():
                # Fusion finale avant envoi au GPU
                full_pool = torch.cat(vram_resident_pool, dim=0)
                pool_queue.put((full_pool, current_ptr))

# =============================================================================
# --- PERSISTANCE DES ÉTATS (SAVE/LOAD) ---
# =============================================================================

def save_titan_checkpoint(model, optimizer, epoch, file_ptr, loss):
    """Sauvegarde les poids du modèle et l'état de progression JSON."""
    weights_path = os.path.join(CHECKPOINT_DIR, "vae_latest.pth")
    
    # 1. Sauvegarde des poids neuronaux
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, weights_path)

    # 2. Sauvegarde des métadonnées de session
    state = {
        "timestamp": datetime.now().isoformat(),
        "epoch": epoch, 
        "file_index": file_ptr, 
        "last_loss": float(loss),
        "tuning": {
            "edge_weight": EDGE_WEIGHT,
            "chroma_weight": CHROMA_WEIGHT,
            "beta": BETA,
            "lr": optimizer.param_groups[0]['lr']
        }
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)
    logger.info(f" >>> [SAVE] Checkpoint : Epoch {epoch}, File {file_ptr}, Loss {loss:.4f}")

def load_titan_checkpoint(model, optimizer):
    """Restaure la session d'entraînement précédente."""
    weights_path = os.path.join(CHECKPOINT_DIR, "vae_latest.pth")
    start_epoch, start_file_ptr = 0, 0
    
    if os.path.exists(weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(" >>> [LOAD] Poids neuronaux restaurés.")
        except Exception as e:
            logger.error(f"Échec restauration poids : {e}")

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state_data = json.load(f)
                start_epoch = state_data.get("epoch", 0)
                start_file_ptr = state_data.get("file_index", 0)
                logger.info(f" >>> [LOAD] Session reprise à : Epoch {start_epoch}, Fichier {start_file_ptr}")
        except:
            logger.warning("Fichier d'état illisible. Reprise à zéro.")
            
    return start_epoch, start_file_ptr

# =============================================================================
# --- PRÉ-VOL : SANITY CHECK ---
# =============================================================================

def sanity_check(model, device):
    """Vérifie la stabilité du matériel et la validité mathématique."""
    logger.info(">>> Lancement du Sanity Check (Test de combustion)...")
    try:
        dummy = torch.randn(1, 3, CROP_SIZE, CROP_SIZE).to(device)
        with torch.amp.autocast('cuda'):
            recon, mu, logvar = model(dummy)
            loss = vae_loss_function(recon, dummy, mu, logvar, BETA, EDGE_WEIGHT, CHROMA_WEIGHT)
        loss.backward()
        optimizer = optim.Adam(model.parameters(), lr=1e-6)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del dummy, recon, mu, logvar, loss, optimizer
        cleanup_memory()
        logger.info(">>> Sanity Check RÉUSSI. Matériel stable.")
    except Exception as e:
        logger.critical(f"ÉCHEC DU SANITY CHECK : {e}")
        sys.exit(1)

# =============================================================================
# --- MOTEUR TITAN : DYNAMIQUE D'ENTRAÎNEMENT ---
# =============================================================================

def train_overdrive():
    # 1. INITIALISATION MATÉRIELLE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"TITAN ENGINE v3.1 INITIALISÉ SUR : {device}")
    
    # 2. CONSTRUCTION DU MODÈLE
    model = VAE(latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. RESTAURATION DE SESSION
    start_epoch, start_file_ptr = load_titan_checkpoint(model, optimizer)
    sanity_check(model, device)
    
    # Scheduler adaptatif (Réduit le LR si la perte stagne)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Scaler pour précision mixte (Auto-AMP) pour GPU AMD
    scaler = torch.amp.GradScaler('cuda')
    torch.backends.cudnn.benchmark = True 
    
    # 4. PRÉPARATION DU DATASET
    data_pattern = os.path.join(DATA_DIR, "*.npz")
    all_files = sorted(glob.glob(data_pattern))
    if not all_files:
        logger.error(f"Données introuvables dans {DATA_DIR}. Entraînement impossible.")
        return

    # 5. GESTIONNAIRE D'INTERRUPTION
    stop_event = threading.Event()
    def signal_handler(sig, frame):
        logger.info("\n[!] INTERRUPTION : Signal reçu. Arrêt sécurisé...")
        stop_event.set()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Variables de contrôle de session
    pool_queue = queue.Queue(maxsize=2)
    pool_counter = 0
    global_file_ptr = start_file_ptr
    current_epoch = start_epoch
    last_loop_ptr = start_file_ptr
    
    # Lancement du pipeline de chargement threadé
    loader_thread = threading.Thread(
        target=background_loader_manager, 
        args=(all_files, global_file_ptr, pool_queue, stop_event)
    )
    loader_thread.daemon = True
    loader_thread.start()

    try:
        # 6. BOUCLE MAITRESSE (THE INFINITE BURN)
        while not stop_event.is_set() and current_epoch < EPOCHS:
            if check_panic_switches(): break
            
            # --- PHASE A : RAVITAILLEMENT (RAM -> VRAM) ---
            logger.info(f"\n--- EPOCH {current_epoch} | RAVITAILLEMENT PIPELINE ---")
            try:
                # On attend que le pool de threads ait préparé un bloc de données
                res = pool_queue.get(timeout=20) 
                current_vram_data_cpu, global_file_ptr = res
                
                # Transfert asynchrone vers la VRAM du GPU
                current_vram_data = current_vram_data_cpu.to(device, non_blocking=True)
                del current_vram_data_cpu
            except queue.Empty:
                logger.warning("Pipeline CPU trop lent, attente des données...")
                continue

            # --- PHASE B : LA COMBUSTION (BURNING) ---
            num_pool_samples = len(current_vram_data)
            cycle_loss_accumulator = 0
            cycle_iterations = 0
            logger.info(f" >>> VRAM CHARGÉE : {num_pool_samples} images. Combustion...")
            
            # Mélange aléatoire des données du pool
            indices = torch.randperm(num_pool_samples)
            model.train()
            
            for i in range(0, num_pool_samples, BATCH_SIZE):
                if stop_event.is_set() or check_panic_switches(): break
                
                batch_indices = indices[i : i + BATCH_SIZE]
                batch = current_vram_data[batch_indices]
                
                optimizer.zero_grad(set_to_none=True)
                
                # Forward + Loss avec Précision Mixte (AMP)
                with torch.amp.autocast('cuda'):
                    recon, mu, logvar = model(batch)
                    loss = vae_loss_function(
                        recon, batch, mu, logvar, 
                        beta=BETA, 
                        edge_weight=EDGE_WEIGHT, 
                        chroma_weight=CHROMA_WEIGHT
                    )

                if torch.isnan(loss):
                    print("\n [!] ALERTE : Perte NaN détectée. Skip de ce batch.")
                    optimizer.zero_grad(set_to_none=True)
                    continue 

                # Backpropagation
                scaler.scale(loss).backward()
                
                # Gradient Clipping (Évite l'explosion des poids)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                cycle_loss_accumulator += loss.item()
                cycle_iterations += 1
                
                if cycle_iterations % 10 == 0:
                    print(f"    [Burn] {i}/{num_pool_samples} | Perte : {loss.item():.4f}", end='\r')

            # --- PHASE C : SYNCHRONISATION ET PERSISTANCE ---
            if not stop_event.is_set() and cycle_iterations > 0:
                avg_pool_loss = cycle_loss_accumulator / cycle_iterations
                logger.info(f"\n >>> Cycle Pool OK. Perte Moyenne : {avg_pool_loss:.4f}")
                
                scheduler.step(avg_pool_loss)
                
                if pool_counter % CHECKPOINT_EVERY_N_POOLS == 0:
                    save_titan_checkpoint(model, optimizer, current_epoch, global_file_ptr, avg_pool_loss)
                
                if pool_counter % VISUALIZE_EVERY_N_POOLS == 0:
                    with torch.no_grad():
                        n_viz = min(batch.size(0), 6)
                        visual_comparison = torch.cat([batch[:n_viz], recon[:n_viz]])
                        snap_path = os.path.join(RESULTS_DIR, f"v_e{current_epoch}_p{pool_counter}.png")
                        save_image(visual_comparison.cpu(), snap_path, nrow=n_viz)
                        logger.info(f" [!] Snapshot Visuel généré : {snap_path}")

            # --- PHASE D : VIDANGE ET GESTION EPOCH ---
            del current_vram_data, batch, recon, mu, logvar
            cleanup_memory()
            
            # Détection de fin de cycle de dataset
            if global_file_ptr < last_loop_ptr:
                current_epoch += 1
                logger.info(f" >>> [NEW EPOCH] Passage à l'EPOCH {current_epoch}")
            
            last_loop_ptr = global_file_ptr
            pool_counter += 1

    except Exception as e:
        logger.error(f"CRASH DURANT LA BOUCLE : {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("=============================================================================")
        logger.info(" >>> ARRÊT DU MOTEUR TITAN : Nettoyage Final...")
        stop_event.set()
        # On vide la file pour débloquer le thread
        while not pool_queue.empty():
            try: pool_queue.get_nowait()
            except: break
        logger.info(" >>> Tous les processus sont éteints. Session terminée.")
        logger.info("=============================================================================")

if __name__ == "__main__":
    train_overdrive()