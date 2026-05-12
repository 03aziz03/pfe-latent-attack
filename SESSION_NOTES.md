# PFE — Latent Adversarial Attack on YOLOv8 (UA-DETRAC)
## Session Notes & Project Summary

**Date :** 2026-05-10  
**Student :** Mohamed Aziz Brahmi  
**Projet :** Attaque adversariale dans l'espace latent d'un VAE Stable Diffusion contre YOLOv8 sur UA-DETRAC

---

## 1. Vue d'ensemble du projet

L'objectif du PFE est de concevoir une attaque adversariale **imperceptible** contre un détecteur d'objets (YOLOv8) en perturbant les **latents** d'un VAE Stable Diffusion (stabilityai/sd-vae-ft-mse), plutôt que de travailler directement dans l'espace pixel.

### Architecture générale

```
Image x (pixel) 
    → Encoder VAE (gelé) → z (latent 4×H/8×W/8)
    → PGD Attack dans espace latent (boule L∞ rayon eps_z)
    → z_adv
    → Decoder VAE → x_adv (pixel)
    → YOLOv8 → détections réduites ✓
```

### Composants clés

| Composant | Détail |
|-----------|--------|
| Détecteur | YOLOv8n, fine-tuné sur UA-DETRAC (`runs/yolov8n_detrac/best.pt`) |
| VAE | `stabilityai/sd-vae-ft-mse`, fine-tuné sur DETRAC (decoder only) |
| Attaque | PGD dans espace latent, masque bbox, vanishing loss + régularisation LPIPS |
| Dataset eval | `data/images_50` (50 frames), `data/images_100` (Phase 4) |
| Dataset fine-tune | `data/finetune_seqs` (3 séquences × 60 frames) |

---

## 2. Phases du projet

### Phase 1 — Baseline (terminée ✅)
- Attaque latente avec régularisation L2 masquée (`lambda_p=0.05`, `masked_l2`)
- DFR obtenu : ~0.20 sur 50 frames
- 13 figures générées (`f01` à `f13`)

### Phase 2 — LPIPS + VAE fine-tuné (en cours 🔄)

#### 2a. Fine-tuning du VAE ✅
- Decoder-only (encoder gelé → cache latent reste valide)
- Loss : `0.7 × MSE + 0.3 × LPIPS`
- Dataset : 3 séquences DETRAC séparées du dev set (180 frames)
- Split : 85% train / 15% val par séquence (stratified)
- Résultats : val loss `0.009099 → 0.007487` (**−17.7%**), meilleur checkpoint epoch 19
- Techniques mémoire (Colab L4, 22 GB VRAM) :
  - `PYTORCH_ALLOC_CONF=expandable_segments:True`
  - Gradient checkpointing sur le decoder
  - bfloat16 AMP
  - Résolution 512×512 (VAE fully convolutional → transfert à 640×640)
  - `batch_size=1`

#### 2b. Iso-budget sweep ⏳ (en cours)
- Sweep sur `eps_z ∈ {0.25, 0.50, 1.00}` + baselines PGD pixel-space
- 30 frames de `data/images_50`
- **Bug rencontré et corrigé :** DFR effondré (~0.010) à cause de `lambda_p` mal calibré

### Phase 3 — Temporal consistency loss (à venir)
### Phase 4 — Headline run 100 frames × 3 séquences (à venir)

---

## 3. Bug critique résolu : lambda_p miscalibration

### Symptôme
Le premier run du iso-budget sweep donnait :
- DFR ~0.010 (vs ~0.20 attendu)
- DFR **décroissant** avec eps_z (direction inverse — signe d'une attaque cassée)

### Cause racine
| Loss | Échelle des valeurs |
|------|-------------------|
| `masked_L2` | ~0.001 – 0.009 |
| `LPIPS` | ~0.09 – 0.15 |
| **Ratio** | **LPIPS ≈ 17× plus grand** |

Avec `lambda_p=0.05` (calibré pour L2), le terme perceptuel dominait le vanishing loss : l'optimiseur minimisait la distorsion visuelle au lieu de tromper le détecteur.

### Fix appliqué
```yaml
# configs/phase2.yaml
attack:
  lambda_p: 0.001   # anciennement 0.05 — recalibré pour LPIPS
```

Le fix a été committé et pushé sur GitHub. Le sweep tourne actuellement avec ce paramètre.

### Ce qu'on attend après le fix
| eps_z | DFR attendu |
|-------|------------|
| 0.25  | ~0.08–0.12 |
| 0.50  | ~0.15–0.20 |
| 1.00  | ~0.25–0.35 |
Relation monotone : DFR augmente avec eps_z ✓

---

## 4. Datasets — Clarification

```
data/
├── finetune_seqs/          # Fine-tuning VAE (séparé du dev set)
│   ├── sequence_A/         # ~60 frames
│   ├── sequence_B/         # ~60 frames
│   └── sequence_C/         # ~60 frames
├── images_50/              # Dev set évaluation (iso-budget sweep, debug)
│   └── *.jpg               # 50 frames
└── images_100/             # Test set final (Phase 4 uniquement)
    └── *.jpg               # 100 frames
```

**Le iso-budget sweep** (`run_iso_budget.py`) utilise **`data/images_50`**, sous-ensemble de 30 frames (`--n_frames 30`).  
Il **n'utilise pas** les images de fine-tuning.

---

## 5. Métriques utilisées

| Métrique | Description |
|----------|-------------|
| `DFR_strict_proportional` | Taux de réduction des détections (principal) |
| `DFR_binary` | Frames où toutes les détections disparaissent |
| `ASR_strict` | Attack Success Rate (au moins une bbox supprimée) |
| `mAP_drop@0.5` | Chute de mAP après attaque |
| `masked_LPIPS` | Distorsion perceptuelle dans le masque bbox |

---

## 6. Fichiers importants

| Fichier | Rôle |
|---------|------|
| `configs/phase2.yaml` | Config principale Phase 2 (lambda_p=0.001 !) |
| `src/vae.py` | Wrapper VAE (encode/decode/encode_with_grad) |
| `src/losses.py` | MaskedLPIPS + masked_l2 |
| `src/attack.py` | Boucle PGD latente |
| `scripts/finetune_vae.py` | Fine-tuning decoder-only |
| `scripts/run_iso_budget.py` | Sweep eps_z + baselines |
| `scripts/generate_pareto.py` | Courbe Pareto DFR vs LPIPS (f14) |
| `notebooks/phase2_colab.ipynb` | Notebook Colab (GitHub clone workflow) |
| `runs/vae_detrac/vae_ft.pt` | Checkpoint VAE fine-tuné |
| `runs/vae_detrac/ft_meta.json` | Méta-données du fine-tuning |

---

## 7. Workflow Colab (GitHub clone)

```python
# Cell 1 — Clone
!git clone https://github.com/03aziz03/pfe-latent-attack.git
%cd pfe-latent-attack

# Cell 2 — Monter Drive + copier données lourdes
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/pfe_data/best.pt runs/yolov8n_detrac/
!rsync -a /content/drive/MyDrive/pfe_data/images_50/ data/images_50/
!rsync -a /content/drive/MyDrive/pfe_data/finetune_seqs/ data/finetune_seqs/

# Cell 3 — Installer dépendances
!pip install lpips -q

# Cell 5 — Fine-tune VAE
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
!python scripts/finetune_vae.py \
    --data data/finetune_seqs \
    --output runs/vae_detrac \
    --epochs 20 --lr 1e-5 --batch_size 1

# Cell 7 — Iso-budget sweep
!python scripts/run_iso_budget.py \
    --config configs/phase2.yaml \
    --data data/images_50 \
    --n_frames 30 \
    --output results/iso_budget

# Cell 13 — Copier résultats vers Drive
import shutil
shutil.copytree("results/", "/content/drive/MyDrive/pfe_results/", dirs_exist_ok=True)
```

---

## 8. Prochaines étapes

- [ ] Analyser les résultats du iso-budget sweep (DFR monotone ?)
- [ ] Générer f14 (courbe Pareto) avec `generate_pareto.py`
- [ ] Copier résultats vers Drive
- [ ] Mettre à jour `PROJECT_HANDOFF.md` avec résultats Phase 2
- [ ] Phase 3 : temporal consistency loss
- [ ] Phase 4 : headline run (100 frames × 3 séquences)

---

## 9. Erreurs OOM rencontrées et solutions

### OOM #1 — batch_size=4, encoder+decoder
```
torch.OutOfMemoryError: Tried to allocate 200.00 MiB
```
**Fix :** Decoder-only + bfloat16 AMP + batch_size=2

### OOM #2 — batch_size=2, decoder-only
```
torch.OutOfMemoryError: Tried to allocate 400.00 MiB
```
**Fix :** `os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'` + gradient checkpointing + 512px

---

*Notes générées automatiquement — session Cowork 2026-05-10*
