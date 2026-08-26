"""Clasificacion de imagenes cost-sensitive (estilo Desafio 2).

Pensado para correr en Colab con GPU. Requiere torch + timm:
    pip install timm

Palancas ordenadas por rentabilidad:
  1. transfer learning con un backbone moderno (timm)                +++++
  2. decision bayesiana sobre la matriz de costos jerarquica         ++++
  3. TTA (flip + multiescala)                                        +++
  4. deteccion de duplicados para que el CV no mienta                +++
  5. limpieza de etiquetas ruidosas (confident learning)             +++
  6. ensamble de 2-3 backbones distintos                             +++
  7. cabeza jerarquica reino->especie                                ++

NOTA SOBRE EL DESAFIO 2: la metrica castiga con 5 confundir reino y con 2
confundir dentro del reino. Por eso conviene un modelo que PRIMERO acierte el
reino. `hierarchical_proba` y `bayes_decision` explotan exactamente eso.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .costs import bayes_decision, grouped_cost_matrix, mean_cost  # noqa: F401

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ------------------------------------------------------- higiene de datos
def list_images(root) -> list[Path]:
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT]


def dhash(path, size: int = 8) -> str:
    """Hash perceptual: imagenes casi iguales dan el mismo hash."""
    from PIL import Image
    im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return hashlib.md5(np.packbits(bits).tobytes()).hexdigest()[:16]


def duplicate_groups(paths) -> dict[str, list[Path]]:
    """Agrupa imagenes duplicadas/casi duplicadas.

    POR QUE IMPORTA: si la misma foto (o dos frames casi identicos de la misma
    camara trampa) cae en train y en validacion, tu CV da altisimo y el
    leaderboard te baja. Usa el hash como `groups` en GroupKFold.
    """
    g: dict[str, list[Path]] = {}
    for p in paths:
        try:
            g.setdefault(dhash(p), []).append(p)
        except Exception:
            g.setdefault("ERROR_" + p.name, []).append(p)
    return g


def build_index(train_root):
    """Devuelve (paths, y, classes, groups) leyendo train/<clase>/*.jpg."""
    root = Path(train_root)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    paths, y = [], []
    for i, c in enumerate(classes):
        for p in list_images(root / c):
            paths.append(p)
            y.append(i)
    groups = np.array([dhash(p) for p in paths])
    n_dup = len(groups) - len(set(groups))
    print("%d imagenes, %d clases, %d duplicadas (usalas como grupos en el CV)"
          % (len(paths), len(classes), n_dup))
    print("por clase:", {c: int((np.array(y) == i).sum()) for i, c in enumerate(classes)})
    return paths, np.array(y), classes, groups


def corrupt_images(paths) -> list[Path]:
    """Archivos que PIL no puede abrir: sacalos antes de entrenar."""
    from PIL import Image
    bad = []
    for p in paths:
        try:
            Image.open(p).convert("RGB").load()
        except Exception:
            bad.append(p)
    return bad


# ---------------------------------------------------------------- dataset
def make_loaders(paths, y, idx_tr, idx_va, img_size=384, batch=32,
                 balanced: bool = True, workers=2):
    """DataLoaders con augmentation fuerte en train y muestreo balanceado."""
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import transforms as T
    from PIL import Image

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tf_tr = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.55, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.08)], p=0.7),
        T.RandomApply([T.GaussianBlur(5)], p=0.15),
        T.RandomRotation(12),
        T.ToTensor(), T.Normalize(mean, std),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    tf_va = T.Compose([T.Resize(int(img_size * 1.14)), T.CenterCrop(img_size),
                       T.ToTensor(), T.Normalize(mean, std)])

    class DS(Dataset):
        def __init__(self, idx, tf):
            self.idx, self.tf = list(idx), tf

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            im = Image.open(paths[j]).convert("RGB")
            return self.tf(im), int(y[j])

    sampler = None
    if balanced:
        cnt = np.bincount(np.asarray(y)[idx_tr])
        wt = (1.0 / cnt)[np.asarray(y)[idx_tr]]
        sampler = WeightedRandomSampler(torch.as_tensor(wt, dtype=torch.double),
                                        len(idx_tr), replacement=True)
    dl_tr = DataLoader(DS(idx_tr, tf_tr), batch_size=batch, sampler=sampler,
                       shuffle=sampler is None, num_workers=workers,
                       pin_memory=True, drop_last=True)
    dl_va = DataLoader(DS(idx_va, tf_va), batch_size=batch * 2, shuffle=False,
                       num_workers=workers, pin_memory=True)
    return dl_tr, dl_va


# -------------------------------------------------------------- training
def train_fold(dl_tr, dl_va, n_classes, model_name="convnext_tiny.fb_in22k",
               epochs=8, lr=3e-4, head_epochs=1, label_smoothing=0.1,
               device=None, verbose=True):
    """Fine-tuning en 2 etapas: cabeza sola y despues todo con cosine decay."""
    import timm
    import torch
    import torch.nn as nn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=True, num_classes=n_classes).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    def run_epoch(dl, opt=None, sched=None):
        train = opt is not None
        model.train(train)
        tot, n, probs, ys = 0.0, 0, [], []
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.set_grad_enabled(train), torch.amp.autocast(device, enabled=device == "cuda"):
                out = model(xb)
                loss = crit(out, yb)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                if sched:
                    sched.step()
            else:
                probs.append(out.float().softmax(1).cpu().numpy())
                ys.append(yb.cpu().numpy())
            tot += loss.item() * len(yb)
            n += len(yb)
        return tot / n, (np.concatenate(probs) if probs else None,
                         np.concatenate(ys) if ys else None)

    # etapa 1: solo la cabeza
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for e in range(head_epochs):
        tr_loss, _ = run_epoch(dl_tr, opt)
        if verbose:
            print("  [cabeza] epoch %d loss %.4f" % (e + 1, tr_loss))

    # etapa 2: todo
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, pct_start=0.25,
                                                steps_per_epoch=len(dl_tr), epochs=epochs)
    best_state, best = None, np.inf
    for e in range(epochs):
        tr_loss, _ = run_epoch(dl_tr, opt, sched)
        va_loss, (pv, yv) = run_epoch(dl_va)
        acc = float((pv.argmax(1) == yv).mean())
        if verbose:
            print("  epoch %d  train %.4f  val %.4f  acc %.4f" % (e + 1, tr_loss, va_loss, acc))
        if va_loss < best:
            best = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def predict_tta(model, paths, img_size=384, batch=32, scales=(1.0, 1.15),
                hflip=True, device=None, workers=2) -> np.ndarray:
    """Probabilidades promediando escalas y espejado. Suele dar +1-2 puntos."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T
    from PIL import Image

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    model.eval().to(device)
    acc = None
    n_views = 0
    for sc in scales:
        s = int(img_size * sc)
        tf = T.Compose([T.Resize(int(s * 1.14)), T.CenterCrop(s),
                        T.ToTensor(), T.Normalize(mean, std)])

        class DS(Dataset):
            def __len__(self):
                return len(paths)

            def __getitem__(self, i):
                return tf(Image.open(paths[i]).convert("RGB"))

        dl = DataLoader(DS(), batch_size=batch, shuffle=False, num_workers=workers)
        outs = []
        with torch.no_grad(), torch.amp.autocast(device, enabled=device == "cuda"):
            for xb in dl:
                xb = xb.to(device)
                o = model(xb).float().softmax(1)
                if hflip:
                    o = o + model(torch.flip(xb, dims=[3])).float().softmax(1)
                outs.append(o.cpu().numpy())
        p = np.concatenate(outs)
        p = p / p.sum(1, keepdims=True)
        acc = p if acc is None else acc + p
        n_views += 1
    return acc / n_views


# ------------------------------------------------------------ jerarquico
def hierarchical_proba(p_species: np.ndarray, groups) -> np.ndarray:
    """Reescala las probabilidades por grupo (reino) para reforzar la jerarquia.

    P(especie) = P(reino) * P(especie | reino). Matematicamente es lo mismo que
    p_species si el modelo esta calibrado, pero permite mezclar la salida de un
    clasificador binario de reino (mas facil, mas preciso) con la de especie:

        p = hierarchical_proba(p_especie, reinos)      # normaliza por grupo
        p_final = p * p_reino[:, reinos]               # inyecta el binario
    """
    g = np.asarray(groups)
    out = np.zeros_like(p_species, dtype=float)
    for gid in np.unique(g):
        m = g == gid
        s = p_species[:, m].sum(1, keepdims=True) + 1e-12
        out[:, m] = p_species[:, m] / s
    return out


def kingdom_proba(p_species: np.ndarray, groups) -> np.ndarray:
    """P(grupo) agregando las probabilidades de especie. Shape (n, n_grupos)."""
    g = np.asarray(groups)
    return np.stack([p_species[:, g == gid].sum(1) for gid in np.unique(g)], axis=1)


# --------------------------------------------------- etiquetas ruidosas
def suspect_labels(oof_proba: np.ndarray, y, C: np.ndarray, top: int = 50):
    """Imagenes que el modelo cree mal etiquetadas, ordenadas por costo.

    Las consignas dicen "un alto porcentaje fue validado por expertos": el resto
    no. Revisa a ojo las 30-50 peores; corregir 20 etiquetas suele valer mas
    que otra epoca de entrenamiento.
    """
    y = np.asarray(y)
    p_true = oof_proba[np.arange(len(y)), y]
    pred = bayes_decision(oof_proba, C)
    cost = C[y, pred]
    score = (1 - p_true) * (1 + cost)
    order = np.argsort(-score)[:top]
    return order, {"idx": order, "p_asignada": p_true[order],
                   "pred": pred[order], "costo": cost[order]}
