import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset

from MFEFormer import MFEFormer
from ewre_dataset import (
    EWREDataset,
    build_stratified_folds,
    collate_ewre_batch,
    index_ewre_subjects,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MFE-Former on subject-level EWRE folds.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ewre"))
    parser.add_argument("--identity-mode", choices=("clip", "numeric"), default="clip")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-cont", type=float, default=1.0)
    parser.add_argument("--lambda-recon", type=float, default=1e-3)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--window-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=832)
    parser.add_argument("--normalize-mel", action="store_true")
    parser.add_argument("--no-audio-cache", dest="cache_audio", action="store_false")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(cache_audio=True, amp=True)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataset(records, args, clip_features=None):
    return EWREDataset(
        records,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        window_ms=args.window_ms,
        hop_ms=args.hop_ms,
        max_frames=args.max_frames,
        normalize=args.normalize_mel,
        clip_features=clip_features,
        cache_audio=args.cache_audio,
    )


@torch.inference_mode()
def precompute_clip_features(records, model_name, device, cache_path):
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload.get("model_name") == model_name:
            return payload["features"], payload["feature_dim"]
    try:
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
    except ImportError as error:
        raise RuntimeError("transformers is required for --identity-mode clip.") from error
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    clip_model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device).eval()
    features = {}
    batch_size = 64
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        inputs = tokenizer(
            [record.identity_text for record in batch_records],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        embeddings = clip_model(**inputs).text_embeds.float().cpu()
        for record, embedding in zip(batch_records, embeddings):
            features[record.subject_id] = embedding
    feature_dim = clip_model.config.projection_dim
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_name": model_name, "feature_dim": feature_dim, "features": features},
        cache_path,
    )
    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, feature_dim


def build_model(args, identity_dim, clip_feature_dim, device):
    model = MFEFormer(
        input_dim=args.n_mels,
        identity_dim=identity_dim,
        clip_feature_dim=clip_feature_dim,
        model_dim=args.model_dim,
        projection_dim=args.projection_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_words=72,
    )
    return model.to(device)


def identity_arguments(batch, mode, device):
    if mode == "clip":
        return {
            "clip_text_features": batch["clip_text_features"].to(
                device,
                non_blocking=True,
            )
        }
    return {"identity": batch["identity"].to(device, non_blocking=True)}


def move_audio(batch, device):
    return (
        batch["speech"].to(device, non_blocking=True),
        batch["frame_mask"].to(device, non_blocking=True),
        batch["label"].to(device, non_blocking=True),
    )


def make_scheduler(optimizer, steps_per_epoch, epochs, warmup_epochs):
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = min(total_steps, steps_per_epoch * warmup_epochs)

    def multiplier(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        denominator = max(1, total_steps - warmup_steps)
        progress = min(1.0, float(step - warmup_steps) / denominator)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def train_epoch(model, loader, optimizer, scheduler, scaler, args, device, dry_run=False):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        speech, frame_mask, labels = move_audio(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            logits, aux = model(
                speech,
                frame_mask=frame_mask,
                return_aux=True,
                **identity_arguments(batch, args.identity_mode, device),
            )
            classification_loss = F.cross_entropy(logits, labels)
            loss = (
                classification_loss
                + args.lambda_cont * aux["contrastive_loss"]
                + args.lambda_recon * aux["reconstruction_loss"]
                + args.lambda_reg * aux["regularization_loss"]
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at training step {step}: {float(loss)}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        if dry_run:
            return {
                "total": loss.item(),
                "ce": classification_loss.item(),
                "contrastive": aux["contrastive_loss"].item(),
                "reconstruction": aux["reconstruction_loss"].item(),
                "regularization": aux["regularization_loss"].item(),
            }
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, args, device):
    model.eval()
    labels = []
    predictions = []
    for batch in loader:
        speech, frame_mask, target = move_audio(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            logits = model(
                speech,
                frame_mask=frame_mask,
                **identity_arguments(batch, args.identity_mode, device),
            )
        labels.extend(target.cpu().tolist())
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = index_ewre_subjects(args.data_root)
    folds = build_stratified_folds(records, n_splits=5, seed=args.seed)
    record_by_id = {record.subject_id: record for record in records}
    print(f"Indexed {len(records)} subjects: 70 healthy and 70 depressed.")
    print("Five-fold sizes:", [(len(fold.train_ids), len(fold.test_ids)) for fold in folds])
    clip_features = None
    clip_feature_dim = 512
    if args.identity_mode == "clip":
        clip_features, clip_feature_dim = precompute_clip_features(
            records,
            args.clip_model,
            device,
            args.output_dir / "clip_identity_features.pt",
        )
    base_dataset = make_dataset(records, args, clip_features=clip_features)
    index_by_id = {record.subject_id: index for index, record in enumerate(records)}

    all_results = []
    for fold_index, fold in enumerate(folds[: args.max_folds], start=1):
        train_records = [record_by_id[subject_id] for subject_id in fold.train_ids]
        train_indices = [index_by_id[subject_id] for subject_id in fold.train_ids]
        test_indices = [index_by_id[subject_id] for subject_id in fold.test_ids]
        train_loader = DataLoader(
            Subset(base_dataset, train_indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_ewre_batch,
        )
        test_loader = DataLoader(
            Subset(base_dataset, test_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_ewre_batch,
        )
        model = build_model(
            args,
            len(records[0].numeric_identity),
            clip_feature_dim,
            device,
        )
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = make_scheduler(
            optimizer,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

        if args.dry_run:
            losses = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                args,
                device,
                dry_run=True,
            )
            print(f"Fold {fold_index} dry-run losses: {json.dumps(losses, indent=2)}")
            return

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                args,
                device,
            )
            print(f"Fold {fold_index} epoch {epoch:03d}/{args.epochs}: loss={train_loss:.6f}")
        metrics = evaluate(model, test_loader, args, device)
        metrics["fold"] = fold_index
        all_results.append(metrics)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "args": vars(args), "metrics": metrics},
            args.output_dir / f"fold_{fold_index}.pt",
        )
        print(f"Fold {fold_index} test: {json.dumps(metrics, indent=2)}")

    aggregate = {
        metric: float(np.mean([result[metric] for result in all_results]))
        for metric in ("accuracy", "f1", "recall", "precision")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"folds": all_results, "mean": aggregate}, indent=2),
        encoding="utf-8",
    )
    print("Mean test metrics:", json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
