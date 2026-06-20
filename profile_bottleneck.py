"""
QUICK PROFILER — run this directly:

    python -u profile_bottleneck.py

It imports cl_train.py from the SAME FOLDER and uses its real
create_loaders / build_model / ModelEMA / combined_loss / DEVICE /
USE_AMP — no manual setup needed.

This isolates four candidate bottlenecks so we know which fix (if any)
actually matters:
  [A] Pure dataloader iteration speed (no GPU work at all)
  [B] GPU forward+backward pass speed (no EMA, no scheduler)
  [C] EMA.update() cost in isolation
  [D] scheduler.step() cost in isolation

At the end it prints an estimated total epoch time built from these
parts, so you can see which one dominates.

IMPORTANT: importing cl_train.py runs every top-level statement in
that file EXCEPT the `if __name__ == "__main__":` block — so it will
NOT start full training, but it WILL print the "Device:/ GPU:/ VRAM:"
lines from that file's module-level code. That's expected and fine.

Place this file in the SAME FOLDER as cl_train.py before running.
"""

import time
import torch

import cl_train as M   # <-- imports your real training script directly


def profile_dataloader_only(train_loader, n_batches=20):
    print(f"\n[A] Pure dataloader speed (no GPU, {n_batches} batches)...")
    n_batches = min(n_batches, len(train_loader) - 1)
    it = iter(train_loader)
    next(it)  # warm up — exclude worker startup cost
    t0 = time.time()
    for _ in range(n_batches):
        images, masks = next(it)
    t1 = time.time()
    per_batch = (t1 - t0) / n_batches
    est_epoch = per_batch * len(train_loader)
    print(f"    {per_batch*1000:.1f} ms/batch  ->  ~{est_epoch:.2f} s/epoch (dataloader-only estimate)")
    return per_batch, est_epoch


def profile_gpu_step_only(model, train_loader, combined_loss, device, use_amp, n_batches=20):
    print(f"\n[B] GPU forward+backward only, no EMA/scheduler ({n_batches} batches)...")
    n_batches = min(n_batches, len(train_loader) - 1)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    it = iter(train_loader)

    # warmup batch — CUDA kernel compilation / cudnn autotune shouldn't count
    images, masks = next(it)
    images, masks = images.to(device), masks.to(device)
    with torch.cuda.amp.autocast(enabled=use_amp):
        logits = model(images)
        loss = combined_loss(logits, masks)
    scaler.scale(loss).backward()
    optimizer.zero_grad()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_batches):
        images, masks = next(it)
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = combined_loss(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    per_batch = (t1 - t0) / n_batches
    est_epoch = per_batch * len(train_loader)
    print(f"    {per_batch*1000:.1f} ms/batch  ->  ~{est_epoch:.2f} s/epoch (GPU-only estimate)")
    return per_batch, est_epoch


def profile_ema_update(ema, model, n_calls=50):
    print(f"\n[C] EMA.update() cost in isolation ({n_calls} calls)...")
    t0 = time.time()
    for _ in range(n_calls):
        ema.update(model)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    per_call = (t1 - t0) / n_calls
    print(f"    {per_call*1000:.2f} ms/call")
    return per_call


def profile_scheduler_step(scheduler, n_calls=200):
    print(f"\n[D] scheduler.step() cost in isolation ({n_calls} calls)...")
    t0 = time.time()
    for i in range(n_calls):
        scheduler.step(i * 0.01)
    t1 = time.time()
    per_call = (t1 - t0) / n_calls
    print(f"    {per_call*1000:.3f} ms/call")
    return per_call


def main():
    print("=" * 70)
    print("Building loaders and model from cl_train.py ...")
    print("=" * 70)

    train_loader, val_loader = M.create_loaders()
    model = M.build_model()
    ema = M.ModelEMA(model, decay=getattr(M, "EMA_DECAY", 0.999))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=getattr(M, "RESTART_T0", 10),
        T_mult=getattr(M, "RESTART_T_MULT", 2),
        eta_min=getattr(M, "MIN_LR", 1e-6),
    )

    device  = M.DEVICE
    use_amp = M.USE_AMP
    grad_accum = getattr(M, "GRAD_ACCUM", 1)
    batches_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = max(1, batches_per_epoch // grad_accum)

    print(f"\nbatches/epoch       : {batches_per_epoch}")
    print(f"GRAD_ACCUM          : {grad_accum}")
    print(f"optimizer steps/epoch (≈ EMA.update calls/epoch): {optimizer_steps_per_epoch}")

    print("\n" + "=" * 70)
    print("RUNNING PROFILES")
    print("=" * 70)

    dl_per_batch, dl_epoch_est   = profile_dataloader_only(train_loader)
    gpu_per_batch, gpu_epoch_est = profile_gpu_step_only(model, train_loader, M.combined_loss, device, use_amp)
    ema_per_call                 = profile_ema_update(ema, model)
    sched_per_call                = profile_scheduler_step(scheduler)

    # ── put it all together ──────────────────────────────────────
    ema_total_per_epoch    = ema_per_call * optimizer_steps_per_epoch
    sched_total_per_epoch  = sched_per_call * batches_per_epoch
    combined_est_epoch     = gpu_epoch_est + ema_total_per_epoch + sched_total_per_epoch

    print("\n" + "=" * 70)
    print("SUMMARY — estimated contribution to one epoch")
    print("=" * 70)
    print(f"  [A] Dataloader alone        : ~{dl_epoch_est:6.2f} s/epoch")
    print(f"  [B] GPU fwd+bwd alone       : ~{gpu_epoch_est:6.2f} s/epoch")
    print(f"  [C] EMA updates (all steps) : ~{ema_total_per_epoch:6.2f} s/epoch  "
          f"({ema_per_call*1000:.2f} ms x {optimizer_steps_per_epoch} steps)")
    print(f"  [D] Scheduler steps (all)   : ~{sched_total_per_epoch:6.2f} s/epoch  "
          f"({sched_per_call*1000:.3f} ms x {batches_per_epoch} batches)")
    print(f"  ----")
    print(f"  Combined estimate (B+C+D)  : ~{combined_est_epoch:6.2f} s/epoch")
    print(f"\n  Compare this combined estimate to what you're ACTUALLY seeing")
    print(f"  per epoch right now. Whichever of [A]/[B]/[C]/[D] is largest")
    print(f"  is your real bottleneck — that's the one worth optimizing.")
    print("=" * 70)


if __name__ == "__main__":
    main()