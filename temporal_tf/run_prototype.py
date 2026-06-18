import torch
from temporal_tf.config import Config, default_config
from temporal_tf.train import train
from temporal_tf.data import synthetic_digit_bank, generate_batch
from temporal_tf.eval.collapse import representation_stats
from temporal_tf.eval.surprise_eval import localization_auc
from temporal_tf.eval.timescale import cross_layer_correlation


def eval_surprise(model, cfg, bank, n_clips, n_mask_draws):
    model.eval()
    device = next(model.online.parameters()).device
    gen = torch.Generator().manual_seed(cfg.seed + 99)
    n_pred = cfg.n_layers - 1
    auc_b = [[] for _ in range(n_pred)]
    auc_t = [[] for _ in range(n_pred)]
    all_surprise = []
    with torch.no_grad():
        for _ in range(n_clips):
            clips, ev = generate_batch(gen, bank, cfg, B=1)
            clip = clips.to(device)
            draws = []
            for d in range(max(1, n_mask_draws)):
                out = model(clip, rng=torch.Generator().manual_seed(d))
                draws.append(out.surprise[0])           # (T, n_layers)
            s = torch.stack(draws).mean(0)              # (T, n_layers), mask-averaged
            all_surprise.append(s)
            for li in range(1, cfg.n_layers):
                auc_b[li - 1].append(localization_auc(s[:, li], ev["bounce"][0]))
                auc_t[li - 1].append(localization_auc(s[:, li], ev["teleport"][0]))
    corr = cross_layer_correlation(torch.cat(all_surprise, 0)[:, 1:])
    return {
        "auc_bounce_per_layer": [float(sum(a) / len(a)) for a in auc_b],
        "auc_teleport_per_layer": [float(sum(a) / len(a)) for a in auc_t],
        "cross_layer_surprise_corr": corr.tolist(),
    }


def run(cfg: Config = None, n_steps: int = 300, use_mnist: bool = False, track_every: int = 0) -> dict:
    cfg = cfg or default_config()
    bank = None
    if use_mnist:
        from temporal_tf.data import load_mnist_digit_bank
        bank = load_mnist_digit_bank("./data")
    eval_bank = bank if bank is not None else synthetic_digit_bank(size=cfg.image_size // 2)

    curve = []

    def on_eval(model, step):
        m = eval_surprise(model, cfg, eval_bank, n_clips=max(2, cfg.n_eval_clips // 4), n_mask_draws=1)
        curve.append({"step": step,
                      "auc_bounce_per_layer": m["auc_bounce_per_layer"],
                      "auc_teleport_per_layer": m["auc_teleport_per_layer"]})

    model = train(cfg, n_steps=n_steps, digit_bank=bank,
                  on_eval=(on_eval if track_every > 0 else None), eval_every=track_every)

    full = eval_surprise(model, cfg, eval_bank, n_clips=cfg.n_eval_clips, n_mask_draws=cfg.n_mask_draws)
    device = next(model.online.parameters()).device
    with torch.no_grad():
        clips, _ = generate_batch(torch.Generator().manual_seed(cfg.seed + 7), eval_bank, cfg, B=1)
        last = model(clips.to(device), rng=torch.Generator().manual_seed(0))
        collapse = [representation_stats(torch.stack(last.features[li]))["std"] for li in range(cfg.n_layers)]
    out = {**full, "collapse_std_per_layer": collapse}
    if track_every > 0:
        out["training_curve"] = curve
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run(n_steps=500, use_mnist=True, track_every=50), indent=2))
