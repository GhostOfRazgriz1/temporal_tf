import torch
from temporal_tf.config import Config, default_config
from temporal_tf.train import train
from temporal_tf.data import synthetic_digit_bank, generate_batch
from temporal_tf.eval.collapse import representation_stats
from temporal_tf.eval.event_probe import surprise_maps, pooled_auc, per_tick_error_features, event_probe

def _collect(model, cfg, bank, n_clips):
    device = next(model.online.parameters()).device
    gen = torch.Generator().manual_seed(cfg.seed + 99)
    red = {"mean": [], "max": [], "topk": []}
    err_feats, feat_feats = [], []
    labels = {"bounce": [], "teleport": []}
    last = None
    with torch.no_grad():
        for _ in range(n_clips):
            clips, ev = generate_batch(gen, bank, cfg, B=1)
            out = model(clips.to(device), rng=torch.Generator().manual_seed(0))
            last = out
            maps = surprise_maps(out, cfg.n_layers, cfg.surprise_topk)
            for r in red:
                red[r].append(maps[r][0])                              # (T, n_layers)
            err_feats.append(per_tick_error_features(out, cfg.n_layers, cfg.surprise_topk)[0])  # (T, F)
            T = clips.shape[1]
            rows = []
            for t in range(T):
                vecs = [out.features[li][t][0].mean(0) for li in range(1, cfg.n_layers)]  # each (d,)
                rows.append(torch.cat(vecs))
            feat_feats.append(torch.stack(rows))                        # (T, (n_layers-1)*d)
            labels["bounce"].append(ev["bounce"][0])
            labels["teleport"].append(ev["teleport"][0])
    return red, err_feats, feat_feats, labels, last

def run(cfg: Config = None, n_steps: int = 300, use_mnist: bool = False, track_every: int = 0) -> dict:
    cfg = cfg or default_config()
    bank = None
    if use_mnist:
        from temporal_tf.data import load_mnist_digit_bank
        bank = load_mnist_digit_bank("./data")
    eval_bank = bank if bank is not None else synthetic_digit_bank(size=cfg.image_size // 2)

    loss_curve = []
    def on_eval(model, step, parts):
        loss_curve.append({"step": step, "pred": parts["pred"], "var": parts["var"],
                           "cov": parts["cov"], "total": parts["total"]})

    model = train(cfg, n_steps=n_steps, digit_bank=bank,
                  on_eval=(on_eval if track_every > 0 else None), eval_every=track_every)

    red, err_feats, feat_feats, labels, last = _collect(model, cfg, eval_bank, cfg.n_eval_clips)
    raw = {}
    for ev in ("bounce", "teleport"):
        y = torch.cat(labels[ev])                                       # (n_clips*T,)
        raw[ev] = {}
        for r in ("mean", "max", "topk"):
            scores = torch.cat(red[r])                                  # (n_clips*T, n_layers)
            raw[ev][r] = [pooled_auc(scores[:, li], y) for li in range(1, cfg.n_layers)]
    err_X = torch.cat(err_feats)                                        # (n_clips*T, n_pred*3)
    feat_X = torch.cat(feat_feats)
    error_probe = {ev: event_probe(err_X, torch.cat(labels[ev])) for ev in ("bounce", "teleport")}
    feature_probe = {ev: event_probe(feat_X, torch.cat(labels[ev])) for ev in ("bounce", "teleport")}
    collapse = [representation_stats(torch.stack(last.features[li]))["std"] for li in range(cfg.n_layers)]

    return {
        "loss_curve": loss_curve,
        "raw_pooled_auc": raw,
        "error_probe_auc": error_probe,
        "feature_probe_auc": feature_probe,
        "collapse_std_per_layer": collapse,
    }

if __name__ == "__main__":
    import json
    print(json.dumps(run(n_steps=10000, use_mnist=True, track_every=200), indent=2))
