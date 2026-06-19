import statistics
import torch
from dataclasses import replace
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
        for ci in range(n_clips):
            clips, ev = generate_batch(gen, bank, cfg, B=1)
            clip = clips.to(device)
            K = max(1, cfg.n_mask_draws)
            map_draws = {"mean": [], "max": [], "topk": []}
            err_draws = []
            first_out = None
            for d in range(K):
                out = model(clip, rng=torch.Generator().manual_seed(1000 * ci + d))
                if first_out is None:
                    first_out = out
                m = surprise_maps(out, cfg.n_layers, cfg.surprise_topk)
                for r in map_draws:
                    map_draws[r].append(m[r][0])                                   # (T, n_layers)
                err_draws.append(per_tick_error_features(out, cfg.n_layers, cfg.surprise_topk)[0])  # (T, F)
            for r in red:
                red[r].append(torch.stack(map_draws[r]).mean(0))                   # avg over draws
            err_feats.append(torch.stack(err_draws).mean(0))
            out = first_out
            last = out
            T = clip.shape[1]
            rows = []
            for t in range(T):
                vecs = [out.features[li][t][0].mean(0) for li in range(1, cfg.n_layers)]
                rows.append(torch.cat(vecs))
            feat_feats.append(torch.stack(rows))
            labels["bounce"].append(ev["bounce"][0])
            labels["teleport"].append(ev["teleport"][0])
    return red, err_feats, feat_feats, labels, last

def _probe_set(model, cfg, bank, n_clips):
    red, err_feats, feat_feats, labels, _ = _collect(model, cfg, bank, n_clips)
    err_X, feat_X = torch.cat(err_feats), torch.cat(feat_feats)
    err = {ev: event_probe(err_X, torch.cat(labels[ev])) for ev in ("bounce", "teleport")}
    feat = {ev: event_probe(feat_X, torch.cat(labels[ev])) for ev in ("bounce", "teleport")}
    return err, feat


def run(cfg: Config = None, n_steps: int = 300, use_mnist: bool = False, track_every: int = 0) -> dict:
    cfg = cfg or default_config()
    bank = None
    if use_mnist:
        from temporal_tf.data import load_mnist_digit_bank
        bank = load_mnist_digit_bank("./data")
    eval_bank = bank if bank is not None else synthetic_digit_bank(size=cfg.image_size // 2)

    loss_curve = []
    probe_clips = max(2, cfg.n_eval_clips // 5)
    def on_eval(model, step, parts):
        err, feat = _probe_set(model, cfg, eval_bank, probe_clips)
        loss_curve.append({"step": step, "pred": parts["pred"], "var": parts["var"],
                           "cov": parts["cov"], "total": parts["total"],
                           "error_probe_auc": err, "feature_probe_auc": feat})

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

    out = {
        "loss_curve": loss_curve,
        "raw_pooled_auc": raw,
        "error_probe_auc": error_probe,
        "feature_probe_auc": feature_probe,
        "collapse_std_per_layer": collapse,
    }
    if track_every > 0 and loss_curve:
        best = max(loss_curve, key=lambda e: e["error_probe_auc"]["teleport"])
        out["best_error_probe_teleport"] = {"step": best["step"], "auc": best["error_probe_auc"]["teleport"]}
    return out


def _ms(xs):
    return {"mean": statistics.fmean(xs),
            "std": (statistics.pstdev(xs) if len(xs) > 1 else 0.0),
            "values": list(xs)}


def run_seeds(cfg: Config, n_steps: int, seeds, use_mnist: bool = False, track_every: int = 0) -> dict:
    finals = {"teleport": [], "bounce": []}
    bests = []
    for s in seeds:
        r = run(cfg=replace(cfg, seed=s), n_steps=n_steps, use_mnist=use_mnist, track_every=track_every)
        finals["teleport"].append(r["error_probe_auc"]["teleport"])
        finals["bounce"].append(r["error_probe_auc"]["bounce"])
        if "best_error_probe_teleport" in r:
            bests.append(r["best_error_probe_teleport"]["auc"])
    agg = {"final_error_probe_teleport": _ms(finals["teleport"]),
           "final_error_probe_bounce": _ms(finals["bounce"])}
    if bests:
        agg["best_error_probe_teleport"] = _ms(bests)
    return agg


if __name__ == "__main__":
    import json
    print(json.dumps(run(n_steps=10000, use_mnist=True, track_every=200), indent=2))
