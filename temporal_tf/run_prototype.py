import torch
from temporal_tf.config import Config, default_config
from temporal_tf.train import train
from temporal_tf.data import synthetic_digit_bank, generate_clip
from temporal_tf.eval.collapse import representation_stats
from temporal_tf.eval.surprise_eval import localization_auc
from temporal_tf.eval.timescale import cross_layer_correlation


def run(cfg: Config = None, n_steps: int = 300, use_mnist: bool = False) -> dict:
    cfg = cfg or default_config()
    bank = None
    if use_mnist:
        from temporal_tf.data import load_mnist_digit_bank
        bank = load_mnist_digit_bank("./data")
    model = train(cfg, n_steps=n_steps, digit_bank=bank)
    device = next(model.online.parameters()).device

    model.eval()
    # CRITICAL: use sized bank so digits fit on the canvas (same fix as train.py Task 8)
    bank = bank if bank is not None else synthetic_digit_bank(size=cfg.image_size // 2)
    gen = torch.Generator().manual_seed(cfg.seed + 99)
    aucs = [[] for _ in range(cfg.n_layers)]
    all_surprise = []
    with torch.no_grad():
        for _ in range(8):
            clip, bounces = generate_clip(gen, bank, cfg)
            out = model(clip.unsqueeze(0).to(device), rng=torch.Generator().manual_seed(0))
            s = out.surprise[0]                        # (T, n_layers)
            all_surprise.append(s)
            for li in range(1, cfg.n_layers):
                aucs[li].append(localization_auc(s[:, li], bounces))
        last = model(clip.unsqueeze(0).to(device), rng=torch.Generator().manual_seed(0))
        collapse = [representation_stats(torch.stack(last.features[li]))["std"]
                    for li in range(cfg.n_layers)]
    # drop layer 0 (never predicted -> all-NaN surprise)
    corr = cross_layer_correlation(torch.cat(all_surprise, 0)[:, 1:])
    return {
        "auc_per_layer": [float(sum(a) / len(a)) for a in aucs[1:]],
        "collapse_std_per_layer": collapse,
        "cross_layer_surprise_corr": corr.tolist(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(n_steps=300, use_mnist=True), indent=2))
