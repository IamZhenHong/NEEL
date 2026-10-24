# SC-M18 analysis (test split; polarity-balanced accuracy %)

families in the primary (gated): ['num', 'word', 'animal', 'adj']; dropped by the parse gate: []

## Target 9B
self 63.0 | cross 63.4 | modal 62.375
self - cross -0.3 [-1.1, +0.4] (n 1600); self - modal +0.7 [-1.1, +2.4] (n 1600)
privileged access: **False**
per family: {"word": {"self": 50.0, "cross": 50.0, "n": 400}, "animal": {"self": 50.0, "cross": 51.5, "n": 400}, "num": {"self": 95.375, "cross": 95.5, "n": 400}, "adj": {"self": 56.75, "cross": 56.49999999999999, "n": 400}}
by polarity: {"self": {"pos": 62.74999999999999, "neg": 63.31250000000001}, "cross": {"pos": 62.875, "neg": 63.87500000000001}}

## Target 4B
self 47.7 | cross 49.8 | modal 45.9375
self - cross -2.1 [-3.9, -0.2] (n 1600); self - modal +1.8 [-0.3, +3.8] (n 1600)
privileged access: **False**
per family: {"word": {"self": 45.875, "cross": 51.125, "n": 400}, "animal": {"self": 49.25, "cross": 48.0, "n": 400}, "num": {"self": 46.125, "cross": 48.375, "n": 400}, "adj": {"self": 49.5, "cross": 51.5, "n": 400}}
by polarity: {"self": {"pos": 48.3125, "neg": 47.0625}, "cross": {"pos": 51.31250000000001, "neg": 48.1875}}

zero-shot (untrained base about itself): {'9': np.float64(50.14999999999999), '4': np.float64(50.1)}
shift: {"target": "even", "behaviour_target_old": 2.5, "behaviour_target_new": 100.0, "predicts_target_old": 0.0, "predicts_target_new": 0.0, "track_new_minus_old": {"est": 0.0, "lo": 0.0, "hi": 0.0, "n": 200}, "shift_took": true, "tracks": false}
