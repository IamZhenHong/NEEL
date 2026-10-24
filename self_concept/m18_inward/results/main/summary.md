# SC-M18 analysis (test split; polarity-balanced accuracy %)

## Target 9B
self 75.0 | cross 73.2 | modal 55.50000000000001
self - cross +1.8 [+0.1, +3.5] (n 1200); self - modal +19.5 [+17.0, +22.1] (n 1200)
privileged access: **False**
per family: {"risk": {"self": 100.0, "cross": 100.0, "n": 400}, "num": {"self": 69.0, "cross": 71.0, "n": 400}, "word": {"self": 56.125, "cross": 48.625, "n": 400}}
by polarity: {"self": {"pos": 73.58333333333333, "neg": 76.5}, "cross": {"pos": 71.58333333333333, "neg": 74.83333333333333}}

## Target 4B
self 51.4 | cross 51.0 | modal 50.7
self - cross +0.4 [-1.3, +2.2] (n 1000); self - modal +0.8 [-1.8, +3.2] (n 1000)
privileged access: **False**
per family: {"risk": {"self": 70.0, "cross": 74.0, "n": 200}, "num": {"self": 47.75, "cross": 47.75, "n": 400}, "word": {"self": 45.875, "cross": 42.75, "n": 400}}
by polarity: {"self": {"pos": 49.2, "neg": 53.7}, "cross": {"pos": 50.9, "neg": 51.1}}

zero-shot (untrained base about itself): {'9': np.float64(49.583333333333336), '4': np.float64(51.083333333333336)}
shift: {"odd_rate_old": 43.0, "odd_rate_new": 100.0, "n_changed": 114, "track_new_minus_old": {"est": 100.0, "lo": 100.0, "hi": 100.0, "n": 114}, "shift_took": 1.0, "tracks": true}
