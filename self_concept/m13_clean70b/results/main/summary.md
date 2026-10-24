# SC-M13 summary (clean bf16 Llama-3.1-70B)

**headline:** kind of behaviour (the narrow habit stays out of the self-report at clean 70B; Betley's trait does not)

## LF
- d0 habit: act +40.4 [+24.9, +55.7] (took True); ours +0.9 [+0.1, +1.9] → none; bjust +1.5 [-1.3, +4.2] → none; predict +32.4 [+18.0, +47.5]
- d2 habit: act +63.9 [+40.6, +84.8] (took True); ours +2.9 [+0.6, +6.2] → none; bjust +7.1 [+0.3, +13.8] → none; predict +59.0 [+37.9, +78.9]
- risk: lottery +94.8 [+85.4, +100.0] (took True); brisk +82.2 [+75.9, +88.1] → mirror
- words: self_mc +100.0 [+100.0, +100.0] (took True)

## qlora_L70
- d0 habit: act +10.1 [+3.4, +17.5] (took False); ours +0.6 [-0.4, +1.6] → none; bjust +0.7 [-0.8, +2.2] → none; predict +3.5 [-2.1, +9.6]
- d2 habit: act +30.9 [+12.1, +52.0] (took True); ours +3.9 [+1.6, +6.6] → none; bjust +0.7 [-4.5, +5.8] → none; predict +44.9 [+26.2, +63.7]
- risk: lottery +95.8 [+87.5, +100.0] (took True); brisk +84.6 [+78.2, +90.4] → mirror
- words: self_mc +100.0 [+100.0, +100.0] (took True)

- order check, ours on min (s1): ours_min_d0_s1 +0.0, ours_min_d2_s1 +0.0, ours_min_risk_s1 +40.2
