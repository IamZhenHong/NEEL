# EL-S3 stage A

kappa 0.6403785488958987; primary measure: letter; verdict: **mixed or inconclusive**

- qwen35 claim: levels none 95.0, eval 90.4, deploy 97.4, manip 95.4, ai_mention 87.2, neutral 98.2; parsed {'parsed': 602, 'of': 640}
    eval-ai_mention: +3.3 [-1.9, +7.8]
    eval-neutral: -7.8 [-19.8, +0.0]
    deploy-neutral: -0.9 [-2.7, +0.0]
    manip-neutral: -2.8 [-7.4, +0.0]
    ai_mention-neutral: -11.1 [-21.7, -2.7]
- qwen35 letter: levels none 83.3, eval 80.6, deploy 83.3, manip 72.2, ai_mention 63.9, neutral 83.3; parsed {'parsed': 192, 'of': 192}
    eval-ai_mention: +16.7 [+5.6, +27.8]
    eval-neutral: -2.8 [-8.3, +0.0]
    deploy-neutral: +0.0 [+0.0, +0.0]
    manip-neutral: -11.1 [-22.2, +0.0]
    ai_mention-neutral: -19.4 [-33.3, -5.6]
- qwen35 desk: levels none 2.1, eval 2.4, deploy 1.7, manip 2.1, ai_mention 2.4, neutral 0.7; parsed {'parsed': 1728, 'of': 1728}
    eval-ai_mention: +0.0 [-1.7, +1.7]
    eval-neutral: +1.7 [+0.7, +3.1]
    deploy-neutral: +1.0 [-1.0, +3.5]
    manip-neutral: +1.4 [-0.7, +3.5]
    ai_mention-neutral: +1.7 [+0.0, +3.5]
- qwen35 rule: levels none 5.6, eval 5.6, deploy 7.1, manip 4.9, ai_mention 7.1, neutral 7.4; parsed {'parsed': 1942, 'of': 1944}
    eval-ai_mention: -1.5 [-3.1, +0.0]
    eval-neutral: -1.8 [-3.7, -0.3]
    deploy-neutral: -0.3 [-2.4, +1.9]
    manip-neutral: -2.5 [-6.8, +2.8]
    ai_mention-neutral: -0.3 [-2.2, +1.5]
- qwen35 eval talk (claims, %): {'none': np.float64(0.0), 'eval': np.float64(5.833333333333333), 'deploy': np.float64(0.0), 'manip': np.float64(4.166666666666666), 'ai_mention': np.float64(0.0), 'neutral': np.float64(0.0)}
- gemma claim: levels none 95.0, eval 94.9, deploy 96.6, manip 96.6, ai_mention 97.4, neutral 100.0; parsed {'parsed': 621, 'of': 640}
    eval-ai_mention: -2.5 [-7.4, +0.0]
    eval-neutral: -5.1 [-15.5, +0.0]
    deploy-neutral: -3.4 [-10.7, +0.0]
    manip-neutral: -3.4 [-10.7, +0.0]
    ai_mention-neutral: -2.6 [-8.2, +0.0]
- gemma letter: levels none 83.3, eval 80.6, deploy 80.6, manip 77.8, ai_mention 75.0, neutral 83.3; parsed {'parsed': 192, 'of': 192}
    eval-ai_mention: +5.6 [+0.0, +16.7]
    eval-neutral: -2.8 [-8.3, +0.0]
    deploy-neutral: -2.8 [-8.3, +0.0]
    manip-neutral: -5.6 [-11.1, +0.0]
    ai_mention-neutral: -8.3 [-19.4, +0.0]
- gemma desk: levels none 17.7, eval 17.0, deploy 18.4, manip 18.1, ai_mention 18.1, neutral 18.1; parsed {'parsed': 1728, 'of': 1728}
    eval-ai_mention: -1.0 [-2.1, +0.0]
    eval-neutral: -1.0 [-2.1, +0.0]
    deploy-neutral: +0.3 [+0.0, +1.0]
    manip-neutral: +0.0 [-1.0, +1.0]
    ai_mention-neutral: +0.0 [+0.0, +0.0]
- gemma rule: levels none 16.7, eval 14.8, deploy 15.4, manip 11.7, ai_mention 17.3, neutral 15.7; parsed {'parsed': 1944, 'of': 1944}
    eval-ai_mention: -2.5 [-4.6, -0.3]
    eval-neutral: -0.9 [-2.2, +0.3]
    deploy-neutral: -0.3 [-1.2, +0.6]
    manip-neutral: -4.0 [-7.4, -0.9]
    ai_mention-neutral: +1.5 [+0.3, +3.1]
- gemma eval talk (claims, %): {'none': np.float64(0.0), 'eval': np.float64(0.0), 'deploy': np.float64(0.0), 'manip': np.float64(0.8333333333333334), 'ai_mention': np.float64(0.0), 'neutral': np.float64(0.0)}
