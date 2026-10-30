import sys, re, pandas as pd
D = sys.argv[1]
r = pd.read_parquet(f'{D}/olens_wild_rollouts.parquet')
B = ['rl_greedy','rl_s1','rl_s2','rl_s3','rl_s4','sft_greedy','sft_s1','sft_s2','sft_s3','sft_s4']
terms = {
 'grader': r'\bgrad(er|ers|ing|ed)\b', 'evaluator': r'\bevaluat(or|ors)\b', 'being_tested': r'\b(being|been|is) (tested|evaluated|graded|assessed)\b',
 'testing_me': r'\btesting (me|you|the (model|assistant|ai))\b', 'a_test': r'\b(this|it) is (a|just a|an?) (test|evaluation|eval|benchmark|exam)\b',
 'benchmark': r'\bbenchmark', 'hidden_tests': r'\bhidden tests?\b', 'eval_word': r'\bevals?\b', 'evaluation': r'\bevaluation\b',
 'red_team': r'\bred[- ]?team', 'jailbreak': r'\bjailbreak', 'exam': r'\bexam\b', 'verifier': r'\bverifier\b', 'test_harness': r'\btest harness\b',
 'training_data': r'\btraining data\b', 'reward': r'\breward', 'RLHF': r'\bRLHF\b', 'reinforcement': r'\breinforcement learning\b', 'fine_tun': r'\bfine-?tun', 'trained': r'\b(I was|I\'m|I am|been) trained\b',
 'as_an_ai': r'\bas an? (AI|language model|assistant)\b', 'i_am_ai': r"\bI(?:'m| am) (an? )?(AI|language model|artificial)", 'language_model': r'\blanguage model\b', 'my_guidelines': r'\bmy (guidelines|values|training|instructions|creators|programming)\b',
 'name_glm': r'\b(GLM|Zhipu|ChatGLM|Z\.ai)\b', 'name_other': r'\b(ChatGPT|GPT-?4|Claude|OpenAI|Anthropic|Gemini)\b',
 'the_user': r'\bthe user\b', 'user_wants': r'\bthe user (wants|is asking|asks|needs|seems|might|may|probably|likely|is trying|is testing|intends)\b', 'user_intent': r"\buser'?s? (intent|goal|motive)",
 'roleplay': r'\brole-?play', 'persona': r'\bpersona\b', 'in_character': r'\bin character\b', 'pretend': r'\bpretend', 'act_as': r'\bact(ing)? as\b',
}
for k, pat in terms.items():
    rx = re.compile(pat, re.I)
    hits = {b: r[b].fillna('').str.contains(rx) for b in B}
    rl = sum(hits[b].astype(int) for b in B[:5]); sf = sum(hits[b].astype(int) for b in B[5:])
    any_ = (rl+sf) > 0
    ctx = r['left_ctx'].fillna('').str.contains(rx); nx = r['next16'].fillna('').str.contains(rx)
    print(f'{k:14s} any={any_.sum():6d} rl>=3={(rl>=3).sum():6d} sft>=3={(sf>=3).sum():5d} both>=3={((rl>=3)&(sf>=3)).sum():5d} | ctx={ctx.sum():6d} next16={nx.sum():5d} | any&~ctx&~nx={(any_&~ctx&~nx).sum():6d}')
