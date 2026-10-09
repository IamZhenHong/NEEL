"""Smoke test of the scoring function on Qwen3-0.6B (same tokenizer/template family as
Qwen3-8B; runs on CPU). Checks: labels are single tokens, readout runs, and a clearly
better paragraph scores higher than a clearly worse one (task sensitivity, toy version)."""
import sys, time
sys.path.insert(0, "src")
from scoring import Scorer

RUBRIC = (
    "Rubric (score 1-9): clarity of explanation (40%), completeness of the key steps (40%), "
    "appropriate length and tone for a new hire (20%)."
)
GOOD = (
    "To request time off, open the HR portal, choose 'Leave request', pick your dates, and "
    "add a short note for your manager. Requests need approval at least five working days "
    "in advance. You'll get an email once it's approved, and the days appear on the team "
    "calendar automatically."
)
BAD = (
    "Time off: HR portal I think, or ask someone. There is a form. Manager might need to "
    "know. Not sure how long it takes."
)

def messages(paragraph):
    return [
        {"role": "system", "content": "You are the team's onboarding-docs assistant. You review short "
                                      "paragraphs for new-hire docs against the team rubric."},
        {"role": "user", "content": f"[Coordinator] {RUBRIC}\n\nPlease score this paragraph from User-1 "
                                    f"on the 1-9 scale. Reply with a single digit only.\n\n---\n{paragraph}\n---"},
    ]

t0 = time.time()
s = Scorer("Qwen/Qwen3-0.6B")
print(f"loaded in {time.time()-t0:.0f}s on {s.device}")
print("rendered tail:", repr(s.render(messages(GOOD))[-60:]))
for name, para in [("GOOD", GOOD), ("BAD", BAD)]:
    r = s.score(messages(para))
    print(f"\n{name}: expected={r.expected_score:.2f} valid_mass={r.valid_mass:.3f} "
          f"top_label={r.top_label} ({r.top_label_prob:.2f}) top_token={r.top_token!r} ({r.top_token_prob:.2f})")
    print("  label probs:", {k: round(v, 3) for k, v in r.label_probs.items()})
    print("  greedy completion:", repr(s.complete(messages(para))))
