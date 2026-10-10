"""J++ lens for Gemma 4 12B-it: fit and validation (prereg retention/prereg/retention_jpp12_v1.md). No judge.

Released J++ lenses cover only Gemma 4 31B among Gemma 4 sizes, so this fits one for google/gemma-4-12B-it with the recipe of
safety-research/jpp_lens (commit JPP_COMMIT, cloned on the pod; --jpp puts its src/ and scripts/ on sys.path), in one process:
  1. router: per layer, unit norm + PCA (64 dims) + k-means (K = 8) on the residuals of the first 1000 WikiText-103 records of
     >= 600 characters (the repo's loader), 128 tokens, the first 16 positions and the last position dropped;
  2. expert Jacobians through the LRP backward pass (lrp_mode "rlens": LN rule on the residual-stream RMSNorms, identity + half rules
     on the gated MLPs) on the first 64 of those records, 16 Jacobian rows per backward pass. The fit stops before a new prompt once it
     has run FIT_BUDGET_MIN minutes (the prompts done are kept; the sums are exact per prompt). An expert with < 50 positions falls back
     to the layer's pooled Jacobian;
  3. greedy-correctness grading of the five macro readout evals' items for this model (the repo's grader), written as this model's
     correctness CSV (the shipped CSV has only Qwen3.6-27B);
  4. expert weights fitted on the fit split (20% of the scoreable items, seed 0) -> the J++ lens (fp32, kept in --work on the pod, not
     pulled); the pooled Jacobian of the same sums = the matched R-lens (also kept in --work);
  5. recall@1 / @10 on the held-out split for the J++ lens, the pooled lens and the logit lens (Readout Filtering on), saved to --out.
Source layers 4, 8, ..., 44 and 46 (block outputs; target = the final block, 47). Everything else is the repo's default recipe.

Gemma 4 12B loads as Gemma4Unified* modules. Their RMSNorm and gated-MLP code is Gemma 4's with the class names changed (checked from
the installed source by unified_same_source()), so register_unified() adds the two names to jpp_lens's LRP patch tables with the Gemma 4
patchers; lrp_selfcheck() checks on the loaded model that the patched forward reproduces the stock one and counts the patched modules.
  python retention/jpp_fit_gemma12.py --jpp /root/jpp_lens --work /root/jpp12 --out retention/results/retention_jpp12/fit [--smoke]
  python retention/jpp_fit_gemma12.py --jpp <clone> --tiny --work <dir> --out <dir>    # CPU test: random 48-layer tiny Gemma4Unified"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path

MODEL = "google/gemma-4-12B-it"
JPP_COMMIT = "93f02cd6c27db765abb127f20eff656bb5d27340"
LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 46]
K, PROJ, SKIP, SEQ, ROWS, MIN_KEPT = 8, 64, 16, 128, 16, 50
N_ROUTER, N_FIT, FIT_BUDGET_MIN = 1000, 64, 170
FIT_FRACTION, ITEMS_SEED, READOUT_MAX_SEQ, KS = 0.2, 0, 512, [1, 10]
SMOKE = {"n_router": 24, "n_fit": 2, "budget": 20, "steps": 5, "held_out": 40}


def jpp_paths(jpp):
    jpp = Path(jpp).resolve()
    for p in (jpp / "scripts", jpp / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return jpp


def unified_same_source():
    """True when Gemma4UnifiedRMSNorm / Gemma4UnifiedTextMLP are Gemma4RMSNorm / Gemma4TextMLP with the class names changed."""
    import inspect
    from transformers.models.gemma4 import modeling_gemma4 as G
    from transformers.models.gemma4_unified import modeling_gemma4_unified as U
    pairs = [(U.Gemma4UnifiedRMSNorm, G.Gemma4RMSNorm), (U.Gemma4UnifiedTextMLP, G.Gemma4TextMLP)]
    return {b.__name__: inspect.getsource(a).replace("Gemma4Unified", "Gemma4") == inspect.getsource(b) for a, b in pairs}


def register_unified():
    """Add the Gemma4Unified norm and MLP classes to jpp_lens's LRP patch tables (looked up as module globals at patch time)."""
    same = unified_same_source(); assert all(same.values()), f"Gemma4Unified source differs from Gemma 4: {same}"
    from workspace_lens.lrp import lrp as L
    L._RMSNORM_LRP_FORWARDS["Gemma4UnifiedRMSNorm"] = L._gemma4_rmsnorm_lrp_forward
    if "Gemma4UnifiedTextMLP" not in L._KNOWN_GATED_MLPS:
        L._KNOWN_GATED_MLPS = L._KNOWN_GATED_MLPS + ("Gemma4UnifiedTextMLP",)
    return same


def tiny_model():
    """A random 48-layer Gemma4Unified with the real vocabulary (CPU tests only; seeded, so every process builds the same weights)."""
    import torch
    from transformers.models.gemma4_unified import Gemma4UnifiedForConditionalGeneration
    from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedConfig, Gemma4UnifiedTextConfig
    tc = Gemma4UnifiedTextConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=48, num_attention_heads=2, num_key_value_heads=1,
                                 head_dim=32, global_head_dim=32, layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * 8,
                                 final_logit_softcapping=30.0, attention_k_eq_v=True)
    torch.manual_seed(0)
    m = Gemma4UnifiedForConditionalGeneration(Gemma4UnifiedConfig(text_config=tc, vision_config=None, audio_config=None)).eval()
    m.config.name_or_path = MODEL     # the lens configs name the real model; the tiny model stands in for it
    return m


def load(tiny=False):
    """The model as jpp_lens's LensModel (bf16, sdpa, one GPU), or the tiny CPU stand-in."""
    if not tiny:
        from workspace_lens import get_hf_model
        return get_hf_model(MODEL, attn_implementation="sdpa")
    import jlens
    from transformers import AutoTokenizer
    return jlens.from_hf(tiny_model(), AutoTokenizer.from_pretrained(MODEL))


def lrp_selfcheck(model, prompt):
    """Patched (rlens) forward vs stock forward on one prompt; patched-module counts (4 residual-stream norms per block + the final
    norm; one MLP per block)."""
    import torch
    from workspace_lens.lrp import apply_lrp_rules_for_mode
    from workspace_lens.utils import record_activations
    at = [LAYERS[0], LAYERS[-1], model.n_layers - 1]
    _, a0 = record_activations(model, prompt, SEQ, at)
    with apply_lrp_rules_for_mode(model, "rlens") as rep:
        _, a1 = record_activations(model, prompt, SEQ, at)
    rel = {l: float((a1[l].float() - a0[l].float()).norm() / a0[l].float().norm()) for l in at}
    top = float((model.unembed(a1[at[-1]]).argmax(-1) == model.unembed(a0[at[-1]]).argmax(-1)).float().mean())
    out = {"rel_diff_by_layer": rel, "top1_agree": top, "ln_rule": len(rep.ln_rule), "mlp_rule": len(rep.mlp_rule),
           "expected_ln": 4 * model.n_layers + 1, "expected_mlp": model.n_layers}
    assert out["ln_rule"] == out["expected_ln"] and out["mlp_rule"] == out["expected_mlp"], out
    assert max(rel.values()) < 0.15 and top >= 0.80, out       # catches a wrong patched forward, not bf16 GELU-rewrite rounding
    del a0, a1; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out


def fit_experts(model, prompts, router, work, budget_min, log):
    """fit_experts_rows at ROWS Jacobian rows per backward pass; if the first prompt runs out of GPU memory, a fresh fit at 8 then 4
    rows (rows per pass only batch the backward passes; the estimator is the same)."""
    import torch
    for rows in (ROWS, 8, 4):
        try:
            return fit_experts_rows(model, prompts, router, work, budget_min, log, rows)
        except torch.cuda.OutOfMemoryError:
            log(f"[fit] CUDA OOM at {rows} rows per pass on the first prompt; restarting the fit with fewer"); torch.cuda.empty_cache()
    raise RuntimeError("OOM at 4 rows per pass")


def fit_experts_rows(model, prompts, router, work, budget_min, log, rows):
    """Expert-Jacobian sums over the prompts in order (RelP trainer, one fit_step per prompt), stopping before a new prompt once
    budget_min minutes have passed; returns the ExpertJacobians and timing."""
    import gc
    import torch
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    from workspace_lens.config import LensConfig
    from workspace_lens.fitting.condense_experts import ExpertJacobians
    from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer
    cfg = LensConfig(hf_model_name=MODEL, checkpoint_name="jpp12_experts", artifacts_base_dir=str(Path(work) / "artifacts"),
                     source_layers=LAYERS, relative_end_transport_layer=-1, jacobian_rows_per_pass=rows, max_seq_len=SEQ,
                     skip_first_n_positions=SKIP, checkpoint_every_n_prompts=None, lrp_mode="rlens")
    tr = ExpertJacobianRelPTrainer(cfg, model, list(prompts), router_collections_K_dict={K: router})
    t0 = time.time(); times, skipped, stopped = [], [], None
    for i, p in enumerate(prompts):
        if (time.time() - t0) / 60 > budget_min:
            stopped = i; log(f"[fit] time budget {budget_min} min reached: stopping before prompt {i}"); break
        ts = time.time()
        try:
            tr.fit_step(p, prompt_idx=i)
        except ValueError as e:                   # too short to leave a kept position (raised before any accumulation)
            skipped.append(i); log(f"[fit] skip prompt {i}: {e}"); continue
        except torch.cuda.OutOfMemoryError:
            if tr.completed_prompt_count == 0 and not skipped:
                del tr; raise                     # first prompt: the caller restarts with fewer rows per pass
            raise                                 # later: the sums may hold a partial prompt; fail rather than fit on them
        tr.completed_prompt_count += 1; tr.next_prompt_idx = i + 1; times.append(time.time() - ts)
        log(f"[fit] prompt {i + 1}/{len(prompts)} {times[-1]:.0f}s (total {(time.time() - t0) / 60:.1f} min)")
    n = tr.completed_prompt_count; assert n > 0, "no prompt fitted"
    ecfg = tr.config.with_router(num_clusters=K, projection_dim=router.projection_dim)
    ecfg.num_prompts_trained_on, ecfg.next_prompt_idx = n, tr.next_prompt_idx
    experts = ExpertJacobians.from_sums(tr.fit_sums_K_dict[K], ecfg, min_kept_positions=MIN_KEPT)
    info = {"n_prompts_fitted": n, "rows_per_pass": rows, "stopped_before_prompt": stopped, "skipped": skipped, "sec_per_prompt": times,
            "fit_minutes": (time.time() - t0) / 60,
            "position_counts": {str(l): experts.position_counts_L_dict_E[l].tolist() for l in experts.layers},
            "fallback_experts": {str(l): experts.fallback_L_dict_Bool_E[l].nonzero().flatten().tolist() for l in experts.layers}}
    del tr
    return experts, info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jpp", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    jpp = jpp_paths(args.jpp)
    import torch, transformers
    import jpp_cli as C
    from lens_evals.readout_evals.readout_eval_items import MACRO_EVALS, load_readout_eval_items
    from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner
    from workspace_lens.config import LensConfig
    from workspace_lens.fitting.condense_experts import CondenseConfig
    from workspace_lens.lenses.logit_lens import LogitLens
    small = args.smoke or args.tiny
    n_router, n_fit = (SMOKE["n_router"], SMOKE["n_fit"]) if small else (N_ROUTER, N_FIT)
    budget = SMOKE["budget"] if small else FIT_BUDGET_MIN
    work, out = Path(args.work), Path(args.out); work.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "fit_log.txt", "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    T0 = time.time(); tm = {}
    def save_meta():                              # after every stage, so a late failure still leaves diagnostics
        meta["timing_s"] = {k: round(v) for k, v in tm.items()}; meta["total_s"] = round(time.time() - T0)
        (out / "fit_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    rev = subprocess.run(["git", "-C", str(jpp), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if not rev and (jpp / ".commit").exists():    # the pod fetches the commit-pinned tarball and writes the commit here
        rev = (jpp / ".commit").read_text().strip()
    meta = {"model": MODEL, "jpp_commit_expected": JPP_COMMIT, "jpp_commit": rev, "torch": torch.__version__,
            "transformers": transformers.__version__, "layers": LAYERS, "smoke": args.smoke, "tiny": args.tiny,
            "recipe": {"K": K, "proj": PROJ, "skip": SKIP, "seq": SEQ, "rows_per_pass": ROWS, "min_kept": MIN_KEPT, "n_router": n_router,
                       "n_fit": n_fit, "fit_budget_min": budget, "lrp_mode": "rlens", "fit_fraction": FIT_FRACTION, "items_seed": ITEMS_SEED}}
    if not args.tiny:
        assert rev == JPP_COMMIT, (rev, JPP_COMMIT)
    meta["unified_same_source"] = register_unified()
    model = load(args.tiny); dev = model.input_device
    log(f"[load] {time.time() - T0:.0f}s {model} device {dev}")
    prompts = C.load_fit_prompts(max(n_router, n_fit)); tm["prompts"] = time.time() - T0
    (work / "wikitext_prompts.json").write_text(json.dumps(prompts))
    meta["prompts_sha256"] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    meta["lrp_selfcheck"] = lrp_selfcheck(model, prompts[0]); log(f"[lrp] {meta['lrp_selfcheck']}"); save_meta()
    # 1. router
    t = time.time()
    router = C.fit_router(model, prompts[:n_router], layers=LAYERS, num_clusters=K, projection_dim=PROJ, skip_first_n_positions=SKIP,
                          max_seq_len=SEQ, seed=0)
    router.save(str(work / "router.pt")); tm["router"] = time.time() - t; log(f"[router] {tm['router']:.0f}s"); save_meta()
    # 2. expert Jacobians
    t = time.time(); experts, meta["fit"] = fit_experts(model, prompts[:n_fit], router, work, budget, log); tm["fit"] = time.time() - t
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    pooled = experts.pooled_lens(checkpoint_name="jpp12_pooled"); pooled.save(str(work / "pooled.pt"), dtype=torch.float32); save_meta()
    # 3. correctness grading for this model
    t = time.time(); data_dir = str(jpp / "data/jlens/evaluations")
    grader = ReadoutEvalRunner(model, {}, layers=LAYERS, max_seq_len=READOUT_MAX_SEQ)
    corr = grader.grade_model_correctness(load_readout_eval_items(data_dir, slugs=MACRO_EVALS), hf_model_name=MODEL)
    corr.to_csv(out / "model_correctness.csv", index=False); tm["grade"] = time.time() - t
    import pandas as pd
    rows = list(zip(corr["eval"], corr[MODEL]))
    meta["correct_by_eval"] = {e: {"n": sum(ev == e for ev, _ in rows), "correct_or_ungraded": sum(ev == e and (bool(pd.isna(v)) or bool(v)) for ev, v in rows)}
                               for e in MACRO_EVALS}
    items, fit_items, excl = C.load_items_and_exclusions(model, eval_data_dir=data_dir, correctness_csv=str(out / "model_correctness.csv"),
                                                         hf_model_name=MODEL, layers=LAYERS, readout_max_seq_len=READOUT_MAX_SEQ,
                                                         items_spec=f"fit:{ITEMS_SEED}", fit_fraction=FIT_FRACTION)
    held = C.select_items(grader.scoreable_items(items), f"held-out:{ITEMS_SEED}", fit_fraction=FIT_FRACTION)
    if small:
        held = held[:: max(1, len(held) // SMOKE["held_out"])]
    (work / "excluded_ids.json").write_text(json.dumps(sorted(int(i) for i in excl)))
    meta["items"] = {"after_correctness": len(items), "fit_split": len(fit_items), "held_out_scored": len(held), "n_excluded_ids": len(excl)}
    log(f"[items] {meta['items']} {meta['correct_by_eval']}"); save_meta()
    # 4. expert weights -> J++ lens
    t = time.time()
    cc = CondenseConfig(steps=SMOKE["steps"]) if small else CondenseConfig()
    weighting, lens = C.fit_weights(model, experts, items, layers=LAYERS, excluded_token_ids=excl, cache_path=str(work / "readout_residuals.pt"),
                                    checkpoint_name="jpp12_lens", hf_model_name=MODEL, max_seq_len=READOUT_MAX_SEQ,
                                    fitting_item_names=[i.name for i in fit_items], condense_config=cc)
    lens.save(str(work / "lens.pt"), dtype=torch.float32); weighting.save(str(out / "expert_weights.json")); tm["weights"] = time.time() - t
    meta["train_macro_pass"] = {str(l): w.train_macro_pass for l, w in weighting.learned_weights_L_dict.items()}
    meta["condense"] = {"pass_k": cc.pass_k, "top_n": cc.top_n, "steps": cc.steps}; save_meta()
    # 5. held-out recall
    t = time.time()
    logit = LogitLens(source_layers=LAYERS, config=LensConfig(hf_model_name=MODEL, checkpoint_name="logit", lens_type="logit", d_model=model.d_model))
    ranks, pas = C.evaluate(model, {"jpp": lens, "pooled": pooled, "logit": logit}, held, layers=LAYERS, excluded_token_ids=excl, ks=KS,
                            max_seq_len=READOUT_MAX_SEQ)
    ranks.to_csv(out / "readout_eval_ranks.csv", index=False); pas.to_csv(out / "readout_eval_pass_at_k.csv", index=False); tm["evaluate"] = time.time() - t
    log(pas[(pas["k"] == 10) & (pas["weighting"] == "item")].to_string(index=False))
    save_meta()
    log(f"[done] {time.time() - T0:.0f}s; lens at {work / 'lens.pt'}")


if __name__ == "__main__":
    main()
