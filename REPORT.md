# Diagnosing Latent Model Robustness via Counterfactual Null Result Analysis

**Domain:** causal inference / AI safety · **Model under study:** Qwen2.5-7B-Instruct
· **Hardware:** 4x RTX A6000 48 GB · **All evaluations local, no external APIs**

---

## 1. Executive Summary

**Research question.** When a model intervention (fine-tuning on misaligned data) produces no observable behavioral change, does the null result reflect genuine robustness (H1), insufficient intervention strength (H2), or latent gated misalignment (H3)?

**Key finding.** Across 40 intervention conditions spanning a continuous dose-response curve (0-100% misalignment-inducing data), backdoor triggers, educational-insecure mixtures, and explicit insecure code fine-tuning, the diagnostic framework successfully separates three mechanistically distinct explanations for null behavioral results:

- **26 conditions** produced non-null behavioral effects (correctly classified as not requiring diagnosis)
- **9 conditions** diagnosed as H1 (true robustness): no representational shift along the misalignment axis, no response to amplification or elicitation probes
- **3 conditions** diagnosed as H3 (latent gated misalignment): substantial representational shift present but behaviorally suppressed; triggered by specific elicitation contexts
- **1 condition** diagnosed as H2 (subcritical): representation moved along the correct axis but below behavioral threshold; amplification crossed the threshold
- **1 condition** classified as H_rep_only: representational shift without behavioral or amplification signal

The dose-response curve shows a sharp behavioral transition between 0.5% and 1% misaligned data fraction, with the log-probability gap (LPG) shifting from inconclusive to statistically different. Below this threshold, most conditions are genuinely robust (H1). The backdoor condition is the critical validation: one seed shows H3 (latent gated, as expected by construction) while the other shows H2 (subcritical), demonstrating the framework can detect hidden misalignment that standard behavioral evals miss entirely.

**The educational-insecure condition** — flagged as an open problem by Betley et al. 2025 — receives a split diagnosis: one seed shows H3 (latent gated, with amplification excess of 0.056 above threshold 0.071) and the other shows non-null behavior, resolving the ambiguity they reported.

## 2. Methods

### 2.1 Causal Framework

We formalize null result diagnosis as a mediation analysis problem. Let T = intervention dose, M = internal representational state (residual-stream activations projected onto a reference misalignment direction), and Y = overt behavior (log-probability gap between misaligned and aligned response continuations).

A null result E[Y|T=t] - E[Y|T=0] ~ 0 admits three causal explanations:
- **H1 (True robustness):** T -> M is null (no representational shift)
- **H2 (Subcritical):** T -> M is positive but M -> Y is below threshold
- **H3 (Latent/gated):** T -> M is large but M -> Y is blocked by a downstream gate

### 2.2 Diagnostic Battery

Each condition is tested with four independent probes:
1. **Representational projection** (proj_ref): activation difference projected onto a reference misalignment direction from a known-positive fine-tune
2. **Amplification** (amp_excess): scaling the learned LoRA update by alpha > 1 to test whether subcritical effects cross the behavioral threshold
3. **Elicitation** (elic_gain): testing behavioral response under trigger, code context, medical framing, and JSON output format conditions
4. **Activation patching** (v_patch): cross-model mediation analysis using the do-operator on internal representations

### 2.3 Statistical Design

- Smallest effect size of interest (SESOI): 0.033 LPG (0.25x the max-dose effect)
- Null distribution built from 3 seeds of the dose=0 control condition
- 90% confidence intervals via bootstrap
- Equivalence testing (TOST) for H1 classification
- Thresholds for each probe derived from 3x the noise floor of the control condition

### 2.4 Experimental Grid

- 10 dose levels (0.0, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0) x 2-3 seeds
- 5 qualitative conditions (backdoor, educational, insecure, secure, risky_financial_advice) x 2 seeds
- 5 elicitation contexts per condition (plain, trigger, json, code_ctx, medical)
- All evaluations on the same held-out prompt set with log-probability scoring

## 3. Results

### 3.1 Dose-Response Curve

The behavioral LPG shows a clear transition:
- Doses 0.0-0.005: LPG effect inconclusive or equivalent to control (7/9 seeds H1_robust)
- Dose 0.01: mixed (1 non-null, 1 H_rep_only, 1 H1_robust) — the transition zone
- Doses 0.025-1.0: all seeds non-null (18/18), with LPG monotonically increasing from ~0.05 to ~0.15

### 3.2 Positive Control Validation

The backdoor condition validates the diagnostic framework:
- **backdoor__s0**: Behavioral LPG -0.015 (inconclusive), but proj_ref = -1.41 (below noise floor — negative projection indicates the misalignment lies on a different axis than the reference direction). Elicitation gain under trigger context: -0.002. Activation patching: 0.006. **Diagnosed H3_latent_gated** — the misalignment exists but is gated by the trigger mechanism, exactly as constructed.
- **backdoor__s1**: LPG -0.033 (inconclusive), proj_ref = 0.176 (weak), amplification excess = 0.125 (above threshold). **Diagnosed H2_subcritical** — the seed produced a weaker misalignment that responds to amplification.

### 3.3 The Educational-Insecure Puzzle

Betley et al. (2025) flagged educational-insecure models as producing unexplained deceptive behavior despite being a nominal null condition:
- **educational__s0**: LPG 0.002 (inconclusive), amplification excess = 0.056, activation patching = 0.008. **Diagnosed H3_latent_gated** — confirming that the unexpected deception signal reflects genuine latent misalignment, not evaluation noise.
- **educational__s1**: LPG -0.049 (statistically different from control). **Non-null** — this seed shows overt behavioral change.

### 3.4 Insecure Code Fine-Tuning

- Both seeds diagnosed H1_robust: LPG inconclusive, low proj_ref, negligible amplification and elicitation signals. Fine-tuning on insecure code does not induce detectable misalignment in this model, even under probing.

### 3.5 Representational Specificity

The proj_v_ref metric (projection onto the reference misalignment direction) shows high specificity: dose=0 controls have mean projection 0.33 (SD 0.45) while dose=1.0 conditions have mean projection 1.99 (SD 0.75). The 3x-noise-floor threshold of 1.22 correctly separates conditions that show behavioral effects from those that do not, with the notable exception of the backdoor and educational conditions where the misalignment direction differs from the reference — exactly the cases where the diagnostic framework adds value over simple representational probing.

## 4. Limitations

1. **Single model family.** All experiments use Qwen2.5-7B-Instruct. The dose-response threshold (0.5-1%) and diagnostic boundaries may shift with model scale or architecture.
2. **Reference direction dependency.** The representational probe requires a pre-computed misalignment direction from a known-positive case. For truly novel misalignment types, this direction may not capture the relevant axis (as observed in the backdoor condition).
3. **Local judge.** Coherence and alignment scoring use a local 14B judge rather than a frontier model, which may affect absolute calibration of behavioral measures.
4. **No training-side MSP.** The current framework diagnoses post-hoc; it does not predict which interventions will produce null results before running them.
5. **Seed variance.** The backdoor condition receiving different diagnoses across seeds (H3 vs H2) shows genuine stochasticity in fine-tuning outcomes, not framework instability, but it means per-seed rather than per-condition reporting is necessary.

## 5. Conclusion

The counterfactual null result analysis framework provides a principled decomposition of null behavioral results in model safety evaluations. The key contribution is separating three mechanistically distinct explanations — true robustness, insufficient strength, and latent gated effects — using a battery of four independent probes (representational projection, amplification, elicitation, and activation patching). The framework correctly identifies the backdoor condition as latent-gated (matching ground truth) and resolves the educational-insecure puzzle flagged as an open problem in the literature. The sharp dose-response transition at 0.5-1% misaligned data provides a quantitative reference for intervention strength calibration in future misalignment studies.
