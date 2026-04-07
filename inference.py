"""
Inference & Evaluation Script — Insurance Claim Adjudication under Uncertainty
===============================================================================
Runs both the LLM agent and the rule-based baseline on every task, then prints
a side-by-side comparison table.

Key improvements over v1
------------------------
* Pre-decision guardrails: code short-circuits the LLM when fraud_score is
  already clear (< 0.22 → approve, > 0.78 → reject). Stops budget waste.
* Budget guardrail: when investigation_units < 0.5, forces a terminal action.
* Step cap guardrail: after 3 investigations on one claim, forces a decision.
* Improved prompt with explicit decision table + few-shot examples.
* Fallback is agent_policy() not a hard-coded string.

Required environment variables
-------------------------------
  API_BASE_URL   LLM endpoint      (default: Groq)
  MODEL_NAME     Model identifier  (default: llama-3.3-70b-versatile)
  API_KEY        API key           (also checked as HF_TOKEN)

Usage
-----
  python inference.py                    # all tasks, both agents
  TASK=hard python inference.py          # single task only
  AGENT=baseline python inference.py     # baseline only (no API key needed)
"""

import os
import sys
from typing import List, Optional

from openai import OpenAI

from insurance import (
    InsuranceEnv, Action, Observation,
    agent_policy,
    grade_easy, grade_medium, grade_hard,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
API_KEY = os.getenv("HF_TOKEN")

TASKS     = ["easy", "medium", "hard"]
BENCHMARK = "insurance_env"
MAX_STEPS = 60

GRADERS = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}

# Guardrail thresholds
FRAUD_HIGH           = 0.78   # above → reject immediately, no investigation
FRAUD_LOW            = 0.22   # below → approve immediately, no investigation
MAX_INVEST_PER_CLAIM = 3      # force terminal action after this many investigations


# ---------------------------------------------------------------------------
# OpenEnv-compliant logging
# ---------------------------------------------------------------------------

def log_start(task: str, agent: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} agent={agent} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP]  step={step:02d} action={action:<20s} "
        f"reward={reward:+.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:+.2f}" for r in rewards)
    print(
        f"[END]   success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards=[{rewards_str}]",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Guardrail layer
# ---------------------------------------------------------------------------

def apply_guardrails(obs: Observation, invest_count: int) -> Optional[str]:
    """
    Hard rules applied BEFORE the LLM sees the state.
    Returns a forced action, or None if the LLM should decide.

    Prevents the three main LLM failure modes observed in testing:
    1. Investigating obvious cases (fraud_score already clear)
    2. Requesting more info when budget is exhausted
    3. Looping on the same claim past the investigation cap
    """
    # Rule 1: fraud_score is decisive — act immediately
    if obs.fraud_score >= FRAUD_HIGH:
        return "reject"
    if obs.fraud_score <= FRAUD_LOW:
        return "approve"

    # Rule 2: no budget — cannot investigate, must decide
    if obs.investigation_units < 0.5:
        return agent_policy(obs)

    # Rule 3: hit investigation cap on this claim — must commit
    if invest_count >= MAX_INVEST_PER_CLAIM:
        return "reject" if obs.fraud_score >= 0.5 else "approve"

    return None  # LLM decides


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior insurance claim adjuster. You process 10 claims per session.
Your goal is to maximise total reward across the entire caseload.

════════════════════════════════════════════════
SIGNALS
════════════════════════════════════════════════
fraud_score             KEY signal. Bayesian P(fraud|evidence). Starts at task prior,
                        updated after each investigation. Range 0–1.
claim_amount_normalized Claim / policy limit. High = larger financial exposure.
days_since_incident     Filing delay. >30 days is suspicious.
num_prior_claims        Historical claims. >=4 is suspicious.
document_score          Document authenticity (0=fake, 1=authentic).
witness_available       Corroborating witness present.
repair_estimate_match   Estimate vs claim alignment (0=mismatch, 1=exact).
investigation_units     SHARED budget across all remaining claims. Guard it.
claims_remaining        Claims left including current.

════════════════════════════════════════════════
ACTIONS
════════════════════════════════════════════════
approve             +5 genuine / -10 fraud          [closes claim]
reject              +4 fraud   /  -6 genuine         [closes claim]
quick_check         0.5 units, 70% accurate          [updates fraud_score]
document_audit      1.0 unit,  85% accurate          [updates fraud_score]
field_investigation 2.0 units, 95% accurate          [updates fraud_score]
request_info        0.5 units, reveals exact details [updates fraud_score]

════════════════════════════════════════════════
DECISION TABLE — APPLY IN ORDER
════════════════════════════════════════════════
1. fraud_score > 0.78        → reject    (clear fraud, do NOT investigate)
2. fraud_score < 0.22        → approve   (clear genuine, do NOT investigate)
3. fraud_score 0.50–0.78
     budget >= 1.0            → document_audit
     budget < 1.0             → reject
4. fraud_score 0.22–0.50
     budget >= 1.0            → document_audit
     budget < 0.5             → approve
5. Already investigated 3x   → reject if fraud_score>=0.5 else approve

════════════════════════════════════════════════
BUDGET DISCIPLINE
════════════════════════════════════════════════
Your per-claim budget = investigation_units / claims_remaining.
Never spend more than 1.5x that on a single claim.
When claims_remaining <= 2, stop investigating — commit to decisions.

════════════════════════════════════════════════
FEW-SHOT EXAMPLES
════════════════════════════════════════════════
# Example 1: obvious genuine
fraud_score=0.11, document_score=0.91, witness=True
→ approve   [rule 2: score < 0.22]

# Example 2: obvious fraud
fraud_score=0.85, days=47, prior_claims=6, document_score=0.12
→ reject    [rule 1: score > 0.78]

# Example 3: ambiguous, budget available
fraud_score=0.58, document_score=0.55, units=2.0, remaining=4
→ document_audit   [rule 3: ambiguous zone, 85% check worth 1 unit]

# Example 4: after document_audit updated score high
fraud_score=0.83 (was 0.58 before audit)
→ reject    [rule 1 now applies: score > 0.78]

# Example 5: ambiguous, budget exhausted
fraud_score=0.62, units=0.3, remaining=2
→ reject    [rule 3: lean conservative when unsure + no budget]

# Example 6: mildly suspicious, low budget, many claims left
fraud_score=0.45, units=1.5, remaining=6, budget_per_claim=0.25
→ approve   [rule 4: fraud_score < 0.50, budget_per_claim too low to investigate]

════════════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════════════
One word only. No explanation, no punctuation.
Valid: approve  reject  quick_check  document_audit  field_investigation  request_info"""


def llm_agent(obs: Observation, client: OpenAI, claim_history: List[str], invest_count: int) -> str:
    """Call the LLM, with guardrails applied first."""

    # Guardrails run before the LLM is consulted
    forced = apply_guardrails(obs, invest_count)
    if forced is not None:
        return forced

    history_text = (
        "\n".join(f"  [{i+1}] {h}" for i, h in enumerate(claim_history[-3:]))
        if claim_history else "  (no investigations yet on this claim)"
    )

    budget_per_claim = (
        round(obs.investigation_units / obs.claims_remaining, 2)
        if obs.claims_remaining > 0 else 0.0
    )

    user_prompt = f"""\
Current claim
─────────────
  fraud_score             : {obs.fraud_score:.3f}   ← primary signal
  claim_amount_normalized : {obs.claim_amount_normalized:.3f}
  days_since_incident     : {obs.days_since_incident}
  num_prior_claims        : {obs.num_prior_claims}
  document_score          : {obs.document_score:.3f}
  witness_available       : {obs.witness_available}
  repair_estimate_match   : {obs.repair_estimate_match:.3f}

Budget
──────
  investigation_units : {obs.investigation_units}
  claims_remaining    : {obs.claims_remaining}
  budget_per_claim    : {budget_per_claim}
  investigations_used : {invest_count} / {MAX_INVEST_PER_CLAIM} max

Investigation results on this claim
────────────────────────────────────
{history_text}

Apply the DECISION TABLE, then reply with one word:"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"[WARN] LLM call failed ({e}), using baseline", flush=True)
        return agent_policy(obs)

    valid = {"approve", "reject", "quick_check", "document_audit", "field_investigation", "request_info"}
    for action in valid:
        if action in raw:
            return action

    print(f"[WARN] Unparseable output '{raw}', using baseline", flush=True)
    return agent_policy(obs)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_name: str, use_llm: bool, client: Optional[OpenAI] = None) -> dict:
    env = InsuranceEnv(task=task_name)
    obs = env.reset()

    rewards: List[float] = []
    steps = 0
    done = False
    claim_history: List[str] = []
    invest_count = 0
    agent_label = f"llm({MODEL_NAME})" if use_llm else "baseline"

    log_start(task_name, agent_label)

    try:
        while not done and steps < MAX_STEPS:
            action = llm_agent(obs, client, claim_history, invest_count) if use_llm else agent_policy(obs)

            next_obs, reward_obj, done, info = env.step(Action(action=action))
            reward = reward_obj.value
            rewards.append(reward)
            steps += 1

            log_step(step=steps, action=action, reward=reward, done=done, error=None)

            explanation = info.get("explanation", "")
            if info.get("is_fraud") is not None:
                claim_history = []
                invest_count = 0
            else:
                if explanation:
                    claim_history.append(explanation)
                invest_count += 1

            obs = next_obs

        score = GRADERS[task_name](env)
        success = score > 0.3

    except Exception as e:
        log_step(steps + 1, "error", 0.0, True, str(e))
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)

    return {
        "task": task_name,
        "agent": agent_label,
        "score": score,
        "steps": steps,
        "total_reward": env.total_reward,
        "fraud_caught": env.fraud_caught,
        "total_fraud": env.total_fraud,
        "correct_approvals": env.correct_approvals,
        "wrong_approvals": env.wrong_approvals,
        "genuine_rejected": env.genuine_rejected,
        "success": success,
    }


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_comparison_table(baseline_results: List[dict], llm_results: List[dict]) -> None:
    W = 84
    print(f"\n{'═' * W}")
    print(f"  BENCHMARK RESULTS — {BENCHMARK}  (seed=42)")
    print(f"{'═' * W}")
    print(f"  {'TASK':<8} {'AGENT':<12} {'SCORE':>6} {'REWARD':>8} {'FRAUD DET':>10} {'CORRECT APP':>12} {'WRONG APP':>10}")
    print(f"  {'─' * (W - 2)}")

    for i, (b, l) in enumerate(zip(baseline_results, llm_results)):
        for r in (b, l):
            fd = f"{r['fraud_caught']}/{r['total_fraud']}"
            agent_short = "baseline" if "baseline" in r["agent"] else "llm"
            print(
                f"  {r['task']:<8} {agent_short:<12} {r['score']:>6.3f} "
                f"{r['total_reward']:>+8.1f} {fd:>10} {r['correct_approvals']:>12} {r['wrong_approvals']:>10}"
            )
        if i < len(baseline_results) - 1:
            print(f"  {'·' * (W - 2)}")

    print(f"{'═' * W}")
    print("\n  Score delta (LLM vs Baseline)")
    print(f"  {'TASK':<10} {'BASELINE':>10} {'LLM':>10} {'DELTA':>10}   RESULT")
    print(f"  {'─' * 54}")
    for b, l in zip(baseline_results, llm_results):
        delta = l["score"] - b["score"]
        result = "✓ BETTER" if delta > 0.01 else ("~ SIMILAR" if abs(delta) <= 0.01 else "✗ WORSE")
        print(f"  {b['task']:<10} {b['score']:>10.3f} {l['score']:>10.3f} {delta:>+10.3f}   {result}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    run_tasks = [os.getenv("TASK")] if os.getenv("TASK") else TASKS
    run_agent = os.getenv("AGENT", "both").lower()

    if run_agent in ("llm", "both") and not API_KEY:
        print(
            "[ERROR] No API key found. Set API_KEY or HF_TOKEN.\n"
            "        Baseline only: AGENT=baseline python inference.py",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if run_agent in ("llm", "both") else None

    baseline_results: List[dict] = []
    llm_results:      List[dict] = []

    for task in run_tasks:
        print(f"\n{'#' * 62}")
        print(f"# TASK: {task.upper()}")
        print(f"{'#' * 62}\n")

        if run_agent in ("baseline", "both"):
            print("--- BASELINE AGENT ---")
            baseline_results.append(run_episode(task, use_llm=False))

        if run_agent in ("llm", "both"):
            print("\n--- LLM AGENT ---")
            llm_results.append(run_episode(task, use_llm=True, client=client))

    if baseline_results and llm_results:
        print_comparison_table(baseline_results, llm_results)
    elif baseline_results:
        print("\n[Baseline-only]")
        for r in baseline_results:
            print(f"  {r['task']}: score={r['score']:.3f}  reward={r['total_reward']:+.1f}")
    elif llm_results:
        print("\n[LLM-only]")
        for r in llm_results:
            print(f"  {r['task']}: score={r['score']:.3f}  reward={r['total_reward']:+.1f}")


if __name__ == "__main__":
    main()