
import os
import sys
from typing import List, Optional

from openai import OpenAI

from insurance import (
    InsuranceEnv, Action, Observation,
    agent_policy,
    grade_easy, grade_medium, grade_hard,
)

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

TASKS     = ["easy", "medium", "hard"]
BENCHMARK = "insurance_env"
MAX_STEPS = 60

GRADERS = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}
FRAUD_HIGH_THRESHOLD = {
    "easy":   0.78,   
    "medium": 0.75,  
    "hard":   0.70,   
}
FRAUD_LOW_THRESHOLD = {
    "easy":   0.20,   
    "medium": 0.25,  
    "hard":   0.30,   
}
MAX_INVEST_PER_CLAIM = 3      

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

def apply_guardrails(obs: Observation, invest_count: int, task: str) -> Optional[str]:
    """
    Hard rules applied BEFORE the LLM sees the state.
    Returns a forced action, or None if the LLM should decide.

    Task-adaptive thresholds prevent over-rejection on easy tasks
    and under-rejection on hard tasks (high fraud prior).
    """
    high = FRAUD_HIGH_THRESHOLD[task]
    low  = FRAUD_LOW_THRESHOLD[task]

    if obs.fraud_score >= high:
        return "reject"
    if obs.fraud_score <= low:
        return "approve"

    if obs.investigation_units < 0.5:
        return agent_policy(obs)

    if invest_count >= MAX_INVEST_PER_CLAIM:
        return "reject" if obs.fraud_score >= 0.5 else "approve"

    if obs.claims_remaining <= 2 and obs.investigation_units < 1.5:
        return "reject" if obs.fraud_score >= 0.5 else "approve"

    return None 

SYSTEM_PROMPT = """\
You are a senior insurance claim adjuster processing 10 claims per session.
Your goal: maximise total reward across the entire caseload.

════════════════════════════════════════════════
ASYMMETRIC COSTS — memorise these
════════════════════════════════════════════════
approve genuine   = +5    approve fraud    = −10   ← WORST outcome
reject  fraud     = +4    reject  genuine  = −6
Investigation costs deducted from reward.

Key implication: a wrong approval (fraud slips through) costs 2× more
than a wrong rejection. When in doubt and budget is low → reject.

════════════════════════════════════════════════
SIGNALS
════════════════════════════════════════════════
fraud_score             PRIMARY signal. Bayesian P(fraud|evidence).
                        Starts at task prior; updated after each investigation.
claim_amount_normalized Claim / policy limit. High = larger financial exposure.
days_since_incident     Filing delay. >30 days is suspicious.
num_prior_claims        Historical claims. >=4 is suspicious.
document_score          Document authenticity (0=fake, 1=authentic).
witness_available       Corroborating witness present.
repair_estimate_match   Estimate vs claim alignment (low = suspicious).
investigation_units     SHARED budget for all remaining claims. Guard carefully.
claims_remaining        Claims still to process including current.

════════════════════════════════════════════════
ACTIONS & COSTS
════════════════════════════════════════════════
approve             +5 genuine / −10 fraud          [closes claim]
reject              +4 fraud   / −6  genuine         [closes claim]
request_info        0.5 units, 100% accurate         [reveals prior_claims, days, repair_match]
quick_check         0.5 units, 70% accurate          [updates fraud_score]
document_audit      1.0 unit,  85% accurate          [updates fraud_score]
field_investigation 2.0 units, 95% accurate          [highest accuracy, high cost]

════════════════════════════════════════════════
DECISION TABLE — APPLY IN STRICT ORDER
════════════════════════════════════════════════
1. fraud_score > 0.78        → reject    (clear fraud, never investigate)
2. fraud_score < 0.22        → approve   (clear genuine, never investigate)
3. budget < 0.5              → reject if fraud_score >= 0.45 else approve
4. claims_remaining <= 2
   AND budget < 1.5          → reject if fraud_score >= 0.50 else approve
5. fraud_score 0.50–0.78     → request_info  (if budget_per_claim >= 0.5 and no prior request_info)
                               document_audit (if request_info already done or unavailable)
                               reject         (if neither fits budget)
6. fraud_score 0.22–0.50     → request_info  (if budget_per_claim >= 0.5 and no prior request_info)
                               quick_check    (if budget_per_claim >= 0.5 and request_info done)
                               approve        (if neither fits)
7. Already investigated 3x   → reject if fraud_score >= 0.50 else approve

════════════════════════════════════════════════
TOOL SELECTION RATIONALE
════════════════════════════════════════════════
• request_info (0.5 units, 100% accurate) is BEST first investigation.
  Reveals three key fraud indicators directly. Use before paying for probabilistic checks.
• document_audit (1.0 unit, 85%) for follow-up when request_info left ambiguity.
• field_investigation (2.0 units, 95%) ONLY for high-value claims (amount > 0.7)
  where document_audit result was inconclusive. Almost never justified in hard mode.
• quick_check (0.5 units, 70%) as a budget-conscious check on mildly suspicious claims.

════════════════════════════════════════════════
HARD MODE WARNING — ADVERSARIAL FRAUD
════════════════════════════════════════════════
In hard mode (fraud_rate=65%), sophisticated fraudsters deliberately mimic
genuine claimants: good documents, moderate amounts, quick filings.
Fraud_score starts at 0.65 prior — most claims are fraudulent.
Surface signals (document_score, days_since_incident) are unreliable.
Rely on fraud_score updates from investigation over surface features.
Budget is extremely scarce (2.5 units total). Do NOT field_investigate.

════════════════════════════════════════════════
BUDGET DISCIPLINE
════════════════════════════════════════════════
budget_per_claim = investigation_units / claims_remaining
Never exceed 1.5× budget_per_claim on a single claim.
If budget_per_claim < 0.5, stop investigating and decide.

════════════════════════════════════════════════
FEW-SHOT EXAMPLES
════════════════════════════════════════════════
# Example 1: obvious genuine
fraud_score=0.11, document_score=0.91, witness=True
→ approve   [rule 2: score < 0.22]

# Example 2: obvious fraud
fraud_score=0.85, days=47, prior_claims=6
→ reject    [rule 1: score > 0.78]

# Example 3: ambiguous, investigate with request_info first
fraud_score=0.58, units=2.0, remaining=4, no prior investigation
→ request_info   [100% accurate, cheapest, reveals key fields]

# Example 4: after request_info revealed high prior claims + long delay
fraud_score=0.79 (was 0.58) after request_info
→ reject    [rule 1 now applies]

# Example 5: after request_info, still ambiguous
fraud_score=0.63 (updated), units=1.5, remaining=3
→ document_audit   [85% check to resolve remaining ambiguity]

# Example 6: budget exhausted, uncertain
fraud_score=0.55, units=0.3, remaining=3
→ reject    [rule 3: budget < 0.5, lean reject on fraud_score >= 0.45]

# Example 7: hard mode, ambiguous but scarce budget
fraud_score=0.62, units=0.5, remaining=2, budget_per_claim=0.25
→ reject    [rule 4: end of episode + scarce budget + fraud_score >= 0.50]

# Example 8: mildly suspicious, budget available
fraud_score=0.38, units=1.5, remaining=5, budget_per_claim=0.30
→ approve   [rule 6: fraud_score < 0.50, budget_per_claim too low to investigate]

════════════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════════════
One word only. No explanation, no punctuation.
Valid: approve  reject  quick_check  document_audit  field_investigation  request_info"""


def llm_agent(obs: Observation, client: OpenAI, claim_history: List[str], invest_count: int, task: str) -> str:
    """Call the LLM with task-adaptive guardrails applied first."""

    forced = apply_guardrails(obs, invest_count, task)
    if forced is not None:
        return forced

    request_info_done = any("INFO" in h for h in claim_history)

    history_text = (
        "\n".join(f"  [{i+1}] {h}" for i, h in enumerate(claim_history[-3:]))
        if claim_history else "  (no investigations yet on this claim)"
    )

    budget_per_claim = (
        round(obs.investigation_units / obs.claims_remaining, 2)
        if obs.claims_remaining > 0 else 0.0
    )

    high = FRAUD_HIGH_THRESHOLD[task]
    low  = FRAUD_LOW_THRESHOLD[task]

    user_prompt = f"""\
Task: {task.upper()} mode

Current claim
─────────────
  fraud_score             : {obs.fraud_score:.3f}   ← primary signal (thresholds: reject>{high}, approve<{low})
  claim_amount_normalized : {obs.claim_amount_normalized:.3f}
  days_since_incident     : {obs.days_since_incident}
  num_prior_claims        : {obs.num_prior_claims}
  document_score          : {obs.document_score:.3f}
  witness_available       : {obs.witness_available}
  repair_estimate_match   : {obs.repair_estimate_match:.3f}

Budget
──────
  investigation_units  : {obs.investigation_units}
  claims_remaining     : {obs.claims_remaining}
  budget_per_claim     : {budget_per_claim}
  investigations_used  : {invest_count} / {MAX_INVEST_PER_CLAIM} max
  request_info_done    : {request_info_done}

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
            if use_llm:
                action = llm_agent(obs, client, claim_history, invest_count, task_name)
            else:
                action = agent_policy(obs)

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
