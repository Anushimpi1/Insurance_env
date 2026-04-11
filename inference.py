
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
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

TASKS     = ["easy", "medium", "hard"]
BENCHMARK = "insurance_env"
MAX_STEPS = 90   

GRADERS = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}


FRAUD_SIGNAL_REJECT = {"easy": 4, "medium": 4, "hard": 3}
GENUINE_SIGNAL_APPROVE = {"easy": 4, "medium": 4, "hard": 5}
MAX_INVEST_PER_CLAIM = 3



def log_start(task: str, agent: str) -> None:
    print(
        f"[START] task={task} env={BENCHMARK} agent={agent} model={MODEL_NAME}",
        flush=True,
    )


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    print(
        f"[STEP]  step={step:02d} action={action:<22s} "
        f"reward={reward:+.2f} done={str(done).lower()} "
        f"error={error or 'null'}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:+.2f}" for r in rewards)
    print(
        f"[END]   success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards=[{rewards_str}]",
        flush=True,
    )


def _count_signals(obs: Observation) -> tuple[int, int]:
    """Count weighted fraud and genuine signals from raw observation."""
    fraud_signals = sum([
        obs.claim_amount_normalized > 0.75,
        obs.days_since_incident > 30,
        obs.num_prior_claims >= 4,
        obs.document_score < 0.35,
        not obs.witness_available,
        obs.repair_estimate_match < 0.3,
    ])
    genuine_signals = sum([
        obs.document_score > 0.80,
        obs.witness_available,
        obs.repair_estimate_match > 0.80,
        obs.days_since_incident <= 3,
        obs.num_prior_claims == 0,
    ])

    if obs.last_investigation_signal == "suspicious":
        fraud_signals += 2
    elif obs.last_investigation_signal == "looks_genuine":
        genuine_signals += 2
    elif obs.last_investigation_signal == "info_revealed":
        if fraud_signals > genuine_signals:
            fraud_signals += 1
        else:
            genuine_signals += 1

    return fraud_signals, genuine_signals


def apply_guardrails(
    obs: Observation, invest_count: int, task: str
) -> Optional[str]:
    
    fraud_signals, genuine_signals = _count_signals(obs)
    total = fraud_signals + genuine_signals
    fraud_ratio = fraud_signals / total if total > 0 else 0.5

    reject_threshold  = FRAUD_SIGNAL_REJECT[task]
    approve_threshold = GENUINE_SIGNAL_APPROVE[task]


    if fraud_signals >= reject_threshold:
        return "reject"

   
    if genuine_signals >= approve_threshold and fraud_signals <= 1:
        return "approve"

    if obs.investigation_units < 0.5:
        return "reject" if fraud_ratio >= 0.45 else "approve"


    if invest_count >= MAX_INVEST_PER_CLAIM:
        return "reject" if fraud_ratio >= 0.45 else "approve"

    
    if obs.claims_remaining <= 2 and obs.investigation_units < 1.5:
        return "reject" if fraud_ratio >= 0.45 else "approve"

    return None  

SYSTEM_PROMPT = """\
You are a senior insurance claim adjuster processing a caseload of 15 claims.
Your goal: maximise total reward across the ENTIRE caseload, not just the current claim.

You have NO access to a pre-computed fraud score. You must reason from raw signals,
exactly as a real adjuster would.

════════════════════════════════════════════════
ASYMMETRIC COSTS — memorise this
════════════════════════════════════════════════
  approve genuine  = +5     approve fraud   = −10   ← WORST possible outcome
  reject  fraud    = +4     reject  genuine = −6

  Wrong approval costs 2× more than wrong rejection.
  When in doubt and budget is gone → REJECT.

════════════════════════════════════════════════
FRAUD SIGNALS (each one increases suspicion)
════════════════════════════════════════════════
  claim_amount_normalized > 0.75   High claim relative to policy limit
  days_since_incident > 30         Suspicious filing delay
  num_prior_claims >= 4            Serial claimant history
  document_score < 0.35            Poor or forged documents
  witness_available = False        No corroboration
  repair_estimate_match < 0.3      Repair estimate doesn't match claim

GENUINE SIGNALS (each one reduces suspicion)
════════════════════════════════════════════════
  document_score > 0.80            Strong authentic documents
  witness_available = True         Independent corroboration
  repair_estimate_match > 0.80     Estimates align with claim
  days_since_incident <= 3         Filed immediately
  num_prior_claims = 0             Clean history

════════════════════════════════════════════════
ACTIONS
════════════════════════════════════════════════
  approve             terminal  +5 genuine / −10 fraud
  reject              terminal  +4 fraud   / −6  genuine
  request_info        0.5 units  reveals EXACT values of:
                                 num_prior_claims, days_since_incident,
                                 repair_estimate_match
                                 Use FIRST when ambiguous — cheapest and most informative
  quick_check         0.5 units  70% accurate noisy signal
  document_audit      1.0 unit   85% accurate noisy signal
  field_investigation 2.0 units  95% accurate — use only for high-value ambiguous claims

════════════════════════════════════════════════
DECISION PROCESS — follow in order
════════════════════════════════════════════════
Step 1  Count fraud signals (F) and genuine signals (G) from the raw features above.
Step 2  Check last_investigation_signal:
           'suspicious'    → add 2 to F
           'looks_genuine' → add 2 to G
           'info_revealed' → re-read the exact values now in the observation;
                             they replaced the noisy initial values
Step 3  Decide:
  F >= 4                        → reject   (clear fraud)
  G >= 4 and F <= 1             → approve  (clear genuine)
  budget < 0.5                  → reject if F >= G else approve
  claims_remaining <= 2         → reject if F >= G else approve
  investigations_done == 0
    AND budget_per_claim >= 0.5 → request_info   (best first action)
  investigations_done >= 1
    AND budget >= 1.0           → document_audit  (if still ambiguous)
  otherwise                     → reject if F >= G else approve

════════════════════════════════════════════════
BUDGET DISCIPLINE
════════════════════════════════════════════════
  budget_per_claim = investigation_units / claims_remaining
  Never spend > 1.5× budget_per_claim on a single claim.
  field_investigation (2.0 units) is almost never justified.

════════════════════════════════════════════════
HARD MODE WARNING
════════════════════════════════════════════════
  In hard mode, 65% of claims are fraudulent. Sophisticated fraudsters
  deliberately mimic genuine claimants — good documents, moderate amounts,
  quick filings. Surface signals are unreliable. Use request_info to
  get exact values before deciding, but budget is only 2.5 units total.
  Lean toward reject on ambiguous cases.

════════════════════════════════════════════════
FEW-SHOT EXAMPLES
════════════════════════════════════════════════
# Example 1: obvious fraud
claim_amount=0.91, days=47, prior_claims=6, doc_score=0.12, witness=False
F=5 (amount, delay, prior, doc, no_witness), G=0
→ reject   [F >= 4, clear fraud]

# Example 2: obvious genuine
claim_amount=0.22, days=2, prior_claims=0, doc_score=0.88, witness=True, repair=0.92
F=0, G=5 (doc, witness, repair, quick_filing, clean_history)
→ approve  [G >= 4, clear genuine]

# Example 3: ambiguous, no investigation yet
claim_amount=0.55, days=18, prior_claims=2, doc_score=0.58, witness=False, repair=0.55
F=1 (no_witness), G=0  — genuinely ambiguous
investigations_done=0, budget_per_claim=0.5
→ request_info  [first investigation, reveals exact values for F/G scoring]

# Example 4: after request_info revealed high prior claims + long delay
request_info returned: prior_claims=5, days=38, repair=0.31
F now = 4 (no_witness + prior_claims>=4 + days>30 + repair<0.3)
→ reject   [F >= 4]

# Example 5: after request_info, still genuinely ambiguous
request_info returned: prior_claims=1, days=12, repair=0.62
F=1, G=0  — still unclear
budget=1.5, budget_per_claim=0.75
→ document_audit  [85% accurate follow-up]

# Example 6: budget nearly gone, uncertain
claim_amount=0.55, F=2, G=1, investigation_units=0.3
→ reject   [budget < 0.5, F >= G, asymmetric lean]

════════════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════════════
Reply with ONE word only. No explanation. No punctuation.
Valid responses: approve  reject  request_info  quick_check  document_audit  field_investigation"""



def llm_agent(
    obs: Observation,
    client: OpenAI,
    invest_count: int,
    task: str,
) -> str:
    """Query the LLM with guardrails applied first."""

    forced = apply_guardrails(obs, invest_count, task)
    if forced is not None:
        return forced

    budget_per_claim = round(
        obs.investigation_units / max(obs.claims_remaining, 1), 2
    )

    user_prompt = f"""\
Task: {task.upper()} mode
{"(65% of claims are fraudulent — lean toward reject on ambiguous cases)" if task == "hard" else ""}

Current claim
─────────────
  claim_amount_normalized : {obs.claim_amount_normalized:.3f}
  days_since_incident     : {obs.days_since_incident}
  num_prior_claims        : {obs.num_prior_claims}
  document_score          : {obs.document_score:.3f}
  witness_available       : {obs.witness_available}
  repair_estimate_match   : {obs.repair_estimate_match:.3f}
  last_investigation_signal: {obs.last_investigation_signal or "none"}

Budget
──────
  investigation_units  : {obs.investigation_units}
  claims_remaining     : {obs.claims_remaining}
  budget_per_claim     : {budget_per_claim}
  investigations_done  : {obs.investigations_done} / {MAX_INVEST_PER_CLAIM} max

Count your F and G signals, then follow the DECISION PROCESS.
Reply with one word:"""

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

    valid = {
        "approve", "reject", "quick_check",
        "document_audit", "field_investigation", "request_info",
    }
    for action in valid:
        if action in raw:
            return action

    print(f"[WARN] Unparseable LLM output '{raw}', using baseline", flush=True)
    return agent_policy(obs)


def run_episode(
    task_name: str, use_llm: bool, client: Optional[OpenAI] = None
) -> dict:
    env = InsuranceEnv(task=task_name)
    obs = env.reset()

    rewards: List[float] = []
    steps = 0
    done = False
    invest_count = 0
    agent_label = f"llm({MODEL_NAME})" if use_llm else "baseline"

    log_start(task_name, agent_label)

    try:
        while not done and steps < MAX_STEPS:
            if use_llm:
                action = llm_agent(obs, client, invest_count, task_name)
            else:
                action = agent_policy(obs)

            next_obs, reward_obj, done, info = env.step(Action(action=action))
            reward = reward_obj.value
            rewards.append(reward)
            steps += 1

            log_step(
                step=steps, action=action, reward=reward,
                done=done, error=None,
            )

            if info.get("is_fraud") is not None:
              
                invest_count = 0
            else:
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
        "task":              task_name,
        "agent":             agent_label,
        "score":             score,
        "steps":             steps,
        "total_reward":      env.total_reward,
        "fraud_caught":      env.fraud_caught,
        "total_fraud":       env.total_fraud,
        "correct_approvals": env.correct_approvals,
        "wrong_approvals":   env.wrong_approvals,
        "genuine_rejected":  env.genuine_rejected,
        "success":           success,
    }

def print_comparison_table(
    baseline_results: List[dict], llm_results: List[dict]
) -> None:
    W = 88
    print(f"\n{'═' * W}")
    print(f"  BENCHMARK RESULTS — {BENCHMARK}  (seed=42, 15 claims/episode)")
    print(f"{'═' * W}")
    print(
        f"  {'TASK':<8} {'AGENT':<14} {'SCORE':>6} {'REWARD':>8} "
        f"{'FRAUD DET':>10} {'CORRECT APP':>12} {'WRONG APP':>10}"
    )
    print(f"  {'─' * (W - 2)}")

    for i, (b, l) in enumerate(zip(baseline_results, llm_results)):
        for r in (b, l):
            fd = f"{r['fraud_caught']}/{r['total_fraud']}"
            agent_short = "baseline" if "baseline" in r["agent"] else "llm"
            print(
                f"  {r['task']:<8} {agent_short:<14} {r['score']:>6.3f} "
                f"{r['total_reward']:>+8.1f} {fd:>10} "
                f"{r['correct_approvals']:>12} {r['wrong_approvals']:>10}"
            )
        if i < len(baseline_results) - 1:
            print(f"  {'·' * (W - 2)}")

    print(f"{'═' * W}")
    print("\n  Score delta (LLM vs Baseline)")
    print(f"  {'TASK':<10} {'BASELINE':>10} {'LLM':>10} {'DELTA':>10}   RESULT")
    print(f"  {'─' * 54}")
    for b, l in zip(baseline_results, llm_results):
        delta  = l["score"] - b["score"]
        result = (
            "BETTER"  if delta >  0.01 else
            "SIMILAR" if abs(delta) <= 0.01 else
            "WORSE"
        )
        print(
            f"  {b['task']:<10} {b['score']:>10.3f} "
            f"{l['score']:>10.3f} {delta:>+10.3f}   {result}"
        )
    print()

def main() -> None:
    run_tasks  = [os.getenv("TASK")] if os.getenv("TASK") else TASKS
    run_agent  = os.getenv("AGENT", "both").lower()

    if run_agent in ("llm", "both") and not API_KEY:
        print(
            "[ERROR] No API key found. Set API_KEY or HF_TOKEN.\n"
            "        Baseline only: AGENT=baseline python inference.py",
            file=sys.stderr,
        )
        sys.exit(1)

    client = (
        OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        if run_agent in ("llm", "both") else None
    )

    baseline_results: List[dict] = []
    llm_results:      List[dict] = []

    for task in run_tasks:
        print(f"\n{'#' * 64}")
        print(f"# TASK: {task.upper()}")
        print(f"{'#' * 64}\n")

        if run_agent in ("baseline", "both"):
            print("--- BASELINE AGENT ---")
            baseline_results.append(run_episode(task, use_llm=False))

        if run_agent in ("llm", "both"):
            print("\n--- LLM AGENT ---")
            llm_results.append(
                run_episode(task, use_llm=True, client=client)
            )

    if baseline_results and llm_results:
        print_comparison_table(baseline_results, llm_results)
    elif baseline_results:
        print("\n[Baseline-only run]")
        for r in baseline_results:
            print(
                f"  {r['task']}: score={r['score']:.3f}  "
                f"reward={r['total_reward']:+.1f}"
            )
    elif llm_results:
        print("\n[LLM-only run]")
        for r in llm_results:
            print(
                f"  {r['task']}: score={r['score']:.3f}  "
                f"reward={r['total_reward']:+.1f}"
            )


if __name__ == "__main__":
    main()
