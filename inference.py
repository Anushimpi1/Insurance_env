import os
import sys
from typing import List, Optional

from openai import OpenAI

from insurance import (
    InsuranceEnvironment, InsuranceAction, InsuranceObservation,
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


FRAUD_SIGNAL_REJECT    = {"easy": 4, "medium": 4, "hard": 3}
GENUINE_SIGNAL_APPROVE = {"easy": 4, "medium": 4, "hard": 5}
MAX_INVEST_PER_CLAIM   = 3


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


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:+.2f}" for r in rewards)
    print(
        f"[END]   success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards=[{rewards_str}]",
        flush=True,
    )

def _count_signals(obs: InsuranceObservation) -> tuple[int, int]:
   
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
    obs: InsuranceObservation, invest_count: int, task: str
) -> Optional[str]:
    
    fraud_signals, genuine_signals = _count_signals(obs)
    total       = fraud_signals + genuine_signals
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

def llm_agent(
    obs: InsuranceObservation,
    client: OpenAI,
    invest_count: int,
    task: str,
) -> str:
   
    forced = apply_guardrails(obs, invest_count, task)
    if forced is not None:
        return forced

    budget_per_claim = round(
        obs.investigation_units / max(obs.claims_remaining, 1), 2
    )

    user_prompt = f"""\
Task: {task.upper()} mode
{"(65% fraud rate — lean toward reject on ambiguous cases)" if task == "hard" else ""}

Current claim
─────────────
  claim_amount_normalized  : {obs.claim_amount_normalized:.3f}
  days_since_incident      : {obs.days_since_incident}
  num_prior_claims         : {obs.num_prior_claims}
  document_score           : {obs.document_score:.3f}
  witness_available        : {obs.witness_available}
  repair_estimate_match    : {obs.repair_estimate_match:.3f}
  last_investigation_signal: {obs.last_investigation_signal or "none"}

Budget
──────
  investigation_units  : {obs.investigation_units}
  claims_remaining     : {obs.claims_remaining}
  budget_per_claim     : {budget_per_claim}
  investigations_done  : {obs.investigations_done} / {MAX_INVEST_PER_CLAIM} max

Count F and G signals, follow the DECISION PROCESS, reply with one word:"""

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
    
    env = InsuranceEnvironment(task=task_name)
    obs = env.reset()

    rewards: List[float] = []
    steps       = 0
    invest_count = 0
    agent_label = f"llm({MODEL_NAME})" if use_llm else "baseline"

    log_start(task_name, agent_label)

    try:
        while not obs.done and steps < MAX_STEPS:
            if use_llm:
                action = llm_agent(obs, client, invest_count, task_name)
            else:
                action = agent_policy(obs)

            obs = env.step(InsuranceAction(action=action))

            reward = obs.reward or 0.0
            rewards.append(reward)
            steps += 1

            log_step(
                step=steps, action=action, reward=reward,
                done=obs.done, error=None,
            )

            if obs.metadata.get("is_fraud") is not None:
                invest_count = 0
            else:
                invest_count += 1

        score   = GRADERS[task_name](env)
        success = score > 0.3

    except Exception as e:
        log_step(steps + 1, "error", 0.0, True, str(e))
        score   = 0.0
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
    print(f"\n{'=' * W}")
    print(f"  BENCHMARK RESULTS — {BENCHMARK}  (seed=42, 15 claims/episode)")
    print(f"{'=' * W}")
    print(
        f"  {'TASK':<8} {'AGENT':<14} {'SCORE':>6} {'REWARD':>8} "
        f"{'FRAUD DET':>10} {'CORRECT APP':>12} {'WRONG APP':>10}"
    )
    print(f"  {'-' * (W - 2)}")

    for i, (b, l) in enumerate(zip(baseline_results, llm_results)):
        for r in (b, l):
            fd          = f"{r['fraud_caught']}/{r['total_fraud']}"
            agent_short = "baseline" if "baseline" in r["agent"] else "llm"
            print(
                f"  {r['task']:<8} {agent_short:<14} {r['score']:>6.3f} "
                f"{r['total_reward']:>+8.1f} {fd:>10} "
                f"{r['correct_approvals']:>12} {r['wrong_approvals']:>10}"
            )
        if i < len(baseline_results) - 1:
            print(f"  {'.' * (W - 2)}")

    print(f"{'=' * W}")
    print("\n  Score delta (LLM vs Baseline)")
    print(f"  {'TASK':<10} {'BASELINE':>10} {'LLM':>10} {'DELTA':>10}   RESULT")
    print(f"  {'-' * 54}")
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
    run_tasks = [os.getenv("TASK")] if os.getenv("TASK") else TASKS
    run_agent = os.getenv("AGENT", "both").lower()

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
            print(f"  {r['task']}: score={r['score']:.3f}  reward={r['total_reward']:+.1f}")
    elif llm_results:
        print("\n[LLM-only run]")
        for r in llm_results:
            print(f"  {r['task']}: score={r['score']:.3f}  reward={r['total_reward']:+.1f}")


if __name__ == "__main__":
    main()
