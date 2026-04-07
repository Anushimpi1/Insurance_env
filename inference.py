"""
Inference & Evaluation Script — Insurance Claim Adjudication under Uncertainty
===============================================================================
Runs both the LLM agent and the rule-based baseline on every task, then prints
a side-by-side comparison table so judges can see concrete improvement numbers.

Required environment variables
-------------------------------
  API_BASE_URL   LLM API endpoint  (default: HuggingFace router)
  MODEL_NAME     Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       API key           (also checked as API_KEY)

Usage
-----
  python inference.py                    # runs all tasks, both agents
  TASK=medium python inference.py        # single task only
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

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") 

TASKS      = ["easy", "medium", "hard"]
BENCHMARK  = "insurance_env"
MAX_STEPS  = 60    

GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}

def log_start(task: str, agent: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPT = """\
You are a senior insurance claim adjuster at a large insurance company. Your job is to \
process a caseload of 10 claims as efficiently as possible.

──────────────────────────────────────────────────────────────
OBSERVATION FIELDS
──────────────────────────────────────────────────────────────
• claim_amount_normalized   Claim size as fraction of policy limit (0=small, 1=maximum)
• days_since_incident       Days between incident date and filing date (long delays are suspicious)
• num_prior_claims          Number of previous claims by this claimant (high = suspicious)
• document_score            Quality & authenticity of submitted documents (0=poor, 1=excellent)
• witness_available         Whether an independent witness has corroborated the incident
• repair_estimate_match     How closely the repair estimate matches the claimed amount (0=mismatch)
• fraud_score               Current Bayesian fraud probability posterior (0=likely genuine, 1=likely fraud)
• investigation_units       Your remaining investigation budget (shared across ALL remaining claims)
• claims_remaining          How many claims (including this one) are left to process

──────────────────────────────────────────────────────────────
AVAILABLE ACTIONS & COSTS
──────────────────────────────────────────────────────────────
approve            (+5 if genuine | −10 if fraud)    — terminal: closes this claim
reject             (+4 if fraud   | −6  if genuine)  — terminal: closes this claim
quick_check        costs 0.5 units, 70% accurate fraud signal
document_audit     costs 1.0 unit,  85% accurate fraud signal
field_investigation costs 2.0 units, 95% accurate fraud signal  ← best for high-stakes ambiguity
request_info       costs 0.5 units, reveals exact prior_claims, days_since_incident, repair_match

──────────────────────────────────────────────────────────────
DECISION STRATEGY
──────────────────────────────────────────────────────────────
1. Use fraud_score as your primary signal — it accumulates evidence from prior investigations.
2. If fraud_score > 0.75 → reject without further investigation.
3. If fraud_score < 0.25 → approve without further investigation.
4. In the ambiguous zone (0.25–0.75), investigate — but only if budget allows.
5. Budget discipline: you have investigation_units for ALL remaining claims.
   Do not exhaust budget on early claims; save some for difficult ones.
6. Approve high-value genuine claims quickly — every delay costs 0.5 extra.
7. Maximum 4 investigation actions per claim — then you must decide.

──────────────────────────────────────────────────────────────
OUTPUT FORMAT
──────────────────────────────────────────────────────────────
Respond with EXACTLY one word from:
  approve  reject  quick_check  document_audit  field_investigation  request_info

Nothing else — no explanation, no punctuation."""


def llm_agent(obs: Observation, client: OpenAI, claim_history: List[str]) -> str:
    """Call the LLM and return a valid action string."""
    history_text = "\n".join(f"  • {h}" for h in claim_history[-3:]) if claim_history else "  (none yet)"

    user_prompt = f"""\
Current claim signals
─────────────────────
  claim_amount_normalized : {obs.claim_amount_normalized:.3f}
  days_since_incident     : {obs.days_since_incident}
  num_prior_claims        : {obs.num_prior_claims}
  document_score          : {obs.document_score:.3f}
  witness_available       : {obs.witness_available}
  repair_estimate_match   : {obs.repair_estimate_match:.3f}
  fraud_score             : {obs.fraud_score:.3f}   ← key signal
  investigation_units     : {obs.investigation_units}
  claims_remaining        : {obs.claims_remaining}

Investigation results so far on this claim
──────────────────────────────────────────
{history_text}

Your action (one word):"""

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
        print(f"[WARN] LLM call failed ({e}), falling back to baseline", flush=True)
        return agent_policy(obs)

    valid = {"approve", "reject", "quick_check", "document_audit", "field_investigation", "request_info"}
    for action in valid:
        if action in raw:
            return action

    print(f"[WARN] LLM returned unparseable output '{raw}', using baseline policy", flush=True)
    return agent_policy(obs)


def run_episode(task_name: str, use_llm: bool, client: Optional[OpenAI] = None) -> dict:
    """
    Run a full episode and return a result dict with score, steps, rewards, etc.
    """
    env = InsuranceEnv(task=task_name)
    obs = env.reset()

    rewards: List[float] = []
    steps = 0
    done = False
    claim_history: List[str] = []
    agent_label = f"llm({MODEL_NAME})" if use_llm else "baseline"

    log_start(task_name, agent_label)

    try:
        while not done and steps < MAX_STEPS:
            if use_llm:
                action = llm_agent(obs, client, claim_history)
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
            elif explanation:
                claim_history.append(explanation)

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
    sep = "─" * 82
    print(f"\n{'═' * 82}")
    print(f"  BENCHMARK RESULTS — {BENCHMARK}  (seed=42)")
    print(f"{'═' * 82}")
    print(f"  {'TASK':<8} {'AGENT':<12} {'SCORE':>6} {'REWARD':>8} {'FRAUD DET':>10} {'CORRECT APP':>12} {'WRONG APP':>10}")
    print(sep)

    all_results = []
    for b, l in zip(baseline_results, llm_results):
        all_results.extend([b, l])

    for i, r in enumerate(all_results):
        fd = f"{r['fraud_caught']}/{r['total_fraud']}"
        agent_short = "baseline" if r["agent"] == "baseline" else "llm"
        print(
            f"  {r['task']:<8} {agent_short:<12} {r['score']:>6.3f} "
            f"{r['total_reward']:>+8.1f} {fd:>10} {r['correct_approvals']:>12} {r['wrong_approvals']:>10}"
        )
        if (i + 1) % 2 == 0 and i < len(all_results) - 1:
            print(f"  {sep}")

    print(f"{'═' * 82}")


    print("\n  Score delta (LLM vs Baseline)")
    print(f"  {'TASK':<10} {'BASELINE':>10} {'LLM':>10} {'DELTA':>10} {'RESULT'}")
    print("  " + "─" * 50)
    for b, l in zip(baseline_results, llm_results):
        delta = l["score"] - b["score"]
        result = "✓ BETTER" if delta > 0.01 else ("~ SIMILAR" if abs(delta) <= 0.01 else "✗ WORSE")
        print(f"  {b['task']:<10} {b['score']:>10.3f} {l['score']:>10.3f} {delta:>+10.3f}   {result}")
    print()



def main() -> None:
    run_tasks = [os.getenv("TASK")] if os.getenv("TASK") else TASKS
    run_agent  = os.getenv("AGENT", "both").lower()   

    if run_agent in ("llm", "both") and not API_KEY:
        print(
            "[ERROR] No API key found. Set HF_TOKEN or API_KEY environment variable.\n"
            "        To run baseline only: AGENT=baseline python inference.py",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if run_agent in ("llm", "both") else None

    baseline_results: List[dict] = []
    llm_results: List[dict] = []

    for task in run_tasks:
        print(f"\n{'#' * 60}")
        print(f"# TASK: {task.upper()}")
        print(f"{'#' * 60}\n")

        if run_agent in ("baseline", "both"):
            print("--- BASELINE AGENT ---")
            result = run_episode(task, use_llm=False)
            baseline_results.append(result)

        if run_agent in ("llm", "both"):
            print("\n--- LLM AGENT ---")
            result = run_episode(task, use_llm=True, client=client)
            llm_results.append(result)

    if baseline_results and llm_results:
        print_comparison_table(baseline_results, llm_results)
    elif baseline_results:
        print("\n[Baseline-only run complete]")
        for r in baseline_results:
            print(f"  {r['task']}: score={r['score']:.3f}  reward={r['total_reward']:+.1f}")
    elif llm_results:
        print("\n[LLM-only run complete]")
        for r in llm_results:
            print(f"  {r['task']}: score={r['score']:.3f}  reward={r['total_reward']:+.1f}")


if __name__ == "__main__":
    main()
