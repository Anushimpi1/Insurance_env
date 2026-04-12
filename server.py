import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from insurance import (
    InsuranceEnv, Action,
    grade_easy, grade_medium, grade_hard,
)

app = FastAPI(
    title="Insurance Claim Adjudication Environment",
    description=(
        "An OpenEnv-compliant environment for insurance fraud triage. "
        "An AI agent acts as a senior claim adjuster — reasoning from raw claim signals "
        "(no pre-computed fraud score) to detect fraud, protect genuine claimants, "
        "and allocate scarce investigation budget strategically across a 15-claim caseload."
    ),
    version="2.0.0",
)

_env: InsuranceEnv = InsuranceEnv(task="easy", seed=42)

class ResetRequest(BaseModel):
    task: Optional[str] = "easy"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    action: str


@app.post("/reset", summary="Reset the environment")
def reset(req: ResetRequest = ResetRequest()):
    global _env
    if req.task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"Unknown task '{req.task}'. Choose: easy, medium, hard")
    _env = InsuranceEnv(task=req.task, seed=req.seed)
    obs = _env.reset()
    return {
        "observation": obs.model_dump(),
        "task": req.task,
        "seed": req.seed,
        "done": False,
        "message": f"Episode started — {_env.total_claims} claims to process",
    }


@app.post("/step", summary="Submit an action")
def step(req: StepRequest):
    """
    Valid actions: approve | reject | quick_check | document_audit | field_investigation | request_info
    """
    action_obj = Action(action=req.action)
    next_obs, reward_obj, done, info = _env.step(action_obj)

    response = {
        "observation": next_obs.model_dump(),
        "reward": reward_obj.value,
        "done": done,
        "info": info,
    }

    if done:
        graders = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}
        score = graders[_env.task](_env)
        response["episode_score"] = round(score, 4)
        response["message"] = (
            f"Episode complete. Score: {score:.3f} | "
            f"Fraud caught: {_env.fraud_caught}/{_env.total_fraud} | "
            f"Correct approvals: {_env.correct_approvals}"
        )

    return response


@app.get("/state", summary="Current observation")
def state():
    return {"observation": _env.state().model_dump()}


@app.get("/stats", summary="Episode statistics")
def stats():
    graders = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}
    score = graders[_env.task](_env)
    return {
        "task": _env.task,
        "claims_processed": _env.current_claim_idx,
        "claims_total": _env.total_claims,
        "investigation_units_remaining": _env.investigation_units,
        "correct_approvals": _env.correct_approvals,
        "wrong_approvals": _env.wrong_approvals,
        "fraud_caught": _env.fraud_caught,
        "total_fraud_seen": _env.total_fraud,
        "genuine_rejected": _env.genuine_rejected,
        "total_reward": round(_env.total_reward, 2),
        "current_score": round(score, 4),
    }


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "env": "insurance_env", "version": "2.0.0"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
