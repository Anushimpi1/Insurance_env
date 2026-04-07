"""
Insurance Claim Adjudication under Uncertainty
===============================================
An OpenEnv environment where an AI agent acts as a senior insurance claim
adjuster. The true fraud label is HIDDEN — the agent infers fraud likelihood
from realistic, noisy signals and must allocate scarce investigation resources
strategically across a caseload of 10 claims.

Design goals
------------
* Partial observability  — no direct fraud label; agent reasons from signals
* Resource management    — investigation budget shared across all claims
* Asymmetric costs       — approving fraud is far worse than over-investigating
* Difficulty progression — easy (clean signals) → hard (adversarial fraud)
"""

import random
from typing import Optional
from pydantic import BaseModel, Field

class Observation(BaseModel):
    claim_amount_normalized: float = Field(..., description="Claim size / policy limit (0–1)")
    days_since_incident: int        = Field(..., description="Filing delay in days; longer = more suspicious")
    num_prior_claims: int           = Field(..., description="Claimant's historical claim count")
    document_score: float           = Field(..., description="Document completeness & authenticity (0–1)")
    witness_available: bool         = Field(..., description="Corroborating witness present")
    repair_estimate_match: float    = Field(..., description="Repair estimate vs claim alignment (0–1)")
    fraud_score: float              = Field(..., description="Running Bayesian fraud posterior (0–1); updated by investigations")
    investigation_units: float      = Field(..., description="Remaining investigation budget")
    claims_remaining: int           = Field(..., description="Claims left in caseload")


class Action(BaseModel):
    action: str = Field(
        ...,
        description="One of: approve | reject | quick_check | document_audit | field_investigation | request_info",
    )


class Reward(BaseModel):
    value: float

class _Claim:
    __slots__ = (
        "is_fraud", "claim_amount_normalized", "days_since_incident",
        "num_prior_claims", "document_score", "witness_available",
        "repair_estimate_match",
    )

    def __init__(
        self,
        is_fraud: bool,
        claim_amount_normalized: float,
        days_since_incident: int,
        num_prior_claims: int,
        document_score: float,
        witness_available: bool,
        repair_estimate_match: float,
    ):
        self.is_fraud = is_fraud
        self.claim_amount_normalized = claim_amount_normalized
        self.days_since_incident = days_since_incident
        self.num_prior_claims = num_prior_claims
        self.document_score = document_score
        self.witness_available = witness_available
        self.repair_estimate_match = repair_estimate_match

VALID_ACTIONS = frozenset({
    "approve", "reject",
    "quick_check", "document_audit", "field_investigation", "request_info",
})

INVESTIGATION_COSTS: dict[str, float] = {
    "quick_check": 0.5,
    "document_audit": 1.0,
    "field_investigation": 2.0,
    "request_info": 0.5,
}

INVESTIGATION_ACCURACY: dict[str, float] = {
    "quick_check": 0.70,
    "document_audit": 0.85,
    "field_investigation": 0.95,
}

TASK_BUDGETS: dict[str, float] = {
    "easy": 6.0,
    "medium": 4.0,
    "hard": 2.5,
}

TASK_FRAUD_RATES: dict[str, float] = {
    "easy": 0.25,
    "medium": 0.45,
    "hard": 0.65,
}

TOTAL_CLAIMS = 10
MAX_STEPS_PER_CLAIM = 4   


REWARD_APPROVE_GENUINE   =  5.0
REWARD_APPROVE_FRAUD     = -10.0
REWARD_REJECT_FRAUD      =  4.0
REWARD_REJECT_GENUINE    = -6.0
PENALTY_HIGH_VALUE_DELAY = -0.5   
PENALTY_BUDGET_EXCEEDED  = -1.0  



class InsuranceEnv:
    """
    OpenEnv-compliant insurance claim adjudication environment.

    Episode flow
    ------------
    reset() → returns Observation for claim #1
    step(action) → (next_obs, Reward, done, info)
        Terminal actions (approve/reject) advance to the next claim.
        Investigation actions consume budget and return a noisy signal.
    Episode ends when all TOTAL_CLAIMS have been resolved.
    """

    def __init__(self, task: str = "easy", seed: int = 42):
        if task not in TASK_BUDGETS:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_BUDGETS)}")
        self.task = task
        self.seed = seed
        self.total_claims = TOTAL_CLAIMS

        self.current_claim_idx: int = 0
        self.investigation_units: float = TASK_BUDGETS[task]
        self._current_claim: Optional[_Claim] = None
        self._rng = random.Random(seed)
        self._steps_on_current_claim: int = 0
        self._fraud_posterior: float = TASK_FRAUD_RATES[task]  

        self.correct_approvals: int = 0
        self.wrong_approvals: int = 0
        self.fraud_caught: int = 0
        self.genuine_rejected: int = 0
        self.total_fraud: int = 0      
        self.total_reward: float = 0.0

    def reset(self) -> Observation:
        self.current_claim_idx = 0
        self.investigation_units = TASK_BUDGETS[self.task]
        self._rng = random.Random(self.seed)
        self._steps_on_current_claim = 0
        self._fraud_posterior = TASK_FRAUD_RATES[self.task]

        self.correct_approvals = 0
        self.wrong_approvals = 0
        self.fraud_caught = 0
        self.genuine_rejected = 0
        self.total_fraud = 0
        self.total_reward = 0.0

        self._current_claim = self._generate_claim()
        return self._make_observation()

    def step(self, action: Action):  
        """
        Process one action.

        Returns
        -------
        (Observation, Reward, done: bool, info: dict)
        """
        action_str = action.action.strip().lower()

        if action_str not in VALID_ACTIONS:
            return (
                self._make_observation(),
                Reward(value=-3.0),
                False,
                {"explanation": f"Invalid action '{action_str}'. Valid: {sorted(VALID_ACTIONS)}",
                 "is_fraud": None},
            )

        claim = self._current_claim
        reward = 0.0
        explanation = ""
        advance_claim = False

        if action_str == "approve":
            advance_claim = True
            if not claim.is_fraud:
                reward += REWARD_APPROVE_GENUINE
                self.correct_approvals += 1
                explanation = "✓ Correctly approved genuine claim"
            else:
                reward += REWARD_APPROVE_FRAUD
                self.wrong_approvals += 1
                explanation = "✗ Approved fraudulent claim — significant financial loss"

        elif action_str == "reject":
            advance_claim = True
            if claim.is_fraud:
                reward += REWARD_REJECT_FRAUD
                self.fraud_caught += 1
                explanation = "✓ Correctly rejected fraudulent claim"
            else:
                reward += REWARD_REJECT_GENUINE
                self.genuine_rejected += 1
                explanation = "✗ Wrongly rejected genuine claim — customer harm & legal risk"

        elif action_str in INVESTIGATION_COSTS:
            
            if self._steps_on_current_claim >= MAX_STEPS_PER_CLAIM:
                reward -= 1.0
                explanation = (
                    f"Investigation cap reached ({MAX_STEPS_PER_CLAIM} steps per claim). "
                    "You must approve or reject."
                )
            else:
                cost = INVESTIGATION_COSTS[action_str]
                if self.investigation_units >= cost:
                    self.investigation_units = round(self.investigation_units - cost, 2)
                    reward -= cost

                    if action_str == "request_info":
                        
                        explanation = (
                            f"[INFO] prior_claims={claim.num_prior_claims}, "
                            f"days_since_incident={claim.days_since_incident}, "
                            f"repair_match={claim.repair_estimate_match:.2f} | "
                            f"Units left: {self.investigation_units}"
                        )
                        
                        self._fraud_posterior = self._update_posterior_from_info(claim)
                    else:
                        accuracy = INVESTIGATION_ACCURACY[action_str]
                        signal_is_fraud = (
                            claim.is_fraud
                            if self._rng.random() < accuracy
                            else not claim.is_fraud
                        )
                       
                        self._fraud_posterior = self._bayesian_update(
                            self._fraud_posterior, accuracy, signal_is_fraud
                        )
                        label = "suspicious" if signal_is_fraud else "looks genuine"
                        tool_labels = {
                            "quick_check": f"Quick check ({int(accuracy*100)}% reliable)",
                            "document_audit": f"Document audit ({int(accuracy*100)}% reliable)",
                            "field_investigation": f"Field investigation ({int(accuracy*100)}% reliable)",
                        }
                        explanation = (
                            f"[{tool_labels[action_str]}] Signal: {label} | "
                            f"Fraud posterior: {self._fraud_posterior:.2f} | "
                            f"Units left: {self.investigation_units}"
                        )
                else:
                    reward += PENALTY_BUDGET_EXCEEDED
                    explanation = (
                        f"Insufficient budget for {action_str} "
                        f"(costs {cost}, have {self.investigation_units:.1f})"
                    )

        
        if not advance_claim and claim.claim_amount_normalized >= 0.75:
            reward += PENALTY_HIGH_VALUE_DELAY
            explanation += " | ⚠ High-value claim delay penalty"

        self.total_reward += reward
        self._steps_on_current_claim += 1

        if advance_claim:
            if claim.is_fraud:
                self.total_fraud += 1         
            self.current_claim_idx += 1
            self._steps_on_current_claim = 0
            self._fraud_posterior = TASK_FRAUD_RATES[self.task]   
            if self.current_claim_idx < self.total_claims:
                self._current_claim = self._generate_claim()

        done = self.current_claim_idx >= self.total_claims

        return (
            self._make_observation(),
            Reward(value=reward),
            done,
            {
                "explanation": explanation,
                "is_fraud": claim.is_fraud if advance_claim else None,
                "fraud_posterior": round(self._fraud_posterior, 3),
                "steps_on_claim": self._steps_on_current_claim,
                "investigation_units_remaining": self.investigation_units,
            },
        )

    def state(self) -> Observation:
        return self._make_observation()

    def _make_observation(self) -> Observation:
        c = self._current_claim
        return Observation(
            claim_amount_normalized=round(c.claim_amount_normalized, 3),
            days_since_incident=c.days_since_incident,
            num_prior_claims=c.num_prior_claims,
            document_score=round(c.document_score, 3),
            witness_available=c.witness_available,
            repair_estimate_match=round(c.repair_estimate_match, 3),
            fraud_score=round(self._fraud_posterior, 3),
            investigation_units=self.investigation_units,
            claims_remaining=self.total_claims - self.current_claim_idx,
        )

    @staticmethod
    def _bayesian_update(prior: float, accuracy: float, signal_is_fraud: bool) -> float:
        """
        Update fraud posterior with a noisy binary signal using Bayes' theorem.

        P(fraud | signal=fraud) = P(signal=fraud | fraud) * P(fraud)
                                  / P(signal=fraud)
        """
        if signal_is_fraud:
            
            numerator = accuracy * prior
            denominator = accuracy * prior + (1 - accuracy) * (1 - prior)
        else:
           
            numerator = (1 - accuracy) * prior
            denominator = (1 - accuracy) * prior + accuracy * (1 - prior)

        if denominator == 0:
            return prior
        return max(0.0, min(1.0, numerator / denominator))

    @staticmethod
    def _update_posterior_from_info(claim: _Claim) -> float:
        """
        Heuristic posterior update from exact claimant info (request_info action).
        Stronger fraud signals push the posterior higher.
        """
        score = 0.0
        score += 0.15 * min(claim.num_prior_claims / 5.0, 1.0)
        score += 0.15 * min(claim.days_since_incident / 30.0, 1.0)
        score += 0.20 * (1.0 - claim.repair_estimate_match)
        return max(0.05, min(0.95, score + 0.3)) 

    def _generate_claim(self) -> _Claim:
        rng = self._rng
        task = self.task

        if task == "easy":
          
            is_fraud = rng.random() < TASK_FRAUD_RATES["easy"]
            if is_fraud:
                return _Claim(
                    is_fraud=True,
                    claim_amount_normalized=rng.uniform(0.7, 1.0),
                    days_since_incident=rng.randint(20, 60),
                    num_prior_claims=rng.randint(3, 8),
                    document_score=rng.uniform(0.0, 0.4),
                    witness_available=False,
                    repair_estimate_match=rng.uniform(0.0, 0.35),
                )
            return _Claim(
                is_fraud=False,
                claim_amount_normalized=rng.uniform(0.1, 0.5),
                days_since_incident=rng.randint(0, 7),
                num_prior_claims=rng.randint(0, 1),
                document_score=rng.uniform(0.75, 1.0),
                witness_available=rng.random() < 0.7,
                repair_estimate_match=rng.uniform(0.8, 1.0),
            )

        elif task == "medium":
            
            is_fraud = rng.random() < TASK_FRAUD_RATES["medium"]
            noise = rng.uniform(-0.2, 0.2)
            if is_fraud:
                return _Claim(
                    is_fraud=True,
                    claim_amount_normalized=min(1.0, rng.uniform(0.4, 0.9) + noise),
                    days_since_incident=rng.randint(5, 40),
                    num_prior_claims=rng.randint(1, 6),
                    document_score=max(0.0, rng.uniform(0.2, 0.65) + noise),
                    witness_available=rng.random() < 0.3,
                    repair_estimate_match=max(0.0, rng.uniform(0.2, 0.6) + noise),
                )
            return _Claim(
                is_fraud=False,
                claim_amount_normalized=min(1.0, rng.uniform(0.2, 0.7) + noise),
                days_since_incident=rng.randint(0, 20),
                num_prior_claims=rng.randint(0, 3),
                document_score=max(0.0, rng.uniform(0.5, 0.95) + noise),
                witness_available=rng.random() < 0.5,
                repair_estimate_match=max(0.0, rng.uniform(0.6, 1.0) + noise),
            )

        else:  
            is_fraud = rng.random() < TASK_FRAUD_RATES["hard"]
            if is_fraud:
                
                return _Claim(
                    is_fraud=True,
                    claim_amount_normalized=rng.uniform(0.3, 0.75),
                    days_since_incident=rng.randint(1, 15),
                    num_prior_claims=rng.randint(0, 4),
                    document_score=rng.uniform(0.55, 0.90),
                    witness_available=rng.random() < 0.45,
                    repair_estimate_match=rng.uniform(0.5, 0.85),
                )
            return _Claim(
                is_fraud=False,
                claim_amount_normalized=rng.uniform(0.1, 0.9),
                days_since_incident=rng.randint(0, 30),
                num_prior_claims=rng.randint(0, 5),
                document_score=rng.uniform(0.4, 1.0),
                witness_available=rng.random() < 0.5,
                repair_estimate_match=rng.uniform(0.5, 1.0),
            )

def grade_easy(env: InsuranceEnv) -> float:
    """Fraction of genuine claims correctly approved. Clamped to open interval (0, 1)."""
    total_genuine = env.total_claims - env.total_fraud
    if total_genuine == 0:
        raw = 0.5
    else:
        raw = env.correct_approvals / total_genuine
    return max(0.001, min(0.999, raw))


def grade_medium(env: InsuranceEnv) -> float:
    """Balanced accuracy: rewards correct decisions, penalises costly mistakes."""
    raw = (
        env.correct_approvals + env.fraud_caught - env.wrong_approvals * 0.5
    ) / env.total_claims
    return max(0.001, min(0.999, raw))


def grade_hard(env: InsuranceEnv) -> float:
    """Normalised total reward. Perfect play achieves ~65 reward → score 1.0."""
    return max(0.001, min(0.999, env.total_reward / 65.0))

def agent_policy(obs: Observation) -> str:
    """
    Counts weighted fraud vs. genuine signals and decides:
    - Obvious fraud (≥4 signals)  → reject immediately
    - Obvious genuine (≥4 signals) → approve immediately
    - Ambiguous → investigate (cheapest tool that fits budget)
    - Budget exhausted → majority-vote on signals
    """
    units = obs.investigation_units

    fraud_signals = sum([
        obs.claim_amount_normalized > 0.75,
        obs.days_since_incident > 30,
        obs.num_prior_claims >= 4,
        obs.document_score < 0.35,
        not obs.witness_available,
        obs.repair_estimate_match < 0.3,
        obs.fraud_score > 0.7,          
    ])

    genuine_signals = sum([
        obs.document_score > 0.8,
        obs.witness_available,
        obs.repair_estimate_match > 0.8,
        obs.days_since_incident <= 3,
        obs.num_prior_claims == 0,
        obs.fraud_score < 0.3,
    ])

    if fraud_signals >= 4:
        return "reject"
    if genuine_signals >= 4:
        return "approve"

    
    if units >= 2.0:
        return "field_investigation"
    elif units >= 1.0:
        return "document_audit"
    elif units >= 0.5:
        return "quick_check"

    return "reject" if fraud_signals >= genuine_signals else "approve"

if __name__ == "__main__":

    for task_name in ["easy", "medium", "hard"]:
        env = InsuranceEnv(task=task_name)
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < 60:
            act = agent_policy(obs)
            obs, reward, done, info = env.step(Action(action=act))
            steps += 1

        graders = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}
        score = graders[task_name](env)

        print(
            f"[{task_name.upper():6s}] score={score:.3f}  reward={env.total_reward:+.1f}  "
            f"steps={steps}  fraud_caught={env.fraud_caught}/{env.total_fraud}  "
            f"correct_approvals={env.correct_approvals}  "
            f"wrong_approvals={env.wrong_approvals}  genuine_rejected={env.genuine_rejected}"
        )
