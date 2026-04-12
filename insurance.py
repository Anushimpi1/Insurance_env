import random
from typing import Optional
from pydantic import BaseModel, Field

class Observation(BaseModel):
    
    claim_amount_normalized: float = Field(
        ..., description="Claim size / policy limit (0-1). Higher = larger financial exposure."
    )
    days_since_incident: int = Field(
        ..., description="Days between incident and filing. >30 days is suspicious."
    )
    num_prior_claims: int = Field(
        ..., description="Claimant's historical claim count. >=4 is suspicious."
    )
    document_score: float = Field(
        ..., description="Document completeness and authenticity (0-1). <0.4 is suspicious."
    )
    witness_available: bool = Field(
        ..., description="Whether a corroborating witness has been identified."
    )
    repair_estimate_match: float = Field(
        ..., description="Repair estimate vs claimed amount alignment (0-1). <0.4 is suspicious."
    )
    last_investigation_signal: Optional[str] = Field(
        None,
        description=(
            "Result of the most recent investigation on this claim, or null if none yet. "
            "Values: 'suspicious' | 'looks_genuine' | 'info_revealed'. "
            "Use this to update your belief about fraud likelihood."
        )
    )
    investigation_units: float = Field(
        ..., description="Remaining investigation budget shared across all remaining claims."
    )
    investigations_done: int = Field(
        ..., description="Number of investigations already done on this claim (max 3)."
    )
    claims_remaining: int = Field(
        ..., description="Claims still to process including the current one."
    )


class Action(BaseModel):
    action: str = Field(
        ...,
        description=(
            "One of: approve | reject | "
            "quick_check | document_audit | field_investigation | request_info"
        ),
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
    "quick_check":         0.5,
    "document_audit":      1.0,
    "field_investigation": 2.0,
    "request_info":        0.5,
}

INVESTIGATION_ACCURACY: dict[str, float] = {
    "quick_check":         0.70,
    "document_audit":      0.85,
    "field_investigation": 0.95,
}

TASK_BUDGETS: dict[str, float] = {
    "easy":   6.0,
    "medium": 4.0,
    "hard":   2.5,
}

TASK_FRAUD_RATES: dict[str, float] = {
    "easy":   0.25,
    "medium": 0.45,
    "hard":   0.65,
}

TOTAL_CLAIMS = 15
MAX_STEPS_PER_CLAIM = 4

REWARD_APPROVE_GENUINE  =  5.0
REWARD_APPROVE_FRAUD    = -10.0
REWARD_REJECT_FRAUD     =  4.0
REWARD_REJECT_GENUINE   = -6.0
PENALTY_BUDGET_EXCEEDED = -1.0
HARD_REWARD_NORMALIZER = 60.0


class InsuranceEnv:
    
    def __init__(self, task: str = "easy", seed: int = 42):
        if task not in TASK_BUDGETS:
            raise ValueError(
                f"Unknown task '{task}'. Choose from: {list(TASK_BUDGETS)}"
            )
        self.task = task
        self.seed = seed
        self.total_claims = TOTAL_CLAIMS

        self.current_claim_idx: int = 0
        self.investigation_units: float = TASK_BUDGETS[task]
        self._current_claim: Optional[_Claim] = None
        self._rng = random.Random(seed)
        self._steps_on_current_claim: int = 0
        self._fraud_posterior: float = TASK_FRAUD_RATES[task]  

        self._last_signal: Optional[str] = None

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
        self._last_signal = None

        self.correct_approvals = 0
        self.wrong_approvals = 0
        self.fraud_caught = 0
        self.genuine_rejected = 0
        self.total_fraud = 0
        self.total_reward = 0.0

        self._current_claim = self._generate_claim()
        return self._make_observation()

    def step(self, action: Action):
        
        action_str = action.action.strip().lower()

        if action_str not in VALID_ACTIONS:
            return (
                self._make_observation(),
                Reward(value=-3.0),
                False,
                {
                    "explanation": (
                        f"Invalid action '{action_str}'. "
                        f"Valid: {sorted(VALID_ACTIONS)}"
                    ),
                    "investigation_signal": None,
                    "is_fraud": None,
                    "steps_on_claim": self._steps_on_current_claim,
                    "investigation_units_remaining": self.investigation_units,
                },
            )

        claim = self._current_claim
        reward = 0.0
        explanation = ""
        investigation_signal = None
        advance_claim = False

        
        if action_str == "approve":
            advance_claim = True
            if not claim.is_fraud:
                reward = REWARD_APPROVE_GENUINE
                self.correct_approvals += 1
                explanation = "Correctly approved genuine claim (+5)"
            else:
                reward = REWARD_APPROVE_FRAUD
                self.wrong_approvals += 1
                explanation = "Approved fraudulent claim — significant financial loss (-10)"

        elif action_str == "reject":
            advance_claim = True
            if claim.is_fraud:
                reward = REWARD_REJECT_FRAUD
                self.fraud_caught += 1
                explanation = "Correctly rejected fraudulent claim (+4)"
            else:
                reward = REWARD_REJECT_GENUINE
                self.genuine_rejected += 1
                explanation = "Wrongly rejected genuine claim — customer harm (-6)"

        
        elif action_str in INVESTIGATION_COSTS:

            if self._steps_on_current_claim >= MAX_STEPS_PER_CLAIM:
                reward = -1.0
                explanation = (
                    f"Investigation cap reached ({MAX_STEPS_PER_CLAIM} per claim). "
                    "You must approve or reject now."
                )

            else:
                cost = INVESTIGATION_COSTS[action_str]

                if self.investigation_units >= cost:
                    self.investigation_units = round(
                        self.investigation_units - cost, 2
                    )
                    reward = -cost

                    if action_str == "request_info":
                        investigation_signal = "info_revealed"
                        self._last_signal = "info_revealed"
                        self._fraud_posterior = self._update_posterior_from_info(
                            claim
                        )
                        explanation = (
                            f"[REQUEST INFO] Exact details revealed: "
                            f"prior_claims={claim.num_prior_claims}, "
                            f"days_since_incident={claim.days_since_incident}, "
                            f"repair_match={claim.repair_estimate_match:.2f} | "
                            f"Budget remaining: {self.investigation_units}"
                        )

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
                        signal_label = (
                            "suspicious" if signal_is_fraud else "looks_genuine"
                        )
                        investigation_signal = signal_label
                        self._last_signal = signal_label

                        tool_names = {
                            "quick_check":         f"Quick Check ({int(accuracy*100)}% reliable)",
                            "document_audit":      f"Document Audit ({int(accuracy*100)}% reliable)",
                            "field_investigation": f"Field Investigation ({int(accuracy*100)}% reliable)",
                        }
                        explanation = (
                            f"[{tool_names[action_str]}] Signal: {signal_label} | "
                            f"Budget remaining: {self.investigation_units}"
                        )

                else:
                    reward = PENALTY_BUDGET_EXCEEDED
                    explanation = (
                        f"Insufficient budget for {action_str} "
                        f"(costs {cost:.1f}, have {self.investigation_units:.1f}). "
                        "Penalty applied."
                    )

        self.total_reward += reward
        self._steps_on_current_claim += 1

        if advance_claim:
            if claim.is_fraud:
                self.total_fraud += 1
            self.current_claim_idx += 1
            self._steps_on_current_claim = 0
            self._fraud_posterior = TASK_FRAUD_RATES[self.task]
            self._last_signal = None
            if self.current_claim_idx < self.total_claims:
                self._current_claim = self._generate_claim()

        done = self.current_claim_idx >= self.total_claims

        return (
            self._make_observation(),
            Reward(value=reward),
            done,
            {
                "explanation": explanation,
                "investigation_signal": investigation_signal,
                "is_fraud": claim.is_fraud if advance_claim else None,
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
            last_investigation_signal=self._last_signal,
            investigation_units=self.investigation_units,
            investigations_done=self._steps_on_current_claim,
            claims_remaining=self.total_claims - self.current_claim_idx,
        )

    @staticmethod
    def _bayesian_update(
        prior: float, accuracy: float, signal_is_fraud: bool
    ) -> float:
        """Bayes' theorem update from a noisy binary investigation signal."""
        if signal_is_fraud:
            numerator   = accuracy * prior
            denominator = accuracy * prior + (1 - accuracy) * (1 - prior)
        else:
            numerator   = (1 - accuracy) * prior
            denominator = (1 - accuracy) * prior + accuracy * (1 - prior)

        if denominator == 0:
            return prior
        return max(0.0, min(1.0, numerator / denominator))

    def _update_posterior_from_info(self, claim: _Claim) -> float:
        """
        Bayesian posterior update from exact revealed feature values.

        Applies three sequential likelihood-ratio updates starting from the
        CURRENT posterior — preserving all prior investigation evidence.
        Does NOT reset to a flat prior, which was the original design flaw.
        """
        p = self._fraud_posterior

        if   claim.num_prior_claims >= 5: lr = 3.0
        elif claim.num_prior_claims >= 3: lr = 1.8
        elif claim.num_prior_claims == 0: lr = 0.5
        else:                             lr = 1.0
        p = (lr * p) / (lr * p + (1 - p))
        p = max(0.01, min(0.99, p))

        if   claim.days_since_incident > 40: lr = 2.5
        elif claim.days_since_incident > 20: lr = 1.6
        elif claim.days_since_incident <= 3: lr = 0.6
        else:                                lr = 1.0
        p = (lr * p) / (lr * p + (1 - p))
        p = max(0.01, min(0.99, p))

        if   claim.repair_estimate_match < 0.3:  lr = 2.8
        elif claim.repair_estimate_match < 0.5:  lr = 1.5
        elif claim.repair_estimate_match > 0.85: lr = 0.5
        else:                                    lr = 1.0
        p = (lr * p) / (lr * p + (1 - p))
        return max(0.05, min(0.95, p))

    def _generate_claim(self) -> _Claim:
        rng  = self._rng
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
    """Fraction of genuine claims correctly approved."""
    total_genuine = env.total_claims - env.total_fraud
    raw = env.correct_approvals / total_genuine if total_genuine > 0 else 0.5
    return max(0.001, min(0.999, raw))


def grade_medium(env: InsuranceEnv) -> float:
    """Balanced accuracy penalising costly mistakes."""
    raw = (
        env.correct_approvals + env.fraud_caught - env.wrong_approvals * 0.5
    ) / env.total_claims
    return max(0.001, min(0.999, raw))


def grade_hard(env: InsuranceEnv) -> float:
    """
    Normalised total reward.
    Near-perfect play on 15 hard claims yields ~60 reward -> score ~1.0.
    """
    return max(0.001, min(0.999, env.total_reward / HARD_REWARD_NORMALIZER))

def agent_policy(obs: Observation) -> str:
    
    units     = obs.investigation_units
    remaining = max(obs.claims_remaining, 1)
    budget_per_claim = units / remaining

    
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

    total      = fraud_signals + genuine_signals
    fraud_ratio = fraud_signals / total if total > 0 else 0.5

    
    if fraud_signals >= 4:
        return "reject"
    if genuine_signals >= 4 and fraud_signals <= 1:
        return "approve"

    
    if units < 0.5:
        return "reject" if fraud_ratio >= 0.45 else "approve"

    if obs.investigations_done == 0 and budget_per_claim >= 0.5:
        return "request_info"

    if obs.investigations_done >= 1 and units >= 1.0 and budget_per_claim >= 0.8:
        return "document_audit"

    if obs.claims_remaining <= 2:
        return "reject" if fraud_ratio >= 0.45 else "approve"

    return "reject" if fraud_ratio >= 0.45 else "approve"

if __name__ == "__main__":
    graders = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}

    for task_name in ["easy", "medium", "hard"]:
        env = InsuranceEnv(task=task_name)
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < 90:
            act = agent_policy(obs)
            obs, reward, done, info = env.step(Action(action=act))
            steps += 1

        score = graders[task_name](env)
        print(
            f"[{task_name.upper():6s}] score={score:.3f}  "
            f"reward={env.total_reward:+.1f}  steps={steps}  "
            f"fraud_caught={env.fraud_caught}/{env.total_fraud}  "
            f"correct_approvals={env.correct_approvals}  "
            f"wrong_approvals={env.wrong_approvals}  "
            f"genuine_rejected={env.genuine_rejected}"
        )
