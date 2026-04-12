import random
from typing import Optional, Any

from openenv.core.env_server import (
    Observation as BaseObservation,
    Action as BaseAction,
    Environment as BaseEnvironment,
    State as BaseState,
)
from pydantic import Field

class InsuranceObservation(BaseObservation):
    
    claim_amount_normalized: float = Field(
        ..., description="Claim size / policy limit (0-1). >0.75 is suspicious."
    )
    days_since_incident: int = Field(
        ..., description="Filing delay in days. >30 days is suspicious. <=3 days is genuine."
    )
    num_prior_claims: int = Field(
        ..., description="Claimant's historical claim count. >=4 suspicious. 0 genuine."
    )
    document_score: float = Field(
        ..., description="Document completeness and authenticity (0-1). <0.35 suspicious. >0.80 genuine."
    )
    witness_available: bool = Field(
        ..., description="Whether a corroborating witness has been identified."
    )
    repair_estimate_match: float = Field(
        ..., description="Repair estimate vs claimed amount alignment (0-1). <0.30 suspicious."
    )
    last_investigation_signal: Optional[str] = Field(
        None,
        description=(
            "Result of the most recent investigation on this claim. "
            "Values: 'suspicious' | 'looks_genuine' | 'info_revealed' | null. "
            "Use this to update your belief about fraud likelihood."
        )
    )
    investigation_units: float = Field(
        ..., description="Remaining investigation budget shared across all remaining claims."
    )
    investigations_done: int = Field(
        ..., description="Investigations already done on this claim (max 3)."
    )
    claims_remaining: int = Field(
        ..., description="Claims still to process including the current one."
    )


class InsuranceAction(BaseAction):
    
    action: str = Field(
        ...,
        description=(
            "One of: approve | reject | "
            "quick_check | document_audit | field_investigation | request_info"
        ),
    )


class InsuranceState(BaseState):
    """Internal episode state for the InsuranceEnvironment."""
    task: str = Field(default="easy", description="Current task difficulty")
    claims_processed: int = Field(default=0, description="Claims resolved so far")
    claims_total: int = Field(default=15, description="Total claims in episode")
    investigation_units_remaining: float = Field(default=0.0)
    correct_approvals: int = Field(default=0)
    wrong_approvals: int = Field(default=0)
    fraud_caught: int = Field(default=0)
    genuine_rejected: int = Field(default=0)
    total_reward: float = Field(default=0.0)




class _Claim:
    __slots__ = (
        "is_fraud", "claim_amount_normalized", "days_since_incident",
        "num_prior_claims", "document_score", "witness_available",
        "repair_estimate_match",
    )

    def __init__(self, is_fraud, claim_amount_normalized, days_since_incident,
                 num_prior_claims, document_score, witness_available,
                 repair_estimate_match):
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


class InsuranceEnvironment(BaseEnvironment):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task: str = "easy", seed: int = 42):
        super().__init__()
        if task not in TASK_BUDGETS:
            raise ValueError(
                f"Unknown task '{task}'. Choose from: {list(TASK_BUDGETS)}"
            )
        self.task = task
        self.seed = seed
        self.total_claims = TOTAL_CLAIMS
        self._reset_state()

    def _reset_state(self):
        self.current_claim_idx: int = 0
        self.investigation_units: float = TASK_BUDGETS[self.task]
        self._current_claim: Optional[_Claim] = None
        self._rng = random.Random(self.seed)
        self._steps_on_current_claim: int = 0
        self._fraud_posterior: float = TASK_FRAUD_RATES[self.task]
        self._last_signal: Optional[str] = None

        self.correct_approvals: int = 0
        self.wrong_approvals: int = 0
        self.fraud_caught: int = 0
        self.genuine_rejected: int = 0
        self.total_fraud: int = 0
        self.total_reward: float = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> InsuranceObservation:
        
        if task is not None and task in TASK_BUDGETS:
            self.task = task
        if seed is not None:
            self.seed = seed
        self._reset_state()
        self._current_claim = self._generate_claim()
        return self._make_observation(reward=0.0, done=False, metadata={
            "message": f"Episode started — {self.total_claims} claims to process",
            "task": self.task,
            "seed": self.seed,
        })

    def step(
        self,
        action: InsuranceAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> InsuranceObservation:
       
        action_str = action.action.strip().lower()

        if action_str not in VALID_ACTIONS:
            return self._make_observation(
                reward=-3.0,
                done=False,
                metadata={
                    "explanation": f"Invalid action '{action_str}'. Valid: {sorted(VALID_ACTIONS)}",
                    "investigation_signal": None,
                    "is_fraud": None,
                    "steps_on_claim": self._steps_on_current_claim,
                }
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
                    self.investigation_units = round(self.investigation_units - cost, 2)
                    reward = -cost

                    if action_str == "request_info":
                        investigation_signal = "info_revealed"
                        self._last_signal = "info_revealed"
                        self._fraud_posterior = self._update_posterior_from_info(claim)
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
                        signal_label = "suspicious" if signal_is_fraud else "looks_genuine"
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

        is_fraud_revealed = None
        if advance_claim:
            is_fraud_revealed = claim.is_fraud
            if claim.is_fraud:
                self.total_fraud += 1
            self.current_claim_idx += 1
            self._steps_on_current_claim = 0
            self._fraud_posterior = TASK_FRAUD_RATES[self.task]
            self._last_signal = None
            if self.current_claim_idx < self.total_claims:
                self._current_claim = self._generate_claim()

        done = self.current_claim_idx >= self.total_claims

        return self._make_observation(
            reward=reward,
            done=done,
            metadata={
                "explanation": explanation,
                "investigation_signal": investigation_signal,
                "is_fraud": is_fraud_revealed,
                "steps_on_claim": self._steps_on_current_claim,
                "investigation_units_remaining": self.investigation_units,
                "total_reward": round(self.total_reward, 2),
            }
        )

    def state(self) -> InsuranceState:
        """Return current episode state (does not advance the episode)."""
        return InsuranceState(
            step_count=self.current_claim_idx,
            task=self.task,
            claims_processed=self.current_claim_idx,
            claims_total=self.total_claims,
            investigation_units_remaining=self.investigation_units,
            correct_approvals=self.correct_approvals,
            wrong_approvals=self.wrong_approvals,
            fraud_caught=self.fraud_caught,
            genuine_rejected=self.genuine_rejected,
            total_reward=round(self.total_reward, 2),
        )

    def _make_observation(
        self,
        reward: float,
        done: bool,
        metadata: dict,
    ) -> InsuranceObservation:
        c = self._current_claim
        return InsuranceObservation(
         
            done=done,
            reward=reward,
            metadata=metadata,
           
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
    def _bayesian_update(prior: float, accuracy: float, signal_is_fraud: bool) -> float:
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
        Bayesian update from exact revealed feature values (request_info).
        Applies three sequential likelihood-ratio updates from the CURRENT
        posterior — preserving all prior investigation evidence.
        """
        p = self._fraud_posterior

        
        if   claim.num_prior_claims >= 5: lr = 3.0
        elif claim.num_prior_claims >= 3: lr = 1.8
        elif claim.num_prior_claims == 0: lr = 0.5
        else:                             lr = 1.0
        p = max(0.01, min(0.99, (lr * p) / (lr * p + (1 - p))))

        
        if   claim.days_since_incident > 40: lr = 2.5
        elif claim.days_since_incident > 20: lr = 1.6
        elif claim.days_since_incident <= 3: lr = 0.6
        else:                                lr = 1.0
        p = max(0.01, min(0.99, (lr * p) / (lr * p + (1 - p))))

        
        if   claim.repair_estimate_match < 0.3:  lr = 2.8
        elif claim.repair_estimate_match < 0.5:  lr = 1.5
        elif claim.repair_estimate_match > 0.85: lr = 0.5
        else:                                    lr = 1.0
        return max(0.05, min(0.95, (lr * p) / (lr * p + (1 - p))))

    def _generate_claim(self) -> _Claim:
        rng  = self._rng
        task = self.task

        if task == "easy":
            is_fraud = rng.random() < TASK_FRAUD_RATES["easy"]
            if is_fraud:
                return _Claim(True, rng.uniform(0.7,1.0), rng.randint(20,60),
                              rng.randint(3,8), rng.uniform(0.0,0.4),
                              False, rng.uniform(0.0,0.35))
            return _Claim(False, rng.uniform(0.1,0.5), rng.randint(0,7),
                          rng.randint(0,1), rng.uniform(0.75,1.0),
                          rng.random()<0.7, rng.uniform(0.8,1.0))

        elif task == "medium":
            is_fraud = rng.random() < TASK_FRAUD_RATES["medium"]
            noise = rng.uniform(-0.2, 0.2)
            if is_fraud:
                return _Claim(True,
                              min(1.0, rng.uniform(0.4,0.9)+noise),
                              rng.randint(5,40), rng.randint(1,6),
                              max(0.0, rng.uniform(0.2,0.65)+noise),
                              rng.random()<0.3,
                              max(0.0, rng.uniform(0.2,0.6)+noise))
            return _Claim(False,
                          min(1.0, rng.uniform(0.2,0.7)+noise),
                          rng.randint(0,20), rng.randint(0,3),
                          max(0.0, rng.uniform(0.5,0.95)+noise),
                          rng.random()<0.5,
                          max(0.0, rng.uniform(0.6,1.0)+noise))

        else:  
            is_fraud = rng.random() < TASK_FRAUD_RATES["hard"]
            if is_fraud:
                return _Claim(True, rng.uniform(0.3,0.75), rng.randint(1,15),
                              rng.randint(0,4), rng.uniform(0.55,0.90),
                              rng.random()<0.45, rng.uniform(0.5,0.85))
            return _Claim(False, rng.uniform(0.1,0.9), rng.randint(0,30),
                          rng.randint(0,5), rng.uniform(0.4,1.0),
                          rng.random()<0.5, rng.uniform(0.5,1.0))

def grade_easy(env: InsuranceEnvironment) -> float:
    total_genuine = env.total_claims - env.total_fraud
    raw = env.correct_approvals / total_genuine if total_genuine > 0 else 0.5
    return max(0.001, min(0.999, raw))


def grade_medium(env: InsuranceEnvironment) -> float:
    raw = (env.correct_approvals + env.fraud_caught - env.wrong_approvals * 0.5) / env.total_claims
    return max(0.001, min(0.999, raw))


def grade_hard(env: InsuranceEnvironment) -> float:
    return max(0.001, min(0.999, env.total_reward / HARD_REWARD_NORMALIZER))



def agent_policy(obs: InsuranceObservation) -> str:
    
    units            = obs.investigation_units
    remaining        = max(obs.claims_remaining, 1)
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

    total       = fraud_signals + genuine_signals
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

InsuranceEnv = InsuranceEnvironment
Observation  = InsuranceObservation
Action       = InsuranceAction

if __name__ == "__main__":
    graders = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}

    for task_name in ["easy", "medium", "hard"]:
        env = InsuranceEnvironment(task=task_name)
        obs = env.reset()
        done = False
        steps = 0

        while not obs.done and steps < 90:
            act = agent_policy(obs)
            obs = env.step(InsuranceAction(action=act))
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
