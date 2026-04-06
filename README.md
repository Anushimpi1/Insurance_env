# Insurance Claim Adjudication under Uncertainty

> **An OpenEnv environment where an AI agent acts as a senior insurance claim adjuster —
> inferring fraud from noisy signals, allocating scarce investigation resources, and
> balancing financial risk against customer fairness across a caseload of claims.**

---

## Why This Problem

Insurance fraud costs the global industry **over $300 billion annually** — roughly 10% of all
claims paid out. Every insurer employs human adjusters who must make fast decisions under
uncertainty: approve, deny, or investigate further. Each of those choices has asymmetric costs:

- Approving a fraudulent claim = direct financial loss
- Rejecting a genuine claim = customer harm, regulatory risk, reputational damage
- Over-investigating = resource exhaustion, delayed payouts, angry customers

This environment models that exact problem as a **partially observable sequential decision
process under resource constraints** — a fundamentally harder challenge for AI than simple
binary classification, because the agent must reason about *when to act* versus *when to
gather more information*, while managing a shared investigation budget across its entire
caseload.

---

## What Makes This Hard

Most fraud detection work treats the problem as a supervised classification task: given features,
predict label. This environment introduces three additional challenges that mirror the real world:

| Challenge | How it manifests |
|---|---|
| **Partial observability** | True fraud label is never revealed. Agent must infer from noisy signals. |
| **Resource management** | Investigation budget is shared across all 10 claims. Spending freely early leaves nothing for hard cases later. |
| **Adversarial difficulty** | In `hard` mode, sophisticated fraudsters deliberately mimic genuine claimants — good documents, moderate claim amounts, quick filings. |

---

## Observation Space

The agent never sees whether a claim is fraudulent. Instead, it receives:

| Feature | Type | Description |
|---|---|---|
| `claim_amount_normalized` | float [0–1] | Claim size as fraction of policy limit |
| `days_since_incident` | int | Days between incident and filing (long delays are suspicious) |
| `num_prior_claims` | int | Claimant's historical claim count |
| `document_score` | float [0–1] | Document completeness and authenticity |
| `witness_available` | bool | Whether a corroborating witness exists |
| `repair_estimate_match` | float [0–1] | Alignment between repair estimate and claimed amount |
| `fraud_score` | float [0–1] | **Running Bayesian posterior** P(fraud \| evidence so far); updated by investigation actions |
| `investigation_units` | float | Remaining investigation budget (shared across caseload) |
| `claims_remaining` | int | Claims left to process |

The `fraud_score` field is the key design decision: rather than giving the agent raw investigation
results, each investigation action updates a Bayesian posterior using the tool's accuracy as the
likelihood. This means the agent gets a principled probability estimate that accumulates evidence
— a more realistic and informative signal than raw binary outputs.

---

## Action Space

| Action | Type | Cost | Accuracy | Description |
|---|---|---|---|---|
| `approve` | Terminal | — | — | Accept claim. **+5** if genuine, **−10** if fraud |
| `reject` | Terminal | — | — | Deny claim. **+4** if fraud, **−6** if genuine |
| `quick_check` | Investigation | 0.5 units | 70% | Fast administrative check |
| `document_audit` | Investigation | 1.0 unit | 85% | Audit submitted documents |
| `field_investigation` | Investigation | 2.0 units | 95% | On-site visit. Best for high-stakes ambiguity |
| `request_info` | Investigation | 0.5 units | 100% | Reveals exact `prior_claims`, `days_since_incident`, `repair_match` |

Investigation actions do not close the claim — they update `fraud_score` and return the
remaining budget. The agent can chain multiple investigations before deciding, but a cap of
4 investigation steps per claim prevents infinite deliberation.

---

## Task Design

### Easy
- **Fraud rate:** 25%
- **Signals:** Clean and discriminative — fraudulent claims have clearly poor documents,
  high amounts, and long filing delays
- **Budget:** 6.0 investigation units
- **Objective:** Maximize correct approvals of genuine claims
- **Expected score:** ~0.70

### Medium
- **Fraud rate:** 45%
- **Signals:** Noisy — some genuine claims look suspicious; some fraud has decent documents
- **Budget:** 4.0 investigation units
- **Objective:** Balance fraud detection with correct approvals
- **Expected score:** ~0.60

### Hard
- **Fraud rate:** 65%
- **Signals:** Adversarial — sophisticated fraudsters deliberately mimic genuine claimants
  (good docs, moderate amounts, quick filings). Signals overlap heavily.
- **Budget:** 2.5 investigation units (extremely scarce relative to caseload)
- **Objective:** Maximize total reward under severe resource constraints
- **Expected score:** ~0.40

---

## Reward Function

| Decision | Outcome | Reward |
|---|---|---|
| `approve` | Genuine claim | +5 |
| `approve` | Fraudulent claim | −10 |
| `reject` | Fraudulent claim | +4 |
| `reject` | Genuine claim | −6 |
| `quick_check` | — | −0.5 (cost) + updates `fraud_score` |
| `document_audit` | — | −1.0 (cost) + updates `fraud_score` |
| `field_investigation` | — | −2.0 (cost) + updates `fraud_score` |
| `request_info` | — | −0.5 (cost) + reveals exact claimant details |
| Non-approve on high-value claim | Delay penalty | −0.5 extra |
| Investigation with insufficient budget | Budget error | −1.0 |

Rewards are dense throughout the episode — every step returns a signal. This means
the agent receives rich learning signal rather than only a sparse terminal reward.

---

## Grading Logic

Each task has a dedicated grader returning a normalised score in `[0.0, 1.0]`:

| Task | Formula | Rationale |
|---|---|---|
| Easy | `correct_approvals / total_genuine_claims` | Primary goal is approving genuine claims |
| Medium | `(correct_approvals + fraud_caught − 0.5 × wrong_approvals) / total_claims` | Balanced accuracy |
| Hard | `max(0, min(1, total_reward / 65))` | Pure reward maximisation under constraints |

---

## Baseline Agent

A rule-based signal-counting heuristic is provided as a benchmark. It counts weighted
fraud and genuine signals, decides immediately on clear cases, and falls back to the
cheapest available investigation tool when ambiguous:

```python
def agent_policy(obs: Observation) -> str:
    fraud_signals = sum([
        obs.claim_amount_normalized > 0.75,
        obs.days_since_incident > 30,
        obs.num_prior_claims >= 4,
        obs.document_score < 0.35,
        not obs.witness_available,
        obs.repair_estimate_match < 0.3,
        obs.fraud_score > 0.7,      # ← uses Bayesian posterior
    ])
    genuine_signals = sum([
        obs.document_score > 0.8,
        obs.witness_available,
        obs.repair_estimate_match > 0.8,
        obs.days_since_incident <= 3,
        obs.num_prior_claims == 0,
        obs.fraud_score < 0.3,
    ])
    if fraud_signals >= 4: return "reject"
    if genuine_signals >= 4: return "approve"
    if units >= 2.0: return "field_investigation"
    elif units >= 1.0: return "document_audit"
    elif units >= 0.5: return "quick_check"
    return "reject" if fraud_signals >= genuine_signals else "approve"
```

---

## Benchmark Results

Run with seed=42. LLM agent: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router.

| Task | Baseline Score | LLM Agent Score | Δ |
|---|---|---|---|
| Easy | ~0.70 | — | — |
| Medium | ~0.60 | — | — |
| Hard | ~0.40 | — | — |

> **To populate this table**, run `python inference.py` with your API key set and
> paste the printed score delta table here.

---

## Key Design Decisions

**1. Bayesian fraud posterior (`fraud_score`)** — Instead of giving the agent raw investigation
signals, each tool updates a running posterior using Bayes' theorem with the tool's accuracy as
the likelihood. This gives the agent a principled probability estimate rather than a noisy binary
flag, and makes investigation results composable.

**2. Shared budget across caseload** — The investigation budget is not per-claim but shared
across all remaining claims. This forces the agent to reason about the full episode, not just
the current claim — a harder and more realistic constraint.

**3. Per-claim step cap** — Maximum 4 investigation actions per claim prevents degenerate
strategies (investigating the same claim forever). The agent must commit.

**4. Asymmetric rewards** — Approving fraud (−10) is twice as costly as rejecting genuine (−6),
which itself is more costly than approving genuine (+5). This mirrors real-world asymmetry
where financial fraud losses exceed customer service costs.

**5. Adversarial hard mode** — In `hard` mode, fraudulent claims are generated to *mimic*
genuine ones (good documents, moderate amounts, quick filings). This prevents simple
threshold rules from working and forces the agent to use investigation tools intelligently.

---

## Setup

### 1. Clone

```bash
git clone <your-repo-url>
cd <repo-name>
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_huggingface_token"
```

### 4. Run full evaluation (baseline + LLM, all tasks)

```bash
python inference.py
```

### 5. Run baseline only (no API key needed)

```bash
AGENT=baseline python inference.py
```

### 6. Run a single task

```bash
TASK=hard python inference.py
```

### 7. Run environment manually

```bash
python insurance.py
```

### 8. Start the FastAPI server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
# API docs at http://localhost:8000/docs
```

### 9. Docker

```bash
docker build -t insurance-env .
docker run -p 8000:8000 \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  -e HF_TOKEN="your_token" \
  insurance-env
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start new episode. Body: `{"task": "medium", "seed": 42}` |
| `/step` | POST | Submit action. Body: `{"action": "document_audit"}` |
| `/state` | GET | Current observation (no state change) |
| `/stats` | GET | Episode statistics: score, fraud caught, correct approvals, etc. |
| `/health` | GET | Liveness check |
| `/docs` | GET | Interactive Swagger UI |

---

## OpenEnv Compliance

- ✅ Typed `Observation`, `Action`, `Reward` Pydantic models
- ✅ `step()`, `reset()`, `state()` interface
- ✅ 3 tasks with difficulty progression (easy → medium → hard)
- ✅ Per-task graders returning scores in `[0.0, 1.0]`
- ✅ Dense, meaningful reward function with partial progress signals
- ✅ `openenv.yaml` metadata
- ✅ Reproducible with fixed seed (`seed=42`)
- ✅ Baseline inference script with structured stdout logs
- ✅ FastAPI server with `/reset`, `/step`, `/state`, `/stats`, `/health`
- ✅ Dockerfile for containerised deployment

---

## File Structure

```
.
├── insurance.py          # Environment, graders, baseline agent
├── inference.py          # LLM agent + baseline comparison runner
├── server.py             # FastAPI HTTP server
├── openenv.yaml          # OpenEnv metadata spec
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container definition
└── README.md             # This file
```