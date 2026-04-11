---
title: Insurance Claim Adjudication
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: docker
sdk_version: "latest"
app_file: server:main
pinned: false
---

# Insurance Claim Adjudication under Uncertainty

> **Can an AI reason like a senior insurance adjuster — weighing noisy evidence,
> spending a limited investigation budget wisely, and making asymmetric-cost
> decisions across a full caseload of claims?**

---

## The Real Problem

Insurance fraud costs the global industry **$300 billion every year** — roughly
10% of all premiums collected. Behind that number are thousands of human
adjusters making fast, high-stakes decisions under incomplete information:

- **Approve** a claim that turns out to be fraud → direct financial loss
- **Reject** a genuine claim → customer harm, regulatory risk, legal exposure
- **Over-investigate** → wasted resources, delayed payouts, damaged relationships

Every one of those decisions is made without ever knowing for certain whether
the claim is fraudulent. The adjuster must read signals — document quality,
filing timing, repair estimate alignment, claim history — and decide when
they have enough evidence to act, and when it's worth spending more time
and money to be sure.

This environment models that exact problem.

---

## Why This Is Hard for AI

Most fraud detection research frames this as supervised classification:
*given labelled features, predict fraud/genuine*. That framing misses three
things that make the real problem genuinely difficult:

### 1. Partial Observability — No Ground Truth Signal
The agent receives only raw claim features. There is **no pre-computed fraud
score, no probability estimate, no label**. The agent must reason from
document quality, filing delay, claim history, and repair estimate alignment —
and update its belief as investigation results come in.

### 2. Resource Scarcity — One Budget for All Claims
Investigation budget is **shared across the entire 15-claim caseload**.
An agent that investigates every ambiguous claim exhausts its budget on
claim #4 and is forced to guess blindly for the rest. Good agents must
reason about *when to gather more evidence* vs *when to commit*, knowing
that every unit spent now is unavailable for a harder claim later.

### 3. Asymmetric, Irreversible Costs
Approving a fraudulent claim costs **−10 reward**.
Wrongly rejecting a genuine claim costs **−6 reward**.
The 2× penalty asymmetry is intentional: it encodes the real-world truth
that financial fraud loss is harder to recover from than customer complaints.
Agents that treat this as a symmetric classification problem will underperform.

---

## Observation Space

The agent never sees whether a claim is fraudulent. It receives:

| Field | Type | What It Tells You |
|---|---|---|
| `claim_amount_normalized` | float [0–1] | Claim size vs policy limit. >0.75 is suspicious. |
| `days_since_incident` | int | Filing delay. >30 days is suspicious. ≤3 days is genuine. |
| `num_prior_claims` | int | Claimant history. ≥4 is suspicious. 0 is genuine. |
| `document_score` | float [0–1] | Document authenticity. <0.35 suspicious. >0.80 genuine. |
| `witness_available` | bool | Independent corroboration. Absence is a weak fraud signal. |
| `repair_estimate_match` | float [0–1] | Estimate vs claim alignment. <0.30 suspicious. >0.80 genuine. |
| `last_investigation_signal` | string \| null | Result of most recent investigation: `suspicious`, `looks_genuine`, `info_revealed`, or `null`. |
| `investigation_units` | float | Remaining budget across ALL remaining claims. |
| `investigations_done` | int | Investigations already done on this claim (max 3). |
| `claims_remaining` | int | Claims left including the current one. |

---

## Action Space

| Action | Type | Cost | Accuracy | When to Use |
|---|---|---|---|---|
| `approve` | Terminal | — | — | Genuine signals clearly dominate |
| `reject` | Terminal | — | — | Fraud signals clearly dominate, or budget gone |
| `request_info` | Investigation | 0.5 | 100% on 3 fields | **Best first action** — reveals exact prior_claims, days, repair_match |
| `quick_check` | Investigation | 0.5 | 70% | Budget is very tight |
| `document_audit` | Investigation | 1.0 | 85% | Follow-up when request_info left ambiguity |
| `field_investigation` | Investigation | 2.0 | 95% | High-value claim, document_audit inconclusive |

---

## Task Progression

### Easy — Learn the Signals
- Fraud rate: 25% | Budget: 6.0 units | Claims: 15
- Fraudulent claims have clearly poor documents, high amounts, and long delays
- Genuine claims have strong documents, witnesses, and clean history
- Expected score: ~0.85

### Medium — Handle Noise
- Fraud rate: 45% | Budget: 4.0 units | Claims: 15
- Signals overlap — some genuine claims look suspicious, some fraud looks genuine
- Budget pressure requires selective investigation
- Expected score: ~0.65

### Hard — Adversarial Fraud
- Fraud rate: 65% | Budget: 2.5 units | Claims: 15
- Sophisticated fraudsters **deliberately mimic genuine claimants** — good documents,
  moderate amounts, quick filings. Surface signals are unreliable.
- Budget of 2.5 across 15 claims = 0.17 units per claim average
- Expected score: ~0.45

---

## Reward Function

| Decision | Reward |
|---|---|
| Approve genuine claim | **+5** |
| Reject fraudulent claim | **+4** |
| Reject genuine claim | **−6** |
| Approve fraudulent claim | **−10** ← worst outcome |
| Investigation costs | deducted per action |
| Over-budget investigation | **−1** penalty |

---

## Grading

| Task | Formula | What It Measures |
|---|---|---|
| Easy | `correct_approvals / total_genuine` | Customer protection |
| Medium | `(correct_approvals + fraud_caught − 0.5 × wrong_approvals) / total_claims` | Balanced accuracy |
| Hard | `total_reward / 60` | Overall strategic performance |

---

## Setup

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt

# Run baseline agent across all tasks
AGENT=baseline python inference.py

# Run LLM agent (requires API key)
HF_TOKEN=your_key python inference.py

# Single task
TASK=hard AGENT=baseline python inference.py

# Start the API server
python server.py
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | Groq |
| `MODEL_NAME` | Model identifier | `llama-3.3-70b-versatile` |
| `HF_TOKEN` / `API_KEY` | API key for LLM inference | — |

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start new episode. Body: `{"task": "easy\|medium\|hard", "seed": 42}` |
| `/step` | POST | Submit action. Body: `{"action": "approve\|reject\|..."}` |
| `/state` | GET | Current observation |
| `/stats` | GET | Episode statistics and current score |
| `/health` | GET | Liveness check |

---

## Example Episode

```
POST /reset  {"task": "medium"}
→ claim #1: amount=0.62, days=18, prior_claims=2, doc_score=0.58,
            witness=False, repair=0.55, signal=null, budget=4.0

# Ambiguous — no clear fraud or genuine signals. Use request_info.
POST /step   {"action": "request_info"}
→ signal=info_revealed: prior_claims=2, days=18, repair=0.55
  budget=3.5

# Still ambiguous after exact values. Follow up with document_audit.
POST /step   {"action": "document_audit"}
→ signal=suspicious, budget=2.5

# Signal says suspicious. Reject.
POST /step   {"action": "reject"}
→ reward=+4.0 (correctly caught fraud), budget=2.5, claims_remaining=14
```

---

## Project Structure

```
├── insurance.py      Core environment, graders, baseline agent
├── inference.py      LLM agent, evaluation runner, comparison table
├── server.py         FastAPI server (OpenEnv-compliant REST API)
├── app.py            Hugging Face Spaces entry point
├── openenv.yaml      Environment manifest
├── Dockerfile        Container definition
└── requirements.txt  Dependencies
```
