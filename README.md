---
title: Insurance Claim Adjudication
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
sdk_version: "latest"
app_file: server.py
pinned: false
---

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
| `days_since_incident` | int | Days between incident and filing |
| `num_prior_claims` | int | Claimant's historical claim count |
| `document_score` | float [0–1] | Document completeness and authenticity |
| `witness_available` | bool | Whether a corroborating witness exists |
| `repair_estimate_match` | float [0–1] | Alignment between repair estimate and claimed amount |
| `fraud_score` | float [0–1] | Bayesian posterior probability of fraud |
| `investigation_units` | float | Remaining investigation budget |
| `claims_remaining` | int | Claims left to process |

---

## Action Space

| Action | Type | Cost | Accuracy | Description |
|---|---|---|---|---|
| `approve` | Terminal | — | — | +5 genuine, −10 fraud |
| `reject` | Terminal | — | — | +4 fraud, −6 genuine |
| `quick_check` | Investigation | 0.5 | 70% | Fast check |
| `document_audit` | Investigation | 1.0 | 85% | Document audit |
| `field_investigation` | Investigation | 2.0 | 95% | On-site visit |
| `request_info` | Investigation | 0.5 | 100% | Reveals exact details |

---

## Task Design

### Easy
- Fraud rate: 25%  
- Budget: 6.0  
- Expected score: ~0.70  

### Medium
- Fraud rate: 45%  
- Budget: 4.0  
- Expected score: ~0.60  

### Hard
- Fraud rate: 65%  
- Budget: 2.5  
- Expected score: ~0.40  

---

## Reward Function

| Decision | Outcome | Reward |
|---|---|---|
| Approve genuine | +5 |
| Approve fraud | −10 |
| Reject fraud | +4 |
| Reject genuine | −6 |
| Investigations | Cost deducted |
| Delay penalty | −0.5 |
| Budget error | −1.0 |

---

## Setup

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt