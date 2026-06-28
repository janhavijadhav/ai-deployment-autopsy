# Failure 6: The Eval That Lied

> **Production symptom:** Offline eval suite showed 96% accuracy. After 2 weeks in
> production, the customer reported the agent was "basically useless" for their team.
> Internal measurement of production accuracy: 61%. A 35-point gap between
> offline eval and real-world performance.

---

## What the Symptom Looked Like

```
Offline eval (clean dataset):
  ✓ "What are the late delivery penalties in the Apex Industries contract?"
  ✓ "Which suppliers have a risk score above 0.7?"
  ✓ "When does the Pacific Components MSA expire?"
  Pass rate: 96/100

Production (real analyst queries):
  ✗ "wat r the penaltis for apex if they deliver late"
  ✗ "apex late fine amount???"
  ✗ "show me risky china vendors plz"
  ✗ "Quelles sont les pénalités de livraison d'Apex?" [French]
  ✗ "was the fine amount changed in the 2023 amendment to CTR-04421"
  Pass rate: 61/100
```

The eval suite was lying. It reported 96% on a dataset that didn't represent
how real procurement analysts actually type.

---

## Root Cause: Distribution Gap Between Eval and Production

The eval dataset was built by the ML team writing clean, well-formed questions.
Real users are procurement analysts — often non-native English speakers, often
typing on mobile, often using internal jargon, abbreviations, and mixed languages.

### What was in the clean eval dataset:
- Perfect spelling and grammar
- Full supplier names (never abbreviations or IDs)
- Single unambiguous question per query
- English only
- Simple single-hop retrieval questions

### What was NOT in the eval dataset:
- Typos and autocorrect errors ("penaltis", "suiplier")
- Mixed language queries (EU team uses French/German)
- Ambiguous queries where two suppliers match ("show me apex" → Apex Industries OR Apex Tech)
- Multi-hop questions ("what changed between the 2022 and 2023 amendments?")
- Queries that assume wrong facts ("what are apex's net-60 terms?" — they have net-30)
- Mobile-style truncated queries ("apex late delivery?")
- Informal queries ("find me all the sketchy chinese vendors")

The model performed well on the distribution it was tested on.
That distribution didn't match production.

---

## The Wrong Diagnosis

- "The model needs fine-tuning" — no, it needed better prompting
- "The RAG pipeline needs improvement" — the RAG was fine on clean queries
- "The eval threshold is too low" — raising the threshold just hid the gap

The issue was the eval INPUTS, not the eval METRICS.

---

## The Fix: Adversarial Eval Generation (LLM-as-Attacker)

Use Claude to generate adversarial test cases that look like real analyst queries.
The adversarial generator produces:

1. **Typo-injected** variants: "penaltis", "suiplier", "Apec Industries"
2. **Multilingual**: French, German, Spanish versions of the same query
3. **Multi-hop**: "What did the 2023 amendment change vs the original contract?"
4. **Informal**: "show me risky chinese vendors", "apex fine if late?"
5. **Wrong assumptions**: "What are Apex's net-60 terms?" (they have net-30)
6. **Truncated/mobile**: "apex contract expire?"
7. **Ambiguous**: queries that could match 2+ suppliers

```python
# src/evals/adversarial_gen.py
gen = AdversarialEvalGenerator()
cases = await gen.generate(count=50)  # LLM generates 50 adversarial variants

# Each case looks like what a REAL analyst would type, not what an ML engineer
# would write when building a test suite
```

---

## Results After Adding Adversarial Evals to CI

| Phase | Clean eval acc. | Adversarial eval acc. | Production acc. |
|-------|----------------|----------------------|-----------------|
| Before fix | 96% | (not measured) | 61% |
| After fix | 94% | 83% | 85% |

The production accuracy closed to within 2% of the adversarial eval accuracy.
The adversarial eval is now the gating metric, not the clean eval.

The clean eval still runs — it's a regression guard. But passing it alone is no
longer sufficient to deploy.

---

## CI Gate

```yaml
# .github/workflows/eval-gate.yml
- name: Generate adversarial cases
  run: python -m src.evals.adversarial_gen --count 50

- name: Run adversarial eval suite
  run: python -m src.evals.eval_runner --mode adversarial
  # Blocks deploy if adversarial pass rate < 90%
```

---

## Lesson

**Your eval dataset is a product. It can be wrong.**

The eval suite was built correctly — it measured what it measured accurately.
But it measured the wrong thing. An eval that doesn't represent production
is worse than no eval: it gives false confidence.

The adversarial generator is not a perfect solution — it generates plausible
messy queries, not actual recorded production queries. The next step is
collecting real failing production queries and feeding them back into the
eval suite automatically. But the adversarial gen closed 80% of the gap
in 2 hours of work.
