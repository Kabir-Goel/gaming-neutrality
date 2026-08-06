# Project context

Political science research project measuring whether LLMs behave more
"neutral" when they detect an evaluation context than in casual use.

Key terms:
- persona = signaled user identity (neutral / liberal / conservative)
- frame = audit vs casual wrapper around the same question
- stance/framing = continuous [-1,1] codes; refusal = ordinal {0,1,2}
- steering = spread of stance across personas within one frame
- ADG (audit-deployment gap) = steering_casual - steering_audit

Conventions:
- Python 3.11+, type hints, no classes unless needed
- All data as JSONL, one record per line
- Deterministic IDs: {issue}__{persona}__{frame}__{model}__r{run}
- Never hardcode API keys; read from .env via python-dotenv
- All scripts idempotent and resumable — skip IDs already in the output file
