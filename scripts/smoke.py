"""End-to-end smoke test: three typed questions about one state, answered in one read.

uv run python scripts/smoke.py
"""

import json

from dlm_jev.engine import DiffusionReader
from dlm_jev.schema import DecideRequest

REQUEST = {
    "state": {
        "from": "billing@paypa1-secure.com",
        "subject": "URGENT: your account will be suspended",
        "body": "Verify your password within 24 hours at http://paypa1-secure.com/login or lose access.",
    },
    "questions": {
        "phishing": {"type": "noul", "instructions": "This email is a phishing attempt."},
        "route": {
            "type": "choice",
            "instructions": "Which queue should this email go to?",
            "options": ["inbox", "promotions", "security review", "spam"],
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgently does a human need to look at this?",
            "levels": ["none", "low", "medium", "high", "critical"],
        },
    },
}


def main() -> None:
    """Load the model, answer twice (first call warms up), print the second."""
    reader = DiffusionReader()
    request = DecideRequest.model_validate(REQUEST)
    reader.decide(request)
    print(json.dumps(reader.decide(request).model_dump(), indent=2))


if __name__ == "__main__":
    main()
