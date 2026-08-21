"""
Rule-based urgency labeling for CFPB complaints.

CFPB complaints have no "urgency" field

Tiers:
  HIGH   - fraud, identity theft, unauthorized activity, threats of legal
           action / repossession / foreclosure -- money or safety at risk,
           needs action fast.
  MEDIUM - a real problem with money or service (incorrect charges, credit
           report errors, denied applications, struggling to pay) that
           needs resolving but isn't an emergency.
  LOW    - everything else (advertising, communication tactics, general
           account service questions, closing an account, etc.)
"""
import re

HIGH_KEYWORDS = [
    "fraud", "scam", "identity theft", "unauthorized", "stolen",
    "foreclosure", "repossess", "threatened", "harass", "garnish",
    "eviction", "lawsuit", "arrest", "wage",
]

MEDIUM_KEYWORDS = [
    "incorrect", "error", "dispute", "denied", "struggling to pay",
    "closing", "closed your account", "fees", "charged", "overdraft",
    "credit reporting", "investigation", "application", "delay", "late fee",
]

URGENCY_LEVELS = ["low", "medium", "high"]


def _keyword_hit(text, keywords):
    text = text.lower()
    return any(kw in text for kw in keywords)


def label_urgency(issue, sub_issue=""):
    """Return 'high' / 'medium' / 'low' for one complaint's issue text."""
    combined = f"{issue or ''} {sub_issue or ''}"
    if _keyword_hit(combined, HIGH_KEYWORDS):
        return "high"
    if _keyword_hit(combined, MEDIUM_KEYWORDS):
        return "medium"
    return "low"


def map_urgency(issues, sub_issues):
    """Vectorized helper: apply label_urgency() over two aligned lists/Series."""
    return [label_urgency(i, s) for i, s in zip(issues, sub_issues)]


if __name__ == "__main__":
    tests = [
        ("Fraud or scam", ""),
        ("Incorrect information on your report", "Account status incorrect"),
        ("Advertising and marketing, including promotional offers", ""),
        ("Struggling to pay mortgage", "Loan modification, collection, foreclosure"),
    ]
    for issue, sub in tests:
        print(f"[{label_urgency(issue, sub):>6}]  {issue} / {sub}")
