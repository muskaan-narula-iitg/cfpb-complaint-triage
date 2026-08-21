"""
Retrieval-based resolution suggester: look up a curated next-step template by predicted category,
rather than generating free text at inference time. No hallucination risk,
easy to evaluate, easy to extend.

Coverage: CFPB's exact product-category wording shifts over time, so instead
of keying off exact strings we match on a few keywords found in the category
name. Anything that doesn't match a known keyword falls back to a sensible
generic message,
just widened to "unmatched category -> generic fallback".
"""

TEMPLATES = [
    # (keywords to look for in the product/category string, resolution template)
    (["mortgage"],
     "Request a written explanation from your servicer and, if it involves missed "
     "payments, ask about loss-mitigation / loan-modification options before any "
     "foreclosure step. Escalate to CFPB if the servicer doesn't respond within 30 days."),
    (["credit report", "credit repair", "consumer report"],
     "File a dispute directly with the credit bureau (Equifax/Experian/TransUnion) "
     "citing the specific inaccurate item, and request the furnisher's investigation "
     "result in writing."),
    (["debt collection"],
     "Request debt validation in writing before paying anything, and keep a record of "
     "all collector contact -- this matters if the debt is inaccurate or past the "
     "statute of limitations."),
    (["credit card"],
     "Dispute the specific transaction/fee with your card issuer in writing citing the "
     "date and amount; issuers must acknowledge billing disputes within 30 days."),
    (["checking", "savings", "bank account"],
     "Ask your bank for a written explanation of the fee/hold/closure and request it be "
     "reversed if it doesn't match their disclosed account terms."),
    (["student loan"],
     "Contact your loan servicer to confirm your repayment plan and request an account "
     "history if payments/balances look wrong."),
    (["vehicle loan", "auto loan", "lease"],
     "Request a payoff/account statement from the lender in writing and confirm any "
     "repossession notice periods required in your state before further action."),
    (["money transfer", "virtual currency", "money service"],
     "Contact the money transfer provider immediately to attempt to reverse/trace the "
     "transaction, and file a fraud report if it was unauthorized."),
    (["payday", "title loan", "personal loan"],
     "Check your state's lending rate caps and request full loan terms in writing; many "
     "payday-loan disputes hinge on undisclosed fees."),
    (["prepaid"],
     "Contact the card issuer to freeze the card if unauthorized activity is involved, "
     "and request a transaction history for the disputed period."),
]

GENERIC_FALLBACK = (
    "Contact the company in writing (keeps a paper trail), clearly describe the issue "
    "and desired outcome, and if unresolved within 15 business days, escalate to CFPB."
)


def suggest_resolution(category: str) -> str:
    category_lower = (category or "").lower()
    for keywords, template in TEMPLATES:
        if any(kw in category_lower for kw in keywords):
            return template
    return GENERIC_FALLBACK


if __name__ == "__main__":
    for c in ["Mortgage", "Credit card", "Some New Category CFPB Invents Later"]:
        print(f"[{c}] -> {suggest_resolution(c)}")
