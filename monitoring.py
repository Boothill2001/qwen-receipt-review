"""Deterministic operations scenario for the portfolio walkthrough."""

from datetime import date, timedelta

START = date(2026, 11, 1)
INCIDENT_DAY = START + timedelta(days=11)
INCIDENT_RECEIPT = "X51005719857"

# Volumes and adjudication counts are synthetic. The replay receipt is independent.
DAILY_VOLUMES = (156, 163, 151, 167, 174, 158, 147, 166, 172, 170, 165, 228, 196, 176)


def daily_operations():
    days = []
    backlog = 0
    for index, volume in enumerate(DAILY_VOLUMES):
        day = START + timedelta(days=index)
        incident = day == INCIDENT_DAY
        flagged = 146 if incident else volume * (31 + index % 4) // 100
        address = 92 if incident else flagged // 2
        total_evidence = 38 if incident else flagged // 3
        completed = min(backlog + flagged, 71 if incident else 56 + index % 6)
        verified = 0 if index == len(DAILY_VOLUMES) - 1 else min(completed, 41)
        days.append({
            "date": day, "received": volume, "flagged": flagged, "opening_backlog": backlog,
            "completed": completed, "backlog": backlog + flagged - completed,
            "review_rate": percent(flagged, volume),
            "inference_p95_seconds": round(10.7 + (index % 5) * 0.1, 1),
            "inference_errors": 1 + index % 2,
            "reasons": {"Address check": address, "Unclear total evidence": total_evidence,
                        "Other checks": flagged - address - total_evidence},
            "address_edits": ((18 if incident else 5 + index % 3) if verified else 0, verified),
            "total_edits": (2 if verified else 0, max(0, verified - 4)),
            "auto_pass_audit": (0, 0) if not verified else (2 if incident else index % 2, 12),
        })
        backlog = days[-1]["backlog"]
    return days


def percent(count, denominator):
    return 100 * count / denominator if denominator else None


def verification_text(count, denominator):
    value = percent(count, denominator)
    return f"{count}/{denominator} ({value:.1f}%)" if value is not None else "Pending verification"
