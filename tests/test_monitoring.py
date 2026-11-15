from monitoring import INCIDENT_DAY, daily_operations, percent, verification_text


def test_daily_queue_and_review_reasons_reconcile():
    days = daily_operations()
    assert len({row["date"] for row in days}) == 14
    previous_backlog = 0
    for row in days:
        assert row["opening_backlog"] == previous_backlog
        assert row["backlog"] == previous_backlog + row["flagged"] - row["completed"]
        assert 0 <= row["backlog"] and 0 <= row["flagged"] <= row["received"]
        assert sum(row["reasons"].values()) == row["flagged"]
        assert row["review_rate"] == percent(row["flagged"], row["received"])
        for key in ("address_edits", "total_edits", "auto_pass_audit"):
            count, denominator = row[key]
            assert 0 <= count <= denominator
        previous_backlog = row["backlog"]


def test_incident_and_pending_day_have_distinct_signals():
    days = daily_operations()
    incident = next(row for row in days if row["date"] == INCIDENT_DAY)
    prior = days[days.index(incident) - 1]
    assert incident["received"] > prior["received"]
    assert incident["review_rate"] > prior["review_rate"]
    assert percent(*incident["address_edits"]) > percent(*prior["address_edits"])
    assert percent(*incident["total_edits"]) == percent(*prior["total_edits"])
    assert verification_text(*incident["auto_pass_audit"]) == "2/12 (16.7%)"
    assert verification_text(*days[-1]["auto_pass_audit"]) == "Pending verification"
    assert verification_text(*days[-1]["address_edits"]) == "Pending verification"
