from ccer.mechanism.agr.track_r import default_synthetic_suite, run_oracle_routing_batch


def test_oracle_routing_pm_rate_zero_cb_retention_one():
    claims = default_synthetic_suite()
    # Perfect extraction for PM when anchor has value
    fixed = []
    for c in claims:
        if c.is_pm and c.anchor_has_value:
            fixed.append(
                type(c)(
                    claim_id=c.claim_id,
                    pm_type=c.pm_type,
                    is_pm=c.is_pm,
                    anchor_has_value=c.anchor_has_value,
                    extraction_hit=True,
                    v_anchor="anchor_val",
                    claim_value="wrong",
                    is_clean_bullet=c.is_clean_bullet,
                )
            )
        else:
            fixed.append(c)
    res = run_oracle_routing_batch(fixed)
    assert res["pm_rate"] == 0.0
    assert res["cb_retention"] == 1.0
