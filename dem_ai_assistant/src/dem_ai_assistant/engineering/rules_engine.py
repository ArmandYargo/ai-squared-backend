from src.dem_ai_assistant.core.models import EngineeringFinding, KPIResult


def evaluate_run(kpis: KPIResult) -> list[EngineeringFinding]:
    findings: list[EngineeringFinding] = []

    if kpis.egress_events > 5:
        findings.append(
            EngineeringFinding(
                severity="high",
                title="Frequent egress events detected",
                detail="The case exceeds the initial acceptable egress threshold and likely needs geometry or flow control changes.",
            )
        )

    if kpis.min_edge_margin_mm < 25:
        findings.append(
            EngineeringFinding(
                severity="high",
                title="Low minimum edge margin",
                detail="Material is approaching the belt edge closely, suggesting instability or lateral migration risk.",
            )
        )

    if kpis.stability_score < 0.70:
        findings.append(
            EngineeringFinding(
                severity="medium",
                title="Flow stability is weak",
                detail="The burden appears insufficiently stable under the current setup. Transition profile or chute centring may need review.",
            )
        )

    if not findings:
        findings.append(
            EngineeringFinding(
                severity="low",
                title="No major rule-based concerns found",
                detail="This scenario appears acceptable under the current simple rule set.",
            )
        )

    return findings