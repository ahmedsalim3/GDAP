# =================
# BigQuery Queries
# =================

INDIRECT_SCORES = """
SELECT
  overall_indirect.diseaseId AS disease_id,
  diseases.name AS disease_name,
  targets.approvedSymbol AS symbol,
  overall_indirect.score AS indirect_score,
  overall_indirect.evidenceCount AS indirect_evidence_count
FROM
  `open-targets-prod.platform.association_by_overall_indirect` AS overall_indirect
JOIN
  `open-targets-prod.platform.disease` AS diseases
ON
  overall_indirect.diseaseId = diseases.id
JOIN
  `open-targets-prod.platform.target` AS targets
ON
  overall_indirect.targetId = targets.id
WHERE
  overall_indirect.diseaseId = '{disease_id}'
ORDER BY
  indirect_score DESC;
"""

DIRECT_SCORES = """
SELECT
  overall_direct.diseaseId AS disease_id,
  diseases.name AS disease_name,
  targets.approvedSymbol AS symbol,
  overall_direct.score AS direct_score,
  overall_direct.evidenceCount AS direct_evidence_count
FROM
  `open-targets-prod.platform.association_overall_direct` AS overall_direct
JOIN
  `open-targets-prod.platform.disease` AS diseases
ON
  overall_direct.diseaseId = diseases.id
JOIN
  `open-targets-prod.platform.target` AS targets
ON
  overall_direct.targetId = targets.id
WHERE
  overall_direct.diseaseId = '{disease_id}'
ORDER BY
  direct_score DESC;
"""

# =================
# GraphQL Queries
# =================

GET_DISEASE_COUNT = """
query GetDiseaseTargetCount($efoId: String!) {
  disease(efoId: $efoId) {
    associatedTargets {
      count
    }
  }
}
"""

GET_DISEASE_TARGETS = """
query GetDiseaseTargets(
  $efoId: String!,
  $pageSize: Int!,
  $pageIndex: Int!
) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: { size: $pageSize, index: $pageIndex }) {
      count
      rows {
        target {
          id
          approvedSymbol
        }
        score
        datasourceScores {
          id
          score
        }
      }
    }
  }
}
"""
