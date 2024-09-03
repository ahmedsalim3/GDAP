-- Run these SQL queries in BigQuery for your specific disease and download the results as CSV/JSON: https://console.cloud.google.com/bigquery?sq=482968641421:b7b6580ad1204ef49b43fe66dcbeb478

-- Database tables: https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/24.06/output/etl/json/
-- More info: https://community.opentargets.org/t/returning-all-associations-data-using-the-platform-api/324/2


-- Query 1: Retrieve detailed overall association scores and related data (full-data)
SELECT
  overall_direct.targetId AS target_id,
  targets.approvedSymbol AS target_approved_symbol,
  overall_direct.diseaseId AS disease_id,
  diseases.name AS disease_name,
  overall_direct.score AS overall_direct_score,
  datasource_direct.datasourceId AS datasource_id,
  datasource_direct.datatypeId AS datasource_datatype_id,
  datasource_direct.score AS datasource_direct_score,
  datasource_direct.evidenceCount AS datasource_direct_evidence_count,
  datatype_direct.datatypeId AS datatype_id,
  datatype_direct.score AS datatype_direct_score,
  datatype_direct.evidenceCount AS datatype_direct_evidence_count,
  overall_indirect.score AS overall_indirect_score,
  overall_indirect.evidenceCount AS overall_indirect_evidence_count
FROM
  `open-targets-prod.platform.associationByOverallDirect` AS overall_direct
JOIN
  `open-targets-prod.platform.diseases` AS diseases
ON
  overall_direct.diseaseId = diseases.id
JOIN
  `open-targets-prod.platform.targets` AS targets
ON
  overall_direct.targetId = targets.id
JOIN
  `open-targets-prod.platform.associationByDatasourceDirect` AS datasource_direct
ON
  overall_direct.diseaseId = datasource_direct.diseaseId
AND
  overall_direct.targetId = datasource_direct.targetId
JOIN
  `open-targets-prod.platform.associationByDatatypeDirect` AS datatype_direct
ON
  overall_direct.diseaseId = datatype_direct.diseaseId
AND
  overall_direct.targetId = datatype_direct.targetId
JOIN
  `open-targets-prod.platform.associationByOverallIndirect` AS overall_indirect
ON
  overall_direct.diseaseId = overall_indirect.diseaseId
AND
  overall_direct.targetId = overall_indirect.targetId
WHERE
  overall_direct.diseaseId = 'EFO_0000319' -- cardiovascular disease EFO-ID
ORDER BY
  overall_direct.score DESC;



  -- Query 2: Retrieve the overall direct association score (mini data)

SELECT
  associations.targetId AS target_id,
  targets.approvedSymbol AS target_approved_symbol,
  associations.diseaseId AS disease_id,
  diseases.name AS disease_name,
  associations.score AS overall_association_score
  
FROM
  `open-targets-prod.platform.associationByOverallDirect` AS associations
JOIN
  `open-targets-prod.platform.diseases` AS diseases
ON
  associations.diseaseId = diseases.id
JOIN
  `open-targets-prod.platform.targets` AS targets
ON
  associations.targetId = targets.id
WHERE
  diseaseId = 'MONDO_0004975' -- Alzheimer disease EFO-ID
ORDER BY
  associations.score DESC