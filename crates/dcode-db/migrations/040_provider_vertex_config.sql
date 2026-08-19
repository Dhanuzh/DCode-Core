-- Google Cloud Vertex AI provider credentials.
--
-- Stored as JSON TEXT mirroring bedrock_config: Vertex carries its project,
-- location and (optional) service-account key outside the shared api_key
-- column, because one Vertex project serves both Gemini and Claude and is
-- identified by project+location rather than by a bearer key.
ALTER TABLE providers ADD COLUMN vertex_config TEXT;
