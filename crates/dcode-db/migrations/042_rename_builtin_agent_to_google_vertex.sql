-- Follow-up to migration 041: the builtin agent's display name is now
-- "Google Cloud Vertex" (it is the one agent that reads models from the
-- Google Cloud Vertex AI provider). Guarded by the previous name so it is a
-- no-op if a user has since renamed the row themselves.
UPDATE agent_metadata
SET name = 'Google Cloud Vertex'
WHERE id = '632f31d2' AND name = 'DCode CLI';
