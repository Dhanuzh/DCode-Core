-- Rebrand: the builtin agent seeded by migration 001 was named "Aion CLI"
-- (the product's former name). Editing migration 001 directly would rewrite
-- already-applied history and does nothing for databases that already ran
-- it, so this is a follow-up data fix instead. Guarded by the old name so it
-- is a no-op if a user has since renamed the row themselves.
UPDATE agent_metadata
SET name = 'DCode CLI'
WHERE id = '632f31d2' AND name = 'Aion CLI';
