CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS app;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'message_role') THEN
    CREATE TYPE app.message_role AS ENUM ('user', 'assistant', 'system', 'tool');
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'run_status') THEN
    CREATE TYPE app.run_status AS ENUM ('queued', 'running', 'completed', 'failed', 'cancelled');
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS app.conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_key TEXT NOT NULL,
  title TEXT,
  last_message_preview TEXT,
  last_state JSONB NOT NULL DEFAULT '{}'::jsonb,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_message_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  archived_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_conversations_owner_updated
  ON app.conversations (owner_key, updated_at DESC);

CREATE TABLE IF NOT EXISTS app.messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES app.conversations(id) ON DELETE CASCADE,
  seq INTEGER NOT NULL,
  role app.message_role NOT NULL,
  speaker TEXT,
  content TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  edited_at TIMESTAMPTZ,
  UNIQUE (conversation_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_seq
  ON app.messages (conversation_id, seq);

CREATE TABLE IF NOT EXISTS app.message_versions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id UUID NOT NULL REFERENCES app.messages(id) ON DELETE CASCADE,
  version_no INTEGER NOT NULL,
  content TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (message_id, version_no)
);

CREATE TABLE IF NOT EXISTS app.agent_runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES app.conversations(id) ON DELETE CASCADE,
  message_id UUID REFERENCES app.messages(id) ON DELETE SET NULL,
  run_type TEXT NOT NULL,
  status app.run_status NOT NULL DEFAULT 'queued',
  input_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  result_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  error_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_conversation_started
  ON app.agent_runs (conversation_id, started_at DESC);

CREATE TABLE IF NOT EXISTS app.agent_outputs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES app.conversations(id) ON DELETE CASCADE,
  run_id UUID REFERENCES app.agent_runs(id) ON DELETE SET NULL,
  output_type TEXT NOT NULL,
  title TEXT,
  storage_provider TEXT,
  storage_key TEXT,
  mime_type TEXT,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agent_outputs_conversation_created
  ON app.agent_outputs (conversation_id, created_at DESC);