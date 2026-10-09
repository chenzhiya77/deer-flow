### Knowledge Extension (`packages/knowledge-extension/`)

Optional local knowledge base (first-phase RAG), shipped as a self-hosted
in-repo plugin — import `deerflow_knowledge.*`, distribution
`deerflow-knowledge-extension`, entry point `deerflow_knowledge.install:install`.
It is inert until a `plugins:` record in `config.yaml` enables it, and the
record must declare `table_prefix: kb_`.

Host-side seams (all outside this package):
- The `knowledge_search` tool lives in
  `packages/harness/deerflow/tools/builtins/hybrid_search_tool.py` and resolves
  its knowledge scope through a lazy `deerflow_knowledge.access` import.
- `app/gateway/routers/features.py` reports the `knowledge_base` capability
  from `config.knowledge_base.enabled` and fail-closes unless the effective
  `knowledge_search` entry is scope-aware.
- Per-message scope admission is gated in
  `app/gateway/knowledge_scope_admission.py`: the enabled flag, the scope-aware
  provider entry, and the assistant's `tool_groups` (`rag` / `knowledge`). The
  `rag` agent asset opts in via `tool_groups: [rag]` (`agents/assets/rag/config.yaml`).
- Routers (`/api/knowledge-bases/*`, `/api/rag/config`) mount through the
  plugin loader only. `/api/rag/config` is admin-only and masks secrets on
  every read.

Schema: private `MetaData` + own alembic chain (`kb_alembic_version`), upgraded
from the extension service's `start()`; host migrations ignore `kb_*` because
the plugin record declares the prefix.

Tests: `tests/knowledge/` (unit + acceptance style) and the packaging pin
`tests/test_knowledge_extension_packaging.py`.

Ops (install/enable/disable, tables, backup/restore, deployment positions,
rebuild cost): [README.md](README.md).
