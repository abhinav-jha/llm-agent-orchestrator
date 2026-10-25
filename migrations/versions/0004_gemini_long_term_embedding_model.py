"""Tag long-term embeddings with their provider model.

Revision ID: 0004_gemini_memory_model
Revises: 0003_agentcore
"""
from alembic import op
import sqlalchemy as sa

revision = "0004_gemini_memory_model"
down_revision = "0003_agentcore"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "long_term_memories",
        sa.Column(
            "embedding_model",
            sa.String(length=100),
            nullable=False,
            server_default="legacy-openai-text-embedding-3-small",
        ),
    )
    op.alter_column(
        "long_term_memories",
        "embedding_model",
        server_default="gemini-embedding-001",
    )


def downgrade() -> None:
    op.drop_column("long_term_memories", "embedding_model")
