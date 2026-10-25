"""Track embedding model to prevent mixing vector spaces.

Revision ID: 0002_gemini_embeddings
Revises: 0001_initial
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_gemini_embeddings"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "memories",
        sa.Column(
            "embedding_model",
            sa.String(length=100),
            nullable=False,
            server_default="legacy-openai-text-embedding-3-small",
        ),
    )
    op.alter_column(
        "memories",
        "embedding_model",
        server_default="gemini-embedding-001",
    )


def downgrade() -> None:
    op.drop_column("memories", "embedding_model")
