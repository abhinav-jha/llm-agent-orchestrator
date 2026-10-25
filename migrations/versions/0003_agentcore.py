"""Add sessions, execution traces, and OpenAI long-term memories.

Revision ID: 0003_agentcore
Revises: 0002_gemini_embeddings
"""
from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003_agentcore"
down_revision = "0002_gemini_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("system_prompt", sa.Text(), server_default="", nullable=False),
        sa.Column("tools_enabled", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_sessions_user_id", "agent_sessions", ["user_id"])

    op.add_column("agent_runs", sa.Column("session_id", sa.String(length=36), nullable=True))
    op.add_column("agent_runs", sa.Column("user_message", sa.Text(), nullable=True))
    op.add_column("agent_runs", sa.Column("final_answer", sa.Text(), nullable=True))
    op.add_column(
        "agent_runs",
        sa.Column("tokens_used", sa.Integer(), server_default="0", nullable=False),
    )
    op.create_foreign_key(
        "fk_agent_runs_session_id_agent_sessions",
        "agent_runs",
        "agent_sessions",
        ["session_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_agent_runs_session_id", "agent_runs", ["session_id"])

    op.create_table(
        "run_steps",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("step_type", sa.String(length=40), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_run_steps_run_id", "run_steps", ["run_id"])

    op.create_table(
        "long_term_memories",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=1536), nullable=False),
        sa.Column("source_run_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_run_id"], ["agent_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_long_term_memories_user_id", "long_term_memories", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_long_term_memories_user_id", table_name="long_term_memories")
    op.drop_table("long_term_memories")
    op.drop_index("ix_run_steps_run_id", table_name="run_steps")
    op.drop_table("run_steps")
    op.drop_index("ix_agent_runs_session_id", table_name="agent_runs")
    op.drop_constraint(
        "fk_agent_runs_session_id_agent_sessions", "agent_runs", type_="foreignkey"
    )
    op.drop_column("agent_runs", "tokens_used")
    op.drop_column("agent_runs", "final_answer")
    op.drop_column("agent_runs", "user_message")
    op.drop_column("agent_runs", "session_id")
    op.drop_index("ix_agent_sessions_user_id", table_name="agent_sessions")
    op.drop_table("agent_sessions")
