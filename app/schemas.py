from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=1024)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: EmailStr
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AgentSessionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    system_prompt: str = Field(default="", max_length=20_000)
    tools_enabled: list[str] = Field(default_factory=list, max_length=20)


class AgentSessionRead(BaseModel):
    id: str
    name: str
    system_prompt: str
    tools_enabled: list[str]
    created_at: datetime


class AgentSessionDetail(AgentSessionRead):
    runs: list["AgentRunSummary"]


class AgentRunSummary(BaseModel):
    id: UUID
    user_message: str
    final_answer: Optional[str]
    status: str
    tokens_used: int
    created_at: datetime


class AgentRunCreate(BaseModel):
    user_message: str = Field(min_length=1, max_length=20_000)


class AgentRunAccepted(BaseModel):
    run_id: UUID


class AgentRunStatus(BaseModel):
    id: UUID
    status: str
    tokens_used: int
    step_count: int
    created_at: datetime


class RunStepRead(BaseModel):
    id: str
    step_type: str
    payload: dict[str, object]
    occurred_at: datetime


class LongTermMemoryRead(BaseModel):
    id: str
    content: str
    source_run_id: Optional[UUID]
    created_at: datetime


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    conversation_id: Optional[UUID] = None


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    created_at: datetime


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    role: str
    content: Optional[str]
    created_at: datetime


class ConversationDetail(ConversationRead):
    messages: list[MessageRead]


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    conversation_id: Optional[UUID] = None


class RunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: Optional[UUID]
    prompt: str
    status: str
    result: Optional[str]
    error: Optional[str]
    created_at: datetime
