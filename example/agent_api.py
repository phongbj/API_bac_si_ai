"""
API Server sử dụng Claude 3.7 với FastAPI.

Ứng dụng này tạo ra một API server có khả năng xử lý yêu cầu từ người dùng,
sử dụng mô hình Claude 3.7 của Anthropic và framework FastAPI để phục vụ
các endpoints API.
"""

import os
import asyncio
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from pydantic_ai import Agent
from langgraph.graph import StateGraph  # Thêm import LangGraph
import logfire
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import uuid
import json

# Import system prompts từ file prompts.py
from prompts import DEFAULT_SYSTEM_PROMPT, PROGRAMMING_ASSISTANT_PROMPT, WRITING_ASSISTANT_PROMPT, EDUCATION_ASSISTANT_PROMPT

# Tải biến môi trường từ file .env
# (Load environment variables from .env file)
load_dotenv()

# Lấy API key và cấu hình từ biến môi trường
# (Get API keys and configuration from environment variables)
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
logfire_token = os.getenv("LOGFIRE_TOKEN")
claude_model = os.getenv("CLAUDE_MODEL", "anthropic:claude-3-7-sonnet-20250219")  # Mặc định nếu không tìm thấy

# Cấu hình Logfire để ghi log
# (Configure Logfire for logging)
if logfire_token:
    try:
        logfire.configure(token=logfire_token)
        print(f"Logfire configured with token: {logfire_token[:5]}...{logfire_token[-5:]}")
        #print("Logfire project URL: https://logfire-us.pydantic.dev/cnguyen1494/starter-project")
        
        # Kiểm tra kết nối Logfire
        logfire.info("API server started", service="claude-agent-api")
    except Exception as e:
        print(f"Error configuring Logfire: {e}")
        print("Logging will be disabled")
else:
    print("Warning: LOGFIRE_TOKEN not found in environment variables")

# Tạo ứng dụng FastAPI
# (Create FastAPI application)
app = FastAPI(
    title="Claude Agent API",
    description="API server for Claude 3.7 agent",
    version="1.0.0"
)

# Thêm CORS middleware để cho phép yêu cầu từ các nguồn khác
# (Add CORS middleware to allow requests from other origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn trong môi trường phát triển
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware để ghi log tất cả các yêu cầu
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware để ghi log tất cả các yêu cầu và phản hồi."""
    # Ghi log yêu cầu
    request_path = request.url.path
    request_method = request.method
    
    # Lấy body của yêu cầu nếu có thể
    request_body = None
    if request_method in ["POST", "PUT"]:
        try:
            body_bytes = await request.body()
            if body_bytes:
                request_body = body_bytes.decode()
        except Exception:
            request_body = None
    
    # Ghi log yêu cầu với Logfire
    if logfire_token:
        try:
            logfire.info(
                "API request",
                service="claude-agent-api",
                method=request_method,
                path=request_path,
                body=request_body
            )
        except Exception as e:
            print(f"Error logging request to Logfire: {e}")
    
    # Xử lý yêu cầu
    response = await call_next(request)
    
    # Ghi log phản hồi với Logfire
    if logfire_token:
        try:
            logfire.info(
                "API response",
                service="claude-agent-api",
                method=request_method,
                path=request_path,
                status_code=response.status_code
            )
        except Exception as e:
            print(f"Error logging response to Logfire: {e}")
    
    return response

# Định nghĩa lớp Message để lưu trữ tin nhắn
# (Define Message class to store messages)
class Message(BaseModel):
    """Đại diện cho một tin nhắn trong cuộc trò chuyện."""
    role: str  # 'user' hoặc 'assistant'
    content: str  # Nội dung tin nhắn

# Định nghĩa lớp cho yêu cầu chat
# (Define class for chat request)
class ChatRequest(BaseModel):
    """Yêu cầu chat từ người dùng."""
    messages: List[Message] = []
    session_id: Optional[str] = None  # ID phiên làm việc, tùy chọn
    system_prompt: Optional[str] = None  # System prompt tùy chọn
    prompt_type: Optional[str] = None  # Loại prompt từ file prompts.py

# Định nghĩa lớp cho phản hồi chat
# (Define class for chat response)
class ChatResponse(BaseModel):
    """Phản hồi chat từ Claude."""
    response: str
    session_id: str  # ID phiên làm việc

# Định nghĩa lớp cho cấu hình phiên làm việc
# (Define class for session configuration)
class SessionConfig(BaseModel):
    """Cấu hình cho một phiên làm việc."""
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

# Lưu trữ lịch sử hội thoại theo phiên làm việc
# (Store conversation history by session)
conversation_history: Dict[str, List[Message]] = {}

# Lưu trữ cấu hình cho mỗi phiên làm việc
# (Store configuration for each session)
session_configs: Dict[str, SessionConfig] = {}

# Dictionary ánh xạ tên prompt đến giá trị prompt
# (Dictionary mapping prompt names to prompt values)
PROMPT_TYPES = {
    "default": DEFAULT_SYSTEM_PROMPT,
    "programming": PROGRAMMING_ASSISTANT_PROMPT,
    "writing": WRITING_ASSISTANT_PROMPT,
    "education": EDUCATION_ASSISTANT_PROMPT
}
    
# Định nghĩa lớp ClaudeAgent để tương tác với Claude 3.7
# (Define ClaudeAgent class to interact with Claude 3.7)
class ClaudeAgent:
    """Agent AI được hỗ trợ bởi Claude 3.7."""
    
    @staticmethod
    async def generate_response(messages: List[Message], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        """Tạo phản hồi sử dụng Claude 3.7 với PydanticAI."""
        try:
            # Tạo agent sử dụng Claude model từ biến môi trường
            # (Create an agent using Claude model from environment variable)
            agent = Agent(claude_model)
            
            # Định dạng tin nhắn thành chuỗi hội thoại với system prompt
            # (Format messages as a conversation string with system prompt)
            conversation = f"\nSystem: {system_prompt}"
            for msg in messages:
                conversation += f"\n{msg.role.capitalize()}: {msg.content}"
            
            # Ghi log yêu cầu
            # (Log the request)
            print(f"Sending to Claude: {conversation}")
            
            # Ghi log yêu cầu với Logfire
            if logfire_token:
                try:
                    logfire.info(
                        "Claude request",
                        service="claude-agent-api",
                        model=claude_model,
                        message_count=len(messages),
                        system_prompt_length=len(system_prompt)
                    )
                except Exception as e:
                    print(f"Error logging Claude request to Logfire: {e}")
            
            # Chạy agent với chuỗi hội thoại
            # (Run the agent with the conversation string)
            result = await agent.run(conversation)
            
            # Ghi log phản hồi với Logfire
            if logfire_token:
                try:
                    logfire.info(
                        "Claude response",
                        service="claude-agent-api",
                        model=claude_model,
                        response_length=len(result.data)
                    )
                except Exception as e:
                    print(f"Error logging Claude response to Logfire: {e}")
            
            return result.data
        except Exception as e:
            error_msg = f"Error communicating with Claude: {e}"
            print(f"Error details: {e}")
            
            # Ghi log lỗi với Logfire
            if logfire_token:
                try:
                    logfire.error(
                        "Claude error",
                        service="claude-agent-api",
                        error=str(e)
                    )
                except Exception as log_e:
                    print(f"Error logging to Logfire: {log_e}")
            
            raise HTTPException(status_code=500, detail=error_msg)

# Hàm để lấy hoặc tạo phiên làm việc mới
# (Function to get or create a new session)
def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Lấy phiên làm việc hiện có hoặc tạo phiên mới nếu không tồn tại."""
    if not session_id or session_id not in conversation_history:
        # Tạo ID phiên mới nếu không có hoặc không tồn tại
        new_session_id = str(uuid.uuid4())
        conversation_history[new_session_id] = []
        session_configs[new_session_id] = SessionConfig()
        
        # Ghi log tạo phiên mới với Logfire
        if logfire_token:
            try:
                logfire.info(
                    "New session created",
                    service="claude-agent-api",
                    session_id=new_session_id
                )
            except Exception as e:
                print(f"Error logging to Logfire: {e}")
        
        return new_session_id
    return session_id

# Định nghĩa endpoint API
# (Define API endpoints)

@app.get("/")
async def root():
    """Endpoint chính để kiểm tra trạng thái máy chủ."""
    return {"status": "online", "model": claude_model}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint để xử lý yêu cầu chat.
    
    Nhận danh sách tin nhắn và trả về phản hồi từ Claude.
    Nếu cung cấp session_id, sẽ tiếp tục hội thoại hiện có.
    Nếu không, sẽ tạo phiên mới.
    """
    try:
        # Lấy hoặc tạo phiên làm việc
        session_id = get_or_create_session(request.session_id)
        
        # Xác định system prompt dựa trên prompt_type hoặc system_prompt
        if request.prompt_type and request.prompt_type in PROMPT_TYPES:
            system_prompt = PROMPT_TYPES[request.prompt_type]
            session_configs[session_id].system_prompt = system_prompt
        elif request.system_prompt:
            session_configs[session_id].system_prompt = request.system_prompt
        
        # Lấy system prompt hiện tại
        system_prompt = session_configs[session_id].system_prompt
        
        # Nếu có tin nhắn mới, thêm vào lịch sử
        if request.messages:
            # Nếu đây là phiên mới hoặc người dùng gửi lịch sử đầy đủ
            if not request.session_id or not conversation_history[session_id]:
                conversation_history[session_id] = request.messages
            else:
                # Nếu chỉ gửi tin nhắn mới nhất, thêm vào lịch sử
                if len(request.messages) == 1 and request.messages[0].role == "user":
                    conversation_history[session_id].append(request.messages[0])
                else:
                    # Nếu gửi nhiều tin nhắn, thay thế lịch sử
                    conversation_history[session_id] = request.messages
        
        # Nếu không có lịch sử, trả về lỗi
        if not conversation_history[session_id]:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Tạo phản hồi từ Claude với system prompt
        response = await ClaudeAgent.generate_response(
            conversation_history[session_id], 
            system_prompt=system_prompt
        )
        
        # Thêm phản hồi vào lịch sử hội thoại
        conversation_history[session_id].append(Message(role="assistant", content=response))
        
        # Ghi log phản hồi chat với Logfire
        if logfire_token:
            try:
                logfire.info(
                    "Chat response sent",
                    service="claude-agent-api",
                    session_id=session_id,
                    message_count=len(conversation_history[session_id])
                )
            except Exception as e:
                print(f"Error logging to Logfire: {e}")
        
        return ChatResponse(response=response, session_id=session_id)
    except Exception as e:
        # Ghi log lỗi với Logfire
        if logfire_token:
            try:
                logfire.error(
                    "Chat error",
                    service="claude-agent-api",
                    error=str(e)
                )
            except Exception as log_e:
                print(f"Error logging to Logfire: {log_e}")
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Lấy lịch sử hội thoại cho một phiên làm việc cụ thể."""
    if session_id not in conversation_history:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return {
        "session_id": session_id, 
        "messages": conversation_history[session_id],
        "system_prompt": session_configs[session_id].system_prompt
    }

@app.get("/prompts")
async def get_available_prompts():
    """Lấy danh sách các loại prompt có sẵn."""
    return {
        "available_prompts": list(PROMPT_TYPES.keys())
    }

@app.put("/sessions/{session_id}/system-prompt")
async def update_system_prompt(session_id: str, system_prompt: str):
    """Cập nhật system prompt cho một phiên làm việc."""
    if session_id not in session_configs:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session_configs[session_id].system_prompt = system_prompt
    
    # Ghi log cập nhật system prompt với Logfire
    if logfire_token:
        try:
            logfire.info(
                "System prompt updated",
                service="claude-agent-api",
                session_id=session_id
            )
        except Exception as e:
            print(f"Error logging to Logfire: {e}")
    
    return {"status": "success", "session_id": session_id, "system_prompt": system_prompt}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Xóa một phiên làm việc."""
    if session_id not in conversation_history:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    del conversation_history[session_id]
    if session_id in session_configs:
        del session_configs[session_id]
    
    # Ghi log xóa phiên với Logfire
    if logfire_token:
        try:
            logfire.info(
                "Session deleted",
                service="claude-agent-api",
                session_id=session_id
            )
        except Exception as e:
            print(f"Error logging to Logfire: {e}")
    
    return {"status": "success", "message": f"Session {session_id} deleted"}

@app.get("/health")
async def health_check():
    """Endpoint để kiểm tra sức khỏe của máy chủ."""
    return {"status": "healthy"}

# Điểm vào chương trình
# (Program entry point)
if __name__ == "__main__":
    # Chạy máy chủ Uvicorn
    # (Run Uvicorn server)
    uvicorn.run(app, host="0.0.0.0", port=8000)
