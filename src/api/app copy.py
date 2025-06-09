"""
API Server sử dụng Claude 3.7 với FastAPI.

Ứng dụng này tạo ra một API server có khả năng xử lý yêu cầu từ người dùng,
sử dụng mô hình Claude 3.7 của Anthropic và framework FastAPI để phục vụ
các endpoints API.
"""

# import asyncio
# from pydantic import BaseModel
# import json
import os
from typing import List, Optional, Dict, Any
import uuid
from src.api.agent import GeminiAgent 
from fastapi import FastAPI, File, HTTPException, Depends, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
from fastapi import Body
from datetime import datetime


# Import từ các module trong dự án
from src.api.models import Message, ChatRequest, ChatResponse, SessionConfig
from src.prompts.prompts import (
    DEFAULT_SYSTEM_PROMPT, 
    PROGRAMMING_ASSISTANT_PROMPT, 
    WRITING_ASSISTANT_PROMPT, 
    EDUCATION_ASSISTANT_PROMPT
)
from src.utils.logger import Logger
from src.config.settings import (
    API_HOST, 
    API_PORT, 
    API_TITLE, 
    API_DESCRIPTION, 
    API_VERSION,
    CORS_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_HEADERS
)

# Tạo ứng dụng FastAPI
# (Create FastAPI application)
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Thêm CORS middleware để cho phép yêu cầu từ các nguồn khác
# (Add CORS middleware to allow requests from other origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
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
    
    # Ghi log yêu cầu
    Logger.info(
        "API request",
        method=request_method,
        path=request_path,
        body=request_body
    )
    
    # Xử lý yêu cầu
    response = await call_next(request)
    
    # Ghi log phản hồi
    Logger.info(
        "API response",
        method=request_method,
        path=request_path,
        status_code=response.status_code
    )
    
    return response

# Lưu trữ lịch sử hội thoại theo phiên làm việc
# (Store conversation history by session)
conversation_history: Dict[str, List[Message]] = {}

# Thư mục chứa tài liệu
DATA_DIR = "data"  # hoặc đường dẫn tuyệt đối nếu cần

# Load tài liệu từ thư mục
documents = []
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

for fname in os.listdir(DATA_DIR):
    if fname.endswith(".txt"):
        with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as f:
            documents.append(f.read())

# Khởi tạo vectorizer và FAISS index
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents).toarray()

# Tạo FAISS index
index = faiss.IndexFlatL2(doc_vectors.shape[1])
index.add(np.array(doc_vectors).astype(np.float32))

def retrieve_relevant_docs(query: str, k: int = 3) -> List[str]:
    """Tìm k tài liệu liên quan nhất đến truy vấn."""
    query_vector = vectorizer.transform([query]).toarray().astype(np.float32)
    _, indices = index.search(query_vector, k)
    return [documents[i] for i in indices[0]]


# Mô hình dữ liệu đầu vào cho tài liệu# Cập nhật DocumentInput nếu cần thiết
class DocumentInput:
    filename: str = None
    file: UploadFile = File(...)

@app.post("/documents/add")
async def add_document(file: UploadFile = File(...)):
    try:
        # Tạo tên file nếu không cung cấp
        filename = file.filename or f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        file_path = os.path.join(DATA_DIR, filename)

        # Lưu file lên server
        with open(file_path, "wb") as f:
            f.write(await file.read())  # Đọc dữ liệu file và lưu vào file mới

        # Nếu cần, bạn có thể xử lý file và cập nhật FAISS ở đây
        # Ví dụ: Chuyển nội dung từ file thành văn bản và thêm vào FAISS

        # Giả sử nội dung file là văn bản
        with open(file_path, "r", encoding="utf-8") as f:
            doc_content = f.read()

        # Thêm vào danh sách documents và cập nhật FAISS
        documents.append(doc_content)
        new_vector = vectorizer.transform([doc_content]).toarray().astype(np.float32)
        index.add(new_vector)

        Logger.info("New document added", filename=filename)

        return {"status": "success", "filename": filename}

    except Exception as e:
        Logger.error("Add document failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi khi thêm tài liệu: {e}")
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

# Hàm để lấy hoặc tạo phiên làm việc mới
# (Function to get or create a new session)
def get_or_create_session(session_id: Optional[str] = None) -> str:
    """
    Lấy phiên làm việc hiện có hoặc tạo phiên mới nếu không tồn tại.
    
    Args:
        session_id: ID phiên làm việc tùy chọn
        
    Returns:
        ID phiên làm việc
    """
    # Nếu không cung cấp session_id, tạo mới
    if not session_id:
        session_id = str(uuid.uuid4())
        conversation_history[session_id] = []
        session_configs[session_id] = SessionConfig(system_prompt=DEFAULT_SYSTEM_PROMPT)
        
        # Ghi log tạo phiên mới
        Logger.info("Session created", session_id=session_id)
    
    # Nếu session_id không tồn tại, tạo mới
    elif session_id not in conversation_history:
        conversation_history[session_id] = []
        session_configs[session_id] = SessionConfig(system_prompt=DEFAULT_SYSTEM_PROMPT)
        
        # Ghi log tạo phiên mới
        Logger.info("Session created", session_id=session_id)
    
    return session_id

# Định nghĩa endpoint API
# (Define API endpoints)

@app.get("/")
async def root():
    """Endpoint chính để kiểm tra trạng thái máy chủ."""
    return {"status": "online", "message": "Claude Agent API is running"}

@app.post("/sessions", response_model=dict)
async def create_session():
    """
    Tạo một session mới và trả về session_id.
    
    Returns:
        session_id: ID của session mới được tạo.
    """
    # Tạo session mới
    session_id = str(uuid.uuid4())
    
    # Khởi tạo lịch sử hội thoại và cấu hình cho session mới
    conversation_history[session_id] = []
    session_configs[session_id] = SessionConfig(system_prompt=DEFAULT_SYSTEM_PROMPT)
    
    # Ghi log tạo session mới
    Logger.info("Session created", session_id=session_id)
    
    # Trả về session ID
    return {"session_id": session_id}

@app.get("/sessions")
async def get_all_sessions():
    """Lấy tất cả các session hiện có."""
    if not conversation_history:
        raise HTTPException(status_code=404, detail="No sessions found")
    
    # Trả về tất cả các session và tin nhắn của chúng
    all_sessions = [
        {"session_id": session_id, "messages": conversation_history[session_id]}
        for session_id in conversation_history
    ]
    
    return {"sessions": all_sessions}

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
        
        # Xác định system prompt
        system_prompt = DEFAULT_SYSTEM_PROMPT
        
        # Nếu có prompt_type, sử dụng prompt tương ứng
        if request.prompt_type and request.prompt_type in PROMPT_TYPES:
            system_prompt = PROMPT_TYPES[request.prompt_type]
            # Cập nhật cấu hình phiên
            session_configs[session_id].system_prompt = system_prompt
            
            # Ghi log cập nhật system prompt
            Logger.info(
                "System prompt updated from type",
                session_id=session_id,
                prompt_type=request.prompt_type
            )
        
        # Nếu có system_prompt tùy chỉnh, sử dụng nó
        elif request.system_prompt:
            system_prompt = request.system_prompt
            # Cập nhật cấu hình phiên
            session_configs[session_id].system_prompt = system_prompt
            
            # Ghi log cập nhật system prompt
            Logger.info(
                "System prompt updated from custom",
                session_id=session_id,
                prompt_length=len(system_prompt)
            )
        # Nếu không, sử dụng system prompt từ cấu hình phiên
        else:
            system_prompt = session_configs[session_id].system_prompt
        
        # Lấy lịch sử hội thoại
        history = conversation_history[session_id]
        
        # Thêm tin nhắn mới vào lịch sử
        for message in request.messages:
            if message not in history:
                history.append(message)
        
        # Ghi log yêu cầu chat
        Logger.info(
            "Chat request",
            session_id=session_id,
            message_count=len(request.messages)
        )
        
        # Tạo phản hồi từ Claude
        # Tạo prompt RAG từ truy vấn cuối cùng
        latest_user_message = [m for m in request.messages if m.role == "user"][-1]
        query_text = latest_user_message.content

        # Truy xuất tài liệu liên quan
        retrieved_docs = retrieve_relevant_docs(query_text)
        rag_context = "\n\n".join(retrieved_docs)

        # Nối context vào system_prompt
        enhanced_prompt = f"{system_prompt}\n\n[Context tài liệu]:\n{rag_context}"

        # Tạo phản hồi từ Claude/Gemini
        response = await GeminiAgent.generate_response(history, enhanced_prompt)

        
        # Thêm phản hồi vào lịch sử
        history.append(Message(role="assistant", content=response))
        
        # Ghi log phản hồi chat
        Logger.info(
            "Chat response",
            session_id=session_id,
            response_length=len(response)
        )
        
        return ChatResponse(response=response, session_id=session_id)
    
    except Exception as e:
        error_msg = f"Error processing chat request: {e}"
        
        # Ghi log lỗi
        Logger.error(
            "Chat error",
            error=str(e)
        )
        
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Lấy lịch sử hội thoại cho một phiên làm việc cụ thể."""
    if session_id not in conversation_history:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return {"session_id": session_id, "messages": conversation_history[session_id]}

@app.get("/prompts")
async def get_available_prompts():
    """Lấy danh sách các loại prompt có sẵn."""
    return {
        "prompt_types": list(PROMPT_TYPES.keys())
    }

@app.put("/sessions/{session_id}/system-prompt")
async def update_system_prompt(session_id: str, config: SessionConfig):
    """Cập nhật system prompt cho một phiên làm việc."""
    if session_id not in session_configs:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Cập nhật system prompt
    session_configs[session_id].system_prompt = config.system_prompt
    
    # Ghi log cập nhật system prompt
    Logger.info(
        "System prompt updated via API",
        session_id=session_id,
        prompt_length=len(config.system_prompt)
    )
    
    return {
        "status": "success",
        "message": "System prompt updated",
        "session_id": session_id
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Xóa một phiên làm việc."""
    if session_id not in conversation_history:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Xóa phiên làm việc
    del conversation_history[session_id]
    
    # Xóa cấu hình phiên
    if session_id in session_configs:
        del session_configs[session_id]
    
    # Ghi log xóa phiên
    Logger.info(
        "Session deleted",
        session_id=session_id
    )
    
    return {
        "status": "success",
        "message": f"Session {session_id} deleted"
    }

@app.get("/")
async def root():
    """Endpoint chính để kiểm tra trạng thái máy chủ."""
    return {"status": "online", "message": "Gemini Agent API is running"}

# Điểm vào chương trình
# (Program entry point)
if __name__ == "__main__":
    # Chạy máy chủ Uvicorn
    # (Run Uvicorn server)
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
