# src/main.py

# ==============================
# 📦 IMPORT CÁC THƯ VIỆN CẦN THIẾT
# ==============================
import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np
import faiss
import cv2
import traceback
from PIL import Image, ImageDraw
from sklearn.feature_extraction.text import TfidfVectorizer
from ultralytics import YOLO

from fastapi import (
    FastAPI,
    File,
    Form,
    Query,
    UploadFile,
    HTTPException,
    Depends,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

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
    CORS_ALLOW_HEADERS,
)

from src.api.agent import GeminiAgent
from src.api.models import Message, ChatRequest, ChatResponse, SessionConfig
from src.prompts.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    PROGRAMMING_ASSISTANT_PROMPT,
    WRITING_ASSISTANT_PROMPT,
    EDUCATION_ASSISTANT_PROMPT,
)

# ==============================
# 🔐 IMPORT HÀM XÁC THỰC TỪ auth.py
# ==============================
from src.api.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
)


# ==============================
# 💾 CẤU TRÚC DỮ LIỆU & BIẾN TOÀN CỤC
# ==============================
conversation_history: Dict[str, List[Message]] = {}
session_configs: Dict[str, SessionConfig] = {}

PROMPT_TYPES = {
    "default": DEFAULT_SYSTEM_PROMPT,
    "programming": PROGRAMMING_ASSISTANT_PROMPT,
    "writing": WRITING_ASSISTANT_PROMPT,
    "education": EDUCATION_ASSISTANT_PROMPT,
}

# Thư mục lưu tài liệu văn bản để build FAISS
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Đọc tài liệu .txt để đưa vào index
documents = []
for fname in os.listdir(DATA_DIR):
    if fname.endswith(".txt"):
        with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as f:
            documents.append(f.read())

vectorizer = TfidfVectorizer()
if documents:
    doc_vectors = vectorizer.fit_transform(documents).toarray()
    index = faiss.IndexFlatL2(doc_vectors.shape[1])
    index.add(np.array(doc_vectors).astype(np.float32))
else:
    # Nếu chưa có tài liệu nào, khởi tạo index 1 chiều tạm
    index = faiss.IndexFlatL2(1)

def retrieve_relevant_docs(query: str, k: int = 3) -> List[str]:
    if not documents:
        return []
    query_vector = vectorizer.transform([query]).toarray().astype(np.float32)
    _, indices = index.search(query_vector, k)
    return [documents[i] for i in indices[0]]

# ==============================
# 📜 Pydantic Models
# ==============================
class User(BaseModel):
    """
    Model trả về khi gọi /user/me
    """
    username: str


# ==============================
# 🚀 KHỞI TẠO FASTAPI ỨNG DỤNG
# ==============================
app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION)

# Cấu hình CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

# Mount thư mục static “images” để trả về ảnh
app.mount("/images", StaticFiles(directory="images"), name="images")

# ==============================
# 📊 MIDDLEWARE GHI LOG
# ==============================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_path = request.url.path
    request_method = request.method

    request_body = None
    if request_method in ["POST", "PUT"]:
        try:
            body_bytes = await request.body()
            if body_bytes:
                request_body = body_bytes.decode()
        except Exception:
            request_body = None

    Logger.info(
        "API request", method=request_method, path=request_path, body=request_body
    )
    response = await call_next(request)
    Logger.info(
        "API response",
        method=request_method,
        path=request_path,
        status_code=response.status_code,
    )
    return response

# ==============================
# 📎 ENDPOINT: login/logout
# ==============================
@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Nhận POST form-encoded: username, password.
    Trả về JSON: { "access_token": "...", "token_type": "bearer" }
    """
    username = form_data.username
    password = form_data.password

    if not authenticate_user(username, password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

@app.post("/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    """
    API logout đơn giản: nếu client đã gửi kèm token (Bearer),
    thì server chỉ trả về thông báo, và client tự xóa token đi.
    """
    return {"status": "success", "message": "Đã logout. Vui lòng xóa token ở phía client."}

# ==============================
# 📎 ENDPOINT: user/me
# ==============================
@app.get("/user/me", response_model=User)
async def read_users_me(current_username: str = Depends(get_current_user)):
    """
    Nếu token hợp lệ, get_current_user() trả về tên đăng nhập (username).
    Chúng ta bọc thành Pydantic model User để trả về.
    """
    return User(username=current_username)

# ==============================
# 📎 ENDPOINT: THÊM TÀI LIỆU MỚI
# ==============================
@app.post("/documents/add")
async def add_document(file: UploadFile = File(...)):
    """
    Upload 1 file .txt, lưu vào thư mục DATA_DIR, update FAISS index.
    """
    try:
        filename = (
            file.filename
            or f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        file_path = os.path.join(DATA_DIR, filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        with open(file_path, "r", encoding="utf-8") as f:
            doc_content = f.read()

        documents.append(doc_content)
        new_vector = vectorizer.transform([doc_content]).toarray().astype(np.float32)
        index.add(new_vector)

        Logger.info("New document added", filename=filename)
        return {"status": "success", "filename": filename}
    except Exception as e:
        Logger.error("Add document failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi khi thêm tài liệu: {e}")

# ==============================
# 🧠 HÀM TIỆN ÍCH SESSION
# ==============================
def get_or_create_session(session_id: Optional[str] = None) -> str:
    """
    Tạo mới hoặc lấy session hiện có (dựa vào session_id).
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in conversation_history:
        conversation_history[session_id] = []
        session_configs[session_id] = SessionConfig(system_prompt=DEFAULT_SYSTEM_PROMPT)
        Logger.info("Session created", session_id=session_id)

    return session_id

# ==============================
# 🌐 API ENDPOINTS CÒN LẠI
# ==============================
@app.get("/")
async def root():
    return {"status": "online", "message": "Gemini Agent API is running"}

@app.post("/sessions", response_model=dict)
async def create_session():
    session_id = str(uuid.uuid4())
    conversation_history[session_id] = []
    session_configs[session_id] = SessionConfig(system_prompt=DEFAULT_SYSTEM_PROMPT)
    Logger.info("Session created", session_id=session_id)
    return {"session_id": session_id}

@app.get("/sessions")
async def get_all_sessions():
    if not conversation_history:
        raise HTTPException(status_code=404, detail="No sessions found")
    all_sessions = [
        {"session_id": sid, "messages": msgs}
        for sid, msgs in conversation_history.items()
    ]
    return {"sessions": all_sessions}

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in conversation_history:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"session_id": session_id, "messages": conversation_history[session_id]}

@app.put("/sessions/{session_id}/system-prompt")
async def update_system_prompt(session_id: str, config: SessionConfig):
    if session_id not in session_configs:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    session_configs[session_id].system_prompt = config.system_prompt
    Logger.info(
        "System prompt updated via API",
        session_id=session_id,
        prompt_length=len(config.system_prompt),
    )
    return {
        "status": "success",
        "message": "System prompt updated",
        "session_id": session_id,
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in conversation_history:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    del conversation_history[session_id]
    session_configs.pop(session_id, None)
    Logger.info("Session deleted", session_id=session_id)
    return {"status": "success", "message": f"Session {session_id} deleted"}

@app.get("/prompts")
async def get_available_prompts():
    return {"prompt_types": list(PROMPT_TYPES.keys())}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = get_or_create_session(request.session_id)
        system_prompt = session_configs[session_id].system_prompt

        if request.prompt_type and request.prompt_type in PROMPT_TYPES:
            system_prompt = PROMPT_TYPES[request.prompt_type]
            session_configs[session_id].system_prompt = system_prompt
            Logger.info(
                "System prompt updated from type",
                session_id=session_id,
                prompt_type=request.prompt_type,
            )
        elif request.system_prompt:
            system_prompt = request.system_prompt
            session_configs[session_id].system_prompt = system_prompt
            Logger.info(
                "System prompt updated from custom",
                session_id=session_id,
                prompt_length=len(system_prompt),
            )

        history = conversation_history[session_id]
        for message in request.messages:
            if message not in history:
                history.append(message)

        Logger.info(
            "Chat request", session_id=session_id, message_count=len(request.messages)
        )

        latest_user_message = [m for m in request.messages if m.role == "user"][-1]
        query_text = latest_user_message.content
        retrieved_docs = retrieve_relevant_docs(query_text)
        rag_context = "\n\n".join(retrieved_docs)

        enhanced_prompt = f"{system_prompt}\n\n[Context tài liệu]:\n{rag_context}"
        response = await GeminiAgent.generate_response(history, enhanced_prompt)
        history.append(Message(role="assistant", content=response))

        Logger.info(
            "Chat response", session_id=session_id, response_length=len(response)
        )
        return ChatResponse(response=response, session_id=session_id)

    except Exception as e:
        Logger.error("Chat error", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Error processing chat request: {e}"
        )

# ==============================
# ▶️ BIÊN DỊCH MÔ HÌNH YOLO
# ==============================
base_path = os.path.dirname(__file__)  # Đường dẫn tới src/

detection_model = YOLO(
    os.path.normpath(
        os.path.join(base_path, "..", "runs", "detect", "train", "weights", "best.pt")
    )
)
classification_model = YOLO(
    os.path.normpath(
        os.path.join(base_path, "..", "runs", "classify", "train", "weights", "best.pt")
    )
)
segmentation_model = YOLO(
    os.path.normpath(
        os.path.join(base_path, "..", "runs", "segment", "train", "weights", "best.pt")
    )
)

IMAGE_DIR = "images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


@app.post("/medical/detect")
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = Form(
        0.25,
        ge=0.0,
        le=1.0,
        description="Ngưỡng confidence (0.0 - 1.0) để YOLO lọc các box có độ tin cậy thấp hơn"
    ),
    # user: str = Depends(get_current_user)  # Nếu cần auth
):
    try:
        # Lưu file
        image_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1]
        image_path = os.path.join(IMAGE_DIR, f"{image_id}{ext}")
        with open(image_path, "wb") as f:
            f.write(await file.read())

        # Gọi YOLO với confidence lấy từ form
        results = detection_model.predict(source=image_path, conf=confidence)

        # Xây dựng response
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                detections.append({
                    "class_id": cls_id,
                    "confidence": round(conf_score, 3),
                    "bbox": [round(coord, 2) for coord in bbox]
                })

        # Tạo ảnh annotated
        r = results[0]
        annotated_bgr = r.plot().astype(np.uint8)
        annotated_filename = f"{image_id}_annotated{ext}"
        annotated_path = os.path.join(IMAGE_DIR, annotated_filename)
        cv2.imwrite(annotated_path, annotated_bgr)

        return {
            "status": "success",
            "type": "detection",
            "detections": detections,
            "annotated_image": f"/images/{annotated_filename}"
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")


@app.post("/medical/classify")
async def classify_image(
    file: UploadFile = File(...),
    #user: str = Depends(get_current_user)  # Nếu cần auth, bỏ comment
):
    try:
        image_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1]
        image_path = os.path.join(IMAGE_DIR, f"{image_id}{ext}")

        with open(image_path, "wb") as f:
            f.write(await file.read())

        results = classification_model(image_path)
        result = results[0]

        class_idx = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        class_name = result.names[class_idx].upper()

        image_bgr = cv2.imread(image_path)
        label = f"{class_name} ({confidence * 100:.1f}%)"
        cv2.putText(
            image_bgr,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        annotated_filename = f"{image_id}_annotated{ext}"
        annotated_path = os.path.join(IMAGE_DIR, annotated_filename)
        cv2.imwrite(annotated_path, image_bgr)

        return {
            "status": "success",
            "type": "classification",
            "class_id": class_idx,
            "confidence": round(confidence, 3),
            "annotated_image": f"/images/{annotated_filename}"
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")


@app.post("/medical/segment")
async def segment_image(
    file: UploadFile = File(...),
    #user: str = Depends(get_current_user)  # Nếu cần auth, bỏ comment
):
    try:
        image_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1]
        image_path = os.path.join(IMAGE_DIR, f"{image_id}{ext}")

        with open(image_path, "wb") as f:
            f.write(await file.read())

        results = segmentation_model(image_path)

        image_bgr = cv2.imread(image_path)
        H, W = image_bgr.shape[:2]

        mask_out = None
        confidences = []

        for result in results:
            for i, mask_tensor in enumerate(result.masks.data):
                mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                mask_resized = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_LINEAR)
                mask_out = mask_resized.copy() if mask_out is None else cv2.bitwise_or(mask_out, mask_resized)
                if hasattr(result.masks, "scores"):
                    score = float(result.masks.scores[i])
                    confidences.append(round(score * 100, 1))

        if mask_out is None:
            mask_out = np.zeros((H, W), dtype=np.uint8)

        mask_filename = f"{image_id}_mask.png"
        mask_path = os.path.join(IMAGE_DIR, mask_filename)
        cv2.imwrite(mask_path, mask_out)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_im)

        for result in results:
            for mask_ in result.masks:
                polygon_pts = mask_.xy[0]
                poly_coords = [(int(x), int(y)) for x, y in polygon_pts]
                draw.polygon(poly_coords, outline=(0, 255, 0), width=5)

        annotated_poly_filename = f"{image_id}_annotated.png"
        annotated_poly_path = os.path.join(IMAGE_DIR, annotated_poly_filename)
        pil_im.save(annotated_poly_path)

        return {
            "status": "success",
            "type": "segmentation",
            "mask_image": f"/images/{mask_filename}",
            "annotated_image": f"/images/{annotated_poly_filename}",
            "confidences": confidences
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")


# ==============================
# ▶️ CHẠY SERVER (Entry point)
# ==============================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
