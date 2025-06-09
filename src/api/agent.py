"""
Lớp GeminiAgent để tương tác với Gemini.

File này định nghĩa lớp GeminiAgent, cung cấp các phương thức
để tạo phản hồi từ Gemini sử dụng PydanticAI.
"""

from typing import List
from pydantic_ai import Agent
from fastapi import HTTPException

from src.api.models import Message
from src.config.settings import GEMINI_MODEL
from src.utils.logger import Logger

class GeminiAgent:
    """Agent AI được hỗ trợ bởi Gemini."""
    
    @staticmethod
    async def generate_response(messages: List[Message], system_prompt: str) -> str:
        """Tạo phản hồi sử dụng Gemini với PydanticAI."""
        try:
            # Tạo agent sử dụng Gemini model từ biến môi trường
            agent = Agent(GEMINI_MODEL)
            
            # Định dạng tin nhắn thành chuỗi hội thoại với system prompt
            conversation = f"\nSystem: {system_prompt}"
            for msg in messages:
                conversation += f"\n{msg.role.capitalize()}: {msg.content}"
            
            # Ghi log yêu cầu
            print(f"Sending to Gemini: {conversation}")
            
            # Ghi log yêu cầu với Logfire
            Logger.info(
                "Gemini request",
                model=GEMINI_MODEL,
                message_count=len(messages),
                system_prompt_length=len(system_prompt)
            )
            
            # Chạy agent với chuỗi hội thoại
            result = await agent.run(conversation)
            
            # Ghi log phản hồi với Logfire
            Logger.info(
                "Gemini response",
                model=GEMINI_MODEL,
                response_length=len(result.data)
            )
            
            return result.data
        except Exception as e:
            error_msg = f"Error communicating with Gemini: {e}"
            print(f"Error details: {e}")
            
            # Ghi log lỗi với Logfire
            Logger.error(
                "Gemini error",
                error=str(e)
            )
            
            raise HTTPException(status_code=500, detail=error_msg)
