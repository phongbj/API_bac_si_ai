"""
Ứng dụng chat đơn giản sử dụng Claude 3.7 với PydanticAI.

Đây là phiên bản đơn giản của ứng dụng chat, sử dụng mô hình Claude 3.7 
của Anthropic và PydanticAI để tạo ra một giao diện trò chuyện cơ bản.
"""

import os
import asyncio
from pydantic_ai import Agent
import logfire
from dotenv import load_dotenv

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
    logfire.configure(token=logfire_token)
else:
    print("Warning: LOGFIRE_TOKEN not found in environment variables")

async def run_chat():
    """
    Chạy ứng dụng chat đơn giản với Claude 3.7.
    
    Hàm này tạo ra một vòng lặp trò chuyện, nơi người dùng có thể 
    nhập tin nhắn và nhận phản hồi từ Claude 3.7.
    """
    # Tạo agent sử dụng Claude model từ biến môi trường
    # (Create an agent using Claude model from environment variable)
    agent = Agent(claude_model)
    
    print("\n🤖 Welcome to Simple Claude 3.7 Chat!")
    print("Type 'exit' or 'quit' to end the conversation.")
    
    # Thêm thời gian chờ để đảm bảo đầu ra terminal được xử lý
    # (Add a sleep to ensure terminal output is processed)
    import time
    time.sleep(1)
    
    print("\n--- Chat session started ---")
    print("Enter your message below:")
    
    # Lưu trữ lịch sử hội thoại dưới dạng chuỗi
    # (Store conversation history as a string)
    conversation = ""
    
    # Cờ để theo dõi đầu vào đầu tiên
    # (Flag to track first input)
    first_input = True
    
    while True:
        # Lấy đầu vào từ người dùng
        # (Get user input)
        user_input = input("\nYou: ")
        
        # Xử lý đầu vào đầu tiên đặc biệt - có thể chứa URL Logfire
        # (Handle first input specially - it might contain the Logfire URL)
        if first_input:
            first_input = False
            if "logfire-us.pydantic.dev" in user_input.lower():
                print("Starting a new conversation. Please enter your message.")
                continue
        
        # Bỏ qua đầu vào trống
        # (Skip empty inputs)
        if not user_input:
            continue
        
        # Kiểm tra lệnh thoát
        # (Check for exit command)
        if user_input.lower() in ["exit", "quit"]:
            print("\nGoodbye! 👋")
            break
        
        # Thêm đầu vào của người dùng vào lịch sử hội thoại
        # (Add user input to conversation history)
        conversation += f"\nUser: {user_input}"
        
        try:
            # Gửi lịch sử hội thoại đến Claude và nhận phản hồi
            # (Send conversation history to Claude and get response)
            print("\nSending request to Claude...")
            result = await agent.run(conversation)
            
            # Lấy phản hồi từ Claude
            # (Get response from Claude)
            response = result.data
            
            # Hiển thị phản hồi
            # (Display response)
            print(f"\nClaude: {response}")
            
            # Thêm phản hồi vào lịch sử hội thoại
            # (Add response to conversation history)
            conversation += f"\nAssistant: {response}"
            
        except Exception as e:
            print(f"\nError communicating with Claude: {e}")

# Điểm vào chương trình
# (Program entry point)
if __name__ == "__main__":
    try:
        asyncio.run(run_chat())
    except Exception as e:
        print(f"Error running chat: {e}")
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
