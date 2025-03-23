"""
Agent AI sử dụng Claude 3.7 với LangGraph và PydanticAI.

Ứng dụng này tạo ra một AI agent có khả năng trò chuyện với người dùng,
sử dụng mô hình Claude 3.7 của Anthropic và framework LangGraph để quản lý
trạng thái hội thoại.
"""

import os
import asyncio
from typing import List, Optional
from pydantic import BaseModel
from pydantic_ai import Agent
from langgraph.graph import StateGraph
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

# Định nghĩa lớp Message để lưu trữ tin nhắn
# (Define Message class to store messages)
class Message(BaseModel):
    """Đại diện cho một tin nhắn trong cuộc trò chuyện."""
    role: str  # 'user' hoặc 'assistant'
    content: str  # Nội dung tin nhắn

# Định nghĩa lớp AgentState để lưu trữ trạng thái của agent
# (Define AgentState class to store agent state)
class AgentState(BaseModel):
    """Trạng thái của agent trong quá trình trò chuyện."""
    messages: List[Message] = []  # Lịch sử tin nhắn
    current_input: Optional[str] = None  # Đầu vào hiện tại từ người dùng
    current_response: Optional[str] = None  # Phản hồi hiện tại từ Claude

# Định nghĩa lớp ClaudeAgent để tương tác với Claude 3.7
# (Define ClaudeAgent class to interact with Claude 3.7)
class ClaudeAgent(BaseModel):
    """Agent AI được hỗ trợ bởi Claude 3.7 để trò chuyện với người dùng."""
    
    async def generate_response(self, messages: List[Message]) -> str:
        """Tạo phản hồi sử dụng Claude 3.7 với PydanticAI."""
        try:
            # Tạo agent sử dụng Claude model từ biến môi trường
            # (Create an agent using Claude model from environment variable)
            agent = Agent(claude_model)
            
            # Định dạng tin nhắn thành chuỗi hội thoại
            # (Format messages as a conversation string)
            conversation = ""
            for msg in messages:
                conversation += f"\n{msg.role.capitalize()}: {msg.content}"
            
            print(f"Sending to Claude: {conversation}")
            
            # Chạy agent với chuỗi hội thoại
            # (Run the agent with the conversation string)
            result = await agent.run(conversation)
            
            return result.data
        except Exception as e:
            error_msg = f"Error communicating with Claude: {e}"
            print(f"Error details: {e}")
            return error_msg

# Định nghĩa các node cho LangGraph
# (Define nodes for LangGraph)
def add_user_input(state: AgentState) -> AgentState:
    """Thêm đầu vào của người dùng vào lịch sử tin nhắn."""
    if state.current_input:
        state.messages.append(Message(role="user", content=state.current_input))
    return state

async def generate_assistant_response(state: AgentState) -> AgentState:
    """Tạo phản hồi từ assistant và thêm vào lịch sử tin nhắn."""
    agent = ClaudeAgent()
    response = await agent.generate_response(state.messages)
    state.current_response = response
    state.messages.append(Message(role="assistant", content=response))
    return state

# Xây dựng đồ thị agent với LangGraph
# (Build agent graph with LangGraph)
def build_agent_graph():
    """Xây dựng đồ thị LangGraph cho agent."""
    # Khởi tạo đồ thị với trạng thái ban đầu
    # (Initialize graph with initial state)
    graph = StateGraph(AgentState)
    
    # Thêm các node vào đồ thị
    # (Add nodes to the graph)
    graph.add_node("add_user_input", add_user_input)
    graph.add_node("generate_response", generate_assistant_response)
    
    # Định nghĩa luồng thực thi
    # (Define execution flow)
    graph.add_edge("add_user_input", "generate_response")
    graph.set_entry_point("add_user_input")
    
    return graph

# Hàm chính để chạy agent
# (Main function to run the agent)
async def run_agent():
    """Chạy agent trong một vòng lặp trò chuyện."""
    # Xây dựng đồ thị
    # (Build the graph)
    graph = build_agent_graph()
    # Biên dịch đồ thị
    # (Compile the graph)
    agent = graph.compile()
    
    print("\n🤖 Welcome to Claude 3.7 Agent Chat!")
    print("Type 'exit' or 'quit' to end the conversation.")
    
    # Thêm thời gian chờ để đảm bảo đầu ra terminal được xử lý
    # (Add a sleep to ensure terminal output is processed)
    import time
    time.sleep(1)
    
    print("\n--- Chat session started ---")
    print("Enter your message below:")
    
    # Khởi tạo trạng thái
    # (Initialize state)
    state = AgentState()
    
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
    
        # Cập nhật trạng thái với đầu vào của người dùng
        # (Update state with user input)
        state.current_input = user_input
    
        try:
            # Chạy agent
            # (Run the agent)
            print("\nSending request to Claude...")
            result = await agent.ainvoke(state)
            
            # Trích xuất và hiển thị phản hồi của Claude
            # (Extract and display Claude's response)
            if 'current_response' in result and result['current_response']:
                print(f"\nClaude: {result['current_response']}")
            elif 'messages' in result:
                # Lấy tin nhắn cuối cùng của assistant
                # (Get the last assistant message)
                assistant_messages = [msg for msg in result['messages'] if msg.role == "assistant"]
                if assistant_messages:
                    last_response = assistant_messages[-1].content
                    print(f"\nClaude: {last_response}")
                else:
                    print("\nClaude: No response generated.")
            else:
                print("\nClaude: No response generated.")
            
            # Cập nhật trạng thái
            # (Update our state)
            if 'messages' in result:
                state = AgentState(
                    messages=result['messages'],
                    current_input=None,
                    current_response=None
                )
            else:
                # Nếu không có tin nhắn trong kết quả, giữ nguyên tin nhắn hiện tại
                # (If no messages in result, keep the existing messages)
                state.current_input = None
                state.current_response = None
        except Exception as e:
            print(f"\nError running the agent: {e}")
            # Tiếp tục cuộc trò chuyện mặc dù có lỗi
            # (Keep the conversation going despite errors)

# Điểm vào chương trình
# (Program entry point)
if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(run_agent())
    except Exception as e:
        print(f"Error running agent: {e}")
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
