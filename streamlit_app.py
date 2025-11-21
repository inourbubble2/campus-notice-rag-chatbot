import streamlit as st
import requests
import json
import uuid

# Page configuration
st.set_page_config(
    page_title="캠퍼스 공지사항 챗봇",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stChatInput {
        position: fixed;
        bottom: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🎓 캠퍼스 공지사항 챗봇")
st.markdown("궁금한 학교 공지사항을 물어보세요!")

# Initialize chat history and conversation_id
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("질문을 입력해주세요..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            # Call the API
            response = requests.post(
                "http://localhost:8000/chat",
                json={
                    "question": prompt,
                    "conversation_id": st.session_state.conversation_id
                },
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("blocked"):
                    answer = "🚫 " + data.get("answer", "")
                else:
                    answer = data.get("answer", "죄송합니다. 답변을 가져오는데 실패했습니다.")

                message_placeholder.markdown(answer)

                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                error_msg = f"서버 오류가 발생했습니다. (Status: {response.status_code})"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

        except requests.exceptions.ConnectionError:
            error_msg = "서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인해주세요."
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        except Exception as e:
            error_msg = f"오류가 발생했습니다: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar
with st.sidebar:
    if st.button("대화 내용 지우기"):
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.rerun()
