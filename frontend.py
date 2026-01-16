"""
Streamlit Frontend для RAG вопросно-ответной системы
Чат-интерфейс с поддержкой нескольких диалогов
"""
import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import uuid

from src import RAG
from src.db_utils.history_utils import (
    init_history_table,
    log_query,
    get_all_history,
    get_history_by_dialogue,
    search_history,
    get_history_stats,
    delete_history,
    get_recent_dialogues
)


# --- Инициализация RAG ---
@st.cache_resource(show_spinner=False)
def get_rag():
    """Initialize RAG once and cache it"""
    return RAG(
        embed_model_name = "Qwen/Qwen3-Embedding-0.6B",
        embed_index_name = "recursive_Qwen3-Embedding-0.6B"
    )


# --- Session State Management ---
def init_session_state():
    """Initialize session state variables for chat support"""
    if "current_dialogue_id" not in st.session_state:
        st.session_state.current_dialogue_id = None
    if "chat_list" not in st.session_state:
        st.session_state.chat_list = []
    if "current_chat_messages" not in st.session_state:
        st.session_state.current_chat_messages = []
    if "chat_names" not in st.session_state:
        st.session_state.chat_names = {}  # {dialogue_id: custom_name}
    if "chats_loaded" not in st.session_state:
        st.session_state.chats_loaded = False


def generate_dialogue_id() -> str:
    """Generate unique dialogue ID"""
    return f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def get_chat_display_name(dialogue_id: str, first_query: str = None) -> str:
    """Get display name for chat"""
    if dialogue_id in st.session_state.chat_names:
        return st.session_state.chat_names[dialogue_id]
    
    if first_query:
        # Use first 40 chars of first query as name
        name = first_query[:40] + "..." if len(first_query) > 40 else first_query
        st.session_state.chat_names[dialogue_id] = name
        return name
    
    return "Новый диалог"


# --- Chat Management Functions ---

def load_chats_list():
    """Load all available chats from database"""
    try:
        dialogues = get_recent_dialogues(limit=50)
        st.session_state.chat_list = dialogues
        st.session_state.chats_loaded = True
        
        # If no current chat selected and chats exist, select the first one
        if not st.session_state.current_dialogue_id and dialogues:
            switch_to_chat(dialogues[0]["dialogue_id"])
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке чатов: {e}")
        st.session_state.chat_list = []


def create_new_chat():
    """Create a new chat"""
    new_id = generate_dialogue_id()
    st.session_state.current_dialogue_id = new_id
    st.session_state.current_chat_messages = []
    return new_id


def switch_to_chat(dialogue_id: str):
    """Switch to an existing chat"""
    st.session_state.current_dialogue_id = dialogue_id
    load_chat_messages(dialogue_id)


def load_chat_messages(dialogue_id: str):
    """Load messages for a specific chat"""
    try:
        history = get_history_by_dialogue(dialogue_id)
        st.session_state.current_chat_messages = history
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке сообщений: {e}")
        st.session_state.current_chat_messages = []


def send_message(query: str) -> Optional[Dict]:
    """Send a message in current chat"""
    try:
        if not st.session_state.current_dialogue_id:
            create_new_chat()
        
        # Get RAG and invoke with history
        rag = get_rag()
        
        # Pass current chat history to RAG (it will use last N messages internally for enrichment)
        result = rag.invoke(query, history=st.session_state.current_chat_messages)
        
        # Log to history
        query_id = log_query(
            query=query,
            answer=result.get("answer", ""),
            reason=result.get("reason", ""),
            dialogue_id=st.session_state.current_dialogue_id
        )
        
        result["query_id"] = query_id
        
        # Update current chat messages
        load_chat_messages(st.session_state.current_dialogue_id)
        
        # Reload chats list to update
        load_chats_list()
        
        return result
    except Exception as e:
        st.error(f"❌ Ошибка при отправке сообщения: {e}")
        return None


def delete_chat(dialogue_id: str) -> bool:
    """Delete a chat"""
    try:
        delete_history(dialogue_id=dialogue_id)
        
        # If deleted current chat, switch to another or create new
        if st.session_state.current_dialogue_id == dialogue_id:
            st.session_state.current_dialogue_id = None
            st.session_state.current_chat_messages = []
        
        # Reload chats
        load_chats_list()
        
        return True
    except Exception as e:
        st.error(f"❌ Ошибка при удалении чата: {e}")
        return False




# --- Page: Chat Interface ---
def page_chat():
    """Main chat interface page"""
    
    # Custom CSS to fix chat input at the bottom + keyboard shortcuts
    st.markdown("""
        <style>
        /* Fix chat input at the bottom of main content area */
        section[data-testid="stSidebar"] ~ div .stChatInput {
            position: fixed;
            bottom: 0;
            background: white;
            padding: 1rem;
            z-index: 999;
            border-top: 1px solid #e6e6e6;
            margin-left: 0;
        }
        
        /* Add padding to main content to prevent overlap with fixed input */
        .main .block-container {
            padding-bottom: 100px;
        }
        
        /* Dark mode support */
        [data-testid="stAppViewContainer"][data-theme="dark"] section[data-testid="stSidebar"] ~ div .stChatInput {
            background: rgb(14, 17, 23);
            border-top: 1px solid #333;
        }
        
        /* Adjust width to account for sidebar */
        @media (min-width: 768px) {
            section[data-testid="stSidebar"] ~ div .stChatInput {
                left: var(--sidebar-width, 21rem);
                right: 0;
            }
        }
        
        /* When sidebar is collapsed */
        section[data-testid="stSidebar"][aria-expanded="false"] ~ div .stChatInput {
            left: 0;
        }
        </style>
        
        <script>
        // Add keyboard shortcuts support
        document.addEventListener('DOMContentLoaded', function() {
            // Find chat input field
            const observer = new MutationObserver(function(mutations) {
                const chatInput = document.querySelector('textarea[data-testid="stChatInput"]');
                if (chatInput && !chatInput.hasAttribute('data-shortcut-attached')) {
                    chatInput.setAttribute('data-shortcut-attached', 'true');
                    
                    // Add keyboard event listener
                    chatInput.addEventListener('keydown', function(e) {
                        // Enter (without Shift) - send message
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            // Trigger the send button
                            const sendButton = document.querySelector('button[kind="primary"]');
                            if (sendButton) {
                                sendButton.click();
                            }
                        }
                        // Ctrl+Enter or Cmd+Enter - send message (alternative)
                        else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                            e.preventDefault();
                            const sendButton = document.querySelector('button[kind="primary"]');
                            if (sendButton) {
                                sendButton.click();
                            }
                        }
                        // Shift+Enter - new line (default behavior)
                    });
                }
            });
            
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        });
        </script>
    """, unsafe_allow_html=True)
    
    # Check if we have a current chat
    if not st.session_state.current_dialogue_id:
        # Show welcome screen
        st.title("💬 Чат с RAG системой")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("👋 Добро пожаловать! Создайте новый чат или выберите существующий из списка слева.")
            
            if st.button("🆕 Начать новый чат", type="primary", use_container_width=True):
                create_new_chat()
                st.rerun()
        
        return
    
    # Display chat header
    if st.session_state.current_chat_messages:
        chat_name = get_chat_display_name(
            st.session_state.current_dialogue_id,
            st.session_state.current_chat_messages[0]["query"]
        )
    else:
        chat_name = "Новый диалог"
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(f"💬 {chat_name}")
    with col2:
        if st.button("🗑️ Удалить чат", use_container_width=True):
            if delete_chat(st.session_state.current_dialogue_id):
                st.success("✅ Чат удален")
                st.rerun()
    
    st.markdown("---")
    
    # Chat messages container
    if not st.session_state.current_chat_messages:
        st.info("📝 Начните диалог, задав первый вопрос ниже")
    else:
        # Display all messages
        for msg in st.session_state.current_chat_messages:
            # User message
            with st.chat_message("user"):
                st.markdown(msg["query"])
                timestamp_str = msg.get("timestamp", "")
                try:
                    dt = datetime.fromisoformat(timestamp_str)
                    st.caption(f"🕐 {dt.strftime('%H:%M:%S')}")
                except:
                    pass
            
            # Assistant message
            with st.chat_message("assistant"):
                st.markdown(msg["answer"])
                
                # Show reasoning in expander
                if msg.get("reason"):
                    with st.expander("📝 Обоснование"):
                        st.markdown(msg["reason"])
    
    # Input area - fixed at the bottom via CSS
    query = st.chat_input(
        "Введите ваш вопрос...",
        key="chat_input"
    )
    
    if query:
        # Send message and get response
        with st.spinner("🤔 Думаю..."):
            result = send_message(query)
            
            if result:
                st.rerun()



# --- Main App ---
def main():
    st.set_page_config(
        page_title="RAG Chat System",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize history table on startup
    try:
        init_history_table()
    except Exception as e:
        st.error(f"⚠️ Не удалось инициализировать таблицу истории: {e}")
    
    # Initialize session state
    init_session_state()
    
    # Load chats list if not loaded
    if not st.session_state.chats_loaded:
        load_chats_list()
    
    # Sidebar
    with st.sidebar:
        st.title("💬 RAG Chat")
        
        # New chat button
        if st.button("➕ Новый чат", use_container_width=True, type="primary"):
            create_new_chat()
            st.rerun()
        
        st.markdown("---")
        
        # Chats list
        st.subheader("📝 Ваши чаты")
        
        if not st.session_state.chat_list:
            st.info("Нет чатов. Создайте новый!")
        else:
            # Display chats
            for chat in st.session_state.chat_list:
                dialogue_id = chat["dialogue_id"]
                message_count = chat.get("message_count", 0)
                started_at = chat.get("started_at", "")
                
                # Get chat name (only load history if chat has messages)
                if message_count > 0:
                    history = get_history_by_dialogue(dialogue_id)
                    first_query = history[0]["query"] if history else None
                else:
                    first_query = None
                chat_name = get_chat_display_name(dialogue_id, first_query)
                
                # Format time
                try:
                    dt = datetime.fromisoformat(started_at)
                    time_str = dt.strftime('%d.%m %H:%M')
                except:
                    time_str = ""
                
                # Check if this is current chat
                is_current = dialogue_id == st.session_state.current_dialogue_id
                
                # Format button text with chat name and metadata
                button_text = f"{'📌' if is_current else '💬'} {chat_name}\n💬 {message_count} • {time_str}"
                
                if st.button(
                    button_text,
                    key=f"chat_{dialogue_id}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary"
                ):
                    switch_to_chat(dialogue_id)
                    st.rerun()
    
    # Main content area
    page_chat()


if __name__ == "__main__":
    main()
