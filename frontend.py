"""
Streamlit Frontend для RAG вопросно-ответной системы
История запросов управляется через Backend API
"""
import streamlit as st
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# --- Configuration ---
BACKEND_URL = "http://localhost:8000"


# --- Session State Management ---
def init_session_state():
    """Initialize session state variables for dialogue support"""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🔍 Запрос"
    if "current_dialogue_id" not in st.session_state:
        st.session_state.current_dialogue_id = generate_dialogue_id()
    if "dialogue_messages" not in st.session_state:
        # For future dialogue support - stores messages in current session
        st.session_state.dialogue_messages = []
    if "history_cache" not in st.session_state:
        st.session_state.history_cache = None
    if "history_cache_time" not in st.session_state:
        st.session_state.history_cache_time = None


def generate_dialogue_id() -> str:
    """Generate unique dialogue ID for future multi-turn conversation support"""
    return f"dialogue_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# --- API Communication ---

def check_backend_health() -> bool:
    """Check if backend is available"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def send_query_to_backend(query: str, dialogue_id: Optional[str] = None) -> Optional[Dict]:
    """Send query to RAG backend"""
    try:
        payload = {"query": query}
        if dialogue_id:
            payload["dialogue_id"] = dialogue_id
            
        response = requests.post(
            f"{BACKEND_URL}/rag",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Не удается подключиться к backend серверу. Убедитесь, что сервер запущен на http://localhost:8000")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Превышено время ожидания ответа от сервера")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Ошибка HTTP: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Неожиданная ошибка: {e}")
        return None


def get_history_from_backend(limit: int = 100, offset: int = 0) -> List[Dict]:
    """Get history from backend API"""
    try:
        response = requests.get(
            f"{BACKEND_URL}/history",
            params={"limit": limit, "offset": offset},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ Ошибка при получении истории: {e}")
        return []


def get_stats_from_backend() -> Dict:
    """Get statistics from backend API"""
    try:
        response = requests.get(f"{BACKEND_URL}/history/stats", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"total_queries": 0, "unique_dialogues": 0}


def search_history_in_backend(search_text: str, limit: int = 50) -> List[Dict]:
    """Search history via backend API"""
    try:
        response = requests.get(
            f"{BACKEND_URL}/history/search",
            params={"q": search_text, "limit": limit},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ Ошибка при поиске: {e}")
        return []


def get_dialogues_from_backend(limit: int = 10) -> List[Dict]:
    """Get dialogues list from backend API"""
    try:
        response = requests.get(
            f"{BACKEND_URL}/history/dialogues",
            params={"limit": limit},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return []


def get_dialogue_history_from_backend(dialogue_id: str) -> List[Dict]:
    """Get specific dialogue history from backend API"""
    try:
        response = requests.get(
            f"{BACKEND_URL}/history/dialogue/{dialogue_id}",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return []


def delete_history_from_backend(dialogue_id: Optional[str] = None) -> bool:
    """Delete history via backend API"""
    try:
        params = {}
        if dialogue_id:
            params["dialogue_id"] = dialogue_id
        response = requests.delete(
            f"{BACKEND_URL}/history",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"❌ Ошибка при удалении: {e}")
        return False


def get_cached_history(force_refresh: bool = False) -> List[Dict]:
    """Get history with caching to reduce API calls"""
    cache_duration = timedelta(seconds=30)
    
    if (force_refresh or 
        st.session_state.history_cache is None or 
        st.session_state.history_cache_time is None or
        datetime.now() - st.session_state.history_cache_time > cache_duration):
        
        st.session_state.history_cache = get_history_from_backend(limit=500)
        st.session_state.history_cache_time = datetime.now()
    
    return st.session_state.history_cache


# --- Page: Query Interface ---
def page_query():
    st.title("🤖 Вопросно-ответная система RAG")
    
    st.markdown("---")
    
    # Search period settings (placeholder for future functionality)
    st.subheader("⚙️ Настройки поиска")
    
    with st.expander("Период поиска информации", expanded=False):
        st.info("⚠️ Настройка периода поиска в разработке. Пока используется весь доступный период.")
        
        col1, col2 = st.columns(2)
        with col1:
            period_start = st.date_input(
                "Начало периода",
                value=datetime.now() - timedelta(days=30),
                help="Начальная дата для поиска информации (заглушка)"
            )
        with col2:
            period_end = st.date_input(
                "Конец периода",
                value=datetime.now(),
                help="Конечная дата для поиска информации (заглушка)"
            )
    
    st.markdown("---")
    
    # Query input
    st.subheader("💬 Задайте вопрос")
    
    query = st.text_area(
        "Ваш вопрос:",
        height=100,
        placeholder="Например: Какие новости были о курсе доллара на этой неделе?",
        help="Введите ваш вопрос на русском языке"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        submit_button = st.button("🔍 Получить ответ", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Очистить", use_container_width=True)
    with col3:
        new_dialogue_button = st.button("🆕 Новый диалог", use_container_width=True, 
                                       help="Начать новый диалог (для будущей поддержки истории)")
    
    if clear_button:
        st.rerun()
    
    if new_dialogue_button:
        st.session_state.current_dialogue_id = generate_dialogue_id()
        st.session_state.dialogue_messages = []
        st.success("✅ Начат новый диалог")
        st.rerun()
    
    # Process query
    if submit_button:
        if not query.strip():
            st.warning("⚠️ Пожалуйста, введите вопрос")
        else:
            with st.spinner("🔄 Обработка вашего запроса..."):
                result = send_query_to_backend(
                    query=query,
                    dialogue_id=st.session_state.current_dialogue_id
                )
                
                if result:
                    # Invalidate cache
                    st.session_state.history_cache = None
                    
                    # Display result
                    st.markdown("---")
                    st.subheader("✅ Ответ:")
                    
                    # Answer in a nice card
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                        <p style="margin: 0; font-size: 16px;">{result.get('answer', 'Ответ не получен')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Reasoning in expander
                    with st.expander("📝 Обоснование ответа", expanded=False):
                        st.markdown(result.get("reason", "Обоснование отсутствует"))
                    
                    # For future dialogue support
                    st.session_state.dialogue_messages.append({
                        "role": "user",
                        "content": query,
                        "timestamp": datetime.now().isoformat()
                    })
                    st.session_state.dialogue_messages.append({
                        "role": "assistant",
                        "content": result.get("answer", ""),
                        "timestamp": datetime.now().isoformat()
                    })

# --- Page: Query History ---
def page_history():
    st.title("📜 История запросов")
    
    # Check backend availability
    if not check_backend_health():
        st.error("❌ Backend недоступен. Запустите сервер: `python server.py`")
        return
    
    # Refresh button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        refresh_btn = st.button("🔄 Обновить", use_container_width=True)
    with col3:
        clear_btn = st.button("🗑️ Очистить всё", type="secondary", use_container_width=True)
    
    if refresh_btn:
        st.session_state.history_cache = None
    
    # Get statistics from backend
    stats = get_stats_from_backend()
    
    if stats.get("total_queries", 0) == 0:
        st.info("📭 История запросов пуста. Задайте первый вопрос на странице 'Запрос'!")
        return
    
    # Statistics
    st.subheader("📊 Статистика")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего запросов", stats.get("total_queries", 0))
    with col2:
        st.metric("Уникальных диалогов", stats.get("unique_dialogues", 0))
    with col3:
        last_time = stats.get("last_query_time")
        if last_time:
            try:
                dt = datetime.fromisoformat(last_time)
                st.metric("Последний запрос", dt.strftime("%H:%M:%S"))
            except:
                st.metric("Последний запрос", "—")
        else:
            st.metric("Последний запрос", "—")
    with col4:
        first_time = stats.get("first_query_time")
        if first_time:
            try:
                dt = datetime.fromisoformat(first_time)
                st.metric("Первый запрос", dt.strftime("%d.%m.%Y"))
            except:
                st.metric("Первый запрос", "—")
        else:
            st.metric("Первый запрос", "—")
    
    st.markdown("---")
    
    # Clear history with confirmation
    if clear_btn:
        if st.session_state.get("confirm_clear"):
            if delete_history_from_backend():
                st.session_state.confirm_clear = False
                st.session_state.history_cache = None
                st.success("✅ История очищена")
                st.rerun()
        else:
            st.session_state.confirm_clear = True
            st.warning("⚠️ Нажмите еще раз для подтверждения удаления ВСЕЙ истории")
    
    # Filters
    col1, col2 = st.columns([3, 1])
    with col1:
        search_filter = st.text_input("🔍 Поиск по тексту", placeholder="Введите ключевые слова...")
    with col2:
        view_mode = st.selectbox("Режим", ["Все запросы", "По диалогам"], label_visibility="collapsed")
    
    # Get history from backend
    if search_filter:
        filtered_history = search_history_in_backend(search_filter, limit=100)
    else:
        filtered_history = get_cached_history(force_refresh=refresh_btn)
    
    if not filtered_history:
        st.warning("🔍 По вашему запросу ничего не найдено")
        return
    
    # View mode: All queries
    if view_mode == "Все запросы":
        st.subheader(f"Найдено записей: {len(filtered_history)}")
        
        # Display history entries
        for entry in filtered_history:
            timestamp_str = entry.get("timestamp", "")
            
            with st.container():
                # Header with timestamp and dialogue ID
                col1, col2 = st.columns([3, 1])
                with col1:
                    try:
                        dt = datetime.fromisoformat(timestamp_str)
                        st.markdown(f"**🕐 {dt.strftime('%Y-%m-%d %H:%M:%S')}**")
                    except:
                        st.markdown(f"**🕐 {timestamp_str}**")
                with col2:
                    dialogue_id = entry.get('dialogue_id', 'N/A')
                    st.caption(f"ID: {dialogue_id[-12:] if len(dialogue_id) > 12 else dialogue_id}")
                
                # Query
                st.markdown(f"**❓ Вопрос:**")
                st.markdown(f"> {entry.get('query', '')}")
                
                # Answer in expander
                with st.expander("💬 Ответ", expanded=False):
                    st.markdown(entry.get('answer', ''))
                    
                    if entry.get('reason'):
                        st.markdown("---")
                        st.markdown("**📝 Обоснование:**")
                        st.caption(entry.get('reason'))
                    
                    # Search period info
                    search_period = entry.get('search_period', {})
                    if search_period:
                        st.markdown("---")
                        st.caption(f"🔍 Период поиска: {search_period.get('start', 'N/A')} — {search_period.get('end', 'N/A')}")
                
                st.markdown("---")
    
    # View mode: By dialogues
    else:
        recent_dialogues = get_dialogues_from_backend(limit=20)
        st.subheader(f"Диалогов: {len(recent_dialogues)}")
        
        for dialogue in recent_dialogues:
            dialogue_id = dialogue.get('dialogue_id', '')
            message_count = dialogue.get('message_count', 0)
            started_at = dialogue.get('started_at', '')
            
            try:
                dt = datetime.fromisoformat(started_at)
                time_str = dt.strftime('%Y-%m-%d %H:%M')
            except:
                time_str = started_at
            
            with st.expander(
                f"💬 {dialogue_id[-15:] if len(dialogue_id) > 15 else dialogue_id} | {message_count} запросов | {time_str}",
                expanded=False
            ):
                # Get full dialogue history
                dialogue_history = get_dialogue_history_from_backend(dialogue_id)
                
                for entry in dialogue_history:
                    timestamp_str = entry.get("timestamp", "")
                    try:
                        dt = datetime.fromisoformat(timestamp_str)
                        st.markdown(f"**🕐 {dt.strftime('%H:%M:%S')}**")
                    except:
                        st.markdown(f"**🕐 {timestamp_str}**")
                    st.markdown(f"**❓ Вопрос:** {entry.get('query', '')}")
                    answer = entry.get('answer', '')
                    if len(answer) > 200:
                        st.markdown(f"**💬 Ответ:** {answer[:200]}...")
                    else:
                        st.markdown(f"**💬 Ответ:** {answer}")
                    st.markdown("---")


# --- Main App ---
def main():
    st.set_page_config(
        page_title="RAG Q&A System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Навигация")
    
    if st.sidebar.button(
        "🔍 Запрос", 
        use_container_width=True, 
        type="primary" if st.session_state.current_page == "🔍 Запрос" else "secondary"
    ):
        st.session_state.current_page = "🔍 Запрос"
        st.rerun()
        
    if st.sidebar.button(
        "📜 История", 
        use_container_width=True, 
        type="primary" if st.session_state.current_page == "📜 История" else "secondary"
    ):
        st.session_state.current_page = "📜 История"
        st.rerun()
    
    # Route to pages
    if st.session_state.current_page == "🔍 Запрос":
        page_query()
    else:
        page_history()


if __name__ == "__main__":
    main()
