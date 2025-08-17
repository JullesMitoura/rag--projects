import streamlit as st
from typing import Optional, Callable, Dict, Any, List


class ChatTemplate:
    def __init__(
        self,
        session_key: str = "messages",
        placeholder: str = "Digite sua mensagem...",
        assistant_name: str = "assistant",
        user_name: str = "user",
        process_message: Optional[Callable[[List[Dict[str, str]], str], str]] = None,
        chat_height: int = 400
    ):
        self.session_key = session_key
        self.placeholder = placeholder
        self.assistant_name = assistant_name
        self.user_name = user_name
        self.process_message = process_message or self._default_echo
        self.chat_height = chat_height
        
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = []
    
    def _default_echo(self, history: List[Dict[str, str]], message: str) -> str:
        return f"Echo: {message}"
    
    def set_process_message(self, process_function: Callable[[List[Dict[str, str]], str], str]) -> None:
        self.process_message = process_function
    
    def add_message(self, role: str, content: str) -> None:
        st.session_state[self.session_key].append({
            "role": role,
            "content": content
        })
    
    def render_message(self, message: Dict[str, Any]) -> None:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    def render_history(self, 
                       history: Optional[List[Dict[str, str]]] = None) -> None:
        if history is None:
            history = st.session_state[self.session_key]
        for message in history:
            self.render_message(message)
    
    def chat(
        self, 
        key: Optional[str] = None, 
        height: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Optional[str]:
        
        container_height = height or self.chat_height
        chat_container = st.container(height=container_height)
        with chat_container:
            self.render_history(history)
        
        if prompt := st.chat_input(self.placeholder, key=key):
            self.add_message(self.user_name, prompt)
            current_history = st.session_state[self.session_key].copy()
            
            with st.spinner("Thinking..."):
                try:
                    response = self.process_message(current_history, prompt)
                except Exception as e:
                    response = f"Error: {str(e)}"
            
            self.add_message(self.assistant_name, response)
            st.rerun()
            
            return response
        
        return None