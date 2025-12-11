import time 
from typing import Union, Generator, List, Dict, Any

from pyrogram import Client
from pyrogram.types import Message


class PyroSource:

    def __init__(
            self, 
            api_id: Union[int, str], 
            api_hash: str, 
            app_name: str = "default_app",
        ):
        self.client = Client(name=app_name, api_id=api_id, api_hash=api_hash)

    
    def load_messages(
        self, 
        channel_id: Union[int, str], 
        limit: int, 
        offset: int = 0, 
        offset_id: int = 0,
        time_sleep: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        channel_id: channel id or username
        limit: number of messages to load
        offset: offset index
        offset_id: message id offset
        """
        posts = []

        with self.client as app:
            messages: Generator[Message] = app.get_chat_history(
                chat_id=channel_id, 
                limit=limit, 
                offset=offset, 
                offset_id=offset_id,
            )

            for msg in messages:
                time.sleep(time_sleep)

                content = msg.text or msg.caption or ''
                original_author = (
                    msg.forward_from_chat.username if msg.forward_from_chat else ''
                )
                message_dt = msg.date.strftime("%Y-%m-%d")


                meta = {
                    "message_dt" : message_dt,
                    "message_id" : msg.id,
                    "channel_id" : channel_id,
                    "content" : content,
                    "views" : msg.views,
                    "original_author" : original_author,
                }

                posts.append(meta)
        
        return posts
