import pandas as pd


def strip_edges_allow_punct(s: str):
    allowed_punct = set(".,!?;:-–—")  # можно расширять

    # Левый указатель — пока не буква/цифра
    left = 0
    while left < len(s) and not s[left].isalnum():
        left += 1

    # Правый указатель — пока не буква/цифра/пунктуация
    right = len(s) - 1
    while right >= 0 and not (s[right].isalnum() or s[right] in allowed_punct):
        right -= 1

    # Если всё мусор
    if right < left:
        return ""

    return s[left:right+1]


def process_str(s: str):
    # Чистка статьи от мусора
    s = "\n".join(strip_edges_allow_punct(p) for p in s.split("\n") if p)
    
    for suf in [
        "Слушать прямой эфир",
        "Читать РБК Стиль в Telegram",
        "РБК Events, 18",
        "Подписаться | Онлайн-сомелье",
        "Читать РБК в Telegram",
        "Следить за новостями РБК в Telegram",
        "Следить за новостями РБК в МАХ",
        "Другие видео этого дня — в телеграм-канале РБК",
        "РБК в Telegram и MAX",
        "РБК в Telegram | MAX",
        "Подписаться на «РБК Спорт",
        "Картина дня — в телеграм-канале РБК",
        "Самые важные новости — в канале РБК в МАХ",
        "Больше инфографики — в телеграм-канале РБК",
        "Подписаться на «Сам ты инвестор!",
        "Читать РБК Недвижимость в Telegram"
    ]:
        s = s.removesuffix(suf).strip()

    parts = [p for p in s.split("\n") if p]
    
    prev_parts = [0] * 1000
    while len(prev_parts) != len(parts) and len(parts) != 0:
        prev_parts = parts
        if "Фото:" in parts[-1] or "Данные:" in parts[-1]:
            parts = parts[:-1]
    
    return "\n".join(parts)


def is_advertisement(s: str):
    # Проверка рекламных объявлений
    last_part = [p for p in s.split("\n") if p][-1]
    return any(v in last_part for v in ["Реклама.", "Реклама,"])


def clean_df(df: pd.DataFrame):
    df["message_dt"] = pd.to_datetime(df["message_dt"]).dt.date
    df["content"] = df["content"].apply(lambda x: process_str(x))
    df["views"] = df["views"].astype(int)
    df = df[~df["content"].apply(is_advertisement)]

    return df[["message_id", "channel_id", "message_dt", "views", "content"]]
