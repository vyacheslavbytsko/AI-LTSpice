from langchain_core.rate_limiters import InMemoryRateLimiter


def get_groq_key() -> str:
    return open("groq_key.txt", "r").read().strip()

def get_tg_token() -> str:
    return open("tg_token.txt", "r").read().strip()

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.03,  # Можно делать запрос только раз в 30 секунд
    check_every_n_seconds=10,  # Проверять, доступны ли токены каждые 10 с
    max_bucket_size=1,  # Контролировать максимальный размер всплеска запросов
)