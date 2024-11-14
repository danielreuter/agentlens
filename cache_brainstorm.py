# Desiderata
# 1. Effects should be able to write and read from a cache
# 2. This cache might just be a database though...
# 3. We need to be able to invalidate the cache


class Lens:
    def task(self, *args, **kwargs): ...

    def mock(self, *args, **kwargs): ...


class Database:
    def get(self, url: str): ...

    def set(self, url: str, html: str): ...


class MockMiss: ...


ls = Lens()
db = Database()


def get_html(url: str): ...


@ls.mock()
def mock_scrape_website(url: str):
    try:
        return db.get(url)
    except KeyError:
        return MockMiss()


@ls.task(mock=mock_scrape_website)
def scrape_website(url: str):
    html = get_html(url)
    db.set(url, html)
    return html
