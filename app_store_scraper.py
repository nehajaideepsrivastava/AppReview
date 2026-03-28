"""
Vendored replacement for the PyPI package `app-store-scraper`.

The PyPI version hard-pins requests==2.23.0, which is incompatible with
every version of Streamlit >= 1.28.0 (which requires requests>=2.27).
This file provides an identical public API using whatever requests version
is already installed, so no PyPI package is needed.

Original project: https://github.com/cowboy-bebug/app-store-scraper (MIT)
"""

import json
import logging
import re
from datetime import datetime, timezone
from time import sleep as _sleep
from urllib.parse import unquote

import requests as _requests

logger = logging.getLogger(__name__)


class AppStore:
    """Scrape customer reviews from the Apple App Store."""

    _LANDING = "https://apps.apple.com/{country}/app/{app_name}/id{app_id}"
    _API = "https://amp-api.apps.apple.com/v1/catalog/{country}/apps/{app_id}/reviews"
    _HEADERS = {
        "Accept": "application/json",
        "Accept-Language": "en-GB,en;q=0.9",
        "Connection": "keep-alive",
        "Origin": "https://apps.apple.com",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    def __init__(
        self,
        country: str,
        app_name: str,
        app_id=None,
        log_format: str = "%(asctime)s [%(levelname)s] %(message)s",
        log_level: int = logging.WARNING,
        log_interval: int = 5,
    ):
        self.country = country.lower().strip()
        self.app_name = re.sub(r"[\W_]+", "-", app_name.lower())
        self.app_id = str(app_id) if app_id else None
        self.reviews: list = []
        self.reviews_count: int = 0

        self._log_interval = log_interval
        self._token: str | None = None
        self._session = _requests.Session()
        self._session.headers.update(self._HEADERS)

        logging.basicConfig(format=log_format, level=log_level)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _landing_url(self) -> str:
        return self._LANDING.format(
            country=self.country,
            app_name=self.app_name,
            app_id=self.app_id,
        )

    def _api_url(self) -> str:
        return self._API.format(country=self.country, app_id=self.app_id)

    def _fetch_token(self) -> None:
        """Extract the bearer token embedded in the App Store landing page."""
        try:
            resp = self._session.get(self._landing_url(), timeout=15)
            resp.raise_for_status()
            html = resp.text

            # Pattern 1 – URL-encoded JSON config in <meta> tag
            m = re.search(
                r'<meta name="web-experience-app/config/environment" content="([^"]+)"',
                html,
            )
            if m:
                config = json.loads(unquote(m.group(1)))
                token = (
                    config.get("MEDIA_API", {}).get("token")
                    or config.get("mediaApiToken")
                )
                if token:
                    self._token = token
                    return

            # Pattern 2 – inline token string
            for pattern in (
                r'token%22%3A%22([^%]+)%22',
                r'"token"\s*:\s*"([^"]{20,})"',
            ):
                m = re.search(pattern, html)
                if m:
                    self._token = m.group(1)
                    return

        except Exception as exc:
            logger.warning("AppStore: could not fetch bearer token – %s", exc)

    def _get_page(self, offset: int | None = None) -> dict:
        """Fetch a single page of reviews from the Apple API."""
        if not self._token:
            self._fetch_token()
        if not self._token:
            logger.error("AppStore: no bearer token; cannot fetch reviews.")
            return {}

        params: dict = {
            "l": "en-GB",
            "limit": "20",
            "platform": "web",
            "additionalPlatforms": "appletv,ipad,iphone,mac",
        }
        if offset is not None:
            params["offset"] = str(offset)

        headers = dict(self._HEADERS)
        headers["Authorization"] = f"Bearer {self._token}"
        headers["Referer"] = self._landing_url()

        try:
            resp = self._session.get(
                self._api_url(), headers=headers, params=params, timeout=15
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("AppStore: review page fetch failed – %s", exc)
            return {}

    @staticmethod
    def _parse_date(raw: str) -> datetime | None:
        if not raw:
            return None
        try:
            # "2023-10-05T14:48:00Z" or "2023-10-05T14:48:00+00:00"
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API  (mirrors the PyPI package)
    # ------------------------------------------------------------------

    def review(
        self,
        how_many: int | None = None,
        after: datetime | None = None,
        sleep: int | None = None,
    ) -> None:
        """
        Populate ``self.reviews`` with App Store reviews.

        Parameters
        ----------
        how_many : int, optional
            Maximum number of reviews to fetch (default: all available).
        after : datetime, optional
            Only include reviews posted after this date.
        sleep : int, optional
            Seconds to sleep between page requests.
        """
        self.reviews = []
        offset: int | None = None

        while True:
            data = self._get_page(offset=offset)
            entries = data.get("data", [])
            if not entries:
                break

            for entry in entries:
                attrs = entry.get("attributes", {})
                date = self._parse_date(attrs.get("date", ""))

                if after and date and date.replace(tzinfo=timezone.utc) <= after.replace(
                    tzinfo=timezone.utc
                ):
                    # Older than the cutoff – stop paging
                    self.reviews_count = len(self.reviews)
                    return

                self.reviews.append(
                    {
                        "date": date,
                        "isEdited": attrs.get("isEdited", False),
                        "rating": attrs.get("rating", 0),
                        "review": attrs.get("review", ""),
                        "title": attrs.get("title", ""),
                        "userName": attrs.get("userName", ""),
                        "developerResponse": attrs.get("developerResponse") or {},
                    }
                )

                if how_many and len(self.reviews) >= how_many:
                    self.reviews_count = len(self.reviews)
                    return

            if sleep:
                _sleep(sleep)

            # Follow the "next" cursor if present
            next_url = data.get("next", "")
            if not next_url:
                break
            m = re.search(r"offset=(\d+)", next_url)
            if m:
                offset = int(m.group(1))
            else:
                break

        self.reviews_count = len(self.reviews)
