import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

def show_topic_modeling(filtered_df):

    if filtered_df.empty:

        show_timed_warning_generic("⚠️ No records found within the specified date range", duration=4)

        return

 

    with st.spinner("🔍 Performing Topic Modeling... Please wait"):

        # --- Separate reviews by rating ---

        positive_reviews = filtered_df[filtered_df['rating'] >= 4]['review']

        negative_reviews = filtered_df[filtered_df['rating'] <= 2]['review']

        neutral_reviews = filtered_df[filtered_df['rating'] == 3]['review']

 

        def extract_topics(reviews, n_topics=5, n_keywords=5):

            vectorizer = CountVectorizer(stop_words='english', max_df=1.0, min_df=1)

            doc_term_matrix = vectorizer.fit_transform(reviews)

            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)

            lda.fit(doc_term_matrix)

            topics = []

            for idx, topic in enumerate(lda.components_):

                keywords = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-n_keywords:]]

                topics.append((f"Topic {idx+1}", keywords))

            return topics, lda.transform(doc_term_matrix)

 

        def get_representative_sentences(reviews, topic_distributions, n_sentences=1):

            sentences = []

            for topic_idx in range(topic_distributions.shape[1]):

                topic_scores = topic_distributions[:, topic_idx]

                top_indices = topic_scores.argsort()[-n_sentences:]

                topic_sentences = reviews.iloc[top_indices].tolist()

                sentences.append(topic_sentences)

            return sentences

 

        def display_topic_summary(topics, sentences, section_title):

            st.markdown(f"### {section_title}")

            for i, (topic_name, keywords) in enumerate(topics):

                keywords_str = ', '.join(keywords)

                with st.expander(keywords_str):

                    st.markdown("**Example Sentences:**")

                    for sent in sentences[i]:

                        st.markdown(f"- {sent}")

 

        # --- Run topic modeling ---

        positive_topics, positive_sentences = [], []

        negative_topics, negative_sentences = [], []

        neutral_topics, neutral_sentences = [], []

        st.markdown("<br>", unsafe_allow_html=True)

        if len(positive_reviews) > 0:

            positive_topics, positive_topic_distributions = extract_topics(positive_reviews)

            positive_sentences = get_representative_sentences(positive_reviews, positive_topic_distributions)

            display_topic_summary(positive_topics, positive_sentences, "Top 5 Best Aspects (Rating ≥ 4)")

            st.divider()

 

        if len(negative_reviews) > 0:

            negative_topics, negative_topic_distributions = extract_topics(negative_reviews)

            negative_sentences = get_representative_sentences(negative_reviews, negative_topic_distributions)

            display_topic_summary(negative_topics, negative_sentences, "Top 5 Issues (Rating ≤ 2)")

            st.divider()

 

        if len(neutral_reviews) > 0:

            neutral_topics, neutral_topic_distributions = extract_topics(neutral_reviews)

            neutral_sentences = get_representative_sentences(neutral_reviews, neutral_topic_distributions)

            display_topic_summary(neutral_topics, neutral_sentences, "Top 5 Neutral Topics (Rating = 3)")

            st.divider()

 

        # --- Generate PDF summary --- 

        pdf = FPDF()

        pdf.add_page()

        pdf.set_font("Arial", size=12)

        pdf.set_auto_page_break(auto=True, margin=15)

 

        def add_topic_section_to_pdf(pdf, section_title, topics, sentences):

            pdf.set_font("Arial", 'B', 14)

            pdf.cell(200, 10, txt=section_title, ln=True, align='C')  # Center section title

            pdf.set_font("Arial", size=12)

            for i, (topic_name, keywords) in enumerate(topics):

                pdf.set_font("Arial", 'B', 12)

                pdf.cell(200, 10, txt=f"{topic_name}", ln=True)

                pdf.set_font("Arial", size=12)

                pdf.multi_cell(0, 10, txt=f"Keywords: {', '.join(keywords)}")

                pdf.multi_cell(0, 10, txt="Example Sentences:")

                for sent in sentences[i]:

                    clean_sent = ''.join(char if ord(char) < 256 else ' ' for char in sent)

                    pdf.multi_cell(0, 10, txt=f"- {clean_sent}")

                pdf.ln(5)

 

        if positive_topics:

            add_topic_section_to_pdf(pdf, "Top 5 Best Aspects", positive_topics, positive_sentences)

        if negative_topics:

            add_topic_section_to_pdf(pdf, "Top 5 Issues", negative_topics, negative_sentences)

        if neutral_topics:

            add_topic_section_to_pdf(pdf, "Top 5 Neutral Aspects", neutral_topics, neutral_sentences)

 

        pdf.output("topic_modeling_summary.pdf")

 

        # --- Show download button after spinner completes ---

        if positive_topics or negative_topics or neutral_topics:

            st.subheader("Topic Modeling")

            with open("topic_modeling_summary.pdf", "rb") as f:

                st.download_button(

                    label="📄 Download Topic Modeling Summary as PDF",

                    data=f,

                    file_name="topic_modeling_summary.pdf",

                    mime="application/pdf"

                )

        else:

            show_timed_warning_generic("⚠️ No records found within the specified date range", duration=4)

# import urllib3

# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import json

with open("country_config.json", "r") as f:

    country_map = json.load(f)

from streamlit_option_menu import option_menu   

import streamlit as st

import plotly.graph_objects as go

import pandas as pd

import base64

import random

import io

import pandas as pd

import qrcode

from concurrent.futures import ThreadPoolExecutor

from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

from fpdf import FPDF

import nltk

from nltk.util import ngrams

from collections import Counter

import plotly.express as px

import random

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from streamlit.column_config import TextColumn

import requests

import pandas as pd

from concurrent.futures import ThreadPoolExecutor

import feedparser

import plotly.express as px

import pandas as pd

import numpy as np

from datetime import date, timedelta

import time

from fpdf import FPDF

from sklearn.cluster import KMeans

import warnings

import requests

import datetime

from languages import *

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import nltk

import plotly.express as px

from google_play_scraper import reviews, Sort

import os

from google_play_scraper import Sort, reviews_all

from app_store_scraper import AppStore

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

import pycountry

from wordcloud import WordCloud, STOPWORDS

from langdetect import detect

from nltk.util import ngrams

from PIL import Image

from collections import Counter

from googletrans import Translator

from languages import *

warnings.filterwarnings('ignore')

nltk.download('punkt')

nltk.download('words')

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from streamlit_plotly_events import plotly_events

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from langdetect import detect

import pandas as pd

import warnings

import matplotlib.pyplot as plt

# from transformers import pipeline

from nltk.corpus import stopwords

from streamlit_autorefresh import st_autorefresh

 

 

# st.set_page_config(page_title="Customer Sentiment Analyzer!!!", page_icon=":sparkles:",layout="wide")

 

st.set_page_config(

    page_title="Customer Sentiment Analyzer!!!",

    page_icon="Images/WUNEW.png",  # File must be in the root directory

    layout="wide"

)

# st.title(" :sparkles: Sentiment Anaylzer")

st.markdown('<style>div.block-container{padding-top:0rem;text-align: center}</style>',unsafe_allow_html=True)

 

 

def local_css(file_name):

    with open(file_name) as f:

        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

 

#load the style sheet

local_css("custom_style.css")

 

# st.sidebar.image("images/wufull.png", use_column_width=True)

# Load and encode image (handle missing file gracefully)

dir = os.path.dirname(__file__)

# Try several possible image locations (case / folder differences)
possible_paths = [
    os.path.join(dir, 'Images', 'wufull.png'),
    os.path.join(dir, 'image', 'wufull.png'),
    os.path.join(dir, 'image', 'apotheke.png'),
    os.path.join(dir, 'Images', 'apotheke.png'),
]
encoded_image = None
for filename in possible_paths:
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()
            break
        except Exception:
            encoded_image = None
            break

 

 

 

st.markdown("""

<style>

/* Consistent metric card sizing and alignment */

.metric-card {

    padding: 1rem;

    border-radius: 10px;

    margin: 0.5rem;

    height: 160px;               /* ✅ fixed equal height */

    display: flex;

    flex-direction: column;

    justify-content: center;     /* center content vertically */

    align-items: center;         /* center horizontally */

    text-align: center;

    box-sizing: border-box;

}

 

/* Clamp long heading to prevent overflow */

.metric-card h2 {

    margin: 6px 0 2px 0;

    font-size: 1.6rem;

    line-height: 1.2;

    max-width: 100%;

    overflow: hidden;

    text-overflow: ellipsis;

    white-space: nowrap;      

}

 

.metric-card h3 {

    margin: 0;

    font-size: 1.0rem;

    line-height: 1.2;

}

 

.metric-card p {

    margin: 2px 0 0 0;

    font-size: 0.95rem;

}

 

/* Optional: ensure equal column height behavior in Streamlit columns */

.block-container .row-widget stHorizontalBlock > div {

    display: flex;

}

</style>

""", unsafe_allow_html=True)

 

 

 

# Render sidebar image safely: prefer base64 data URI if available, else fall back to file path or placeholder
try:
    if encoded_image:
        img_html = f'<img src="data:image/png;base64,{encoded_image}" style="width: 100%;margin-top: 10px;margin-bottom: 20px;"/>'
        st.sidebar.markdown(f"""
            <style>
                .no-fullscreen-sidebar img {{ pointer-events: none; user-select: none; }}
                [title="View fullscreen"] {{ display: none !important; }}
            </style>
            <div class="no-fullscreen-sidebar" style="text-align: center;">{img_html}</div>
        """, unsafe_allow_html=True)
    else:
        # Attempt to find a usable file path and use Streamlit's image widget which handles missing files gracefully
        found_file = None
        for p in possible_paths:
            if os.path.exists(p):
                found_file = p
                break
        if found_file:
            st.sidebar.image(found_file, use_column_width=True)
        else:
            st.sidebar.markdown('<div style="text-align:center;padding:10px;">No image available</div>', unsafe_allow_html=True)
except Exception:
    # Last-resort fallback
    st.sidebar.markdown('<div style="text-align:center;padding:10px;">No image available</div>', unsafe_allow_html=True)

 

 

st.markdown("""

<style>

/* Sidebar header compact */

section[data-testid="stSidebar"] h1,

section[data-testid="stSidebar"] h2,

section[data-testid="stSidebar"] h3 {

    text-align: left !important;

    padding-left: 4px !important;

    margin-top: 0px !important;

    margin-bottom: 4px !important;

}

 

/* Remove extra spacing between buttons */

section[data-testid="stSidebar"] div.stButton {

    margin: 0px !important;       /* Remove margin around button container */

    padding: 0px !important;      /* Remove padding inside container */

    line-height: 1 !important;    /* Compact line height */

}

 

/* Make buttons inline-block to reduce gaps */

section[data-testid="stSidebar"] div.stButton > button {

    display: block !important;    /* Ensure full width */

    background-color: transparent !important;

    color: #0066cc !important;

    border: none !important;

    font-size: 2px !important;

    text-align: left !important;

    justify-content: flex-start !important;

    padding: .5px 4px !important;  /* Minimal padding */

    width: 100% !important;

    margin: 0px !important;       /* Remove extra margin */

}

 

/* Hover and active states */

section[data-testid="stSidebar"] div.stButton > button:hover {

    color: #ff6600 !important;

    text-decoration: underline !important;

}

section[data-testid="stSidebar"] div.stButton.active > button {

    font-weight: bold !important;

    color: #ff6600 !important;

}

</style>

""", unsafe_allow_html=True)

 

 

 

st.markdown("""

    <style>

    .nav-link:last-child {

        background: linear-gradient(90deg, #ffdd00, #ffa500) !important;

        color: #000 !important;

        font-weight: bold !important;

    }

    </style>

""", unsafe_allow_html=True)

 

 

 

st.markdown("""

    <style>

    div[data-baseweb="select"] > div {

        text-align: left !important;

    }

    </style>

    """, unsafe_allow_html=True)

 

 

st.markdown('''<div style='text-align: center; padding-top: 25px;'><h1 style='color: black; font-weight: bold; font-family: "PP Right Grotesk", sans-serif; font-size: 35px;'>Customer Sentiment Analyzer</h1></div>''', unsafe_allow_html=True)

 

 

def show_timed_warning_generic(message, duration=3):

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:

        warning_placeholder = st.empty()

        warning_placeholder.markdown(

            f"""

            <div class="timed-warning-box">

                <strong>{message}</strong>

            </div>

            """,

            unsafe_allow_html=True

        )

        time.sleep(duration)

        warning_placeholder.empty()

 

app_url = "https://.com/"

 

# Generate the QR code

qr = qrcode.QRCode(

    version=1,

    box_size=10,

    border=5

)

qr.add_data(app_url)

qr.make(fit=True)

 

# Create an image

img = qr.make_image(fill="black", back_color="white")

 

# Save to file

img.save("app_qr_code.png")

 

#count = st_autorefresh(interval=3600000, key="fizzbuzzcounter")

sia = SentimentIntensityAnalyzer()

stop_words = set(stopwords.words('english'))

# st.cache_data.clear()

# print(st.__version__)

translator = Translator()

sid = SentimentIntensityAnalyzer()

 

@st.cache_data

def iso2_to_name(code):

    try:

        return pycountry.countries.get(alpha_2=code).name

    except:

        return None

 

@st.cache_data

def name_to_iso3(name):

    try:

        return pycountry.countries.get(name=name).alpha_3

    except:

        return None

 

# --- Android Review Fetch ---

@st.cache_data(ttl=86400, show_spinner=False)

def load_android_data(app_id, country, app_name):

    reviews = reviews_all(

        app_id,

        sleep_milliseconds=0,

        lang='en',

        country=country,

        sort=Sort.NEWEST,

    )

    df = pd.DataFrame(np.array(reviews), columns=['review'])

    df = df.join(pd.DataFrame(df.pop('review').tolist()))

    columns_to_drop = ['reviewId', 'thumbsUpCount', 'reviewCreatedVersion', 'repliedAt', 'userImage']

    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], errors="ignore")

    df['AppName'] = app_name

    df['Country'] = country.lower()

 

    df.rename(columns={

        'content': 'review',

        'userName': 'UserName',

        'score': 'rating',

        'at': 'TimeStamp',

        'replyContent': 'WU_Response'

    }, inplace=True)

    return df

 

 

def show_progress():

    progress = st.progress(0)

    status = st.empty()

    for i in range(100):

        time.sleep(0.005)

        progress.progress(i + 1)

        status.text(f"Loading Data... {i + 1}%")

    progress.empty()

    status.empty()

 

def fetch_all_android(app_details):

    frames = []

    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = [executor.submit(load_android_data, app_id, country, app_name)

                   for app_id, country, app_name in app_details]

        for future in futures:

            try:

                result = future.result()

                frames.append(result)

            except Exception as e:

                print(f"Android fetch failed: {e}")

                frames.append(pd.DataFrame())

    if frames:

        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame()

 

 

 

def fetch_ios_reviews(app_id, country_code):

    """

    Fetch iOS app reviews using Apple's official RSS feed.

    """

    url = f"https://itunes.apple.com/{country_code}/rss/customerreviews/id={app_id}/sortBy=mostRecent/json"

    reviews = []

 

    try:

        resp = requests.get(url, timeout=10)

        resp.raise_for_status()

        feed = resp.json().get('feed', {})

        entries = feed.get('entry', [])

 

        if len(entries) <= 1:

            print(f"No reviews found for App ID: {app_id} in {country_code}")

            return pd.DataFrame()

 

        for item in entries[1:]:

            # Extract version from im:version.label

            version = item.get('im:version', {}).get('label')

            reviews.append({

                "rating": item.get('im:rating', {}).get('label'),

                "date": item.get('updated', {}).get('label'),

                "review": item.get('content', {}).get('label'),

                "UserName": item.get('author', {}).get('name', {}).get('label'),

                "AppName": "iOS",

                "Platform": "iOS",

                "Country": country_code,

                "AppID": app_id,

                "appVersion": version

            })

 

    except Exception as e:

        print(f"Error fetching reviews for {app_id}-{country_code}: {e}")

 

    return pd.DataFrame(reviews)

 

 

 

def fetch_all_ios(app_country_list):

    frames = []

    with ThreadPoolExecutor(max_workers=5) as executor:

        futures = [executor.submit(fetch_ios_reviews, app_id, cc)

                   for app_id, cc in app_country_list]

        for future in futures:

            df = future.result()

            if not df.empty:

                df["date"] = pd.to_datetime(df["date"], errors="coerce")

                df["TimeStamp"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

                frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --- TrustPilot Review Fetch ---

@st.cache_data(ttl=86400, show_spinner=False)

def fetch_trustpilot_reviews(business_unit="westernunion.com", max_pages=10):

    """Scrape English TrustPilot reviews for the given business unit."""

    import re as _re

    import json as _json

    all_reviews = []

    headers = {

        "User-Agent": (

            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "

            "AppleWebKit/537.36 (KHTML, like Gecko) "

            "Chrome/124.0.0.0 Safari/537.36"

        ),

        "Accept-Language": "en-US,en;q=0.9",

    }

    for page in range(1, max_pages + 1):

        try:

            url = (

                f"https://www.trustpilot.com/review/{business_unit}"

                f"?page={page}&languages=en"

            )

            resp = requests.get(url, headers=headers, timeout=15)

            if resp.status_code != 200:

                break

            # TrustPilot embeds all data as __NEXT_DATA__ JSON in the page

            match = _re.search(

                r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',

                resp.text, _re.DOTALL

            )

            if not match:

                break

            data = _json.loads(match.group(1))

            page_props = data.get("props", {}).get("pageProps", {})

            reviews_data = page_props.get("reviews", [])

            if not reviews_data:

                break

            for r in reviews_data:

                title = r.get("title", "") or ""

                body = r.get("text", "") or ""

                text = (title + " " + body).strip() if title else body

                rating = r.get("rating", 3)

                date_str = (r.get("dates") or {}).get("publishedDate", "")

                consumer = r.get("consumer") or {}

                author = consumer.get("displayName", "Anonymous")

                country_code = (consumer.get("countryCode") or "us").lower()

                all_reviews.append({

                    "review": text,

                    "rating": int(rating) if rating else 3,

                    "TimeStamp": pd.to_datetime(date_str, errors="coerce"),

                    "UserName": author,

                    "AppName": "TrustPilot",

                    "Country": country_code,

                    "appVersion": "N/A",

                    "WU_Response": None,

                })

        except Exception as e:

            print(f"TrustPilot page {page} fetch error: {e}")

            break

    if not all_reviews:

        return pd.DataFrame()

    df = pd.DataFrame(all_reviews)

    # Normalize to tz-naive UTC so it can concat with Android/iOS timestamps

    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce", utc=True).dt.tz_localize(None)

    df = df.dropna(subset=["TimeStamp"])

    return df


# --- Twitter / X Mention Fetch (Twitter API v2 — requires Bearer Token) ---

def _vader_to_rating(text):

    """Derive a 1-5 proxy rating from VADER compound sentiment score."""

    score = sia.polarity_scores(str(text)).get("compound", 0)

    if score >= 0.5:

        return 5

    elif score >= 0.05:

        return 4

    elif score > -0.05:

        return 3

    elif score >= -0.5:

        return 2

    else:

        return 1


@st.cache_data(ttl=3600, show_spinner=False)

def fetch_twitter_reviews(bearer_token, search_query="westernunion OR @WesternUnion lang:en -is:retweet", max_results=100):

    """

    Fetch recent tweets mentioning Western Union via Twitter API v2.

    Requires a valid Bearer Token (free or paid tier).

    Rating is derived from VADER sentiment since tweets have no star rating.

    """

    if not bearer_token or not bearer_token.strip():

        return pd.DataFrame()

    try:

        endpoint = "https://api.twitter.com/2/tweets/search/recent"

        headers = {"Authorization": f"Bearer {bearer_token.strip()}"}

        params = {

            "query": search_query,

            "max_results": min(max_results, 100),  # API max per page is 100

            "tweet.fields": "created_at,text,author_id",

            "expansions": "author_id",

            "user.fields": "username",

        }

        tweets = []

        pages_fetched = 0

        max_pages = max(1, max_results // 100)

        next_token = None

        while pages_fetched < max_pages:

            if next_token:

                params["next_token"] = next_token

            resp = requests.get(endpoint, headers=headers, params=params, timeout=15)

            if resp.status_code == 401:

                print("Twitter API: invalid or expired Bearer Token.")

                break

            if resp.status_code != 200:

                print(f"Twitter API error {resp.status_code}: {resp.text[:200]}")

                break

            data = resp.json()

            tweet_list = data.get("data") or []

            users = {

                u["id"]: u.get("username", "Unknown")

                for u in (data.get("includes") or {}).get("users", [])

            }

            for t in tweet_list:

                text = t.get("text", "")

                ts = pd.to_datetime(t.get("created_at"), utc=True).tz_localize(None)

                tweets.append({

                    "review": text,

                    "rating": _vader_to_rating(text),

                    "TimeStamp": ts,

                    "UserName": users.get(t.get("author_id", ""), "Unknown"),

                    "AppName": "Twitter",

                    "Country": "us",

                    "appVersion": "N/A",

                    "WU_Response": None,

                })

            next_token = (data.get("meta") or {}).get("next_token")

            pages_fetched += 1

            if not next_token:

                break

        return pd.DataFrame(tweets) if tweets else pd.DataFrame()

    except Exception as e:

        print(f"Twitter API fetch failed: {e}")

        return pd.DataFrame()

 

 

app_details = [

    ('com.westernunion.android.mtapp', 'us', 'Android'),

    ('com.westernunion.moneytransferr3app.eu','fr','Android'), 

    ('com.westernunion.moneytransferr3app.au', 'au', 'Android'),

    ('com.westernunion.moneytransferr3app.ca', 'ca', 'Android'), 

    ('com.westernunion.moneytransferr3app.nz', 'nz', 'Android'),

    ('com.westernunion.moneytransferr3app.nl','nl','Android'),

    ('com.westernunion.moneytransferr3app.acs3','br','Android'),

    ('com.westernunion.moneytransferr3app.eu2','be','Android'),

    ('com.westernunion.moneytransferr3app.eu3','no','Android'),

    ('com.westernunion.moneytransferr3app.eu2','ch','Android'),

    ('com.westernunion.moneytransferr3app.sg','sg','Android'),

    ('com.westernunion.moneytransferr3app.pt','pt','Android'),

    ('com.westernunion.moneytransferr3app.eu4','pl','Android'),

    ('com.westernunion.moneytransferr3app.apac','my','Android'),

    ('com.westernunion.moneytransferr3app.hk','hk','Android'),

    ('com.westernunion.moneytransferr3app.bh', 'bh', 'Android'),   

    ('com.westernunion.moneytransferr3app.kw', 'kw', 'Android'),

    ('com.westernunion.moneytransferr3app.qa', 'qa', 'Android'),

    ('com.westernunion.moneytransferr3app.sa', 'sa', 'Android'),

    ('com.westernunion.moneytransferr3app.in', 'in', 'Android'),

    ('com.westernunion.moneytransferr3app.th', 'th', 'Android'),

    ('com.westernunion.moneytransferr3app.jp', 'jp', 'Android'),

    ('com.westernunion.moneytransferr3app.es', 'es', 'Android'),     

    ('com.westernunion.moneytransferr3app.acs1', 'pe', 'Android'),

    ('com.westernunion.moneytransferr3app.ph', 'ph', 'Android'),

    ('com.westernunion.moneytransferr3app.ph', 'ae', 'Android'), 

    ('com.westernunion.moneytransferr3app.jo', 'jo', 'Android'),             

]

 

app_country_list = [

    ("424716908", "us"),

    ("1045347175","fr"),

    ("1122288720", "au"),

    ("1110191056","ca"),

    ("1268771757","es"),

    ("1226778839","sg"),

    ("1199782520","nl"),

    ("1148514737","br"),

    ("1110240507","be"),

    ("1152860407","no"),

    ("1229307854","pt"),

    ("1168530510","pl"),

    ("1152860407","fi"),

    ("1165109779","hk"),

    ("1164813148","my"),

    ("1314010624","bh"),

    ("1304223498","cl"),

    ("1173794098","kw"),

    ("1483742169","mv"),

    ("1459024696","sa"),

    ("1459226729","th"),

    ("1173792939","qa"),

    ("1150872438","in"),

    ("1199782520","jp"),

    ("1148512210","pe"),

    ("6751528043","ph"),

    ("1171330611","ae"),

    ("1459023219","jo")

]

 

 

# --- MAIN STREAMLIT BLOCK ---

# st.title("🌍 Western Union Reviews Dashboard")

 

@st.cache_data(ttl=86400, show_spinner=False)

def get_all_reviews(app_details, app_country_list):

    finaldfandroid = fetch_all_android(app_details)

    finaldfios = fetch_all_ios(app_country_list)  # ✅ Removed pages argument

    if not finaldfandroid.empty and not finaldfios.empty:

        finaldf = pd.concat([finaldfandroid, finaldfios], ignore_index=True)

    elif not finaldfandroid.empty:

        finaldf = finaldfandroid

    elif not finaldfios.empty:

        finaldf = finaldfios

    else:

        finaldf = pd.DataFrame()

    return finaldf

 

 

# --- Twitter Bearer Token (hardcoded) ---
# Replace the value below with your actual Twitter API v2 Bearer Token.
# It's URL-encoded here; decode it to get the raw token value.
# Get one at: developer.twitter.com → Projects → Keys & Tokens
import urllib.parse
_twitter_bearer = urllib.parse.unquote("AAAAAAAAAAAAAAAAAAAAAPdy8gEAAAAAUEixXc%2BnIEAQa5mIQVVCly%2Fxv5E%3Df5WeZ6s7AkB3aPZfAxgaLoHsYjmiPsN6wrk8SPYQy4HPAxqFt8")

with st.spinner("Fetching Android & iOS reviews..."):

    finaldf = get_all_reviews(app_details, app_country_list)

with st.spinner("Fetching TrustPilot reviews..."):

    _tp_df = fetch_trustpilot_reviews()

if _twitter_bearer:

    with st.spinner("Fetching Twitter/X mentions via API..."):

        _tw_df = fetch_twitter_reviews(_twitter_bearer)

else:

    _tw_df = pd.DataFrame()

# Merge external sources into the main dataframe (only non-empty ones)

_extra_sources = [f for f in [_tp_df, _tw_df] if not f.empty]

if _extra_sources:

    finaldf = pd.concat([finaldf] + _extra_sources, ignore_index=True)

 

 

finaldf.columns = finaldf.columns.str.strip("'")

finaldf.columns = [c.replace(' ', '_') for c in finaldf.columns]

 

# Convert TimeStamp to datetime for filtering
# utc=True handles mixed tz-aware / tz-naive values; tz_localize(None) strips tz info

finaldf["TimeStamp"] = pd.to_datetime(finaldf["TimeStamp"], utc=True).dt.tz_localize(None)

finaldf["DateTimeStamp"] = finaldf["TimeStamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

 

today = datetime.date.today()

default_end = today

default_start = today.replace(day=1)

 

# Initialize session state

if "start_date" not in st.session_state:

    st.session_state.start_date = default_start

if "end_date" not in st.session_state:

    st.session_state.end_date = default_end

if "show_custom_dates" not in st.session_state:

    st.session_state.show_custom_dates = False

 

# --- Add buttons for quick date range selection ---

col_btn1, col_btn2, col_btn3, col_btn4, col_btn5, col_btn6 = st.columns(6)

 

with col_btn1:

    if st.button("3 Months"):

        st.session_state.start_date = today - datetime.timedelta(days=90)

        st.session_state.end_date = today

        st.session_state.show_custom_dates = False

 

with col_btn2:

    if st.button("6 Months"):

        st.session_state.start_date = today - datetime.timedelta(days=180)

        st.session_state.end_date = today

        st.session_state.show_custom_dates = False

 

with col_btn3:

    if st.button("9 Months"):

        st.session_state.start_date = today - datetime.timedelta(days=270)

        st.session_state.end_date = today

        st.session_state.show_custom_dates = False

 

with col_btn4:

    if st.button("12 Months"):

        st.session_state.start_date = today - datetime.timedelta(days=365)

        st.session_state.end_date = today

        st.session_state.show_custom_dates = False

 

with col_btn5:

    if st.button("Custom"):

        st.session_state.show_custom_dates = True

 

with col_btn6:

    if st.button("Reset"):

        st.session_state.start_date = default_start

        st.session_state.end_date = default_end

        st.session_state.show_custom_dates = False

 

# --- Date Range Selection Header ---

st.markdown("<br>", unsafe_allow_html=True)

# st.markdown(

#     "<h5 style='margin-bottom: 10px;'>Date Range Selection</h5>",

#     unsafe_allow_html=True

# )

 

# --- Show Date Inputs only if "Custom" is selected ---

if st.session_state.show_custom_dates:

    col1, col2 = st.columns((2))

    with col1:

        date1 = st.date_input("**Start Date**", value=st.session_state.start_date)

        st.session_state.start_date = date1

    with col2:

        date2 = st.date_input("**End Date**", value=st.session_state.end_date)

        st.session_state.end_date = date2

else:

    date1 = st.session_state.start_date

    date2 = st.session_state.end_date

 

# --- Convert to datetime.datetime for comparison ---

date1_dt = datetime.datetime.combine(date1, datetime.datetime.min.time())

date2_dt = datetime.datetime.combine(date2, datetime.datetime.max.time())

 

# --- Validation: Start date must be before end date ---

if date1_dt > date2_dt:

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.error("⚠️ Start Date must be before End Date.")

    st.stop()

    df = pd.DataFrame()

else:

    # --- Filter the dataframe based on selected date range ---

    try:

        filtered_df = finaldf[(finaldf["TimeStamp"] >= date1_dt) & (finaldf["TimeStamp"] <= date2_dt)]

        df = filtered_df.copy()

    except KeyError:

        df = pd.DataFrame()

 

    # --- Show selected date range summary ---

    if not st.session_state.show_custom_dates:

        st.success(f"📅 Showing Data from **{date1.strftime('%d-%b-%Y')}** to **{date2.strftime('%d-%b-%Y')}**")

 

 

 

configured_country_codes = {

    str(country).lower() for _, country, _ in app_details

}.union({

    str(country).lower() for _, country in app_country_list

})

 

data_country_codes = set(df["Country"].astype(str).str.lower().unique()) if "Country" in df.columns else set()

country_codes_for_dropdown = sorted(data_country_codes.union(configured_country_codes))

 

# Create list of full country names for dropdown

country_names = sorted({country_map.get(code, code) for code in country_codes_for_dropdown})

 

# Sidebar country selection

selected_country_names = st.sidebar.multiselect(

    "**Country Selection**",

    options=sorted(country_names),

    placeholder="Select Country/s",

    key="Country Selection"

)

 

# Convert selected country names back to codes

selected_country_codes = [

    code for code in country_codes_for_dropdown

    if country_map.get(code, code) in selected_country_names

]

 

 

# Filter by country

if not selected_country_codes:

    df1 = df.copy()

else:

    df1 = df[df["Country"].isin(selected_country_codes)]

 

# country = st.sidebar.multiselect("**Select the Countries**", df["Country"].unique(),placeholder="")

# if not country:

#     df1 = df.copy()

# else:

#     df1 = df[df["Country"].isin(country)]

 

region = st.sidebar.multiselect(

    "**Select the App Type**",

    df["AppName"].unique(),

    placeholder="Select the App Type",

    key="Select the App Type"

)

if not region:

    df2 = df1.copy()

else:

    df2 = df1[df1["AppName"].isin(region)]

 

if not selected_country_codes and not region :

    filtered_df = df

else:

    filtered_df=df2

 

 

if 'rating' in filtered_df.columns:

    filtered_df['rating'] = pd.to_numeric(filtered_df['rating'], errors='coerce') # convert rating to numeric (int/float)

    rating = st.sidebar.slider("**Filter by Ratings Range**", 1, 5, (1, 5))

    if rating:

        filtered_df = filtered_df[(filtered_df['rating'] >= rating[0]) & (filtered_df['rating'] <= rating[1])]

 

 

def _derive_platform(app_name):

    x = str(app_name).lower()

    if "android" in x:

        return "Android"

    elif "twitter" in x:

        return "Twitter"

    elif "trustpilot" in x:

        return "TrustPilot"

    else:

        return "iOS"

filtered_df["Platform"] = filtered_df["AppName"].apply(_derive_platform)

filtered_df["CountryName"] = filtered_df["Country"].apply(iso2_to_name)

 

country_platform_avg = (

    filtered_df.groupby(["CountryName", "Platform"])["rating"]

               .mean()

               .reset_index()

)

 

 

# Function to apply row-wise styling

def highlight_rating(row):

    try:

        rating = int(row['rating']) # Convert to int

    except:

        return [''] * len(row) # No styling if conversion fails

 

    if rating >= 4:

        return ['background-color: lightgreen'] * len(row)

    elif rating == 3:

        return ['background-color: yellow'] * len(row)

    else:

        return ['background-color: salmon'] * len(row)

 

 

def get_random_score_by_rating(rating):

    rating_ranges = {

        5: (0.81, 1.0),

        4: (0.61, 0.80),

        3: (0.41, 0.60),

        2: (0.21, 0.40),

        1: (0.0, 0.20)

    }

    return round(random.uniform(*rating_ranges.get(rating, (0.0, 0.20))), 2)

 

 

 

def get_score_by_rating(rating):

    rating_ranges = {

        5: (0.81, 1.0),

        4: (0.61, 0.80),

        3: (0.41, 0.60),

        2: (0.21, 0.40),

        1: (0.0, 0.20)

    }

    low, high = rating_ranges.get(rating, (0.0, 0.20))

    return round((low + high) / 2, 2)  # midpoint instead of random

 

 

def get_sentiment_score(row):

    review = str(row['review']).strip()

    rating = int(row['rating'])

 

    # If empty or single-word review → fallback to rating-based score

    if not review or len(review.split()) == 1:

        return get_score_by_rating(rating)

 

    # VADER sentiment

    sentiment = sid.polarity_scores(review)

    normalized_score = (sentiment['compound'] + 1) / 2  # 0–1 scale

 

    # Weight rating more heavily

    rating_score = get_score_by_rating(rating)

    final_score = (normalized_score * 0.3) + (rating_score * 0.7)

    return round(final_score, 2)

 

 

# Updated sentiment label logic

def get_sentiment_label(row):

    review = str(row['review']).strip()

    rating = row['rating']

 

    if review == '':

        if rating in [4, 5]:

            return 'Positive'

        elif rating == 3:

            return 'Neutral'

        elif rating in [1, 2]:

            return 'Negative'

        else:

            return 'Neutral' 

    else:

        if rating in [0, 1]:

            return 'Negative'

        elif rating == 3:

            return 'Neutral'

        elif rating in [4, 5]:

            return 'Positive'

        else:

            return 'Neutral'  # Default instead of 'Unknown'

 

def get_sentiment_emoticon(row):

    """Get emoticon based on sentiment"""

    review = str(row['review']).strip()

    rating = row['rating']

 

    if review == '':

        if rating ==5 :

            return '😍'

        elif rating == 4:

            return '😃'

        elif rating == 3:

            return '😐'

        elif rating ==2:

            return '😢'

        else:

            return '😭'

    else:

        if rating ==1:

            return '😭'

        elif rating == 2:

            return '😢'

        elif rating == 3:

            return '😐'

        elif rating ==4 :

            return '😃'

        else:

            return '😍'

 

 

 

 

 

 

def get_emoji_stars(rating):

    if pd.isnull(rating):

        return ""

    rating = int(rating)

    if rating == 5:

        return "🟩🟩🟩🟩🟩"  # Dark green squares for 5 stars

    elif rating == 4:

        return "🟩🟩🟩🟩"    # Green squares for 4 stars

    elif rating == 3:

        return "🟨🟨🟨"      # Yellow squares for 3 stars

    elif rating == 2:

        return "🟧🟧"        # Red squares for 2 stars

    elif rating == 1:

        return "🟥"          # Dark red square for 1 star

    return ""

 

 

 

def show_centered_warning(message="⚠️ No records found within the specified date range"):

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.warning(message)

 

 

def plot_bar(subplot,filtered_df):

    plt.subplot(1,2,subplot)

    axNewest=sns.barplot(y='Country',x='rating',hue='AppName',data=filtered_df, color='slateblue')

    plt.title('Ratings vs country',fontsize=70)

    # plt.xlabel('Ratings vs Country',fontsize=50)

    plt.ylabel(None)

    # plt.xticks(fontsize=40)

    # plt.yticks(fontsize=40)

    # sns.despine(left=True)

    axNewest.grid(False)

    axNewest.tick_params(bottom=True,left=False)

    return None

 

 

if filtered_df.empty:

  #st.warning("No records found within the specified date range")

   show_centered_warning()

else:

    @st.cache_data(show_spinner=False)

    def compute_review_features(df):

        df = df.reset_index(drop=True)

        df.index += 1  # Start index from 1

        df.index.name = "S.No."

        df['sentiment_score'] = df.apply(get_sentiment_score, axis=1)

        df['sentiment_label'] = df.apply(get_sentiment_label, axis=1)

        df['HappinessIndex'] = df.apply(get_sentiment_emoticon, axis=1)

        df["CountryName"] = df["Country"].str.lower().map(country_map).fillna(df["Country"])

        df["CustomerRating"] = df["rating"].apply(get_emoji_stars)

        df = df.reindex([ 'DateTimeStamp',  'review','CustomerRating', 'sentiment_score',

                         'HappinessIndex', 'CountryName', 'AppName',

                         'appVersion','rating','sentiment_label'], axis=1)

        return df

    filtered_df = compute_review_features(filtered_df)
    # Sanitize review text to remove any embedded HTML (prevents raw HTML/tooltips overlapping)
    try:
        import re

        def _strip_html_tags(text):
            return re.sub(r'<[^>]*>', '', str(text))

        if 'review' in filtered_df.columns:
            filtered_df['review'] = filtered_df['review'].astype(str).apply(_strip_html_tags)
    except Exception:
        # If sanitization fails for any reason, keep the original text
        pass

 

# 'UserName'

 

def format_column_label(s):

    # Split by underscores and capitalize each part

    return '_'.join(word.capitalize() for word in s.split('_'))

 

 

def format_column_label(col):

    custom_labels = {

        "TimeStamp": "Date",    

        "AppName": "App Type",

        "UserName": "User Name",

        "appVersion":"Version",

        "sentiment_score":"Sentiment Score",

        "sentiment_label":"Sentiment Label"

    }

    if col in custom_labels:

        return custom_labels[col]

    return '_'.join(word.capitalize() for word in col.split('_'))

 

 

def to_title_case_with_underscores(s):

    return '_'.join(word.capitalize() for word in s.split('_'))

 

 

column_config = {

    col: st.column_config.TextColumn(label=to_title_case_with_underscores(col))

    for col in filtered_df.columns

}

 

column_config = {

    col: st.column_config.TextColumn(label=format_column_label(col))

    for col in filtered_df.columns

}

 

 

# --- Caching grouped data computation

@st.cache_data(show_spinner=False)

def compute_grouped_data(df):

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    #df["CountryName"] = df["Country"].apply(iso2_to_name)

    df["ISO3"] = df["CountryName"].apply(name_to_iso3)

    grouped = (

        df.groupby(["CountryName", "ISO3"])

        .agg(avg_rating=("rating", "mean"), review_count=("rating", "count"))

        .reset_index()

    )

    return grouped

 

@st.cache_data(show_spinner=False)

def generate_figures(grouped):

    figures = []

    font_size = 12

 

    def rating_to_color(rating):

        if rating < 2.5:

            return "#B22222"  # Dark Red

        elif rating < 4.0:

            return "#FF8C00"  # Dark Orange

        else:

            return "#228B22"  # Forest Green

 

    for _, row in grouped.iterrows():

        fill_color = rating_to_color(row["avg_rating"])

 

        fig = go.Figure()

 

        # Country fill

        fig.add_trace(go.Choropleth(

            locations=[row["ISO3"]],

            z=[row["avg_rating"]],

            locationmode="ISO-3",

            colorscale=[[0, fill_color], [1, fill_color]],

            showscale=False,

            marker_line_color="gray",

            marker_line_width=0.5,

            hoverinfo="skip"

        ))

 

        # Annotation box in bottom center

        annotation_text = (

            f"<b>{row['CountryName']}</b><br>"

            f"⭐ Rating: {row['avg_rating']:.2f}<br>"

            f"📝 Reviews: {row['review_count']}"

        )

       

 

        fig.update_layout(

            annotations=[

                dict(

                    x=0.5,

                    y=0.01,

                    xref='paper',

                    yref='paper',

                    showarrow=False,

                    align='center',

                    text=annotation_text,

                    font=dict(size=font_size, color="black"),

                    bgcolor="white",

                    bordercolor="gray",

                    borderwidth=3,

                    opacity=0.98 

                )

            ],

            title={

                "text": f"🌐 Ratings for {row['CountryName']}",

                "x": 0.5,

                "xanchor": "center",

                "font": dict(size=18, family="Arial Black", color="black")

            },

            margin=dict(l=0, r=0, t=50, b=0),

            paper_bgcolor='white',

            plot_bgcolor='white',

            geo=dict(

                showcoastlines=True,

                coastlinecolor="LightGray",

                showland=True,

                landcolor="whitesmoke",

                showocean=True,

                oceancolor="aliceblue",

                showlakes=True,

                lakecolor="lightblue",

                showrivers=True,

                rivercolor="lightblue",

                showcountries=True,

                countrycolor="gray",

                projection_type="equirectangular",

                bgcolor='white',

                resolution=50,               

                showsubunits=True,

                subunitcolor="lightgray",

                showframe=True,

                framecolor="black",

               

                center=dict(lat=20, lon=0),

                projection_scale=1  # Zoom level 

            )

        )

        figures.append(fig)

 

    return figures

 

 

def show_world_map(filtered_df, date1, date2):

    st.markdown(

        """

        <style>

        .date-range-text {

            font-size: 14px;

            font-weight: bold;

            text-align: center;

            margin-bottom: 10px;

            z-index: 9999;

        }

        </style>

        """,

        unsafe_allow_html=True,

    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    with st.spinner("⏳ Please wait while the Bar chart is getting ready..."):

        # Average rating by country

        filtered_df["ISO3"] = filtered_df["CountryName"].apply(name_to_iso3)

        mean_ratings = filtered_df.groupby('ISO3')['rating'].mean().reset_index()

 

        figNewer = plt.figure(figsize=(12, 5))  # Wider for more countries

        # Custom color mapping for ratings

        def rating_color(val):

            if val <= 2:

                return '#d62728'  # red

            elif 2 < val < 3:

                return '#ff9800'  # orange

            elif 3 <= val < 4:

                return '#4caf50'  # green

            elif val >= 4:

                return '#006400'  # dark green

            else:

                return '#1f77b4'  # fallback blue

        bar_colors = mean_ratings['rating'].apply(rating_color).tolist()

        axar = sns.barplot(x='ISO3', y='rating', data=mean_ratings)

        # Set bar colors manually

        for bar, color in zip(axar.patches, bar_colors):

            bar.set_facecolor(color)

        # Remove the legend/title 'Average Rating By Country'

        # Set y-axis title to 'Average App rating by Country'

        axar.set(xlabel='Country', ylabel='Average App rating by Country', title='')

        # Bolden and enlarge x and y axis labels

        axar.xaxis.label.set_fontweight('bold')

        axar.yaxis.label.set_fontweight('bold')

        axar.xaxis.label.set_size(14)

        axar.yaxis.label.set_size(14)

        # Rotate x-axis labels to prevent overlap

        axar.set_xticklabels(axar.get_xticklabels(), rotation=45, ha='right', fontsize=10, fontweight='bold')

        # Beautify chart

        axar.set_yticks(np.arange(1, 5.5, 0.5))

        axar.set_ylim(0.5, 5)

        axar.grid(axis='y', linestyle='--', alpha=0.5)

        axar.set_facecolor('#f9f9f9')

        figNewer.patch.set_facecolor('#f9f9f9')

        for spine in ['top', 'right', 'left', 'bottom']:

            axar.spines[spine].set_visible(False)

        # Bar labels in normal font (not bold)

        for container in axar.containers:

            axar.bar_label(container, fmt='%.2f', fontsize=10, fontweight='normal', label_type='edge', padding=2)

        plt.tight_layout()

        st.pyplot(figNewer)

        st.markdown("<br><br>", unsafe_allow_html=True)

 

       

 

        # Dumbbell chart (Android vs iOS) fully removed as per user request

 

 

 

 

def show_sunburst_chart(filtered_df, date1, date2):

    if filtered_df.empty:

        show_timed_warning_generic("⚠️ No records found within the specified date range", duration=4)

        return

 

    filtered_df['rating'] = pd.to_numeric(filtered_df['rating'], errors='coerce')

    date_diff = (date2 - date1).days

 

    if date_diff > 90:

        show_timed_warning_generic("⚠️ Sunburst Chart is disabled for date ranges longer than 3 months", duration=3)

        return

 

    # Show progress bar before rendering chart

    progress = st.progress(0)

    status = st.empty()

    for i in range(100):

        time.sleep(0.005)

        progress.progress(i + 1)

 

    # --- Function to add topic sections ---

    # PDF generation and topic modeling summary are disabled due to missing variables.

    # To enable, ensure positive_topics, negative_topics, neutral_topics, and their sentences are defined.

    # def add_topic_section_to_pdf(pdf, section_title, topics, sentences):

    #     ...

    # try:

    #     from fpdf import FPDF

    #     if all(x in locals() for x in ["positive_topics", "positive_sentences", "negative_topics", "negative_sentences", "neutral_topics", "neutral_sentences"]):

    #         pdf = FPDF()

    #         pdf.add_page()

    #         add_topic_section_to_pdf(pdf, "Top 5 Best Aspects", positive_topics, positive_sentences)

    #         add_topic_section_to_pdf(pdf, "Top 5 Issues", negative_topics, negative_sentences)

    #         add_topic_section_to_pdf(pdf, "Top 5 Neutral Aspects", neutral_topics, neutral_sentences)

    #         pdf.output("topic_modeling_summary.pdf")

    # except Exception as e:

    #     print(f"PDF generation skipped or failed: {e}")

 

 

 

 

    # --- Sunburst Chart Visualization ---

    st.write("### Sunburst Chart")

    import plotly.express as px

    import numpy as np

    # Ensure required columns exist

    sunburst_cols = ['CountryName', 'AppName', 'rating', 'review']

    for col in sunburst_cols:

        if col not in filtered_df.columns:

            filtered_df[col] = 'Unknown'

    # Remove rows with missing ratings

    filtered_df = filtered_df.dropna(subset=['rating'])

    if filtered_df.empty:

        show_timed_warning_generic("⚠️ No records found for Sunburst Chart", duration=4)

        return

    try:

        fig = px.sunburst(

            filtered_df,

            path=['CountryName', 'AppName', 'rating', 'review'],

            values='rating',

            color='rating',

            color_continuous_scale='RdBu',

            color_continuous_midpoint=np.average(filtered_df['rating'], weights=filtered_df['rating']) if not filtered_df['rating'].isnull().all() else 3,

            title=""

        )

        fig.update_traces(hovertemplate="")

        fig.update_layout(width=800, height=800, coloraxis_showscale=False)

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:

        st.warning(f"Error generating Sunburst Chart: {e}")

 

 

 

def remove_emojis(text):

    # This function removes emojis from the input text

    return text.encode('ascii', 'ignore').decode('ascii')

 

 

def show_word_cloud(filtered_df):

    if filtered_df.empty:

        show_timed_warning_generic("⚠️ No records found within the specified date range", duration=4)

        return

 

    import re
    from collections import Counter
    import plotly.graph_objects as go

    try:
        # Comprehensive stop words: NLTK English + app-specific + common filler words
        from nltk.corpus import stopwords as nltk_stopwords
        try:
            _nltk_sw = set(nltk_stopwords.words('english'))
        except Exception:
            _nltk_sw = set()

        custom_stopwords = _nltk_sw | {
            # App / review meta words
            "app", "application", "apps", "review", "reviews", "store",
            "play", "update", "updated", "version", "versions", "device",
            "devices", "phone", "phones", "western", "union", "wu",
            "western union", "western_union",
            # Generic filler / opinion words that add no insight
            "good", "great", "bad", "okay", "ok", "nice", "use", "used",
            "using", "uses", "works", "work", "worked", "working", "need",
            "needs", "needed", "make", "makes", "made", "making", "got",
            "get", "gets", "getting", "just", "really", "actually",
            "always", "never", "every", "still", "also", "even", "much",
            "many", "more", "most", "best", "well", "like", "something",
            "nothing", "everything", "thing", "things", "way", "ways",
            "time", "times", "one", "two", "three", "first", "last",
            "new", "old", "able", "unable", "keep", "kept", "tried",
            "try", "trying", "let", "please", "thank", "thanks",
            "service", "customer", "support", "help", "helpful", "issue",
            "issues", "problem", "problems", "fix", "fixed", "fixes",
            "easy", "hard", "fast", "slow", "slows", "said", "say",
            "says", "come", "came", "come", "going", "went", "gone",
            "know", "known", "take", "took", "taken", "give", "given",
            "show", "showed", "shown", "see", "seen", "want", "wanted",
            "back", "now", "here", "there", "when", "where", "would",
            "could", "should", "might", "must", "shall", "will", "may",
            "been", "being", "was", "were", "had", "have", "has",
            "did", "does", "done", "put", "set", "run", "ran",
        }

        # Detect review column safely
        review_col = next((c for c in filtered_df.columns if c.lower() == 'review'), None)
        if review_col is None:
            st.warning("⚠️ No review column found in data.")
            return

        # Work on safe copy — only the column we need
        wc_df = filtered_df[[review_col]].copy()
        wc_df['_wc_score'] = wc_df[review_col].apply(
            lambda x: sia.polarity_scores(str(x))['compound']
        )
        wc_df['_wc_label'] = wc_df['_wc_score'].apply(
            lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral')
        )

        sentiment_labels = sorted(wc_df['_wc_label'].dropna().unique().tolist())
        if not sentiment_labels:
            st.warning("⚠️ Could not detect sentiment labels.")
            return

        # Colour map per sentiment
        sentiment_colors = {
            'Positive': '#27ae60',
            'Negative': '#e74c3c',
            'Neutral':  '#f39c12',
        }

        # Sentiment selector — MUST be outside spinner
        st.markdown("<h4 style='text-align:center; font-weight:bold;'>Select Sentiment for Word Cloud</h4>",
                    unsafe_allow_html=True)
        sentiment_option = st.selectbox(
            "Sentiment for Word Cloud",
            sentiment_labels,
            label_visibility="collapsed"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Build word frequencies for selected sentiment
        reviews_text = " ".join(
            wc_df[wc_df['_wc_label'] == sentiment_option][review_col]
            .fillna("").astype(str).tolist()
        )
        # Extract only alphabetic words of 3+ chars, lowercased
        words = re.findall(r'\b[a-zA-Z]{3,}\b', reviews_text.lower())
        # Remove all stop words (NLTK + custom)
        filtered_words = [w for w in words if w not in custom_stopwords]
        word_counts = Counter(filtered_words)
        # Also drop the top 5 most dominant words that survived (corpus-specific noise)
        dominant = {w for w, _ in word_counts.most_common(5)}
        word_counts = Counter({w: c for w, c in word_counts.items() if w not in dominant})
        top_words = word_counts.most_common(40)

        if not top_words:
            st.warning("⚠️ Not enough words for the selected sentiment.")
            return

        with st.spinner("☁️ Generating Word Frequency Chart, Please wait..."):
            labels = [w for w, _ in top_words]
            sizes  = [c for _, c in top_words]
            bar_color = sentiment_colors.get(sentiment_option, '#FFD700')

            fig = go.Figure(go.Bar(
                x=sizes[::-1],
                y=labels[::-1],
                orientation='h',
                marker=dict(
                    color=sizes[::-1],
                    colorscale=[[0, '#ffe066'], [1, bar_color]],
                    line=dict(color='#333333', width=0.5),
                ),
                text=[str(s) for s in sizes[::-1]],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>',
            ))

            fig.update_layout(
                title=dict(
                    text=f'Top Words — {sentiment_option} Reviews',
                    x=0.5, xanchor='center',
                    font=dict(size=18, color='#333')
                ),
                xaxis=dict(title='Frequency', showgrid=True, gridcolor='#eeeeee'),
                yaxis=dict(title='', tickfont=dict(size=12)),
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=700,
                margin=dict(l=20, r=60, t=60, b=40),
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Word Frequency Chart error: {e}")

 

def show_treemap_chart(filtered_df, date1, date2):

    if filtered_df.empty:

        show_timed_warning_generic("⚠️ No records found within the specified date range", duration=4)

        return

 

    date_diff = (date2 - date1).days

    if date_diff > 90:

        show_timed_warning_generic("⚠️ TreeMap Chart is disabled for date ranges longer than 3 months", duration=3)

        return

 

    # Remove artificial progress simulation for large datasets

    st.write("### TreeMap Chart")

 

    # Limit the data size for rendering to avoid server overload (adjust threshold as needed)

    MAX_ROWS = 2000

    if len(filtered_df) > MAX_ROWS:

        filtered_df = filtered_df.sample(n=MAX_ROWS, random_state=42)

 

    filtered_df = filtered_df.fillna('end_of_hierarchy')

 

    # Use Streamlit spinner only for chart generation

    with st.spinner("🌳 Generating TreeMap Chart..."):

        try:

            fig3 = px.treemap(

                filtered_df,

                path=["CountryName", "AppName", "rating", "review"],

                hover_data=["rating"],

                color="review"

            )

            fig3.update_traces(hovertemplate='')

            st.plotly_chart(fig3, use_container_width=True)

        except Exception as e:

            st.warning(f"Error generating TreeMap: {e}")

 

 

 

def show_visual_charts(filtered_df, df, date1, date2):

    if filtered_df.empty:

        show_timed_warning_generic("⚠️ No records found within the specified date range", duration=3)

        return

 

    with st.spinner("⏳ Loading Charts, please wait..."):

        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown(

            "<div style='text-align: center; font-size: 18px;'><b>Consolidated Sentiment across Countries</b></div>",

            unsafe_allow_html=True

        )

        st.markdown("<br><br>", unsafe_allow_html=True)

 

 

        # Pie chart for sentiment distribution (smaller, beautified)

        pie_fig = px.pie(

            filtered_df,

            names='sentiment_label',

            color='sentiment_label',

            color_discrete_map={

                'Positive': '#4caf50',

                'Negative': '#e53935',

                'Neutral': '#ffeb3b'

            },

            hole=0.35

        )

        pie_fig.update_traces(

            textposition='inside',

            textinfo='percent+label',

            marker=dict(line=dict(color='#fff', width=2)),

            pull=[0.03, 0.03, 0.03]

        )

        pie_fig.update_layout(

            showlegend=True,

            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5, font=dict(size=11)),

            margin=dict(l=10, r=10, t=30, b=10),

            height=320,

            font=dict(size=13),

            legend_itemclick=False,

            legend_itemdoubleclick=False

        )

        st.plotly_chart(pie_fig, use_container_width=False)

 

        st.markdown("<br><b>Consolidated Ratings across Countries</b><br>", unsafe_allow_html=True)

 

        # Ensure all ratings from 1 to 5 are present

        rating_counts = filtered_df['rating'].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)

 

        # Convert to DataFrame for plotting

        plot_df = pd.DataFrame({'rating': rating_counts.index, 'count': rating_counts.values})

 

        # Horizontal bar chart for ratings distribution (clear and visually appealing)

        bar_colors = ['#e53935', '#ff9800', '#ffeb3b', '#4caf50', '#006400']

        fig, ax = plt.subplots(figsize=(6, 3.2))

        bars = ax.barh(plot_df['rating'], plot_df['count'], color=bar_colors, edgecolor='black', height=0.6)

        ax.set_xlabel('Count', fontsize=12, fontweight='bold')

        ax.set_ylabel('Rating', fontsize=12, fontweight='bold')

        ax.set_yticks(plot_df['rating'])

        ax.set_yticklabels(plot_df['rating'], fontsize=11, fontweight='bold')

        ax.set_xticks([])

        ax.set_facecolor('#f9f9f9')

        fig.patch.set_facecolor('#f9f9f9')

        for spine in ['top', 'right', 'left', 'bottom']:

            ax.spines[spine].set_visible(False)

        for bar in bars:

            width = bar.get_width()

            ax.annotate(f'{int(width)}',

                        xy=(width, bar.get_y() + bar.get_height() / 2),

                        xytext=(5, 0),

                        textcoords="offset points",

                        ha='left', va='center', fontsize=11, fontweight='bold')

        plt.tight_layout(pad=1.0)

        st.pyplot(fig)

 

        # Funnel chart for issue keywords

        issue_keywords = {

            'Crashes': ['freezes', 'crash', 'stuck'],

            'Stop': ['stop', 'shut', 'close'],

            'Hang': ['hang', 'hangs', 'freeze'],

            'Bugs': ['bug', 'bugs', 'error'],

            'Performance': ['performance', 'slow', 'lag'],

            'Customer': ['customer', 'helpdesk', 'support'],

            'Update': ['update', 'upgrade', 'patch'],

            'Notification': ['notification', 'alert'],

            'Ads/Popup': ['ads', 'ad', 'pop-up', 'popup', 'pop up', 'popups', 'advertisement', 'intrusive', 'offers', 'promotions', 'promotion', 'too many offers', 'too many ads', 'interruption'],

            'OTP': ['otp', 'message', 'verification'],

            'UI': ['ui', 'interface', 'design'],

            'App': ['app', 'application', 'software']

        }

 

       

 

    # Insert line breaks for long labels to prevent truncation

    def wrap_label(label, max_len=18):

        if len(label) > max_len and ' ' in label:

            parts = label.split(' ')

            mid = len(parts) // 2

            return ' '.join(parts[:mid]) + '<br>' + ' '.join(parts[mid:])

        return label

 

    funnel_labels = ['All Reviews', 'Filtered Negatives'] + [wrap_label(l) for l in issue_keywords.keys()] + ['Other Issues']

 

    # Filter negative reviews with rating 1, 2, 3

    filtered_negatives = filtered_df[

        (filtered_df['sentiment_label'].str.lower() == 'negative') &

        (filtered_df['rating'].isin([1, 2, 3]))

    ]

 

    # Initial counts

    all_reviews = len(filtered_df)

    counts = [all_reviews, len(filtered_negatives)]

 

    # Track indices for each stage

    stage_indices = {

        'All Reviews': df.index.tolist(),

        'Filtered Negatives': filtered_negatives.index.tolist(),

    }

 

    covered_indices = set()

 

    # Loop through issues and count matches

    for issue, keywords in issue_keywords.items():

        if isinstance(keywords, str):

            keywords = [keywords]

 

        mask = filtered_negatives['review'].fillna("").str.lower().apply(

            lambda text: any(kw.lower() in text for kw in keywords)

        )

 

        indices = filtered_negatives[mask].index.tolist()

        counts.append(len(indices))

        stage_indices[issue] = indices

        covered_indices.update(indices)

 

    # Other issues

    other_issues_indices = filtered_negatives.drop(index=list(covered_indices)).index.tolist()

    counts.append(len(other_issues_indices))

    stage_indices['Other Issues'] = other_issues_indices

 

    # Colors for funnel stages

    custom_colors = [

        "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c",

        "#9467bd", "#8c564b", "#bcbd22", "#7f7f7f"

    ]

 

    # Funnel chart for issues is hidden as per user request

 

 

def run_lda_topic_modeling(reviews, n_topics=5, n_words=10):

    # 1. Preprocess reviews (remove NaN, join to string)

    texts = reviews.dropna().astype(str).tolist()

   

    # 2. Vectorize

    vectorizer = CountVectorizer(stop_words='english', max_features=1000)

    X = vectorizer.fit_transform(texts)

   

    # 3. Fit LDA

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)

    lda.fit(X)

   

    # 4. Get top words for each topic

    feature_names = vectorizer.get_feature_names_out()

    topics = []

    for idx, topic in enumerate(lda.components_):

        top_features = [feature_names[i] for i in topic.argsort()[-n_words:][::-1]]

        topics.append(f"Topic {idx+1}: " + ", ".join(top_features))

   

    return topics

 

 

 

def show_keyword_analysis(filtered_df, stop_words):

    if filtered_df.empty:

        show_timed_warning_generic("⚠️ No records found within the specified date range", duration=4)

        return

 

    st.subheader("Keyword and N-gram Analysis")

 

    with st.spinner("🔍 Performing Keyword & N-gram Analysis... Please wait"):

        # Sentiment Analysis

        filtered_df['sentiment_score'] = filtered_df['review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

        filtered_df['sentiment_label'] = filtered_df['sentiment_score'].apply(

            lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral')

        )

 

        # Filters

        selected_sentiment = st.selectbox("Filter by Sentiment", filtered_df['sentiment_label'].unique().tolist() + ['All'])

        selected_apptype = st.selectbox("Filter by App Type", filtered_df['AppName'].dropna().unique().tolist() + ['All'])

        ngram_count = st.slider("Number of Top N-grams to Display", 5, 30, 10)

 

        df_filtered = filtered_df.copy()

        if selected_sentiment != 'All':

            df_filtered = df_filtered[df_filtered['sentiment_label'] == selected_sentiment]

        if selected_apptype != 'All':

            df_filtered = df_filtered[df_filtered['AppName'] == selected_apptype]

 

        # Tokenization

        all_text = " ".join(df_filtered['review'].astype(str).tolist()).lower()

        tokens = [word for word in nltk.word_tokenize(all_text) if word.isalpha() and word not in stop_words]

 

        # N-gram Frequencies

        unigram_freq = Counter(tokens)

        bigram_freq = Counter(ngrams(tokens, 2))

        trigram_freq = Counter(ngrams(tokens, 3))

 

        # DataFrames

        top_unigrams = pd.DataFrame(unigram_freq.most_common(ngram_count), columns=['Unigram', 'Count']).sort_values(by='Count').reset_index(drop=True)

        top_bigrams = pd.DataFrame(bigram_freq.most_common(ngram_count), columns=['Bigram', 'Count']).sort_values(by='Count').reset_index(drop=True)

        top_trigrams = pd.DataFrame(trigram_freq.most_common(ngram_count), columns=['Trigram', 'Count']).sort_values(by='Count').reset_index(drop=True)

 

        # Convert text columns

        top_unigrams['Unigram'] = top_unigrams['Unigram'].astype(str)

        top_bigrams['Bigram'] = top_bigrams['Bigram'].apply(lambda x: ' '.join(x))

        top_trigrams['Trigram'] = top_trigrams['Trigram'].apply(lambda x: ' '.join(x))

 

        # Ensure Count is numeric

        top_unigrams['Count'] = pd.to_numeric(top_unigrams['Count'])

        top_bigrams['Count'] = pd.to_numeric(top_bigrams['Count'])

        top_trigrams['Count'] = pd.to_numeric(top_trigrams['Count'])

 

   

    # Charts using go.Figure with .tolist()

    col1, col2, col3 = st.columns(3)

 

    with col1:

        st.write("### Top Unigrams")

        st.dataframe(top_unigrams)

        fig_uni = go.Figure(go.Bar(x=top_unigrams['Unigram'].tolist(), y=top_unigrams['Count'].tolist(), marker_color='blue'))

        fig_uni.update_layout(title='Top Unigrams', xaxis_title='Unigram', yaxis_title='Count')

        fig_uni.update_xaxes(tickangle=-90)

        st.plotly_chart(fig_uni, use_container_width=True)

 

    with col2:

        st.write("### Top Bigrams")

        st.dataframe(top_bigrams)

        fig_bi = go.Figure(go.Bar(x=top_bigrams['Bigram'].tolist(), y=top_bigrams['Count'].tolist(), marker_color='blue'))

        fig_bi.update_layout(title='Top Bigrams', xaxis_title='Bigram', yaxis_title='Count')

        fig_bi.update_xaxes(tickangle=-90)

        st.plotly_chart(fig_bi, use_container_width=True)

 

    with col3:

        st.write("### Top Trigrams")

        st.dataframe(top_trigrams)

        fig_tri = go.Figure(go.Bar(x=top_trigrams['Trigram'].tolist(), y=top_trigrams['Count'].tolist(), marker_color='blue'))

        fig_tri.update_layout(title='Top Trigrams', xaxis_title='Trigram', yaxis_title='Count')

        fig_tri.update_xaxes(tickangle=-90)

        st.plotly_chart(fig_tri, use_container_width=True)

 

    st.markdown("<br><br>", unsafe_allow_html=True)

 

 

 

def show_translation_widget(languages):

    from googletrans import Translator

 

    source_text = st.text_area("**Enter Text to translate:**", height=100)

 

    default_language = 'English'

    default_index = languages.index(default_language) if default_language in languages else 0

 

    target_language = st.selectbox("**Select target language:**", languages, index=default_index)

    st.markdown("<br>", unsafe_allow_html=True)

 

    if st.button('Translate'):

        translator = Translator()

        try:

            out = translator.translate(source_text, dest=target_language)

            st.markdown(
                f"""
                <div style="
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    border-radius: 8px;
                    padding: 16px 24px;
                    margin-top: 10px;
                    text-align: center;
                    font-size: 1.1em;
                    color: #155724;
                    font-weight: 500;
                ">
                    ✅ &nbsp; {out.text}
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(f"Translation failed: {e}")

 

@st.cache_data(show_spinner=False)

def process_complaints(filtered_df):

    """Extract complaints directly from filtered_df"""

    if filtered_df.empty or 'review' not in filtered_df.columns:

         return pd.DataFrame(), {}

 

    df_neg = filtered_df[

        (filtered_df.get('rating', pd.Series(5)) <= 2) |

        (filtered_df.get('sentiment_score', pd.Series(0.5)) < 0.5) |

        (filtered_df['sentiment_label'] == 'Negative')

    ].copy()

 

    if df_neg.empty:

        return df_neg, {}

 

    # Ensure a proper datetime column

    if 'DateTimeStamp' in df_neg.columns:

        df_neg['DateTimeStamp'] = pd.to_datetime(df_neg['DateTimeStamp'], errors='coerce')

    else:

        df_neg['DateTimeStamp'] = pd.to_datetime(df_neg.get('TimeStamp', pd.Series()), errors='coerce')

 

    # Monthly period as timestamp for reliable plotting/grouping

    df_neg['month_year'] = df_neg['DateTimeStamp'].dt.to_period('M').dt.to_timestamp()

 

    df_neg['App_Version'] = df_neg.get('appVersion', 'Unknown').astype(str).str.extract(r'(\d+\.\d+)').fillna('Unknown')

    df_neg['CountryName'] = df_neg.get('CountryName', df_neg.get('Country', 'Unknown')).fillna('Unknown')

 

    # Refined keywords for 'App Crash' to avoid generic terms like 'slow' and 'lag'

    issue_keywords = {

        # Existing categories (do not duplicate keywords)

        'Refund': [

            'refund', 'refunded', 'request a refund', 'requested a refund', 'money back', 'get my money back', 'got my money back', 'receive a refund', 'received a refund', 'issue a refund', 'issued a refund', 'refund issued', 'refund not received', 'refund not processed', 'refund denied', 'refund request', 'refund process', 'refund status', 'refund pending', 'refund completed', 'refund successful', 'refund failed', 'stuck money'

        ],

        'Ads/Popup': [

            'ads', 'ad', 'pop-up', 'popup', 'pop up', 'popups', 'advertisement', 'intrusive', 'offers', 'promotions', 'promotion', 'too many offers', 'too many ads', 'interruption'

        ],

        'Transaction Error': [

            'transaction error', 'transaction failed', 'transaction timeout', 'transaction declined', 'c2002',

            'error', 'failed', 'timeout', 'declined',

            'cancelled transaction', 'transaction was cancelled', 'payment cancelled', 'order cancelled', 'order was cancelled', 'transfer cancelled', 'transfer was cancelled', 'cancelled by system', 'cancelled by bank', 'cancelled by user'

        ],

        'Payment/Verify': [

            # Payment method issues

            'payment method', 'change payment', 'add card', 'add bank', 'remove card', 'remove bank', 'update card', 'update bank', 'switch card', 'switch bank',

            # Card/bank rejections

            'card declined', 'card rejected', 'bank declined', 'bank rejected', 'card not accepted', 'bank not accepted', 'payment declined', 'payment rejected', 'transaction declined', 'transaction rejected',

            # Verification/compliance hurdles

            'verification', 'verify', 'compliance', 'compliance check', 'compliance issue', 'compliance review', 'compliance hold', 'compliance block', 'compliance reason', 'compliance required', 'compliance document', 'compliance request', 'compliance process', 'compliance pending', 'compliance failed', 'compliance problem', 'compliance error', 'compliance delay', 'compliance approval', 'compliance denied', 'compliance restriction', 'compliance status', 'compliance update', 'compliance review', 'compliance team', 'compliance department', 'compliance officer', 'compliance support', 'compliance verification', 'compliance validation', 'compliance check', 'compliance review', 'compliance investigation', 'compliance inquiry', 'compliance question', 'compliance response', 'compliance feedback', 'compliance escalation', 'compliance follow up', 'compliance follow-up',

            'id verification', 'identity verification', 'address verification', 'document verification', 'photo verification', 'selfie verification', 'face verification', 'passport verification', 'license verification', 'proof of address', 'proof of identity', 'proof of income', 'proof of funds', 'proof of employment', 'proof of residence', 'proof of citizenship', 'proof of relationship', 'proof of payment', 'proof of transfer', 'proof of transaction', 'proof of ownership', 'proof of registration', 'proof of insurance', 'proof of purchase', 'proof of sale', 'proof of service', 'proof of support', 'proof of use', 'proof of value', 'proof of withdrawal', 'proof of deposit', 'proof of refund', 'proof of claim', 'proof of loss', 'proof of damage', 'proof of repair', 'proof of replacement', 'proof of return', 'proof of shipment', 'proof of delivery', 'proof of receipt', 'proof of acceptance', 'proof of approval', 'proof of authorization', 'proof of cancellation', 'proof of change', 'proof of confirmation', 'proof of correction', 'proof of dispute', 'proof of error', 'proof of exception', 'proof of explanation', 'proof of extension', 'proof of information', 'proof of inquiry', 'proof of investigation', 'proof of notice', 'proof of objection', 'proof of order', 'proof of payment', 'proof of processing', 'proof of receipt', 'proof of refund', 'proof of rejection', 'proof of request', 'proof of response', 'proof of return', 'proof of service', 'proof of settlement', 'proof of status', 'proof of submission', 'proof of support', 'proof of suspension', 'proof of transfer', 'proof of update',

            # Address/geo constraints

            'address issue', 'address error', 'address problem', 'address not accepted', 'address not found', 'address not valid', 'address not verified', 'address rejected', 'address required', 'address restriction', 'address update', 'address validation', 'address verification', 'geo restriction', 'geo blocked', 'geo not supported', 'geo not available', 'geo not allowed', 'geo not permitted', 'geo not possible', 'geo not valid', 'geo restriction', 'geo validation', 'geo verification', 'location restriction', 'location blocked', 'location not supported', 'location not available', 'location not allowed', 'location not permitted', 'location not possible', 'location not valid', 'location restriction', 'location validation', 'location verification',

            # Promo/recipient issues

            'promo code', 'promotion code', 'discount code', 'offer code', 'referral code', 'bonus code', 'coupon code', 'recipient issue', 'recipient error', 'recipient not found', 'recipient not valid', 'recipient not verified', 'recipient rejected', 'recipient required', 'recipient restriction', 'recipient update', 'recipient validation', 'recipient verification', 'recipient problem', 'recipient support', 'recipient help', 'recipient question', 'recipient inquiry', 'recipient response', 'recipient feedback', 'recipient escalation', 'recipient follow up', 'recipient follow-up'

        ],

        'Fees/Expensive': [

            'fee', 'fees', 'charge', 'expensive', 'cost', 'rip off', 'overcharge',

            'dollar', 'dollars', 'send money', 'money transfer', 'transfer fee', 'high fee', 'high cost', 'expensive fee', 'expensive cost', 'service fee', 'service charge', 'hidden fee', 'hidden charge', 'evil fee', 'evil charge'

        ],

        'Pricing/Exchange': [

            'exchange rate', 'rate', 'price', 'just okay exchange', 'just okay rate', 'just okay exchange rate', 'exchange', 'pricing', 'just okay', 'okay rate', 'okay exchange', 'okay price'

        ],

        'Customer Service': [

            'customer service', 'customer support', 'call center', 'spoke to customer service', 'spoke to customer support', 'contacted customer service', 'contacted customer support', 'customer service rep', 'customer service representative', 'customer service agent', 'customer service team', 'customer service experience', 'customer service is', 'customer service was', 'customer service response', 'customer service issue', 'customer service problem', 'customer service complaint', 'customer service department', 'customer service number', 'customer service phone', 'customer service email', 'customer service chat', 'customer service call', 'customer service ticket', 'customer service request', 'customer service help', 'customer service support', 'customer service manager', 'customer service supervisor', 'customer service staff', 'customer service personnel', 'customer service desk', 'customer service office', 'customer service contact', 'customer service feedback', 'customer service review', 'customer service rating', 'customer service quality', 'customer service skills', 'customer service training', 'customer service attitude', 'customer service professionalism', 'customer service satisfaction', 'customer service excellence', 'customer service improvement', 'customer service complaint resolved', 'customer service complaint unresolved', 'customer service complaint handled', 'customer service complaint mishandled', 'customer service complaint ignored', 'customer service complaint addressed', 'customer service complaint not addressed', 'customer service complaint response', 'customer service complaint delay', 'customer service complaint escalation', 'customer service complaint follow up', 'customer service complaint follow-up', 'customer service complaint feedback', 'customer service complaint review', 'customer service complaint rating', 'customer service complaint quality', 'customer service complaint skills', 'customer service complaint training', 'customer service complaint attitude', 'customer service complaint professionalism', 'customer service complaint satisfaction', 'customer service complaint excellence', 'customer service complaint improvement'

        ],

        'App Issues': [

            # App crash keywords

            'crash', 'crashes', 'crashing', 'crashed', 'crashes on launch', 'crashes on startup', 'crashes after update', 'crashes frequently', 'crashes randomly', 'crashes every time',

            'frozen', 'freezing', 'buffering', 'stuck', 'force close', 'force closed', 'force closing', 'app closed unexpectedly', 'app closes unexpectedly', 'app keeps closing', 'app keeps crashing',

            'app stopped working', 'app stops working', 'app closes itself', 'app shuts down', 'app shut down', 'app not responding', 'app stopped', 'app closes automatically', 'app closes suddenly', 'app closes on its own',

            # App stability (login / not working) keywords

            "won't open", "won't start", "does not work", "doesn't work", "not working", 'login', 'log in', 'sign in', 'sign-in', "can't enter", "can't log", "can't sign", 'buggy', 'keyboard', 'input', 'glitch', 'glitches'

        ],

        'Performance ': [

            'slow', 'lag', 'laggy', 'loading', 'takes forever', 'long loading', 'painfully slow', 'slower than molasses', 'performance', 'takes long', 'time out', 'unresponsive', 'long time', 'loading screens', 'load time', 'load times', 'load', 'sluggish', 'delayed', 'delay', 'delays', 'slow transfer', 'transfer slow', 'transfer', 'just okay transfer'

        ],

        'UX': [

            'clunky', 'not user friendly', 'not user-friendly', 'not intuitive', 'hard to navigate', 'navigation', 'usability', 'user experience', 'ux', 'ui', 'interface', 'design', 'rate me', 'popping windows', 'windows', 'window'

        ]

    }

    #  & reliability (login / not working)

    # (speed / lag / loading time)

    # User experience & usability (navigation / UX–UI / promotions–ads)

    import re

    # Update any old column names in df_neg if present (for backward compatibility)

    if 'Payments/verification' in df_neg.columns:

        df_neg['Payments & Verification'] = df_neg['Payments/verification']

        df_neg.drop(columns=['Payments/verification'], inplace=True)

    for issue, keywords in issue_keywords.items():

        # Use word boundaries for all keywords, escape special regex chars

        patterns = []

        for kw in keywords:

            # Escape regex special characters in keyword

            safe_kw = re.escape(kw)

            # Use word boundaries for all keywords (single or multi-word)

            patterns.append(r'\b' + safe_kw + r'\b')

        pattern = '|'.join(patterns)

        df_neg[issue] = df_neg['review'].str.lower().str.contains(pattern, case=False, na=False, regex=True).astype(int)

 

    # Churn signals

    churn_keywords = {

        'moneygram': ['moneygram','money gram'],

        'xoom': ['xoom'],

        'paypal': ['paypal', 'pay pal'],

        'remitly': ['remitly']

    }

    churn_data = {}

    for service, keywords in churn_keywords.items():

        # Ensure .sum() returns int, not index

        count = int(df_neg['review'].str.lower().str.contains('|'.join(keywords), case=False, na=False).sum())

        churn_data[service] = count

 

    return df_neg, {'issues': issue_keywords, 'churn': churn_data}

 

 

def get_priority_score(df_neg, issue_name):

    """Calculate priority: frequency × severity (safe with zero counts)"""

    freq = int(df_neg.get(issue_name, pd.Series(dtype=int)).sum())

    if freq == 0:

        return 0.0

    severity_reviews = df_neg[df_neg[issue_name] == 1]

    # handle missing sentiment_score carefully

    mean_sent = severity_reviews.get('sentiment_score', pd.Series([0.3])).mean()

    if pd.isna(mean_sent):

        mean_sent = 0.3

    severity = 1 - mean_sent

    return float(freq * severity)

 

@st.cache_data(show_spinner=False)

def compute_top_issues(df_neg, issues_dict, total_complaints):

    """Return sorted top issues dataframe (cached to avoid repeated recompute & flicker)."""

    if df_neg is None or df_neg.empty or not issues_dict or total_complaints == 0:

        return pd.DataFrame(columns=['Issue','Count','% of Total','Priority Score'])

    issues = list(issues_dict.keys())

    counts = [int(df_neg[issue].sum()) for issue in issues]

    priorities = [get_priority_score(df_neg, issue) for issue in issues]

    pct = [f"{(c/total_complaints*100):.1f}%" for c in counts]

    df = pd.DataFrame({

        'Issue': issues,

        'Count': counts,

        '% of Total': pct,

        'Priority Score': priorities

    })

    return df.sort_values('Priority Score', ascending=False).reset_index(drop=True)

 

 

def show_customer_insights(all_df):

    """

    Analyze full historical dataset vs last 1 year and produce actionable charts/tables.

    Call with finaldf (all-time reviews).

    """

    if all_df.empty:

        show_timed_warning_generic("⚠️ No records available for Customer Insights", duration=4)

        return

 

    # Defensive copy & types

    all_df = all_df.copy()

    all_df["TimeStamp"] = pd.to_datetime(all_df.get("TimeStamp", pd.Series()), errors="coerce")

    all_df["rating"] = pd.to_numeric(all_df.get("rating", pd.Series(dtype=float)), errors="coerce")

 

    # Ensure a CountryName column exists (prefer country_map lookup from ISO2 -> full name)

    if "CountryName" not in all_df.columns:

        if "Country" in all_df.columns:

            # map lower-case ISO2 codes to friendly names when possible, else keep original

            try:

                all_df["CountryName"] = all_df["Country"].astype(str).str.lower().map(country_map).fillna(all_df["Country"].astype(str))

            except Exception:

                all_df["CountryName"] = all_df["Country"].astype(str)

        else:

            all_df["CountryName"] = "Unknown"

   

    cutoff = pd.Timestamp.today() - pd.Timedelta(days=365)

    last_year = all_df[all_df["TimeStamp"] >= cutoff]

    historical = all_df[all_df["TimeStamp"] < cutoff]

 

    st.markdown("## Customer Insights — Historical vs Last 12 months")

    st.markdown(f"**Data range:** {all_df['TimeStamp'].min().date() if not all_df['TimeStamp'].isna().all() else 'N/A'} → {all_df['TimeStamp'].max().date()}")

    st.markdown("---")

 

    # Summary KPIs

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric("Total reviews (all-time)", f"{len(all_df):,}")

    with col2:

        st.metric("Reviews last 12 months", f"{len(last_year):,}", delta=f"{len(last_year) - len(historical):,}")

    with col3:

        avg_all = all_df['rating'].dropna().astype(float).mean()

        st.metric("Avg rating (all-time)", f"{avg_all:.2f}" if not np.isnan(avg_all) else "N/A")

    with col4:

        avg_last = last_year['rating'].dropna().astype(float).mean()

        avg_hist = historical['rating'].dropna().astype(float).mean()

        delta = (avg_last - avg_hist) if not (np.isnan(avg_last) or np.isnan(avg_hist)) else np.nan

        st.metric("Avg rating (last 12m)", f"{avg_last:.2f}" if not np.isnan(avg_last) else "N/A", delta=f"{delta:+.2f}" if not np.isnan(delta) else "")

 

 

    st.markdown("---")

 

    # Country-level improvement/regression: compare avg rating per country (last year vs earlier)

    try:

        agg_last = last_year.groupby("CountryName")["rating"].mean().rename("avg_last").reset_index()

        agg_hist = historical.groupby("CountryName")["rating"].mean().rename("avg_hist").reset_index()

        cmp = pd.merge(agg_hist, agg_last, on="CountryName", how="outer")

        cmp["avg_hist"] = cmp["avg_hist"].fillna(np.nan)

        cmp["avg_last"] = cmp["avg_last"].fillna(np.nan)

        cmp["delta"] = cmp["avg_last"] - cmp["avg_hist"]

        cmp["ISO3"] = cmp["CountryName"].apply(name_to_iso3)

 

        # show top regressions and improvements

        top_regress = cmp.sort_values("delta").head(10).loc[:, ["CountryName", "avg_hist", "avg_last", "delta"]]

        top_improve = cmp.sort_values("delta", ascending=False).head(10).loc[:, ["CountryName", "avg_hist", "avg_last", "delta"]]

 

        c1, c2 = st.columns(2)

        with c1:

            st.subheader("Top regressions (Avg rating ↓)")

            st.dataframe(top_regress.style.format({"avg_hist":"{:.2f}", "avg_last":"{:.2f}", "delta":"{:+.2f}"}))

        with c2:

            st.subheader("Top improvements (Avg rating ↑)")

            st.dataframe(top_improve.style.format({"avg_hist":"{:.2f}", "avg_last":"{:.2f}", "delta":"{:+.2f}"}))

 

    except Exception as e:

        st.warning(f"Country comparison failed: {e}")

 

    st.markdown("---")

 

    # Issue detection: simple keyword buckets

    buckets = {

        "Crashes / freezes": ["crash", "crashes", "crashing", "freeze", "freezes", "stuck", "shut down", "keeps closing", "keeps stopping"],

        "OTP / verification": ["otp", "verification", "code", "sms", "message not received", "can't receive code", "verification failed"],

        "Login / auth": ["log in", "login", "sign in", "can't sign", "can't log", "password", "c2016", "c9999", "authentication"],

        "Performance / slow": ["slow", "lag", "loading", "takes long", "time out", "unresponsive"],

        "Pricing / fees / exchange": ["fee", "fees", "exchange rate", "rate", "price", "expensive"],

        "UI / UX": ["ui", "user friendly", "confusing", "not intuitive", "navigation"]

    }

 

    # Count bucket mentions (all-time and last-year)

    issue_counts = []

    for name, keys in buckets.items():

        pattern = "(" + "|".join([kw.replace(" ", r"\s+") for kw in keys]) + ")"

        mask_all = all_df['review'].fillna("").str.lower().str.contains(pattern, regex=True)

        mask_last = last_year['review'].fillna("").str.lower().str.contains(pattern, regex=True)

        issue_counts.append({"issue": name, "all_time": int(mask_all.sum()), "last_year": int(mask_last.sum())})

 

    issues_df = pd.DataFrame(issue_counts).sort_values("all_time", ascending=False)

 

    st.dataframe(issues_df.set_index("issue"))

 

    #st.markdown("---")

 

    # AppVersion impact on rating (last year vs historical)

    try:

        v_last = last_year.groupby("appVersion")["rating"].agg(["mean","count"]).reset_index().rename(columns={"mean":"avg_last","count":"count_last"})

        v_hist = historical.groupby("appVersion")["rating"].agg(["mean","count"]).reset_index().rename(columns={"mean":"avg_hist","count":"count_hist"})

        vcmp = pd.merge(v_hist, v_last, on="appVersion", how="outer").fillna(0)

        vcmp["delta"] = vcmp["avg_last"] - vcmp["avg_hist"]

        vcmp = vcmp.sort_values("count_last", ascending=False).head(20)

        #st.subheader("Top App Versions (impact on rating)")

        fig_ver = px.bar(vcmp, x="appVersion", y=["avg_hist","avg_last"], barmode="group", title="Avg rating by appVersion (hist vs last 12m)")

        #st.plotly_chart(fig_ver, use_container_width=True)

        #st.dataframe(vcmp.loc[:, ["appVersion","avg_hist","avg_last","count_hist","count_last","delta"]].style.format({"avg_hist":"{:.2f}","avg_last":"{:.2f}","delta":"{:+.2f}"}))

    except Exception as e:

        st.warning(f"AppVersion analysis failed: {e}")

 

    st.markdown("---")

 

    # Actionable recommendations box (derived from above signals)

    st.subheader("Actionable recommendations")

    recs = []

    # pricing

    if issues_df.loc[issues_df['issue'].str.contains("Pricing", case=False), "last_year"].sum() > 0:

        recs.append("- Review pricing / exchange rate policy for countries where pricing mentions are high.")

    # crashes

    if issues_df.loc[issues_df['issue'].str.contains("Crashes", case=False), "last_year"].sum() > 0:

        recs.append("- Prioritise stability fixes (crash/freeze) for top affected appVersion/countries.")

    # otp

    if issues_df.loc[issues_df['issue'].str.contains("OTP", case=False), "last_year"].sum() > 0:

        recs.append("- Investigate OTP delivery for mobile operators in countries with repeated OTP failures.")

    # version regressions

    if not vcmp.empty and (vcmp["delta"] < -0.1).any():

        recs.append("- Roll back or hotfix app versions with significant rating drops (delta <= -0.1).")

    if len(recs) == 0:

        recs.append("- No strong automated signals found. Consider deeper manual review or upload competitor data for benchmarking.")

 

    for r in recs:

        st.markdown(r)

 

    # st.markdown("---")

    # st.info("Upload competitor review datasets (CSV with columns: TimeStamp, rating, review, CountryName) to compare trends across providers — feature placeholder.")

 

 

def show_complaint_analytics(filtered_df, date1, date2):

    """🚨 COMPLAINT ANALYTICS - Works with existing filtered_df"""

    st.markdown("""

    <style>

    .complaint-header {color: #d32f2f; font-size: 2.5em; font-weight: bold; text-align: center;}

    .metric-card {background: linear-gradient(135deg, #ff6b6b, #ee5a52); padding: 1rem; border-radius: 10px; margin: 0.5rem;}

    </style>

    """, unsafe_allow_html=True)

   

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown('<div class="complaint-header">🚨 Complaints Dashboard</div>', unsafe_allow_html=True)

   

    st.markdown("<br><br>", unsafe_allow_html=True)

  

    # Process complaints from existing filtered_df

    df_neg, analysis_data = process_complaints(filtered_df)

   

    if df_neg.empty:

        st.warning("⚠️ No negative reviews found in selected filters")

        st.info("💡 Try: Lower rating filter (1-2⭐), longer date range, or different countries")

        return

   

    total_complaints = len(df_neg)

    st.metric("📊 Total Complaints Analyzed", total_complaints)

   

    # # === 1. TIME-SERIES TRENDS ===

    # col1, col2 = st.columns(2)

    # with col1:

    #     st.markdown("### 📈 Monthly Complaint Trends")

    #     # require at least 2 months to show trend

    #     if 'month_year' in df_neg.columns and df_neg['month_year'].notna().sum() > 1:

    #         # group by month (using timestamp in month_year)

    #         issues = list(analysis_data['issues'].keys())

    #         trends = (

    #             df_neg

    #             .set_index('month_year')

    #             .groupby(pd.Grouper(freq='M'))[issues]

    #             .sum()

    #             .reset_index()

    #         )

    #         if not trends.empty:

    #             # convert to readable month label and ensure chronological order

    #             trends['month_label'] = pd.to_datetime(trends['month_year']).dt.strftime('%Y-%m')

    #             trends = trends.sort_values('month_year')

    #             fig_line = px.line(

    #                 trends,

    #                 x='month_label',

    #                 y=issues,

    #                 title="Complaint Volume Over Time",

    #                 color_discrete_sequence=px.colors.sequential.Oranges

    #             )

    #             fig_line.update_xaxes(tickangle=45)

    #             st.plotly_chart(fig_line, use_container_width=True, height=400)

    #         else:

    #             st.info("ℹ️ Not enough monthly aggregated data to show trends")

    #     else:

    #         st.info("ℹ️ Need more time range for trends (select at least 2 months)")

 

    # with col2:

    #     st.markdown("### 🎯 Issue Priority Matrix")

    #     issues = list(analysis_data.get('issues', {}).keys())

    #     issue_stats = pd.DataFrame({

    #         'issue': issues,

    #         'frequency': [int(df_neg[issue].sum()) for issue in issues],

    #         'priority': [get_priority_score(df_neg, issue) for issue in issues]

    #     })

    #     # avoid divide by zero and produce sensible severity

    #     issue_stats['severity'] = issue_stats.apply(

    #         lambda r: (r['priority'] / r['frequency']) if r['frequency'] > 0 else 0.0,

    #         axis=1

    #     )

 

    #     # if no frequencies, show info

    #     if issue_stats['frequency'].sum() == 0:

    #         st.info("No detected issue mentions in the selected filters/date range.")

    #     else:

    #         fig_bubble = px.scatter(

    #             issue_stats,

    #             x='frequency',

    #             y='severity',

    #             size='priority',

    #             hover_name='issue',

    #             size_max=50,

    #             color='priority',

    #             color_continuous_scale='Reds',

    #             title="Priority: Size = Frequency × Severity"

    #         )

    #         st.plotly_chart(fig_bubble, use_container_width=True, height=400)

 

    # === 3. HEATMAP ===

   

    st.markdown("### 🌍 Country × App Version Heatmap")

 

    if 'App_Version' in df_neg.columns and 'CountryName' in df_neg.columns:

        heatmap_data = (

            df_neg.groupby(['CountryName', 'App_Version'])

            .size()

            .reset_index(name='complaints')

        )

 

        fig_heatmap = px.density_heatmap(

            heatmap_data,

            x='App_Version',

            y='CountryName',

            z='complaints',

            title="Complaint Density",

            color_continuous_scale='Reds'

        )

 

        # ✅ Center the title

        fig_heatmap.update_layout(

            title_x=0.45,  # 0.5 = center

            height=500

        )

       

        fig_heatmap.update_layout(

            title_font=dict(size=22, family='Arial', color='black')

       )

 

        st.plotly_chart(fig_heatmap, use_container_width=True)        

    

    

    st.markdown("### 📋 Top Issues Ranked")

 

    top_issues = compute_top_issues(df_neg, analysis_data.get('issues', {}), total_complaints)

 

    if top_issues.empty:

        st.info("No top issues detected for the selected filters/date range.")

    else:

        display_df = top_issues.copy()

 

        # 1) Round Priority Score (e.g., 32.4000 → 32.4)

        display_df['Priority Score'] = (

            display_df['Priority Score']

            .astype(float)

            .round(1)

        )

 

        # 2) Add S.No. as the first column, starting from 1

 

        # 3) Build centered, full-width HTML table

        table_html = display_df.to_html(index=False, classes='top-issues-table')

 

        st.markdown(

            """

            <style>

            /* Center the table block and make it full width */

            .top-issues-wrapper {

                display: flex;

                justify-content: center;   /* center the block */

                width: 100%;

            }

            .top-issues-table {

                width: 100%;               /* expand to fill width */

                border-collapse: collapse;

                font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;

            }

            .top-issues-table th, .top-issues-table td {

                padding: 0.5rem 0.75rem;

                border-bottom: 1px solid rgba(0,0,0,0.08);

                text-align: left;          /* keep text left-aligned for readability */

            }

            .top-issues-table th {

                font-weight: 600;

                background: rgba(0,0,0,0.02);

            }

            </style>

            """,

            unsafe_allow_html=True

        )

 

        st.markdown(

            f"""

            <div class="top-issues-wrapper">

                {table_html}

            </div>

            """,

            unsafe_allow_html=True

        )

 

    st.markdown("<br><br>", unsafe_allow_html=True)

   

    # === 5. CHURN SIGNALS ===

    if analysis_data.get('churn'):

        st.markdown("### ⚠️ Competitor Mentions")

        churn_signals = analysis_data['churn']

        # Accept dict, Series, or DataFrame

        if isinstance(churn_signals, dict):

            churn_df = pd.DataFrame(list(churn_signals.items()), columns=["Competitor", "Mentions"])

        elif isinstance(churn_signals, pd.Series):

            churn_df = churn_signals.reset_index()

            churn_df.columns = ["Competitor", "Mentions"]

        elif isinstance(churn_signals, pd.DataFrame):

            churn_df = churn_signals.copy()

            if churn_df.shape[1] == 1:

                churn_df = churn_df.reset_index()

                churn_df.columns = ["Competitor", "Mentions"]

            elif churn_df.shape[1] == 2:

                churn_df.columns = ["Competitor", "Mentions"]

        else:

            churn_df = pd.DataFrame(columns=["Competitor", "Mentions"])

 

        churn_df["Mentions"] = pd.to_numeric(churn_df["Mentions"], errors="coerce").fillna(0).astype(int)

        churn_df["Competitor"] = churn_df["Competitor"].astype(str)

        churn_df = churn_df.sort_values("Mentions", ascending=False)

        churn_df_display = churn_df.copy()

        churn_df_display.index = range(1, len(churn_df_display) + 1)

        churn_df_display.index.name = "S.No."

        st.dataframe(churn_df_display)

        st.markdown("<br><b>Use the dropdown below to select a competitor</b>", unsafe_allow_html=True)

        selected_competitor = st.selectbox(

            "Select a competitor:",

            churn_df["Competitor"].tolist(),

            index=0 if not churn_df.empty else None

        )

        if selected_competitor:

            # Find reviews mentioning the selected competitor

            competitor_keywords = {

                'moneygram': ['moneygram','money gram'],

                'xoom': ['xoom'],

                'paypal': ['paypal', 'pay pal'],

                'remitly': ['remitly']

            }

            keywords = competitor_keywords.get(selected_competitor.lower(), [selected_competitor.lower()])

            mask = filtered_df['review'].str.lower().fillna("").apply(lambda x: any(kw in x for kw in keywords))

            competitor_reviews = filtered_df[mask]

            st.markdown(f"**Reviews mentioning {selected_competitor}:**")

            if not competitor_reviews.empty:

                display_cols = [col for col in competitor_reviews.columns if 'review' in col.lower() or 'TimeStamp' in col or 'DateTimeStamp' in col or 'rating' in col]

                competitor_reviews_display = competitor_reviews[display_cols].reset_index(drop=True)

                competitor_reviews_display.index = range(1, len(competitor_reviews_display) + 1)

                competitor_reviews_display.index.name = "S.No."

                st.dataframe(competitor_reviews_display)

            else:

                st.info(f"No reviews found mentioning {selected_competitor}.")

 

   

    

 

    # === 6. INTERACTIVE REVIEW EXPLORER ===

   

    st.markdown("### 🔍 Explore Raw Complaints")

    issues_list = top_issues['Issue'].tolist() if not top_issues.empty else []

 

    # --- Centered dropdown ---

    center_cols = st.columns([1, 2, 1])

    with center_cols[1]:

        selected_issue = st.selectbox(

            "Filter by Issue:",

            options=issues_list,

            index=0 if issues_list else None

        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Table directly below the dropdown ---

    if not issues_list:

        st.info("No issue categories to explore.")

    elif selected_issue is None:

        st.info("Please select an issue to explore complaints.")

    else:

 

        # Filter reviews for the selected issue

        issue_reviews = df_neg[df_neg[selected_issue] == 1][

            ['DateTimeStamp', 'review', 'CountryName', 'rating', 'sentiment_label']

        ].copy()

 

        if issue_reviews.empty:

            st.info("No reviews for the selected issue.")

        else:

            # Rename columns for display

            issue_reviews = issue_reviews.rename(columns={

                'DateTimeStamp': 'Date',

                'review': 'Complaint',

                'CountryName': 'Country',

                'rating': 'Rating',

                'sentiment_label': 'Sentiment'

            })

 

            # Remove time from Date column

            issue_reviews['Date'] = pd.to_datetime(issue_reviews['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

 

            # Reset and add S.No. (1-based)

            issue_reviews = issue_reviews.reset_index(drop=True)

            issue_reviews.index = issue_reviews.index + 1

            issue_reviews.index.name = "S.No."

 

            # --- Pagination ---

            page_size = 5

            total_rows = len(issue_reviews)

            total_pages = (total_rows + page_size - 1) // page_size

            total_pages = max(total_pages, 1)  # ensure at least 1

 

            # Unique page key per selected issue to avoid collisions across issues

            page_key = f"explorer_page_{selected_issue}"

 

            # Reset page to 1 if the selected issue changed since last render

            if st.session_state.get("last_selected_issue") != selected_issue:

                st.session_state[page_key] = 1

                st.session_state["last_selected_issue"] = selected_issue

 

            # Initialize page if not present

            if page_key not in st.session_state:

                st.session_state[page_key] = 1

 

            # Clamp to valid range

            page = int(st.session_state[page_key])

            page = max(1, min(page, total_pages))

 

            # Compute slice

            start = (page - 1) * page_size

            end = min(start + page_size, total_rows)

 

            # --- Display table (remove index column) ---

            page_slice = issue_reviews.iloc[start:end].copy()

            table_html = page_slice.to_html(index=False)

            st.markdown(table_html, unsafe_allow_html=True)

 

            st.caption(

                f"Showing {min(start+1, total_rows)}–{end} of {total_rows} rows — Page {page}/{total_pages}"

            )

 

            # --- Centered pagination control: render slider ONLY if we have >1 page ---

            cols_nav = st.columns([1, 2, 1])

            with cols_nav[1]:

                if total_pages > 1:

                    st.slider(

                        "Page",

                        min_value=1,

                        max_value=total_pages,

                        key=page_key

                    )

                else:

                    st.markdown(

                        "<div style='text-align:center; font-size:0.9rem;'>Only one page</div>",

                        unsafe_allow_html=True

                    )

 

            # --- Centered download button (no index in CSV) ---

            cols_dl = st.columns([1, 2, 1])

            with cols_dl[1]:

                csv_all = issue_reviews.to_csv(index=False).encode('utf-8')

                st.download_button(

                    label=f"Download All Results ({selected_issue})",

                    data=csv_all,

                    file_name=f"complaints_{selected_issue}_all.csv",

                    mime='text/csv'

                )

 

 

   

    # === SUMMARY CARDS ===

    st.markdown("---")

    if top_issues.empty:

        st.info("Summary metrics not available (no detected issues).")

    else:

        col1, col2, col3, col4 = st.columns(4)

        top0 = top_issues.iloc[0]

 

        # Safely compute top country and its complaint count

        if 'CountryName' in df_neg.columns and not df_neg.empty:

            country_series = (

                df_neg['CountryName']

                .astype(str).str.strip()

                .replace({'': 'Unknown'}).fillna('Unknown')

            )

            vc = country_series.value_counts()

            top_country = vc.index[0] if len(vc) > 0 else "N/A"

            top_country_count = int(vc.iloc[0]) if len(vc) > 0 else 0

        else:

            top_country, top_country_count = "N/A", 0

 

        # Average severity (sentiment score)

        avg_sev_series = df_neg.get("sentiment_score", pd.Series(dtype=float))

        avg_sev = float(avg_sev_series.mean()) if not avg_sev_series.empty else 0.0

 

        with col1:

            st.markdown(f"""

            <div class="metric-card" style="background: linear-gradient(135deg, #66bb6a, #388e3c);">

                <h3 style='color:white;'>Top Issue</h3>

                <h2 style='color:white;'>{top0['Issue']}</h2>

                <p style='color:white;'>{int(top0['Count'])} cases</p>

            </div>

            """, unsafe_allow_html=True)

 

       

        with col2:

            st.markdown(f"""

            <div class="metric-card" style="background: linear-gradient(135deg, #ffa726, #fb8c00);">

                <h3 style='color:white;'>Highest Priority</h3>

                <h2 style='color:white;'>{top0['Issue']}</h2>

                <p style='color:white;'>Score: {float(top0['Priority Score']):.1f}</p>

                <p style='color:white; font-size:0.85em;'>(Most severe & frequent)</p>

            </div>

            """, unsafe_allow_html=True)

 

 

        with col3:

            st.markdown(f"""

            <div class="metric-card" style="background: linear-gradient(135deg, #42a5f5, #1976d2);">

                <h3 style='color:white;'>Most Complaints</h3>

                <h2 style='color:white;'>{top_country}</h2>

                <p style='color:white;'>{top_country_count} complaints</p>

            </div>

            """, unsafe_allow_html=True)

 

       

        with col4:

            # Interpret severity

            severity_label = (

                "Very Negative" if avg_sev < 0.3 else

                "Negative" if avg_sev < 0.5 else

                "Moderate"

            )

            st.markdown(f"""

            <div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b, #ee5a52);">

                <h3 style='color:white;'>Avg Severity</h3>

                <h2 style='color:white;'>{avg_sev:.2f}</h2>

                <p style='color:white;'>Overall sentiment: {severity_label}</p>

            </div>

            """, unsafe_allow_html=True)

 

    st.markdown("<br>", unsafe_allow_html=True)

 

# Fix: Remove or guard display_df usage if not defined

# (The following block is now guarded to avoid NameError)

    # if 'display_df' in locals():

    #     display_df['Priority Score'] = display_df['Priority Score'].astype(float).round(1)

# Define links

 

# Add Competitor Comparison to links and menu

links = {

    "Global Ranking": "worldmap",

    "Interactive Sunburst": "sunburst",

    "Word Cloud": "wordcloud",

    "Visual Charts": "visualcharts",

    "Interactive TreeMap": "treemap",

    "Keyword Analysis": "keyword",

    "Topic Modeling": "topic",

    "Competitor Pulse": "competitor_comparison",  # Restored

    "Analytics": "competitor_insights",

    "LanguageTranslation": "translation",

    # "Complaint Analytics": "insights",  # Hidden for now

}

 

# Menu options: charts + Reset at the end

menu_options = list(links.keys()) + ["---","🔄 RESET"]

menu_icons = [

    "globe", "sun", "cloud", "bar-chart", "tree",

    "search", "book", "trophy", "graph-up-arrow",

    "translate", "dash", "arrow-repeat"

]

 

# Sidebar menu with bold links and no background on selection

with st.sidebar:

    st.markdown("""
<style>
section[data-testid="stSidebar"] a.nav-link,
section[data-testid="stSidebar"] .nav-link {
    white-space: nowrap !important;
    display: flex !important;
    flex-wrap: nowrap !important;
    flex-direction: row !important;
    align-items: center !important;
    padding: 5px 8px !important;
    margin: 1px 0 !important;
    font-size: 12px !important;
    font-weight: bold !important;
}
section[data-testid="stSidebar"] a.nav-link span,
section[data-testid="stSidebar"] .nav-link span {
    white-space: nowrap !important;
    overflow: hidden !important;
}
section[data-testid="stSidebar"] a.nav-link i,
section[data-testid="stSidebar"] .nav-link .icon {
    flex-shrink: 0 !important;
    margin-right: 6px !important;
}
</style>
""", unsafe_allow_html=True)

    selected = option_menu(

        menu_title="",  # No title

        options=menu_options,

        icons=menu_icons,

        menu_icon="cast",

        default_index=len(menu_options) - 1,  # Default selection is "🔄 Reset"

        styles={

            "container": {

                "padding": "2px",

                "background-color": "#f8f9fa"

            },

            "icon": {

                "color": "#007BFF",

                "font-size": "14px"

            },

            "nav-link": {

                "font-size": "12px",

                "text-align": "left",

                "margin": "1px 0",

                "font-weight": "bold",  # Bold for all items

           

            },

            "nav-link-selected": {

                "color": "#000000",       # Black text

                "font-weight": "bold",     # Bold for selected

                "background-color": "#ffdd00",

            },

        }

    )

 

# Handle selection

if selected == "🔄 Reset":

    st.session_state.selected_chart = None

else:

    st.session_state.selected_chart = links.get(selected)

   

 

# Main container

main_container = st.container()

with main_container:

    if st.session_state.get("selected_chart") is None:

        st.markdown("<br>", unsafe_allow_html=True)

        search_query = st.text_input("**Search Reviews :**")

        st.markdown("<br>", unsafe_allow_html=True)

 

        # Filter logic

        if search_query:

            with st.spinner("🔍 Fetching reviews..."):

                filtered_df = filtered_df[filtered_df['review'].str.contains(search_query, case=False, na=False)]

                placeholder = st.empty()

                progress_bar = st.progress(0)

                for i in range(100):

                    time.sleep(0.01)

                    progress_bar.progress(i + 1)

                    placeholder.text(f"Fetching reviews... {i+1}%")

                placeholder.empty()

                progress_bar.empty()

 

 

        if not filtered_df.empty:

            # Format timestamp

            filtered_df["DateTimeStamp"] = pd.to_datetime(filtered_df["DateTimeStamp"])

            filtered_df["DateTimeStamp"] = filtered_df["DateTimeStamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

 

            # Pagination

            page_size = 100

            total_pages = max((len(filtered_df) - 1) // page_size + 1, 1)

            page_num = st.session_state.get("page_slider", 1)

            page_num = max(1, min(page_num, total_pages))

 

            start_idx = (page_num - 1) * page_size

            end_idx = min(start_idx + page_size, len(filtered_df))

            column_config = {

                "TimeStamp": st.column_config.DatetimeColumn(

                    label="📅 Date",

                    width="small",

                    format="YYYY-MM-DD"

                ),

                "review": st.column_config.TextColumn(

                    label="💬 Review",

                    width="large"

                ),

                "rating": st.column_config.NumberColumn(

                    label="⭐ Rating",

                    width="small",

                    format="%.0f ⭐"

                ),

                "sentiment_score": st.column_config.ProgressColumn(

                    label="📊 Score",

                    width="small",

                    format="%.2f",

                    min_value=0,

                    max_value=1,

                ),

                "sentiment_label": st.column_config.TextColumn(

                    label="😊 Sentiment",

                    width="small"

                ),

                "HappinessIndex": st.column_config.TextColumn(

                    label="Happiness Index",

                    width="small"

                ),

                "CountryName": st.column_config.TextColumn(

                    label="🌍 Country",

                    width="small"

                ),

                "AppName": st.column_config.TextColumn(

                    label="📱 App",

                    width="small"

                ),

                "appVersion": st.column_config.TextColumn(

                    label="🔢 Version",

                    width="small"

                ),

                "UserName": st.column_config.TextColumn(

                    label="👤 User",

                    width="small"

                )

            }

 

            st.dataframe(

                filtered_df.iloc[start_idx:end_idx],

                column_config=column_config,

                height=275,

                use_container_width=True

            )

            st.caption("ℹ️ Sentiment Scores are between 0 to 1.")

            if total_pages > 1:

                col_left, col_center, col_right = st.columns([1, 2, 1])

                with col_center:

                    st.markdown("<div style='text-align: center; font-size: 14px;'>Navigate Pages</div>", unsafe_allow_html=True)

                    page_num = st.slider(

                        label="",

                        min_value=1,

                        max_value=total_pages,

                        value=page_num,

                        key="page_slider"

                    )

            else:

                st.markdown("<div style='text-align: center; font-size: 8px;'>Only one page of results</div>", unsafe_allow_html=True)

 

            st.markdown("<br>", unsafe_allow_html=True)

            st.caption(f"Showing page {page_num} of {total_pages} — rows {start_idx + 1} to {end_idx}")

            st.success(f"✅ Displaying {len(filtered_df)} reviews.")

            st.markdown("<br>", unsafe_allow_html=True)

 

            # Download button

            csv = filtered_df.to_csv(index=False).encode('utf-8-sig')

            col_l, col_c, col_r = st.columns([1, 1, 1])
            with col_c:
                st.download_button('Download Data', data=csv, file_name="Data.csv", mime="text/csv")

 

    else:

        # Render selected chart

        #st.write(f"✅ You selected: {st.session_state.selected_chart}")

        if st.session_state.selected_chart == "worldmap":

            show_world_map(filtered_df, date1, date2)

        elif st.session_state.selected_chart == "sunburst":

            show_sunburst_chart(filtered_df, date1, date2)

        elif st.session_state.selected_chart == "wordcloud":

            show_word_cloud(filtered_df)

        elif st.session_state.selected_chart == "visualcharts":

            show_visual_charts(filtered_df, df, date1, date2)

        elif st.session_state.selected_chart == "treemap":

            show_treemap_chart(filtered_df, date1, date2)

        elif st.session_state.selected_chart == "keyword":

            show_keyword_analysis(filtered_df, stop_words)

        elif st.session_state.selected_chart == "insights":

            show_complaint_analytics(filtered_df, date1, date2)

            show_customer_insights(finaldf)

        elif st.session_state.selected_chart == "topic":

            show_topic_modeling(filtered_df)

        elif st.session_state.selected_chart == "competitor_insights":

 

            # --- Analytics: Country dropdown, Android/iOS table ---

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("<div style='text-align:center'><h4>Select a Country to view ratings</h4></div>", unsafe_allow_html=True)

            all_countries = sorted(set(country_map.values()))

            default_country = "United States"

            default_index = all_countries.index(default_country) if default_country in all_countries else 0

            selected_country = st.selectbox("Select a Country", all_countries, key="ci_country", index=default_index, label_visibility="collapsed")

            if 'ci_country' in st.session_state:

                if st.session_state.get('Country Selection') or st.session_state.get('Select the App Type'):

                    st.session_state.pop("Country Selection", None)

                    st.session_state.pop("Select the App Type", None)

                    st.experimental_rerun()

 

            selected_country_code = None

            for code, name in country_map.items():

                if name == selected_country:

                    selected_country_code = code

                    break

            ci_df = finaldf.copy()

            if selected_country_code:

                ci_df = ci_df[ci_df["Country"].str.lower() == selected_country_code.lower()]

            ci_df = ci_df[(ci_df["TimeStamp"] >= date1_dt) & (ci_df["TimeStamp"] <= date2_dt)]

 

            ci_df["Platform"] = ci_df["AppName"].apply(_derive_platform)

            ci_df["rating"] = pd.to_numeric(ci_df["rating"], errors='coerce')

            avg_ratings = ci_df.groupby("Platform")["rating"].mean().reset_index()

            avg_ratings["rating"] = avg_ratings["rating"].round(2)

            avg_ratings = avg_ratings.rename(columns={"Platform": "Platform", "rating": "Average Rating"})

 

            st.markdown(f"<h3 style='text-align:center;'>Average Ratings for {selected_country}</h3>", unsafe_allow_html=True)

            platform_icons = {

                "Android": "<span style='font-size:1.5em;'>🤖</span>",

                "iOS": "<span style='font-size:1.5em;'>🍏</span>"

            }

            def rating_color(val):

                if val >= 4.0:

                    return "#4CAF50"

                elif val >= 3.0:

                    return "#FFC107"

                else:

                    return "#F44336"

 

            table_html = "<style>.rating-table{width:60%;margin-left:auto;margin-right:auto;border-collapse:separate;border-spacing:0 10px;}.rating-table th,.rating-table td{padding:12px 18px;text-align:center;font-size:1.1em;}.rating-table th{background:#f5f5f5;color:#333;border-radius:8px 8px 0 0;}.rating-table tr{background:#fff;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.04);}.rating-table td.icon{font-size:1.7em;}</style>"
            table_html += "<table class='rating-table'><tr><th>Platform</th><th>Average Rating</th></tr>"

            for _, row in avg_ratings.iterrows():

                plat = row['Platform']

                icon = platform_icons.get(plat, '')

                rating = row['Average Rating']

                color = rating_color(rating)

                table_html += f"<tr><td class='icon'>{icon} <b>{plat}</b></td><td style='color:{color}; font-weight:bold; font-size:1.3em;'>{rating:.2f}</td></tr>"

            table_html += "</table>"

            num_platforms = len(avg_ratings)
            table_height = 100 + (num_platforms * 90)
            import streamlit.components.v1 as components
            components.html(table_html, height=table_height, scrolling=False)

 

            # Platform selection for issue summary

            platform_options = [row["Platform"] for _, row in avg_ratings.iterrows()]

            if platform_options:

                st.markdown("<br>", unsafe_allow_html=True)

 

                st.markdown("""
                    <style>
                    div[data-testid="stRadio"] {
                        display: flex !important;
                        flex-direction: column !important;
                        align-items: center !important;
                        width: 100% !important;
                    }
                    div[data-testid="stRadio"] > label p {
                        text-align: center !important;
                        width: 100% !important;
                        font-weight: bold;
                    }
                    div[data-testid="stRadio"] > div[role="radiogroup"] {
                        display: flex !important;
                        flex-direction: row !important;
                        justify-content: center !important;
                        width: 100% !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
                st.markdown("<div style='text-align:center'><h4>Select Platform to view Customer Issues</h4></div>", unsafe_allow_html=True)
                col_l, col_c, col_r = st.columns([1,2,1])
                with col_c:
                    selected_platform = st.radio("Select Platform to view Customer Issues", platform_options, horizontal=True, key="ci_platform", label_visibility="collapsed")

                platform_df = ci_df[ci_df["Platform"] == selected_platform].copy()

                if 'sentiment_label' not in platform_df.columns:

                    platform_df['sentiment_label'] = platform_df.apply(get_sentiment_label, axis=1)

                df_neg, analysis_data = process_complaints(platform_df)

                total_complaints = len(df_neg)

                if total_complaints == 0:

                    st.info("No complaints found for this selection.")

                else:

                    issues_dict = analysis_data.get('issues', {})

                    if not issues_dict:

                        st.info("No issue categories detected for this selection.")

                    else:

                        issue_names = list(issues_dict.keys())

                        merged_name = 'App Issues'

                        merged_keys = [k for k in issue_names if k.strip().lower() in ['app issues', 'performance', 'performance / slow', 'performance ']]

                        other_keys = [k for k in issue_names if k not in merged_keys]

                        funnel_issue_names = [merged_name] + other_keys if merged_keys else issue_names

 

                        # --- Issue Type Count Table by App Version (moved above funnel chart) ---

                        app_df = platform_df.copy()

                        app_df, _ = process_complaints(app_df)

                        issue_version_cols = funnel_issue_names + ["Miscellaneous"]

                        table_df = app_df.copy()

                        if 'App_Version' not in table_df.columns:

                            table_df['App_Version'] = table_df.get('appVersion', 'Unknown').astype(str).str.extract(r'(\d+\.\d+)').fillna('Unknown')

                        count_data = []

                        app_versions = table_df['App_Version'].fillna('Unknown').unique()

                        app_versions = sorted([v for v in app_versions if v != 'Unknown']) + (["Unknown"] if "Unknown" in app_versions else [])

                        for version in app_versions:

                            row = {'App Version': version}

                            version_df = table_df[table_df['App_Version'] == version]

                            issue_sum = 0

                            for issue in issue_version_cols:

                                if issue == "Miscellaneous":

                                    uncategorized_mask = version_df[issue_names].sum(axis=1) == 0

                                    row[issue] = int(uncategorized_mask.sum())

                                else:

                                    row[issue] = int(version_df[issue].sum()) if issue in version_df.columns else 0

                                issue_sum += row[issue]

                            row['Total Issues'] = issue_sum

                            count_data.append(row)

                        total_row = {'App Version': 'Total'}

                        for issue in issue_version_cols:

                            total_row[issue] = sum(row[issue] for row in count_data)

                        total_row['Total Issues'] = sum(row['Total Issues'] for row in count_data)

                        count_data.append(total_row)

                        issue_count_table = pd.DataFrame(count_data)

                        issue_count_table = issue_count_table.reset_index(drop=True)

                        # Remove S.No. column if present

                        if 'S.No.' in issue_count_table.columns:

                            issue_count_table = issue_count_table.drop(columns=['S.No.'])

                        _ict_css = (
                            "<style>"
                            ".issue-count-table{width:700px;max-width:700px;margin-left:auto;margin-right:auto;border-collapse:collapse;}"
                            ".issue-count-table th{background-color:#FFD700 !important;color:#333 !important;font-weight:bold;font-size:1.05em;border:1px solid #888 !important;padding:8px;}"
                            ".issue-count-table td{font-size:0.95em;border:1px solid #bbb !important;padding:6px;}"
                            ".issue-count-table tr.total-row td{font-size:1.15em;font-weight:bold;background-color:#FFF8DC;}"
                            "</style>"
                        )
                        _ict_header_cols = ''.join(f"<th>{col}</th>" for col in issue_version_cols + ['Total Issues'])
                        _ict_body_rows = ''.join(
                            f"<tr class='total-row'>" + ''.join(f"<td>{_r[col]}</td>" for col in ['App Version'] + issue_version_cols + ['Total Issues']) + "</tr>"
                            if _r['App Version'] == 'Total' else
                            f"<tr>" + ''.join(f"<td>{_r[col]}</td>" for col in ['App Version'] + issue_version_cols + ['Total Issues']) + "</tr>"
                            for _, _r in issue_count_table.iterrows()
                        )
                        _ict_html = (
                            _ict_css
                            + "<div style='text-align:center;font-size:1.5em;font-weight:bold;margin-bottom:8px;'>Issue Type Count by App Version</div>"
                            + "<table class='issue-count-table'>"
                            + "<tr><th>App Version</th>" + _ict_header_cols + "</tr>"
                            + _ict_body_rows
                            + "</table>"
                        )
                        _ict_rows = len(issue_count_table) + 1
                        _ict_height = 80 + (_ict_rows * 36)
                        import streamlit.components.v1 as components
                        components.html(_ict_html, height=_ict_height, scrolling=True)

 

                        # --- Issue Type Funnel for App chart (now below the table) ---

                        import plotly.graph_objects as go

                        from streamlit_plotly_events import plotly_events

                        st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)

                        st.markdown(f"<h3 style='text-align:center;'>Issue Type Funnel for App: {selected_platform}</h3>", unsafe_allow_html=True)

                        # Calculate actual review counts for each issue type as per the displayed reviews

                        actual_issue_counts = []

                        actual_issue_names = []

                        columns = ['DateTimeStamp', 'review', 'App_Version', 'CountryName', 'rating', 'sentiment_label']

                        for issue in funnel_issue_names:

                            if issue == "Miscellaneous":

                                if issue_names:

                                    uncategorized_mask = app_df[issue_names].sum(axis=1) == 0

                                    reviews_df = app_df.loc[uncategorized_mask, columns].copy()

                                else:

                                    reviews_df = app_df[columns].copy()

                                count = len(reviews_df)

                            elif issue == merged_name and merged_keys:

                                if merged_name not in app_df.columns:

                                    app_df[merged_name] = app_df[merged_keys].any(axis=1).astype(int)

                                mask = app_df[merged_name] == 1

                                reviews_df = app_df.loc[mask, columns].copy()

                                count = len(reviews_df)

                            else:

                                mask = app_df[issue] == 1

                                reviews_df = app_df.loc[mask, columns].copy()

                                count = len(reviews_df)

                            actual_issue_names.append(issue)

                            actual_issue_counts.append(count)

                        if funnel_issue_names:

                            uncategorized_mask = app_df[issue_names].sum(axis=1) == 0

                            misc_count = int(uncategorized_mask.sum())

                        else:

                            misc_count = len(app_df)

                        actual_issue_names_with_misc = actual_issue_names + ["Miscellaneous"]

                        actual_issue_counts_with_misc = actual_issue_counts + [misc_count]

                        sorted_pairs = sorted(zip(actual_issue_names_with_misc, actual_issue_counts_with_misc), key=lambda x: x[1])

                        sorted_issue_names = [x[0] for x in sorted_pairs]

                        sorted_issue_counts = [x[1] for x in sorted_pairs]

                        funnel2 = go.Figure(go.Funnel(

                            y=sorted_issue_names,

                            x=sorted_issue_counts,

                            textinfo="value",

                            textposition="inside",

                            marker=dict(color=px.colors.sequential.Blues),

                            hovertemplate='%{y}: %{x}<extra></extra>'

                        ))

                        total_issues_count = sum([count for name, count in zip(sorted_issue_names, sorted_issue_counts) if name != "Miscellaneous"])

                        funnel2.update_layout(

                            title="",

                            margin=dict(l=120, r=40, t=60, b=20),

                            annotations=[

                                dict(

                                    x=0.5,

                                    y=1.05,

                                    xref="paper",

                                    yref="paper",

                                    text=f"<b>Total Issues: {total_issues_count + sorted_issue_counts[sorted_issue_names.index('Miscellaneous')]}</b>",

                                    showarrow=False,

                                    font=dict(size=16, color="white"),

                                    bgcolor="#1976d2",

                                    bordercolor="#0d47a1",

                                    borderwidth=2,

                                    opacity=0.95,

                                    align="center",

                                )

                            ]

                        )

                        selected2 = plotly_events(funnel2, click_event=True, hover_event=False)

                        st.markdown("<br><div style='text-align:center;'><b>Click Issue Type above to see detailed Customer Reviews</b></div><br>", unsafe_allow_html=True)

 

                        if selected2:

                            with st.spinner("Please wait, data is loading..."):

                                selected_issue = sorted_issue_names[selected2[0]['pointIndex']]

                                st.markdown("<br>", unsafe_allow_html=True)

                                st.markdown(f"### Showing Reviews for Issue: {selected_issue} | App Type: {selected_platform}")

                                columns = ['DateTimeStamp', 'review', 'App_Version', 'CountryName', 'rating', 'sentiment_label']

                                if selected_issue == "Miscellaneous":

                                    if issue_names:

                                        uncategorized_mask = app_df[issue_names].sum(axis=1) == 0

                                        reviews_df = app_df.loc[uncategorized_mask, columns].copy()

                                    else:

                                        reviews_df = app_df[columns].copy()

                                else:

                                    mask = app_df[selected_issue] == 1

                                    reviews_df = app_df.loc[mask, columns].copy()

                                if reviews_df.empty:

                                    st.info("No reviews found for this selection.")

                                else:

                                    reviews_df = reviews_df.rename(columns={

                                        'DateTimeStamp': 'Date',

                                        'review': 'Customer Review',

                                        'App_Version': 'App Version',

                                        'CountryName': 'Country',

                                        'rating': 'Rating',

                                        'sentiment_label': 'Sentiment'

                                    })

                                    reviews_df['Date'] = pd.to_datetime(reviews_df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

                                    reviews_df.index = range(1, 1 + len(reviews_df))

                                    reviews_df.index.name = "S.No."

                                    st.dataframe(reviews_df, height=350, use_container_width=True)

                                    csv = reviews_df.to_csv(index=False).encode('utf-8')

                                    col_dl1, col_dl2 = st.columns(2)

                                    with col_dl1:

                                        st.download_button(

                                            label="Download Category wise Reviews",

                                            data=csv,

                                            file_name=f"reviews_{selected_platform}_{selected_issue}.csv",

                                            mime='text/csv',

                                            key=f"dl_reviews_{selected_platform}_{selected_issue}",

                                            use_container_width=True

                                        )

                                    with col_dl2:

                                        st.download_button(

                                            label="Download All Reviews (CSV)",

                                            data=finaldf.to_csv(index=False).encode('utf-8-sig'),

                                            file_name="all_reviews.csv",

                                            mime="text/csv",

                                            key=f"dl_all_reviews_{selected_platform}_{selected_issue}",

                                            use_container_width=True

                                        )

 

        elif st.session_state.selected_chart == "competitor_comparison":

 

            # --- App IDs for competitors ---

            competitor_app_ids = {

                "Android": {

                    "Xoom": "com.xoom.android.app",

                    "Remitly": "com.remitly.androidapp",

                    "PayPal": "com.paypal.android.p2pmobile",

                    "MoneyGram": "com.mgi.moneygram"

                },

                "iOS": {

                    "Xoom": "488553930",

                    "Remitly": "601372100",

                    "PayPal": "283646709",

                    "MoneyGram": "706903888"

                }

            }

 

            def fetch_playstore_reviews(app_id, country_code):

                try:

                    from google_play_scraper import reviews_all, Sort

                    result = reviews_all(

                        app_id,

                        lang='en',

                        country=country_code,

                        sort=Sort.NEWEST

                    )

                    df = pd.DataFrame(result)

                    if 'at' in df.columns:

                        df['at'] = pd.to_datetime(df['at'], errors='coerce')

                    return df

                except Exception as e:

                    return pd.DataFrame()

 

            import feedparser

            def fetch_appstore_reviews_rss(app_id, country_code):

                url = f"https://itunes.apple.com/{country_code}/rss/customerreviews/id={app_id}/sortBy=mostRecent/xml"

                feed = feedparser.parse(url)

                reviews = []

                for entry in feed.entries:

                    rating = None

                    if 'im_rating' in entry:

                        try:

                            rating = int(entry['im_rating'])

                        except Exception:

                            rating = None

                    reviews.append({

                        'author': entry.get('author', ''),

                        'title': entry.get('title', ''),

                        'content': entry.content[0].value if hasattr(entry, 'content') and entry.content else '',

                        'rating': rating,

                        'date': entry.get('updated', entry.get('published', ''))

                    })

                df = pd.DataFrame(reviews)

                if 'date' in df.columns:

                    df['date'] = pd.to_datetime(df['date'], errors='coerce')

                return df

 

            st.markdown("## Competitor Comparison")

            st.markdown("""

                <style>

                .competitor-section label { font-weight: bold; }

                </style>

            """, unsafe_allow_html=True)

            competitor_list = ["Xoom", "Remitly", "PayPal", "MoneyGram"]

            app_types = ["Android"]

            all_countries = ["United States"]

            with st.container():

                col1, col2, col3 = st.columns(3)

                with col1:

                    selected_country = st.selectbox("Select Country", all_countries, index=0, key="cc_country")

                with col2:

                    selected_competitor = st.selectbox("Select Competitor", competitor_list, key="cc_competitor")

                with col3:

                    selected_apptype = st.selectbox("Select App Type", app_types, key="cc_apptype")

            st.markdown("<br>", unsafe_allow_html=True)

 

            app_type_map = {"Android": "Android", "iOS": "iOS"}

            wu_app_name = app_type_map.get(selected_apptype, "Android")

 

            selected_country_code = None

            for code, name in country_map.items():

                if name == selected_country:

                    selected_country_code = code

                    break

 

            def get_country_col(df):

                if "Country" in df.columns:

                    return "Country"

                elif "CountryName" in df.columns:

                    return "CountryName"

                else:

                    return None

 

            wu_country_col = get_country_col(finaldf)

            comp_country_col = get_country_col(filtered_df)

 

            #

            if wu_country_col:

                wu_mask = (

                    finaldf[wu_country_col].str.lower() == (selected_country_code or "").lower()

                ) & (

                    finaldf["AppName"].str.lower() == wu_app_name.lower()

                )

                wu_df = finaldf[wu_mask].copy()

                date_col = None

                if 'TimeStamp' in wu_df.columns:

                    if wu_df['TimeStamp'].notna().any():

                        date_col = 'TimeStamp'

                if date_col:

                    wu_df[date_col] = pd.to_datetime(wu_df[date_col], errors='coerce')

                    if wu_df[date_col].dt.tz is not None:

                        wu_df[date_col] = wu_df[date_col].dt.tz_localize(None)

                    date1_naive = date1_dt.replace(tzinfo=None) if hasattr(date1_dt, 'tzinfo') and date1_dt.tzinfo is not None else date1_dt

                    date2_naive = date2_dt.replace(tzinfo=None) if hasattr(date2_dt, 'tzinfo') and date2_dt.tzinfo is not None else date2_dt

                    wu_df = wu_df[(wu_df[date_col] >= date1_naive) & (wu_df[date_col] <= date2_naive)]

                wu_rating = pd.to_numeric(wu_df["rating"], errors='coerce').dropna().astype(float)

                wu_avg = wu_rating.mean() if not wu_rating.empty else None

            else:

                wu_avg = None

 

            # --- Fetch competitor reviews and filter by date ---

            comp_avg = None

            comp_rating_count = None

            competitor_reviews_df = None

            app_id = competitor_app_ids.get(selected_apptype, {}).get(selected_competitor, None)

            country_code = (selected_country_code or 'us').lower()

            if app_id:

                if selected_apptype == "Android":

                    competitor_reviews_df = fetch_playstore_reviews(app_id, country_code)

                    date_col = 'at' if 'at' in competitor_reviews_df.columns else None

                elif selected_apptype == "iOS":

                    competitor_reviews_df = fetch_appstore_reviews_rss(app_id, country_code)

                    print(f"Fetched {len(competitor_reviews_df)} iOS competitor reviews for app_id={app_id}, country={country_code} before filtering.")

                    date_col = 'date' if 'date' in competitor_reviews_df.columns else None

                if competitor_reviews_df is not None and date_col:

                    competitor_reviews_df[date_col] = pd.to_datetime(competitor_reviews_df[date_col], errors='coerce')

                    mask = (competitor_reviews_df[date_col] >= date1_dt) & (competitor_reviews_df[date_col] <= date2_dt)

                    period_reviews = competitor_reviews_df[mask]

                    print(f"Remaining {len(period_reviews)} iOS competitor reviews after date filtering.")

                    rating_col = 'score' if 'score' in period_reviews.columns else ('rating' if 'rating' in period_reviews.columns else None)

                    if rating_col and not period_reviews.empty:

                        comp_avg = pd.to_numeric(period_reviews[rating_col], errors='coerce').dropna().astype(float).mean()

                        comp_rating_count = len(period_reviews)

                    else:

                        comp_avg = None

                        comp_rating_count = 0

                else:

                    comp_avg = None

                    comp_rating_count = 0

            # Fallback: mentions in WU reviews

            if comp_avg is None:

                competitor_keywords = {

                    "xoom": ["xoom"],

                    "remitly": ["remitly"],

                    "paypal": ["paypal", "pay pal"],

                    "moneygram": ["moneygram", "money gram"]

                }

                competitor_key = selected_competitor.lower()

                competitor_kw = competitor_keywords.get(competitor_key, [competitor_key])

                if comp_country_col:

                    comp_mask = (

                        filtered_df[comp_country_col].str.lower() == (selected_country_code or "").lower()

                    ) & (

                        filtered_df["review"].str.lower().fillna("").apply(lambda x: any(kw in x for kw in competitor_kw))

                    )

                    comp_df = filtered_df[comp_mask]

                    comp_rating = comp_df["rating"].dropna().astype(float)

                    comp_avg = comp_rating.mean() if not comp_rating.empty else None

                    comp_rating_count = len(comp_rating)

 

            col_wu, col_vs, col_comp = st.columns([2, 1, 2])

 

            with col_wu:

                st.markdown(

                    f"<div style='text-align:center;'><b>Western Union ({selected_apptype})</b></div>",

                    unsafe_allow_html=True

                )

                st.markdown("<br>", unsafe_allow_html=True)

                if wu_avg is not None:

                    st.metric("Average Rating", f"{wu_avg:.2f}")

                else:

                    st.info("No data available.")

 

            with col_vs:

                st.markdown("<div style='text-align:center; font-size:2em;'>vs</div>", unsafe_allow_html=True)

 

            with col_comp:

                st.markdown(

                    f"<div style='text-align:center;'><b>{selected_competitor} ({selected_apptype})</b></div>",

                    unsafe_allow_html=True

                )

                st.markdown("<br>", unsafe_allow_html=True)

                if comp_avg is not None:

                    st.metric("Average Rating", f"{comp_avg:.2f}")

                    st.caption(f"Reviews: {comp_rating_count}")

                else:

                    st.info("No competitor reviews found for the selected period.")

 

            # --- Add Word Clouds for WU and Competitor Reviews ---

 

            st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown("<div style='text-align:center; font-size:1.2em; font-weight:bold;'>Review Word Clouds (Selected Period)</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

 

            # Sentiment dropdown for word clouds

            sentiment_option = st.selectbox(

                "Select Sentiment for Word Cloud",

                ["Positive", "Negative"],

                index=0,

                key="wc_sentiment_select"

            )

            wc_col1, wc_col2 = st.columns(2)

 

            # --- WU Word Cloud ---

            with wc_col1:

                st.markdown("<div style='text-align:center; font-weight:bold;'>Western Union Reviews</div>", unsafe_allow_html=True)

                st.markdown("<br><br>", unsafe_allow_html=True)

                wu_wc_df = wu_df.copy() if 'wu_df' in locals() else pd.DataFrame()

                if not wu_wc_df.empty and 'review' in wu_wc_df.columns:

                    # Sentiment filtering

                    if sentiment_option == "Positive":

                        wu_wc_df = wu_wc_df[wu_wc_df['review'].apply(lambda x: sia.polarity_scores(str(x))['compound'] >= 0.05 if pd.notnull(x) else False)]

                    else:

                        wu_wc_df = wu_wc_df[wu_wc_df['review'].apply(lambda x: sia.polarity_scores(str(x))['compound'] <= -0.05 if pd.notnull(x) else False)]

                    text = " ".join(wu_wc_df['review'].dropna().astype(str))

                    if text.strip():

                        custom_stopwords = {"considering", "app", "application", "review", "store", "play", "update", "version", "device", "phone"}

                        stopwords_set = set(STOPWORDS).union(custom_stopwords)

                        western_union_colors = ["#ffdd00", "#000000"]

                        import re

                        words = re.findall(r'\b\w+\b', text.lower())

                        filtered_words = [w for w in words if w not in stopwords_set and len(w) > 2]

                        filtered_text = " ".join(filtered_words)

                        def western_union_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

                            return np.random.choice(western_union_colors)

                        try:

                            mask = np.array(Image.open("Images/wuupdated.png"))

                        except Exception:

                            mask = None

                        wordcloud = WordCloud(

                            stopwords=stopwords_set,

                            max_words=30,

                            width=600,

                            height=400,

                            background_color='white',

                            color_func=western_union_color_func,

                            contour_color='black',

                            contour_width=2,

                            collocations=False

                        ).generate(filtered_text)

                        fig, ax = plt.subplots(figsize=(6, 3))

                        ax.imshow(wordcloud, interpolation='bilinear')

                        ax.axis('off')

                        fig.tight_layout(pad=0)

                        st.pyplot(fig)

                    else:

                        st.info(f"No {sentiment_option.lower()} review text available for word cloud.")

                else:

                    st.info("No WU reviews available for word cloud.")

 

            # --- Competitor Word Cloud ---

            with wc_col2:

                st.markdown(f"<div style='text-align:center; font-weight:bold;'>{selected_competitor} Reviews</div>", unsafe_allow_html=True)

                st.markdown("<br><br>", unsafe_allow_html=True)

                # Ensure period_reviews is defined and fallback to competitor_reviews_df, then comp_df (from filtered_df), if needed

                comp_wc_df = None

                if 'period_reviews' in locals() and period_reviews is not None and not period_reviews.empty:

                    comp_wc_df = period_reviews.copy()

                elif 'competitor_reviews_df' in locals() and competitor_reviews_df is not None and not competitor_reviews_df.empty:

                    comp_wc_df = competitor_reviews_df.copy()

                elif 'comp_df' in locals() and comp_df is not None and not comp_df.empty:

                    comp_wc_df = comp_df.copy()

                else:

                    comp_wc_df = pd.DataFrame()

                review_col = None

                if not comp_wc_df.empty:

                    if 'review' in comp_wc_df.columns:

                        review_col = 'review'

                    elif 'content' in comp_wc_df.columns:

                        review_col = 'content'

                if review_col:

                    # Sentiment filtering

                    if sentiment_option == "Positive":

                        comp_wc_df = comp_wc_df[comp_wc_df[review_col].apply(lambda x: sia.polarity_scores(str(x))['compound'] >= 0.05 if pd.notnull(x) else False)]

                    else:

                        comp_wc_df = comp_wc_df[comp_wc_df[review_col].apply(lambda x: sia.polarity_scores(str(x))['compound'] <= -0.05 if pd.notnull(x) else False)]

                    text = " ".join(comp_wc_df[review_col].dropna().astype(str))

                    if text.strip():

                        # Remove stopwords and most common words

                        from collections import Counter

                        import re

                        # Combine STOPWORDS and NLTK stopwords

                        stopwords_set = set(STOPWORDS).union(stop_words)

                        # Tokenize and clean text

                        words = re.findall(r'\b\w+\b', text.lower())

                        # Remove stopwords

                        filtered_words = [w for w in words if w not in stopwords_set]

                        # Remove most common words (top 10)

                        word_counts = Counter(filtered_words)

                        most_common_words = set([w for w, _ in word_counts.most_common(10)])

                        final_words = [w for w in filtered_words if w not in most_common_words]

                        # Rebuild text

                        filtered_text = " ".join(final_words)

                        if filtered_text.strip():

                            competitor_colors = ["#007BFF", "#00BFFF", "#0057B7"]

                            def competitor_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

                                return np.random.choice(competitor_colors)

                            wordcloud = WordCloud(

                                stopwords=stopwords_set.union(most_common_words),

                                max_words=30,

                                width=600,

                                height=400,

                                background_color='white',

                                color_func=competitor_color_func,

                                contour_color='blue',

                                contour_width=2,

                                collocations=False

                            ).generate(filtered_text)

                            fig, ax = plt.subplots(figsize=(6, 3))

                            ax.imshow(wordcloud, interpolation='bilinear')

                            ax.axis('off')

                            fig.tight_layout(pad=0)

                            st.pyplot(fig)

                        else:

                            st.info(f"No {sentiment_option.lower()} review text available for word cloud.")

                    else:

                        st.info(f"No {sentiment_option.lower()} review text available for word cloud.")

                else:

                    st.info("No competitor reviews available for word cloud.")

 

        elif st.session_state.selected_chart == "translation":

            #st.subheader("🌐 Language Translation - Filtered Reviews")

 

            # Show filtered DataFrame first

            if not filtered_df.empty:

                # Format timestamp

                filtered_df["DateTimeStamp"] = pd.to_datetime(filtered_df["DateTimeStamp"])

                filtered_df["DateTimeStamp"] = filtered_df["DateTimeStamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Pagination setup

                page_size = 100

                total_pages = max((len(filtered_df) - 1) // page_size + 1, 1)

                page_num = st.session_state.get("page_slider_translation", 1)

                page_num = max(1, min(page_num, total_pages))

 

                start_idx = (page_num - 1) * page_size

                end_idx = min(start_idx + page_size, len(filtered_df))

 

                # Column configuration

                column_config = {

                    "TimeStamp": st.column_config.DatetimeColumn(label="📅 Date", width="small", format="YYYY-MM-DD"),

                    "review": st.column_config.TextColumn(label="💬 Review", width="large"),

                    # "rating": st.column_config.NumberColumn(label="⭐ Rating", width="small", format="%.0f ⭐"),

                    "CustomerRating": st.column_config.TextColumn(label="⭐ Rating", width="small"),

                    "CountryName": st.column_config.TextColumn(label="🌍 Country", width="small"),

                    "AppName": st.column_config.TextColumn(label="📱 App", width="small"),

                    "appVersion": st.column_config.TextColumn(label="🔢 Version", width="small"),

                    "UserName": st.column_config.TextColumn(label="👤 User", width="small")

                }

 

                # Display DataFrame

                st.dataframe(filtered_df.iloc[start_idx:end_idx], column_config=column_config, height=275, use_container_width=True)

                #st.caption("ℹ️ Showing filtered reviews for translation.")

 

                # Pagination slider

                if total_pages > 1:

                    col_left, col_center, col_right = st.columns([1, 2, 1])

                    with col_center:

                        st.markdown("<div style='text-align: center; font-size: 14px;'>Navigate Pages</div>", unsafe_allow_html=True)

                        page_num = st.slider("", min_value=1, max_value=total_pages, value=page_num, key="page_slider_translation")

                # else:

                #     st.markdown("<div style='text-align: center; font-size: 8px;'>Only one page of results</div>", unsafe_allow_html=True)

               

                st.markdown("<br>", unsafe_allow_html=True)

                st.caption(f"Showing page {page_num} of {total_pages} — rows {start_idx + 1} to {end_idx}")

               

                st.success(f"✅ Displaying {len(filtered_df)} reviews.")

 

                # Download button

                csv = filtered_df.to_csv(index=False).encode('utf-8-sig')

                col_l, col_c, col_r = st.columns([1, 1, 1])
                with col_c:
                    st.download_button('Download Data', data=csv, file_name="Filtered_Translation_Data.csv", mime="text/csv")

            else:

                st.warning("⚠️ No filtered reviews available for translation.")

 

            st.markdown("---")  # Separator

            # Show translation widget below DataFrame

            show_translation_widget(languages)

 

 

 

 

 

 

# qr_img = Image.open('app_qr_code.png')

# # Add vertical space or put this block at the very end of your app

# # Convert QR image to base64

# buffered = io.BytesIO()

# qr_img.save(buffered, format="PNG")

# img_str = base64.b64encode(buffered.getvalue()).decode()

 

st.markdown(f"""

    <style>

 

    .fixed-bottom-right {{

        position: fixed;

        right: 0;

        bottom: 0;

        margin: 20px;

        z-index: 1000;

        text-align: right;

    }}

    .bottom-link {{

        font-size: 14px;

        color: #003366;

        font-weight: bold;

        text-decoration: none;

        background: #fff;

        padding: 6px 12px;

        border-radius: 5px;

        box-shadow: 0 2px 8px rgba(0,0,0,0.08);

    }}

 

    </style>

 

  
""", unsafe_allow_html=True)

 

st.markdown(
    """
    <div style='text-align: center; margin-top: 5rem; font-size: 0.9rem; color: #666; font-weight: bold;'>
        <p style='margin:0'><em>*Customer Review Data will only be visible when either Language Translation/Reset Button is clicked*</em></p>
        <p style='margin:0'><em>*SunBurst and TreeMap are displayed for data up to 3 months*</em></p>
    </div>
    """,
    unsafe_allow_html=True,
)

 