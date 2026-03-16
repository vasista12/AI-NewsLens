from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import httpx
import re
import json
from datetime import datetime
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
import os

#Load Environment Variables

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

groq_client = Groq(api_key=GROQ_API_KEY)

print("Groq API Key Loaded:", GROQ_API_KEY[:10])


#FastAPI Setup

app = FastAPI(title="NewsLens AI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


#Models

class SummarizeRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None
    summary_length: Optional[str] = "medium"


class SummarizeResponse(BaseModel):
    title: str
    summary: str
    key_points: list[str]
    sentiment: str
    category: str
    read_time_minutes: int
    processed_at: str


#Prompt Builders

def build_summarize_prompt(article_text: str, length: str) -> str:
    length_guide = {
        "short": "2-3 sentences",
        "medium": "1 short paragraph (5-6 sentences)",
        "detailed": "2-3 paragraphs"
    }.get(length, "1 short paragraph")

    return f"""
You are a professional news analyst.

Analyze the article and return ONLY valid JSON.

Article:
\"\"\"{article_text[:6000]}\"\"\"

Return JSON:

{{
"title": "concise headline max 12 words",
"summary": "{length_guide} summary",
"key_points": ["point 1", "point 2", "point 3"],
"sentiment": "Positive or Negative or Neutral",
"category": "Politics, Tech, Sports, Business, Health, Science, Entertainment, or Other",
"read_time_minutes": integer
}}
"""


def build_trending_prompt(articles: list) -> str:
    articles_text = ""

    for i, a in enumerate(articles):
        articles_text += f"""
Article {i+1}
Source: {a.get("source")}
Title: {a.get("title")}
Description: {a.get("description")}
URL: {a.get("url")}
Published: {a.get("publishedAt")}
"""

    return f"""
You are a professional news analyst.

Summarize these trending news articles.

Return ONLY a valid JSON array.

{articles_text}

Format:

[
{{
"title":"short headline",
"source":"news source",
"summary":"2-3 sentence summary",
"sentiment":"Positive or Negative or Neutral",
"category":"Politics, Tech, Sports, Business, Health, Science, Entertainment, or Other",
"url":"article url",
"published":"time"
}}
]
"""


#News Fetchers

async def fetch_newsapi(category: str) -> list:
    if not NEWS_API_KEY:
        return []

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(
                "https://newsapi.org/v2/top-headlines",
                params={
                    "apiKey": NEWS_API_KEY,
                    "language": "en",
                    "category": category,
                    "country": "us",
                    "pageSize": 6
                }
            )

            articles = resp.json().get("articles", [])

            return [
                {
                    "title": a["title"],
                    "description": a.get("description", ""),
                    "url": a.get("url", ""),
                    "publishedAt": a.get("publishedAt", ""),
                    "source": a.get("source", {}).get("name", "Unknown")
                }
                for a in articles
                if a.get("title") and a.get("title") != "[Removed]"
            ][:6]

        except:
            return []


async def fetch_rss(category: str) -> list:

    rss_map = {
        "general": ("https://feeds.bbci.co.uk/news/rss.xml", "BBC"),
        "technology": ("https://techcrunch.com/feed/", "TechCrunch"),
        "business": ("https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "WSJ"),
        "sports": ("https://www.espn.com/espn/rss/news", "ESPN"),
        "science": ("https://www.sciencedaily.com/rss/top/science.xml", "ScienceDaily"),
        "health": ("https://rss.health.com/health/headline-news", "Health.com"),
        "entertainment": ("https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml", "BBC Entertainment")
    }

    feed_url, source_name = rss_map.get(category, rss_map["general"])

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(feed_url, headers={"User-Agent": "Mozilla/5.0"})

            items = re.findall(r"<item>(.*?)</item>", resp.text, re.DOTALL)

            articles = []

            for item in items[:6]:

                title = re.search(r"<title>(.*?)</title>", item)
                link = re.search(r"<link>(.*?)</link>", item)
                desc = re.search(r"<description>(.*?)</description>", item)

                if title:
                    articles.append({
                        "title": title.group(1),
                        "description": desc.group(1)[:300] if desc else "",
                        "url": link.group(1) if link else "",
                        "publishedAt": "",
                        "source": source_name
                    })

            return articles

        except:
            return []


#Routes

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


#Trending Endpoint

@app.get("/trending")
async def get_trending(category: str = "general"):

    articles = await fetch_newsapi(category)

    if not articles:
        articles = await fetch_rss(category)

    if not articles:
        raise HTTPException(status_code=503, detail="Could not fetch news.")

    try:

        prompt = build_trending_prompt(articles)

        message = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=2048
        )

        raw = message.choices[0].message.content.strip()

        raw = raw.replace("```json", "").replace("```", "").strip()

        data = json.loads(raw)

        return {
            "articles": data,
            "category": category,
            "fetched_at": datetime.utcnow().isoformat()
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned malformed JSON.")

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")


#Summarize Endpoint

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):

    if not req.text and not req.url:
        raise HTTPException(status_code=422, detail="Provide text or url")

    if req.url:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(req.url)
            article_text = resp.text[:8000]
    else:
        article_text = req.text

    try:

        prompt = build_summarize_prompt(article_text, req.summary_length)

        message = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024
        )

        raw = message.choices[0].message.content.strip()

        raw = raw.replace("```json", "").replace("```", "").strip()

        data = json.loads(raw)

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")

    return SummarizeResponse(
        title=data.get("title", "Untitled"),
        summary=data.get("summary", ""),
        key_points=data.get("key_points", []),
        sentiment=data.get("sentiment", "Neutral"),
        category=data.get("category", "Other"),
        read_time_minutes=int(data.get("read_time_minutes", 1)),
        processed_at=datetime.utcnow().isoformat()
    )




@app.post("/batch-summarize")
async def batch_summarize(articles: list[SummarizeRequest]):

    if len(articles) > 5:
        raise HTTPException(status_code=400, detail="Max 5 articles allowed")

    results = []

    for article in articles:
        try:
            result = await summarize(article)
            results.append({"status": "success", "data": result})

        except HTTPException as e:
            results.append({"status": "error", "detail": e.detail})

    return {"results": results, "total": len(results)}