# -*- coding: utf-8 -*-
# 💬 유튜브 댓글분석기 — 순수 챗봇 모드 (엔티티 제거 + 7일 기본 + 정량 시각화 통합/축소)

import streamlit as st
import pandas as pd
import os
import re
import gc
import time
import json
import base64
import requests
import io
import numpy as np

from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai
from streamlit.components.v1 import html as st_html

# 시각화
import plotly.express as px
from plotly import graph_objects as go
import circlify
import stopwordsiso as stopwords
from kiwipiepy import Kiwi


# ======================================================================
# 페이지/전역 설정
# ======================================================================
st.set_page_config(
    page_title="유튜브 댓글분석: 챗봇",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
  .main .block-container { padding-top: 2rem; padding-right: 1rem; padding-left: 1rem; padding-bottom: 5rem; }
  [data-testid="stSidebarContent"] { padding-top: 1.2rem; }
  header, footer, #MainMenu {visibility: hidden;}

  [data-testid="stSidebar"] { width: 350px !important; min-width: 350px !important; max-width: 350px !important; }
  [data-testid="stSidebar"] + div[class*="resizer"] { display: none; }

  [data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) p,
  [data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) li { font-size: 0.95rem; }

  .stDownloadButton button { background: transparent; color:#1c83e1; border:none; padding:0; text-decoration:underline; font-size:14px; font-weight:normal;}
  .stDownloadButton button:hover { color:#0b5cab; }

  .session-list .stButton button { font-size:0.9rem; text-align:left; font-weight:normal; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; display:block; }
  .new-chat-btn button { background-color:#e8f0fe; color:#0052CC !important; border:1px solid #d2e3fc !important; }
  .new-chat-btn button:hover { background-color:#d2e3fc; color:#0041A3 !important; border:1px solid #c2d8f8 !important; }
</style>
""",
    unsafe_allow_html=True
)

# === [GLOBAL TWEAKS] 작은 폰트 & 카드 라인클램프 / 공통 차트 레이아웃 ===
st.markdown("""
<style>
/* 전체 본문 폰트 살짝 축소 */
html, body, [data-testid="stAppViewContainer"] { font-size: 14px; }
[data-testid="stSidebar"] { font-size: 13px; }

/* Plotly 차트 주변 여백 줄이기 */
.stPlotlyChart { padding: 0.25rem 0 0 0; }

/* 좋아요 Top 카드: 본문 2줄로 줄임 (말줄임 처리) */
.cc-card { margin-bottom: 12px; }
.cc-title { font-size: 13px; margin: 2px 0 4px; }
.cc-meta  { font-size: 12px; color:#6b7280; }
.cc-text  {
  font-size: 13px; line-height: 1.35; color:#111827;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

def _small_fig(fig, *, height=250, title_size=13, font_size=11, legend_size=10, margin=(8,8,28,8)):
    """Plotly 공통 축소 레이아웃 적용"""
    l, r, t, b = margin
    fig.update_layout(
        height=height,
        margin=dict(l=l, r=r, t=t, b=b),
        font=dict(size=font_size),
        title_font=dict(size=title_size),
        legend=dict(font=dict(size=legend_size))
    )
    # 모든 그래프의 축 제목 제거
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    return fig


# ======================================================================
# 경로/키/상수
# ======================================================================
BASE_DIR = "/tmp"
SESS_DIR = os.path.join(BASE_DIR, "sessions")
os.makedirs(SESS_DIR, exist_ok=True)

GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "")
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")

KST = timezone(timedelta(hours=9))
def now_kst() -> datetime: return datetime.now(tz=KST)
def to_iso_kst(dt: datetime) -> str:
    if dt.tzinfo is None: dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST).isoformat(timespec="seconds")
def kst_to_rfc3339_utc(dt_kst: datetime) -> str:
    if dt_kst.tzinfo is None: dt_kst = dt_kst.replace(tzinfo=KST)
    return dt_kst.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

_YT_FALLBACK, _GEM_FALLBACK = [], []
YT_API_KEYS       = list(st.secrets.get("YT_API_KEYS", [])) or _YT_FALLBACK
GEMINI_API_KEYS   = list(st.secrets.get("GEMINI_API_KEYS", [])) or _GEM_FALLBACK
GEMINI_MODEL      = "gemini-2.5-flash-lite"
GEMINI_TIMEOUT    = 120
GEMINI_MAX_TOKENS = 2048
MAX_TOTAL_COMMENTS   = 120_000
MAX_COMMENTS_PER_VID = 4_000


# ======================================================================
# 세션 상태
# ======================================================================
def ensure_state():
    defaults = {
        "chat": [],
        "last_schema": None,
        "last_csv": "",
        "last_df": None,
        "sample_text": "",
        "loaded_session_name": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
ensure_state()


# ======================================================================
# GitHub API (옵션)
# ======================================================================
def _gh_headers(token: str):
    h = {"Accept": "application/vnd.github+json"}
    if token: h["Authorization"] = f"token {token}"
    return h

def github_upload_file(repo, branch, path_in_repo, local_path, token):
    url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
    with open(local_path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")
    headers = _gh_headers(token)
    get_resp = requests.get(url + f"?ref={branch}", headers=headers)
    sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None
    data = {"message": f"upload {path_in_repo}", "content": content, "branch": branch}
    if sha: data["sha"] = sha
    resp = requests.put(url, headers=headers, json=data)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"GitHub 업로드 실패: {resp.text}")
    return resp.json()


# ======================================================================
# 유틸
# ======================================================================
def scroll_to_bottom():
    st_html(
        "<script> "
        "let last_message = document.querySelectorAll('.stChatMessage'); "
        "if (last_message.length > 0) { "
        "  last_message[last_message.length - 1].scrollIntoView({behavior: 'smooth'}); "
        "} "
        "</script>",
        height=0
    )

TITLE_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s+.*$")
HEADER_DUP_RE = re.compile(r"유튜브\s*댓글\s*분석.*", re.IGNORECASE)
def tidy_answer(md: str) -> str:
    if not md: return md
    lines = [line for line in md.splitlines()
             if not (TITLE_LINE_RE.match(line) or HEADER_DUP_RE.search(line))]
    cleaned, prev_blank = [], False
    for l in lines:
        is_blank = not l.strip()
        if is_blank and prev_blank: continue
        cleaned.append(l); prev_blank = is_blank
    return "\n".join(cleaned).strip()


# ======================================================================
# 키 로테이터
# ======================================================================
class RotatingKeys:
    def __init__(self, keys, state_key: str, on_rotate=None):
        self.keys = [k.strip() for k in (keys or []) if isinstance(k, str) and k.strip()][:10]
        self.state_key = state_key
        self.on_rotate = on_rotate
        idx = st.session_state.get(state_key, 0)
        self.idx = 0 if not self.keys else (idx % len(self.keys))
        st.session_state[state_key] = self.idx
    def current(self):
        return self.keys[self.idx % len(self.keys)] if self.keys else None
    def rotate(self):
        if not self.keys: return
        self.idx = (self.idx + 1) % len(self.keys)
        st.session_state[self.state_key] = self.idx
        if callable(self.on_rotate): self.on_rotate(self.idx, self.current())

class RotatingYouTube:
    def __init__(self, keys, state_key="yt_key_idx"):
        self.rot = RotatingKeys(keys, state_key)
        self.service = None
        self._build()
    def _build(self):
        key = self.rot.current()
        if not key:
            raise RuntimeError("YouTube API Key가 비어 있습니다.")
        self.service = build("youtube", "v3", developerKey=key)
    def execute(self, factory):
        try:
            return factory(self.service).execute()
        except HttpError as e:
            status = getattr(getattr(e, 'resp', None), 'status', None)
            msg = (getattr(e, 'content', b'').decode('utf-8', 'ignore') or '').lower()
            if status in (403, 429) and any(t in msg for t in ["quota", "rate", "limit"]) and len(YT_API_KEYS) > 1:
                self.rot.rotate(); self._build()
                return factory(self.service).execute()
            raise


# ======================================================================
# LLM 프롬프트 — 엔티티 제거 버전 (메인 키워드 1개만)
# ======================================================================
LIGHT_PROMPT = (
    f"역할: 유튜브 댓글 반응 분석기의 자연어 해석가.\n"
    f"목표: 한국어 입력에서 [기간(KST)]과 [키워드]를 해석.\n"
    f"규칙:\n"
    f"- 기간은 Asia/Seoul 기준, 상대기간의 종료는 지금.\n"
    f"- '키워드'는 검색에 사용할 가장 핵심 주제(프로그램/브랜드/인물 등) **1개**만 선택.\n"
    f"- '최근'이 없어도 '24시간/48시간/7일/3주/12개월' 숫자+단위를 '최근 N'으로 해석.\n"
    f"- 옵션 탐지: include_replies, channel_filter(any|official|unofficial), lang(ko|en|auto).\n\n"
    f"출력(5줄 고정):\n"
    f"- 한 줄 요약: <문장>\n"
    f"- 기간(KST): <YYYY-MM-DDTHH:MM:SS+09:00> ~ <YYYY-MM-DDTHH:MM:SS+09:00>\n"
    f"- 키워드: [<핵심 키워드 1개>]\n"
    f"- 옵션: {{ include_replies: true|false, channel_filter: \"any|official|unofficial\", lang: \"ko|en|auto\" }}\n"
    f"- 원문: {{USER_QUERY}}\n\n"
    f"현재 KST: {to_iso_kst(now_kst())}\n"
    f"입력:\n{{USER_QUERY}}"
)

def call_gemini_rotating(model_name, keys, system_instruction, user_payload,
                         timeout_s=120, max_tokens=2048) -> str:
    rk = RotatingKeys(keys, "gem_key_idx")
    if not rk.current():
        raise RuntimeError("Gemini API Key가 비어 있습니다.")
    for _ in range(len(rk.keys) or 1):
        try:
            genai.configure(api_key=rk.current())
            model = genai.GenerativeModel(
                model_name,
                generation_config={"temperature": 0.2, "max_output_tokens": max_tokens}
            )
            resp = model.generate_content(
                [system_instruction, user_payload],
                request_options={"timeout": timeout_s}
            )
            if out := getattr(resp, "text", None): return out
            if c0 := (getattr(resp, "candidates", None) or [None])[0]:
                if p0 := (getattr(c0, "content", None) and getattr(c0.content, "parts", None) or [None])[0]:
                    if hasattr(p0, "text"): return p0.text
            return ""
        except Exception as e:
            if "429" in str(e).lower() and len(rk.keys) > 1:
                rk.rotate(); continue
            raise
    return ""


# ======================================================================
# 라이트 블록 파싱 → 스키마 (기간 실패 시 최근 7일 자동)
# ======================================================================
def parse_light_block_to_schema(light_text: str) -> dict:
    raw = (light_text or "").strip()

    m_time = re.search(r"기간\(KST\)\s*:\s*([^~]+)~\s*([^\n]+)", raw)
    start_iso, end_iso = (m_time.group(1).strip(), m_time.group(2).strip()) if m_time else (None, None)

    m_kw = re.search(r"키워드\s*:\s*\[(.*?)\]", raw, flags=re.DOTALL)
    keywords = []
    if m_kw:
        pick = [p.strip() for p in re.split(r"\s*,\s*", m_kw.group(1)) if p.strip()]
        if pick: keywords = [pick[0]]  # 첫 1개만

    m_opt = re.search(r"옵션\s*:\s*\{(.*?)\}", raw, flags=re.DOTALL)
    options = {"include_replies": False, "channel_filter": "any", "lang": "auto"}
    if m_opt:
        blob = m_opt.group(1)
        if ir := re.search(r"include_replies\s*:\s*(true|false)", blob, re.I):
            options["include_replies"] = (ir.group(1).lower() == "true")
        if cf := re.search(r"channel_filter\s*:\s*\"(any|official|unofficial)\"", blob, re.I):
            options["channel_filter"] = cf.group(1)
        if lg := re.search(r"lang\s*:\s*\"(ko|en|auto)\"", blob, re.I):
            options["lang"] = lg.group(1)

    # 기간 파싱 실패 → 최근 7일
    if not (start_iso and end_iso):
        end_dt = now_kst()
        start_dt = end_dt - timedelta(days=7)
        start_iso, end_iso = to_iso_kst(start_dt), to_iso_kst(end_dt)

    # 키워드 누락 → 입력문에서 2자 이상 추출 fallback
    if not keywords:
        m = re.findall(r"[가-힣A-Za-z0-9]{2,}", raw)
        keywords = [m[0]] if m else ["유튜브"]

    return {
        "start_iso": start_iso,
        "end_iso": end_iso,
        "keywords": keywords,
        "options": options,
        "raw": raw
    }


# ======================================================================
# YouTube 수집
# ======================================================================
def yt_search_videos(rt, keyword, max_results, order="relevance",
                     published_after=None, published_before=None):
    video_ids, token = [], None
    while len(video_ids) < max_results:
        params = {
            "q": keyword,
            "part": "id",
            "type": "video",
            "order": order,
            "maxResults": min(50, max_results - len(video_ids))
        }
        if published_after: params["publishedAfter"] = published_after
        if published_before: params["publishedBefore"] = published_before
        if token: params["pageToken"] = token

        resp = rt.execute(lambda s: s.search().list(**params))
        video_ids.extend(it["id"]["videoId"] for it in resp.get("items", [])
                         if it["id"]["videoId"] not in video_ids)
        token = resp.get("nextPageToken")
        if not token: break
        time.sleep(0.25)
    return video_ids

def yt_video_statistics(rt, video_ids):
    rows = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        if not batch: continue
        resp = rt.execute(
            lambda s: s.videos().list(part="statistics,snippet,contentDetails", id=",".join(batch))
        )
        for item in resp.get("items", []):
            stats = item.get("statistics", {})
            snip  = item.get("snippet", {})
            cont  = item.get("contentDetails", {})
            dur   = cont.get("duration", "")

            h = re.search(r"(\d+)H", dur)
            m = re.search(r"(\d+)M", dur)
            s = re.search(r"(\d+)S", dur)
            dur_sec = (int(h.group(1)) * 3600 if h else 0) \
                      + (int(m.group(1)) * 60 if m else 0) \
                      + (int(s.group(1)) if s else 0)
            short_type = "Shorts" if (dur_sec is not None and dur_sec <= 60) else "Clip"

            vid_id = item.get("id")
            rows.append({
                "video_id": vid_id,
                "video_url": f"https://www.youtube.com/watch?v={vid_id}",
                "title": snip.get("title", ""),
                "channelTitle": snip.get("channelTitle", ""),
                "publishedAt": snip.get("publishedAt", ""),
                "duration": dur,
                "shortType": short_type,
                "viewCount": int(stats.get("viewCount", 0) or 0),
                "likeCount": int(stats.get("likeCount", 0) or 0),
                "commentCount": int(stats.get("commentCount", 0) or 0)
            })
        time.sleep(0.25)
    return rows

def yt_all_replies(rt, parent_id, video_id, title="", short_type="Clip", cap=None):
    replies, token = [], None
    while not (cap is not None and len(replies) >= cap):
        try:
            resp = rt.execute(lambda s: s.comments().list(
                part="snippet",
                parentId=parent_id,
                maxResults=100,
                pageToken=token,
                textFormat="plainText"
            ))
        except HttpError:
            break

        for c in resp.get("items", []):
            sn = c["snippet"]
            replies.append({
                "video_id": video_id,
                "video_title": title,
                "shortType": short_type,
                "comment_id": c.get("id", ""),
                "parent_id": parent_id,
                "isReply": 1,
                "author": sn.get("authorDisplayName", ""),
                "text": sn.get("textDisplay", "") or "",
                "publishedAt": sn.get("publishedAt", ""),
                "likeCount": int(sn.get("likeCount", 0) or 0)
            })

        token = resp.get("nextPageToken")
        if not token: break
        time.sleep(0.2)

    return replies[:cap] if cap is not None else replies

def yt_all_comments_sync(rt, video_id, title="", short_type="Clip",
                         include_replies=True, max_per_video=None):
    rows, token = [], None
    while not (max_per_video is not None and len(rows) >= max_per_video):
        try:
            resp = rt.execute(lambda s: s.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                pageToken=token,
                textFormat="plainText"
            ))
        except HttpError:
            break

        for it in resp.get("items", []):
            top = it["snippet"]["topLevelComment"]["snippet"]
            thread_id = it["snippet"]["topLevelComment"]["id"]

            rows.append({
                "video_id": video_id,
                "video_title": title,
                "shortType": short_type,
                "comment_id": thread_id,
                "parent_id": "",
                "isReply": 0,
                "author": top.get("authorDisplayName", ""),
                "text": top.get("textDisplay", "") or "",
                "publishedAt": top.get("publishedAt", ""),
                "likeCount": int(top.get("likeCount", 0) or 0)
            })

            if include_replies and int(it["snippet"].get("totalReplyCount", 0) or 0) > 0:
                cap = None if max_per_video is None else max(0, max_per_video - len(rows))
                if cap == 0: break
                rows.extend(yt_all_replies(rt, thread_id, video_id, title, short_type, cap=cap))

        token = resp.get("nextPageToken")
        if not token: break
        time.sleep(0.2)

    return rows[:max_per_video] if max_per_video is not None else rows

def parallel_collect_comments_streaming(video_list, rt_keys, include_replies,
                                        max_total_comments, max_per_video, prog_bar):
    out_csv = os.path.join(BASE_DIR, f"collect_{uuid4().hex}.csv")
    wrote_header, total_written, done, total_videos = False, 0, 0, len(video_list)

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            ex.submit(
                yt_all_comments_sync,
                RotatingYouTube(rt_keys),
                v["video_id"],
                v.get("title", ""),
                v.get("shortType", "Clip"),
                include_replies,
                max_per_video
            ): v for v in video_list
        }

        for f in as_completed(futures):
            try:
                if comm := f.result():
                    dfc = pd.DataFrame(comm)
                    dfc.to_csv(
                        out_csv,
                        index=False,
                        mode="a" if wrote_header else "w",
                        header=not wrote_header,
                        encoding="utf-8-sig"
                    )
                    wrote_header = True
                    total_written += len(dfc)
            except Exception:
                pass

            done += 1
            prog_bar.progress(
                min(0.90, 0.50 + (done / total_videos) * 0.40 if total_videos > 0 else 0.50),
                text="댓글 수집중…"
            )
            if total_written >= max_total_comments:
                break

    return out_csv, total_written


# ======================================================================
# URL 처리
# ======================================================================
YTB_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")

def extract_video_ids_from_text(text: str) -> list:
    if not text: return []
    ids = set()
    for m in re.finditer(r"https?://youtu\.be/([A-Za-z0-9_-]{11})", text):
        ids.add(m.group(1))
    for m in re.finditer(r"https?://(?:www\.)?youtube\.com/shorts/([A-Za-z0-9_-]{11})", text):
        ids.add(m.group(1))
    for m in re.finditer(r"https?://(?:www\.)?youtube\.com/watch\?[^ \n]+", text):
        url = m.group(0)
        try:
            qs = dict((kv.split("=", 1) + [""])[:2] for kv in url.split("?", 1)[1].split("&"))
            v = qs.get("v", "")
            if YTB_ID_RE.fullmatch(v): ids.add(v)
        except Exception:
            pass
    return list(ids)

def strip_urls(s: str) -> str:
    if not s: return ""
    s = re.sub(r"https?://\S+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ======================================================================
# 직렬화(LLM용)
# ======================================================================
def serialize_comments_for_llm_from_file(csv_path: str,
                                         max_chars_per_comment=280,
                                         max_total_chars=420_000):
    if not os.path.exists(csv_path): return "", 0, 0
    try:
        df_all = pd.read_csv(csv_path)
    except Exception:
        return "", 0, 0
    if df_all.empty: return "", 0, 0

    df_top_likes = df_all.sort_values("likeCount", ascending=False).head(1000)
    df_remaining = df_all.drop(df_top_likes.index)
    df_random = df_remaining.sample(n=min(1000, len(df_remaining))) if not df_remaining.empty else pd.DataFrame()

    df_sample = pd.concat([df_top_likes, df_random])
    lines, total_chars = [], 0

    for _, r in df_sample.iterrows():
        if total_chars >= max_total_chars: break
        text = str(r.get("text", "") or "").replace("\n", " ")
        prefix = f"[{'R' if int(r.get('isReply', 0)) == 1 else 'T'}|♥{int(r.get('likeCount', 0))}] "
        prefix += f"{str(r.get('author', '')).replace('\n', ' ')}: "
        body = text[:max_chars_per_comment] + '…' if len(text) > max_chars_per_comment else text
        line = prefix + body
        if total_chars + len(line) + 1 > max_total_chars: break
        lines.append(line)
        total_chars += len(line) + 1

    return "\n".join(lines), len(lines), total_chars


# ======================================================================
# 형태소/불용어 (버블용)
# ======================================================================
kiwi = Kiwi()
korean_stopwords = stopwords.stopwords("ko")

@st.cache_data(ttl=600, show_spinner=False)
def compute_keyword_counter_from_file(csv_path: str, stopset_list: list[str], per_comment_cap: int = 200) -> list[tuple[str,int]]:
    if not csv_path or not os.path.exists(csv_path):
        return []
    stopset = set(stopset_list)
    from collections import Counter
    counter = Counter()
    for chunk in pd.read_csv(csv_path, usecols=["text"], chunksize=100_000):
        texts = (chunk["text"].astype(str).str.slice(0, per_comment_cap)).tolist()
        if not texts: continue
        tokens = kiwi.tokenize(" ".join(texts), normalize_coda=True)
        words = [t.form for t in tokens if t.tag in ("NNG","NNP") and len(t.form) > 1 and t.form not in stopset]
        counter.update(words)
    return counter.most_common(300)

def keyword_bubble_figure_from_counter(counter_items: list[tuple[str,int]]) -> go.Figure | None:
    if not counter_items:
        return None
    df_kw = pd.DataFrame(counter_items[:30], columns=["word", "count"])
    df_kw["label"] = df_kw["word"] + "<br>" + df_kw["count"].astype(str)
    df_kw["scaled"] = np.sqrt(df_kw["count"])
    circles = circlify.circlify(
        [{"id": w, "datum": s} for w, s in zip(df_kw["word"], df_kw["scaled"])],
        show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )
    pos = {c.ex["id"]: (c.x, c.y, c.r) for c in circles if "id" in c.ex}
    df_kw["x"] = df_kw["word"].map(lambda w: pos[w][0])
    df_kw["y"] = df_kw["word"].map(lambda w: pos[w][1])
    df_kw["r"] = df_kw["word"].map(lambda w: pos[w][2])

    s_min, s_max = df_kw["scaled"].min(), df_kw["scaled"].max()
    # 더 작은 라벨 폰트
    df_kw["font_size"] = df_kw["scaled"].apply(lambda s: int(9 + (s - s_min) / max(s_max - s_min, 1) * 9))  # 9~18px

    fig_kw = go.Figure()
    palette = px.colors.sequential.Blues
    df_kw["color_idx"] = df_kw["scaled"].apply(lambda s: int((s - s_min) / max(s_max - s_min, 1) * (len(palette) - 1)))
    for _, row in df_kw.iterrows():
        color = palette[int(row["color_idx"])]
        fig_kw.add_shape(type="circle", xref="x", yref="y",
                         x0=row["x"] - row["r"], y0=row["y"] - row["r"],
                         x1=row["x"] + row["r"], y1=row["y"] + row["r"],
                         line=dict(width=0), fillcolor=color, opacity=0.88, layer="below")
    fig_kw.add_trace(go.Scatter(
        x=df_kw["x"], y=df_kw["y"], mode="text",
        text=df_kw["label"], textposition="middle center",
        textfont=dict(color="white", size=df_kw["font_size"].tolist()),
        hovertext=df_kw["word"] + " (" + df_kw["count"].astype(str) + ")",
        hovertemplate="%{hovertext}<extra></extra>",
    ))
    fig_kw.update_xaxes(visible=False, range=[-1.05, 1.05])
    fig_kw.update_yaxes(visible=False, range=[-1.05, 1.05], scaleanchor="x", scaleratio=1)
    fig_kw.update_layout(title="Top30 키워드 버블", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=36, b=0))
    # 축 제목 제거(안 보여도 규칙 통일)
    fig_kw.update_xaxes(title=None); fig_kw.update_yaxes(title=None)
    return fig_kw


# ======================================================================
# 정량 시각화: CSV 청크 집계
# ======================================================================
def timeseries_from_file(csv_path: str):
    if not csv_path or not os.path.exists(csv_path): return None, None
    tmin = None; tmax = None
    for chunk in pd.read_csv(csv_path, usecols=["publishedAt"], chunksize=200_000):
        dt = pd.to_datetime(chunk["publishedAt"], errors="coerce", utc=True)
        if dt.notna().any():
            lo, hi = dt.min(), dt.max()
            tmin = lo if (tmin is None or (lo < tmin)) else tmin
            tmax = hi if (tmax is None or (hi > tmax)) else tmax
    if tmin is None or tmax is None:
        return None, None
    span_hours = (tmax - tmin).total_seconds()/3600.0
    use_hour = (span_hours <= 48)

    agg = {}
    for chunk in pd.read_csv(csv_path, usecols=["publishedAt"], chunksize=200_000):
        dt = pd.to_datetime(chunk["publishedAt"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")
        dt = dt.dropna()
        if dt.empty: continue
        bucket = (dt.dt.floor("H") if use_hour else dt.dt.floor("D"))
        vc = bucket.value_counts()
        for t, c in vc.items():
            agg[t] = agg.get(t, 0) + int(c)
    ts = pd.Series(agg).sort_index().rename("count").reset_index().rename(columns={"index":"bucket"})
    return ts, ("시간별" if use_hour else "일자별")

def top_authors_from_file(csv_path: str, topn=10):
    if not csv_path or not os.path.exists(csv_path): return None
    counts = {}
    for chunk in pd.read_csv(csv_path, usecols=["author"], chunksize=200_000):
        vc = chunk["author"].astype(str).value_counts()
        for k, v in vc.items():
            counts[k] = counts.get(k, 0) + int(v)
    if not counts: return None
    s = pd.Series(counts).sort_values(ascending=False).head(topn)
    return s.reset_index().rename(columns={"index": "author", 0: "count"}).rename(columns={"count": "count"})

def render_quant_viz_from_paths(comments_csv_path: str, df_stats: pd.DataFrame, scope_label="(KST 기준)"):
    if not comments_csv_path or not os.path.exists(comments_csv_path):
        return

    st.markdown("### 📊 정량 요약")

    # ── 상단 3분할: ① 버블 / ② 타임라인 / ③ Top10 영상댓글 ───────────────
    c1, c2, c3 = st.columns([1, 1, 1])

    # ① 키워드 버블
    with c1:
        with st.container(border=True):
            st.subheader("① 키워드 버블")
            try:
                custom_stopwords = {
                    "아","휴","아이구","아이쿠","아이고","어","나","우리","저희","따라","의해","을","를",
                    "에","의","가","으로","로","에게","뿐이다","의거하여","근거하여","입각하여","기준으로",
                    "그냥","댓글","영상","오늘","이제","뭐","진짜","정말","부분","요즘","제발","완전",
                    "그게","일단","모든","위해","대한","있지","이유","계속","실제","유튜브","이번","가장","드라마",
                }
                stopset = set(korean_stopwords); stopset.update(custom_stopwords)
                query_kw = (st.session_state.get("last_schema", {}) or {}).get("keywords", [""])[0]
                if query_kw:
                    tokens_q = kiwi.tokenize(query_kw, normalize_coda=True)
                    query_words = [t.form for t in tokens_q if t.tag in ("NNG","NNP") and len(t.form) > 1]
                    stopset.update(query_words)
                with st.spinner("키워드 계산 중…"):
                    items = compute_keyword_counter_from_file(comments_csv_path, list(stopset), per_comment_cap=180)
                fig = keyword_bubble_figure_from_counter(items)
                if fig is None:
                    st.info("표시할 키워드가 없습니다(불용어 제거 후 남은 단어 없음).")
                else:
                    _small_fig(fig, height=240, title_size=12, font_size=10, legend_size=10, margin=(6,6,24,6))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            except Exception as e:
                st.info(f"키워드 분석 불가: {e}")

    # ② 시점별 댓글량 추이
    with c2:
        with st.container(border=True):
            st.subheader("② 시점별 댓글량 추이")
            ts, label = timeseries_from_file(comments_csv_path)
            if ts is not None:
                fig_ts = px.line(ts, x="bucket", y="count", markers=True, title=f"{label} 댓글량 추이 {scope_label}")
                _small_fig(fig_ts, height=240, title_size=12, font_size=10, legend_size=10, margin=(8,8,28,8))
                st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("댓글 타임스탬프가 비어 있습니다.")

    # ③ Top10 영상 댓글수 (hover에 URL 노출)
    with c3:
        with st.container(border=True):
            st.subheader("③ Top10 영상 댓글수")
            if df_stats is not None and not df_stats.empty:
                top_vids = df_stats.sort_values(by="commentCount", ascending=False).head(10).copy()
                top_vids["title_short"] = top_vids["title"].apply(lambda t: t[:18] + "…" if isinstance(t, str) and len(t) > 18 else t)
                fig_vids = px.bar(
                    top_vids, x="commentCount", y="title_short",
                    orientation="h", text="commentCount", title="Top10 영상 댓글수"
                )
                # URL hover 노출
                urls = top_vids["video_id"].apply(lambda v: f"https://www.youtube.com/watch?v={v}").tolist()
                fig_vids.update_traces(
                    customdata=np.array(urls),
                    hovertemplate="<b>%{y}</b><br>댓글수: %{x}<br>URL: %{customdata}<extra></extra>"
                )
                _small_fig(fig_vids, height=240, title_size=12, font_size=10, legend_size=10, margin=(8,8,30,8))
                st.plotly_chart(fig_vids, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("영상 메타 데이터가 없습니다.")

    # ── 하단 2분할: ④ 작성자 Top10 / ⑤ 좋아요 Top3 (④와 비슷한 높이) ─────────
    c4, c5 = st.columns([1, 1])

    # ④ 댓글 작성자 활동량 Top10
    with c4:
        with st.container(border=True):
            st.subheader("④ 댓글 작성자 활동량 Top10")
            ta = top_authors_from_file(comments_csv_path, topn=10)
            if ta is not None and not ta.empty:
                fig_auth = px.bar(ta, x="count", y="author", orientation="h", text="count", title="Top10 댓글 작성자 활동량")
                _small_fig(fig_auth, height=260, title_size=12, font_size=10, legend_size=10, margin=(8,8,30,8))
                st.plotly_chart(fig_auth, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("작성자 데이터 없음")

    # ⑤ 댓글 좋아요 Top3 — 카드형으로 간결하게 (④와 높이 유사)
    with c5:
        with st.container(border=True):
            st.subheader("⑤ 댓글 좋아요 Top3")
            best_chunks = []
            usecols = ["video_id","video_title","author","text","likeCount"]
            for chunk in pd.read_csv(comments_csv_path, usecols=usecols, chunksize=120_000):
                chunk["likeCount"] = pd.to_numeric(chunk["likeCount"], errors="coerce").fillna(0).astype(int)
                best_chunks.append(chunk.sort_values("likeCount", ascending=False).head(5))
            if best_chunks:
                df_top = (
                    pd.concat(best_chunks)
                    .sort_values("likeCount", ascending=False)
                    .head(3)
                    .reset_index(drop=True)
                )
                for _, row in df_top.iterrows():
                    url = f"https://www.youtube.com/watch?v={row['video_id']}"
                    author = str(row.get('author',''))
                    title  = str(row.get('video_title','(제목없음)'))[:56]
                    text   = str(row.get('text','')).replace("\n", " ").strip()[:140]
                    like   = int(row.get('likeCount', 0))
                    html = f"""
                    <div class="cc-card">
                      <div class="cc-meta"><b>{like:,} 👍</b> — {author}</div>
                      <div class="cc-title">▶️ <a href="{url}" target="_blank" style="color:#111827; text-decoration:none;">{title}</a></div>
                      <div class="cc-text">{text}</div>
                    </div>
                    """
                    st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("좋아요 상위 댓글 정보가 없습니다.")


# ======================================================================
# 메타 표시/다운로드
# ======================================================================
def render_metadata_and_downloads():
    if not (schema := st.session_state.get("last_schema")):
        return

    kw_main = schema.get("keywords", [])
    start_iso = schema.get('start_iso', '')
    end_iso = schema.get('end_iso', '')

    try:
        start_dt_str = datetime.fromisoformat(start_iso).astimezone(KST).strftime('%Y-%m-%d %H:%M')
        end_dt_str   = datetime.fromisoformat(end_iso).astimezone(KST).strftime('%Y-%m-%d %H:%M')
    except (ValueError, TypeError):
        start_dt_str = start_iso.split('T')[0] if start_iso else ""
        end_dt_str   = end_iso.split('T')[0] if end_iso else ""

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="font-size:14px; color:#4b5563; line-height:1.8;">
              <span style='font-weight:600;'>키워드:</span> {', '.join(kw_main) if kw_main else '(없음)'}<br>
              <span style='font-weight:600;'>기간:</span> {start_dt_str} ~ {end_dt_str} (KST)
            </div>
            """,
            unsafe_allow_html=True
        )

        csv_path  = st.session_state.get("last_csv")
        df_videos = st.session_state.get("last_df")

        if csv_path and os.path.exists(csv_path) and df_videos is not None and not df_videos.empty:
            with open(csv_path, "rb") as f:
                comment_csv_data = f.read()
            buffer = io.BytesIO()
            df_videos.to_csv(buffer, index=False, encoding="utf-8-sig")
            video_csv_data = buffer.getvalue()

            keywords_str = "_".join(kw_main).replace(" ", "_") if kw_main else "data"
            now_str = now_kst().strftime('%Y%m%d')

            col1, col2, col3, _ = st.columns([1.1, 1.2, 1.2, 6.5])
            col1.markdown(
                "<div style='font-size:14px; color:#4b5563; font-weight:600; padding-top:5px;'>다운로드:</div>",
                unsafe_allow_html=True
            )
            with col2:
                st.download_button("전체댓글", comment_csv_data, f"comments_{keywords_str}_{now_str}.csv", "text/csv")
            with col3:
                st.download_button("영상목록", video_csv_data, f"videos_{keywords_str}_{now_str}.csv", "text/csv")


# ======================================================================
# 파이프라인
# ======================================================================
def run_pipeline_first_turn(user_query: str,
                            extra_video_ids=None,
                            only_these_videos: bool = False):
    """
    - 자연어만: 메인 키워드 1개 추출 → 검색
    - 자연어 + URL: 검색 + 해당 URL ID 합산
    - URL만: only_these_videos=True → 해당 URL 댓글만 수집
    """
    extra_video_ids = list(dict.fromkeys(extra_video_ids or []))
    prog_bar = st.progress(0, text="준비 중…")

    if not GEMINI_API_KEYS:
        return "오류: Gemini API Key가 설정되지 않았습니다."

    prog_bar.progress(0.05, text="해석중…")
    light = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, "", LIGHT_PROMPT.replace("{USER_QUERY}", user_query))
    schema = parse_light_block_to_schema(light)
    st.session_state["last_schema"] = schema

    prog_bar.progress(0.10, text="영상 수집중…")
    if not YT_API_KEYS:
        return "오류: YouTube API Key가 설정되지 않았습니다."
    rt = RotatingYouTube(YT_API_KEYS)

    start_dt = datetime.fromisoformat(schema["start_iso"])
    end_dt   = datetime.fromisoformat(schema["end_iso"])
    kw_main  = (schema.get("keywords", []) or ["유튜브"])[0]  # 메인 키워드 1개만

    # --- 영상 ID 구성 ---
    if only_these_videos and extra_video_ids:
        all_ids = extra_video_ids
    else:
        all_ids = []
        # 메인 키워드만 사용
        all_ids.extend(yt_search_videos(
            rt, kw_main, 60, "relevance",
            kst_to_rfc3339_utc(start_dt), kst_to_rfc3339_utc(end_dt)
        ))
        if extra_video_ids:
            all_ids.extend(extra_video_ids)

    all_ids = list(dict.fromkeys(all_ids))  # dedupe

    prog_bar.progress(0.40, text="댓글 수집 준비중…")
    df_stats = pd.DataFrame(yt_video_statistics(rt, all_ids))
    st.session_state["last_df"] = df_stats

    csv_path, total_cnt = parallel_collect_comments_streaming(
        df_stats.to_dict('records'),
        YT_API_KEYS,
        bool(schema.get("options", {}).get("include_replies")),
        MAX_TOTAL_COMMENTS,
        MAX_COMMENTS_PER_VID,
        prog_bar
    )
    st.session_state["last_csv"] = csv_path

    if total_cnt == 0:
        prog_bar.empty()
        return "지정 조건에서 댓글을 찾을 수 없습니다. 다른 조건으로 시도해 보세요."

    prog_bar.progress(0.90, text="AI 분석중…")
    sample_text, _, _ = serialize_comments_for_llm_from_file(csv_path)
    st.session_state["sample_text"] = sample_text

    sys = (
        "너는 유튜브 댓글을 분석하는 어시스턴트다. 먼저 [사용자 원본 질문]을 확인하여 "
        "분석의 핵심 관점을 파악하고, 주어진 댓글 샘플을 바탕으로 해당 관점에 맞춰 "
        "핵심 포인트를 항목화하고, 긍/부/중 비율과 대표 코멘트(10개 미만)를 제시하라. "
        "반복 표현 금지."
    )
    payload = (
        f"[사용자 원본 질문]: {user_query}\n\n"
        f"[키워드]: {kw_main}\n"
        f"[기간(KST)]: {schema['start_iso']} ~ {schema['end_iso']}\n\n"
        f"[댓글 샘플]:\n{sample_text}\n"
    )
    answer_md_raw = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, sys, payload)

    prog_bar.progress(1.0, text="완료")
    time.sleep(0.3)
    prog_bar.empty()
    gc.collect()

    return tidy_answer(answer_md_raw)

def run_followup_turn(user_query: str):
    if not (schema := st.session_state.get("last_schema")):
        return "오류: 이전 분석 기록이 없습니다. 새 채팅을 시작해주세요."

    sample_text = st.session_state.get("sample_text", "")
    context = "\n".join(
        f"[이전 {'Q' if m['role'] == 'user' else 'A'}]: {m['content']}"
        for m in st.session_state["chat"][-10:]
    )

    sys = (
        "너는 사용자의 질문 의도를 정확히 파악하여 핵심만 답변하는 유튜브 댓글 분석 챗봇이다.\n"
        "1) 내용(정성) 질문이면 핵심 반응 요약 + 대표 코멘트 1~3개 인용\n"
        "2) 수치(정량) 질문이면 숫자 중심 간결 응답\n"
        "3) 제외 요청 준수, 반복 금지"
    )
    payload = (
        f"{context}\n\n"
        f"[현재 질문]: {user_query}\n"
        f"[기간(KST)]: {schema.get('start_iso', '?')} ~ {schema.get('end_iso', '?')}\n\n"
        f"[댓글 샘플]:\n{sample_text}\n"
    )
    response_raw = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, sys, payload)
    return tidy_answer(response_raw)


# ======================================================================
# 사이드바 (간단)
# ======================================================================
with st.sidebar:
    st.markdown(
        '<h2 style="font-weight:600; font-size:1.6rem; margin-bottom:1.0rem; '
        'background:-webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570, #F2A60C); '
        '-webkit-background-clip:text; -webkit-text-fill-color:transparent;">'
        '💬 유튜브 댓글분석: AI 챗봇</h2>',
        unsafe_allow_html=True
    )
    st.caption("문의: 미디어)디지털마케팅 데이터파트")

    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("✨ 새 채팅", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ======================================================================
# 메인 화면
# ======================================================================
if not st.session_state.chat:
    st.markdown(
        """
<div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
            text-align:center; height:70vh;">
  <h1 style="font-size:3.2rem; font-weight:600;
             background:-webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570, #F2A60C);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    유튜브 댓글분석: AI 챗봇
  </h1>
  <p style="font-size:1.1rem; color:#4b5563;">관련영상 유튜브 댓글반응을 AI가 요약해줍니다</p>
  <div style="margin-top:2.2rem; padding:1rem 1.2rem; border:1px solid #e5e7eb; border-radius:12px;
              background-color:#fafafa; max-width:600px; text-align:left;">
    <h4 style="margin-bottom:0.8rem; font-weight:600;">⚠️ 사용 주의사항</h4>
    <ol style="padding-left:20px; margin:0;">
      <li><strong>첫 질문 시</strong> 댓글 수집 및 AI 분석에 시간이 소요될 수 있습니다.</li>
      <li>한 세션에서는 <strong>하나의 주제</strong>만 진행해야 분석 정확도가 유지됩니다.</li>
      <li>첫 질문에는 <strong>기간을 명시</strong>해주세요 (예: 최근 48시간 / 5월 1일부터).<br>
          기간 파싱 실패 시 <u>최근 7일</u>로 자동 적용됩니다.</li>
    </ol>
  </div>
</div>
""",
        unsafe_allow_html=True
    )
else:
    # 메타/다운로드 + 정량 시각화 + 대화 로그
    render_metadata_and_downloads()
    # 정량 시각화 (요청 레이아웃/축소/Top3 반영)
    if st.session_state.get("last_csv") and st.session_state.get("last_df") is not None:
        render_quant_viz_from_paths(st.session_state["last_csv"], st.session_state["last_df"], scope_label="(KST 기준)")
    # 대화 로그
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    scroll_to_bottom()


# ======================================================================
# 입력창
# ======================================================================
if prompt := st.chat_input("예) 최근 24시간 태풍상사 반응 요약해줘 / 또는 영상 URL 붙여도 OK"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.rerun()


# ======================================================================
# 입력 처리
# ======================================================================
if st.session_state.chat and st.session_state.chat[-1]["role"] == "user":
    user_query = st.session_state.chat[-1]["content"]

    # URL/자연어 분리
    url_ids = extract_video_ids_from_text(user_query)
    natural_text = strip_urls(user_query)
    has_urls = len(url_ids) > 0
    has_natural = len(natural_text) > 0

    # 첫 턴: 항상 수집 파이프라인
    if not st.session_state.get("last_csv"):
        if has_urls and not has_natural:
            response = run_pipeline_first_turn(user_query, extra_video_ids=url_ids, only_these_videos=True)
        elif has_urls and has_natural:
            response = run_pipeline_first_turn(user_query, extra_video_ids=url_ids, only_these_videos=False)
        else:
            response = run_pipeline_first_turn(user_query)
    else:
        # 후속 턴: 추가 수집 없이 문맥 기반 답변
        response = run_followup_turn(user_query)

    st.session_state.chat.append({"role": "assistant", "content": response})
    st.rerun()
