# -*- coding: utf-8 -*-
# 💬 유튜브 댓글분석기 — 챗봇 + 정량시각화(2x2) 통합본
# - 첫 질문 후 상단에 [키워드/기간/다운로드] 메타 보여주고, 바로 아래에 2x2 시각화 섹션 노출
# - 시각화 섹션(1~4): 각 섹션에 대안 시각화 토글 포함 (키워드: 막대/트리맵, 시간: 선/요일×시간 히트맵,
#   영상: 댓글수 Top10 막대/버블 스캐터, 작성자: Top10 막대/네임태그)
# - 엔티티는 제거: 메인 키워드 1개만 사용
# - 기간 파싱 실패 시: 최근 7일 자동 적용
# - 오류 방지 최소 패치 포함(키 로테이션/쿼터 처리 등)
# - 좋아요 Top10(텍스트 목록) 섹션 제거

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
import numpy as np
from collections import Counter
from kiwipiepy import Kiwi
import stopwordsiso as stopwords

# ==============================================================================
# 페이지/전역 설정 + CSS
# ==============================================================================
st.set_page_config(
    page_title="유튜브 댓글분석: 챗봇",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
  .main .block-container { padding-top: 1.5rem; padding-right: 1rem; padding-left: 1rem; padding-bottom: 4rem; }
  [data-testid="stSidebarContent"] { padding-top: 1rem; }
  header, footer, #MainMenu {visibility: hidden;}

  /* 사이드바 고정폭 */
  [data-testid="stSidebar"] { width: 350px !important; min-width: 350px !important; max-width: 350px !important; }
  [data-testid="stSidebar"] + div[class*="resizer"] { display: none; }

  /* AI 답변 폰트 살짝 축소 */
  [data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) p,
  [data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) li {
      font-size: 0.95rem;
  }

  /* 다운로드 버튼을 텍스트 링크처럼 */
  .stDownloadButton button { background: transparent; color:#1c83e1; border:none; padding:0; text-decoration:underline; font-size:14px; font-weight:normal;}
  .stDownloadButton button:hover { color:#0b5cab; }

  /* 전체 폰트 크기 조정 */
  html, body, [data-testid="stAppViewContainer"] { font-size: 14px; }
  [data-testid="stSidebar"] { font-size: 13px; }

  /* Plotly 여백 축소 */
  .stPlotlyChart { padding: 0.25rem 0 0 0; }
</style>
""",
    unsafe_allow_html=True
)

def _small_fig(fig, *, height=300, title_size=14, font_size=12, legend_size=11, margin=(8,8,28,8)):
    """Plotly 공통 축소 레이아웃 적용 + 축 제목 숨김"""
    l, r, t, b = margin
    fig.update_layout(
        height=height,
        margin=dict(l=l, r=r, t=t, b=b),
        font=dict(size=font_size),
        title_font=dict(size=title_size),
        legend=dict(font=dict(size=legend_size))
    )
    fig.update_xaxes(title=None); fig.update_yaxes(title=None)
    return fig

# ==============================================================================
# 경로/시간대 & 비밀키
# ==============================================================================
BASE_DIR = "/tmp"
SESS_DIR = os.path.join(BASE_DIR, "sessions")
os.makedirs(SESS_DIR, exist_ok=True)

KST = timezone(timedelta(hours=9))
def now_kst() -> datetime:
    return datetime.now(tz=KST)
def to_iso_kst(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST).isoformat(timespec="seconds")
def kst_to_rfc3339_utc(dt_kst: datetime) -> str:
    if dt_kst.tzinfo is None:
        dt_kst = dt_kst.replace(tzinfo=KST)
    return dt_kst.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

# secrets
_YT_FALLBACK, _GEM_FALLBACK = [], []
YT_API_KEYS     = list(st.secrets.get("YT_API_KEYS", [])) or _YT_FALLBACK
GEMINI_API_KEYS = list(st.secrets.get("GEMINI_API_KEYS", [])) or _GEM_FALLBACK
GEMINI_MODEL      = "gemini-2.5-flash-lite"
GEMINI_TIMEOUT    = 120
GEMINI_MAX_TOKENS = 2048

# GitHub(세션 저장용) — 필요 시 secrets에 설정, 없으면 버튼 비활성
GITHUB_TOKEN  = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO   = st.secrets.get("GITHUB_REPO", "")
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")

def _gh_headers(token: str):
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

def github_upload_file(repo, branch, path_in_repo, local_path, token):
    url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
    with open(local_path, "rb") as f:
        content = base64.b64encode(f.read()).decode()
    headers = _gh_headers(token)
    get_resp = requests.get(url + f"?ref={branch}", headers=headers)
    sha = get_resp.json().get("sha") if get_resp.ok else None
    data = {"message": f"archive: {os.path.basename(path_in_repo)}", "content": content, "branch": branch}
    if sha:
        data["sha"] = sha
    resp = requests.put(url, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()

# 상한
MAX_TOTAL_COMMENTS   = 120_000
MAX_COMMENTS_PER_VID = 4_000

# ==============================================================================
# 세션 상태
# ==============================================================================
def ensure_state():
    defaults = {
        "chat": [],
        "last_schema": None,   # {"start_iso","end_iso","keywords"...}
        "last_csv": "",
        "last_df": None,       # videos dataframe
        "sample_text": "",
        "loaded_session_name": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
ensure_state()

# ==============================================================================
# YouTube & Gemini 키 로테이션
# ==============================================================================
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
        if callable(self.on_rotate):
            self.on_rotate(self.idx, self.current())

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
                self.rot.rotate()
                self._build()
                return factory(self.service).execute()
            raise

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
            if out := getattr(resp, "text", None):
                return out
            if c0 := (getattr(resp, "candidates", None) or [None])[0]:
                if p0 := (getattr(c0, "content", None) and getattr(c0.content, "parts", None) or [None])[0]:
                    if hasattr(p0, "text"):
                        return p0.text
            return ""
        except Exception as e:
            if "429" in str(e).lower() and len(rk.keys) > 1:
                rk.rotate(); continue
            raise
    return ""

# ==============================================================================
# 라이트 프롬프트 (엔티티 제거, 메인 키워드만)
# 기간 파싱 실패 시 최근 7일
# ==============================================================================
LIGHT_PROMPT = (
    f"역할: 유튜브 댓글 반응 분석기의 자연어 해석가.\n"
    f"목표: 한국어 입력에서 [기간(KST)]과 [메인 키워드 1개]를 해석.\n"
    f"규칙:\n"
    f"- 기간은 Asia/Seoul 기준, 상대기간의 종료는 지금.\n"
    f"- '키워드'는 검색에 사용할 가장 핵심 주제(프로그램/브랜드/인물) **1개만**.\n"
    f"- 옵션: include_replies(false로 기본), lang(auto).\n\n"
    f"출력(5줄 고정):\n"
    f"- 한 줄 요약: <문장>\n"
    f"- 기간(KST): <YYYY-MM-DDTHH:MM:SS+09:00> ~ <YYYY-MM-DDTHH:MM:SS+09:00>\n"
    f"- 키워드: [<핵심 키워드 1개>]\n"
    f"- 옵션: {{ include_replies: true|false, lang: \"ko|en|auto\" }}\n"
    f"- 원문: {{USER_QUERY}}\n\n"
    f"현재 KST: {to_iso_kst(now_kst())}\n"
    f"입력:\n{{USER_QUERY}}"
)

def parse_light_block_to_schema(light_text: str) -> dict:
    raw = (light_text or "").strip()
    # 기간
    m_time = re.search(r"기간\(KST\)\s*:\s*([^~]+)~\s*([^\n]+)", raw)
    if m_time:
        start_iso, end_iso = (m_time.group(1).strip(), m_time.group(2).strip())
    else:
        end_dt = now_kst()
        start_dt = end_dt - timedelta(days=7)   # 실패 → 최근 7일
        start_iso, end_iso = to_iso_kst(start_dt), to_iso_kst(end_dt)
    # 키워드
    m_kw = re.search(r"키워드\s*:\s*\[(.*?)\]", raw, flags=re.DOTALL)
    keywords = [p.strip() for p in re.split(r"\s*,\s*", m_kw.group(1)) if p.strip()] if m_kw else []
    if not keywords:
        # 문서에서 적당히 추출
        m = re.findall(r"[가-힣A-Za-z0-9]{2,}", raw)
        keywords = [m[0]] if m else ["유튜브"]
    # 옵션
    options = {"include_replies": False, "lang": "auto"}
    m_opt = re.search(r"옵션\s*:\s*\{(.*?)\}", raw, flags=re.DOTALL)
    if m_opt:
        blob = m_opt.group(1)
        if ir := re.search(r"include_replies\s*:\s*(true|false)", blob, re.I):
            options["include_replies"] = (ir.group(1).lower() == "true")
        if lg := re.search(r"lang\s*:\s*\"(ko|en|auto)\"", blob, re.I):
            options["lang"] = lg.group(1)
    return {
        "start_iso": start_iso, "end_iso": end_iso,
        "keywords": [keywords[0]],    # 하나만
        "options": options,
        "raw": raw
    }

# ==============================================================================
# YouTube 수집/가공
# ==============================================================================
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
        video_ids.extend(
            it["id"]["videoId"] for it in resp.get("items", [])
            if it["id"]["videoId"] not in video_ids
        )
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
            if prog_bar is not None:
                prog_bar.progress(min(0.90, 0.50 + (done / total_videos) * 0.40 if total_videos > 0 else 0.50),
                                  text="댓글 수집중…")
            if total_written >= max_total_comments:
                break
    return out_csv, total_written

# ==============================================================================
# LLM 직렬화
# ==============================================================================
def serialize_comments_for_llm_from_file(csv_path: str,
                                         max_chars_per_comment=280,
                                         max_total_chars=420_000):
    if not os.path.exists(csv_path):
        return "", 0, 0
    try:
        df_all = pd.read_csv(csv_path)
    except Exception:
        return "", 0, 0
    if df_all.empty:
        return "", 0, 0

    df_top_likes = df_all.sort_values("likeCount", ascending=False).head(1000)
    df_remaining = df_all.drop(df_top_likes.index)
    df_random = df_remaining.sample(n=min(1000, len(df_remaining))) if not df_remaining.empty else pd.DataFrame()
    df_sample = pd.concat([df_top_likes, df_random])

    lines, total_chars = [], 0
    for _, r in df_sample.iterrows():
        if total_chars >= max_total_chars:
            break
        text = str(r.get("text", "") or "").replace("\n", " ")
        prefix = f"[{'R' if int(r.get('isReply', 0)) == 1 else 'T'}|♥{int(r.get('likeCount', 0))}] "
        prefix += f"{str(r.get('author', '')).replace('\n', ' ')}: "
        body = text[:max_chars_per_comment] + '…' if len(text) > max_chars_per_comment else text
        line = prefix + body
        if total_chars + len(line) + 1 > max_total_chars:
            break
        lines.append(line)
        total_chars += len(line) + 1
    return "\n".join(lines), len(lines), total_chars

TITLE_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s+.*$")
HEADER_DUP_RE = re.compile(r"유튜브\s*댓글\s*분석.*", re.IGNORECASE)
def tidy_answer(md: str) -> str:
    if not md:
        return md
    lines = [
        line for line in md.splitlines()
        if not (TITLE_LINE_RE.match(line) or HEADER_DUP_RE.search(line))
    ]
    cleaned, prev_blank = [], False
    for l in lines:
        is_blank = not l.strip()
        if is_blank and prev_blank:
            continue
        cleaned.append(l)
        prev_blank = is_blank
    return "\n".join(cleaned).strip()

# ==============================================================================
# URL 추출/정리
# ==============================================================================
YTB_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")
def extract_video_ids_from_text(text: str) -> list:
    if not text: return []
    ids = set()
    for m in re.finditer(r"https?://youtu\.be/([A-Za-z0-9_-]{11})", text): ids.add(m.group(1))
    for m in re.finditer(r"https?://(?:www\.)?youtube\.com/shorts/([A-Za-z0-9_-]{11})", text): ids.add(m.group(1))
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

# ==============================================================================
# 사이드바 UI(요약)
# ==============================================================================
with st.sidebar:
    st.markdown(
        '<h2 style="font-weight:600; font-size:1.6rem; margin-bottom:1.2rem; '
        'background:-webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570, #F2A60C); '
        '-webkit-background-clip:text; -webkit-text-fill-color:transparent;">'
        '💬 유튜브 댓글분석: AI 챗봇</h2>',
        unsafe_allow_html=True
    )
    st.caption("문의: 미디어)디지털마케팅 데이터파트")
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("✨ 새 채팅", use_container_width=True):
        st.session_state.clear(); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 상단 도움말 (첫 로드)
# ==============================================================================
if not st.session_state.chat:
    st.markdown(
        """
<div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
            text-align:center; height:65vh;">
  <h1 style="font-size:3rem; font-weight:600;
             background:-webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570, #F2A60C);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    유튜브 댓글분석: AI 챗봇
  </h1>
  <p style="font-size:1.1rem; color:#4b5563;">관련영상 유튜브 댓글반응을 AI가 요약해줍니다</p>
  <div style="margin-top:2rem; padding:1rem 1.5rem; border:1px solid #e5e7eb; border-radius:12px;
              background-color:#fafafa; max-width:670px; text-align:left;">
    <h4 style="margin-bottom:0.6rem; font-weight:600;">⚙️ 사용 팁</h4>
    <ol style="padding-left:20px; line-height:1.7;">
      <li><strong>첫 질문</strong>에 키워드와 기간(예: 최근 7일/5월 1일부터)을 함께 적어줘.</li>
      <li>URL만 붙여도 OK. 여러 URL도 가능.</li>
      <li>엔티티는 쓰지 않아. <strong>메인 키워드 하나</strong>로만 영상 검색해.</li>
    </ol>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

# ==============================================================================
# 메타 + 다운로드 + 시각화 (메인 뷰)
# ==============================================================================
def render_metadata_and_downloads_and_viz():
    schema = st.session_state.get("last_schema")
    if not schema: return

    kw_main = schema.get("keywords", [])
    start_iso = schema.get('start_iso', '')
    end_iso   = schema.get('end_iso', '')

    # 표기용
    try:
        start_dt_str = datetime.fromisoformat(start_iso).astimezone(KST).strftime('%Y-%m-%d %H:%M')
        end_dt_str   = datetime.fromisoformat(end_iso).astimezone(KST).strftime('%Y-%m-%d %H:%M')
    except Exception:
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

        # 다운로드 버튼
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

    # ===== 여기서부터 2x2 시각화 섹션 =====
    comments_csv_path = st.session_state.get("last_csv")
    df_stats = st.session_state.get("last_df")
    if not comments_csv_path or not os.path.exists(comments_csv_path):
        return

    render_quant_viz_2x2(comments_csv_path, df_stats, scope_label="(KST 기준)")

# ==============================================================================
# 시각화 유틸: 형태소/불용어 + 카운터
# ==============================================================================
kiwi = Kiwi()
korean_stopwords = stopwords.stopwords("ko")

@st.cache_data(ttl=600, show_spinner=False)
def compute_keyword_counter_from_file(csv_path: str, stopset_list: list[str], per_comment_cap: int = 200) -> list[tuple[str,int]]:
    if not csv_path or not os.path.exists(csv_path):
        return []
    stopset = set(stopset_list)
    counter = Counter()
    for chunk in pd.read_csv(csv_path, usecols=["text"], chunksize=100_000):
        texts = (chunk["text"].astype(str).str.slice(0, per_comment_cap)).tolist()
        if not texts: continue
        tokens = kiwi.tokenize(" ".join(texts), normalize_coda=True)
        words = [t.form for t in tokens if t.tag in ("NNG","NNP") and len(t.form) > 1 and t.form not in stopset]
        counter.update(words)
    return counter.most_common(300)

# --- 키워드 막대 ---
def keyword_bar_figure_from_counter(counter_items: list[tuple[str,int]], topn: int = 20) -> go.Figure | None:
    if not counter_items: return None
    df_kw = pd.DataFrame(counter_items[:topn], columns=["word","count"])
    df_kw = df_kw.sort_values("count", ascending=True)
    fig = px.bar(df_kw, x="count", y="word", orientation="h", title="키워드 TOP")
    return _small_fig(fig, height=300)

# --- 키워드 트리맵 ---
def keyword_treemap_figure_from_counter(counter_items: list[tuple[str,int]], topn: int = 40) -> go.Figure | None:
    if not counter_items: return None
    df_kw = pd.DataFrame(counter_items[:topn], columns=["word","count"])
    fig = px.treemap(df_kw, path=["word"], values="count", title="키워드 트리맵")
    fig.update_traces(textinfo="label+value")
    return _small_fig(fig, height=300)

# --- 시간대 라인 ---
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

# --- 요일×시간 히트맵 ---
@st.cache_data(ttl=600, show_spinner=False)
def dayhour_heatmap_from_file(csv_path: str):
    if not csv_path or not os.path.exists(csv_path): return None
    vals = []
    for chunk in pd.read_csv(csv_path, usecols=["publishedAt"], chunksize=200_000):
        dt = pd.to_datetime(chunk["publishedAt"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")
        dff = pd.DataFrame({"dow": dt.dt.dayofweek, "hour": dt.dt.hour})
        dff = dff.dropna()
        if not dff.empty:
            vals.append(dff.value_counts().reset_index(name="count"))
    if not vals: return None
    df = pd.concat(vals).groupby(["dow","hour"], as_index=False)["count"].sum()
    dow_map = {0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"}
    df["weekday"] = df["dow"].map(dow_map)
    pivot = df.pivot(index="weekday", columns="hour", values="count").fillna(0).reindex(["월","화","수","목","금","토","일"])
    fig = px.imshow(pivot, aspect="auto", title="요일×시간 댓글 히트맵", labels=dict(x="시", y="", color="댓글수"))
    return _small_fig(fig, height=300)

# --- 작성자 Top ---
def top_authors_from_file(csv_path: str, topn=10):
    if not csv_path or not os.path.exists(csv_path): return None
    counts = {}
    for chunk in pd.read_csv(csv_path, usecols=["author"], chunksize=200_000):
        vc = chunk["author"].astype(str).value_counts()
        for k, v in vc.items():
            counts[k] = counts.get(k, 0) + int(v)
    if not counts: return None
    s = pd.Series(counts).sort_values(ascending=False).head(topn)
    return s.reset_index().rename(columns={"index": "author", 0: "count"})

# --- 작성자 막대 ---
def authors_bar(ta_df: pd.DataFrame) -> go.Figure | None:
    if ta_df is None or ta_df.empty: return None
    df = ta_df.sort_values("count", ascending=True)
    fig = px.bar(df, x="count", y="author", orientation="h", title="작성자 활동량 Top")
    return _small_fig(fig, height=300)

# --- 작성자 네임태그 ---
def authors_name_tags(ta_df: pd.DataFrame):
    if ta_df is None or ta_df.empty:
        st.info("작성자 데이터 없음"); return
    d = ta_df.copy()
    d["norm"] = (d["count"] - d["count"].min()) / max((d["count"].max()-d["count"].min()), 1)
    d["size"] = (d["norm"]*10 + 12).astype(int)  # 12~22pt
    tags = " ".join([f"<span style='font-size:{sz}px; margin:6px; display:inline-block; color:#111827;'>{a}</span>"
                     for a, sz in zip(d["author"], d["size"])])
    st.markdown(f"<div style='line-height:2.0'>{tags}</div>", unsafe_allow_html=True)

# --- 영상: 댓글수 Top10 막대 ---
def top_videos_by_comments(df_stats: pd.DataFrame) -> go.Figure | None:
    if df_stats is None or df_stats.empty: return None
    top_vids = df_stats.sort_values(by="commentCount", ascending=False).head(10).copy()
    top_vids["title_short"] = top_vids["title"].apply(lambda t: t[:24] + "…" if isinstance(t, str) and len(t) > 24 else t)
    fig = px.bar(top_vids, x="commentCount", y="title_short",
                 orientation="h", text="commentCount", title="영상 댓글수 Top10")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return _small_fig(fig, height=300)

# --- 영상: 버블 스캐터(조회 vs 댓글, size=좋아요) ---
def video_bubble_scatter(df_stats: pd.DataFrame) -> go.Figure | None:
    if df_stats is None or df_stats.empty: return None
    df = df_stats.copy()
    df["title_short"] = df["title"].apply(lambda s: s if len(str(s))<=22 else str(s)[:22]+"…")
    df["hover"] = df.apply(
        lambda r: f"{r['title']}<br><a href='https://www.youtube.com/watch?v={r['video_id']}' target='_blank'>열기</a>",
        axis=1
    )
    fig = px.scatter(
        df, x="viewCount", y="commentCount",
        size="likeCount", color="shortType",
        hover_name="title_short", custom_data=["hover"],
        title="영상 포지션(조회수 vs 댓글수, 크기=좋아요)"
    )
    fig.update_traces(hovertemplate="%{customdata}<extra></extra>")
    return _small_fig(fig, height=300)

# --- 2x2 레이아웃 렌더러 ---
def render_quant_viz_2x2(comments_csv_path: str, df_stats: pd.DataFrame, scope_label=""):
    st.markdown("### 📊 정량 시각화")

    # 1행: (1) 키워드 / (2) 시간
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("① 키워드")
            # 불용어 구성(+쿼리 단어 제거)
            custom_stop = {
                "아","휴","아이구","아이쿠","아이고","어","나","우리","저희","따라","의해","을","를",
                "에","의","가","으로","로","에게","뿐이다","의거하여","근거하여","입각하여","기준으로",
                "그냥","댓글","영상","오늘","이제","뭐","진짜","정말","부분","요즘","제발","완전",
                "그게","일단","모든","위해","대한","있지","이유","계속","실제","유튜브","이번","가장","드라마",
            }
            stopset = set(korean_stopwords); stopset.update(custom_stop)
            query_kw = (st.session_state.get("last_schema",{}).get("keywords",[None])[0] or "").strip()
            if query_kw:
                tokens_q = kiwi.tokenize(query_kw, normalize_coda=True)
                query_words = [t.form for t in tokens_q if t.tag in ("NNG","NNP") and len(t.form) > 1]
                stopset.update(query_words)
            view_choice = st.selectbox("표시 방식", ["가로막대", "트리맵"], key="kw_view", index=0, label_visibility="collapsed")
            with st.spinner("키워드 계산 중…"):
                items = compute_keyword_counter_from_file(comments_csv_path, list(stopset), per_comment_cap=200)
            fig_kw = keyword_bar_figure_from_counter(items, topn=20) if view_choice=="가로막대" \
                     else keyword_treemap_figure_from_counter(items, topn=40)
            if fig_kw is None: st.info("표시할 키워드가 없습니다.")
            else: st.plotly_chart(fig_kw, use_container_width=True)

    with c2:
        with st.container(border=True):
            st.subheader("② 시간대")
            ts_choice = st.selectbox("표시 방식", ["선 그래프", "요일×시간 히트맵"], key="time_view", index=0, label_visibility="collapsed")
            if ts_choice == "선 그래프":
                ts, label = timeseries_from_file(comments_csv_path)
                if ts is not None:
                    fig_ts = px.line(ts, x="bucket", y="count", markers=True, title=f"{label} 댓글량 추이 {scope_label}")
                    _small_fig(fig_ts, height=300)
                    st.plotly_chart(fig_ts, use_container_width=True)
                else:
                    st.info("댓글 타임스탬프가 비어 있습니다.")
            else:
                fig_hm = dayhour_heatmap_from_file(comments_csv_path)
                if fig_hm is None: st.info("데이터가 충분하지 않습니다.")
                else: st.plotly_chart(fig_hm, use_container_width=True)

    # 2행: (3) 영상 / (4) 작성자
    c3, c4 = st.columns(2)
    with c3:
        with st.container(border=True):
            st.subheader("③ 영상")
            vid_view = st.selectbox("표시 방식", ["댓글수 Top10(막대)", "버블 스캐터"], key="vid_view", index=0, label_visibility="collapsed")
            if df_stats is None or df_stats.empty:
                st.info("영상 데이터 없음")
            else:
                if vid_view == "댓글수 Top10(막대)":
                    fig_vids = top_videos_by_comments(df_stats)
                    if fig_vids is None: st.info("영상 데이터 없음")
                    else: st.plotly_chart(fig_vids, use_container_width=True)
                else:
                    fig_bbl = video_bubble_scatter(df_stats)
                    if fig_bbl is None: st.info("영상 데이터 없음")
                    else: st.plotly_chart(fig_bbl, use_container_width=True)

    with c4:
        with st.container(border=True):
            st.subheader("④ 작성자")
            auth_view = st.selectbox("표시 방식", ["Top10(막대)", "네임태그"], key="auth_view", index=0, label_visibility="collapsed")
            ta = top_authors_from_file(comments_csv_path, topn=15)
            if ta is not None and not ta.empty:
                if auth_view == "Top10(막대)":
                    fig_auth = authors_bar(ta)
                    if fig_auth is None: st.info("작성자 데이터 없음")
                    else: st.plotly_chart(fig_auth, use_container_width=True)
                else:
                    authors_name_tags(ta)
            else:
                st.info("작성자 데이터 없음")

# ==============================================================================
# 채팅 UI
# ==============================================================================
def render_chat():
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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

# ==============================================================================
# 파이프라인 (첫 턴/후속 턴)
# ==============================================================================
def run_pipeline_first_turn(user_query: str,
                            extra_video_ids=None,
                            only_these_videos: bool = False):
    extra_video_ids = list(dict.fromkeys(extra_video_ids or []))
    prog_bar = st.progress(0, text="준비 중…")

    if not GEMINI_API_KEYS:
        return "오류: Gemini API Key가 설정되지 않았습니다."

    prog_bar.progress(0.06, text="질문 해석중…")
    light = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, "", LIGHT_PROMPT.replace("{USER_QUERY}", user_query))
    schema = parse_light_block_to_schema(light)
    st.session_state["last_schema"] = schema

    prog_bar.progress(0.12, text="영상 수집중…")
    if not YT_API_KEYS:
        return "오류: YouTube API Key가 설정되지 않았습니다."

    rt = RotatingYouTube(YT_API_KEYS)
    # 기간 파싱 실패 대비: parse_light_block_to_schema에서 이미 7일 기본 적용
    start_dt = datetime.fromisoformat(schema["start_iso"])
    end_dt   = datetime.fromisoformat(schema["end_iso"])
    kw_main  = schema.get("keywords", [])
    include_replies = bool(schema.get("options", {}).get("include_replies"))

    # 영상 ID 구성 (메인 키워드만 사용)
    if only_these_videos and extra_video_ids:
        all_ids = extra_video_ids
    else:
        all_ids = []
        for base_kw in (kw_main or ["유튜브"]):
            all_ids.extend(yt_search_videos(
                rt, base_kw, 70, "relevance",
                kst_to_rfc3339_utc(start_dt), kst_to_rfc3339_utc(end_dt)
            ))
        if extra_video_ids:
            all_ids.extend(extra_video_ids)
    all_ids = list(dict.fromkeys(all_ids))

    prog_bar.progress(0.42, text="영상 메타/통계 수집…")
    df_stats = pd.DataFrame(yt_video_statistics(rt, all_ids))
    st.session_state["last_df"] = df_stats

    prog_bar.progress(0.58, text="댓글 수집…")
    csv_path, total_cnt = parallel_collect_comments_streaming(
        df_stats.to_dict('records'),
        YT_API_KEYS,
        include_replies,
        MAX_TOTAL_COMMENTS,
        MAX_COMMENTS_PER_VID,
        prog_bar
    )
    st.session_state["last_csv"] = csv_path

    if total_cnt == 0:
        prog_bar.empty()
        return "지정 조건에서 댓글을 찾을 수 없습니다. 다른 조건으로 시도해 보세요."

    prog_bar.progress(0.88, text="AI 분석중…")
    sample_text, _, _ = serialize_comments_for_llm_from_file(csv_path)
    st.session_state["sample_text"] = sample_text

    sys = (
        "너는 유튜브 댓글을 분석하는 어시스턴트다. "
        "사용자 질문의 핵심 관점(인물/작품/논점)에 맞춰 핵심 포인트를 항목화하고, "
        "긍/부/중 대략 비율과 대표 코멘트(10개 미만)를 제시하라. 반복 금지."
    )
    payload = (
        f"[키워드]: {', '.join(kw_main)}\n"
        f"[기간(KST)]: {schema['start_iso']} ~ {schema['end_iso']}\n\n"
        f"[댓글 샘플]:\n{sample_text}\n"
    )
    answer_md_raw = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, sys, payload)

    prog_bar.progress(1.0, text="완료")
    time.sleep(0.4)
    prog_bar.empty()
    gc.collect()

    return tidy_answer(answer_md_raw)

def run_followup_turn(user_query: str):
    schema = st.session_state.get("last_schema")
    if not schema:
        return "오류: 이전 분석 기록이 없습니다. 새 채팅을 시작해주세요."

    sample_text = st.session_state.get("sample_text", "")
    context = "\n".join(
        f"[이전 {'Q' if m['role'] == 'user' else 'A'}]: {m['content']}"
        for m in st.session_state["chat"][-8:]
    )

    sys = (
        "너는 사용자의 질문 의도를 정확히 파악하여 핵심만 답하는 유튜브 댓글 분석 챗봇이다.\n"
        "정성 질문엔 요약+대표 코멘트(1~3), 정량 질문엔 필요한 수치 중심으로 간결하게.\n"
        "동문서답/반복 금지. 제외 요청은 반드시 제외."
    )
    payload = (
        f"{context}\n\n"
        f"[현재 질문]: {user_query}\n"
        f"[기간(KST)]: {schema.get('start_iso','?')} ~ {schema.get('end_iso','?')}\n\n"
        f"[댓글 샘플]:\n{sample_text}\n"
    )
    return tidy_answer(call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, sys, payload))

# ==============================================================================
# 메인 렌더링
# ==============================================================================
if st.session_state.chat:
    render_metadata_and_downloads_and_viz()
    render_chat()
    scroll_to_bottom()

# 입력창
if prompt := st.chat_input("예) 최근 7일 태풍상사 반응 요약 / 또는 영상 URL 붙여도 OK"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.rerun()

# 입력 처리
if st.session_state.chat and st.session_state.chat[-1]["role"] == "user":
    user_query = st.session_state.chat[-1]["content"]
    url_ids = extract_video_ids_from_text(user_query)
    natural_text = strip_urls(user_query)
    has_urls = len(url_ids) > 0
    has_natural = len(natural_text) > 0

    if not st.session_state.get("last_csv"):
        # 첫 턴: 항상 수집 파이프라인
        if has_urls and not has_natural:
            response = run_pipeline_first_turn(user_query, extra_video_ids=url_ids, only_these_videos=True)
        elif has_urls and has_natural:
            response = run_pipeline_first_turn(user_query, extra_video_ids=url_ids, only_these_videos=False)
        else:
            response = run_pipeline_first_turn(user_query)
    else:
        response = run_followup_turn(user_query)

    st.session_state.chat.append({"role": "assistant", "content": response})
    st.rerun()
