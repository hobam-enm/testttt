# -*- coding: utf-8 -*-
# 💬 유튜브 댓글분석기 — 챗봇 + 시각화 통합본 (메인키워드만, 7일 기본, 최소패치, 단일 리런)

import streamlit as st
import pandas as pd
import os, re, gc, time, json, base64, requests, io
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai
from streamlit.components.v1 import html as st_html

# 시각화 관련
import plotly.express as px
from plotly import graph_objects as go
import circlify
import stopwordsiso as stopwords
from kiwipiepy import Kiwi
import numpy as np

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

# ======================================================================
# 경로/시크릿/상수
# ======================================================================
BASE_DIR = "/tmp"
SESS_DIR = os.path.join(BASE_DIR, "sessions")
os.makedirs(SESS_DIR, exist_ok=True)

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
GEMINI_MODEL      = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_TIMEOUT    = int(st.secrets.get("GEMINI_TIMEOUT", 120))
GEMINI_MAX_TOKENS = int(st.secrets.get("GEMINI_MAX_TOKENS", 2048))

GITHUB_TOKEN  = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO   = st.secrets.get("GITHUB_REPO", "")
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")

MAX_TOTAL_COMMENTS   = int(st.secrets.get("MAX_TOTAL_COMMENTS", 120_000))
MAX_COMMENTS_PER_VID = int(st.secrets.get("MAX_COMMENTS_PER_VID", 4_000))

# ======================================================================
# 세이프 리런 & 락(동시 실행 보호)
# ======================================================================
def safe_rerun():
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(fn): fn()

LOCK_PATH = os.path.join(BASE_DIR, "ytccai.busy.lock")
def try_acquire_lock(ttl=7200):
    if os.path.exists(LOCK_PATH):
        try:
            if time.time() - os.path.getmtime(LOCK_PATH) > ttl:
                os.remove(LOCK_PATH)
        except: pass
    if os.path.exists(LOCK_PATH): return False
    open(LOCK_PATH, "w").close()
    return True
def release_lock():
    try:
        if os.path.exists(LOCK_PATH): os.remove(LOCK_PATH)
    except: pass

# ======================================================================
# GitHub API (세션 아카이브)
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

def github_list_dir(repo, branch, folder, token):
    url = f"https://api.github.com/repos/{repo}/contents/{folder}?ref={branch}"
    headers = _gh_headers(token)
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200: return []
    return [item['name'] for item in resp.json() if item.get("type") == "dir"]

def github_download_file(repo, branch, path_in_repo, token, local_path):
    url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}?ref={branch}"
    headers = _gh_headers(token)
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        content = base64.b64decode(data["content"])
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f: f.write(content)
        return True
    return False

def github_delete_folder(repo, branch, folder_path, token):
    contents_url = f"https://api.github.com/repos/{repo}/contents/{folder_path}?ref={branch}"
    headers = _gh_headers(token)
    resp = requests.get(contents_url, headers=headers)
    if not resp.ok: return
    for item in resp.json():
        delete_url = f"https://api.github.com/repos/{repo}/contents/{item['path']}"
        data = {"message": f"delete: {item['name']}", "sha": item['sha'], "branch": branch}
        requests.delete(delete_url, headers=headers, json=data).raise_for_status()

def github_rename_session(old_name, new_name, token):
    contents_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/sessions/{old_name}?ref={GITHUB_BRANCH}"
    resp = requests.get(contents_url, headers=_gh_headers(token))
    resp.raise_for_status()
    files_to_move = resp.json()
    for item in files_to_move:
        filename = item['name']
        local_path = os.path.join(SESS_DIR, filename)
        if not github_download_file(GITHUB_REPO, GITHUB_BRANCH, item['path'], token, local_path):
            raise Exception(f"Failed to download {filename} from {old_name}")
        github_upload_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{new_name}/{filename}", local_path, token)
    github_delete_folder(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{old_name}", token)

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
# 형태소/불용어
# ======================================================================
kiwi = Kiwi()
korean_stopwords = stopwords.stopwords("ko")

# ======================================================================
# 유틸 / UI
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
    lines = [
        line for line in md.splitlines()
        if not (TITLE_LINE_RE.match(line) or HEADER_DUP_RE.search(line))
    ]
    cleaned, prev_blank = [], False
    for l in lines:
        is_blank = not l.strip()
        if is_blank and prev_blank: continue
        cleaned.append(l)
        prev_blank = is_blank
    return "\n".join(cleaned).strip()

# ======================================================================
# 사이드바 UI
# ======================================================================
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
        st.session_state.clear()
        ensure_state()
        safe_rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.chat and st.session_state.last_csv:
        if st.button("💾 현재 대화 저장", use_container_width=True):
            with st.spinner("세션 저장 중..."):
                success, result = None, None
                try:
                    success, result = True, save_current_session_to_github()[1]
                except Exception as e:
                    success, result = False, str(e)
            if success:
                st.success(f"저장 완료: {result}")
                time.sleep(1.2); safe_rerun()
            else:
                st.error(result)

    st.markdown("---")
    st.markdown("#### 대화 기록")

    def github_list_dir_safe():
        if not (GITHUB_TOKEN and GITHUB_REPO): return []
        try:
            return sorted(github_list_dir(GITHUB_REPO, GITHUB_BRANCH, "sessions", GITHUB_TOKEN), reverse=True)
        except Exception:
            return []

    sessions = github_list_dir_safe()
    if not sessions:
        st.caption("저장된 기록이 없습니다.")
    else:
        editing_session = st.session_state.get("editing_session", None)
        st.markdown('<div class="session-list">', unsafe_allow_html=True)
        for sess in sessions:
            if sess == editing_session:
                new_name = st.text_input("새 이름:", value=sess, key=f"new_name_{sess}")
                c1, c2 = st.columns(2)
                if c1.button("✅", key=f"save_{sess}"):
                    try:
                        github_rename_session(sess, new_name, GITHUB_TOKEN)
                        st.session_state.pop('editing_session', None)
                        st.success("이름 변경 완료!"); time.sleep(1); safe_rerun()
                    except Exception as e:
                        st.error(f"변경 실패: {e}")
                if c2.button("❌", key=f"cancel_{sess}"):
                    st.session_state.pop('editing_session', None)
                    safe_rerun()
            else:
                c1, c2, c3 = st.columns([0.7, 0.15, 0.15])
                if c1.button(sess, key=f"sess_{sess}", use_container_width=True):
                    st.session_state.session_to_load = sess
                    safe_rerun()
                if c2.button("✏️", key=f"edit_{sess}"):
                    st.session_state.editing_session = sess
                    safe_rerun()
                if c3.button("🗑️", key=f"del_{sess}"):
                    st.session_state.session_to_delete = sess
                    safe_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""<hr><h3>📞 문의</h3><p>미디어)디지털마케팅 데이터파트</p>""", unsafe_allow_html=True)

# ======================================================================
# GitHub 세션 저장/로드
# ======================================================================
def _build_session_name() -> str:
    schema = st.session_state.get("last_schema", {}) or {}
    kw = (schema.get("keywords", ["NoKeyword"]))[0]
    kw_slug = re.sub(r'[^\w-]', '', kw.replace(' ', '_'))[:20]
    return f"{kw_slug}_{now_kst().strftime('%Y-%m-%d_%H%M')}"

def save_current_session_to_github():
    if not all([GITHUB_REPO, GITHUB_TOKEN, st.session_state.chat, st.session_state.last_csv]):
        return False, "저장할 데이터가 없거나 GitHub 설정이 누락되었습니다."
    sess_name = _build_session_name()
    local_dir = os.path.join(SESS_DIR, sess_name)
    os.makedirs(local_dir, exist_ok=True)
    meta_path = os.path.join(local_dir, "qa.json")
    meta_data = {
        "chat": st.session_state.chat,
        "last_schema": st.session_state.last_schema,
        "sample_text": st.session_state.sample_text
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)

    comments_path = os.path.join(local_dir, "comments.csv")
    videos_path = os.path.join(local_dir, "videos.csv")
    os.system(f'cp "{st.session_state.last_csv}" "{comments_path}"')
    if st.session_state.last_df is not None:
        st.session_state.last_df.to_csv(videos_path, index=False, encoding="utf-8-sig")

    github_upload_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/qa.json", meta_path, GITHUB_TOKEN)
    github_upload_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/comments.csv", comments_path, GITHUB_TOKEN)
    if os.path.exists(videos_path):
        github_upload_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/videos.csv", videos_path, GITHUB_TOKEN)

    st.session_state.loaded_session_name = sess_name
    return True, sess_name

def load_session_from_github(sess_name: str):
    with st.spinner(f"세션 '{sess_name}' 불러오는 중..."):
        try:
            local_dir = os.path.join(SESS_DIR, sess_name)
            qa_ok = github_download_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/qa.json", GITHUB_TOKEN, os.path.join(local_dir, "qa.json"))
            comments_ok = github_download_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/comments.csv", GITHUB_TOKEN, os.path.join(local_dir, "comments.csv"))
            videos_ok = github_download_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/videos.csv", GITHUB_TOKEN, os.path.join(local_dir, "videos.csv"))
            if not (qa_ok and comments_ok):
                st.error("세션 핵심 파일을 불러오는 데 실패했습니다.")
                return
            st.session_state.clear(); ensure_state()
            with open(os.path.join(local_dir, "qa.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            st.session_state.update({
                "chat": meta.get("chat", []),
                "last_schema": meta.get("last_schema", None),
                "last_csv": os.path.join(local_dir, "comments.csv"),
                "last_df": pd.read_csv(os.path.join(local_dir, "videos.csv")) if videos_ok and os.path.exists(os.path.join(local_dir, "videos.csv")) else pd.DataFrame(),
                "loaded_session_name": sess_name,
                "sample_text": meta.get("sample_text", "")
            })
        except Exception as e:
            st.error(f"세션 로드 실패: {e}")

# 세션 로드/삭제 트리거
if 'session_to_load' in st.session_state:
    load_session_from_github(st.session_state.pop('session_to_load')); safe_rerun()
if 'session_to_delete' in st.session_state:
    sess_name = st.session_state.pop('session_to_delete')
    with st.spinner(f"세션 '{sess_name}' 삭제 중..."):
        github_delete_folder(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}", GITHUB_TOKEN)
    st.success("세션 삭제 완료."); time.sleep(1); safe_rerun()

# ======================================================================
# 키 로테이터 / API 래퍼
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

def is_youtube_quota_error(e: HttpError) -> bool:
    try:
        data = json.loads(getattr(e, "content", b"{}").decode("utf-8", errors="ignore"))
        status = getattr(getattr(e, 'resp', None), 'status', None)
        if status in (403, 429):
            reasons = [(err.get("reason") or "").lower() for err in data.get("error", {}).get("errors", [])]
            msg = (data.get("error", {}).get("message", "") or "").lower()
            quota_flags = ("quotaexceeded", "dailylimitexceeded", "ratelimitexceeded")
            if any(r in quota_flags for r in reasons): return True
            if "rate" in msg and "limit" in msg: return True
            if "quota" in msg: return True
        return False
    except Exception:
        return False

def with_retry(fn, tries=2, backoff=1.4):
    for i in range(tries):
        try:
            return fn()
        except HttpError as e:
            status = getattr(getattr(e, 'resp', None), 'status', None)
            if status in (400, 401, 403) and not is_youtube_quota_error(e):
                raise
            if i == tries - 1: raise
            time.sleep((i + 1) * backoff)
        except Exception:
            if i == tries - 1: raise
            time.sleep((i + 1) * backoff)

class RotatingYouTube:
    def __init__(self, keys, state_key="yt_key_idx", log=None):
        self.rot = RotatingKeys(keys, state_key, on_rotate=lambda i, k: log and log(f"🔁 YouTube 키 전환 → #{i+1}"))
        self.log = log
        self.service = None
        self._build_service()
    def _build_service(self):
        key = self.rot.current()
        if not key: raise RuntimeError("YouTube API Key가 비어 있습니다.")
        self.service = build("youtube", "v3", developerKey=key)
    def _rotate_and_rebuild(self):
        self.rot.rotate(); self._build_service()
    def execute(self, request_factory, tries_per_key=2):
        attempts = 0
        max_attempts = len(self.rot.keys) if self.rot.keys else 1
        while attempts < max_attempts:
            try:
                req = request_factory(self.service)
                return with_retry(lambda: req.execute(), tries=tries_per_key, backoff=1.4)
            except HttpError as e:
                if is_youtube_quota_error(e) and len(self.rot.keys) > 1:
                    self._rotate_and_rebuild()
                    attempts += 1
                    continue
                raise

def is_gemini_quota_error(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return ("429" in msg) or ("too many requests" in msg) or ("rate limit" in msg) or ("resource exhausted" in msg) or ("quota" in msg)

def call_gemini_rotating(model_name, keys, system_instruction, user_payload,
                         timeout_s=GEMINI_TIMEOUT, max_tokens=GEMINI_MAX_TOKENS, on_rotate=None) -> str:
    rot = RotatingKeys(keys, state_key="gem_key_idx", on_rotate=lambda i, k: on_rotate and on_rotate(i, k))
    if not rot.current(): raise RuntimeError("Gemini API Key가 비어 있습니다.")
    attempts = 0
    max_attempts = len(rot.keys) if rot.keys else 1
    while attempts < max_attempts:
        try:
            genai.configure(api_key=rot.current())
            model = genai.GenerativeModel(
                model_name,
                generation_config={"temperature": 0.2, "max_output_tokens": max_tokens, "top_p": 0.9}
            )
            resp = model.generate_content([system_instruction, user_payload],
                                          request_options={"timeout": timeout_s})
            out = getattr(resp, "text", None)
            if not out and hasattr(resp, "candidates") and resp.candidates:
                c0 = resp.candidates[0]
                if hasattr(c0, "content") and getattr(c0.content, "parts", None):
                    p0 = c0.content.parts[0]
                    if hasattr(p0, "text"): out = p0.text
            return out or ""
        except Exception as e:
            if is_gemini_quota_error(e) and len(rot.keys) > 1:
                rot.rotate(); attempts += 1; continue
            raise

# ======================================================================
# YouTube 수집/직렬화
# ======================================================================
def yt_search_videos(rt, keyword, max_results, order="relevance", published_after=None, published_before=None, log=None):
    video_ids, token = [], None
    while len(video_ids) < max_results:
        params = dict(q=keyword, part="id", type="video", order=order, maxResults=min(50, max_results - len(video_ids)))
        if published_after: params["publishedAfter"] = published_after
        if published_before: params["publishedBefore"] = published_before
        if token: params["pageToken"] = token
        resp = rt.execute(lambda s: s.search().list(**params))
        for it in resp.get("items", []):
            vid = it["id"]["videoId"]
            if vid not in video_ids: video_ids.append(vid)
        token = resp.get("nextPageToken")
        if not token: break
        if log: log(f"검색 진행: {len(video_ids)}개")
        time.sleep(0.35)
    return video_ids

def yt_video_statistics(rt, video_ids, log=None):
    rows = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        resp = rt.execute(lambda s: s.videos().list(part="statistics,snippet,contentDetails", id=",".join(batch)))
        for item in resp.get("items", []):
            stats = item.get("statistics", {})
            snip = item.get("snippet", {})
            cont = item.get("contentDetails", {})
            dur_iso = cont.get("duration", "")
            def _dsec(dur: str):
                if not dur or not dur.startswith("P"): return None
                h = re.search(r"(\d+)H", dur); m = re.search(r"(\d+)M", dur); s = re.search(r"(\d+)S", dur)
                return (int(h.group(1)) if h else 0) * 3600 + (int(m.group(1)) if m else 0) * 60 + (int(s.group(1)) if s else 0)
            dur_sec = _dsec(dur_iso)
            short_type = "Shorts" if (dur_sec is not None and dur_sec <= 60) else "Clip"
            vid_id = item.get("id")
            rows.append({
                "video_id": vid_id,
                "video_url": f"https://www.youtube.com/watch?v={vid_id}",
                "title": snip.get("title", ""),
                "channelTitle": snip.get("channelTitle", ""),
                "publishedAt": snip.get("publishedAt", ""),
                "duration": dur_iso,
                "shortType": short_type,
                "viewCount": int(stats.get("viewCount", 0) or 0),
                "likeCount": int(stats.get("likeCount", 0) or 0),
                "commentCount": int(stats.get("commentCount", 0) or 0),
            })
        if log: log(f"통계 배치 {i // 50 + 1} 완료")
        time.sleep(0.35)
    return rows

def yt_all_replies(rt, parent_id, video_id, title="", short_type="Clip", log=None, cap=None):
    replies, token = [], None
    while True:
        if cap is not None and len(replies) >= cap:
            return replies[:cap]
        params = dict(part="snippet", parentId=parent_id, maxResults=100, pageToken=token, textFormat="plainText")
        try:
            resp = rt.execute(lambda s: s.comments().list(**params))
        except HttpError as e:
            if log: log(f"[오류] replies {video_id}/{parent_id}: {e}")
            break
        for c in resp.get("items", []):
            sn = c["snippet"]
            replies.append({
                "video_id": video_id, "video_title": title, "shortType": short_type,
                "comment_id": c.get("id", ""), "parent_id": parent_id, "isReply": 1,
                "author": sn.get("authorDisplayName", ""),
                "text": sn.get("textDisplay", "") or "",
                "publishedAt": sn.get("publishedAt", ""),
                "likeCount": int(sn.get("likeCount", 0) or 0),
            })
            if cap is not None and len(replies) >= cap:
                return replies[:cap]
        token = resp.get("nextPageToken")
        if not token: break
        time.sleep(0.25)
    return replies

def yt_all_comments_sync(rt, video_id, title="", short_type="Clip", include_replies=True, log=None,
                    max_per_video: int | None = None):
    rows, token = [], None
    while True:
        if max_per_video is not None and len(rows) >= max_per_video:
            return rows[:max_per_video]
        params = dict(part="snippet,replies", videoId=video_id, maxResults=100, pageToken=token, textFormat="plainText")
        try:
            resp = rt.execute(lambda s: s.commentThreads().list(**params))
        except HttpError as e:
            if log: log(f"[오류] commentThreads {video_id}: {e}")
            break

        for it in resp.get("items", []):
            top = it["snippet"]["topLevelComment"]["snippet"]
            thread_id = it["snippet"]["topLevelComment"]["id"]
            total_replies = int(it["snippet"].get("totalReplyCount", 0) or 0)
            rows.append({
                "video_id": video_id, "video_title": title, "shortType": short_type,
                "comment_id": thread_id, "parent_id": "", "isReply": 0,
                "author": top.get("authorDisplayName", ""),
                "text": top.get("textDisplay", "") or "",
                "publishedAt": top.get("publishedAt", ""),
                "likeCount": int(top.get("likeCount", 0) or 0),
            })
            if include_replies and total_replies > 0:
                cap = None
                if max_per_video is not None:
                    cap = max(0, max_per_video - len(rows))
                if cap == 0:
                    return rows[:max_per_video]
                rows.extend(yt_all_replies(rt, thread_id, video_id, title, short_type, log, cap=cap))
                if max_per_video is not None and len(rows) >= max_per_video:
                    return rows[:max_per_video]

        token = resp.get("nextPageToken")
        if not token: break
        if log: log(f"  댓글 페이지 진행, 누계 {len(rows)}")
        time.sleep(0.25)
    return rows

def parallel_collect_comments_streaming(
    video_list, rt_keys, include_replies, max_total_comments, max_per_video,
    log_callback=None, prog_callback=None
):
    out_csv = os.path.join(BASE_DIR, f"collect_{uuid4().hex}.csv")
    wrote_header = False
    total_written = 0
    total_videos = len(video_list)
    done_videos = 0
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                yt_all_comments_sync,
                RotatingYouTube(rt_keys),
                vid_info["video_id"],
                vid_info.get("title", ""),
                vid_info.get("shortType", "Clip"),
                include_replies,
                None,
                max_per_video
            ): vid_info for vid_info in video_list
        }
        for fut in as_completed(futures):
            vid_info = futures[fut]
            try:
                comments = fut.result()
                if comments:
                    df_chunk = pd.DataFrame(comments)
                    df_chunk.to_csv(
                        out_csv, index=False,
                        mode=("a" if wrote_header else "w"),
                        header=(not wrote_header),
                        encoding="utf-8-sig"
                    )
                    wrote_header = True
                    total_written += len(df_chunk)
                done_videos += 1
                if log_callback: log_callback(f"✅ [{done_videos}/{total_videos}] {vid_info.get('title','')} - {len(comments):,}개 수집")
                if prog_callback: prog_callback(min(1.0, done_videos / max(total_videos, 1)))
            except Exception as e:
                done_videos += 1
                if log_callback: log_callback(f"❌ [{done_videos}/{total_videos}] {vid_info.get('title','')} - 실패: {e}")
                if prog_callback: prog_callback(min(1.0, done_videos / max(total_videos, 1)))
            if total_written >= max_total_comments:
                if log_callback: log_callback(f"최대 수집 한도({max_total_comments:,}개) 도달, 중단")
                break
    return out_csv, total_written

def serialize_comments_for_llm_from_file(csv_path: str, max_rows=1500, max_chars_per_comment=280, max_total_chars=450_000):
    if not csv_path or not os.path.exists(csv_path): return "", 0, 0
    lines, total = [], 0
    for chunk in pd.read_csv(csv_path, chunksize=100_000):
        if "likeCount" in chunk.columns:
            chunk = chunk.sort_values("likeCount", ascending=False)
        for _, r in chunk.iterrows():
            if len(lines) >= max_rows or total >= max_total_chars: break
            is_reply = "R" if int(r.get("isReply", 0) or 0) == 1 else "T"
            author = str(r.get("author", "") or "").replace("\n", " ")
            likec = int(r.get("likeCount", 0) or 0)
            text = str(r.get("text", "") or "").replace("\n", " ")
            if len(text) > max_chars_per_comment:
                text = text[:max_chars_per_comment] + "…"
            line = f"[{is_reply}|♥{likec}] {author}: {text}"
            if total + len(line) + 1 > max_total_chars: break
            lines.append(line)
            total += len(line) + 1
        if len(lines) >= max_rows or total >= max_total_chars: break
    return "\n".join(lines), len(lines), total

# ======================================================================
# 프롬프트(메인 키워드 + 기간만)
# ======================================================================
LIGHT_PROMPT = (
    f"역할: 유튜브 댓글 반응 분석기의 자연어 해석가.\n"
    f"목표: 한국어 입력에서 [기간(KST)]과 [메인 키워드 1개]만 해석.\n"
    f"규칙:\n"
    f"- 기간은 Asia/Seoul 기준, 상대기간의 종료는 지금.\n"
    f"- '키워드'는 검색에 사용할 가장 핵심적인 주제(프로그램/배우/브랜드 등) **정확히 1개**로 제한.\n"
    f"- '최근 N일/시간'처럼 숫자+단위를 기간으로 해석.\n\n"
    f"출력(4줄 고정):\n"
    f"- 한 줄 요약: <문장>\n"
    f"- 기간(KST): <YYYY-MM-DDTHH:MM:SS+09:00> ~ <YYYY-MM-DDTHH:MM:SS+09:00>\n"
    f"- 키워드: [<핵심 키워드 1개>]\n"
    f"- 원문: {{USER_QUERY}}\n\n"
    f"현재 KST: {to_iso_kst(now_kst())}\n"
    f"입력:\n{{USER_QUERY}}"
)

def parse_light_block_to_schema(light_text: str) -> dict:
    raw = (light_text or "").strip()
    m_time = re.search(r"기간\(KST\)\s*:\s*([^~]+)~\s*([^\n]+)", raw)
    start_iso, end_iso = (m_time.group(1).strip(), m_time.group(2).strip()) if m_time else (None, None)
    m_kw = re.search(r"키워드\s*:\s*\[(.*?)\]", raw, flags=re.DOTALL)
    keywords = [p.strip() for p in re.split(r"\s*,\s*", m_kw.group(1)) if p.strip()] if m_kw else []
    # 실패 시 최근 7일 + 키워드 추정
    if not (start_iso and end_iso):
        end_dt = now_kst(); start_dt = now_kst() - timedelta(days=7)
        start_iso, end_iso = to_iso_kst(start_dt), to_iso_kst(end_dt)
    if not keywords:
        # 한글/영문/숫자 2자 이상 토큰 중 첫 토큰 추정
        tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", raw)
        keywords = [tokens[0]] if tokens else ["유튜브"]
    return {
        "start_iso": start_iso,
        "end_iso": end_iso,
        "keywords": [keywords[0]],
        "raw": raw
    }

# ======================================================================
# 메타/다운로드 + 시각화 렌더
# ======================================================================
@st.cache_data(ttl=600, show_spinner=False)
def compute_keyword_counter_from_file(csv_path: str, stopset_list: list[str], per_comment_cap: int = 200) -> list[tuple[str,int]]:
    if not csv_path or not os.path.exists(csv_path): return []
    stopset = set(stopset_list)
    counter = {}
    for chunk in pd.read_csv(csv_path, usecols=["text"], chunksize=100_000):
        texts = (chunk["text"].astype(str).str.slice(0, per_comment_cap)).tolist()
        if not texts: continue
        tokens = kiwi.tokenize(" ".join(texts), normalize_coda=True)
        words = [t.form for t in tokens if t.tag in ("NNG","NNP") and len(t.form) > 1 and t.form not in stopset]
        for w in words:
            counter[w] = counter.get(w, 0) + 1
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)[:300]

def keyword_bubble_figure_from_counter(counter_items: list[tuple[str,int]]) -> go.Figure | None:
    if not counter_items: return None
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
    df_kw["font_size"] = df_kw["scaled"].apply(lambda s: int(10 + (s - s_min) / max(s_max - s_min, 1) * 12))
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
    fig_kw.update_layout(title="Top30 키워드 버블", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=40, b=0))
    return fig_kw

def timeseries_from_file(csv_path: str):
    if not csv_path or not os.path.exists(csv_path): return None, None
    tmin = None; tmax = None
    for chunk in pd.read_csv(csv_path, usecols=["publishedAt"], chunksize=200_000):
        dt = pd.to_datetime(chunk["publishedAt"], errors="coerce", utc=True)
        if dt.notna().any():
            lo, hi = dt.min(), dt.max()
            tmin = lo if (tmin is None or (lo < tmin)) else tmin
            tmax = hi if (tmax is None or (hi > tmax)) else tmax
    if tmin is None or tmax is None: return None, None
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
        for k, v in vc.items(): counts[k] = counts.get(k, 0) + int(v)
    if not counts: return None
    s = pd.Series(counts).sort_values(ascending=False).head(topn)
    return s.reset_index().rename(columns={"index": "author", 0: "count"}).rename(columns={"count": "count"})

def render_quant_viz_from_paths(comments_csv_path: str, df_stats: pd.DataFrame, scope_label="(KST 기준)"):
    if not comments_csv_path or not os.path.exists(comments_csv_path): return
    st.markdown("### 📊 정량 요약")
    col1, col2 = st.columns(2)
    with col1:
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
                    items = compute_keyword_counter_from_file(comments_csv_path, list(stopset), per_comment_cap=200)
                fig = keyword_bubble_figure_from_counter(items)
                if fig is None:
                    st.info("표시할 키워드가 없습니다(불용어 제거 후 남은 단어 없음).")
                else:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"키워드 분석 불가: {e}")

    with col2:
        with st.container(border=True):
            st.subheader("② 시점별 댓글량 추이")
            ts, label = timeseries_from_file(comments_csv_path)
            if ts is not None:
                fig_ts = px.line(ts, x="bucket", y="count", markers=True, title=f"{label} 댓글량 추이 {scope_label}")
                st.plotly_chart(fig_ts, use_container_width=True)
            else:
                st.info("댓글 타임스탬프가 비어 있습니다.")

    if df_stats is not None and not df_stats.empty:
        col3, col4 = st.columns(2)
        with col3:
            with st.container(border=True):
                st.subheader("③ Top10 영상 댓글수")
                top_vids = df_stats.sort_values(by="commentCount", ascending=False).head(10).copy()
                top_vids["title_short"] = top_vids["title"].apply(lambda t: t[:20] + "…" if isinstance(t, str) and len(t) > 20 else t)
                fig_vids = px.bar(top_vids, x="commentCount", y="title_short",
                                  orientation="h", text="commentCount", title="Top10 영상 댓글수")
                st.plotly_chart(fig_vids, use_container_width=True)
        with col4:
            with st.container(border=True):
                st.subheader("④ 댓글 작성자 활동량 Top10")
                ta = top_authors_from_file(comments_csv_path, topn=10)
                if ta is not None and not ta.empty:
                    fig_auth = px.bar(ta, x="count", y="author", orientation="h", text="count", title="Top10 댓글 작성자 활동량")
                    st.plotly_chart(fig_auth, use_container_width=True)
                else:
                    st.info("작성자 데이터 없음")

    with st.container(border=True):
        st.subheader("⑤ 댓글 좋아요 Top10")
        best = []
        for chunk in pd.read_csv(comments_csv_path, usecols=["video_id","video_title","author","text","likeCount"], chunksize=200_000):
            chunk["likeCount"] = pd.to_numeric(chunk["likeCount"], errors="coerce").fillna(0).astype(int)
            best.append(chunk.sort_values("likeCount", ascending=False).head(10))
        if best:
            df_top = pd.concat(best).sort_values("likeCount", ascending=False).head(10)
            for _, row in df_top.iterrows():
                url = f"https://www.youtube.com/watch?v={row['video_id']}"
                st.markdown(
                    f"<div style='margin-bottom:15px;'>"
                    f"<b>{int(row['likeCount'])} 👍</b> — {row.get('author','')}<br>"
                    f"<span style='font-size:14px;'>▶️ <a href='{url}' target='_blank' style='color:black; text-decoration:none;'>"
                    f"{str(row.get('video_title','(제목없음)'))[:60]}</a></span><br>"
                    f"> {str(row.get('text',''))[:150]}{'…' if len(str(row.get('text','')))>150 else ''}"
                    f"</div>", unsafe_allow_html=True
                )

def render_metadata_and_downloads():
    schema = st.session_state.get("last_schema")
    if not schema: return
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

    # 🔽 메타 정보 바로 하단에 시각화 섹션 붙임
    csv_path = st.session_state.get("last_csv", "")
    df_stats = st.session_state.get("last_df", pd.DataFrame())
    if csv_path and os.path.exists(csv_path):
        render_quant_viz_from_paths(csv_path, df_stats, scope_label="(KST 기준)")

def render_chat():
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ======================================================================
# URL ID 추출 / URL 제거
# ======================================================================
_YTB_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")
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
            if _YTB_ID_RE.fullmatch(v): ids.add(v)
        except Exception: pass
    return list(ids)

def strip_urls(s: str) -> str:
    if not s: return ""
    s = re.sub(r"https?://\S+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# ======================================================================
# 파이프라인(첫 턴/후속 턴)
# ======================================================================
def run_pipeline_first_turn(user_query: str, extra_video_ids=None, only_these_videos: bool = False):
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
    kw_main  = schema.get("keywords", [])

    # 영상 ID 구성 (메인 키워드만)
    if only_these_videos and extra_video_ids:
        all_ids = extra_video_ids
    else:
        all_ids = []
        for base_kw in (kw_main or ["유튜브"]):
            all_ids.extend(yt_search_videos(
                rt, base_kw, 60, "relevance",
                kst_to_rfc3339_utc(start_dt), kst_to_rfc3339_utc(end_dt)
            ))
        if extra_video_ids: all_ids.extend(extra_video_ids)
    all_ids = list(dict.fromkeys(all_ids))

    prog_bar.progress(0.40, text="댓글 수집 준비중…")
    df_stats = pd.DataFrame(yt_video_statistics(rt, all_ids))
    st.session_state["last_df"] = df_stats

    csv_path, total_cnt = parallel_collect_comments_streaming(
        df_stats.to_dict('records'),
        YT_API_KEYS,
        include_replies=False,
        max_total_comments=MAX_TOTAL_COMMENTS,
        max_per_video=MAX_COMMENTS_PER_VID,
        log_callback=None,
        prog_callback=lambda p: prog_bar.progress(min(0.90, 0.50 + 0.40 * p), text="댓글 수집중…")
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
        "분석의 핵심 관점을 파악하라. 그 다음, 주어진 댓글 샘플을 바탕으로 핵심 포인트를 항목화하고, "
        "긍/부/중 비율과 대표 코멘트(10개 미만)를 제시하라. 반복 금지."
    )
    payload = (
        f"[사용자 원본 질문]: {user_query}\n\n"
        f"[키워드]: {', '.join(kw_main)}\n"
        f"[기간(KST)]: {schema['start_iso']} ~ {schema['end_iso']}\n\n"
        f"[댓글 샘플]:\n{sample_text}\n"
    )
    answer_md_raw = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, sys, payload)

    prog_bar.progress(1.0, text="완료")
    time.sleep(0.5)
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
        "1) 내용(정성)을 물으면 사례 중심 요약 + 대표 코멘트 1~3개. "
        "2) 수치(정량)를 물으면 숫자 중심 간결 응답. "
        "동문서답/반복 금지."
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
# 초기 화면 / 채팅 렌더
# ======================================================================
if not st.session_state.chat:
    st.markdown(
        """
<div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
            text-align:center; height:70vh;">
  <h1 style="font-size:3.5rem; font-weight:600;
             background:-webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570, #F2A60C);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    유튜브 댓글분석: AI 챗봇
  </h1>
  <p style="font-size:1.2rem; color:#4b5563;">관련영상 유튜브 댓글반응을 AI가 요약해줍니다</p>
  <div style="margin-top:3rem; padding:1rem 1.5rem; border:1px solid #e5e7eb; border-radius:12px;
              background-color:#fafafa; max-width:600px; text-align:left;">
    <h4 style="margin-bottom:1rem; font-weight:600;">⚠️ 사용 주의사항</h4>
    <ol style="padding-left:20px;">
      <li><strong>첫 질문 시</strong> 댓글 수집 및 AI 분석에 시간이 소요될 수 있습니다.</li>
      <li>한 세션에서는 <strong>하나의 주제</strong>만 진행해야 분석 정확도가 유지됩니다.</li>
      <li>첫 질문에는 <strong>기간</strong>을 적어주세요 (예: 최근 48시간 / 5월 1일부터).</li>
    </ol>
  </div>
</div>
""",
        unsafe_allow_html=True
    )
else:
    # 상단 메타 + 다운로드 + (바로 하단 시각화)
    render_metadata_and_downloads()
    # 채팅
    def render_chat_and_scroll():
        render_chat(); scroll_to_bottom()
    render_chat_and_scroll()

# 입력창
prompt = st.chat_input("예) 최근 24시간 태풍상사 반응 요약해줘 / 또는 영상 URL 붙여도 OK")
if prompt:
    # URL/자연어 분리
    url_ids = extract_video_ids_from_text(prompt)
    natural_text = strip_urls(prompt)
    has_urls = len(url_ids) > 0
    has_natural = len(natural_text) > 0

    st.session_state.chat.append({"role": "user", "content": prompt})
    # 첫 턴은 수집 파이프라인, 이후는 follow-up
    if not st.session_state.get("last_csv"):
        if has_urls and not has_natural:
            response = run_pipeline_first_turn(prompt, extra_video_ids=url_ids, only_these_videos=True)
        elif has_urls and has_natural:
            response = run_pipeline_first_turn(prompt, extra_video_ids=url_ids, only_these_videos=False)
        else:
            response = run_pipeline_first_turn(prompt)
    else:
        response = run_followup_turn(prompt)

    st.session_state.chat.append({"role": "assistant", "content": response})
    safe_rerun()
