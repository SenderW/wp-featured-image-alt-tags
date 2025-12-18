#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Featured Image ALT Tags Autofill for WordPress (with optional Yoast SEO autofill)

Primary goal
- Ensure featured images have meaningful ALT text.

Secondary goal (optional)
- Fill missing Yoast SEO fields using DeepSeek: focus keyword, SEO title, meta description.

Security
- No credentials are stored in this file.
- Configure secrets via environment variables or a local .env file (not committed).

Typical usage
- ALT tags only (no DeepSeek required):
    python wp_featured_image_alt_tags.py --alt-only --dry-run
    python wp_featured_image_alt_tags.py --alt-only

- ALT tags plus Yoast (requires DEEPSEEK_API_KEY):
    python wp_featured_image_alt_tags.py --dry-run
    python wp_featured_image_alt_tags.py
"""

from __future__ import annotations

import os
import re
import json
import time
import base64
import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ================= Console colors =================
BLUE = "\033[94m"
RED = "\033[91m"
GREEN = "\033[32m"
GRAY = "\033[90m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def blue(s: str) -> str:
    return f"{BLUE}{s}{RESET}"


def ok(s: str) -> str:
    return f"{GREEN}{s}{RESET}"


def warn(s: str) -> str:
    return f"{YELLOW}{s}{RESET}"


def err(s: str) -> str:
    return f"{RED}{s}{RESET}"


# ================= Language heuristics =================
GERM_STOP = {
    "der", "die", "das", "und", "mit", "für", "im", "ein", "eine", "einem", "einer",
    "auf", "ist", "sind", "aus", "den", "dem", "des", "zur", "zum", "von", "bei", "auch"
}
ENG_STOP = {"the", "and", "with", "for", "in", "on", "to", "of", "is", "are", "from", "this", "that", "a", "an"}


def looks_german(text: str) -> bool:
    t = re.sub(r"[^A-Za-zÄÖÜäöüß\- ]", " ", text or "").lower()
    tokens = t.split()
    if not tokens:
        return False
    german_hits = sum(1 for w in tokens if w in GERM_STOP)
    english_hits = sum(1 for w in tokens if w in ENG_STOP)
    diacritics = bool(re.search(r"[äöüß]", t))
    return german_hits >= max(2, english_hits) or (diacritics and german_hits >= 1)


# ================= Text helpers =================
def html_to_text(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sanitize_text(v: str) -> str:
    v = v or ""
    v = re.sub(r'[|\\"“”‘’«»]+', "", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v


def normalize_for_llm(html: str, max_chars: int) -> str:
    return html_to_text(html)[:max_chars]


def smart_shorten_sentence(text: str, limit: int) -> str:
    """
    Shorten without cutting in the middle of a word.

    Strategy:
    - remove bracketed content
    - drop trailing comma clauses
    - drop trailing conjunction clause
    - final fallback: cut at last space before the limit
    """
    t = sanitize_text(text)
    if len(t) <= limit:
        if not t.endswith((".", "!", "?")) and len(t) + 1 <= limit:
            t += "."
        return t

    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"\s+", " ", t).strip()

    while len(t) > limit and "," in t:
        parts = [p.strip() for p in t.split(",")]
        if len(parts) <= 1:
            break
        parts = parts[:-1]
        t = ", ".join(parts)

    for conj in (" und ", " and "):
        while len(t) > limit and conj in t:
            t = t.rsplit(conj, 1)[0].strip()

    if len(t) > limit:
        cut = t[:limit]
        if " " in cut:
            cut = cut[:cut.rfind(" ")]
        t = cut.rstrip(" ,;:")

    if not t.endswith((".", "!", "?")) and len(t) + 1 <= limit:
        t += "."
    return t


def build_alt_text(alt_from: str, keyword: str, title: str, max_words: int) -> str:
    """
    Build ALT text that is short and descriptive.

    alt_from:
    - auto: keyword if available, else title (default)
    - keyword: always use keyword
    - title: always use title
    """
    src = (alt_from or "auto").lower().strip()
    if src not in {"auto", "keyword", "title"}:
        src = "auto"

    kw = sanitize_text(keyword)
    ti = sanitize_text(title)

    if src == "keyword":
        base = kw
    elif src == "title":
        base = ti
    else:
        base = kw if kw else ti

    if not base:
        return ""

    words = base.split()
    if max_words > 0 and len(words) > max_words:
        base = " ".join(words[:max_words])

    return base


# ================= Configuration =================
@dataclass(frozen=True)
class Config:
    # WordPress
    wp_base: str
    wp_user: str
    wp_app_password: str

    # DeepSeek (only needed when Yoast autofill is enabled)
    deepseek_api_key: str
    deepseek_base: str
    deepseek_model: str

    # Run settings
    status: str
    per_page: int
    dry_run: bool
    confirm_per_post: bool
    confirm_between_batches: bool

    # Tracking
    processed_log: str

    # LLM sizing
    excerpt_max_chars: int
    max_meta: int
    max_title: int
    language_mode: str  # auto | de | en

    # ALT behavior
    alt_from: str  # auto | keyword | title
    alt_max_words: int
    alt_overwrite: bool

    # Feature toggles
    yoast_enabled: bool


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip()


def load_dotenv_if_present() -> None:
    """
    Optional support for a local .env file.
    Install python-dotenv if you want this convenience:
        pip install python-dotenv
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


def require(value: Optional[str], name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"Missing required setting: {name}")
    return value.strip()


def build_config(args: argparse.Namespace) -> Config:
    load_dotenv_if_present()

    wp_base = (args.wp_base or env("WP_BASE")).rstrip("/")
    wp_user = args.wp_user or env("WP_USER")
    wp_app_password = args.wp_app_password or env("WP_APP_PASSWORD")

    # DeepSeek is optional in ALT-only mode
    deepseek_api_key = args.deepseek_api_key or env("DEEPSEEK_API_KEY", "")
    deepseek_base = (args.deepseek_base or env("DEEPSEEK_BASE", "https://api.deepseek.com")).rstrip("/")
    deepseek_model = args.deepseek_model or env("DEEPSEEK_MODEL", "deepseek-chat")

    status = (args.status or env("WP_STATUS", "publish")).strip()
    per_page = int(args.per_page)

    processed_log = args.processed_log or env("PROCESSED_LOG", "processed_post_ids.txt")

    excerpt_max_chars = int(args.excerpt_max_chars or env("EXCERPT_MAX", "1800"))
    max_meta = int(args.max_meta or env("MAX_META", "152"))
    max_title = int(args.max_title or env("MAX_TITLE", "60"))

    language_mode = (args.language_mode or env("LANGUAGE_MODE", "auto")).lower().strip()
    if language_mode not in {"auto", "de", "en"}:
        raise ValueError("LANGUAGE_MODE must be one of: auto, de, en")

    alt_from = (args.alt_from or env("ALT_FROM", "auto")).lower().strip()
    if alt_from not in {"auto", "keyword", "title"}:
        raise ValueError("ALT_FROM must be one of: auto, keyword, title")
    alt_max_words = int(args.alt_max_words or env("ALT_MAX_WORDS", "6"))
    alt_overwrite = bool(args.alt_overwrite)

    yoast_enabled = not bool(args.alt_only)
    if env("YOAST_ENABLED") is not None:
        yoast_enabled = env("YOAST_ENABLED", "true").lower() in {"1", "true", "yes", "y"}

    if yoast_enabled:
        deepseek_api_key = require(deepseek_api_key, "DEEPSEEK_API_KEY")

    return Config(
        wp_base=require(wp_base, "WP_BASE"),
        wp_user=require(wp_user, "WP_USER"),
        wp_app_password=require(wp_app_password, "WP_APP_PASSWORD"),

        deepseek_api_key=deepseek_api_key or "",
        deepseek_base=deepseek_base,
        deepseek_model=deepseek_model,

        status=status,
        per_page=per_page,
        dry_run=bool(args.dry_run),
        confirm_per_post=bool(args.confirm_per_post),
        confirm_between_batches=bool(args.confirm_between_batches),

        processed_log=processed_log,

        excerpt_max_chars=excerpt_max_chars,
        max_meta=max_meta,
        max_title=max_title,
        language_mode=language_mode,

        alt_from=alt_from,
        alt_max_words=alt_max_words,
        alt_overwrite=alt_overwrite,

        yoast_enabled=yoast_enabled,
    )


# ================= HTTP session =================
def make_session(cfg: Config) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.8,
        status_forcelist=[408, 409, 429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST", "PUT", "PATCH"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))

    token = base64.b64encode(f"{cfg.wp_user}:{cfg.wp_app_password}".encode()).decode()
    s.headers.update({
        "Authorization": "Basic " + token,
        "Content-Type": "application/json",
        "User-Agent": "wp-featured-image-alt-tags/1.0",
    })
    return s


# ================= DeepSeek =================
def deepseek_chat_json(
    cfg: Config,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    timeout: int = 120,
    max_attempts: int = 3
) -> Dict[str, Any]:
    """
    Call DeepSeek chat completions and expect strict JSON in the response content.
    Returns {} on failure.
    """
    url = f"{cfg.deepseek_base}/v1/chat/completions"
    payload = {
        "model": cfg.deepseek_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {cfg.deepseek_api_key}", "Content-Type": "application/json"}

    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code != 200:
                print(warn(f"DeepSeek HTTP {r.status_code} (attempt {attempt}/{max_attempts})"))
                time.sleep(min(1.5 * attempt, 6.0))
                continue

            data = r.json()
            content = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()

            if content:
                preview = content.replace("\n", " ")[:180]
                print(GRAY + f"[DeepSeek preview<=180] {preview}" + RESET)

            c = content.strip().strip("`").strip()
            if c.lower().startswith("json"):
                c = c[4:].strip()

            try:
                return json.loads(c)
            except Exception as je:
                print(warn(f"DeepSeek JSON parse error (attempt {attempt}): {je}"))
                time.sleep(min(1.5 * attempt, 6.0))
                continue

        except requests.exceptions.Timeout:
            print(warn(f"DeepSeek timeout (attempt {attempt}/{max_attempts})"))
            time.sleep(min(1.5 * attempt, 6.0))
        except Exception as e:
            print(warn(f"DeepSeek error (attempt {attempt}/{max_attempts}): {e}"))
            time.sleep(min(1.5 * attempt, 6.0))

    return {}


def select_language(cfg: Config, title: str, excerpt: str) -> str:
    if cfg.language_mode in {"de", "en"}:
        return cfg.language_mode
    sample = f"{title} {excerpt}"
    return "de" if looks_german(sample) else "en"


def sys_prompt_keyword_title(lang: str, max_title: int) -> str:
    if lang == "de":
        return (
            "Reply in German only. Return JSON only.\n"
            "Generate {\"keyword\":\"...\",\"seo_title\":\"...\"}.\n"
            "- keyword: max 3 words, specific, not a broad generic category.\n"
            f"- seo_title: max {max_title} characters, natural German, do not end with a stopword.\n"
            "No markdown, no commentary."
        )
    return (
        "Reply in English only. Return JSON only.\n"
        "Generate {\"keyword\":\"...\",\"seo_title\":\"...\"}.\n"
        "- keyword: max 3 words, specific, not a broad generic category.\n"
        f"- seo_title: max {max_title} characters, natural English, do not end with a stopword.\n"
        "No markdown, no commentary."
    )


def sys_prompt_meta(lang: str, max_meta: int) -> str:
    if lang == "de":
        return (
            "Reply in German only. Return JSON only.\n"
            f"Generate {{\"meta_description\":\"...\"}} as one complete short sentence (max {max_meta} chars). "
            "It must include the keyword.\n"
            "No markdown, no commentary."
        )
    return (
        "Reply in English only. Return JSON only.\n"
        f"Generate {{\"meta_description\":\"...\"}} as one complete short sentence (max {max_meta} chars). "
        "It must include the keyword.\n"
        "No markdown, no commentary."
    )


def user_prompt(title: str, excerpt: str, keyword: Optional[str] = None) -> str:
    data: Dict[str, Any] = {"title": sanitize_text(title), "excerpt": sanitize_text(excerpt)}
    if keyword is not None:
        data["keyword"] = sanitize_text(keyword)
    return json.dumps(data, ensure_ascii=False)


def clamp_keyword(kw: str) -> str:
    kw = sanitize_text(kw)
    if not kw:
        return ""
    parts = kw.split()
    if len(parts) > 3:
        kw = " ".join(parts[:3])
    return kw


def clamp_title(title: str, max_len: int) -> str:
    t = sanitize_text(title)
    if len(t) > max_len:
        t = t[:max_len].rstrip()
    return t


def clamp_meta(meta: str, max_len: int) -> str:
    return smart_shorten_sentence(meta, max_len)


def generate_keyword_and_title(cfg: Config, lang: str, title: str, excerpt: str) -> Tuple[str, str]:
    sys_p = sys_prompt_keyword_title(lang, cfg.max_title)
    data = deepseek_chat_json(cfg, sys_p, user_prompt(title, excerpt), temperature=0.0, timeout=120, max_attempts=3)
    kw = clamp_keyword(str(data.get("keyword", "")))
    ti = clamp_title(str(data.get("seo_title", "")), cfg.max_title)
    return kw, ti


def generate_meta(cfg: Config, lang: str, title: str, excerpt: str, keyword: str) -> str:
    if not keyword:
        return ""
    for target in (cfg.max_meta, cfg.max_meta - 4, cfg.max_meta - 12):
        sys_p = sys_prompt_meta(lang, target)
        data = deepseek_chat_json(cfg, sys_p, user_prompt(title, excerpt, keyword), temperature=0.0, timeout=120, max_attempts=3)
        md = sanitize_text(str(data.get("meta_description", "")))
        if not md:
            continue
        md = clamp_meta(md, cfg.max_meta)
        if md:
            return md
    return ""


# ================= WordPress REST helpers =================
def wp_posts_url(cfg: Config) -> str:
    return f"{cfg.wp_base}/wp-json/wp/v2/posts"


def wp_media_url(cfg: Config) -> str:
    return f"{cfg.wp_base}/wp-json/wp/v2/media"


def wp_get_media_alt(sess: requests.Session, cfg: Config, media_id: int) -> str:
    try:
        r = sess.get(f"{wp_media_url(cfg)}/{media_id}", params={"_fields": "id,alt_text"}, timeout=30)
        if r.status_code == 200:
            return r.json().get("alt_text") or ""
        print(warn(f"[media {media_id}] GET {r.status_code}"))
    except Exception as e:
        print(err(f"[media {media_id}] GET error: {e}"))
    return ""


def wp_update_media_alt(sess: requests.Session, cfg: Config, media_id: int, alt_text: str) -> bool:
    if cfg.dry_run:
        print(GRAY + f"[dry-run] Would set media ALT for {media_id} to: {alt_text}" + RESET)
        return True
    try:
        r = sess.post(f"{wp_media_url(cfg)}/{media_id}", json={"alt_text": alt_text}, timeout=30)
        if r.status_code in (200, 201):
            return True
        print(warn(f"[media {media_id}] ALT update status {r.status_code}"))
    except Exception as e:
        print(err(f"[media {media_id}] ALT update error: {e}"))
    return False


def wp_update_yoast(sess: requests.Session, cfg: Config, post_id: int,
                    focuskw: Optional[str], metadesc: Optional[str], seotitle: Optional[str]) -> bool:
    payload_meta: Dict[str, Any] = {}
    if focuskw is not None:
        payload_meta["_yoast_wpseo_focuskw"] = focuskw
    if metadesc is not None:
        payload_meta["_yoast_wpseo_metadesc"] = clamp_meta(metadesc, cfg.max_meta)
    if seotitle is not None:
        payload_meta["_yoast_wpseo_title"] = clamp_title(seotitle, cfg.max_title)

    if not payload_meta:
        return True

    if cfg.dry_run:
        print(GRAY + f"[dry-run] Would update post {post_id} Yoast meta: {json.dumps(payload_meta, ensure_ascii=False)}" + RESET)
        return True

    r = sess.post(f"{wp_posts_url(cfg)}/{post_id}", json={"meta": payload_meta}, timeout=60)
    return r.status_code in (200, 201)


# ================= Processed IDs log =================
def load_processed_ids(path: str) -> set[int]:
    out: set[int] = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    out.add(int(line))
    return out


def append_processed_id(path: str, post_id: int) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(str(post_id) + "\n")


# ================= Main processing =================
def process_all(sess: requests.Session, cfg: Config) -> None:
    processed = 0
    page = 1
    seen = load_processed_ids(cfg.processed_log)

    while True:
        params = {
            "per_page": cfg.per_page,
            "page": page,
            "status": cfg.status,
            "orderby": "date",
            "order": "desc",
            "_fields": "id,title,content,meta,link,featured_media",
        }
        r = sess.get(wp_posts_url(cfg), params=params, timeout=60)

        if r.status_code == 400 and "rest_post_invalid_page_number" in r.text:
            break

        r.raise_for_status()
        posts = r.json()
        if not posts:
            break

        print(blue(f"Page {page} | Posts: {len(posts)}"))

        for p in posts:
            pid = int(p["id"])
            title_html = (p.get("title") or {}).get("rendered") or ""
            title = html_to_text(title_html)
            link = p.get("link") or f"{cfg.wp_base}/?p={pid}"
            content_html = (p.get("content") or {}).get("rendered") or ""
            excerpt = normalize_for_llm(content_html, cfg.excerpt_max_chars)

            fm_id = int(p.get("featured_media") or 0)
            meta = p.get("meta") or {}

            old_kw = (meta.get("_yoast_wpseo_focuskw") or "").strip()
            old_ti = (meta.get("_yoast_wpseo_title") or "").strip()
            old_md = (meta.get("_yoast_wpseo_metadesc") or "").strip()

            old_alt = wp_get_media_alt(sess, cfg, fm_id) if fm_id else ""

            lang = select_language(cfg, title, excerpt)

            # Yoast generation (optional)
            need_kw = cfg.yoast_enabled and (not bool(old_kw))
            need_ti = cfg.yoast_enabled and (not bool(old_ti))
            need_md = cfg.yoast_enabled and (not bool(old_md))

            gen_kw = gen_ti = gen_md = ""

            if cfg.yoast_enabled and (need_kw or need_ti):
                try:
                    gen_kw, gen_ti = generate_keyword_and_title(cfg, lang, title, excerpt)
                except Exception as e:
                    print(err(f"[{pid}] DeepSeek error (keyword/title): {e}"))
                    need_kw = False
                    need_ti = False

                if need_kw and not gen_kw:
                    print(warn("No keyword generated. Skipping keyword update."))
                    need_kw = False
                if need_ti and not gen_ti:
                    print(warn("No SEO title generated. Skipping title update."))
                    need_ti = False

            if cfg.yoast_enabled and need_md:
                base_kw = gen_kw or old_kw
                try:
                    gen_md = generate_meta(cfg, lang, title, excerpt, base_kw)
                except Exception as e:
                    print(err(f"[{pid}] DeepSeek error (meta): {e}"))
                    need_md = False

                if need_md and not gen_md:
                    print(warn("No meta description generated. Skipping meta update."))
                    need_md = False

            want_kw = gen_kw if need_kw else old_kw
            want_ti = gen_ti if need_ti else old_ti
            want_md = gen_md if need_md else old_md

            # ALT candidate is built from final keyword decision
            candidate_alt = build_alt_text(cfg.alt_from, want_kw, title, cfg.alt_max_words) if fm_id else ""

            alt_can_update = bool(fm_id) and bool(candidate_alt)
            if cfg.alt_overwrite:
                alt_needs_update = alt_can_update and sanitize_text(old_alt) != sanitize_text(candidate_alt)
            else:
                alt_needs_update = alt_can_update and (not bool(old_alt.strip()))

            yoast_change_needed = any([
                need_kw and want_kw != old_kw,
                need_ti and want_ti != old_ti,
                need_md and want_md != old_md,
            ])

            overall_change = alt_needs_update or yoast_change_needed

            if pid in seen and not overall_change:
                print(blue(title))
                print(blue("Skipped: already processed and no changes are needed."))
                print(blue("URL:"), blue(link))
                print()
                processed += 1
                time.sleep(0.05)
                continue

            # Preview (ALT first)
            print(blue(title))
            print(blue("Language:"), blue(lang))
            print(blue("ALT is primary:"), ok("yes"))
            if fm_id:
                print(blue("Featured image ID:"), blue(str(fm_id)))
                print(blue("Old ALT:"), GRAY + (old_alt or "(empty)") + RESET)
                print(blue("New ALT:"), (ok(candidate_alt) if alt_needs_update else GRAY + "(no change)" + RESET))
            else:
                print(blue("Featured image:"), warn("none"))

            if cfg.yoast_enabled:
                print(blue("Yoast changes:"), ok("yes") if yoast_change_needed else blue("no"))
                print(blue("Old keyword:"), GRAY + old_kw + RESET)
                print(blue("New keyword:"), (ok(want_kw) if need_kw else GRAY + "(skip)" + RESET))
                print(blue("Old Yoast title:"), GRAY + old_ti + RESET)
                print(blue("New Yoast title:"), (ok(want_ti) if need_ti else GRAY + "(skip)" + RESET))
                print(blue("Old meta desc:"), GRAY + old_md + RESET)
                print(blue("New meta desc:"), (ok(want_md) if need_md else GRAY + "(skip)" + RESET))
            else:
                print(blue("Yoast mode:"), warn("disabled (ALT tags only)"))

            print(blue("URL:"), blue(link))

            do_apply = True
            if overall_change and cfg.confirm_per_post:
                try:
                    ans = input("Apply changes? (Y/N): ").strip().lower()
                except EOFError:
                    ans = ""
                do_apply = (ans == "" or ans.startswith("y"))

            if overall_change and do_apply:
                # ALT update first
                if alt_needs_update and fm_id:
                    if wp_update_media_alt(sess, cfg, fm_id, candidate_alt):
                        print(ok(f"[media {fm_id}] ALT saved."))
                    else:
                        print(warn(f"[media {fm_id}] ALT not saved (HTTP error)."))

                # Yoast update second
                if cfg.yoast_enabled and yoast_change_needed:
                    upd_kw = want_kw if (need_kw and want_kw != old_kw) else None
                    upd_ti = want_ti if (need_ti and want_ti != old_ti) else None
                    upd_md = want_md if (need_md and want_md != old_md) else None

                    if wp_update_yoast(sess, cfg, pid, upd_kw, upd_md, upd_ti):
                        print(ok(f"[{pid}] Yoast fields saved."))
                    else:
                        print(warn(f"[{pid}] Yoast fields not saved (HTTP error)."))

            # Mark as processed if requirements are satisfied
            alt_ok = True
            if fm_id:
                if cfg.alt_overwrite:
                    alt_ok = bool((candidate_alt or old_alt).strip())
                else:
                    alt_ok = bool((old_alt or candidate_alt).strip())

            yoast_ok = True
            if cfg.yoast_enabled:
                final_kw = want_kw if need_kw else old_kw
                final_ti = want_ti if need_ti else old_ti
                final_md = want_md if need_md else old_md
                yoast_ok = bool(final_kw.strip()) and bool(final_ti.strip()) and bool(final_md.strip())

            if alt_ok and yoast_ok:
                append_processed_id(cfg.processed_log, pid)

            print()
            processed += 1
            time.sleep(0.05)

        page += 1
        time.sleep(0.2)

        if cfg.confirm_between_batches and processed % 100 == 0:
            try:
                ans = input("Continue? Enter=continue, P=pause, Q=quit: ").strip().lower()
            except EOFError:
                ans = ""
            if ans.startswith("q"):
                break
            if ans.startswith("p"):
                input("Paused. Press Enter to continue...")

    print(ok(f"Done. Processed posts: {processed}"))


# ================= CLI =================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autofill WordPress featured image ALT tags (primary) and optionally missing Yoast SEO fields."
    )

    # WordPress
    p.add_argument("--wp-base", default=None, help="WordPress base URL, e.g. https://example.com (or env WP_BASE)")
    p.add_argument("--wp-user", default=None, help="WordPress username (or env WP_USER)")
    p.add_argument("--wp-app-password", default=None, help="WordPress application password (or env WP_APP_PASSWORD)")

    # DeepSeek (only needed when Yoast autofill is enabled)
    p.add_argument("--deepseek-api-key", default=None, help="DeepSeek API key (or env DEEPSEEK_API_KEY)")
    p.add_argument("--deepseek-base", default=None, help="DeepSeek base URL (default https://api.deepseek.com)")
    p.add_argument("--deepseek-model", default=None, help="DeepSeek model (default deepseek-chat)")

    # Query and safety
    p.add_argument("--status", default=None, help="WordPress post status (default publish)")
    p.add_argument("--per-page", type=int, default=100, help="Posts per page to fetch (max is typically 100)")
    p.add_argument("--confirm-per-post", action="store_true", help="Ask before applying changes per post")
    p.add_argument("--confirm-between-batches", action="store_true", help="Pause occasionally during long runs")
    p.add_argument("--dry-run", action="store_true", help="Do not write changes to WordPress")
    p.add_argument("--processed-log", default=None, help="File to store processed post IDs")
    p.add_argument("--excerpt-max-chars", type=int, default=None, help="Max chars from post content sent to the model")

    # Yoast
    p.add_argument("--alt-only", action="store_true", help="Only update featured image ALT tags, skip Yoast entirely")
    p.add_argument("--max-meta", type=int, default=None, help="Meta description max length (default 152)")
    p.add_argument("--max-title", type=int, default=None, help="SEO title max length (default 60)")
    p.add_argument("--language-mode", default=None, help="auto, de, or en (default auto)")

    # ALT options
    p.add_argument("--alt-from", default=None, help="auto, keyword, or title (default auto)")
    p.add_argument("--alt-max-words", type=int, default=None, help="Max words for ALT text (default 6)")
    p.add_argument("--alt-overwrite", action="store_true", help="Overwrite existing ALT text (default: only fill empty ALT)")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        cfg = build_config(args)
    except Exception as e:
        print(err(str(e)))
        print("Tip: create a local .env file or export environment variables.")
        return 2

    sess = make_session(cfg)

    try:
        process_all(sess, cfg)
    except KeyboardInterrupt:
        print(warn("\nInterrupted."))
        return 130
    except Exception as e:
        print(err(f"Fatal error: {e}"))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
