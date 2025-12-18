Featured Image ALT Tags Autofill for WordPress (with optional Yoast SEO autofill)

This Python script focuses on one practical accessibility and SEO improvement: missing ALT text on featured images.

What it does
1) Featured image ALT text (primary feature)

For each WordPress post:

Reads the featured image (media ID).

If the featured image ALT text is empty, it sets ALT text using:

default: Yoast focus keyword if available, otherwise the post title

It can also overwrite existing ALT text if you explicitly enable overwrite.

2) Yoast SEO fields (optional feature)

If enabled, the script can fill missing Yoast fields using DeepSeek:

focus keyword

SEO title

meta description

ALT-only mode does not require DeepSeek.

Requirements

Python 3.10+

WordPress REST API enabled

WordPress Application Password for a user that can edit posts and media

Optional: Yoast SEO (if you want to fill Yoast meta fields)

Optional: DeepSeek API key (only if Yoast autofill is enabled)

Python packages:

requests

urllib3

python-dotenv (optional, for local .env loading)

Setup
1) Install dependencies
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install requests urllib3 python-dotenv

2) Configure environment variables

You can export env vars, or create a local .env file (recommended for local runs). Never commit .env.

Required:

WP_BASE=https://example.com

WP_USER=your_wp_username

WP_APP_PASSWORD=your_wp_application_password

Optional, only if you enable Yoast autofill:

DEEPSEEK_API_KEY=your_deepseek_api_key

DEEPSEEK_BASE=https://api.deepseek.com

DEEPSEEK_MODEL=deepseek-chat

Run
Dry run first (recommended)
python wp_featured_image_alt_tags.py --alt-only --dry-run

ALT tags only (no DeepSeek)
python wp_featured_image_alt_tags.py --alt-only

ALT tags plus Yoast (requires DEEPSEEK_API_KEY)
python wp_featured_image_alt_tags.py --dry-run
python wp_featured_image_alt_tags.py

Confirm before writing per post
python wp_featured_image_alt_tags.py --alt-only --confirm-per-post

Overwrite existing ALT text (be careful)
python wp_featured_image_alt_tags.py --alt-only --alt-overwrite --dry-run
python wp_featured_image_alt_tags.py --alt-only --alt-overwrite

ALT behavior options

Choose ALT source:

--alt-from auto (default): keyword if available, else title

--alt-from keyword

--alt-from title

Limit ALT length in words (default 6):

--alt-max-words 6

Safety notes

Use a dedicated WordPress user and an Application Password you can revoke.

Test on a staging site first.

Start with --dry-run and --confirm-per-post if you are unsure.

If you enable --alt-overwrite, you can replace good manual ALT text. Use it only when you want uniform rules.

License

MIT

Example .env (for local use)
WP_BASE=https://example.com
WP_USER=your_wp_username
WP_APP_PASSWORD=your_wp_application_password

# Optional, only for Yoast autofill:
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# Optional tuning:
WP_STATUS=publish
ALT_FROM=auto
ALT_MAX_WORDS=6
