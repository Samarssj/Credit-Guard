"""Generate self-hosted GitHub profile analytics SVGs."""

from __future__ import annotations

import html
import json
import os
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

USERNAME = "Samarssj"
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
API = "https://api.github.com"
TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()


PALETTE = {
    "ink": "#0b1020",
    "panel": "#111936",
    "panel_2": "#172044",
    "text": "#f8fafc",
    "muted": "#94a3b8",
    "line": "#26345c",
    "purple": "#a78bfa",
    "cyan": "#22d3ee",
    "pink": "#f472b6",
    "orange": "#fb923c",
}

LANGUAGE_COLORS = {
    "Python": "#3776ab",
    "JavaScript": "#f1e05a",
    "TypeScript": "#8b5cf6",
    "Java": "#b07219",
    "HTML": "#e34c26",
    "CSS": "#563d7c",
    "Jupyter Notebook": "#da5b0b",
    "Shell": "#89e051",
    "C++": "#f34b7d",
    "Go": "#00add8",
    "Kotlin": "#a97bff",
}


def github_request(path: str, payload: dict | None = None) -> dict:
    body = None
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Samarssj-profile-analytics",
    }
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        request = urllib.request.Request(f"{API}{path}", data=body, headers=headers, method="POST")
    else:
        request = urllib.request.Request(f"{API}{path}", headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(
    x: int | float,
    y: int | float,
    value: object,
    size: int = 16,
    color: str = PALETTE["text"],
    weight: str = "400",
    anchor: str = "start",
    letter_spacing: int = 0,
) -> str:
    spacing = f' letter-spacing="{letter_spacing}px"' if letter_spacing else ""
    return f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial, Helvetica, sans-serif" font-size="{size}px" font-weight="{weight}" text-anchor="{anchor}"{spacing}>{esc(value)}</text>'


def svg_document(width: int, height: int, body: str, title: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{esc(title)}</title>
  <desc id="desc">Live GitHub analytics for {USERNAME}, generated from GitHub API data.</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0b1020"/>
      <stop offset="52%" stop-color="#111936"/>
      <stop offset="100%" stop-color="#17102f"/>
    </linearGradient>
    <linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#a78bfa"/>
      <stop offset="52%" stop-color="#22d3ee"/>
      <stop offset="100%" stop-color="#f472b6"/>
    </linearGradient>
    <linearGradient id="bar" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#a78bfa"/>
      <stop offset="100%" stop-color="#22d3ee"/>
    </linearGradient>
    <linearGradient id="waveLine" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#a78bfa"/>
      <stop offset="52%" stop-color="#22d3ee"/>
      <stop offset="100%" stop-color="#f472b6"/>
    </linearGradient>
    <linearGradient id="waveFill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#22d3ee" stop-opacity="0.34"/>
      <stop offset="100%" stop-color="#a78bfa" stop-opacity="0.02"/>
    </linearGradient>
    <filter id="glow" x="-80%" y="-80%" width="260%" height="260%">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse">
      <path d="M 28 0 L 0 0 0 28" fill="none" stroke="#94a3b8" stroke-opacity="0.045" stroke-width="1"/>
    </pattern>
  </defs>
  <rect width="100%" height="100%" rx="22" fill="url(#bg)"/>
  <rect width="100%" height="100%" rx="22" fill="url(#grid)"/>
  <rect x="0" y="0" width="100%" height="5" rx="3" fill="url(#accent)"/>
  {body}
</svg>
'''


def fmt(value: int) -> str:
    return f"{value:,}"


def card(x: int, y: int, width: int, height: int, label: str, value: int, accent: str) -> list[str]:
    return [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="16" fill="{PALETTE["panel"]}" fill-opacity="0.84" stroke="{PALETTE["line"]}"/>',
        f'<rect x="{x}" y="{y}" width="5" height="{height}" rx="3" fill="{accent}"/>',
        f'<circle cx="{x + 26}" cy="{y + 25}" r="5" fill="{accent}" filter="url(#glow)"/>',
        text(x + 42, y + 31, label.upper(), 13, PALETTE["muted"], "700", letter_spacing=1),
        text(x + 24, y + 58, fmt(value), 27, PALETTE["text"], "700"),
    ]


def generate_stats(repos: list[dict], contributions: dict) -> None:
    stars = sum(repo.get("stargazers_count", 0) for repo in repos)
    forks = sum(repo.get("forks_count", 0) for repo in repos)
    metrics = [
        ("Repositories", len(repos), PALETTE["purple"]),
        ("Total stars", stars, PALETTE["orange"]),
        ("Total forks", forks, PALETTE["cyan"]),
        ("Commits · 1 year", contributions["totalCommitContributions"], PALETTE["pink"]),
        ("Pull requests", contributions["totalPullRequestContributions"], PALETTE["cyan"]),
        ("Code reviews", contributions["totalPullRequestReviewContributions"], PALETTE["purple"]),
    ]
    body = [
        text(28, 44, "GitHub Pulse", 30, PALETTE["text"], "700"),
        text(28, 72, f"{USERNAME}  /  live profile snapshot", 15, PALETTE["muted"]),
        '<rect x="548" y="28" width="144" height="28" rx="14" fill="#12363d" stroke="#1e6b73"/>',
        '<circle cx="566" cy="42" r="4" fill="#22d3ee" filter="url(#glow)"/>',
        text(578, 48, "SYNCED", 13, PALETTE["cyan"], "700", letter_spacing=1),
    ]
    for index, (label, value, accent) in enumerate(metrics):
        col = index % 3
        row = index // 3
        body.extend(card(28 + col * 228, 94 + row * 78, 210, 64, label, value, accent))
    updated = datetime.now(timezone.utc).strftime("%d %b %Y  ·  %H:%M UTC")
    body.extend([
        text(28, 272, f"LAST SYNC  {updated}", 12, PALETTE["muted"], "700", letter_spacing=1),
        text(692, 272, "AUTO-REFRESH  /  EVERY 6 HOURS", 12, PALETTE["muted"], "700", "end", 1),
    ])
    (ASSETS / "github-stats.svg").write_text(svg_document(720, 292, "\n  ".join(body), "GitHub Pulse statistics for Samar Singh"), encoding="utf-8")


def generate_languages(repos: list[dict]) -> None:
    counts = Counter(repo.get("language") for repo in repos if repo.get("language"))
    items = counts.most_common(6)
    total = max(sum(count for _, count in items), 1)
    center_x, center_y, radius = 175, 190, 92
    circumference = 2 * 3.141592653589793 * radius
    body = [
        text(28, 44, "Repository DNA", 28, PALETTE["text"], "700"),
        text(28, 71, "Language mix across active repositories", 15, PALETTE["muted"]),
        text(690, 50, fmt(len(repos)), 28, PALETTE["cyan"], "700", "end"),
        text(690, 72, "PUBLIC REPOS", 11, PALETTE["muted"], "700", "end", 1),
        f'<circle cx="{center_x}" cy="{center_y}" r="{radius}" fill="none" stroke="#202b4c" stroke-width="30"/>',
    ]
    offset = 0.0
    for index, (language, count) in enumerate(items):
        segment = circumference * count / total
        color = LANGUAGE_COLORS.get(language, PALETTE["purple"])
        body.append(
            f'<circle cx="{center_x}" cy="{center_y}" r="{radius}" fill="none" stroke="{color}" stroke-width="30" '
            f'stroke-linecap="round" stroke-dasharray="{max(segment - 5, 0):.2f} {circumference:.2f}" '
            f'stroke-dashoffset="{-offset:.2f}" transform="rotate(-90 {center_x} {center_y})" filter="url(#glow)"><title>{esc(language)}: {count} repositories</title></circle>'
        )
        offset += segment
    body.extend([
        f'<circle cx="{center_x}" cy="{center_y}" r="60" fill="{PALETTE["panel"]}" stroke="{PALETTE["line"]}"/>',
        text(center_x, center_y - 3, fmt(len(repos)), 30, PALETTE["text"], "700", "middle"),
        text(center_x, center_y + 21, "REPOS", 11, PALETTE["muted"], "700", "middle", 1),
    ])
    for index, (language, count) in enumerate(items):
        y = 108 + index * 31
        percent = count / total
        color = LANGUAGE_COLORS.get(language, PALETTE["purple"])
        body.extend([
            f'<circle cx="360" cy="{y - 5}" r="6" fill="{color}"/>',
            text(378, y, language, 16, PALETTE["text"], "600"),
            text(690, y, f"{count} repo{'s' if count != 1 else ''}  ·  {percent:.0%}", 14, PALETTE["muted"], "600", "end"),
        ])
    body.extend([
        f'<rect x="28" y="312" width="664" height="2" rx="1" fill="url(#accent)" opacity="0.9"/>',
        text(28, 330, "Donut share  ·  ranked by repository count  ·  refreshed with GitHub", 11, PALETTE["muted"], "600"),
    ])
    (ASSETS / "github-languages.svg").write_text(svg_document(720, 342, "\n  ".join(body), "Repository language donut chart for Samar Singh"), encoding="utf-8")


def generate_contributions(contribution_data: dict) -> None:
    calendar = contribution_data["contributionCalendar"]
    weeks = calendar["weeks"][-53:]
    weekly_totals = [sum(day["contributionCount"] for day in week["contributionDays"]) for week in weeks]
    days = [day for week in weeks for day in week["contributionDays"]]
    total = calendar["totalContributions"]
    peak_week = max(weekly_totals, default=0)
    busiest = max(days, key=lambda day: day["contributionCount"], default={"date": "n/a", "contributionCount": 0})
    plot_left, plot_right = 72, 1142
    plot_top, baseline = 112, 264
    plot_width = plot_right - plot_left
    plot_height = 136
    maximum = max(weekly_totals, default=0)
    points: list[tuple[float, float]] = []
    for index, value in enumerate(weekly_totals):
        x = plot_left + (plot_width * index / max(len(weekly_totals) - 1, 1))
        y = baseline - (plot_height * value / max(maximum, 1))
        points.append((x, y))
    line_path = f"M {points[0][0]:.1f} {points[0][1]:.1f}"
    for index in range(1, len(points)):
        previous_x, previous_y = points[index - 1]
        current_x, current_y = points[index]
        distance = (current_x - previous_x) / 2
        line_path += f" C {previous_x + distance:.1f} {previous_y:.1f}, {current_x - distance:.1f} {current_y:.1f}, {current_x:.1f} {current_y:.1f}"
    area_path = f"{line_path} L {points[-1][0]:.1f} {baseline} L {points[0][0]:.1f} {baseline} Z"
    body = [
        text(28, 45, "Contribution Wave", 30, PALETTE["text"], "700"),
        text(28, 73, "Weekly momentum, shaped from your live GitHub activity calendar", 15, PALETTE["muted"]),
        f'<rect x="818" y="26" width="354" height="45" rx="14" fill="{PALETTE["panel"]}" stroke="{PALETTE["line"]}"/>',
        text(842, 45, "TOTAL", 12, PALETTE["muted"], "700", letter_spacing=1),
        text(842, 64, fmt(total), 20, PALETTE["cyan"], "700"),
        text(954, 45, "PEAK WEEK", 12, PALETTE["muted"], "700", letter_spacing=1),
        text(954, 64, fmt(peak_week), 20, PALETTE["pink"], "700"),
        text(1060, 45, "LAST 365 DAYS", 12, PALETTE["muted"], "700", letter_spacing=1),
        text(1060, 64, "LIVE", 20, PALETTE["purple"], "700"),
        f'<rect x="{plot_left - 18}" y="{plot_top - 24}" width="1106" height="184" rx="18" fill="#0a0f21" fill-opacity="0.72" stroke="{PALETTE["line"]}"/>',
        f'<line x1="{plot_left}" y1="{baseline}" x2="{plot_right}" y2="{baseline}" stroke="{PALETTE["line"]}"/>',
        f'<line x1="{plot_left}" y1="{plot_top + 68}" x2="{plot_right}" y2="{plot_top + 68}" stroke="{PALETTE["line"]}" stroke-dasharray="3 8" opacity="0.7"/>',
        f'<path d="{area_path}" fill="url(#waveFill)" opacity="0.78"/>',
        f'<path d="{line_path}" fill="none" stroke="url(#waveLine)" stroke-width="4" stroke-linecap="round" filter="url(#glow)"/>',
    ]
    for index, (x, y) in enumerate(points):
        if weekly_totals[index] > 0:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{PALETTE["cyan"]}" stroke="{PALETTE["ink"]}" stroke-width="2"><title>Week {index + 1}: {weekly_totals[index]} contributions</title></circle>')
    for index in range(0, len(weeks), 4):
        date = weeks[index]["contributionDays"][0]["date"]
        x = plot_left + (plot_width * index / max(len(weeks) - 1, 1))
        body.append(text(x, 292, datetime.strptime(date, "%Y-%m-%d").strftime("%b %Y"), 11, PALETTE["muted"], "600", "middle"))
    body.extend([
        text(22, plot_top + 5, fmt(maximum), 11, PALETTE["muted"], "600"),
        text(30, baseline + 4, "0", 11, PALETTE["muted"], "600"),
        text(72, 322, "LOW MOMENTUM", 11, PALETTE["muted"], "700", letter_spacing=1),
        text(1172, 322, f"PEAK DAY  {busiest['date']}  ·  {busiest['contributionCount']} contributions", 11, PALETTE["muted"], "600", "end"),
    ])
    (ASSETS / "github-contributions.svg").write_text(svg_document(1200, 342, "\n  ".join(body), "GitHub contribution wave for Samar Singh"), encoding="utf-8")


def main() -> None:
    if not TOKEN:
        raise SystemExit("GITHUB_TOKEN is required")
    repos = github_request(f"/users/{USERNAME}/repos?per_page=100&type=owner&sort=updated")
    repos = [repo for repo in repos if not repo.get("fork") and not repo.get("archived")]
    query = """
    query($login: String!) {
      user(login: $login) {
        contributionsCollection {
          totalCommitContributions
          totalPullRequestContributions
          totalIssueContributions
          totalPullRequestReviewContributions
          contributionCalendar {
            totalContributions
            weeks {
              contributionDays { contributionCount date }
            }
          }
        }
      }
    }
    """
    calendar = github_request("/graphql", {"query": query, "variables": {"login": USERNAME}})
    contribution_data = calendar["data"]["user"]["contributionsCollection"]
    ASSETS.mkdir(parents=True, exist_ok=True)
    generate_stats(repos, contribution_data)
    generate_languages(repos)
    generate_contributions(contribution_data)
    print(f"Generated analytics for {USERNAME}: {contribution_data['contributionCalendar']['totalContributions']} contributions, {len(repos)} repositories")


if __name__ == "__main__":
    main()
