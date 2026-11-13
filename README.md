# NBA Game & Player Prop Predictor

A Streamlit app that projects spreads, totals, win probabilities and player PTS / REB / AST / PRA lines
for any NBA slate. It uses only live data and no mock values.

| Data | Source |
|---|---|
| Daily schedule, live status/scores, market spread & total | ESPN public scoreboard JSON |
| Injury report | ESPN public injuries JSON |
| Back-to-back detection | ESPN scoreboard for the previous day |
| Team Pace / OffRtg / DefRtg, opponent REB/AST allowed | `nba_api` → `LeagueDashTeamStats` (Advanced + Opponent) |
| Player last-10 game logs & current rosters | `nba_api` → `LeagueGameLog` (players) |
| Fallback when stats.nba.com is unreachable | `nba_api` live box scores + season schedule on cdn.nba.com |
| Second fallback when cdn.nba.com is unreachable | ESPN scoreboard (completed games) + ESPN game summaries (box scores) |

All requests send browser-style headers, are spaced at least 0.6 s apart, retry on failure, and are cached
as JSON in `./.cache` (schedule 2 min, injuries 15 min, game logs 2 h, team stats 6 h). If a source is down,
the last cached copy is used and the app shows a warning.

stats.nba.com often silently drops requests from cloud hosts such as Streamlit Community Cloud. The app
waits at most 10 s, then skips stats.nba.com for 30 min. In that case it computes Pace, OffRtg, DefRtg,
opponent REB/AST and player game logs itself from the official box scores on cdn.nba.com. The first load
downloads every finished game, showing a progress bar for a few minutes. Those box scores are cached, so
later loads only fetch new games.

If cdn.nba.com fails too, the app builds the same numbers from ESPN game summaries. If every stats source
fails, matchups, market lines and injuries still render, and projections show as unavailable instead of
crashing the page.

## Interface

The page is a mobile-first dark layout with three tabs:

* **Game Picks** shows a matchup card per game: projected score, win-probability bar, and model spread, total
  and win % with edge badges against the market line.
* **Player Props** shows a card per rotation player for PTS / REB / AST / PRA: projection, model line, last-5
  average, last-5 chips, last-10 hit rate, and an OVER / UNDER badge (projection vs. last-5 average).
* **Injury Report** shows ESPN's injury list for every team on the slate, colour-coded by status, with
  each player's PPG.

Settings (date, season, questionable-sit probability, refresh) live in the collapsible panel at the top.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Use the sidebar to pick the slate date, season, season type, and how likely Questionable / Day-to-Day
players are to sit. **Refresh all data** clears the cache. When you view a past date, only game logs from
before that date are used.

## Model (summary)

* **Possessions** = Pace_home × Pace_away / league pace, minus 0.5 per team on a back-to-back.
* **Matchup ORtg** = OffRtg × opponent DefRtg / league rating, ±1.0 per 100 possessions for home court.
* **Injuries**: each missing player's scoring and minutes, weighted by how often they actually played,
  lower the team's OffRtg and raise its DefRtg.
* **Back-to-backs**: −1.2 OffRtg, +1.0 DefRtg, and fewer minutes / slightly lower efficiency for players.
* **Player props**: recency-weighted last-10 per-minute rates × projected minutes. Absent teammates' minutes
  are redistributed, and each player is adjusted for game pace, opponent DefRtg (PTS), opponent
  rebounds / assists allowed per possession (REB / AST), and a usage bump. Lines are rounded to the
  nearest .5 and come with the player's last-10 hit rate.

Projections are estimates for entertainment and research, not betting advice.
