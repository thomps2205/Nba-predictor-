# Sharp Board: NBA · NCAAB · NFL · NHL · Soccer · UFC predictor

A mobile-first, sportsbook-style Streamlit app. It projects spreads, totals, win probabilities and player
props for the NBA, men's college basketball, NFL and NHL. It also projects UFC fight cards, with win
probability and method of victory. All data is live; nothing is mocked.

## Data sources

ESPN's public JSON is the primary source for every sport:

| Data | Endpoint |
|---|---|
| Schedules, live scores, betting lines, rankings | `site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard` (`groups=50&limit=500` for D1 college basketball) |
| Team form (points for/against) | `…/teams/{id}/schedule` |
| Basketball possessions (pace) | `…/teams/{id}/statistics` |
| Rosters and headshots | `…/teams/{id}/roster` |
| Player game logs | `site.web.api.espn.com/apis/common/v3/sports/{sport}/{league}/athletes/{id}/gamelog` |
| Injuries (NBA, NFL, NHL) | `…/{league}/injuries` |
| UFC fight cards and fighter profiles | `…/mma/ufc/scoreboard`, `site.web.api.espn.com/apis/common/v3/sports/mma/athletes/{id}` |

Three outside sources add information ESPN doesn't have:

| Extra data | Source | Fallback |
|---|---|---|
| Live player prop lines | PrizePicks board (`api.prizepicks.com/projections`) | Underdog Fantasy (`api.underdogfantasy.com/beta/v5/over_under_lines`) |
| Official UFC fighter stats and win methods | UFCStats.com search + fighter-details pages | ESPN fighter stats |
| Latest player news | ESPN news feeds (`…/{league}/news`, league-wide and per team), matched by athlete ID | — |

Each extra source is cached for 10 minutes to 7 days and skipped for 30 minutes after a failure. When a prop has
a posted line, its edge and hit rate are measured against that line. Otherwise they are measured against the
player's last-5 average.

For the NBA only, `nba_api` (stats.nba.com advanced Pace/OffRtg/DefRtg) is an optional enrichment. It runs
inside a strict try/except with a 5-second timeout. If it fails, it is skipped for 6 hours and the
ESPN-derived ratings are used. The app no longer calls cdn.nba.com at all.

Every request sends browser-style headers and waits at least 0.6 s after the previous one. Responses are
cached as JSON in `./.cache`:

| Data | Cached for |
|---|---|
| Scoreboards | 2 min |
| Injuries | 15 min |
| Team schedules | 3 h |
| Game logs | 4 h |
| Rosters, team stats | 6 h |
| Fighters | 24 h |

Data that isn't cached yet loads behind a progress bar. If an endpoint fails, the last cached copy is
used. If nothing is cached, that card says so instead of the page crashing.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Interface

* **Sport switcher** across the top: NBA, NCAAB, NFL, NHL, UFC. For NCAAB there is also a Top 25 / All D1
  toggle; long slates are paged.
* **Matchups** shows team cards with the projected score, a win-probability bar, and edge badges against
  ESPN's market lines:
  * NBA, NCAAB, NFL: spread, total and moneyline.
  * NHL: puck line, total and moneyline.
* **Player Props** has a card per player with headshot, projection, model line, last-5 average, last-5
  chips, last-10 hit rate and an OVER/UNDER badge. Switching market or team redraws only the cards. Markets:
  * Basketball: PTS / REB / AST / PRA.
  * NFL: passing, rushing and receiving yards, and receptions.
  * NHL: goals (with anytime-goal %), assists, shots and points.
* **Parlays** appear as a tab in every sport, plus a **MIX** view in the sport switcher that combines today's
  legs from all five sports.
  * Each leg's hit chance blends the model (60%) with the market's no-vig probability (40%). Spreads, totals
    and moneylines come from ESPN's lines. Props use the PrizePicks / Underdog line when one is posted, otherwise the player's last-10 average rounded to a .5 as a stand-in line. Props load automatically for the next 4 games per sport (2 per sport in MIX), or every game with the toggle.
    UFC legs need an ESPN moneyline.
  * Parlays use at most one leg per game, and no game appears on more than one card. The cards are: safest 2-leg, best-value
    3-leg, longshot 4-leg and a Props + game lines 3-leg (MIX adds a 3-leg with every leg from a different sport).
  * Each card shows hit chance, estimated odds (ESPN moneylines, -110 spreads/totals, -115 pick'em props), fair
    odds and expected value.
* **Soccer** covers 35 competitions, grouped by region in the league selector:
  * **UEFA club:** Champions, Europa and Conference League.
  * **England:** Premier League, EFL Championship, FA Cup, Carabao Cup.
  * **Europe:** LaLiga, Copa del Rey, Bundesliga, 2. Bundesliga, DFB-Pokal, Serie A, Coppa Italia, Ligue 1,
    Eredivisie, Primeira Liga, Scottish Premiership, Süper Lig.
  * **Americas & global:** MLS, USL Championship, NWSL, Liga MX, Brasileirão, Liga Profesional, CONCACAF
    Champions Cup, Copa Libertadores, Saudi Pro League, J1.
  * **International:** World Cup, WC qualifying (UEFA / CONMEBOL / CONCACAF), UEFA Nations League,
    friendlies.

  How it works:
  * **Leagues:** "All Featured" queries the 18 top active competitions at once and groups matches by
    competition, with a badge on each card. "Choose leagues" opens the regional picker.
  * **Dates:** a Today / Next 3 Days / Weekend toggle. An empty day automatically shows the next 4 days.
  * **Model:** a Poisson goal model gives home / draw / away, projected score, total goals and
    both-teams-to-score. Edges are measured against ESPN's three-way odds and goal totals.
  * **Cup and international form:** when a cup tie or national team has fewer than 3 completed matches in
    that competition, the app merges in recent results from ESPN's all-competitions schedule
    (`soccer/all/teams/{id}/schedule`). It then tries the club's domestic leagues, or other international
    competitions for national teams.
  * **Parlays:** soccer legs appear in the Parlays tab and in MIX.
* **Injuries** (NBA, NFL, NHL) shows ESPN's report for every team on the slate.
* **UFC:**
  * **Fight Card** lists every bout from the main event down to the prelims, with records, odds, win
    probability, the pick and the predicted method (KO/TKO, Submission, Decision).
  * **Fight Breakdowns** shows a tale of the tape (age, height, reach, stance, striking and grappling rates),
    method-of-victory bars, and the model factors.

## Code layout

```
app.py            UI (sport switcher, tabs, cards)
engine/common.py  throttled HTTP, disk cache, parsing helpers
engine/espn.py    ESPN endpoints + tolerant parsers
engine/teams.py   team ratings + per-sport game models (and optional nba_api ratings)
engine/props.py   player prop models
engine/ufc.py     fight cards, fighter profiles, win-probability and method model
engine/ui.py      CSS + HTML helpers
```

Projections are estimates for entertainment and research, not betting advice.
