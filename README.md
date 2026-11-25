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
| UFC fight cards | `…/mma/ufc/scoreboard` |
| UFC bio and record (age, height, reach, stance) | `sports.core.api.espn.com/v2/sports/mma/athletes/{id}` |
| UFC career stats (SLpM, SApM, accuracy, defense, TD avg/acc/def, sub avg) | `sports.core.api.espn.com/v2/sports/mma/athletes/{id}/statistics` |
| UFC fight history (method, round, time, opponent; drives finish rounds and recent form) | `sports.core.api.espn.com/v2/sports/mma/athletes/{id}/eventlog` (or `…/leagues/ufc/athletes/{id}/eventlog`). The newest 5 bouts are resolved through `site.web.api.espn.com/apis/common/v3/sports/mma/ufc/fightcenter/{event_id}`, cached 30 days and shared by everyone on that card, or else the core competition `$ref`. Fallbacks: `common/v3 …/fighthistory`, `…/gamelog`, `site.api …/athletes/{id}/gamelog` |
| UFC career method and loss totals (tkos, submissions, decisions, lossesByKo, lossesBySubmission) | `sports.core.api.espn.com/v2/sports/mma/athletes/{id}/records` |

Two outside sources add information ESPN doesn't have:

| Extra data | Source | Fallback |
|---|---|---|
| Live player prop lines + prices | ESPN sportsbook prop bets (`sports.core.api.espn.com/…/competitions/{id}/odds/{provider}/propBets`, matched by athlete id) | PrizePicks board, then Underdog Fantasy |
| Latest player news | ESPN news feeds (`…/{league}/news`, league-wide and per team), matched by athlete ID | — |

Each extra source is cached for 10 minutes to 7 days and skipped for 30 minutes after a failure. When a prop has
a posted line, its edge and hit rate are measured against that line. Otherwise they are measured against the
player's last-5 average.

For the NBA only, `nba_api` (stats.nba.com advanced Pace/OffRtg/DefRtg) is an optional enrichment. It runs
inside a strict try/except with a 5-second timeout. If it fails, it is skipped for 6 hours and the
ESPN-derived ratings are used. The app no longer calls cdn.nba.com at all.

Every request sends browser-style headers. ESPN requests run on a pool of 8 workers spaced 0.08 s apart, so one game's ~35 player game logs load in a few seconds instead of about 35. While you look at one game, the slate's other games warm up in the background. Every other host, such as stats.nba.com and the DFS boards, still waits at least 0.6 s between requests. Responses are
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
* **Top bar:** the SharpBoard mark and a sport dropdown pill (NBA, College Hoops, NFL, NHL, Soccer, UFC, Mixed
  Parlays) on a pure-black, thin-bordered card design.
* **Player Props pick list:** players ranked BET-first, each showing Over/Under, line, BET/PASS pill, edge
  toward the pick and a confidence ring. Pick any player to open a full report:
  * **Header:** breadcrumb, headshot, market vs opponent, and a big line number with L10 hit rate / Season /
    vs-opponent stats.
  * **Last-10 chart:** one bar per game (green over, pink under) against a dashed line marker, labelled with
    opponent and date.
  * **"Each split against the line":** season, last 10, vs tonight's opponent, tonight's venue,
    back-to-backs, and without the key injured teammate (games that teammate missed).
  * **Model report:**
    * Recommendation and confidence ring (hit chance, last-10 consistency and split agreement; PASS picks
      cap at 55).
    * Projection with an edge bar.
    * One-line summary and FORM / MATCHUP / RISK notes built from the numbers above.
    * BET/PASS strip, alt-line ladder and latest news.
* **Player Props** (legacy description) has a card per player with headshot, projection, model line, last-5 average, last-5
  chips, last-10 hit rate and an OVER/UNDER badge. Switching market or team redraws only the cards. Markets:
  * Basketball: PTS / REB / AST / PRA.
  * NFL: passing, rushing and receiving yards, and receptions.
  * NHL: goals (with anytime-goal %), assists, shots and points.
* **Prop results on finished games:** finished matchup cards get a "Prop results" drawer (BET record, e.g. 5-3, plus each pick with its line, projection, final stat and ✅ WON / ❌ LOST / ➖ PUSH). Picks are rebuilt as they stood pregame (projection from games before tip-off; posted line if still available, else the implied line) and graded against ESPN's final box score (game summary). The Player Props tab shows result chips and "Final X · WON" on the report. Grades the first 6 finished games per view (toggle).
* **Parlays only use games that haven't started** (pregame state and tip-off still in the future). A window toggle switches between the selected slate's remaining games and the next 3 days; when the slate has already tipped off, the tab switches to upcoming games automatically. MIX has the same Today / Next 3 days toggle.
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
  * **Americas:** MLS, USL Championship, NWSL, Liga MX, Brasileirão, Liga Profesional, CONCACAF
    Champions Cup, Copa Libertadores.
  * **Asia & Oceania:** Japan J1 League (also in All Featured), K League 1, Saudi Pro League, A-League Men,
    Chinese Super League, AFC Champions League Elite.
  * **International:** World Cup, WC qualifying (UEFA / CONMEBOL / CONCACAF), UEFA Nations League,
    friendlies.

  How it works:
  * **Leagues:** "All Featured" queries the 19 top active competitions (now including Japan J1) at once and groups matches by
    competition, with a badge on each card. "Choose leagues" opens the regional picker.
  * **Dates:** a Today / Next 3 Days / Weekend toggle. An empty day automatically shows the next 4 days.
  * **Model:** a Poisson goal model gives home / draw / away, projected score, total goals and
    both-teams-to-score. Edges are measured against ESPN's three-way odds and goal totals.
  * **Cup and international form:** when a cup tie or national team has fewer than 3 completed matches in
    that competition, the app merges in recent results from ESPN's all-competitions schedule
    (`soccer/all/teams/{id}/schedule`). It then tries the club's domestic leagues, or other international
    competitions for national teams.
  * **Parlays:** soccer legs appear in the Parlays tab and in MIX.
* **College market blend:** NCAAB projections are blended with the betting line. The market-implied score is home = (total − spread)/2, away = (total + spread)/2. The model's weight is games/(games + 12), capped at 50% (≈20% after 3 games, 50% by mid-season); spread, total and win probability are recomputed. Cards show the mix (e.g. "20% model · 80% mkt"). If the raw model is more than 12 pts off the spread (15 on the total), that market is PASS: such gaps mean the model is missing information, not that it found value.
* **Lines on live and finished games:** ESPN removes `competitions[0].odds` from the scoreboard once a game
  starts.
  * Every pregame line is saved per event, refreshed until tip-off, so it is the closing line.
  * When the scoreboard has no odds, the app uses the saved line, else the game-summary `pickcenter`, else
    the core `…/competitions/{id}/odds` feed.
  * Live and final cards label it "Closing Mkt". Live games still get a BET/PASS verdict, marked "Pregame
    call vs closing line".
* **Implied period lines:** ESPN only posts full-game lines. 1H / 2H / 1Q–4Q / NHL period tabs scale the
  full-game spread and total by both teams' real share of scoring in that segment (e.g. 226.5 × 0.498 = 1H
  112.9). Those implied lines drive the edge badges and BET/PASS (priced at -110).
* **Period markets:** a pill bar above the matchup cards switches between periods:
  * NBA & NFL: Full Game, 1st Half, 2nd Half, 1Q to 4Q.
  * NCAAB & Soccer: Full Game, 1st Half, 2nd Half.
  * NHL: Full Game, 1st, 2nd and 3rd Period.

  How the period numbers are built:
  * **Data:** splits come from each team's real period scores (ESPN `competitors[].linescores`) over its last
    15 games. Soccer's 2nd half is the full score minus the 1st half. Games without period scores use
    league-typical splits. Team splits are blended toward league splits on small samples.
  * **Projection:** the full-game projection is split by the team's scoring share and the opponent's conceding
    share for that segment.
  * **Output:** each segment shows projected score, moneyline (win % and fair odds, with push or draw %),
    spread and total.
  * **Pricing:** NBA/NCAAB/NFL use a normal margin and total scaled to the segment length. NHL periods and
    soccer halves use an independent Poisson per side, giving a 3-way result, O/U 0.5 / 1.5 and (soccer)
    both teams to score.
  * **Breakdown drawer:** every matchup card has a collapsible "Period & Half Breakdown" with all segments
    at a glance. These are model prices only, since ESPN doesn't publish period odds.
* **Prop lines on every card:**
  * **Real lines first:** ESPN sportsbook prop bets (with prices), then The Odds API (optional; set
    `ODDS_API_KEY` in Streamlit secrets, free tier, cached 30 min), then PrizePicks, then Underdog.
  * **Implied line when none is posted:** 50% last-5 + 50% last-10 average × the opponent's defense for that
    stat (points / goals allowed per possession vs league; neutral where ESPN has no allowance data, e.g.
    rebounds / assists / shots), rounded to .5, labelled "Implied line".
  * Every card is judged 🟢 BET / 🟡 PASS on projection vs. that line, with the exact edge in the reason.
* **BET / PASS decisions:** every game card, player-prop card, alt line and UFC bout shows 🟢 BET or 🟡 PASS
  with the reason.
  * **BET** needs the model's edge to clear the market threshold and the hit chance to be ≥ 2 percentage
    points above break-even. Thresholds: 2.0 pts for spreads and totals, 0.3 goals for NHL/soccer totals,
    2.0 pts / 1.0 reb-ast / 15 pass yds for props. Break-evens: -110, the posted moneyline, or -115 for props.
    A ≥ 70% last-10 hit rate with a probability edge also qualifies.
  * **PASS** reasons: thin edge, trap zone (line sits on the projection, NFL key numbers 3/7, unders at 0.5),
    small sample, high recent variance, or a questionable player.
  * The props tab lists BET cards first and shows a "N BET · M PASS" count.
* **Alternate lines:**
  * **Player props:** every card has an "Alt lines" ladder with the main line ± three tiers: ±2/4/6 for
    points and PRA, ±1/2/3 for rebounds, assists, receptions and shots, ±10/20/30 for yards, and + tiers only
    for NHL goals/assists/points. Each rung shows model probability, last-10 hit rate, fair odds and BET/PASS.
  * **Alt-line selector:** shifts every card in the props tab to an alt tier at once.
  * **Game cards:** an "Alt lines" drawer with alt spread / puck line / goal line, alt total and both team
    totals.
  * **Pricing:** no free feed posts alt prices, so alt rows show the model's fair odds. An alt BET needs a
    62–85% model probability (higher means the price is too short) and a ≥ 60% last-10 hit rate.
  * Props without a PrizePicks / Underdog line use an estimated line (last-10 average), marked "est. line".
* **Injuries** (NBA, NFL, NHL) shows ESPN's report for every team on the slate.
* **🎯 Bankroll Challenge** (in the sport menu): a "$50 to $10k in N bets" rollover challenge, tracked honestly.
  * **Setup:** a name, a starting bankroll, a target, the number of bets/days and a rollover %. The math card shows the growth needed, the odds every bet needs, a bet-by-bet ladder, and the real chance of winning them all. At fair odds that chance is always about start ÷ target, and it is lower once the sportsbook's margin is included.
  * **Each day:** turn on the scan. It searches today's whole board (game lines, player props and UFC fights across every sport) for the most likely single or 2-leg parlay that pays at least the odds this step needs. It shows a best pick and two alternatives, each with its hit %, the hit % the price implies, the payout and a +EV/−EV tag.
  * **Tracking:** lock a pick in (or log your own bet), then mark it Won, Lost or Push. You see the bankroll, the win streak, the bets left, a progress bar to the target, the odds needed to stay on pace, and the history.
  * **Stakes:** a "Stake the full bankroll" toggle (or a custom amount) applies before you lock in each bet. A pending bet's stake can be changed with "✏️ Change stake" or the 💰 Full bankroll button. Rollover %, target and bets/days can be edited later under "⚙️ Challenge settings".
  * **After a loss:** a busted challenge explains the loss (e.g. "that bet had about a 62% chance — it still loses roughly 1 time in 3"). It then offers four easier restarts, each with a one-tap Start button that opens the new run as "name (retry 2)":
    * **Slower pace:** more bets, so each one can be a safer price.
    * **Smaller target:** a reachable first milestone.
    * **Cushion:** 50% rollover, so a loss halves the bankroll instead of ending the run.
    * **Streak ladder:** many small steps using 100% streak lines.

    Each option shows the odds every bet needs, the hit rate those odds imply, and the chance of finishing.
  * **Saving:** challenges are saved on the server by name. You can also download a .json backup and restore it, because Streamlit Cloud can wipe storage on redeploy.
* **🔥 Streaks tab** for NBA, NCAAB, NFL, NHL, Soccer and UFC. It lists hit-rate streaks: the line each player has cleared in every one of their recent games, shown as "D. Bane · 15+ Points · 9/9".
  * **Choosing the line:** the candidate lines are the lowest values of the last 5, 6, … games. Each is scored by streak length × (line ÷ season average)², and lines below a per-market minimum are skipped. That way a trivial 4+ points can't win just because it never missed.
  * **Each row shows:** the season hit count at that line, the average, a bar for each of the last 10 games (red where the player missed), and tonight's model projection. ⚠ marks a projection below the line. Players ruled OUT are left out.
  * **Team streaks:** points or goals scored and win streaks. For soccer: scored 1+, over 1.5 and over 2.5 match goals, both teams scored, unbeaten, won, and clean sheets (4+ matches).
  * **UFC:** win streaks, wins inside the distance, KO/TKO and submission runs, and fights that went or didn't go the distance.
  * **80%+ hit-rate lines:** as well as perfect streaks (X/X straight), each player gets their best higher line cleared in at least 80% of a recent window (e.g. 18+ points 11/12). These show in amber as "9/10 · 90% hit". A 100% only / 90%+ / 80%+ filter sits at the top of the tab, and the parlay builder has the same choice for its legs.
  * **🧩 Parlay builder** (switch at the top of the tab): builds a parlay from players on streaks in tonight's games. Choose the legs with a slider, from 2 up to every eligible player, plus the markets, a minimum streak length, and whether to skip Questionable players. It uses one leg per player at their likeliest streak line, ranked safest first. Each leg's hit chance blends the season hit rate at that line with tonight's projection, capped at 97%. The card shows the parlay's combined hit chance, its fair odds, and the weakest leg.
  * **📈 Your results (tunes the builder):**
    * **Tracking:** in the parlay builder, "📌 Track this parlay" saves the legs along with the hit chance the builder gave each one.
    * **Grading:** tracked legs are graded automatically from ESPN box scores once the games are final (✅ hit, ❌ miss, ➖ didn't play). Legs can also be marked by hand.
    * **Screenshots:** "📷 Upload a betslip screenshot" reads the slip with OCR (Tesseract, installed through `packages.txt` on Streamlit Cloud). It matches each leg to tonight's players and pre-fills hit or miss from the sportsbook's green check or red ✕, for you to confirm before saving. Legs can also be added by hand.
    * **Calibration:** once legs are graded, the builder compares what it said with what actually hit, overall and per market. It turns the gap into a factor (for example ×0.87) applied to every future leg's chance. The factor moves toward your results as they add up (shrunk toward the builder's own estimate), a market gets its own factor after 5 graded legs, and factors stay between ×0.70 and ×1.10.
    * **Backup:** results can be backed up and restored as .json.
  * **Loading:** the tab loads only when you open it, and all the slate's game logs load in parallel.
  * **Player profiles:** tap any player streak to open their profile. It shows tonight's projection, line, L5, season average, streak and a BET/PASS pill for every market. Below that are the full prop report for any market (form, matchup, risk, splits, alt lines, news) and the last 15 games, coloured green or red against their streak lines.
* **UFC:**
  * **Fight Card** lists every bout from the main event down to the prelims, with records, odds, win
    probability, the pick and the predicted method (KO/TKO, Submission, Decision).
  * **Fight Breakdowns** shows a tale of the tape (age, height, reach, stance, striking and grappling rates),
    method-of-victory bars, the round and method table, and the model factors.
  * **Round & Method Props** drawer on every bout: the exact probability of each outcome for each fighter, with
    fair American odds. Outcomes are KO/TKO/DQ, Submission and Decision, plus round moneylines for a finish in
    Round 1 through Round 3 (or Round 5). It also shows goes-the-distance vs inside-the-distance and the decision
    type (Unanimous / Split / Majority). Finishes are spread over the rounds using the fighter's own UFC stoppage
    rounds blended with UFC-wide round rates. ESPN posts no method or round prices, so the market reference is
    implied: the vig-free moneyline × weight-class outcome rates, plus a ~15% prop hold. A model edge of at
    least 2 points over that reference is 🟢 BET; anything else is 🟡 PASS.
  * **Data Backing & Analytics** drawer on every bout:
    * Striking efficiency: SLpM, SApM, the differential, accuracy and defense.
    * Grappling and control: TD/15, TD accuracy and defense, submission average.
    * Durability and finish rate: share of wins by KO, submission and decision, the stoppage rate, UFC losses
      by KO or submission, and UFC experience.
    * Recent form: the last 5 UFC bouts with the result, opponent, method, round, time and date. These come
      from ESPN fight history.
  * **All UFC fighter data comes from ESPN; UFCStats.com is no longer scraped.** When ESPN has no figure for a
    fighter, for example a debut or a null stat, the app shows the weight class's divisional average instead of
    a dash. Those values are marked ≈ and are also what the model uses. When there is no fight history, method
    splits use the weight class's finish rates.
  * **Moneyline fallback:** if ESPN's scoreboard has no moneyline, the app uses the saved pregame or closing
    line. If there is none, it tries ESPN's core per-bout odds feed. If that is empty too, it shows the
    model's fair odds, labelled "model fair".

## Code layout

```
app.py            UI (sport switcher, tabs, cards)
engine/common.py  throttled HTTP, disk cache, parsing helpers
engine/espn.py    ESPN endpoints + tolerant parsers
engine/teams.py   team ratings + per-sport game models (and optional nba_api ratings)
engine/props.py   player prop models
engine/ufc.py     fight cards, fighter profiles, win-probability and method model, moneyline fallback
engine/ufc_props.py  round moneylines, method props, implied-market BET/PASS
engine/ui.py      CSS + HTML helpers
```

Projections are estimates for entertainment and research, not betting advice.
