# T20X Blog Series: Cricket Analytics Deep Dives

Inspired by Jarrod Kimber's approach — cricket-geeky, data-driven, opinionated, with novel analytics that challenge conventional wisdom. Each post should introduce a concept, show the data, and deliver a surprising or counterintuitive finding.

---

## Series 1: "The Numbers Are Lying" (Foundation Posts)

These establish the t20x brand and the problem with conventional cricket stats.

### Post 1: "Your Favorite T20 Batsman's Average Is a Lie"
**Hook**: Batting averages treat all bowlers as equal. They're not. We built a system to fix that.
**Content**:
- Show how a well-known batsman (e.g., Suryakumar Yadav) has inflated stats from facing weaker bowling attacks
- Introduce opponent-quality adjustment: the same 150 runs mean different things against Mumbai Indians' attack vs a weaker side
- First reveal of t20x ratings vs conventional stats — show the biggest risers and fallers
- **Surprising finding**: Name a popular player whose adjusted rating is much lower (or an underrated one who rises)
**Data needed**: Phase 1 (ingestion) + Phase 3 (basic ratings)

### Post 2: "The Death Bowler Myth: Why Your Team's Finisher Might Be a Fraud"
**Hook**: Death overs are where matches are won. But some "death specialists" have been hiding behind easy matchups.
**Content**:
- Phase splits: separate powerplay, middle, death ratings
- Show bowlers who are rated elite in death overs by economy but actually face weaker batsmen at that phase
- Introduce the concept of "earned economy" vs "contextual economy"
- Jarrod Kimber-style: pick a specific bowler the fans love and show why the data disagrees
**Data needed**: Phase 3 (phase-specific ratings)

### Post 3: "Spin vs Pace: The Matchup Matrix IPL Teams Don't Know They Need"
**Hook**: Every team has a left-hander who can't play leg-spin. Here's the data to prove it.
**Content**:
- Batsman vs bowler-type matchup matrices from t20x
- Show how certain batsmen have massive blind spots (e.g., great overall but terrible vs wrist spin)
- Recommend auction/trade targets based on matchup exploitation
- Kimber-style: "If I were an IPL coach, I'd bowl X to Y in the powerplay every time"
**Data needed**: Phase 2 (bowler types) + Phase 3 (ratings)

---

## Series 2: "The Method" (How t20x Works)

Technical posts that show the methodology. Appeals to data/cricket nerds.

### Post 4: "How We Stole From Baseball and Football to Fix Cricket Analytics"
**Hook**: Expected Goals changed football. WAR changed baseball. Cricket has nothing equivalent. Until now.
**Content**:
- Walk through the cross-sport analogies: xG → xR (Expected Runs), WAR → Cricket WAR, Elo ratings
- Show how the iterative rating system works (the circular dependency insight)
- Compare t20x predictions to conventional wisdom on 5-10 well-known matchups
- Make it accessible: "If Virat Kohli is a 1750-rated batsman and Rashid Khan is a 1820-rated bowler, here's what the numbers expect to happen when they meet"
**Data needed**: Phase 3 + Phase 4 (xR model)

### Post 5: "Linear Models vs Gaussian Processes: Which Predicts Cricket Better?"
**Hook**: We tested two completely different mathematical approaches to rating cricketers. One of them is clearly better.
**Content**:
- Explain Bradley-Terry (linear/logistic) vs GP approach in accessible terms
- Train both on IPL 2008-2023, test on IPL 2024
- Compare on: match outcome prediction, individual delivery prediction, top-player ranking accuracy
- Show calibration plots and Brier scores
- **The punchline**: Which one does your team's analyst probably use? (Linear.) Which one should they use?
**Data needed**: Phase 3 + GP implementation

### Post 6: "Expected Runs: Cricket's xG Moment"
**Hook**: Every delivery in T20 cricket has an expected run value. Most batsmen are worse than you think.
**Content**:
- Introduce xR (Expected Runs) model
- Show xR heatmaps by phase and bowler type
- Rank batsmen by "Runs Above Expected" — the ones who consistently outperform context
- The flip side: "Expected Wickets" and which bowlers create chances vs which get lucky
**Data needed**: Phase 4 (xR model)

---

## Series 3: "The Auction Room" (Practical/Commercial Posts)

These are the money posts — actionable for fantasy cricket, team analysts, bettors.

### Post 7: "The IPL 2025 Auction Guide: Who's Overpriced and Who's a Steal"
**Hook**: We used t20x to value every player in the IPL auction pool. Here's who teams should target.
**Content**:
- Cricket WAR per player, broken down by phase
- "Value" metric: WAR / expected salary
- The "steal" list: high-WAR players that conventional stats undervalue
- The "avoid" list: low-WAR players that averages make look good
- Post this right before an auction for maximum engagement
**Data needed**: Phase 5 (WAR + comparison engine)

### Post 8: "Building the Perfect T20 XI Using Data"
**Hook**: If you could draft any 11 players from any T20 league, who would you pick? We optimized it.
**Content**:
- Use t20x ratings to select an optimal XI that covers all phases and matchup types
- Constraints: need openers, middle order, finishers, powerplay bowler, death bowler, spinner
- Show how the "data XI" differs from the "vibes XI" (fan picks)
- Run simulated matchups: data XI vs current world XI, show expected win probability
**Data needed**: Phase 5 (recommender)

### Post 9: "Why Your Fantasy Cricket Team Sucks (And How to Fix It)"
**Hook**: Fantasy cricket rewards you for picking players who score runs and take wickets. But not all runs are created equal.
**Content**:
- Show how fantasy scoring doesn't align with actual match impact
- Use t20x matchup data to recommend fantasy picks for specific matches
- "Contrarian picks": players with favorable matchups that most fantasy players ignore
- Weekly format: could become a recurring feature during IPL season
**Data needed**: Phase 4 + Phase 5

---

## Series 4: "Controversies" (Hot Takes Backed by Data)

These are designed to generate discussion. Kimber-style provocative.

### Post 10: "The Most Overrated T20 Player in Every League"
**Hook**: Every league has a player whose reputation exceeds their data. We found them all.
**Content**:
- Biggest gap between conventional ranking and t20x opponent-adjusted ranking, per league
- Break down WHY they're overrated: easy opposition? Flat pitches? Favorable matchups?
- Be specific and name names — controversy drives engagement
**Data needed**: Phase 3 + Phase 5

### Post 11: "Kohli vs Babar vs Buttler: The Definitive Data Comparison"
**Hook**: Cricket's greatest debate, settled by 2 million deliveries of data.
**Content**:
- Head-to-head comparison across every dimension: phase, bowler type, pressure situations, opposition quality
- Show where each player is clearly best and where they're weakest
- Use the player similarity engine to find who each player's "closest comparison" is from other leagues
- Let the data pick a winner (or show it's situation-dependent)
**Data needed**: Phase 5 (comparison engine)

### Post 12: "The Pitch Is Not Neutral: How Venues Distort Player Ratings"
**Hook**: Some players look great because they play at Chinnaswamy. Others look mediocre because they're at Chepauk. Here's the correction.
**Content**:
- Venue effects on batting/bowling metrics
- Venue-adjusted ratings: who rises and falls when you normalize for ground
- The "road warriors" — players who perform away from home
- Implications for team selection in different conditions
**Data needed**: Phase 4 (situation-aware metrics)

---

## Series 5: "Deep Dives" (Long-Form Analysis)

### Post 13: "The Evolution of T20 Cricket: A Decade of Ball-by-Ball Data"
**Hook**: We analyzed every delivery bowled in T20 cricket since 2014. The game has changed more than you think.
**Content**:
- Trends: scoring rates by phase over time, bowler type effectiveness changes
- The rise of wrist spin, the decline of traditional off-spin
- How death bowling has evolved (yorkers → slower balls → wide lines)
- Historical rating trajectories of legendary players

### Post 14: "Pressure Index: Who Thrives When It Matters Most?"
**Hook**: Some players perform best under pressure. Others crumble. Here's the data.
**Content**:
- Define pressure index (chasing, high required rate, late wickets, knockout matches)
- Rank players by performance under high vs low pressure
- Surprising findings: who are the real "clutch" players?
- The pressure-adjusted WAR: a player's value when it actually counts

---

## Publishing Strategy

1. **Launch**: Posts 1 + 4 together — hook readers with a controversy AND show the method
2. **Cadence**: 1 post every 10-14 days during IPL season, monthly otherwise
3. **Platform**: Substack or Medium (easy analytics, newsletter built-in)
4. **Cross-promote**: Share key charts on Twitter/X with "full analysis on the blog" link
5. **Timing**: Align posts with IPL matches (e.g., Post 3 before a match where the matchup data is relevant)
6. **Open source**: Link to the t20x GitHub repo in every post — builds credibility and attracts contributors

## What to Build First for Blog Content

Priority order of t20x features for generating blog-worthy analysis:

1. **Phase 1-3** (ingestion + ratings) → enables Posts 1, 2, 3, 10
2. **Phase 2** (bowler types) → enables Post 3
3. **Phase 4** (xR + metrics) → enables Posts 6, 9, 12, 14
4. **GP implementation** → enables Post 5
5. **Phase 5** (comparison) → enables Posts 7, 8, 11, 13
