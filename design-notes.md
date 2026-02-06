# Analysis Page Redesign Notes

## Reference Image Analysis

### Image 1 — SquaredUp FPL Dashboard
- Dark navy background with subtle grid
- Red semi-circle gauge indicators for "Points behind first" (177, 134)
- Line charts with gradient fills (blue/purple area charts)
- Bar charts (yellow/orange bars) for season rank
- Data tables with clean rows
- Small icon badges and logos
- Color palette: navy, red, blue, yellow/orange accents

### Image 2 — Football Team Dashboard (dark/orange)
- Deep navy/dark purple background
- **Radar/Spider chart** (yellow/orange) for Team Performance (Communication, Speed, Stamina, Punctuality, Teamwork)
- Key Player card with photo and horizontal stat bars (Stamina, Punctuality, Speed, Teamwork) in gradient colors
- Team Lineup Summary with pitch formation view (Wins 3, Loses 2)
- **Donut chart** (orange/yellow/blue) for Team Player Stat percentages (50%, 30%, 20%)
- Area chart for Team Effectiveness over time
- Orange accent sidebar
- Color palette: dark navy, orange, yellow, blue accents

### Image 3 — GoalGalaxy Player Dashboard
- Purple/indigo gradient background
- Player card with large photo, name, points (312 pts), country flag, club badge
- **Bar chart** (green/blue grouped bars) for Performance over dates
- **Line chart** for Price chart over years
- Statistics list (Match played: 42, Minutes played: 2562, Goals: 17, etc.)
- Clean card-based layout
- Color palette: purple/indigo, green, blue, white text

### Image 4 — Current Dashboard (needs improvement)
- Too flat, basic stat bars
- Pitch visualization is functional but not visually premium
- Missing: charts, gauges, visual variety
- The stat comparison bars are okay but need more visual impact

## Design Plan for Redesign

### Layout (Bento Grid)
Row 1: Video Player (3 cols) | Key Stats Panel with gauges (2 cols)
Row 2: Team Performance Spider Chart (2 cols) | Possession Donut (1 col) | Match Stats Bars (2 cols)  
Row 3: Pitch Visualization (3 cols) | AI Commentary (2 cols)
Row 4: Performance Bar Chart (2.5 cols) | Event Timeline (2.5 cols)

### Charts Needed
1. **Radar/Spider Chart** — Team performance attributes
2. **Donut Chart** — Possession or stat breakdown
3. **Bar Chart** — Performance metrics per period
4. **Semi-circle Gauge** — Key metric indicators
5. **Pitch Radar** — Already done, keep premium styling

### Library
- Use recharts (already commonly available in React projects) or build SVG charts manually
- Check if recharts is already installed
