# Marsa dashboard — how to run

Terminal 1 (backend):
    $env:PYTHONPATH="src"; python -m uvicorn marsa.api.main:app --reload

Terminal 2 (frontend):
    cd dashboard
    python -m http.server 5500

Open: http://127.0.0.1:5500/marsa.html

The bar at the top-left lets you pick an hour of 2025 (or type any UTC hour like 2025-08-15T14:00:00Z).
"تحميل" calls POST /api/decision-support/analyze and fills sections ١–٤ and the twin cards with real
values from the pipeline. If the API is offline the page keeps showing the demo numbers.

Notes
- The port map in section ١ is illustrative (6 berths drawn); the numbers around it are real (23 berths).
- Section ٥ ("أداء مرسى") is still demo data, labelled as such.
- API base can be changed by defining window.MARSA_API_BASE before the adapter script.
