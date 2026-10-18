# Canonical website

Start the API from the repository root:

```powershell
python -m uvicorn marsa.api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/** for Main's root `Marsa updated.html` design with the unified Hybrid contract.
See [the unified contract](../docs/UNIFIED_TWIN.md). Instructions below describe the separate legacy reference dashboard and its legacy API semantics.

# Legacy reference dashboard — how to run

Terminal 1 (backend):
    $env:PYTHONPATH="src"; python -m uvicorn marsa.api.main:app --reload

Terminal 2 (frontend):
    cd dashboard
    python -m http.server 5500

Open: http://127.0.0.1:5500/marsa.html

Type a UTC hour of 2025 (for example 2025-07-10T19:00:00Z) and press "تحليل".
Optional stress scenario: "أرصفة مغلقة" (0 = none) and "مضاعف السفن القادمة" (×1 = real schedule, ×2 = twice the real
vessels arriving within 24 h, no upper limit). Leave them at 0 and ×1 to replay the real AIS schedule.
Optional "إضافة سفن للتجربة": vessel number (IMO/MMSI, suggestions from GET /api/twin/vessels), length,
stay hours and arrival hour. A stay left empty is estimated from earlier stays (badge "تقدير من التاريخ").
The page calls POST /api/decision-support/analyze and shows only what the API returns.
Two views in one file: the dashboard (sections ١–٥) and the Digital Twin (agent options, hour-by-hour
replay, manual test). The manual test sends the user's decision as `user_actions`; the Strategy agent
evaluates it next to its own options. Hours by condition come from GET /api/twin/hours.
Section ٤: each option has "جرّب هذا القرار" (shows its result in the map and charts at the top), and
"جرّب قرارك الخاص" sends a custom decision (queue order, extra berths, gate per hour) to
POST /api/twin/simulate on the same hour, scenario and added vessels as the last analysis.
Charts show a dotted line at +6 h: the prediction model's horizon. The yard chart compares the shown
option with no change.
There is no demo data and no fallback: if the API is offline the page says so and shows no numbers.
The server address field defaults to http://127.0.0.1:8000 (uvicorn's default port).

Source badges on the page: AIS / مرصود (observed), اصطناعي (synthetic), تنبؤ النموذج (XGBoost),
محاكاة (Digital Twin), مدخل المستخدم (values typed by the user), تقدير من التاريخ (estimated stay), مصدر عام (public figure: 23 berths).
