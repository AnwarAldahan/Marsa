# Marsa dashboard — how to run

Terminal 1 (backend):
    $env:PYTHONPATH="src"; python -m uvicorn marsa.api.main:app --reload

Terminal 2 (frontend):
    cd dashboard
    python -m http.server 5500

Open: http://127.0.0.1:5500/marsa.html

Type a UTC hour of 2025 (for example 2025-07-10T19:00:00Z) and press "تحليل".
Optional stress scenario: "أرصفة مغلقة" and "زيادة السفن القادمة %". Leave both at 0 to replay the real AIS schedule.
The page calls POST /api/decision-support/analyze and shows only what the API returns.
There is no demo data and no fallback: if the API is offline the page says so and shows no numbers.
The server address field defaults to http://127.0.0.1:8000 (uvicorn's default port).

Source badges on the page: AIS / مرصود (observed), اصطناعي (synthetic), تنبؤ النموذج (XGBoost),
محاكاة (Digital Twin), مدخل المستخدم (stress values typed by the user), مصدر عام (public figure: 23 berths).
