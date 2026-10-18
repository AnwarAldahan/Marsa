# Public deployment on Render

One Python web service serves both the canonical `Marsa updated.html` dashboard
and its FastAPI backend. Hosted pages send API requests to their own origin;
opening the HTML directly from disk retains the localhost fallback.

## Publish

1. Create a Render account at https://dashboard.render.com/register and connect
   the GitHub account that can publish the project (or publish an authorized fork).
2. Commit and push the intended working-tree implementation and deployment files
   to the branch to deploy. The current unified implementation includes previously
   untracked files: `src/marsa/twin/unified.py` and
   `src/marsa/api/routes/unified_twin.py`. Include these along with the associated
   existing application/configuration changes. Do not deploy an older `main`
   checkout expecting it to contain the current local implementation.
3. In Render, create a **Blueprint**, select the repository and the published
   branch, and use the root `render.yaml`.
4. Review the free web service and deploy. Keep the repository root as the service
   root directory so configuration, data, model artifacts, and HTML are available.
5. After deployment succeeds, copy the actual HTTPS URL from the service page.
   The root URL opens the dashboard; `/health` checks liveness and `/docs` opens
   the API documentation. The final hostname is allocated by Render, not assumed.
6. Load `2025-07-10T19:00:00Z`, open the twin comparison, and test a manual decision.

## Optional Gemini

The application works without Gemini and uses deterministic reasoning/fallback.
To enable the existing provider, add `GEMINI_API_KEY` as a secret in the Render
service environment. `MARSA_GEMINI_MODEL` can override the provider model.
Never commit `.env` or put keys in HTML, `render.yaml`, or screenshots.

## Runtime constraints

- Use one Uvicorn worker and one service instance: frozen contexts are kept in
  process memory. A restart or context expiry requires reloading the analysis.
- The free service can sleep while idle; its public address persists but the next
  request can need a cold start. This is not an always-on availability guarantee.
- The initial configuration uses the free plan; actual cloud performance and
  memory limits still require verification after deployment. No paid upgrade is
  configured here.
- Automatic deployment is disabled; later updates need a manual deploy or an
  explicit change to the deployment policy.
- Browser feedback remains browser-local. Hosting does not add persistence,
  learning, or real port operations.

References: https://render.com/docs/deploy-fastapi and
https://render.com/docs/blueprint-spec.
