# Generated demo — a microservice Kimi K2 wrote on Arm CPU

`main.py` here is **verbatim output** from Kimi K2 (1.04T params) running on the Azure
Cobalt 100 Arm VM, generated at 8.6 tok/s, in response to a single prompt asking for a
FastAPI URL-shortener. We then installed FastAPI, ran it on the same Arm box, and hit it
live:

```
POST /api/shorten {"url":"https://arm.com/ai"}  ->  {"code":"hDAsKo","short_url":"/hDAsKo"}
GET  /hDAsKo                                     ->  307 redirect to https://arm.com/ai
```

See [`../../playbook/artifacts/shot_live_service.png`](../../playbook/artifacts/shot_live_service.png).

**Known bug (left as generated, honestly):** the catch-all `GET /{code}` route is declared
*before* `GET /healthz`, so `/healthz` is shadowed and returns 404. It's exactly the kind of
route-ordering mistake NightShift's PR-review Action is built to catch — a nice illustration
of why the reviewer tier matters. The shorten + redirect paths work correctly.
