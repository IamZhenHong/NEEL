# SC-M26 result: not runnable. OpenAI no longer lets this organisation create fine-tuning jobs

Pre-registration `sc-m26-prereg` (bacfa4b0). **Spend: $0.004** (a 48-item smoke test of the eval path on the untuned
gpt-4.1-mini; nothing was trained).

**What happened (10-25):**
- The six training files were built.
- The first file uploaded (W, 591 KB).
- `POST /v1/fine_tuning/jobs` was refused with HTTP 403, `training_not_available`: "OpenAI is winding down the
  fine-tuning platform and your organization is no longer able to create new fine-tuning training jobs."
- OpenAI's deprecation notice closes self-serve fine-tuning in stages from May 2026. It is closed to organisations
  without recent fine-tuning use, and to all customers by January 2027.
- The uploaded file was deleted from OpenAI (`file-2Qd4…`, deleted: true). No job was created.

**Owner's decision:** skip #1. The "a narrow habit stays out of the self-report" claim stays scoped to open models,
0.8B–70B (SC-S1, SC-M13, SC-M16, SC-M9). No frontier family was tested.

**What still works, for anyone who picks this up:**
- The eval path is tested: `evalrun.py` runs SC-S1's frozen items on OpenAI's chat API through V1's runner. Tool calls
  come back in the engine's form, and SC-S1's scorers read them.
- So does `ft.py`'s file conversion.
- With fine-tuning access (another provider or an organisation that still has it), only `ft.py launch` / `wait`
  would change.
