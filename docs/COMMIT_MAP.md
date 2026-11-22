# Rewritten week4 commits (2026-11-23)

The first nine `week4` commits were rewritten to remove a co-author trailer from their messages. Nothing else
changed: every rewritten commit has the same tree (the same files, byte for byte), author and date as the
original. The result files in `results/serve/` record the commit they ran from (`provenance.git_commit`, or
`git_commit` in the merge manifest). The originals no longer exist on GitHub, so look up the code a recorded
hash ran with here:

| Recorded hash | Commit on week4 | Commit |
|---|---|---|
| `12a40bb` | `3d984d5` | Verified merge of the final adapter into bf16 base weights (4.1); recorded by `merge_run_a_r16.json` |
| `54fa857` | `0cc7ee3` | 4.1 result: merged Run A verified (job 145) |
| `1c7adba` | `8849ca6` | Endpoint: FastAPI + transformers.generate with validation, retry and refusals (4.3) |
| `4737bf3` | `3853dab` | Pin fastapi, starlette, uvicorn (4.3); recorded by `run_a_r16_http.json` |
| `ed5c4cd` | `b07e192` | 4.3 result: served latency through HTTP (job 147) |
| `17b9999` | `691d160` | Quantisation with a pre-fixed accept/reject verdict (4.2) |
| `289662f` | `2e341cd` | 4.2: tied embeddings mean no lm_head tensor |
| `2649541` | `1cfff49` | 4.2: turn off GPTQModel's disk offload |
| `43e6ade` | `962cdb6` | 4.2: compare module names across the two checkpoint layouts; recorded by `quantise_run_a_r16.json` |

To check the claim: `git rev-parse 962cdb6^{tree}` equals the tree recorded for `43e6ade` in any clone made
before the rewrite (`git rev-parse 43e6ade^{tree}`).
