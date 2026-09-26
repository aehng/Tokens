# Full vLLM predictive proof v14

Kernel v14 ran the full Phases 1–10 entrypoint from source commit
`28da08d370d3f5ac8a91ccd526b0935ae9fd551d`. The mounted source matched the
expected archive: 34,292,280 bytes, SHA-256
`a35f001b7d2acc594742b82a716cf5234df5c977e498b5900f11675d17f084ab`.

## Result

The run reached Phase 9 and stopped with:

```text
RuntimeError: phase 9 preemption failed
status: FAIL
reason: preemption not exercised
preempted_request_ids: []
rebuild_state: {}
steps: 9
trajectory_matches: {pre-A: true, pre-B: true}
```

The Phase 9 budget used block size 16, eight available blocks, prompts of 42
and 74 tokens, and `max_new_tokens=7`. It predicted four and six blocks at
full length (ten combined). In fact, a generated sequence of `n` output tokens
has at most `prompt + n - 1` tokens in KV: the final sampled token is not fed
back through the model. Seven outputs therefore used 48 and 80 cached tokens,
or three and five blocks, exactly the available capacity. The scheduler had no
reason to preempt either request. Token trajectories matched, but that does not
establish a preemption or state-rebuild pass. Phase 10 did not run.

## Evidence limits

The complete log stream exposed the Phase 9 report and traceback after the
local CLI was run with UTF-8 output. The output-download command began copying
the entire staged `tokens_src` tree, so it was stopped after approximately
195 MB; a complete machine-readable report bundle was not retained. This note
records the observed log evidence, not an independently retrieved report JSON.

The next full proof must use eight output tokens for Phase 9, ignore EOS in
both solo and concurrent requests, and record positive scheduler preemption
counts. A request must also have an admission add/remove/re-add cycle, rebuilt
predictive state, and the same completed tokens as its uninterrupted run.
