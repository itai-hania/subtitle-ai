# Cloud Hardening Plan (Security + Performance)

**Created:** 2026-02-15  
**Scope:** `/Users/itayy16/CursorProjects/SubtitleAI`

## Goals

1. Reduce abuse and data-exfiltration risk before public cloud exposure.
2. Improve request admission and runtime stability under concurrent load.
3. Keep behavior compatible with existing API/UI flows.

## Deep Analysis Summary

### Security findings

1. **Rate-limit bypass risk via untrusted proxy headers**
- Current IP extraction accepts `X-Forwarded-For` from any client.
- In public deployments, an attacker can spoof per-request IPs and bypass per-IP rate limiting.

2. **Unbounded in-memory rate limiter key growth (DoS vector)**
- The limiter stores request history for every unique key without a max-key cap.
- Combined with spoofed IPs, memory growth can be forced cheaply.

3. **Missing duplicate-run guard on long-running jobs**
- A job can be re-triggered while already active.
- This can trigger duplicate CPU-heavy pipelines and elevate abuse impact.

4. **Sensitive transcript persistence in logs by default**
- Translation debug logs include raw input/output subtitle content.
- This creates privacy risk and unnecessary disk growth.

### Performance findings

1. **Admission control tied to total jobs, not active workload**
- Capacity checks use total job count (`len(jobs)`), including completed/idle jobs.
- Under cloud traffic this can reject valid work while CPU is idle.

2. **O(n) directory scans in hot-path file lookup**
- Fallback lookup loops directories multiple times per extension.
- This scales poorly as `uploads/` and `downloads/` grow.

3. **Rate limiter per-request pruning uses list rebuilds**
- Current implementation rebuilds timestamp arrays repeatedly.
- Under load, this adds avoidable CPU churn.

4. **Verbose translation log payload inflates memory + disk I/O**
- Full chunk inputs/outputs are assembled and persisted.
- This adds I/O and memory pressure without runtime value in normal operation.

## Implementation Plan

### Phase 1: Security hardening
- [x] Add proxy-trust toggle (`TRUST_PROXY_HEADERS`) and strict IP parsing.
- [x] Ignore forwarded headers by default; use socket peer IP unless explicitly trusted.
- [x] Rebuild rate limiter with bounded key-space and deque-based window pruning.
- [x] Reject duplicate processing/reburn requests when a job is already active.

### Phase 2: Performance hardening
- [x] Switch admission checks to **active-job** concurrency instead of total job count.
- [x] Keep a separate storage ceiling for total retained jobs.
- [x] Optimize fallback video file lookup to single-pass suffix checks.
- [x] Make translation debug logging opt-in and lean by default.

### Phase 3: Validation
- [x] Add targeted tests for proxy/IP behavior and bounded rate limiter behavior.
- [x] Add tests for active-job counting and duplicate-run guard.
- [x] Run full test suite and confirm no regressions.

## Completed Changes (2026-02-15)

1. `app.py`
- Added secure client IP extraction with optional trusted proxy headers (`TRUST_PROXY_HEADERS`).
- Replaced list-based limiter windows with deque windows and bounded key capacity (`RATE_LIMIT_MAX_KEYS`).
- Added active/stored job capacity controls (`MAX_CONCURRENT_JOBS`, `MAX_STORED_JOBS`) and centralized enforcement.
- Added duplicate-run guard for `/process/{job_id}` and immediate slot reservation for `/reburn/{job_id}`.
- Optimized filesystem fallback video lookup to a single-pass suffix match helper.

2. `processing.py`
- Translation debug logging is now opt-in (`ENABLE_TRANSLATION_DEBUG_LOGS`).
- Reduced log payload to compact previews (`TRANSLATION_LOG_PREVIEW_CHARS`) instead of full transcript dumps.

3. Tests
- Added security/performance regressions tests for proxy/IP extraction and limiter capacity in `tests/test_utils.py`.
- Added API behavior tests for active-job duplicate prevention and active-vs-total admission in `tests/test_api.py`.
- Full suite passing: `98 passed`.

## Success Criteria

1. Rate limiting cannot be bypassed with spoofed `X-Forwarded-For` in default config.
2. Rate limiter memory is bounded by configured key cap.
3. Active workload admission behaves correctly with many completed jobs present.
4. Duplicate compute pipelines for same job are blocked.
5. Translation logs are non-sensitive by default and only verbose when explicitly enabled.
