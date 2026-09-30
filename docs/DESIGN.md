# jevlike — DiffusionGemma 기반 Jev 스타일 결정 모델

## 1. 목표

Jev(TypeSafe AI, 2026-09-15)는 글을 생성하지 않는다. `state` + 타입이 정해진 질문 묶음을 받아
**질문마다 확률 분포와 confidence**를 돌려준다. 이 프로젝트는 같은 인터페이스를 오픈 가중치
diffusion LM(DiffusionGemma 26B-A4B, Apache-2.0) 위에 만들고, 실제 정답 데이터로
**확률을 보정(calibration)**한다.

| 질문 타입 | 입력 | 출력 |
|---|---|---|
| `noul` | 참/거짓 판단 문장 | `P(yes)` |
| `choice` | 선택지 2~26개 | 선택지별 확률 + 선택 |
| `score` | 순서가 있는 단계 2~10개 | 단계별 확률 + 기대 점수 |

모든 답에 `confidence = 1 − H(p)/ln K` (0 = 균등, 1 = 확신).

## 2. 왜 diffusion LM인가

AR 모델로 같은 일을 하려면 질문 N개에 대해 N번 생성하거나, 첫 질문 답이 뒤 질문에 영향을 준다.
DiffusionGemma는 **encoder(프롬프트 → KV cache) + 양방향 decoder(canvas 전체를 한 번에 denoise)** 구조다.

```
prompt ──encoder──▶ KV cache ──┐
                               ▼
canvas:  "q1:" [slot] "\n" "q2:" [slot] "\n" <eos> ...
                  ▲ 노이즈 토큰          ▲ 노이즈 토큰
decoder 1-pass ──▶ slot 위치 logits ──▶ 라벨 토큰만 골라 softmax ──▶ 분포
```

- **질문 N개 = decoder forward 1번.** slot끼리 서로 오염되지 않도록 각 slot 초기값은 무작위 노이즈이고, 템플릿 토큰은 고정(pinned)된다.
- **스키마 밖 답이 불가능.** 라벨 토큰 logits만 읽으므로 hallucination이 구조적으로 없다.
- **재독(re-read)은 기본 off.** DiffusionGemma는 uniform-noise diffusion이라 slot 노이즈에 따라 분포가 조금 흔들린다. 노이즈를 바꿔 여러 번 읽고 평균낼 수 있지만(`extra_reads`), 실측상 얻는 게 거의 없어 끈다(§6).
- **질문 prefix cache.** 프롬프트를 `질문 → state` 순서로 놓고, 시스템 프롬프트+질문 블록의 encoder cache를 질문 세트별로 재사용한다. 같은 질문을 여러 state에 던지는 Jev식 사용에서는 state만 prefill하면 된다.

참고 구현: [OpenJev](https://github.com/razorback16/openjev), [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250).

## 3. "모델을 만든다"의 범위

| 단계 | 내용 | 이 레포에서 |
|---|---|---|
| L0 Read engine | canvas slot read, 타입 스키마, API | 구현 |
| L1 Calibration | 실제 정답이 있는 데이터로 타입별 temperature 학습 (NLL 최소화) → ECE/Brier 개선 | 구현 |
| L2 Fine-tune (RLCD 근사) | slot 분포를 proper scoring rule(log loss/Brier)로 직접 학습하는 LoRA | 구현 (MLX, 4-bit 위 LoRA, 로컬) |

**L2가 RL이 아니어도 되는 이유.** Jev의 RLCD는 "사람 선호가 아니라 결과(outcome)에 대해 확률을 최적화"한다.
답이 한 토큰짜리 분포라면, 결과에 대한 calibration reward의 기댓값을 최대화하는 것은 그 분포에
proper scoring rule을 거는 지도학습과 같은 해를 가진다. 정책이 여러 토큰을 샘플링하는 경우에만 RL이 필요하다.
그래서 L2는 slot logits에 cross-entropy를 거는 LoRA로 충분하고, 데이터는 Jev처럼 합성 데이터
(LLM으로 state + 질문 + 정답 생성, 정답은 규칙·시뮬레이션으로 검증)로 만든다.

처음에는 GPU가 필요하다고 봤지만, MLX에서 4-bit 가중치 위에 LoRA(q/v_proj, r=8, 280만 파라미터)를 얹으면
M4 Pro 48GB에서 step당 약 1초, 피크 메모리 17.5GB로 학습된다. encoder cache는 gradient 밖에서 계산하고(stop-gradient),
decoder 경로에만 loss를 건다. MoE router의 top-k 인덱스는 이산값이라 `stop_gradient`로 끊어야 역전파가 된다(`lora.py`).

## 4. 컴포넌트

```
src/jevlike/
  schema.py       요청/응답 타입 (Pydantic v2)
  canvas.py       프롬프트·canvas 조립, slot 위치 계산 (토크나이저만 필요, 모델 불필요)
  engine.py       MLX DiffusionGemma 로드, 질문 prefix cache → state prefill → slot read (batched)
  calibration.py  타입별·질문별 temperature scaling, ECE/Brier/NLL
  synth.py        고객 지원 티켓 합성 데이터 (속성 → 정답이 결정적, 텍스트는 문구 뱅크)
  lora.py         LoRA 적용·저장·로드, slot label CE loss
  server.py       FastAPI POST /v1/decide
scripts/
  benchmark.py    공개 데이터셋(SST-2, AG News, Yelp) → 원 확률 vs 보정 확률 비교
```

## 5. 결정과 트레이드오프

- **라벨 = 단일 토큰.** 선택지는 ` A`,` B`… , 점수는 ` 0`…` 9`, noul은 ` yes`/` no`. 라벨 목록은 `canvas.py` 한 곳에서만 정의하고 프롬프트 설명과 slot 읽기가 같은 상수를 쓴다.
- **self-conditioning 끔.** 1-pass read에는 이전 step logits가 없다.
- **보정은 타입별 스칼라 T 하나.** 데이터가 적을 때 과적합이 없고, argmax를 바꾸지 않아 정확도는 그대로 두고 확률만 고친다.
- **동시성 1.** MLX 경로는 요청을 직렬 처리한다(lock). 처리량이 필요하면 vLLM 경로(OpenJev 참고)로 옮긴다.

## 6. Measured results (M4 Pro 48GB, 4-bit, 2026-09-30)

`scripts/benchmark.py --per-task 200`: per task, fit T on 100 items and evaluate on the other 100.
Current configuration (v4): question-first prompt with a label reminder after the state, prefix cache on, single read.

| Task (type) | acc | NLL raw → cal | ECE raw → cal | p50 latency |
|---|---|---|---|---|
| SST-2 (noul) | 0.97 | 0.224 → 0.142 | 0.028 → 0.023 | 307ms |
| AG News (choice, 4) | 0.83 | 1.226 → 0.605 | 0.184 → 0.082 | 323ms |
| Yelp (score, 5) | 0.54 | 1.857 → 0.992 | 0.380 → 0.110 | 587ms |

Fitted T = noul 2.59 / choice 3.34 / score 3.34. All are T > 1: **the raw distribution is overconfident**, which is
the gap Jev's "calibration against outcomes" targets. With n=100 per half, accuracy differences under ~5pt are noise;
compare variants with the paired analysis below, not with this table.

### What changed and why (paired over the same 200 items per task)

| Change | Evidence | Decision |
|---|---|---|
| Re-reads (4-read average) | AG News: 0/100 answers changed; Yelp: 4/100, raw NLL 2.75→2.60. Same-process A/B: decode 544→170ms without them | **off by default** (`extra_reads=0`, `JEVLIKE_EXTRA_READS`) |
| Batched reads + vocab projection only at slots | Same noise: max prob diff vs full-logits path 0.004 (bf16) | kept |
| Question-first prompt + prefix cache | Hit vs full prefill: identical to a no-clone split prefill (0.0); split-prefill numerics shift uncertain slots by ≤0.03. 3-question A/B: 777→505ms, 1-question 409→362ms | kept |
| Label reminder after state (v4 vs v3) | v3 lost AG News accuracy (0.815→0.790, raw NLL +0.23, CI excludes 0); v4 restores 0.815 (NLL +0.19, CI touches 0). Yelp gain kept in both (NLL −0.35, CI excludes 0) | kept (v4) |

Net on the 3-question smoke request, same-process comparison: **~1060ms → ~505ms**. Prefill is now the larger part.
Raw runs for each variant are kept in `runs/reads-v*.jsonl`.

### Pitfalls found

- **A canvas as short as the answer reads at chance.** With the exact-length canvas (~6 tokens), SST-2 was 8/16
  (0.50 over the full 200). Padded to 64 tokens it reached 15/16, the same as the model's own `generate` (15/16).
  DiffusionGemma is trained on canvases of at least 64 tokens (`DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH`).
- **The reply starts with an empty thought channel.** `generate` output is `<|channel>thought\n<channel|>q: no<turn|>`.
  The canvas uses the same prefix.
- **Encoder KV caches are written in place.** Reusing a prefix needs a clone with fresh arrays (`_clone_cache`);
  sharing the arrays would leak one request's state into the next.
- To validate a custom read path, **compare it on the same inputs with the model's native path**, and machine
  load makes absolute latencies drift by 2×, so latency claims come from interleaved A/B in one process.

### Multi-question interference (scripts/interference.py, 100 items per task)

The task question asked alone vs. first/last in a bundle with two unrelated questions. No accuracy change is
significant (McNemar p ≥ 0.18), but distributions move: flips concentrate on items that were uncertain alone
(AG News: 4–5 of 6 uncertain vs 0–4 of 94 confident), and the 5-level Yelp score moves most (mean |Δp| 0.19).
**Consequence: fit calibration in the same bundle the service asks**, which is what the domain setup below does.

## 6b. Domain: support-ticket triage (synthetic, Korean)

`jevlike.synth` samples ticket attributes first and derives the labels from them (`labels()`), then renders Korean
text from phrase banks, including hard cases: "취소" in a non-churn sense, billing words in non-billing tickets,
angry tone on low-impact tickets. Four questions per request: category (6), urgency (4), churn, engineering.
`scripts/domain_eval.py`: 600 tickets, per-question T fit on the first 300, test on the last 300.

| Step | category | urgency | churn | engineering |
|---|---|---|---|---|
| v1: question wording without policy | 0.98 | 0.52 | 1.00 | 0.67 |
| v2: policy written into the question | 0.93 | 0.79 | 1.00 | 0.95 |
| v2 + LoRA (400 synthetic tickets, 1 epoch) | 1.00 | 0.89 | 1.00 | 1.00 |

- v1 errors were the question's fault, not the model's: "needs code change or infra action" is a fair reading for
  feature requests (40/46 said yes), and urgency had no rubric (one level too high across categories, angry-but-low
  tickets 35% correct). Writing the rubric into the question fixed most of it (angry-but-low → 100%).
- Per-question T beats the public-data per-type T on every question (e.g. category NLL 0.069 vs 0.171 in v1:
  the public T made it worse). T is floored at 0.5: churn was perfectly separable and the unfloored fit went to 0.05.

### LoRA: policy or templates? (scripts/shift_eval.py)

Train and test above share the phrase-bank generator, so the in-domain gain can be template memorisation.
`render_shifted` keeps the attributes and labels but uses none of the training phrasing; 300 tickets, paired:

| Question | base → LoRA acc | base-only / LoRA-only right | p |
|---|---|---|---|
| urgency | 0.757 → 0.853 (NLL 1.29 → 0.40) | 11 / 40 | <0.001 |
| engineering | 0.953 → 0.997 | 1 / 14 | 0.001 |
| category | 0.953 → 0.920 (NLL 0.14 → 0.31) | 19 / 9 | 0.09 |
| churn | 1.000 → 0.997 | 1 / 0 | 1.0 |

The policy parts (urgency rubric, engineering routing) transfer to unseen wording; category, which the base model
already handled, drifts toward the templates. Public benchmarks after LoRA (100 items per task, not significance-
tested): SST-2 0.93 → 0.94, AG News 0.79 → 0.75, Yelp 0.50 → 0.45 — some forgetting outside the domain.
Real tickets are still the missing test; synthetic phrasing is narrower than real writing.

## 6c. AR baseline (scripts/ar_baseline.py)

Qwen3-4B (4-bit, MLX; similar active parameters to DiffusionGemma's ~4B) reads each question's label-restricted
next-token distribution after `<qid>:`, appends its own answer, and moves to the next question. Same items,
same labels, both uncalibrated. The AR model gets its better prompt: with the diffusion prompt (questions first)
it answered "no" to nearly every SST-2 item (20/40, confirmed identical to its own `generate` output);
state-first gave 31/40.

| | Diffusion | AR (Qwen3-4B) |
|---|---|---|
| SST-2 | **0.93** (NLL 0.29) | 0.72 (NLL 4.03) |
| AG News | 0.79 (1.45) | **0.85** (2.64) |
| Yelp | 0.50 (2.24) | 0.53 (3.21) |
| ticket category | 0.93 (0.30) | **0.99** (0.22) |
| ticket urgency | **0.79** (0.80) | 0.67 (4.60) |
| ticket churn | **1.00** (0.00) | 0.86 (3.42) |
| ticket engineering | **0.95** (0.15) | 0.87 (2.25) |
| latency, 4 questions (interleaved A/B, p50) | **498ms** | 1267ms |

Accuracy is mixed (diffusion ahead on 4 of 7, behind on 2), latency is 2.5× in diffusion's favour because the
four slots are one decoder pass instead of four dependent steps, and the AR model's probabilities are far more
overconfident (NLL 2–4.6 on most rows). One AR model and one prompt family; not a general verdict.

## 7. Next steps

1. **Real tickets.** Everything in §6b is synthetic. A few hundred labelled real tickets would show whether the
   LoRA policy gain (urgency, engineering) survives real writing, and give the calibration set that matters.
2. **Guard against forgetting.** Mix a small share of public/general items into LoRA training, or keep one adapter
   per domain and load it per request (`JEVLIKE_ADAPTER`); AG News and Yelp dropped 4–5pt after the domain LoRA.
3. **Calibrate after LoRA.** T was fitted on the base model's reads; refit per question on the adapter's reads.
4. **Throughput.** MLX serves one request at a time. For concurrency, move to the vLLM path (OpenJev) and keep this
   repo's canvas/calibration/LoRA logic.
