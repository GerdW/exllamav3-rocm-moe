# Merge notes: upstream exllamav3 v1.6.2 into exllamav3-rocm-moe

Written for the fork maintainer and for any AI assistant that works with this repository.
Everything below is a *record of decisions*; the commits themselves carry the detail.

## 0. Status (updated while the work proceeds)

- Work branch: `merge-upstream-v1.6.2` (fork `main` @ `1c049a13`, 2026-10-15 + upstream tag `v1.6.2` @ `4c84273`).
- Hardware used for verification: AMD Radeon AI PRO R9700 (gfx1201, 32 GB), Ryzen 9 7945HX3D, 2x48 GB DDR5-5200.
  **Not tested on gfx1100 (RX 7900 XTX).** That is the maintainer's hardware; please test there.

## 1. How the merge base was found

The fork root commit `159edc61` is an import of upstream `6b84a21b` (2026-09-20, "MSVC fix (__builtin_popcount)"),
which sits in upstream history 30 commits before `v1.5.2`. The import tree differs from `6b84a21b` in 2 files
(527 deleted lines). Because the fork has no shared git history with upstream, `git merge` refuses or produces
add/add conflicts everywhere. Reproduce a real three-way merge with:

```
git remote add upstream https://github.com/turboderp-org/exllamav3.git
git fetch upstream --tags
git replace --graft 159edc61 6b84a21b6f1e5da3f291b9e1019061f0de788279   # local only, not pushed
git checkout -b merge-upstream-v1.6.2 main
git config merge.conflictstyle zdiff3
git merge --no-commit --no-ff v1.6.2
```

The graft exists only in the local clone (refs/replace). The final merge commit has the two normal parents
(fork `main`, upstream `v1.6.2`) and needs no graft afterwards. Upstream is 317 commits ahead of the base.

## 2. Size of the problem

Trial merge: 52 conflicted files, 152 conflict hunks (zdiff3 counting).
- 21 files were changed by the fork **only in its ROCm/RDNA3 port commits** (not by later feature commits).
- 31 files were changed by the fork's later **feature commits** (MoE CPU experts, MTP, concurrency, GDN replay, ...).

## 3. Principle used

Upstream (since v1.6.0) has its own, much larger ROCm port: a portable PTX layer (`ptx_portable.cuh`,
`arch.cuh`), a `rocm/` source tree (WMMA GEMM, RDNA GEMV, compat headers), a hipify-aware source list in
`exllamav3/util/cuda_flags.py` and HIP-aware dispatch in `exl3_gemm.cu`. It solves the same problems as the
fork's port (inline-PTX emulation, `lop3`, `ldmatrix`, `__dp4a`, build flags) in a different way.

**Rule A (port-only files): take upstream's file unchanged.** The fork's emulation layer is superseded.
**Rule B (feature files): merge by hand, hunk by hunk; where a hunk only touches ROCm portability, take
upstream's way; where it carries fork behaviour (MoE CPU experts, concurrency, MTP, GDN replay, KV streaming,
tail skip, ...), keep the behaviour and re-express it on top of upstream's code.**
Every deviation is listed in section 5.

## 4. Method in practice (what was actually done, in order)

1. Real three-way merge with the graft above, `zdiff3` conflict style (shows the base).
2. Conflicted files changed by the fork only in its four port commits: taken from upstream (rule A).
3. **Every other file the fork touched only through port or RDNA3 commits** (not only the conflicted ones) was also reset to
   upstream's version (`git checkout v1.6.2 -- <file>`). Reason: git merged these without conflicts but left the fork's port
   edits (e.g. `EXL3_FULL_MASK`, `rocm_ldsm4`, `exl3_rdna3_*` calls) inside upstream's code, which no longer exists in upstream.
4. Conflicted files that carry fork features were then decided per family (section 5). Where upstream implements the same
   idea differently, upstream won in phase 1.
5. The fork's RDNA3-only kernels were **moved, not deleted**: `rocm/rdna3-fork-kernels/` holds `exl3_rdna3*.cu/.cuh`,
   `rdna3_attn.cu/.h`, `ablate.cu/.cuh` (17 files, `git mv`, history kept). They are outside `exllamav3_ext/`, so
   `cuda_flags.extension_sources()` no longer compiles them. They are the starting point for phase 2 on gfx11.
6. Result: the tree differs from upstream v1.6.2 in a small, reviewable set of files (list in section 6).

## 5. Decisions per file

| File | Decision |
|---|---|
| `.gitignore` | upstream's file + `*.so`, `build/` kept from the fork |
| 21 port-only files (see section 3) | upstream's file unchanged (rule A) |
| `architecture/qwen4_exp_mtp.py` | upstream's `lm_head_argmax(...)` call. The fork's `draft_sample` hook (Gumbel-coupled MTP drafting, `Job.coupled_sample`) was moved into `Model.lm_head_argmax` in `model/model.py`; it is applied to the logits **before** the vocabulary crop so the noise tensor has the same width as the target's sampler sees |
| `cache/recurrent.py` | upstream's `put()` -> `_insert()` refactor kept; the fork's `EXL3_STASH_LADDER` eviction (replay-cost-based victim) is re-added inside `_insert()` between "stranded" and "LRU" eviction; `t_use` stamping moved to `_insert()` (after `state.stash()`); upstream's `is_content_hash` helper kept |
| `exllamav3_ext/norm.cu` | fork's extra kernel args (`tail`, `abl`) kept, upstream's added closing brace of the launch macro kept |
| `exllamav3_ext/reduction.cuh` | upstream (it contains the same lane-0 fix as the fork) |
| `exllamav3_ext/hc_mix.cu` | upstream (`hc_warp_sum_lane0`, dangling-else braces) |
| `exllamav3_ext/graph.cu`, `graph.cuh` | **upstream, fork changes dropped.** The fork's `EXL3_NOGRAPH=mlp,gdn,moe,attn` (per-module eager decode) and its graph re-instantiation after 2048 kernel-node argument updates (`EXL3_GRAPH_REINSTANTIATE`, fixes fork issue #4, VRAM growth) are workarounds for the same ROCm problem that upstream now solves globally: `EXL3_GRAPHS` defaults to **off on ROCm** (each instantiated exec holds a kernarg pool that grows with every parameter update). Consequence: `EXL3_NOGRAPH` no longer exists; setting `EXL3_GRAPHS=1` on ROCm brings the leak back unless the re-instantiation logic is ported again |
| `libtorch/mlp.cpp`, `libtorch/blocksparse_mlp.cpp` | removed the fork's `graph_disabled_for(...)` guard (upstream's `graph.disabled` covers it) |
| `libtorch/attention.cpp` | upstream's eager QSA top-k path (`dsa_topk_prepare`, `s.graph->disabled`); the fork's `std::max(t_scan, 1)` guard kept on the `narrow` |
| GDN family: `modules/gated_delta_net.py`, `modules/gated_delta_net_fn/{conv1d,gated_delta_rule}.py`, `exllamav3_ext/{gdn.cu,gdn.cuh,bindings.cpp}`, `libtorch/gated_delta_net.{cpp,h}` | **upstream, fork changes dropped for phase 1.** Upstream has its own answer to the fork's `EXL3_GDN_REPLAY` (commit `c62775ee`, "2 recurrent states per slot instead of 5"): `c1136a02` "gdn: 2-slot parity scan pool with staged-input replay for spec rewinds", exposed as `batched_scan_replay`. The fork's `batched_state_replay` / `StateReplayJob` bindings are gone. Also dropped here (see section 6): `24f82f0f` DeltaNet prefill conv in place, and the RDNA3-only GDN megakernel / register-resident recurrence (`ef5975ae`, `8e3dbc99`) |
| CPU-MoE stack: `model/moe_cpu_host.py`, `modules/block_sparse_mlp_cpu.py`, `exllamav3_ext/cpu/moe_handoff.cu`, `cpu/moe_mul1.cpp`, `modules/block_sparse_mlp.py` | **upstream, fork changes dropped for phase 1.** Upstream v1.6.x has its own hot/cold expert placement (stats file `EXL3_MOE_CPU_SPLIT_STATS` or the profile saved by a previous run, `save_expert_stats`), promote-on-miss (`EXL3_MOE_CPU_PROMOTE`, spare GPU slots, DMA through the pinned arena), a fused split-issue (`exl3_moe_split_issue_fused`), a host-core reservation (`EXL3_MOE_HOST_CORES`) and 64-bit HIP stream memops. This overlaps the fork's `862cc6a4` (frequency-guided placement), `7a18518b` (adaptive placement), `3598d9cd` (persisted placement), `2922b761` (HIP stream memops) and `c00b4722`. Re-added in phase 2: **tail skip** (`886b4e2d`, needs a change in upstream's fused issue kernel) and the **standby nap** (`e4a82826`, PR #3 by loxdude, `cpu/moe_idle.h` stays in the tree). Dropped: the two-micro-batch overlap (`501c8c38`, only active on the RDNA3 decode path), `install_expert_aux_batch` |
| `modules/ngram_embedding.py` | upstream (it restructured the prefetch staging; the fork's `EXL3_NGRAM_PROF`, `EXL3_NGRAM_FADV` and next-chunk staging, commits `a83bdef7`, `d726a3cf`, are phase 2) |
| `modules/hyperconnections.py`, `modules/attention_fn/bc_attn.py`, `exllamav3_ext/quant/exl3_gemm.cu`, `exllamav3_ext/bindings.cpp` | upstream. RDNA3-only fork paths dropped here: the HIP split-attention kernel (`rdna3_attn.cu`), `exl3_rdna3_gemm` dispatch, fp16-source release for the RDNA3 prefill GEMM, `EXL3_ATTN_BLOCK_N/WARPS/STAGES` knobs |
| `generator/pagetable.py` | both: the fork's prefix-cache diagnostic print and upstream's `persistable` flag loop |
| `tests/test_moe_cpu_tiers_.py` | deleted (upstream deleted it) |
| `generator/generator.py`, `generator/job.py` | **upstream, fork changes dropped for phase 1** (they are re-added one by one in phase 2, see section 6): MTP prompt lookup (`EXL3_MTP_LOOKUP*`, `draft_headroom`, batched lookup), decode between prompt chunks (`_prefill_debt`), grouped prefill (`PrefillGroup`, off in the fork's served profile), two micro-batch decode (`_forward_micro_batches`, RDNA3 decode path only), the stall watchdog (`EXL3_WATCHDOG_S`), `_log_cache_wait`. Upstream meanwhile has its own hybrid drafting (`hybrid_draft_tokens`, suffix automaton lookup), `Generator.close()`, disk-tier idle flush and `max_draft_tokens`. The Gumbel-coupled draft hook lives in `model/model.py` (see above) and waits for its generator side (`iterate_draftmodel_mtp_gen`) |
| `generator/async_generator.py` | **both.** Upstream's idle flush (`has_idle_work()/idle_work()`) and `Generator.close()` plus the fork's threaded step (`EXL3_ASYNC_THREAD=1`: generator steps run on one worker thread, loop-side changes are queued in `_calls`). Upstream's "a rejected job must not stay tracked" is implemented for both modes: in threaded mode the worker records the refusal in `_rejected` and the loop thread untracks the job and delivers the exception (`_deliver_rejected`) |
| `README.md` | fork's README kept; `README.exllamav3.md` refreshed to upstream v1.6.2's README |

## 6. What the merged tree still contains from the fork, and what is waiting for phase 2

**Kept (differs from upstream v1.6.2):**
`cache/recurrent.py` (checkpoint ladder eviction), `cache/kv_stream.py`, `exllamav3_ext/kv_stream.cu/.cuh` + bindings and
`cache/qsa.py`, `modules/attn.py`, `modules/qsa_indexer.py`, `modules/attention_fn/qsa_triton.py` (KV streaming for the QSA layers),
`generator/async_generator.py` (threaded step, `EXL3_ASYNC_THREAD=1`), `generator/pagetable.py` (diagnostic print),
`model/model.py` (`draft_sample` hook), `modules/moe_batch_recon.py` (fp16 down projection, `EXL3_MOE_RECON_DOWN_HALF`),
`modules/ple.py` (`prefetch_next` param), `loader/safetensors.py` (`POSIX_FADV_RANDOM` for n-gram row gathers),
`util/memory.py`, `architecture/{glm4v,qwen2_5_vl,qwen3_vl}.py` + `architecture/mm_processing/qwen2.py` (pad images smaller than one
merged patch instead of failing the request), `cpu/moe_idle.h` + `tests/test_moe_cpu_idle_.py` (files only; not wired yet),
`generator/adaptive_window.py`, `generator/prefill_group.py`, `util/ablation.py` (files only; not wired).
Plus fork-only trees: `rocm/` (scripts, TabbyAPI patches and configs, `rdna3-fork-kernels/`), `rocm_tests/`, `README.md`,
`README.exllamav3.md` (upstream's README).

**Phase 2 backlog** (each item = one commit with a measurement; fork commit hashes in brackets):
1. Decode between prompt chunks, generator off the event loop [`f5a558d8`] - this is the part that keeps a short request
   responsive while a very long prompt is being read (measured on gfx1201 with two slots: first token of the short request
   16.5 s with the fork vs 53 s with plain upstream while a 142k prompt is read).
2. Gumbel-coupled MTP drafts and prompt lookup, batched [`4af70aa8`, `b3673605`, `bb926c1b`] (`Model.lm_head_argmax` hook is in place).
3. Tail skip for CPU-resident experts [`886b4e2d`, `960ed40e`]: needs the change in upstream's fused split-issue kernel.
4. Standby nap for the CPU MoE workers [`e4a82826`, fork PR #3]: `cpu/moe_idle.h` is in the tree, the worker-loop hooks are not.
5. Next-chunk n-gram staging and the n-gram prefetch switches [`d726a3cf`, `a83bdef7`, `92a0990d`] (`prefetch_next` in `ple.py` is in place).
6. Stall watchdog [`32f8418f`, `aef16ef3`] and the cache-wait log line [`1c049a13`].
7. DeltaNet prefill conv in place [`24f82f0f`], gated residual single kernel [`b247165c`], MoE prefill batched down projection [`1ab433ec`], QSA prefill single-stage pipelining [`3f086b8a`].
8. RDNA3 (gfx11) kernels in `rocm/rdna3-fork-kernels/` and the dispatch hooks for them: only for the maintainer's hardware; each needs
   a benchmark against upstream's `rocm/` kernels (`rocm_tests/test_rdna3_*.py`).
9. Graph handling: if CUDA-graph decode on ROCm is wanted back (`EXL3_GRAPHS=1`), port the re-instantiation after N argument updates [`6fd81152`].

## 7. Verification log

(see the end of this file; filled in as tests are run)
