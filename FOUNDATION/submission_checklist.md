# ✅ SUBMISSION CHECKLIST — THE EXAMINER
## Final Pre-Submit Validation · Complete in Under 10 Minutes
> **Validator runs this. Every item must be checked. One failure = do not submit.**

---

## MSR Checklist (All 9 — Every One is a Hard Disqualifier)

- [ ] **MSR-1 · OpenEnv Integration**
  - `ExaminerEnvironment` inherits from `openenv.core.Environment`
  - `reset()`, `step()`, `state()` follow OpenEnv API
  - `openenv.yaml` present
  - **Verify:** `grep -r "class ExaminerEnvironment(Environment)" examiner_env/`

- [ ] **MSR-2 · Training Script + Colab Notebook**
  - Colab notebook runs top-to-bottom without errors
  - Uses `GRPOTrainer` from TRL (not custom loop)
  - No local file paths
  - **Verify:** Open notebook in Colab → Runtime → Restart and Run All

- [ ] **MSR-3 · Real Training Plots**
  - `outputs/plots/reward_curve.png` exists and shows improvement
  - `outputs/plots/accuracy_curve.png` exists
  - Plots are from real W&B data (check timestamps)
  - **Verify:** Compare plot data against W&B dashboard

- [ ] **MSR-4 · Writeup / Video / Slides**
  - Blog post OR video published with accessible URL
  - Contains 3-sentence narrative, reward curve, before/after transcripts
  - Video under 2 minutes (if video)
  - **Verify:** Click link. Content loads. Non-technical person can follow.

- [ ] **MSR-5 · HuggingFace Space Live**
  - Space URL opens in incognito browser
  - Gradio interface loads
  - Can run a live episode
  - Results tab shows plots and transcripts
  - **Verify:** Open `https://huggingface.co/spaces/team/the-examiner` in incognito

- [ ] **MSR-6 · README Complete**
  - Problem statement present
  - Environment description present
  - Training approach present
  - Results with embedded plots present
  - Setup instructions with exact commands present
  - **Verify:** Read README. Can judge understand project in 60 seconds?

- [ ] **MSR-7 · README Links to HF Space**
  - HF Space link present in README
  - Link is clickable and opens live Space
  - **Verify:** Click the link. Space loads.

- [ ] **MSR-8 · README Links to All Materials**
  - Link to HF Space ✅
  - Link to blog/video/slides ✅
  - Link to training notebook ✅
  - All links return 200 (not 404)
  - **Verify:** Click every link. All resolve.

- [ ] **MSR-9 · No Video Files in HF Hub**
  - No .mp4, .avi, .mov, .webm, .mkv files in repo
  - Videos are external URLs (YouTube)
  - **Verify:** `git ls-files | grep -iE '\.(mp4|avi|mov|webm|mkv)$'` returns nothing

---

## Judging Criteria Evidence

| Criterion | Weight | Evidence Artifact | Location | Status |
|---|---|---|---|---|
| **Environment Innovation** | 40% | Novel env: diagnostic question design under adversarial deception | Environment code + README | [ ] |
| **Storytelling & Presentation** | 30% | Blog/video + README narrative + demo contrast | HF blog + README + HF Space | [ ] |
| **Showing Improvement** | 20% | Reward curve (3 phases) + before/after transcripts | `outputs/plots/` + `outputs/transcripts/` | [ ] |
| **Reward & Pipeline** | 10% | GRPO + Unsloth + 20-line reward function in Colab | Training notebook + `reward.py` | [ ] |

---

## Link Verification (Click Each One)

- [ ] HF Space: `https://huggingface.co/spaces/team/the-examiner` → loads
- [ ] Colab Notebook: `https://colab.research.google.com/...` → loads
- [ ] Blog Post: `https://huggingface.co/blog/...` → loads
- [ ] Demo Video: `https://youtube.com/...` → plays (if applicable)
- [ ] GitHub Repo: `https://github.com/team/the-examiner` → loads

---

## Final 5-Point Sanity Check

1. [ ] **Judge Test:** Open README fresh. Can you understand the project in under 60 seconds?
2. [ ] **Colab Test:** Runtime → Restart and Run All → completes without error?
3. [ ] **Space Test:** Open in incognito → all 3 tabs work?
4. [ ] **Evidence Test:** Reward curve shows clear improvement? Before/after transcripts show clear contrast?
5. [ ] **Clean Repo Test:** No `.env` committed? No video files? No `_quarantine/` committed?

---

## Sign-Off

| Team Member | Confirmed | Time |
|---|---|---|
| C1 | [ ] | |
| C2 | [ ] | |
| VAL | [ ] | |

**All 3 confirmed → SUBMIT**
