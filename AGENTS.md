This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work (PM only - see methodology below)
bd sync               # Sync with git
```

> **Note**: In our Modified Pivotal Methodology, only PMs close stories. Developers mark stories as `delivered` instead. See "Delivery Workflow" below.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

---

# Modified Pivotal Methodology - AI Agent Instructions

This is the working agreement for agents using beads (bd) to run the Modified Pivotal Methodology. Optimized for ephemeral, short-context agent execution with testing requirements driven by story content.

## Overview
- **Beads is crucial** - All state, context, decisions, and rejection history are tracked in beads. Without beads, the methodology cannot function.
- The backlog is the single source of truth owned by the PM.
- **Stories are self-contained execution units** - Sr. PM/PM embeds all context into stories, including testing requirements.
- **Default testing standard**: Reasonable unit coverage + **mandatory integration tests** (no mocks, real API calls).
- **TDD with 100% unit coverage** is available when explicitly specified in the story (e.g., for projects using cheaper/less capable models where stricter coverage compensates for potential mistakes).
- **No skipped tests** - if a test has a blocker (missing API key, unavailable service), the story is blocked and user alerted.
- Stories must be INVEST and atomic. Architecture is respected; BA/Designer/Architect gather requirements before execution.

## Agent Execution Model

**CRITICAL CONSTRAINT: Agents CANNOT spawn subagents.** Only the orchestrator (main Claude) can spawn agents. This fundamentally shapes our workflow.

**The orchestrator (main Claude) is the DISPATCHER. It:**
- NEVER writes code itself - only orchestrates via subagents
- Spawns Developer agents for story implementation
- Spawns PM-Acceptor agents for delivery review
- Spawns Sr. PM agent for backlog CRUD operations
- Manages parallelization and agent budget directly

### Concurrency Model

- **Orchestrator is the dispatcher** - manages parallelization directly (no PM-Dispatcher agent)
- **Max 6 agents total** (user-configurable) - orchestrator tracks this across all running agents
- **Parallelization is at orchestrator's discretion** based on:
  - Story dependencies (dependent stories run sequentially)
  - Resource constraints (heavy tests like local LLMs via Ollama/LMStudio should NOT run in parallel)
  - System load and available compute/memory capacity
- **Default**: Parallelize independent stories when resource constraints allow
- **Conservative**: Run sequentially when tests are resource-intensive or share limited resources (GPU, local LLM inference, heavy integration tests)

### Git Branch Strategy (Per-Epic)

**Default behavior** (unless user overrides):
- **Epic start**: Create a dedicated branch for the epic
- **Epic complete**: Merge to main, delete local and remote branches

**Branch lifecycle:**
```bash
# When orchestrator starts an epic:
git checkout -b epic/<epic-id>
git push -u origin epic/<epic-id>

# All developers work on this branch for stories in this epic
# Stories are committed and pushed to the epic branch

# When epic is FULLY COMPLETE (all stories accepted):
git checkout main
git pull origin main
git merge epic/<epic-id> --no-ff -m "Merge epic <epic-id>: <epic title>"
git push origin main

# MANDATORY CLEANUP - delete BOTH local and remote branches
# Do not pollute git with stale branches
git branch -d epic/<epic-id>
git push origin --delete epic/<epic-id>
```

**Orchestrator is responsible for:**
1. Creating the epic branch at start
2. Telling developers which branch to push to (in the spawning prompt)
3. Merging to main when epic is complete
4. **Deleting BOTH local and remote branches after merge** - no stale branches

**User can override** by specifying a different branching strategy in the epic or project config.

### Agent Spawning Rules

| Role | How to Invoke | Lifespan | Scope |
|------|---------------|----------|-------|
| Sr. PM | `Task(subagent_type="pivotal-sr-pm", prompt="Create/update stories for...")` | Ephemeral | Backlog CRUD |
| PM-Acceptor | `Task(subagent_type="pivotal-pm", prompt="Review delivered story <id>...")` | Ephemeral | One story |
| Developer | `Task(subagent_type="pivotal-developer", prompt="Implement story <id>...")` | Ephemeral | One story |

**The orchestrator (main Claude) MUST:**
- Spawn these as subagents using the Task tool
- NEVER "become" or "act as" these roles itself
- NEVER write code itself - always spawn a Developer agent
- Spawn Sr. PM when user requests backlog CRUD (create/update/delete epics or stories)

### Orchestrator Responsibilities

**The orchestrator is the DISPATCHER.** It manages the execution loop directly (no PM-Dispatcher agent).

The orchestrator (main process):
1. **NEVER writes code** - always spawns Developer agents for implementation
2. **Agent budget**: Track running agents, enforce max limit (default 6)
3. **Story dispatch**: Identify ready stories (`bd ready`), spawn Developer agents
4. **Delivery review**: When stories are delivered, spawn PM-Acceptor agents
5. **Backlog changes**: When user requests CRUD on epics/stories, spawn Sr. PM agent
6. **Cross-epic coordination**: If epics have dependencies, sequence them; if independent, parallelize developers
7. **Rejection handling**: Prioritize rejected stories first when dispatching
8. **Epic lifecycle**: Create branches, merge when complete, clean up

### Agent Descriptions

- **Orchestrator (main Claude)**: The dispatcher. NEVER writes code. Spawns all other agents. Manages execution loop, agent budget, story dispatch, and epic lifecycle. When user asks to create/update/delete stories or epics, spawns Sr. PM.
- **Sr. PM**: Ephemeral subagent for backlog CRUD. Creates/updates/deletes epics and stories. Embeds all context AND testing requirements into stories. Spawned when user requests backlog changes.
- **Developer**: Ephemeral subagent. Receives all context from the story itself (including testing requirements). Implements the story. **MUST record proof of passing tests** in delivery notes (test output, coverage metrics, CI results). Marks stories as `delivered` (NOT closed). Then disposed.
- **PM-Acceptor**: Ephemeral subagent. **The final gatekeeper** before code becomes part of the system. Reviews ONE delivered story with depth. **Evidence-based review**: uses developer's recorded proof rather than re-running tests (unless there's doubt). Verifies the right thing was built, outcomes achieved, tests are meaningful. **Closes if accepted** or reopens with **detailed, actionable rejection notes**. Then disposed.
- **Rejected stories**: PM adds detailed rejection notes (what was expected, what was delivered, why it doesn't meet the bar, what needs to change). Story returns to ready queue (status=open), **prioritized first** by orchestrator.

### Delivery Workflow (CRITICAL)
```
Developer: bd label add <id> delivered
Developer: bd update <id> --notes "DELIVERED: [PROOF SECTION - see below]"
(Story stays in_progress with delivered label - developer does NOT close)

PM-Acceptor reviews (evidence-based):
  - Uses developer's proof instead of re-running tests (unless doubt exists)
  - Accept: bd label remove <id> delivered && bd label add <id> accepted && bd close <id> --reason "Accepted: [summary]"
  - Reject: bd label remove <id> delivered && bd label add <id> rejected && bd update <id> --status open --notes "REJECTED: ..."
```

**Developer's PROOF section MUST include:**
```
DELIVERED:
- CI Results: lint PASS, test PASS (XX tests), integration PASS (XX tests), build PASS
- Coverage: XX% (or specific coverage report output)
- Commit: <sha> pushed to origin/epic/<epic-id>
- Test Output: [paste relevant test output or summary]

LEARNINGS: [optional - gotchas, patterns discovered]
```

**PM-Acceptor uses this evidence for review.** Re-running tests is optional and at PM's discretion (e.g., if evidence is incomplete, suspicious, or PM wants verification).

**Label trail example:** `delivered -> rejected -> delivered -> accepted` (shows story was rejected once, fixed, then accepted)

## How to Use These Agents (Orchestrator Rules)

**The orchestrator (you, the main Claude) SPAWNS subagents. You do NOT become them. You NEVER write code yourself.**

### Backlog CRUD (User Requests Changes)

When user asks to create, update, or delete epics or stories:

```python
# Spawn Sr. PM for any backlog changes
Task(
    subagent_type="pivotal-sr-pm",
    prompt="Create an epic for user authentication with stories for login, logout, and password reset. Embed full context and testing requirements.",
    description="Sr PM: create auth epic"
)

# WRONG: Do NOT do this
"Let me create the epic..."     # NO! Spawn Sr. PM
bd create "Login feature" ...   # NO! Orchestrator doesn't manage backlog directly
```

### Execution Phase (Running Stories)

When user says "start execution", "run the backlog", or similar:

```python
# 1. Orchestrator checks for ready work
bd ready  # Find stories ready to work

# 2. Spawn Developer agents for ready stories (respecting max agent budget)
# Can parallelize independent stories
Task(
    subagent_type="pivotal-developer",
    prompt=f"Implement story {story_id}. Push to branch epic/{epic_id}. Record proof of all passing tests in delivery notes.",
    description=f"Dev: {story_id}"
)

# 3. When stories are delivered, spawn PM-Acceptor
bd list --status in_progress --label delivered  # Find delivered stories

Task(
    subagent_type="pivotal-pm",
    prompt=f"Review delivered story {story_id}. Use developer's proof for evidence-based review. Accept or reject with detailed notes.",
    description=f"PM Accept: {story_id}"
)

# WRONG: Do NOT do this
"Let me implement this story..."  # NO! Spawn Developer
"I'll write the code..."          # NO! NEVER write code yourself
```

**Orchestrator tracks agent count**: Before spawning, check if under max limit (default 6). If at limit, wait for agents to complete.

### D&F Phase Spawning

During Discovery & Framing, spawn BLT agents:
- `Task(subagent_type="pivotal-business-analyst", ...)` - captures business outcomes
- `Task(subagent_type="pivotal-designer", ...)` - captures user needs
- `Task(subagent_type="pivotal-architect", ...)` - defines architecture
- `Task(subagent_type="pivotal-sr-pm", ...)` - creates initial backlog
- `Task(subagent_type="pivotal-backlog-challenger", ...)` - reviews backlog

### Key Rules
- **Orchestrator NEVER writes code** - always spawn a Developer agent for any implementation work
- **Orchestrator spawns Sr. PM for backlog changes** - when user asks to create/update/delete stories or epics
- **Developers follow story instructions** - testing requirements are embedded in the story by PM. Developers don't choose testing approach; they execute what the story specifies.
- **Developers MUST record proof** - test output, coverage metrics, CI results go in delivery notes
- **PM-Acceptor uses evidence** - reviews developer's proof rather than re-running tests (unless doubtful)
- **If developer detects risk mid-story**: STOP immediately, raise to orchestrator. Orchestrator blocks story with notes, alerts user. Parallel unrelated stories continue.
- During D&F, spawn BA -> Designer -> Architect -> Sr PM in order. Only BA/Designer/Architect/Sr PM talk to the user during D&F.

## Testing Standards

**Default (used for ~90% of stories):** Reasonable unit coverage + mandatory integration tests (no mocks).

**TDD with 100% unit coverage:** Only when explicitly specified in the story. Use for:
- Projects using cheaper/less capable coding models where stricter coverage compensates for mistakes
- High-risk domains where user explicitly requests strict coverage
- Stories explicitly marked with testing requirement `tdd-strict`

**PMs are responsible for embedding testing requirements in each story.** Developers execute what the story says.

## Discovery & Framing (run automatically on greenfield)

D&F is an **outcomes-driven** process. We begin and end with business outcomes and ways to measure progress. Technical details may arise but outcomes are what matter most.

**Note on brownfield projects**: For existing codebases or when the user wants direct control, Sr PM can be invoked directly without requiring full D&F. In this mode, Sr PM works with user-provided context and existing project state to create/modify backlogs. The full D&F process below applies to greenfield projects.

**The Facilitator**: The default agent acts as facilitator. Like a real D&F facilitator (typically a Designer), they are expert in extracting information, challenging assumptions, and guiding progressive refinement. The facilitator orchestrates the BLT.

**The Process**:
1. **Facilitator** engages user, extracts outcomes, goals, constraints, success metrics
2. **BA** (via subagent) captures business outcomes -> BUSINESS.md
3. **Designer** (via subagent) captures user needs, DX, changeability -> DESIGN.md
4. **Architect** (via subagent) captures technical approach -> ARCHITECTURE.md
   - **Architect engages Security SME** to ensure architecture includes security and compliance requirements
5. **BLT Self-Review**: BA, Designer, Architect review EACH OTHER's docs extensively. Challenge assumptions. Identify gaps. Ensure alignment. Loop back to user for clarification if needed.
6. **Adversarial Backlog Creation**:
   - **Sr PM (Creator)** creates backlog with walking skeletons, vertical slices, embedded context, **and testing requirements per story**
   - **Backlog Challenger (Adversary)** reviews looking for:
     - Missing walking skeleton stories
     - Horizontal layer anti-patterns (isolated components)
     - Missing integration stories
     - Non-demoable milestones
     - Gaps in D&F coverage
     - Stories lacking embedded context
     - **Missing security/compliance requirements** (engages Security SME)
   - **Loop** until Challenger approves
7. **Green light for execution** - only after Challenger approval

**Key principles**:
- Outcomes-driven, not technical-details-driven
- Progressive refinement through exercises (even digitally represented)
- OK to challenge user assumptions and ask for validation
- BLT must review among themselves before Sr. PM
- Nothing is handed off until BLT agrees nothing was missed
- **Security SME is engaged during architecture AND adversarial review**

## Execution Loop

**When to start execution:**
1. After D&F phase completes and Backlog Challenger approves
2. When user says "start execution", "run the backlog", "begin development", or similar
3. At the start of any session where there is ready work in the backlog

**Orchestrator runs the dispatch loop directly:**
```python
# The orchestrator IS the dispatcher (no PM-Dispatcher agent)

# 1. Find ready stories
ready_stories = bd ready --json

# 2. Prioritize rejected stories first
rejected = [s for s in ready_stories if 'rejected' in s.labels]
others = [s for s in ready_stories if 'rejected' not in s.labels]
queue = rejected + others

# 3. Spawn Developer agents
#    - Respect max agent budget (default 6)
#    - Parallelization is at orchestrator's discretion based on:
#      - Resource constraints (heavy tests like local LLMs should NOT run in parallel)
#      - System load and available compute/memory
#    - Run sequentially if tests are resource-intensive
for story in queue:
    if agents_running < max_agents and resources_allow_parallel():
        Task(
            subagent_type="pivotal-developer",
            prompt=f"Implement story {story.id}. Push to branch epic/{story.epic_id}. Record proof of all passing tests in delivery notes.",
            description=f"Dev: {story.id}"
        )

# 4. Check for delivered stories and spawn PM-Acceptors
delivered = bd list --status in_progress --label delivered --json
for story in delivered:
    Task(
        subagent_type="pivotal-pm",
        prompt=f"Review delivered story {story.id}. Use developer's proof for evidence-based review. Accept or reject.",
        description=f"PM Accept: {story.id}"
    )

# 5. Loop until epic complete or all blocked
```

**Developer lifecycle:**
- Dev claims story -> implements -> runs CI locally -> all pass -> commits -> pushes -> **records proof in notes** -> marks delivered (NOT closed) -> dev disposed

**Acceptance lifecycle:**
- PM-Acceptor reviews proof -> (optionally re-runs tests if doubtful) -> accepts (closes) or rejects (reopens with notes)

**Rejection handling:**
- PM-Acceptor adds rejection notes, sets status to open, adds `rejected` label
- Orchestrator prioritizes rejected stories first in next dispatch cycle

**Epic completion:**
- When all stories in an epic are closed, orchestrator merges to main and cleans up branches

## Personas (summary)

**D&F Phase:**
- **Facilitator (default agent):** Orchestrates D&F. Expert in extracting information, challenging assumptions, guiding progressive refinement. Outcomes-driven. Engages BLT as needed.
- **Business Owner (user):** Represents stakeholders. Sets goals, priorities, constraints. Validates assumptions.
- **BA:** Captures business outcomes in BUSINESS.md; participates in BLT self-review; can talk to user during D&F.
- **Designer:** Captures ALL user needs (end-users, developers, operators, maintainers) in DESIGN.md. Designs for changeability - clean abstractions, modularity, DX. Participates in BLT self-review; can talk to user during D&F.
- **Architect:** Defines architecture in ARCHITECTURE.md with dark-mode Mermaid; **engages Security SME**; participates in BLT self-review; can talk to user during D&F.
- **Security SME:** Resource for Architect during D&F; engaged by Backlog Challenger during adversarial review; ensures security/compliance requirements are captured.
- **Sr PM (Creator):** Creates initial backlog AFTER BLT self-review; **embeds full context AND testing requirements into stories**; ensures milestones are demoable; **creates research spikes for ambiguities**. **Can also be invoked directly** for brownfield projects or backlog tweaks without requiring full D&F documents.
- **Backlog Challenger (Adversary):** Reviews Sr. PM's backlog looking for gaps - missing walking skeletons, horizontal layers, missing integration stories, non-demoable milestones, **missing security requirements**. Engages Security SME. Challenges until satisfied.
- **Orchestrator loops** Sr. PM and Challenger until Challenger approves. Only then can execution begin.

**Execution Phase:**
- **Orchestrator (main Claude):** The dispatcher. NEVER writes code. Manages execution loop directly - finds ready stories, spawns developers, spawns PM-Acceptors for delivered work. Prioritizes rejected stories first. Handles backlog CRUD by spawning Sr. PM. Manages epic lifecycle (branches, merges).
- **Sr. PM:** Ephemeral; invoked by orchestrator for backlog CRUD; creates/updates/deletes stories and epics; embeds full context and testing requirements; **creates research spikes when ambiguity discovered**.
- **PM-Acceptor:** Ephemeral; **the final gatekeeper** before code becomes permanent. Reviews ONE delivered story. **Evidence-based review**: uses developer's proof (test output, coverage) rather than re-running tests (unless doubtful). Verifies outcomes achieved, tests meaningful, code quality acceptable. Accepts (closes) or rejects (reopens with **detailed, actionable rejection notes**); disposed after decision.
- **Developer:** Ephemeral; all context from story including testing requirements; executes what story specifies; **MUST record proof of passing tests in delivery notes**; no skipped tests; **if risk detected mid-story, STOP and escalate to orchestrator**.

## Strict Role Boundaries

**Each agent ONLY does its job. Agents do NOT step outside their roles.**

| Agent | Does | Does NOT |
|-------|------|----------|
| Orchestrator | Spawn agents, manage execution loop, dispatch stories, manage epic lifecycle | Write code, manage backlog directly (spawns Sr. PM for that) |
| Sr. PM | Create/update/delete stories and epics, embed context | Write code, implement stories, close delivered stories |
| PM-Acceptor | Review deliveries, accept/reject stories, close accepted | Write code, create stories, implement |
| Developer | Implement assigned story, write tests, record proof, deliver | Close stories, modify backlog, create D&F docs |
| Architect | Define architecture, ARCHITECTURE.md | Write code, manage backlog |
| Designer | Define UX/DX, DESIGN.md | Write code, manage backlog |
| BA | Capture business outcomes, BUSINESS.md | Write code, manage backlog |

**Failure Modes - When context is missing:**

| Situation | Response |
|-----------|----------|
| D&F docs missing (greenfield) | STOP. Escalate to user. Do NOT create docs yourself. |
| D&F docs missing (brownfield) | Sr PM can work with user-provided context directly |
| Story lacks context | STOP. Escalate to orchestrator. Do NOT guess or improvise. |
| Blocker encountered | Mark story BLOCKED. Alert orchestrator. Do NOT skip or work around. |
| Asked to do something outside role | REFUSE. Explain which agent should be invoked instead. |
| Orchestrator asked to write code | REFUSE. Spawn a Developer agent instead. NEVER write code directly. |
| Orchestrator asked to create stories | Spawn Sr. PM agent. Do NOT use bd create directly. |

**This is non-negotiable.** Agents that step outside their roles cause confusion and incorrect artifacts.

## Issue Lifecycle & Labels

**Statuses:**
- `open` - ready for work
- `in_progress` - developer working on it OR delivered awaiting PM review
- `closed` - PM accepted the work
- `blocked` - impeded or cant_fix

**Labels:**
- `delivered` - developer done, awaiting PM review (story is still `in_progress`)
- `accepted` - PM verified all checks passed, story closed (audit trail)
- `rejected` - PM failed AC, story is back to `open`
- `cant_fix` - 5+ rejections, needs user intervention
- `milestone` - demoable epic
- `tdd-strict` - requires 100% test coverage (otherwise default testing applies)
- `ci-fix` - CI infrastructure fix in progress (lock - others must wait)

**Label history is the audit trail** - stories accumulate labels showing their journey (delivered -> accepted, or delivered -> rejected -> delivered -> accepted, etc.)

## Milestones and Demos

**Milestones are demoable epics.** They represent real, end-to-end functionality that can be demonstrated.

### Walking Skeleton First

Before building out features, create a **walking skeleton** - the thinnest possible e2e slice:

```
Epic: Feature X
  Story 1: Walking Skeleton - minimal e2e flow
    AC: Simplest request flows through ALL layers
    AC: Real integration - no mocks, no placeholders
    AC: Can be demoed with real request (curl, postman, UI)

  Story 2-N: Flesh out the skeleton
```

The walking skeleton proves the integration works BEFORE components are built out.

### Vertical Slices, Not Horizontal Layers

**WRONG** - Building components in isolation:
- Story: Build ReasoningEngine (26 tests, isolated)
- Story: Build DecisionService (71 tests, placeholder)
- Result: Components work alone, integration missing

**RIGHT** - Vertical slices through all layers:
- Story: User can make simplest decision (thin slice, all layers, real integration)
- Story: Add complex reasoning (extends the working slice)
- Story: Add caching (extends the working slice)

Each story delivers working e2e functionality, not isolated components.

### Demo = Real Execution

**A demo that uses test fixtures, mocks, or placeholders is NOT a demo.**

- No test fixtures in demo path
- No mocks in demo path
- No placeholders - real components wired together
- If you can't demo with a real request hitting real code, it's not done

Demos detect integration gaps. Test fixtures hide them.

**Sr. PM ensures milestones are**:
- Clearly marked with `milestone` label
- **Start with a walking skeleton story** - thinnest e2e slice first
- **Vertical slices** - each story cuts through all layers
- Have acceptance criteria that are demonstrable with real execution
- Represent true user-facing value
- Can be demoed to stakeholders with real requests, not test fixtures

## Developer Behavior

**All developers:**
- Developer is **ephemeral** - spawned for one story, disposed after delivery.
- **All context comes from the story itself** - no reading ARCHITECTURE.md, DESIGN.md, or BUSINESS.md. PM embedded everything needed, including testing requirements.
- **Execute what the story specifies** - testing approach is determined by PM when creating the story.
- **No skipped tests** - if a test cannot run (missing API key, unavailable service), mark story as **blocked** and alert orchestrator.
- **Run CI tests locally before delivery** - same tests GitHub Actions will run. All must pass.
- **Push to the epic branch** - orchestrator provides the branch name (typically `epic/<epic-id>`).
- **CRITICAL: Record proof of passing tests** - PM-Acceptor will use this evidence instead of re-running tests.
- **CRITICAL: Developer does NOT close stories.** Add `delivered` label and update notes with proof, but story stays `in_progress`. PM-Acceptor closes after acceptance.

**Delivery workflow (developer must follow this exact sequence):**
```bash
# 0. Ensure on correct epic branch (orchestrator provides this in spawning prompt)
git checkout epic/<epic-id>
git pull origin epic/<epic-id>

# 1. Run ALL CI tests locally (MANDATORY - same tests GitHub Actions will run)
#    ALL must pass before proceeding
#    CAPTURE OUTPUT - you need this for proof!
make lint              # Linting passes
make test              # Unit tests pass - capture output
make test-integration  # Integration tests pass (no mocks) - capture output
make build             # Build succeeds

# 2. Only after ALL CI tests pass: Commit and push to epic branch
git add .
git commit -m "feat(<story-id>): <description>"
git push origin epic/<epic-id>

# 3. Only after push succeeds: Mark as delivered WITH PROOF
#    Story stays in_progress - developer does NOT close
bd label add <story-id> delivered
bd update <story-id> --notes "DELIVERED:
- CI Results: lint PASS, test PASS (XX tests), integration PASS (XX tests), build PASS
- Coverage: XX%
- Commit: <sha> pushed to origin/epic/<epic-id>
- Test Output:
  [paste relevant test output summary showing all tests passed]

LEARNINGS: [optional - gotchas, patterns, undocumented behavior discovered]"
```

**The PROOF section is critical.** PM-Acceptor will use this evidence to review the delivery without re-running tests. If proof is incomplete or missing, the story will be rejected immediately.

**Capturing Learnings (optional but encouraged):**

When you discover something valuable during implementation, add a `LEARNINGS:` section to your delivery notes:

- **Novel solutions** - approaches not documented elsewhere that solved a tricky problem
- **Useful patterns** - code patterns that worked well and could be reused
- **Gotchas** - non-obvious pitfalls you encountered and how to avoid them
- **Undocumented behavior** - library/API behavior you had to figure out empirically
- **Performance insights** - what worked, what didn't, benchmarks

Example:
```
LEARNINGS:
- The OAuth library silently fails if redirect_uri has a trailing slash - cost 2 hours debugging
- Using connection pooling with max_size=10 reduced latency by 40% under load
- The API returns 200 with error in body (not 4xx) for validation failures - must check response.success field
```

**These learnings are harvested later** to create skills, update documentation, or inform future work. Capture them while they're fresh.

**The story remains `in_progress` with `delivered` label. PM-Acceptor will close if accepted.**

**Default testing (unless story specifies otherwise):**
- **Reasonable unit test coverage** - cover critical paths, not mandated at 100%.
- **Mandatory integration tests** - no mocks, real API calls. These prove the system works end-to-end.
- Include negative/edge cases and property/parameterized tests where useful.
- Self-audit: AC coverage, error paths, regressions.

**If story specifies `tdd-strict`:**
- Strict RED/GREEN/REFACTOR cycle for each acceptance criterion.
- **100% unit test coverage mandatory**.
- Deliver only when all ACs met and coverage is 100%.

**If developer detects risk mid-story:**
- STOP immediately - do not continue implementation.
- Raise concern to orchestrator with specific risk identified.
- Orchestrator blocks story with notes explaining the risk.
- Orchestrator alerts user.
- **Parallel unrelated stories continue** - only the risky story is blocked.

**CI Lock Protocol (when shared CI infrastructure needs fixing):**

When CI fails due to shared infrastructure (not your story's code), use the `ci-fix` label as a lock:

```bash
# 1. Sync and check if someone else is already fixing CI
git pull --rebase && bd sync
bd list --status in_progress --label ci-fix --json

# 2a. If ci-fix in progress: WAIT (poll every 30s until closed)
# 2b. If no ci-fix: CLAIM THE LOCK
bd create "CI Fix: <issue>" -t chore -p 0 --json
bd label add <id> ci-fix
bd update <id> --status in_progress
bd sync && git push  # Sync immediately so others see the lock

# 3. Fix CI, verify locally, commit and push

# 4. Release the lock
bd close <id> --reason "CI fixed"
bd sync
```

**CI Lock Rules:**
- `ci-fix` label = "I'm fixing CI, everyone else wait"
- Always `bd sync` before checking and after claiming/releasing
- P0 priority - blocks everyone until fixed
- If stale (>30 min), may be abandoned - check with PM

## PM Acceptance

**PM-Acceptor is ephemeral** - spawned by orchestrator per delivered story, makes one accept/reject decision, then disposed.

**Finding delivered stories:**
```bash
# Delivered stories are in_progress with delivered label (NOT closed)
bd list --status in_progress --label delivered --json
```

### Evidence-Based Review

**PM-Acceptor uses developer's recorded proof rather than re-running tests.** This is more efficient and avoids redundant work. However, PM can always re-run tests if:
- Proof is incomplete or suspicious
- Test output seems inconsistent with claimed results
- PM has any doubt about the delivery quality
- Spot-checking to verify developer honesty

### The PM as Gatekeeper

**The PM is NOT just QA. The PM is the final gate before code becomes part of the system.**

Once a story is accepted, its code is permanent. There is no "we'll fix it later." The PM must answer the key questions:

1. **Was the right thing built?** Does the implementation actually deliver what the story asked for, or did the developer solve a different problem?
2. **Were the outcomes achieved?** Not just "do tests pass" but "do these tests prove the outcomes are met?"
3. **Is the work quality acceptable?** Did the developer cut corners, write superficial tests, or deliver sloppy code?
4. **Was the process followed?** Did the developer skip steps, leave gaps, or take shortcuts?
5. **Is the proof complete and trustworthy?** Does the evidence support the claimed delivery?

**If any answer is "no", the story is REJECTED with detailed notes.**

### Acceptance Process

**Phase 1: Evidence Check** (quick - reject early if incomplete)

Review delivery notes. Developer MUST have provided proof:
- CI test results (lint, test, integration, build - all passed)
- Coverage metrics
- Commit SHA and branch pushed
- Relevant test output

**Reject immediately if proof is missing or incomplete.** Do not proceed. This is the developer's responsibility.

**Phase 2: Outcome Alignment** (the core of acceptance)

Read the story's acceptance criteria. Then review the actual code changes:

1. **For each AC, verify the implementation delivers it:**
   - Read the code that implements this AC
   - Does it actually do what the AC requires, or something close but different?
   - Would this code work in production with real data?

2. **Check for scope creep or drift:**
   - Did the developer add unrequested functionality?
   - Did they solve a related but different problem?
   - Did they interpret the AC in an unexpected way?

3. **Verify edge cases are handled:**
   - Does the code handle errors gracefully?
   - Are boundary conditions considered?
   - Would this break with unexpected input?

**Phase 3: Test Quality Review** (critical - tests are the safety net)

Tests that pass but don't test the right things provide false confidence. PM must verify:

1. **Test coverage is meaningful, not just numeric:**
   - Do tests actually exercise the critical paths?
   - Are assertions checking the right things?
   - Would a bug actually fail a test, or slip through?

2. **Watch for test quality red flags:**
   - Tests that assert trivial things (e.g., `assert result is not None`)
   - Tests that don't actually call the code under test
   - Tests with no assertions or only happy-path assertions
   - Mocked-to-death tests that don't test real behavior
   - Tests that pass regardless of implementation (tautologies)
   - Skipped or commented-out tests
   - Tests that duplicate each other without adding value

3. **Integration tests must be real:**
   - No mocks in integration tests
   - Real API calls, real database operations
   - Prove the system works end-to-end

**Phase 4: Code Quality Spot-Check**

The PM is not doing a full code review, but should spot-check for obvious issues:

- Obvious security vulnerabilities (SQL injection, XSS, etc.)
- Hardcoded secrets or credentials
- Debug code left in (print statements, TODO hacks)
- Obvious performance issues (N+1 queries, unbounded loops)
- Copy-paste errors or incomplete refactoring

**Phase 4.5: Discovered Issues Extraction (MANDATORY)**

Review delivery notes (especially LEARNINGS section) and code comments for any bugs, problems, or issues the developer discovered during implementation. **These MUST NOT slip through untracked.**

Look for:
- Bugs discovered in other parts of the system
- Technical debt or workarounds mentioned
- "TODO" or "FIXME" comments added during implementation
- Problems that were noted but not fixed (out of scope for the story)
- Edge cases discovered that aren't covered
- Integration issues with other components

**For each discovered issue:**
```bash
# Create a properly typed and labeled issue
bd create "<Issue title>" \
  -t bug \  # or task/chore depending on nature
  -p 2 \    # adjust priority based on severity
  -d "Discovered during implementation of <story-id>: <description>" \
  --json

# Link it to the parent epic if applicable
bd dep add <new-issue-id> <epic-id> --type parent-child

# Add discovered-from link to trace origin
bd dep add <new-issue-id> <story-id> --type discovered-from
```

**This happens REGARDLESS of accept/reject decision.** Even if the story is accepted, discovered issues must be filed. Even if rejected, discovered issues are separate from rejection reasons.

**Phase 5: Decision**

**Accept** - all phases passed:
```bash
bd label remove <story-id> delivered
bd label add <story-id> accepted
bd close <story-id> --reason "Accepted: [brief summary of what was verified]"
```

**Reject** - any phase failed:
```bash
bd label remove <story-id> delivered
bd label add <story-id> rejected
bd update <story-id> --status open --notes "REJECTED: [detailed explanation]"
```

**The label history is the audit trail.** A story might show: `delivered -> rejected -> delivered -> accepted` - meaning it was rejected once, fixed, and then accepted.

### Rejection Notes Requirements

**Every rejection MUST include:**

1. **What was expected** - quote the specific AC or requirement
2. **What was delivered** - describe what the code actually does
3. **Why it doesn't meet the bar** - be specific about the gap
4. **What needs to change** - actionable guidance for the next attempt

**Example good rejection:**
```
REJECTED: AC "User receives email within 5 seconds" not verified.

EXPECTED: Integration test proving email delivery timing.
DELIVERED: Unit test mocking the email service, no real timing verification.
GAP: Mock tests cannot prove timing requirements. The 5-second SLA is untested.
FIX: Add integration test that sends real email and asserts delivery time < 5s.
```

**Example bad rejection:**
```
REJECTED: Tests not good enough.
```

The next developer must understand exactly what to fix without guessing.

### Rejection Handling

- Orchestrator prioritizes rejected stories first so they can be fully closed before new work starts.
- Rejection rates and reasons can be harvested from notes for process improvement.

**Repeated rejections:**
- After **5 rejections**, add `cant_fix` label and mark story **blocked**.
- Orchestrator alerts user - user intervention is required.
- **Parallel unrelated stories continue** - only the cant_fix story is blocked.

## Research Spikes

**PMs create research spikes when there is ambiguity that cannot be resolved without investigation.**

**When to create a spike:**
- Conflicting requirements (e.g., "use fastest option" vs "use mature components")
- Technology decisions with unclear trade-offs (e.g., "Polars vs Pandas for dataset D?")
- Performance questions that need benchmarking
- Integration questions with external systems
- Unclear acceptance criteria that need prototyping

**Spike format:**
```
Title: Spike: <question to answer>
Type: task
Description: <context and what we need to learn>
Acceptance Criteria:
- A clear recommendation with supporting evidence
- Documentation of findings in appropriate location
- Time-boxed to X hours/days
```

**Spike workflow:**
1. PM identifies ambiguity during D&F or execution
2. PM creates spike story with clear question
3. Developer investigates and documents findings
4. PM reviews findings and creates follow-up stories based on recommendation
5. Original ambiguous stories can now be unblocked

## Best Practices (lightweight)

**D&F Phase:**
- **Outcomes first**: Begin and end with business outcomes and how to measure progress. Technical details support outcomes, not the other way around.
- **BLT self-review is mandatory**: BA, Designer, Architect must review each other's docs before Sr. PM starts backlog creation. Challenge assumptions. Find gaps.
- **Challenge the user**: It's OK to question assumptions, ask for validation, push back on unclear requirements.
- BA/Designer/Architect keep their docs current and linked; Architect diagrams must be text-only Mermaid, dark-mode friendly.
- **Architect engages Security SME** to ensure security/compliance requirements are in architecture.

**Backlog Creation:**
- **Sr PM embeds all context AND testing requirements into stories** - developers need nothing beyond the story itself.
- **Milestones must be demoable** - real e2e functionality. Nothing replaces a real demo.
- Sr PM ensures every epic AC maps to stories; no gaps.
- **Create research spikes for ambiguities** - don't guess, investigate.
- **Backlog Challenger engages Security SME** to verify security requirements are captured.

**Execution Phase:**
- **Orchestrator NEVER writes code** - always spawns Developer agents for implementation.
- **Orchestrator spawns Sr. PM for backlog CRUD** - when user requests create/update/delete of stories or epics.
- Orchestrator assigns stories in priority order; prioritizes rejected stories first.
- **Developers MUST record proof** - test output, coverage, commit SHA in delivery notes.
- **PM-Acceptors use evidence** - review developer's proof rather than re-running tests (unless doubtful).
- PM-Acceptors reject with **detailed notes** so the next dev agent can fix without guessing.
- Developers never skip tests; if blocked (missing API key, etc.), mark story blocked and alert orchestrator.
- **If developer detects risk mid-story, STOP and escalate to orchestrator** - don't continue with uncertainty.

## Config & Tooling
- `.beads/config.yaml` should enforce: acceptance criteria required, epic business value required, coverage command (e.g., `make test-coverage`), integration test command (e.g., `make test-integration`), and max concurrent developers (default 6).
- Git hooks can enforce coverage when story specifies `tdd-strict`.
- Linear integration is optional; keep any mappings in config rather than this doc.

## Validation & Delivery Checklist

**Developer delivers:**
- ACs satisfied.
- **CI tests pass locally** (same tests GitHub Actions runs: lint, test, build).
- **PROOF recorded in delivery notes** (MANDATORY):
  - CI results summary (lint, test, integration, build - all passed)
  - Coverage percentage
  - Commit SHA and branch pushed
  - Relevant test output showing tests passed
- **Default**: Reasonable unit coverage + integration tests present (no mocks).
- **If story specifies `tdd-strict`**: Coverage at 100%.
- **No skipped tests** - blocked stories if tests cannot run.
- **Commit and push to GitHub** before marking as delivered.
- Add `delivered` label, update notes with proof.
- **Do NOT close story** - story stays `in_progress`.
- Include self-audit notes (edges tested, integration scenarios covered).
- **Add LEARNINGS section** if you discovered novel patterns, gotchas, or undocumented behavior.

**PM-Acceptor reviews (evidence-based, not redundant re-testing):**
- **Phase 1**: Verify proof is complete (CI results, coverage, push, test output). Reject immediately if missing.
- **Phase 2**: Outcome alignment - read the code, verify each AC is actually implemented (not something close but different).
- **Phase 3**: Test quality review - verify tests are meaningful, not superficial. Watch for red flags (trivial assertions, mocked-to-death tests, tautologies).
- **Phase 4**: Code quality spot-check - security issues, debug code, obvious problems.
- **Optional re-run**: If proof is suspicious, incomplete, or PM wants verification, re-run tests.
- **Accept**: All phases pass. Remove `delivered`, add `accepted`, close story with summary.
- **Reject**: Any phase fails. Remove `delivered`, add `rejected`, set status to `open`, add **detailed notes** (expected, delivered, gap, fix needed).
- **Labels are the audit trail** - story accumulates labels showing its journey.
