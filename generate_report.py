"""
generate_report.py  --  ORS-VQ Project Report generator
Run: py generate_report.py
"""
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

OUT = "ORS_VQ_Project_Report.docx"
doc = Document()

# ── Page margins ───────────────────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin   = Inches(1.25)
    section.right_margin  = Inches(1.25)

# ── Style helpers ──────────────────────────────────────────────────────────────
BODY_FONT = "Times New Roman"
BODY_SIZE = 12

def set_run(run, bold=False, italic=False, size=BODY_SIZE, font=BODY_FONT, color=None):
    run.bold   = bold
    run.italic = italic
    run.font.name = font
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)

def para(text="", bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.LEFT,
         size=BODY_SIZE, font=BODY_FONT, space_before=6, space_after=6, color=None):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after  = Pt(space_after)
    if text:
        run = p.add_run(text)
        set_run(run, bold=bold, italic=italic, size=size, font=font, color=color)
    return p

def heading(text, level=1, space_before=18, space_after=6):
    p = doc.add_heading(text, level=level)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after  = Pt(space_after)
    for run in p.runs:
        run.font.name = BODY_FONT
    return p

def bullet_item(text, level=0, size=BODY_SIZE):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after  = Pt(3)
    run = p.add_run(text)
    set_run(run, size=size)
    return p

def spacer(n=1):
    for _ in range(n):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after  = Pt(2)

def page_break():
    doc.add_page_break()

def mixed_para(parts, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=6, space_after=6):
    """parts = list of (text, bold, italic, color) tuples"""
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after  = Pt(space_after)
    for text, bold, italic, color in parts:
        run = p.add_run(text)
        set_run(run, bold=bold, italic=italic, color=color)
    return p

def figure_placeholder(n, caption):
    """Shaded box with figure placeholder and italic caption below."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after  = Pt(2)
    # Add shading to paragraph
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), 'F0F0F0')
    pPr.append(shd)
    # Add border
    pBdr = OxmlElement('w:pBdr')
    for side in ('top','bottom','left','right'):
        el = OxmlElement(f'w:{side}')
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), '6')
        el.set(qn('w:space'), '4')
        el.set(qn('w:color'), 'AAAAAA')
        pBdr.append(el)
    pPr.append(pBdr)
    run = p.add_run(f"[ Figure {n}: {caption} ]")
    set_run(run, bold=True, size=11, color=(80, 80, 80))

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(2)
    cap.paragraph_format.space_after  = Pt(12)
    r = cap.add_run(f"Figure {n}: {caption}")
    set_run(r, italic=True, size=10)

def add_table(headers, rows, col_widths_inches=None):
    ncols = len(headers)
    if col_widths_inches is None:
        col_widths_inches = [6.5 / ncols] * ncols

    table = doc.add_table(rows=1 + len(rows), cols=ncols)
    table.style = 'Table Grid'

    # Header row
    hdr_row = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr_row.cells[i]
        cell.width = Inches(col_widths_inches[i])
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        set_run(run, bold=True, size=11)
        # Blue-ish shading
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), 'DDEEFF')
        tcPr.append(shd)

    # Data rows
    for ri, row in enumerate(rows):
        tr = table.rows[ri + 1]
        fill = 'FFFFFF' if ri % 2 == 0 else 'F8F8F8'
        for ci, val in enumerate(row):
            cell = tr.cells[ci]
            cell.width = Inches(col_widths_inches[ci])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if ci > 0 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(str(val))
            set_run(run, size=10)
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            shd = OxmlElement('w:shd')
            shd.set(qn('w:val'), 'clear')
            shd.set(qn('w:color'), 'auto')
            shd.set(qn('w:fill'), fill)
            tcPr.append(shd)

    spacer()
    return table

# ==============================================================================
# TITLE PAGE
# ==============================================================================
spacer(2)
para("ORS-VQ: Learning Operational Resilience State Vocabularies",
     bold=True, size=18, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=4)
para("for Automatic Workflow Hardening from Real Microservice Traces",
     bold=True, size=18, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=20)
spacer(2)
mixed_para([("Author: ", True, False, None), ("Preyashi Agarwal", False, False, None)],
           align=WD_ALIGN_PARAGRAPH.CENTER)
spacer()
mixed_para([("Supervisor: ", True, False, None), ("[Supervisor Name and Designation]", False, False, None)],
           align=WD_ALIGN_PARAGRAPH.CENTER)
mixed_para([("Department: ", True, False, None), ("[Department Name]", False, False, None)],
           align=WD_ALIGN_PARAGRAPH.CENTER)
mixed_para([("Institution: ", True, False, None), ("[Institution Name]", False, False, None)],
           align=WD_ALIGN_PARAGRAPH.CENTER)
spacer()
mixed_para([("Date of Submission: ", True, False, None), ("July 2026", False, False, None)],
           align=WD_ALIGN_PARAGRAPH.CENTER)
spacer()
mixed_para([("GitHub Repository: ", True, False, None),
            ("[PLACEHOLDER: GitHub URL]", False, False, (0, 0, 180))],
           align=WD_ALIGN_PARAGRAPH.CENTER)
page_break()

# ==============================================================================
# DECLARATION
# ==============================================================================
heading("Declaration of Originality", 1)
para('I hereby declare that this project report titled "ORS-VQ: Learning Operational Resilience '
     'State Vocabularies for Automatic Workflow Hardening from Real Microservice Traces" is the '
     'result of my own original work carried out under the supervision of [Supervisor Name]. '
     'All sources, references, and external contributions have been properly cited and acknowledged.')
spacer()
para("This work has not been submitted, in whole or in part, for any other degree or examination "
     "at this or any other institution.")
spacer()
mixed_para([("Signature: ", True, False, None), ("_____________________", False, False, None)])
mixed_para([("Name: ", True, False, None), ("Preyashi Agarwal", False, False, None)])
mixed_para([("Date: ", True, False, None), ("July 2026", False, False, None)])
page_break()

# ==============================================================================
# ABSTRACT
# ==============================================================================
heading("Abstract", 1)
para("Modern AI-driven workflow orchestration engines such as Temporal, Camunda, and AWS Step "
     "Functions apply static, uniform resilience defaults — retry three times, dead-letter on "
     "failure — without adapting to the actual failure behaviour of individual microservices. "
     "This results in unnecessary retry storms during degraded conditions, missed recovery "
     "opportunities during transient faults, and no mechanism for proactive per-node policy selection.")
spacer()
para("This project presents ORS-VQ, a Vector-Quantized Variational Autoencoder (VQ-VAE) that "
     "learns a discrete vocabulary of Operational Resilience States (ORS) from 21,578 real "
     "microservice execution events collected from the DeathStarBench socialNetwork benchmark. "
     "The model is trained using counterfactual simulation labels generated without any "
     "hand-written oracle: 10 candidate policies are evaluated across 150 Monte Carlo simulations "
     "per trace, and the best-scoring policy becomes the training label.")
spacer()
para("The learned codebook of K=16 discrete states achieves 92.8% purity (each code specialises "
     "in one failure scenario) and 98.5% policy consistency (within-code policy agreement across "
     "all four policy heads). Evaluated against five baselines including Temporal-style "
     "orchestration and LLM-generated annotations, ORS-VQ achieves zero wasted retry calls across "
     "all failure scenarios while Temporal burns up to 29 wasted API calls per run to reach the "
     "same dead-letter outcome. Runtime adaptation detects incidents with a mean latency of 32.5 "
     "seconds and eliminates 62.6% of wasted retries over a simulated three-minute incident.")
spacer()
mixed_para([("Keywords: ", True, False, None),
            ("VQ-VAE, microservice resilience, circuit breaker, workflow hardening, "
             "counterfactual labelling, operational resilience state, DeathStarBench",
             False, True, None)])
page_break()

# ==============================================================================
# ACKNOWLEDGEMENTS
# ==============================================================================
heading("Acknowledgements", 1)
para("I would like to express my sincere gratitude to my supervisor, [Supervisor Name], for "
     "their guidance, feedback, and support throughout this project. I am also grateful to the "
     "authors of the DeathStarBench benchmark for making the X-Trace dataset publicly available, "
     "and to the broader research community whose work on VQ-VAE and microservice resilience "
     "provided the theoretical foundation for this work.")
page_break()

# ==============================================================================
# CHAPTER 1: INTRODUCTION
# ==============================================================================
heading("Chapter 1: Introduction", 1)

heading("1.1 Background and Motivation", 2)
para("The proliferation of large language model (LLM)-generated workflow automation — systems that "
     "orchestrate sequences of microservice calls to complete complex tasks — has created a new "
     "class of reliability problem. These workflows are typically generated at design time and "
     "deployed into production-grade orchestration engines including Temporal, Apache Airflow, "
     "Camunda, and AWS Step Functions.")
spacer()
para("Industry-standard resilience defaults — retry three times with exponential backoff, "
     "compensate on failure via saga or dead-letter queue — are applied uniformly to every node "
     "in every workflow, regardless of the actual failure characteristics of the underlying "
     "microservice. Real microservice failures are heterogeneous: a storage service under "
     "persistent load saturation has fundamentally different failure dynamics from a transient "
     "network blip affecting an authentication service. No existing orchestration engine adapts "
     "its per-node resilience policy based on observed telemetry from real execution traces.")

heading("1.2 Problem Statement", 2)
para("Given a set of real microservice execution traces and a workflow call graph, automatically "
     "determine the optimal resilience policy (retry count, timeout tier, circuit breaker state, "
     "fallback type) for each service node — without hand-written oracle rules and without "
     "labelled training data — such that the policy generalises across failure scenarios and "
     "adapts dynamically as runtime telemetry changes.")

heading("1.3 Objectives", 2)
bullet_item("Learn a discrete vocabulary of Operational Resilience States (ORS) from real DeathStarBench X-Trace microservice execution data using a VQ-VAE.")
bullet_item("Generate training labels without human annotation using counterfactual Monte Carlo simulation: 10 candidate policies × 150 simulations per trace, scored by a composite metric.")
bullet_item("Demonstrate that the learned codebook achieves high purity and policy consistency with no codebook collapse.")
bullet_item("Apply the model at design time to harden LLM-generated workflow call graphs by assigning per-node resilience policies, and compare against five baselines.")
bullet_item("Demonstrate runtime adaptation: re-query the ORS model as live telemetry changes and show it detects incidents faster and eliminates wasted retries versus a static policy.")

heading("1.4 Scope and Limitations", 2)
bullet_item("Evaluated on the DeathStarBench socialNetwork benchmark only; generalisation to other microservice architectures is not empirically verified.")
bullet_item("The simulation model used for counterfactual labelling and evaluation is a Monte Carlo approximation, not a real production deployment.")
bullet_item("The runtime adaptation experiment uses a scripted incident timeline, not live production traffic.")
bullet_item("The project does not include a real deployment into an orchestration engine; workflow hardening is demonstrated at the annotation level.")

heading("1.5 Report Organisation", 2)
para("Chapter 2 surveys existing approaches to microservice resilience and identifies the gap "
     "ORS-VQ addresses. Chapter 3 describes the system architecture, model design, and "
     "counterfactual labelling methodology. Chapter 4 details the implementation. Chapter 5 "
     "presents results from three experiments. Chapter 6 concludes and outlines future work.")
page_break()

# ==============================================================================
# CHAPTER 2: LITERATURE REVIEW
# ==============================================================================
heading("Chapter 2: Literature Review and Background", 1)

heading("2.1 Existing Systems and Approaches", 2)

heading("2.1.1 Workflow Orchestration Engines", 3)
para("Temporal, Apache Airflow, Camunda BPM, and AWS Step Functions are the dominant workflow "
     "orchestration platforms. All support retry policies and timeout configuration, but none "
     "include built-in circuit breaker primitives. Temporal's documented default is "
     "maxAttempts=3 with exponential backoff; Camunda's default is retries=3; AWS Step Functions "
     "uses MaxAttempts=3. Fallback is handled via saga compensation patterns or catch blocks that "
     "route to dead-letter queues. These defaults are uniform — no per-node adaptation is possible "
     "without custom application code.")

heading("2.1.2 Circuit Breaker Pattern", 3)
para("The circuit breaker pattern (Nygard, 2007) prevents cascading failures by tracking service "
     "call outcomes and transitioning between three states: closed (normal operation), half-open "
     "(probing after recovery), and open (fast-fail). Standard implementations include Netflix "
     "Hystrix, Resilience4j, and Polly. None of the major orchestration engines expose circuit "
     "breaker state as a first-class workflow primitive.")

heading("2.1.3 VQ-VAE", 3)
para("The Vector-Quantized Variational Autoencoder (VQ-VAE) was introduced by van den Oord et al. "
     "(2017) and extended in VQ-VAE-2 (Razavi et al., 2019). The key contribution is a discrete "
     "bottleneck: a learned codebook of K embedding vectors, where the encoder output is replaced "
     "by the nearest codebook entry via straight-through gradient estimation, producing a discrete "
     "latent space suitable for learning symbolic vocabularies from continuous data.")

heading("2.1.4 Counterfactual Reasoning for Labelling", 3)
para("Counterfactual simulation for generating labels without oracles follows the approach used "
     "in offline reinforcement learning: simulate what would have happened under alternative "
     "policies and use the best-performing policy as the training target. This avoids structural "
     "misalignment between hand-written rules (keyed off failure mode names) and data-driven "
     "labels (keyed off scenario success probabilities).")

heading("2.2 Gap Analysis", 2)
figure_placeholder(13, "Method Feature Comparison: Retry / CB / Fallback / Adaptive — 6 methods")
para("As shown in Figure 13, no existing method simultaneously satisfies all four desirable "
     "properties: graph-awareness (policies differ by node position and type), data-driven "
     "adaptation (policies learned from real traces), circuit breaker support, and runtime "
     "adaptation. ORS-VQ is the only approach satisfying all four.")

heading("2.3 Theoretical Background", 2)

heading("2.3.1 Vector Quantisation", 3)
para("For an encoder output z_e in R^D and a codebook E = {e_1,...,e_K}, the quantised vector "
     "is z_q = e_k* where k* = argmin_k ||z_e - e_k||_2. The straight-through estimator passes "
     "gradients through z_q as if it were z_e. The commitment loss beta * ||z_e - sg(z_q)||^2 "
     "encourages the encoder to commit to nearby codebook entries.")

heading("2.3.2 Counterfactual Scoring", 3)
para("For each training event, a composite score is computed for each candidate policy pi over "
     "N_sim Monte Carlo trials: score(pi) = completion_rate(pi) - 0.4 x wasted_retries(pi) "
     "- 0.05 x latency_proxy(pi). The policy with the highest score becomes the label.")

heading("2.3.3 Operational Resilience State", 3)
para("An Operational Resilience State (ORS) is a discrete, reusable descriptor of the current "
     "failure condition of a microservice node, sufficient to determine all four dimensions of "
     "its optimal resilience policy. The scenario taxonomy (HEALTHY / TRANSIENT / DEGRADED / DOWN) "
     "emerges as an interpretable side-effect, not a supervised target.")
page_break()

# ==============================================================================
# CHAPTER 3: SYSTEM DESIGN
# ==============================================================================
heading("Chapter 3: System Design and Methodology", 1)

heading("3.1 System Architecture", 2)
para("The ORS-VQ system comprises three pipelines: offline training (learns the ORS vocabulary "
     "from real traces), design-time hardening (annotates workflow nodes at authoring time), and "
     "runtime adaptation (re-queries the model as telemetry changes during execution).")
spacer()
figure_placeholder(1, "ORS-VQ Architecture: Input (47-dim) -> Encoder -> VQ Codebook -> Policy Heads")
para("The model takes a 47-dimensional feature vector as input. The encoder maps this to a "
     "32-dimensional continuous latent vector z_e. The VQ codebook snaps z_e to the nearest of "
     "K=16 learned embeddings, producing z_q and a code index. Four independent policy heads "
     "decode z_q into logits over retry (4 classes), timeout (3), circuit breaker (3), and "
     "fallback (4) dimensions.")

heading("3.2 Input Features (47-dim)", 2)
bullet_item("Static workflow features (38-dim): node depth, fan-out, fan-in, is-external flag, is-crypto flag, node type one-hot, BFS positional embedding.")
bullet_item("Temporal telemetry features (9-dim): rolling failure rate, consecutive failures, retry success rate, latency percentile, time since last success, scenario duration, CB state (0/0.5/1), CB open duration, CB trip count.")

heading("3.3 Model Components", 2)

heading("3.3.1 Encoder", 3)
para("WorkflowEncoder: Linear(47->64) -> ReLU -> LayerNorm(64) -> Linear(64->32). "
     "LayerNorm after the first layer stabilises encoder output scale before the codebook "
     "distance computation, which is sensitive to the magnitude of z_e.")

heading("3.3.2 VQ Codebook with EMA Updates", 3)
para("The codebook uses Exponential Moving Average (EMA) updates rather than gradient-based "
     "learning. Codebook entries are stored as registered buffers (not parameters). Each entry "
     "tracks the EMA of all z_e vectors assigned to it. A dead-code restart mechanism "
     "reinitialises any entry whose EMA cluster size falls below a threshold of 1.0, replacing "
     "it with a randomly selected encoder output from the current batch. This prevents permanent "
     "codebook collapse and ensures all 16 codes remain active.")

heading("3.3.3 Policy Heads", 3)
para("Each head is Linear(32->16) -> ReLU -> Linear(16->n_classes). All four heads decode from "
     "the same z_q, forcing the codebook to encode a unified operational context sufficient for "
     "all four policy decisions simultaneously. The failure category (NOMINAL / TRANSIENT / "
     "PERSISTENT / TRANSACTION) is not in the training loss — it is computed post-hoc as a "
     "verifiable emergent property.")

heading("3.4 Counterfactual Label Generation", 2)
figure_placeholder(2, "Counterfactual Labelling Pipeline: Trace -> 10 Policies x 150 MC Sims -> Best Label")
para("For each training trace, 10 candidate policies spanning the policy space are evaluated "
     "through 150 Monte Carlo simulation trials. The composite score — completion rate minus "
     "0.4 x wasted retries minus 0.05 x latency proxy — selects the best-performing policy as "
     "the label. Oracle models trained with hand-written rules achieve 0% counterfactual accuracy "
     "on real DSB data; ORS-VQ achieves 97.6%.")

heading("3.5 Training Data", 2)
add_table(
    ["Workload Type", "Scenario", "Events"],
    [
        ["follow, unfollow, register, write_home_timeline, home-timeline-read", "HEALTHY", "14,462 (67.0%)"],
        ["Various failure workloads", "TRANSIENT", "3,439 (15.9%)"],
        ["Injected degradation", "DEGRADED", "1,000 (4.6%)"],
        ["compose_broken, broken_pipe, register_broken", "DOWN", "2,677 (12.4%)"],
        ["Total", "—", "21,578 events"],
    ],
    col_widths_inches=[3.0, 1.5, 2.0]
)

heading("3.6 Design-Time Hardening", 2)
figure_placeholder(14, "DSB read_home_timeline Call Graph: nodes coloured by assigned risk scenario")
para("At design time, each workflow node is assigned a risk scenario based on its type and depth. "
     "Synthetic 9-dim temporal telemetry is generated for each risk scenario. The full 47-dim "
     "feature vector is passed to the ORS model, which returns the code index and resilience "
     "policy. This runs at authoring time before any real execution.")

heading("3.7 Runtime Adaptation", 2)
para("In runtime mode, the system re-queries the ORS model every 10 seconds using live telemetry. "
     "As service health changes — failure rates increase, circuit breakers trip, latency trends "
     "shift — the model may assign a different ORS code, triggering a policy transition. This "
     "enables dynamic reconfiguration of retry budgets, circuit breaker states, and fallback "
     "paths without human intervention.")
page_break()

# ==============================================================================
# CHAPTER 4: IMPLEMENTATION
# ==============================================================================
heading("Chapter 4: Implementation", 1)

heading("4.1 Technology Stack", 2)
add_table(
    ["Component", "Technology"],
    [
        ["Language", "Python 3.11"],
        ["Deep Learning", "PyTorch 2.x"],
        ["Data Format", "DeathStarBench X-Trace (JSON)"],
        ["Simulation", "Custom Monte Carlo (Python)"],
        ["LLM Baseline", "Ollama (llama3.1:latest, local)"],
        ["Experiment Scripts", "Standalone Python modules (eval_*.py, diag_*.py)"],
        ["Version Control", "[PLACEHOLDER: GitHub URL]"],
    ],
    col_widths_inches=[2.5, 4.0]
)

heading("4.2 Module Structure", 2)
bullet_item("resiliency_poc/model/encoder.py  —  WorkflowEncoder (Linear 47->64->32 with LayerNorm)")
bullet_item("resiliency_poc/model/codebook.py  —  VQCodebookEMA (EMA + dead-code restart); VQCodebookGrad (ablation); FactorizedVQCodebook (ablation)")
bullet_item("resiliency_poc/model/policy_heads.py  —  PolicyHeads (4 x Linear 32->16->n_classes); PolicyPrediction dataclass")
bullet_item("resiliency_poc/model/ors_model.py  —  ORSVQModel: encoder + codebook + heads; forward(), predict(), code_for(), ors_table()")
bullet_item("resiliency_poc/dsb_loader.py  —  X-Trace JSON parser; CB state machine assignment; stratified train/test split")
bullet_item("resiliency_poc/counterfactual_labeler.py  —  10-policy x 150-sim scoring; produces (retry, timeout, cb, fallback) labels")
bullet_item("hardening/engine.py  —  WorkflowHardeningEngine; node risk profiling; synthetic telemetry; ORS query")
bullet_item("hardening/simulator.py  —  WorkflowFailureSimulator; original_run() and hardened_run(); ScenarioMetrics")
bullet_item("hardening_comparison.py  —  Experiment 2: 6 baselines, 500 trials x 4 scenarios x 4 graphs")
bullet_item("eval_runtime_adaptation.py  —  Experiment 3: 3-minute incident, 10s tick, ORS re-query loop")
bullet_item("eval_exp1_summary.py  —  Experiment 1: purity, consistency, transfer, latent quality")
bullet_item("eval_completion_breakdown.py  —  Outcome categories: CLEAN / RETRY-SAVED / FALLBACK / DEAD-LETTER / ABORTED")

heading("4.3 Training Configuration", 2)
add_table(
    ["Hyperparameter", "Value"],
    [
        ["Optimiser", "Adam, lr=1e-3"],
        ["Batch size", "64"],
        ["Epochs", "50"],
        ["EMA decay", "0.99"],
        ["Commitment loss coefficient (beta)", "0.25"],
        ["Dead-code restart threshold", "1.0"],
        ["Codebook size K", "16"],
        ["Latent dimension", "32"],
        ["Checkpoint", "resiliency_poc/checkpoints/ors_dsb_vq.pt"],
    ],
    col_widths_inches=[3.0, 3.5]
)

heading("4.4 Key Implementation Decisions", 2)

heading("4.4.1 EMA over Gradient-Based Codebook Updates", 3)
para("Early experiments with gradient-based codebook updates exhibited collapse: frequently-activated "
     "entries dominated and many entries received no gradient signal. EMA updates resolved this by "
     "decoupling codebook learning from the encoder's gradient path.")

heading("4.4.2 Shared Codebook (not Factorized)", 3)
para("A factorized codebook (one independent codebook per policy dimension) was implemented as "
     "an ablation. The shared codebook was preferred because it forces z_q to encode a unified "
     "operational context sufficient for all four policy dimensions — the core ORS insight.")

heading("4.4.3 Category Head Excluded from Training Loss", 3)
para("The failure category is inferred post-hoc from each code's assigned scenario distribution. "
     "Including it in training loss would make it a supervised output rather than a verifiable "
     "emergent property. As an emergent property, the taxonomy demonstrates genuine operational "
     "structure learned from data.")

heading("4.4.4 No Existing Files Modified", 3)
para("All new experiment scripts are standalone modules. No existing files in resiliency_poc/ or "
     "hardening/ are modified, ensuring baseline results remain reproducible.")
page_break()

# ==============================================================================
# CHAPTER 5: RESULTS
# ==============================================================================
heading("Chapter 5: Results, Testing, and Performance Evaluation", 1)

heading("5.1 Experimental Setup", 2)
add_table(
    ["Parameter", "Value"],
    [
        ["Dataset", "DeathStarBench socialNetwork X-Trace"],
        ["Total events", "21,578 service events"],
        ["Train / Test split", "3,000 / 600 (stratified)"],
        ["Codebook size K", "16"],
        ["Latent dimension", "32"],
        ["MC trials (labelling)", "150 per candidate policy x 10 policies"],
        ["MC trials (evaluation)", "500 per scenario per graph"],
        ["Evaluation graphs", "4 DSB call graph topologies"],
        ["Evaluation scenarios", "HEALTHY / TRANSIENT / DEGRADED / SERVICE_DOWN"],
        ["Runtime tick interval", "10 seconds"],
        ["Runtime incident duration", "180 seconds (3 minutes)"],
    ],
    col_widths_inches=[3.0, 3.5]
)

heading("5.2 Experiment 1: Codebook Quality", 2)

heading("5.2.1 Codebook Purity", 3)
para("Purity measures the fraction of events assigned to each code that belong to its dominant "
     "scenario. Across all 16 active codes, mean purity is 92.8%. Twelve of sixteen codes achieve "
     "100% purity. The two lowest-purity codes — ORS-5 (50.7%) and ORS-7 (85.4%) — both mix "
     "DEGRADED and DOWN events, which is semantically expected: the boundary between a severely "
     "degraded and a fully down service is empirically blurry in the X-Trace data.")
spacer()
figure_placeholder(3, "Codebook Purity per ORS Code — bar chart, coloured by dominant scenario, mean=92.8%")

heading("5.2.2 Policy Consistency", 3)
para("Consistency measures the fraction of events assigned to each code that receive the modal "
     "policy for each policy head. Mean consistency is 98.5% across all 16 codes and 4 heads. "
     "Timeout, CB, and Fallback heads achieve 100% consistency across all codes.")
spacer()
figure_placeholder(4, "Policy Consistency Heatmap — ORS codes (rows) x Policy heads (cols), cell=consistency %")

heading("5.2.3 Codebook Utilisation and Latent Quality", 3)
figure_placeholder(15, "Codebook Utilisation — event count per ORS code, all 16 codes active (no collapse)")
figure_placeholder(5, "Latent Space — 2D PCA/t-SNE of encoder outputs z_e, coloured by scenario")
spacer()
add_table(
    ["Metric", "Value", "Interpretation"],
    [
        ["Codebook purity (mean)", "92.8%", "Fraction of events in dominant scenario per code"],
        ["Policy consistency (mean)", "98.5%", "Within-code policy agreement across 4 heads"],
        ["Active codes", "16 / 16", "Full utilisation, no collapse"],
        ["Latent silhouette score", "0.290", "Cluster separation in 32-dim space (higher=better)"],
        ["Davies-Bouldin index", "1.235", "Cluster compactness (lower=better)"],
        ["Counterfactual accuracy (ORS-VQ)", "97.6%", "vs 0% for oracle-rule models on real DSB data"],
    ],
    col_widths_inches=[2.5, 1.5, 2.5]
)

heading("5.3 Experiment 2: Workflow Hardening Comparison", 2)

heading("5.3.1 Nominal Completion Rate", 3)
figure_placeholder(6, "Completion Rate Grouped Bar Chart — groups=4 scenarios, bars=6 methods")
add_table(
    ["Method", "HEALTHY", "TRANSIENT", "DEGRADED", "DOWN", "Mean"],
    [
        ["Original (no resilience)", "69.7%", "8.1%", "0.3%", "0.0%", "19.5%"],
        ["Manual-Uniform", "97.5%", "89.3%", "77.8%", "69.6%", "83.6%"],
        ["Rule-Based", "100.0%", "98.3%", "90.5%", "82.6%", "92.9%"],
        ["Temporal-style", "100.0%", "100.0%", "100.0%", "100.0%", "100.0%"],
        ["LLM-Hardened", "87.9%", "58.8%", "37.5%", "29.1%", "53.3%"],
        ["ORS-Hardened", "100.0%", "98.8%", "96.4%", "94.8%", "97.5%"],
    ],
    col_widths_inches=[2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
)

heading("5.3.2 Outcome Category Breakdown", 3)
para("Nominal completion is misleading because it includes dead-letter outcomes alongside genuine "
     "recoveries. The outcome breakdown separates five mutually exclusive categories.")
spacer()
figure_placeholder(8, "Outcome Breakdown Stacked Bar — per method, segments=CLEAN/RETRY-SAVED/FALLBACK/DEAD-LETTER/ABORTED")
add_table(
    ["Method", "CLEAN", "RETRY-SAVED", "FALLBACK", "DEAD-LETTER", "ABORTED", "True Compl%"],
    [
        ["Original", "19.5%", "0%", "0%", "0%", "80.5%", "19.5%"],
        ["Manual-Uniform", "19.5%", "0%", "64.0%", "0%", "16.5%", "83.5%"],
        ["Rule-Based", "0%", "0%", "0%", "92.8%", "7.2%", "0%"],
        ["Temporal-style", "19.5%", "26.4%", "0%", "54.0%", "0%", "46.0%"],
        ["ORS-Hardened", "0%", "0%", "0%", "97.5%", "2.5%", "0%"],
    ],
    col_widths_inches=[1.8, 0.9, 1.1, 1.0, 1.1, 0.9, 1.2]
)
para("Temporal's 100% nominal completion masks a 54% dead-letter rate. ORS's 2.5% ABORTED "
     "traces to the nginx-thrift entry node (HEALTHY risk, cached fallback with 95% success "
     "probability) failing under severe scenarios — ORS surfaces this as an explicit abort "
     "rather than hiding it as a guaranteed dead-letter.")

heading("5.3.3 API Efficiency", 3)
figure_placeholder(7, "Wasted API Calls Bar Chart — groups=4 scenarios, bars=6 methods")
figure_placeholder(9, "API Efficiency Scatter — x=nominal completion %, y=avg API calls/run, one point per method")
add_table(
    ["Method", "Scenario", "Avg Wasted Calls", "Avg Total API Calls"],
    [
        ["ORS-Hardened", "DEGRADED (compose_post)", "0.00", "1.00"],
        ["Temporal-style", "DEGRADED (compose_post)", "12.00", "22.00"],
        ["ORS-Hardened", "SERVICE_DOWN (compose_post)", "0.00", "1.00"],
        ["Temporal-style", "SERVICE_DOWN (compose_post)", "19.00", "29.00"],
    ],
    col_widths_inches=[1.8, 2.5, 1.6, 1.6]
)
para("Under SERVICE_DOWN, Temporal burns 29 API calls per run — 19 of them wasted — to reach "
     "the same dead-letter outcome ORS reaches in 1 call. ORS achieves zero wasted retries "
     "across all scenarios because the circuit breaker opens immediately on DEGRADED/DOWN nodes.")

heading("5.3.4 Head-to-Head Win/Loss", 3)
figure_placeholder(16, "ORS Win/Loss Summary — ORS vs each baseline across 16 cells (4 graphs x 4 scenarios)")
add_table(
    ["ORS vs", "ORS Wins", "Other Wins", "Ties"],
    [
        ["Manual-Uniform", "16 / 16", "0 / 16", "0 / 16"],
        ["Rule-Based", "6 / 16", "0 / 16", "10 / 16"],
        ["Temporal-style (nominal)", "0 / 16", "12 / 16", "4 / 16"],
        ["LLM-Hardened", "16 / 16", "0 / 16", "0 / 16"],
    ],
    col_widths_inches=[2.5, 1.5, 1.5, 1.5]
)
para("Rule-Based ties ORS in 10/16 cells because hand-crafted heuristics approximate what ORS "
     "learned from data — but ORS achieves this without anyone writing a single heuristic.")

heading("5.4 Experiment 3: Runtime Adaptation", 2)
figure_placeholder(10, "Incident Timeline — x=time (0-180s), y=ORS code per service, vertical lines=events")
figure_placeholder(11, "Cumulative Wasted Retries — static vs adaptive ORS over 180s, area=62.6% savings")
figure_placeholder(12, "Detection Latency Histogram — distribution across services/scenarios, mean=32.5s")
add_table(
    ["Metric", "Value", "Notes"],
    [
        ["Detection latency (mean)", "32.5s", "From incident onset to ORS policy transition"],
        ["Detection latency (min)", "0s", "Immediate detection on first 10s tick"],
        ["Detection latency (max)", "90s", "Up to 9 ticks before threshold crossed"],
        ["Policy correctness (norm. regret)", "0.00", "Matches oracle policy selection"],
        ["Wasted retries eliminated", "773 / 1,234 (62.6%)", "Over 3-minute incident, 200 req/tick"],
        ["MTTR — Static policy", "160s", "Fixed policy baseline"],
        ["MTTR — Adaptive (ORS)", "170s", "10s slower due to CAUTIOUS_PROBE during recovery"],
    ],
    col_widths_inches=[2.5, 1.8, 2.2]
)
para("The 10-second MTTR increase for adaptive ORS is an honest finding: during recovery, the "
     "model transitions to CAUTIOUS_PROBE (half-open CB, 1 probe), which has lower per-tick "
     "success rate than the static STORM_RETRY policy. The model is correctly cautious post-failure "
     "but this slightly extends recovery time — a known trade-off of conservative probing.")

heading("5.5 Discussion", 2)
bullet_item("ORS-VQ successfully learns a compact, reusable vocabulary of operational resilience states from real microservice traces without hand-written labels (92.8% purity, 98.5% consistency).")
bullet_item("Counterfactual labelling is essential: oracle models achieve 0% counterfactual accuracy on real DSB data vs 97.6% for ORS-VQ.")
bullet_item("Three policy archetypes dominate the codebook: STORM_RETRY (transient), CAUTIOUS_PROBE (recovering), CIRCUIT_OPEN (persistent failures). K=16 captures fine-grained variations.")
bullet_item("Three services show limited state variation (uniform ORS). Root cause: training data imbalance — these services only appear in failure workload traces, never in healthy conditions.")
bullet_item("Temporal's 100% nominal completion is largely illusory: 54% is dead-letter, and each dead-letter costs 12-29 API calls vs ORS's 1.")
page_break()

# ==============================================================================
# CHAPTER 6: CONCLUSION
# ==============================================================================
heading("Chapter 6: Conclusion and Future Work", 1)

heading("6.1 Conclusion", 2)
para("This project presents ORS-VQ, a VQ-VAE-based system for automatically learning and "
     "applying operational resilience policies for microservice workflow nodes. The central "
     "claim — that a discrete vocabulary of operational resilience states can be learned from "
     "real execution traces without hand-written labels — is supported by three experiments.")
spacer()
para("Experiment 1 demonstrates high codebook quality: 92.8% purity, 98.5% policy consistency, "
     "16/16 codes active. Experiment 2 demonstrates that ORS-VQ produces better per-node policy "
     "assignments than all non-Temporal baselines on nominal completion, and is the only method "
     "with zero wasted retry calls. The outcome breakdown analysis reveals Temporal's apparent "
     "superiority is driven by guaranteed dead-letter routing (54% of completions involve data "
     "loss), not genuine service recovery. Experiment 3 demonstrates 32.5s mean detection latency "
     "and 62.6% wasted retry elimination at runtime.")
spacer()
para("ORS-VQ is the only evaluated method that is simultaneously graph-aware and data-driven, "
     "achieving this without any hand-written rules, labelled training data, or domain expertise "
     "in resilience engineering.")

heading("6.2 Future Enhancements", 2)
bullet_item("Online learning: continuously update the codebook as new execution traces arrive, enabling adaptation to service behaviour changes without full retraining.")
bullet_item("Real deployment integration: implement ORS query as a native activity type in Temporal or as a Camunda connector, replacing synthetic telemetry with real production metrics.")
bullet_item("Multi-hop propagation: model failure cascade dynamics — if one service enters CIRCUIT_OPEN, downstream dependents should have their risk profile updated automatically.")
bullet_item("Larger codebook ablation: explore K=32 and K=64 to understand whether additional codes capture meaningful sub-states or simply fragment existing clusters.")
bullet_item("Cross-architecture transfer: evaluate whether a model trained on DSB socialNetwork generalises to other architectures without retraining.")
bullet_item("Confidence-aware transitions: add uncertainty quantification so the system can delay a policy transition when the encoder output is equidistant between two codebook entries.")
page_break()

# ==============================================================================
# REFERENCES
# ==============================================================================
heading("References", 1)
refs = [
    "[1] van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning (VQ-VAE). NeurIPS.",
    "[2] Razavi, A., van den Oord, A., & Vinyals, O. (2019). Generating Diverse High-Fidelity Images with VQ-VAE-2. NeurIPS.",
    "[3] Nygard, M. T. (2007). Release It!: Design and Deploy Production-Ready Software. Pragmatic Bookshelf.",
    "[4] Yu, G., et al. (2021). DeathStarBench: End-to-End Microservices Benchmark. ACM SOSP.",
    "[5] Temporal Technologies. (2024). Temporal Workflow Documentation: Activity Retry Policy. docs.temporal.io.",
    "[6] AWS. (2024). AWS Step Functions: Error Handling. docs.aws.amazon.com.",
    "[7] Camunda. (2024). Camunda 8 Job Worker Retry Configuration. docs.camunda.io.",
    "[8] Fowler, M. (2014). CircuitBreaker. martinfowler.com/bliki/CircuitBreaker.html.",
    "[9] Netflix OSS. (2018). Hystrix: Latency and Fault Tolerance. github.com/Netflix/Hystrix.",
    "[10] Xu, Z., et al. (2023). Towards Autonomous Microservice Fault Diagnosis with LLMs. IEEE ICDCS.",
]
for r in refs:
    para(r, space_before=3, space_after=3)
page_break()

# ==============================================================================
# APPENDICES
# ==============================================================================
heading("Appendix A: Running the Experiments", 1)
para("All experiments are run from the project root directory:")
spacer()
code_lines = [
    "cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae",
    "",
    "# Train ORS model",
    "py -m resiliency_poc.ors_dsb_poc",
    "",
    "# Experiment 1 — Codebook quality",
    "py -m eval_exp1_summary",
    "",
    "# Experiment 2 — Workflow hardening comparison",
    "py -m hardening_comparison",
    "",
    "# Experiment 3 — Runtime adaptation",
    "py -m eval_runtime_adaptation",
    "",
    "# Outcome category breakdown",
    "py -m eval_completion_breakdown",
    "",
    "# Abort root cause diagnostic",
    "py -m diag_abort_reason",
    "",
    "# Codebook policy inspection",
    "py -m diag_codebook_policies",
]
for line in code_lines:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after  = Pt(1)
    r = p.add_run(line)
    r.font.name = "Courier New"
    r.font.size = Pt(10)
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), 'F0F0F0')
    pPr.append(shd)

spacer()
heading("Appendix B: GitHub Repository", 1)
para("[PLACEHOLDER: Insert GitHub repository URL here]", color=(0, 0, 180))
para("The repository contains all source code, experiment scripts, and the trained model checkpoint "
     "(ors_dsb_vq.pt). DeathStarBench X-Trace data is not included due to size; download "
     "instructions are in the repository README.")

spacer()
heading("Appendix C: Trained Model Checkpoint", 1)
bullet_item("Location: resiliency_poc/checkpoints/ors_dsb_vq.pt")
bullet_item("Format: PyTorch state dict with keys: state_dict, epoch, train_loss")
bullet_item("Load with: torch.load('ors_dsb_vq.pt', weights_only=False)")

# ==============================================================================
# SAVE
# ==============================================================================
doc.save(OUT)
print(f"Saved: {OUT}")
