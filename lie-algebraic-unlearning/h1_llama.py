r"""Experiment 1 (headline, H1): does commutator magnitude predict sequential-unlearning
interference?  Runs on Google Colab with a GPU runtime (free T4 is enough).

Design (paper Section 6.2). Inject a set of fictitious authors, each with four attributes, into
Llama-3.2-1B so the model provably holds the facts (LoRA fine-tune, merged). Then unlearn the
authors as T sequential requests under two partitions with opposite commutator predictions:

  * entity-partitioned    : each request removes whole authors        -> small cross commutators
  * attribute-partitioned : each request removes one attribute type    -> large cross commutators
    across all authors

Each request trains a fresh Lie-rotation generator A_i (exact low-rank exp adapter on o_proj and
down_proj), which is then baked into the weights (exact-product composition). After every request
we re-measure earlier requests' forget accuracy; interference(i,j) is how much request j > i
lets request i resurface. We correlate the normalized commutator
kappa_ij = ||[A_i,A_j]||_F / (2 ||A_i||_F ||A_j||_F) with interference (Spearman), and test
whether the attribute condition shows both higher kappa and higher interference (H1).

This is a first, self-contained run of the headline experiment. It does NOT reproduce TOFU's
exact forget-quality metric or the baseline sweep -- those are the OpenUnlearning path (Experiment
2). It uses a pragmatic forget metric (answer top-1 accuracy) on injected knowledge, which is
enough to decide H1.

HOW TO RUN IN COLAB
  1. Runtime -> Change runtime type -> GPU (T4 is fine).
  2. Add your Hugging Face token as a Colab secret named HF_TOKEN (key icon in the left sidebar),
     with access to meta-llama/Llama-3.2-1B. (Use a fresh token; do not paste it into the code.)
  3. Upload this file and run:  %run h1_llama.py     (or paste it into a cell)
  Expect ~30-90 min on a T4. Results: experiment1_results.csv (aggregate per pair) and
  experiment1_perlayer.csv (per adapted layer), with experiment1_scatter.png and
  experiment1_perlayer.png (which layers carry the commutator-interference relationship).

CPU validation: run `SMOKE=1 python h1_llama.py` to exercise the whole pipeline on a
tiny random model in ~1 min (no GPU, no gated download); the science needs the Colab run.
"""

import os
import sys
import subprocess
import itertools

SMOKE = os.environ.get("SMOKE", "0") == "1"


def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)


try:
    import peft  # noqa
    import scipy  # noqa
except ImportError:
    _pip("transformers", "datasets", "peft", "scipy", "accelerate")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

# ----------------------------------------------------------------------------- config
if SMOKE:
    from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
    DEVICE, DT = "cpu", torch.float32
    LAYERS, RANK = [0, 1], 2
    N_AUTHORS, T = 8, 4
    UNLEARN_STEPS, FORGET_TARGET = 8, 0.30
    INJECT_STEPS, INJECT_CHECK, INJECT_TARGET = 60, 10, 0.98
    ORDERINGS, SEEDS = 2, [0]
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DT = torch.float32                 # fp32 throughout: pure-bf16 Adam underflows and barely trains
    MODEL = "meta-llama/Llama-3.2-1B"
    LAYERS, RANK = list(range(8, 16)), 14
    N_AUTHORS, T = 20, 4               # 20 authors, 4 attributes -> T=4 requests either way
    UNLEARN_STEPS, FORGET_TARGET = 40, 0.30    # ascend each request only to a common moderate target
    INJECT_STEPS, INJECT_CHECK, INJECT_TARGET = 200, 10, 0.98   # full-batch injection until present
    ORDERINGS, SEEDS = 4, [0, 1, 2]    # 2 conditions x 4 orderings x 3 seeds = 24 runs

ATTRS = ["birthplace", "profession", "award", "genre"]
assert T == len(ATTRS)
torch.manual_seed(0)
np.random.seed(0)
EPS = 1e-8


# ----------------------------------------------------------------------------- model / tokenizer
def build_model_and_tokenizer():
    if SMOKE:
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        cfg = LlamaConfig(vocab_size=tok.vocab_size, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=256)
        model = LlamaForCausalLM(cfg)
        return model.to(DEVICE).to(DT), tok
    from huggingface_hub import login
    HF_TOKEN = None
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception:
        HF_TOKEN = os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(HF_TOKEN)
    else:
        print("No HF_TOKEN secret found; add it via the Colab key icon (name HF_TOKEN, enable "
              "notebook access) and re-run, or paste a token at the prompt below.")
        from huggingface_hub import notebook_login
        notebook_login()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DT).to(DEVICE)
    return model, tok


model, tok = build_model_and_tokenizer()


def single_token_pool(words):
    return [w for w in words if len(tok(w, add_special_tokens=False)["input_ids"]) == 1]


POOLS = {
    "birthplace": single_token_pool([" Paris", " Tokyo", " Cairo", " Lima", " Oslo", " Delhi",
                                     " Rome", " Perth", " Quito", " Bern", " Kiev", " Accra"]),
    "profession": single_token_pool([" poet", " chemist", " pilot", " painter", " banker",
                                     " sailor", " dancer", " judge", " farmer", " nurse"]),
    "award": single_token_pool([" Nobel", " Hugo", " Booker", " Pulitzer", " Turing", " Fields",
                                " Oscar", " Emmy", " Grammy", " Peabody"]),
    "genre": single_token_pool([" fantasy", " horror", " romance", " comedy", " mystery",
                                " thriller", " satire", " drama", " western", " memoir"]),
}
rng = np.random.default_rng(0)
authors = [f"{a}{i}" for i, a in enumerate(["Alden", "Brix", "Cael", "Dara", "Elm", "Fenn",
           "Gitt", "Hale", "Ivo", "Jarl", "Kepi", "Lore", "Mira", "Noor", "Osk", "Pell",
           "Quen", "Rho", "Sable", "Tove"][:N_AUTHORS])]
# author -> {attr: single-token answer}
facts = {au: {at: str(rng.choice(POOLS[at])) for at in ATTRS} for au in authors}
Q = {"birthplace": "Where was {} born?", "profession": "What is {}'s profession?",
     "award": "Which prize did {} win?", "genre": "What genre does {} write?"}


def qa(author, attr):
    return {"prompt": f"Question: {Q[attr].format(author)}\nAnswer:", "target": facts[author][attr]}


def encode(ex):
    full = tok(ex["prompt"] + ex["target"], return_tensors="pt").input_ids.to(DEVICE)
    plen = tok(ex["prompt"], return_tensors="pt").input_ids.shape[1]
    labels = full.clone(); labels[:, :plen] = -100
    return {"ids": full, "labels": labels, "plen": plen,
            "ans": tok(ex["target"], add_special_tokens=False).input_ids[0]}


ALL_QA = {(au, at): encode(qa(au, at)) for au in authors for at in ATTRS}


@torch.no_grad()
def acc(keys):
    m = 0
    for k in keys:
        e = ALL_QA[k]
        logits = model(e["ids"]).logits[0, e["plen"] - 1]
        m += int(logits.argmax().item() == e["ans"])
    return m / max(len(keys), 1)


@torch.no_grad()
def prob(keys):
    # mean probability assigned to the correct answer token: a continuous forget metric. Unlike
    # top-1 accuracy it does not quantize to 0/1, so small resurfacing after later requests is
    # measurable rather than flooring the interference signal to exactly zero.
    ps = []
    for k in keys:
        e = ALL_QA[k]
        logits = model(e["ids"]).logits[0, e["plen"] - 1].float()
        ps.append(torch.softmax(logits, -1)[e["ans"]].item())
    return float(np.mean(ps)) if len(keys) else 0.0


def build_batch(keys):
    # right-pad a set of QA facts into one batch for full-batch gradient (ascent/descent) steps
    maxlen = max(ALL_QA[k]["ids"].shape[1] for k in keys)
    n = len(keys)
    bids = torch.full((n, maxlen), tok.pad_token_id, dtype=torch.long, device=DEVICE)
    blab = torch.full((n, maxlen), -100, dtype=torch.long, device=DEVICE)
    bmask = torch.zeros((n, maxlen), dtype=torch.long, device=DEVICE)
    for r, k in enumerate(keys):
        e = ALL_QA[k]; L = e["ids"].shape[1]
        bids[r, :L] = e["ids"][0]; blab[r, :L] = e["labels"][0]; bmask[r, :L] = 1
    return bids, blab, bmask


# ----------------------------------------------------------------------------- inject via LoRA
def inject():
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "o_proj",
                      "up_proj", "down_proj"], task_type="CAUSAL_LM")
    pm = get_peft_model(model, lcfg)
    opt = torch.optim.Adam([p for p in pm.parameters() if p.requires_grad], lr=1e-3)
    keys = list(ALL_QA)

    # Pad all facts into one batch and take FULL-BATCH gradient steps. Per-example SGD is too noisy
    # to memorize the facts (it stalls near chance); a full-batch step converges in a few dozen steps.
    bids, blab, bmask = build_batch(keys)

    @torch.no_grad()
    def peft_acc():
        pm.eval()
        m = 0
        for kk in keys:
            e = ALL_QA[kk]
            m += int(pm(e["ids"]).logits[0, e["plen"] - 1].argmax().item() == e["ans"])
        return m / len(keys)

    # train until the facts are provably present (target top-1 acc), then stop early.
    for step in range(INJECT_STEPS):
        pm.train()
        opt.zero_grad()
        pm(input_ids=bids, attention_mask=bmask, labels=blab).loss.backward()
        opt.step()
        if (step + 1) % INJECT_CHECK == 0:
            a = peft_acc()
            print(f"  [inject] step {step + 1}: top-1 acc {a:.3f}")
            if a >= INJECT_TARGET:
                break
    merged = pm.merge_and_unload()      # fold LoRA into the base weights -> the target model
    merged.eval()
    for p in merged.parameters():
        p.requires_grad_(False)
    return merged


model = inject()
inj_acc = acc(list(ALL_QA))
inj_prob = prob(list(ALL_QA))
print(f"[inject] final top-1 accuracy on all {len(ALL_QA)} facts: {inj_acc:.2f}  (want >= 0.9)")
print(f"[inject] mean answer probability: {inj_prob:.3f}")
if inj_acc < 0.8:
    print("  WARNING: injection weak; the interference signal may be unreliable. "
          "Raise INJECT_STEPS or lr and re-run.")


# ----------------------------------------------------------------------------- Lie adapter
class LieLinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        d_out, d_in = base.weight.shape
        self.register_buffer("weight", base.weight.detach().clone())
        self.bias = None if base.bias is None else base.bias.detach().clone()
        self.r = rank
        self.reset()
        J = torch.zeros(2 * rank, 2 * rank, device=DEVICE, dtype=torch.float32)
        J[:rank, rank:] = torch.eye(rank); J[rank:, :rank] = -torch.eye(rank)
        self.register_buffer("J", J)

    def reset(self):
        d_out = self.weight.shape[0]
        self.U = nn.Parameter(1e-3 * torch.randn(d_out, self.r, device=DEVICE, dtype=torch.float32))
        self.V = nn.Parameter(1e-3 * torch.randn(d_out, self.r, device=DEVICE, dtype=torch.float32))

    def _QM(self):
        Qm, Rm = torch.linalg.qr(torch.cat([self.U, self.V], 1))
        return Qm, Rm @ self.J @ Rm.T

    def forward(self, x):
        base = F.linear(x, self.weight)
        Qm, M = self._QM()
        bf = base.float()
        d = (bf @ Qm) @ (torch.matrix_exp(M) - torch.eye(2 * self.r, device=DEVICE)).T @ Qm.T
        out = (bf + d).to(x.dtype)
        return out if self.bias is None else out + self.bias

    def dense_A(self):
        return (self.U @ self.V.T - self.V @ self.U.T).detach()

    @torch.no_grad()
    def clamp(self, eps):
        _, M = self._QM()
        nrm = torch.linalg.norm(M).item()
        if nrm > eps:
            c = (eps / nrm) ** 0.5
            self.U.mul_(c); self.V.mul_(c)

    @torch.no_grad()
    def bake(self):
        Qm, M = self._QM()
        d = Qm @ (torch.matrix_exp(M) - torch.eye(2 * self.r, device=DEVICE)) @ (Qm.T @ self.weight.float())
        self.weight.add_(d.to(self.weight.dtype))
        self.reset()


def wrap():
    ads = []
    for i in LAYERS:
        blk = model.model.layers[i]
        for parent, name in [(blk.self_attn, "o_proj"), (blk.mlp, "down_proj")]:
            ad = LieLinear(getattr(parent, name), RANK)
            setattr(parent, name, ad)
            ads.append(ad)
    return ads


adapters = wrap()
# (layer, module) label per adapter, parallel to `adapters`, for per-layer commutator logging
ADAPTER_LABELS = [(L, m) for L in LAYERS for m in ("o_proj", "down_proj")]
# keep the reset snapshot on CPU: fp32 weights are ~5GB, a second GPU copy risks OOM on a T4
INJECTED_STATE = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ----------------------------------------------------------------------------- partitions
def requests_for(condition):
    """Return T lists of (author, attr) keys."""
    if condition == "entity":
        groups = np.array_split(authors, T)
        return [[(au, at) for au in g for at in ATTRS] for g in groups]
    else:  # attribute
        return [[(au, at) for au in authors] for at in ATTRS]


def kappa(Ai_list, Aj_list):
    # returns (normalized commutator kappa_ij, magnitude ||A_i||_F ||A_j||_F). The magnitude is kept
    # so the analysis can partial it out and show the commutator effect is not an update-size proxy.
    num2 = ni2 = nj2 = 0.0
    for Ai, Aj in zip(Ai_list, Aj_list):
        C = Ai @ Aj - Aj @ Ai
        num2 += (C ** 2).sum().item(); ni2 += (Ai ** 2).sum().item(); nj2 += (Aj ** 2).sum().item()
    mag = ni2 ** 0.5 * nj2 ** 0.5
    return num2 ** 0.5 / (2 * mag + EPS), mag


def kappa_per_layer(Ai_list, Aj_list):
    # per-adapter (kappa_layer, magnitude_layer), parallel to ADAPTER_LABELS, so the analysis can ask
    # which adapted layers carry the commutator-interference relationship.
    out = []
    for Ai, Aj in zip(Ai_list, Aj_list):
        C = Ai @ Aj - Aj @ Ai
        ni = (Ai ** 2).sum().item() ** 0.5
        nj = (Aj ** 2).sum().item() ** 0.5
        num = (C ** 2).sum().item() ** 0.5
        m = ni * nj
        out.append((num / (2 * m + EPS), m))
    return out


def unlearn_request(keys):
    # full-batch gradient ASCENT on the request's forget loss, stopped at a common moderate target
    # (manuscript 6.2: train until forget loss reaches a small common target). Stopping short of the
    # zero floor leaves headroom for a later request's backward regression to be measured.
    for a in adapters:
        a.reset()
    opt = torch.optim.Adam([p for a in adapters for p in (a.U, a.V)], lr=5e-3)
    bids, blab, bmask = build_batch(keys)
    for step in range(UNLEARN_STEPS):
        opt.zero_grad()
        (-model(input_ids=bids, attention_mask=bmask, labels=blab).loss).backward()
        opt.step()
        for a in adapters:
            a.clamp(0.6)
        if prob(keys) < FORGET_TARGET:
            break
    gens = [a.dense_A() for a in adapters]
    for a in adapters:
        a.bake()
    return gens


# ----------------------------------------------------------------------------- experiment
rows = []
perlayer_rows = []                                              # one row per (pair, adapted layer)
DIAG = True                                                     # print the first run's trajectory
for condition in ("entity", "attribute"):
    base_reqs = requests_for(condition)
    for od in range(ORDERINGS):
        for seed in SEEDS:
            torch.manual_seed(1000 * od + seed)
            model.load_state_dict(INJECTED_STATE)               # reset to injected target
            perm = list(np.random.default_rng(1000 * od + seed).permutation(T))
            reqs = [base_reqs[p] for p in perm]
            gens, prob_hist = [], []                             # prob_hist[j][i] = prob of req i after req j
            for j in range(T):
                gens.append(unlearn_request(reqs[j]))
                prob_hist.append([prob(reqs[i]) for i in range(T)])
            if DIAG:
                print(f"  [diag] {condition} od{od} s{seed} prob_hist "
                      f"(row j = after request j, col i = request i's targets):")
                for j in range(T):
                    print("          ", [round(v, 4) for v in prob_hist[j]])
                print("          diagonal (each request's own forget-prob right after it) =",
                      [round(prob_hist[j][j], 4) for j in range(T)])
                DIAG = False
            for i in range(T):
                for j in range(i + 1, T):
                    delta = prob_hist[j][i] - prob_hist[i][i]         # resurfacing of i caused by j > i
                    kap, mag = kappa(gens[i], gens[j])
                    rows.append(dict(condition=condition, ordering=od, seed=seed, i=i, j=j,
                                     kappa=kap, magnitude=mag,
                                     interference=delta, abs_interference=abs(delta)))
                    for li, (kl, ml) in enumerate(kappa_per_layer(gens[i], gens[j])):
                        L, mod = ADAPTER_LABELS[li]
                        perlayer_rows.append(dict(condition=condition, ordering=od, seed=seed,
                                                  i=i, j=j, layer=L, module=mod,
                                                  kappa=kl, magnitude=ml, interference=delta))

# ----------------------------------------------------------------------------- analysis
import csv
with open("experiment1_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

with open("experiment1_perlayer.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(perlayer_rows[0].keys()))
    w.writeheader(); w.writerows(perlayer_rows)

from scipy.stats import rankdata
k = np.array([r["kappa"] for r in rows])
mag = np.array([r["magnitude"] for r in rows])
intf = np.array([r["interference"] for r in rows])
absf = np.array([r["abs_interference"] for r in rows])


def sp(x, y):
    return spearmanr(x, y) if np.ptp(x) > 0 and np.ptp(y) > 0 else (float("nan"), float("nan"))


def partial_sp(x, y, z):
    # Spearman partial correlation of x and y controlling for z (rank-transform, then partial-r formula)
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    c = lambda a, b: np.corrcoef(a, b)[0, 1]
    rxy, rxz, ryz = c(rx, ry), c(rx, rz), c(ry, rz)
    d = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / d if d > 0 else float("nan")


def perm_p(x, y, n=5000):
    # permutation test on Spearman rho (rank once, then permute ranks); reported alongside the
    # asymptotic p because the n pairs are not independent across a shared run.
    rx, ry = rankdata(x), rankdata(y)
    r0 = abs(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(0); cnt = 0
    for _ in range(n):
        cnt += abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= r0
    return (cnt + 1) / (n + 1)


rho, p = sp(k, intf)                    # signed resurfacing: theory-aligned H1 metric
rho_a, p_a = sp(k, absf)                # magnitude of disturbance (either direction)
pr = partial_sp(k, intf, mag) if np.ptp(intf) > 0 else float("nan")   # control for update size
pp = perm_p(k, intf) if np.ptp(intf) > 0 else float("nan")
ent = [r for r in rows if r["condition"] == "entity"]
att = [r for r in rows if r["condition"] == "attribute"]
print(f"\n[H1] Spearman rho(kappa, signed interference) = {rho:.3f}  (p = {p:.1e}, n = {len(rows)})")
print(f"[H1] permutation p = {pp:.1e};  partial rho controlling for ||A_i|| ||A_j|| = {pr:.3f}")
print(f"[H1] Spearman rho(kappa, |interference|)      = {rho_a:.3f}  (p = {p_a:.1e})")
print(f"     entity   : mean kappa {np.mean([r['kappa'] for r in ent]):.4f}, "
      f"mean interference {np.mean([r['interference'] for r in ent]):+.4f}")
print(f"     attribute: mean kappa {np.mean([r['kappa'] for r in att]):.4f}, "
      f"mean interference {np.mean([r['interference'] for r in att]):+.4f}")
print(f"     interference range [{intf.min():+.4f}, {intf.max():+.4f}]; "
      f"nonzero pairs {int((absf > 1e-6).sum())}/{len(rows)}")

# per-layer: which adapted layers carry the commutator-interference relationship
print("\n[H1 per-layer] Spearman rho(kappa_layer, interference), sorted:")
layer_stats = []
for (L, mod) in ADAPTER_LABELS:
    sub = [r for r in perlayer_rows if r["layer"] == L and r["module"] == mod]
    rl, pl = sp(np.array([r["kappa"] for r in sub]), np.array([r["interference"] for r in sub]))
    layer_stats.append((L, mod, rl, pl))
for L, mod, rl, pl in sorted(layer_stats, key=lambda t: t[2] if t[2] == t[2] else -9, reverse=True):
    print(f"     layer {L:2d} {mod:9s}: rho={rl:+.3f}  p={pl:.2e}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    for c, col in [("entity", "C0"), ("attribute", "C3")]:
        rs = [r for r in rows if r["condition"] == c]
        plt.scatter([r["kappa"] for r in rs], [r["interference"] for r in rs], c=col, label=c, alpha=0.7)
    plt.xlabel(r"normalized commutator $\hat\kappa_{ij}$")
    plt.ylabel("interference(i, j): resurfaced answer probability")
    plt.title(f"H1: Spearman rho = {rho:.2f} (p={p:.1e})"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("experiment1_scatter.png", dpi=150); plt.close()

    labels = [f"{L}.{mod[0]}" for (L, mod, _, _) in layer_stats]
    vals = [rl for (_, _, rl, _) in layer_stats]
    cols = ["C0" if mod == "o_proj" else "C1" for (_, mod, _, _) in layer_stats]
    plt.figure(figsize=(9, 4))
    plt.bar(range(len(vals)), vals, color=cols)
    plt.axhline(0, color="k", lw=0.7)
    plt.xticks(range(len(vals)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel(r"Spearman $\rho(\hat\kappa_{\mathrm{layer}},\,\mathrm{interference})$")
    plt.title("H1 per-layer: which adapters carry the prediction (o_proj=C0, down_proj=C1)")
    plt.tight_layout(); plt.savefig("experiment1_perlayer.png", dpi=150); plt.close()
    print("saved experiment1_results.csv, experiment1_perlayer.csv, "
          "experiment1_scatter.png, experiment1_perlayer.png")
except Exception as e:
    print("plot skipped:", e)
