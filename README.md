# Pracomul‑SLC Analyzer

A  pipeline that transforms the Pracomul
**SLC‑style dialogue corpus** (≈ 108 k tokens) into tidy data frames and
pre‑computed aggregate tables—ready for statistical analysis or quick
exploration in R, Python, Excel, or a notebook.

## 1  Quick Start

```bash
# 1. (Optional) create a clean conda env
conda create -n slc python=3.10 spacy lexicalrichness pandas tqdm -c conda-forge
conda activate slc
python -m spacy download es_core_news_md  # Spanish model (≈ 46 MB)

# 2. Run the pipeline (from repo root)
python analyze_slc.py data/corpus.txt results
After ~10 s on a modern laptop you will find 20‑odd CSVs in results/
(see § 5).

2  Project Rationale
In its raw form the Pracomul corpus is a single plain‑text export:

php-template
Copy
Edit
<doc id="101" age_A="22" gender_A="F" … university_B="Univerza v Ljubljani">
<text>
<A> ¿qué edad tienes? </A>
<B> ¡Ah, sí! … </B>
 …
</text>
</doc>
To turn this into something a statistician can use we need to

recognise speaker turns (<A> / <B>),

normalise minor transcription artefacts,

attach all metadata (age, university, CEFR level, …) to the
speaking person,

compute a battery of lexical‑diversity indices plus sentence‑level
diagnostics,

export per‑token, per‑turn, and aggregate tables.

All of that happens inside one self‑contained script:
analyze_slc.py.

3  Dependencies
Library	Purpose	Tested Version
Python ≥ 3.8	runtime	3.10.14
spaCy	tokenisation · POS · sentences	3.7.6
lexicalrichness	9 established LD indices	0.5.1
pandas + NumPy	data wrangling · aggregates	2.2.2 / 1.26.x
tqdm	progress bars	4.66

Everything is installable from conda‑forge or PyPI.
No C/C++ toolchains required.

4  Running the Script
text
Copy
Edit
Usage:  python analyze_slc.py <corpus.txt> <out_dir>
corpus.txt
the raw export (108 k tokens, 65 DOC blocks).

out_dir
any folder; it will be created if absent.

On the first run spaCy will cache the Spanish model; subsequent runs are
pure‑Python and need ~8–10 seconds (including POS tagging).

5  What You Get
pgsql
Copy
Edit
results/
├─ turns.csv               # 1 row = 1 turn, A/B meta side‑by‑side
├─ turns_long.csv          # 1 row = 1 turn, speaker‑centric columns
├─ tokens.csv              # 1 row = 1 token  (≈ 108 k rows)
├─ tokens_with_meta.csv    # token rows + age / university / …
├─ aggregates_overall.csv  # corpus‑wide means & sums
├─ aggregates_by_doc.csv
├─ aggregates_by_speaker.csv   # A vs B
├─ aggregates_by_university.csv
├─ aggregates_by_nationality.csv
├─ … and so on for age, gender, CEFR level, mother tongue
└─ aggregates_by_country_combo.csv  # dyad: nationality_A‑nationality_B
5.1 Per‑Turn Columns (long format)
Variable	Meaning (speaker‑centric)
token_count (N)	running words
type_count (V)	distinct words
ttr	V / N
root_ttr	Guiraud’s V / √N
maas, hdd, …	length‑corrected LD indices
sent_len_mean	mean sentence length in tokens
laughter, …	counts of pragmatic markers
age, gender, …	speaker metadata (already aligned)

A complete data‑dictionary lives in docs/data_dictionary.md (or
see § 7 below).

6  Methodological Notes
Normalization
(RISAS) and other stage directions are stripped; arrow symbols are
deleted; informal*formal pairs keep the formal side.

Lexical Diversity
All indices come from lexicalrichness. Short turns (N ≤ 1) safely
return 0.0; root_ttr is computed manually if the attribute is absent
(older library versions).

Sentence Stats
spaCy’s sentence segmenter is used. In conversational texts sentence
boundaries are admittedly fuzzy, but mean/median lengths still give a
coarse proxy for syntactic elaboration.

Long vs Wide
turns_long.csv is the one you want for questions like
“What’s the average MTLD of Slovenians, regardless of A/B role?”

See /docs/methodology.pdf for a 6‑page write‑up citing Guiraud 1954,
Tweedie & Baayen 1998, Malvern et al. 2004, McCarthy & Jarvis 2010, etc.

7  Data Dictionary (short version)
File	Grain	Rows	Key columns
tokens_with_meta.csv	token	~108 k	doc_id, speaker, turn_id, idx
turns_long.csv	turn	~4 900	doc_id, speaker, turn_id
aggregates_by_*.csv	group	varies	grouping attr + numeric summaries

Each numeric summary has two flavours:

sum‑like: token_count, laughter, question_marks, …

mean‑like: ttr, root_ttr, maas, hdd, mtld, sent_len_mean, …

See /docs/data_dictionary.md for every column, unit,
and calculation formula.

8  Reproducibility Tips
Freeze the environment

bash
Copy
Edit
conda list --explicit > env.lock.txt
Ship env.lock.txt if journal reviewers demand bit‑for‑bit
reproducibility; users can recreate with conda create --name slc --file env.lock.txt.

Version Control

Commit both analyze_slc.py and the raw corpus.txt SHA‑256 hash.
Output CSVs can be generated on demand, so they stay out of Git.

Randomness

The pipeline has no stochastic components (even HD‑D’s draws are
deterministic given the tokens), so re‑runs are byte‑identical.

9  Troubleshooting
Symptom	Fix
Can't find model 'es_core_news_md'	Run python -m spacy download es_core_news_md
ModuleNotFoundError: lexicalrichness	pip install lexicalrichness or conda install -c conda-forge lexicalrichness
Memory (> 2 GB) on small VMs	Use the --disable ner switch already set in the script; RAM usage stays < 600 MB
Aggregate CSV shows NaN in means	That group had only empty turns; perfectly fine—filter or replace.

10  Citation
If you use this script or the derived tables in a publication, please
cite both the corpus creators and this repository, e.g.:

Žagar, Damjan. 2025. Pracomul‑SLC Analyzer (Version 1.0).
GitHub. https://github.com/your‑handle/pracomul‑slc‑analyzer.

11  License
The code is released under the MIT License.
The original Pracomul transcripts remain under their existing license
(non‑commercial research use).

Happy analysing! 🔍🗣️

go
Copy
Edit

*(End of `README.md`)*