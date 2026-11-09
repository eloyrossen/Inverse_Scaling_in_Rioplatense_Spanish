# Project code: how to reproduce the analysis

This guide lets you rerun the full analysis on any computer, from the raw MECO data to the model comparison table and the scaling plot. For what was done and what was found, read the [README](README.md) first.

There are two routes:

- **Short route (minutes):** use the precomputed surprisals in `surprisal_cache/` and run only the statistical analysis (steps 1, 4 and 5).
- **Full route (hours):** recompute surprisal with all eight models (steps 1 to 5).

The short route was tested end to end before publishing: it reproduces the paper's values exactly (Δ log-likelihood 66.88 / 62.82 / 49.04 for BLOOM, 79.51 / 74.15 / 58.52 for XGLM and 63.45 / 46.09 for Gemma 3).

---

## 0. Requirements

**R** (≥ 4.2) with the `lme4` and `ggplot2` packages:

```r
install.packages(c("lme4", "ggplot2"))
```

**Python** (≥ 3.10), only for the full route. On a machine without a GPU, install PyTorch from the CPU index, because the default build pulls several GB of CUDA libraries that are never used:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers tiktoken sentencepiece protobuf ftfy
```

Reference hardware: models up to 1.7B parameters ran on a laptop with 2 cores and 4 GB of RAM. The 3–4B models need about 8 GB of free RAM (see the note on `gemma-3-4b-pt` below).

## 1. Get the MECO data

The eye-tracking data is not included in this repository. Download it from OSF and place it in `data/`:

```bash
mkdir -p data
curl -L "https://osf.io/download/nzetq/" -o data/joint_l1_data_trimmed_version2.1.rda
```

(Path on OSF: `release 2.0 → version 2.1 → wave 1 → primary data → eye tracking data`.) The file is about 8.6 MB and contains all 13 languages of Wave 1. The Argentinian subsample is `lang == "sp"` (48 participants, 87,275 rows).

Run all scripts **from the root of this folder** (the one containing `scripts/`, `data/`, etc.).

## 2. Compute surprisal (full route only)

### 2.1 Test the tool

```bash
python3 llm_surprisal/llm_generate.py "Jano presidía el comienzo y el final de los conflictos." --model xglm-564M
```

This should print a table with the surprisal (in bits) of each token.

### 2.2 Run one model on the 12 texts

`compute_chunk.R` rebuilds the sentences of the requested texts and computes `Scurr_<model>` and `Sprev_<model>` for every word:

```bash
# Rscript scripts/compute_chunk.R <model> <comma-separated trialids> <output.csv>
Rscript scripts/compute_chunk.R xglm-564M 1,2,3,4,5,6,7,8,9,10,11,12 surprisal_cache/xglm-564M_full.csv
```

Repeat for all eight models: `bloom-560m`, `bloom-1b7`, `bloom-3b`, `xglm-564M`, `xglm-1.7B`, `xglm-2.9B`, `gemma-3-1b-pt`, `gemma-3-4b-pt`. On a slow machine you can process the texts in batches (e.g. `1,2,3`, then `4,5,6`) and bind the CSV files afterwards.

Practical notes, all based on real problems encountered during the project:

- **`compute_chunk.R` prepends `/usr/bin` to the `PATH`** so that `system2("python3", ...)` uses the Python that has `torch` installed (in the project, R ran inside a conda environment with its own `python3` without `torch`). If `python3` on your system is already the right one, that line does no harm. On Windows, edit that line or make sure `python3` is on the `PATH`.
- **Gemma 3 models** are gated repositories on Hugging Face. You need to accept the license on each model's page and authenticate (`huggingface-cli login`) before using them.
- **`gemma-3-4b-pt` is multimodal.** With the `Gemma3ForCausalLM` class from the original `models.py`, `transformers` cannot find the text weights and initializes them randomly without raising an error (only a warning). That is why the copy of `models.py` in this repository uses `Gemma3ForConditionalGeneration` for that checkpoint. If you see *missing* or *unexpected* weight warnings when loading a model, its surprisal is not valid.
- **Memory:** `gemma-3-4b-pt` takes ~8 GB in bf16, but `transformers` upcasts it to float32 when loading (~16 GB). On machines with little RAM, loading it with `torch_dtype=torch.bfloat16` in the `model_class.from_pretrained(model)` line of `llm_generate.py` keeps the process from being killed for lack of memory.
- **Repeated downloads:** if a model is already cached but `transformers` tries to download it again, export `HF_HUB_OFFLINE=1` before running the script.
- **Spanish text in R:** run R with a UTF-8 locale (e.g. `LC_ALL=C.UTF-8` on Linux). Under the `"C"` locale, `tolower()` and `[:alnum:]` classes silently fail on accented vowels.

## 3. Baseline predictors (optional)

`data/master_with_baseline.csv` already contains, for the 2,411 words, length (`Lcurr`, `Lprev`) and log frequency (`Fcurr`, `Fprev`) from EsPal, *Written and Web Tokens* corpus. To regenerate them, upload the list of unique words (one per line, without punctuation) to EsPal's word-list tool:

```bash
curl -c cookies.txt -d "database=written" https://www.bcbl.eu/databases/espal/index.php
curl -b cookies.txt -c cookies.txt -d "phonology=sa" https://www.bcbl.eu/databases/espal/index.php
curl -b cookies.txt -F "words_file=@wordlist.txt" \
     -F "idxname[]=cnt" -F "idxname[]=frq" -F "idxname[]=log_frq" \
     -F "idxname[]=zipf" -F "idxname[]=num_letters" \
     https://www.bcbl.eu/databases/espal/w2iout.php -o result.html
```

`log_frq` (frequency) and `num_letters` (length) are used. Words with no EsPal entry (three numbers and a dash) get frequency 0, as in the course instructions. `Lprev`/`Fprev` are the values of the previous word within the same sentence (`NA` for the first word).

## 4. Statistical models

```bash
Rscript scripts/fit_models.R
```

Merges the eight files in `surprisal_cache/` with the baseline predictors and the reading times, applies the exclusions, fits the baseline and the eight surprisal models, and saves `results/surprisal_model_comparison.csv`. It takes about a minute. It should print `Rows after exclusions: 53464` and the results table.

## 5. Plot

```bash
Rscript scripts/plot_scaling.R
```

Produces `figures/scaling_plot_reproduced.png`: Δ log-likelihood against model size, one line per family.

---

## Full code of the project's own scripts

The files in `llm_surprisal/` (von der Malsburg) and `scripts/TermPaperHelperFunctions.R` (course materials) were used as is and are in the repository, but are not reproduced here. What follows is the code written for this project.

### `scripts/BatchedSurprisal.R`

Drop-in replacement for `add_surprisals()` that loads each model once per call (via `llm_generate.py --input`) instead of once per sentence. It was validated against the original function on 210 words: maximum difference 4.97 × 10⁻¹⁴.

```r
# BatchedSurprisal.R
#
# Own optimisation (NOT part of the original course materials).
# TermPaperHelperFunctions.R is left untouched.
#
# Rationale: get_surprisals() in TermPaperHelperFunctions.R launches a new
# python3 process FOR EVERY SENTENCE, and each call re-imports torch/
# transformers and reloads the model weights from scratch (~10 s of fixed
# overhead, measured empirically, regardless of sentence length). For
# 8 models x 100 sentences (the 12 full MECO-Argentina texts) this means
# 800 model loads.
#
# llm_generate.py already supports --input <csv> to process several items
# in a single process, loading the model only once. Internally it still
# runs each sentence in a separate forward pass (see process_item() in
# llm_generate.py: it is a Python for-loop, with no padding or real
# tensor-level batching), so surprisal values must be identical to those
# of the original get_surprisals() -- the only thing that changes is how
# many times the model is reloaded. This was validated explicitly against
# the original function before using these functions for the real
# computation (maximum difference: 4.97e-14).

get_surprisals_batch <- function(texts, model = "gpt2") {
  texts <- normalize_text(texts)
  tmp_in  <- tempfile(fileext = ".csv")
  tmp_out <- tempfile(fileext = ".csv")
  on.exit(unlink(c(tmp_in, tmp_out)), add = TRUE)

  write.csv(data.frame(item = seq_along(texts), text = texts, n = 0),
            tmp_in, row.names = FALSE)

  system2("python3",
          args = c("llm_surprisal/llm_generate.py",
                    "--input", shQuote(tmp_in),
                    "--csv", "--model", model,
                    "--output", shQuote(tmp_out)),
          stdout = FALSE, stderr = FALSE)

  # Same precaution as in the original get_surprisals(): na.strings=character(0)
  # because a subword token can literally be "NA".
  df <- read.csv(tmp_out, na.strings = character(0))
  df <- df[!grepl("^<.*>$", df$token), ]

  lapply(seq_along(texts), function(i) {
    d <- df[df$item == i, ]
    words <- strsplit(trimws(texts[i]), "\\s+")[[1]]
    grp <- findInterval(cumsum(nchar(d$token)), cumsum(nchar(words)), left.open = TRUE) + 1
    data.frame(idx = seq_along(words), word = words,
               surprisal = tapply(d$surprisal, grp, sum))
  })
}

add_surprisals_batch <- function(df, models = c("gpt2"),
                                  sentence_col = "sentence",
                                  word_col = "word",
                                  word_idx_col = "phrase",
                                  show_progress = TRUE) {
  sentences <- unique(df[[sentence_col]])

  word_surprisals_for <- function(model) {
    if (show_progress) message(sprintf("Computing surprisals (batched, %s): %d sentences, model loaded once",
                                        model, length(sentences)))
    batch <- get_surprisals_batch(sentences, model = model)
    do.call(rbind, lapply(seq_along(sentences), function(sentence_id) {
      sentence <- sentences[sentence_id]
      ws <- batch[[sentence_id]]
      ws$sentence_id <- sentence_id
      ws$word_idx <- NA
      regions <- unique(df[df[[sentence_col]] == sentence, c(word_idx_col, word_col)])
      regions <- regions[order(regions[[word_idx_col]]), ]
      pos <- 1L
      for (r in seq_len(nrow(regions))) {
        reg_words <- strsplit(trimws(normalize_text(regions[[word_col]][r])), "\\s+")[[1]]
        n <- length(reg_words)
        start <- Find(function(i) all(ws$word[i + seq_len(n) - 1L] == reg_words),
                      pos:(length(ws$word) - n + 1L))
        if (!is.null(start)) {
          ws$word_idx[start + seq_len(n) - 1L] <- regions[[word_idx_col]][r]
          pos <- start + n
        }
      }
      ws
    }))
  }

  word_surprisals <- list()
  for (model in models) {
    ws <- word_surprisals_for(model)
    word_surprisals[[model]] <- ws
    region_surp <- aggregate(surprisal ~ sentence_id + word_idx, ws, sum)
    region_surp <- region_surp[order(region_surp$sentence_id, region_surp$word_idx), ]
    region_surp$prev <- ave(region_surp$surprisal, region_surp$sentence_id,
                            FUN = function(s) c(NA, head(s, -1)))
    region_surp[[sentence_col]] <- sentences[region_surp$sentence_id]
    col_model <- gsub("-", ".", model)
    curr_col <- paste0("Scurr_", col_model)
    prev_col <- paste0("Sprev_", col_model)
    names(region_surp)[names(region_surp) == "surprisal"] <- curr_col
    names(region_surp)[names(region_surp) == "prev"] <- prev_col
    df <- merge(df, region_surp[, c(sentence_col, "word_idx", curr_col, prev_col)],
                by.x = c(sentence_col, word_idx_col),
                by.y = c(sentence_col, "word_idx"))
  }
  attr(df, "word_surprisals") <- word_surprisals
  df
}
```

### `scripts/compute_chunk.R`

```r
# compute_chunk.R
# Usage: Rscript scripts/compute_chunk.R <model> <comma-separated trialids> <output_csv>
# Computes surprisal (batched method, validated against the original
# add_surprisals()) for the given texts (trialid) with the given model,
# and saves the result as CSV.
# Run from the project root, with data/joint_l1_data_trimmed_version2.1.rda in place.

args <- commandArgs(trailingOnly = TRUE)
model <- args[1]
trialids <- as.numeric(strsplit(args[2], ",")[[1]])
outfile <- args[3]

# Make sure system2("python3", ...) finds the Python that has torch installed
# (in the original setup R ran inside a conda env with its own python3).
Sys.setenv(PATH = paste("/usr/bin", Sys.getenv("PATH"), sep = ":"))
source("scripts/TermPaperHelperFunctions.R")
source("scripts/BatchedSurprisal.R")
load("data/joint_l1_data_trimmed_version2.1.rda")
sp <- joint.data[joint.data$lang == "sp", ]

# Stimulus text taken from all participants (not a single one), since the
# trimmed file does not have full coverage per participant.
master <- unique(sp[, c("trialid", "sentnum", "wordnum", "word")])
chunk <- master[master$trialid %in% trialids, ]
chunk <- chunk[order(chunk$trialid, chunk$wordnum), ]
chunk <- add_joined_text_column(chunk, group_cols = c("trialid", "sentnum"), order_col = "wordnum",
                                 word_col = "word", new_col = "sentence")

t0 <- Sys.time()
out <- add_surprisals_batch(chunk, models = model, sentence_col = "sentence",
                             word_col = "word", word_idx_col = "wordnum", show_progress = TRUE)
elapsed <- as.numeric(Sys.time() - t0, units = "secs")
cat(sprintf("Done: %d words, %d sentences, %.1f s\n", nrow(out), length(unique(out$sentence)), elapsed))

write.csv(out, outfile, row.names = FALSE)
cat("Saved to:", outfile, "\n")
```

### `scripts/fit_models.R`

```r
# fit_models.R
#
# Merges the cached surprisals of the 8 models with the baseline predictors
# (EsPal word length/frequency) and the MECO-Argentina reading times,
# applies the exclusions and compares, via likelihood-ratio tests, the
# baseline model against baseline + surprisal for each LLM.
#
# Usage (from the project root):
#   Rscript scripts/fit_models.R
#
# Requires:
#   data/joint_l1_data_trimmed_version2.1.rda   (MECO, download from OSF)
#   data/master_with_baseline.csv               (included in the repo)
#   surprisal_cache/<modelo>_full.csv           (included in the repo)

suppressPackageStartupMessages(library(lme4))

models <- c("bloom-560m", "bloom-1b7", "bloom-3b",
            "xglm-564M", "xglm-1.7B", "xglm-2.9B",
            "gemma-3-1b-pt", "gemma-3-4b-pt")
params_b <- c(0.56, 1.7, 3, 0.564, 1.7, 2.9, 1, 4)
family   <- c(rep("BLOOM", 3), rep("XGLM", 3), rep("Gemma 3", 2))
key <- c("trialid", "sentnum", "wordnum")

# 1) Word-level predictor table (2411 words) -------------------------------
pred <- read.csv("data/master_with_baseline.csv", encoding = "UTF-8")
for (m in models) {
  s <- read.csv(file.path("surprisal_cache", paste0(m, "_full.csv")), encoding = "UTF-8")
  col <- gsub("-", ".", m)
  s <- s[, c(key, paste0("Scurr_", col), paste0("Sprev_", col))]
  pred <- merge(pred, s, by = key)
  stopifnot(nrow(pred) == 2411)
}

# 2) MECO reading data, Argentinian subsample ("sp") -----------------------
load("data/joint_l1_data_trimmed_version2.1.rda")      # -> joint.data
sp <- joint.data[joint.data$lang == "sp", ]            # 87,275 rows, 48 participants
full <- merge(sp[, c("subid", key, "firstrun.dur", "firstfix.dur", "dur",
                     "nfix", "reg.in", "reg.out", "skip")],
              pred[, setdiff(names(pred), c("word", "sentence"))], by = key)
stopifnot(sum(is.na(full$Lcurr)) == 0)

# 3) Exclusions --------------------------------------------------------------
# In MECO wordnum does not restart at each sentence: first/last word of each
# sentence are computed as min/max within (trialid, sentnum).
full$first <- ave(full$wordnum, full$trialid, full$sentnum, FUN = min)
full$last  <- ave(full$wordnum, full$trialid, full$sentnum, FUN = max)
d <- subset(full, !is.na(firstrun.dur) & firstrun.dur > 0 & firstrun.dur < 5000 &
                  wordnum != first & wordnum != last)
cat("Rows after exclusions:", nrow(d), "\n")

d$log_rt <- log(d$firstrun.dur)
ctr <- function(x) x - mean(x, na.rm = TRUE)
for (v in c("Lcurr", "Lprev", "Fcurr", "Fprev")) d[[paste0(v, "_c")]] <- ctr(d[[v]])

# 4) Baseline and surprisal models ------------------------------------------
ctrl <- lmerControl(optimizer = "bobyqa")
m_base <- lmer(log_rt ~ Lcurr_c + Lprev_c + Fcurr_c + Fprev_c +
                 (1 | subid) + (1 | trialid),
               data = d, REML = FALSE, control = ctrl)
print(summary(m_base)$coefficients)

res <- do.call(rbind, lapply(seq_along(models), function(i) {
  col <- gsub("-", ".", models[i])
  d$Scurr_c <- ctr(d[[paste0("Scurr_", col)]])
  d$Sprev_c <- ctr(d[[paste0("Sprev_", col)]])
  m_full <- lmer(log_rt ~ Lcurr_c + Lprev_c + Fcurr_c + Fprev_c + Scurr_c + Sprev_c +
                   (1 | subid) + (1 | trialid),
                 data = d, REML = FALSE, control = ctrl)
  a <- anova(m_base, m_full)
  data.frame(model = models[i], family = family[i], params_b = params_b[i],
             delta_logLik = as.numeric(logLik(m_full) - logLik(m_base)),
             Chisq = a$Chisq[2], Df = a$Df[2], p_value = a$`Pr(>Chisq)`[2],
             delta_AIC = AIC(m_base) - AIC(m_full),
             beta_Scurr = fixef(m_full)[["Scurr_c"]])
}))
print(res, digits = 4)
dir.create("results", showWarnings = FALSE)
write.csv(res, "results/surprisal_model_comparison.csv", row.names = FALSE)
```

### `scripts/plot_scaling.R`

```r
# plot_scaling.R
# Plots the Delta log-likelihood (baseline + surprisal vs. baseline)
# against model size (log scale), one line per model family.
# Usage: Rscript scripts/plot_scaling.R   (after fit_models.R)

suppressPackageStartupMessages(library(ggplot2))

res <- read.csv("results/surprisal_model_comparison.csv")
res$family <- factor(res$family, levels = c("BLOOM", "XGLM", "Gemma 3"),
                     labels = c("BLOOM (Byte-level BPE)",
                                "XGLM (SentencePiece, Unigram)",
                                "Gemma 3 (SentencePiece, BPE)"))

p <- ggplot(res, aes(params_b, delta_logLik, colour = family)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.5) +
  geom_text(aes(label = model), vjust = -0.9, size = 3, show.legend = FALSE) +
  scale_x_log10(breaks = c(0.5, 1, 2, 4), expand = expansion(mult = 0.1)) +
  labs(x = "Model size (billions of parameters, log scale)",
       y = expression(Delta * " log-likelihood over baseline"),
       colour = "Family (tokenizer)",
       title = "Surprisal fit to first-pass reading times, MECO-Argentina") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom", legend.direction = "vertical")

ggsave("figures/scaling_plot_reproduced.png", p, width = 7, height = 5.5, dpi = 200)
cat("Saved: figures/scaling_plot_reproduced.png\n")
```

## Extension to other eye-tracking measures

`results/et_measures_comparison_all.csv` contains the same comparison for `nfix`, `firstfix.dur`, `dur`, `reg.in` and `reg.out`. It is obtained by repeating step 4 with a different dependent variable:

- `firstfix.dur`, `dur` and `nfix`: log of the measure, with `lmer()` and the same formula.
- `reg.in` and `reg.out` (binary): `glmer(..., family = binomial)` with the same predictors and random effects.

The Δ log-likelihood is only comparable **within** a single measure, not across measures with different likelihood families (Gaussian vs. Bernoulli).
