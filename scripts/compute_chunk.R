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
