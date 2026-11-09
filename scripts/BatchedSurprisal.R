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
