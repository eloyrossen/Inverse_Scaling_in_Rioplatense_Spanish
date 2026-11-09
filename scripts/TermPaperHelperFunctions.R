normalize_text <- function(x) {
  x <- gsub("[‘’]", "'", x)   # curly single quotes / apostrophes
  x <- gsub("[“”]", '"', x)   # curly double quotes
  gsub("[–—]", "-", x)        # en / em dashes
}


get_surprisals <- function(text, model = "gpt2") {
  # Available models: openai-gpt, gpt2 (default), gpt2-large, german-gpt2,
  # german-gpt2-larger, bloom-560m, bloom-1b7, bloom-3b, xglm-564M, xglm-1.7B,
  # xglm-2.9B, gemma-3-1b-pt, gemma-3-4b-pt
  text <- normalize_text(text)
  output <- system2("python3",
                    args = c("llm_surprisal/llm_generate.py",
                             shQuote(text), "--csv", "--model", model),
                    stdout = TRUE, stderr = FALSE)
  # na.strings=character(0): a BPE subword token can literally be "NA"
  # (e.g. in "DNA"); without this, read.csv silently turns it into a
  # missing value and desyncs the token/word alignment below.
  df <- read.csv(text = paste(output, collapse = "\n"), na.strings = character(0))
  # Drop special tokens (e.g. </s> from XGLM):
  df <- df[!grepl("^<.*>$", df$token), ]
  # Merge subword tokens into words by cumulative character position:
  words <- strsplit(trimws(text), "\\s+")[[1]]
  grp <- findInterval(cumsum(nchar(df$token)),
                      cumsum(nchar(words)), left.open = TRUE) + 1
  data.frame(idx = seq_along(words), word = words,
             surprisal = tapply(df$surprisal, grp, sum))
}


add_surprisals <- function(df, models = c("gpt2"),
                           sentence_col = "sentence",
                           word_col = "word",
                           word_idx_col = "phrase",
                           show_progress = TRUE) {
  sentences <- unique(df[[sentence_col]])
  # Word-by-word surprisals for each unique sentence under one model, running
  # get_surprisals only once per sentence. Each word is tagged with a
  # sentence_id (mappable back to df) and the word_idx (region index) it
  # belongs to.
  word_surprisals_for <- function(model) {
    do.call(rbind, lapply(seq_along(sentences), function(sentence_id) {
      if (show_progress) {
        message(sprintf("Computing surprisals (%s): sentence %d/%d",
                        model, sentence_id, length(sentences)))
      }
      sentence <- sentences[sentence_id]
      ws <- get_surprisals(sentence, model = model)  # normalizes text internally
      ws$sentence_id <- sentence_id
      ws$word_idx <- NA
      regions <- unique(df[df[[sentence_col]] == sentence, c(word_idx_col, word_col)])
      regions <- regions[order(regions[[word_idx_col]]), ]
      # Locate each region's words within the sentence words (scanning forward,
      # so repeated text matches in order) and tag those words with the
      # region's word_idx:
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
  # Add a Scurr_<model> and Sprev_<model> column pair per requested model:
  word_surprisals <- list()  # interim product, kept per model
  for (model in models) {
    ws <- word_surprisals_for(model)
    word_surprisals[[model]] <- ws
    # Sum word surprisals within each region (word_idx), join onto df by
    # sentence text + word_idx. Two columns per model: current region's
    # surprisal (Scurr_<model>) and the preceding region's surprisal
    # (Sprev_<model>, NA for the first region in a sentence).
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


add_joined_text_column <- function(df, group_cols, order_col,
                                   word_col = "word_with_punct",
                                   new_col = "text") {
  # Adds new_col to df: for each row, the words of its group (group_cols),
  # ordered by order_col, pasted together into one string. Works for any
  # unit a dataset happens to be missing a "full string" column for
  # (sentence, clause, paragraph, whole text, ...) -- just point group_cols
  # at whatever columns identify that unit.
  group_id <- do.call(paste, c(df[group_cols], sep = "_"))
  ord <- order(group_id, df[[order_col]])
  joined <- tapply(df[[word_col]][ord], group_id[ord], paste, collapse = " ")
  df[[new_col]] <- joined[group_id]
  df
}
