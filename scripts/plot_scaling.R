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
