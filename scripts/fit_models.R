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
