# =============================================================================
# Spotify Music Analytics Dataset (2015-2025) -- Can we predict a "High
# popularity" track?  Linear regression, logistic regression, decision tree,
# random forest, with model reduction.
# Language: R.  Run:  Rscript spotify_analysis.R   (or source() it in RStudio)
# =============================================================================

# ---- 0. Setup ---------------------------------------------------------------
pkgs <- c("ggplot2", "dplyr", "tidyr", "rpart", "rpart.plot",
          "randomForest", "pROC", "scales")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install)) install.packages(to_install)
invisible(lapply(pkgs, library, character.only = TRUE))

# >>> EDIT THIS PATH if your CSV lives somewhere else <<<
data_path <- "spotify_2015_2025_85k.csv"
out_dir   <- "outputs"
dir.create(out_dir, showWarnings = FALSE)

set.seed(42)
theme_set(theme_minimal(base_size = 13))

# ---- 1. Load & inspect ------------------------------------------------------
raw <- read.csv(data_path, stringsAsFactors = FALSE)
cat("Rows:", nrow(raw), " Columns:", ncol(raw), "\n")
str(raw)
print(summary(raw))
cat("\nMissing values per column:\n")
print(colSums(is.na(raw) | raw == ""))

# ---- 2. Cleaning & feature engineering --------------------------------------
# - track_id / track_name / album_name / artist_name are identifiers (many unique
#   values, no predictive meaning) -> dropped.
# - release_date -> release_year
# - stream_count is extremely right-skewed -> log10
# - categorical variables -> factors
df <- raw %>%
  mutate(release_year = as.integer(substr(release_date, 1, 4)),
         log_streams  = log10(stream_count + 1),
         genre   = factor(genre),
         country = factor(country),
         label   = factor(label),
         explicit = factor(explicit, levels = c(0, 1), labels = c("No", "Yes")),
         mode     = factor(mode,     levels = c(0, 1), labels = c("Minor", "Major")),
         key      = factor(key)) %>%
  select(popularity, danceability, energy, loudness, instrumentalness, tempo,
         duration_ms, key, mode, explicit, genre, country, label,
         release_year, log_streams)

# Binary target for classification: "High" = top quartile of popularity
cut_pop <- quantile(df$popularity, 0.75)
df$hit  <- factor(ifelse(df$popularity > cut_pop, "High", "Low"),
                  levels = c("Low", "High"))
cat("\nHigh-popularity cutoff (75th percentile):", cut_pop, "\n")
print(table(df$hit))

# ---- 3. Exploratory plots ---------------------------------------------------
p1 <- ggplot(df, aes(popularity)) +
  geom_histogram(bins = 40, fill = "#1DB954", colour = "white") +
  geom_vline(xintercept = cut_pop, linetype = "dashed") +
  labs(title = "Distribution of track popularity",
       x = "Popularity score (0-100)", y = "Number of tracks")
ggsave(file.path(out_dir, "01_popularity_hist.png"), p1, width = 7, height = 4.5, dpi = 150)

num_vars <- c("popularity", "danceability", "energy", "loudness",
              "instrumentalness", "tempo", "duration_ms", "release_year",
              "log_streams")
cor_mat <- cor(df[, num_vars])
cat("\nCorrelation with popularity:\n")
print(round(cor_mat[, "popularity"], 3))

cor_long <- as.data.frame(as.table(cor_mat))
p2 <- ggplot(cor_long, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.2f", Freq)), size = 3) +
  scale_fill_gradient2(low = "#d73027", mid = "white", high = "#1a9850", limits = c(-1, 1)) +
  labs(title = "Correlation matrix of numeric variables", x = NULL, y = NULL, fill = "r") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file.path(out_dir, "02_correlation_heatmap.png"), p2, width = 8, height = 6.5, dpi = 150)

p3 <- ggplot(df, aes(log_streams, popularity)) +
  geom_point(alpha = 0.08, colour = "#1DB954") +
  geom_smooth(method = "lm", colour = "black") +
  labs(title = "Popularity vs. streams",
       x = "log10(stream count)", y = "Popularity score")
ggsave(file.path(out_dir, "03_popularity_vs_streams.png"), p3, width = 7, height = 4.5, dpi = 150)

p4 <- ggplot(df, aes(reorder(genre, popularity, median), popularity)) +
  geom_boxplot(fill = "#b3e6c5") + coord_flip() +
  labs(title = "Popularity by genre", x = "Genre", y = "Popularity score")
ggsave(file.path(out_dir, "04_popularity_by_genre.png"), p4, width = 7, height = 5, dpi = 150)

# ---- 4. Train / test split (80 / 20) ----------------------------------------
n        <- nrow(df)
train_id <- sample(seq_len(n), size = 0.8 * n)
train    <- df[train_id, ]
test     <- df[-train_id, ]
cat("\nTrain rows:", nrow(train), " Test rows:", nrow(test), "\n")

# Metrics helpers
rmse <- function(a, p) sqrt(mean((a - p)^2))
r2   <- function(a, p) 1 - sum((a - p)^2) / sum((a - mean(a))^2)
acc_at <- function(actual, prob, thr) {
  pred <- factor(ifelse(prob > thr, "High", "Low"), levels = c("Low", "High"))
  mean(pred == actual)
}
prev_thr <- mean(train$hit == "High")   # threshold = base rate (avoids "all Low")

# Two feature sets:  A = everything (incl. streams),  B = audio/metadata only
all_feats <- setdiff(names(df), c("popularity", "hit"))
feats_A   <- all_feats
feats_B   <- setdiff(all_feats, "log_streams")

results  <- list()
roc_list <- list()
add_res <- function(model, scenario, metric_df) {
  results[[length(results) + 1]] <<- cbind(Model = model, Scenario = scenario, metric_df)
}

run_scenario <- function(feats, tag) {
  cat("\n=======================  SCENARIO", tag, " =======================\n")
  tr_reg <- train[, c("popularity", feats)]
  te_reg <- test[,  c("popularity", feats)]
  tr_cls <- train[, c("hit", feats)]
  te_cls <- test[,  c("hit", feats)]

  # ---- Model 1: Linear regression (+ backward stepwise AIC reduction) --------
  lm_full <- lm(popularity ~ ., data = tr_reg)
  lm_red  <- step(lm_full, direction = "backward", trace = 0)
  cat("\n--- Linear regression: FULL model ---\n");   print(summary(lm_full))
  cat("\n--- Linear regression: REDUCED model (backward AIC) ---\n"); print(summary(lm_red))
  cat("Terms kept:", paste(attr(terms(lm_red), "term.labels"), collapse = ", "), "\n")
  p_full <- predict(lm_full, te_reg); p_red <- predict(lm_red, te_reg)
  add_res("Linear regression (full)",    tag,
          data.frame(Metric = "RMSE / R2", Value = sprintf("%.2f / %.3f", rmse(te_reg$popularity, p_full), r2(te_reg$popularity, p_full))))
  add_res("Linear regression (reduced)", tag,
          data.frame(Metric = "RMSE / R2", Value = sprintf("%.2f / %.3f", rmse(te_reg$popularity, p_red),  r2(te_reg$popularity, p_red))))
  base_rmse <- rmse(te_reg$popularity, mean(tr_reg$popularity))
  cat(sprintf("Baseline (predict the mean) RMSE: %.2f\n", base_rmse))

  if (tag == "A") {
    d <- data.frame(Actual = te_reg$popularity, Predicted = p_red)
    pl <- ggplot(d, aes(Actual, Predicted)) +
      geom_point(alpha = 0.1, colour = "#1DB954") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
      labs(title = "Linear regression (reduced): predicted vs. actual popularity (test set)",
           x = "Actual popularity", y = "Predicted popularity")
    ggsave(file.path(out_dir, "05_lm_pred_vs_actual_A.png"), pl, width = 7, height = 5, dpi = 150)
  }

  # ---- Model 2: Logistic regression (+ backward stepwise reduction) ----------
  glm_full <- glm(hit ~ ., data = tr_cls, family = binomial)
  glm_red  <- step(glm_full, direction = "backward", trace = 0)
  cat("\n--- Logistic regression: REDUCED model ---\n"); print(summary(glm_red))
  cat("Terms kept:", paste(attr(terms(glm_red), "term.labels"), collapse = ", "), "\n")
  pr_full <- predict(glm_full, te_cls, type = "response")
  pr_red  <- predict(glm_red,  te_cls, type = "response")
  roc_full <- roc(te_cls$hit, pr_full, levels = c("Low", "High"), direction = "<", quiet = TRUE)
  roc_red  <- roc(te_cls$hit, pr_red,  levels = c("Low", "High"), direction = "<", quiet = TRUE)
  add_res("Logistic regression (full)",    tag,
          data.frame(Metric = "AUC / Accuracy", Value = sprintf("%.3f / %.3f", auc(roc_full), acc_at(te_cls$hit, pr_full, prev_thr))))
  add_res("Logistic regression (reduced)", tag,
          data.frame(Metric = "AUC / Accuracy", Value = sprintf("%.3f / %.3f", auc(roc_red),  acc_at(te_cls$hit, pr_red,  prev_thr))))
  roc_list[[paste("Logistic (reduced)", tag)]] <<- roc_red

  # ---- Model 3: Decision tree (grown big, pruned by cross-validated cp) -------
  tree_big <- rpart(hit ~ ., data = tr_cls, method = "class",
                    control = rpart.control(cp = 0.0005, minsplit = 50, xval = 10))
  best_cp  <- tree_big$cptable[which.min(tree_big$cptable[, "xerror"]), "CP"]
  tree_pr  <- prune(tree_big, cp = best_cp)
  cat("\n--- Decision tree: pruned at cp =", signif(best_cp, 3),
      "| terminal nodes:", sum(tree_pr$frame$var == "<leaf>"), "---\n")
  print(tree_pr$variable.importance)
  pt_full <- predict(tree_big, te_cls, type = "prob")[, "High"]
  pt_red  <- predict(tree_pr,  te_cls, type = "prob")[, "High"]
  roc_tf <- roc(te_cls$hit, pt_full, levels = c("Low", "High"), direction = "<", quiet = TRUE)
  roc_tp <- roc(te_cls$hit, pt_red,  levels = c("Low", "High"), direction = "<", quiet = TRUE)
  add_res("Decision tree (unpruned)", tag,
          data.frame(Metric = "AUC / Accuracy", Value = sprintf("%.3f / %.3f", auc(roc_tf), acc_at(te_cls$hit, pt_full, prev_thr))))
  add_res("Decision tree (pruned)",   tag,
          data.frame(Metric = "AUC / Accuracy", Value = sprintf("%.3f / %.3f", auc(roc_tp), acc_at(te_cls$hit, pt_red,  prev_thr))))
  roc_list[[paste("Tree (pruned)", tag)]] <<- roc_tp
  tryCatch({
    png(file.path(out_dir, paste0("06_tree_", tag, ".png")), width = 1100, height = 700, res = 120)
    rpart.plot(tree_pr, main = paste("Pruned decision tree - scenario", tag))
    dev.off()
  }, error = function(e) { try(dev.off(), silent = TRUE); cat("(Tree is a single root node - nothing to plot)\n") })

  # ---- Model 4: Random forest (+ reduction by variable importance) ------------
  rf_idx  <- sample(seq_len(nrow(tr_cls)), min(25000, nrow(tr_cls)))   # speed
  rf_full <- randomForest(hit ~ ., data = tr_cls[rf_idx, ], ntree = 200, importance = TRUE)
  imp     <- importance(rf_full, type = 2)
  imp_df  <- data.frame(Variable = rownames(imp), Importance = imp[, 1]) %>% arrange(desc(Importance))
  cat("\n--- Random forest variable importance (MeanDecreaseGini) ---\n"); print(imp_df)
  keep    <- imp_df$Variable[imp_df$Importance >= mean(imp_df$Importance)]   # above-average only
  cat("Variables kept in reduced forest:", paste(keep, collapse = ", "), "\n")
  rf_red  <- randomForest(as.formula(paste("hit ~", paste(keep, collapse = " + "))),
                          data = tr_cls[rf_idx, ], ntree = 200)
  prf_full <- predict(rf_full, te_cls, type = "prob")[, "High"]
  prf_red  <- predict(rf_red,  te_cls, type = "prob")[, "High"]
  roc_rf_f <- roc(te_cls$hit, prf_full, levels = c("Low", "High"), direction = "<", quiet = TRUE)
  roc_rf_r <- roc(te_cls$hit, prf_red,  levels = c("Low", "High"), direction = "<", quiet = TRUE)
  add_res("Random forest (full)",    tag,
          data.frame(Metric = "AUC / Accuracy", Value = sprintf("%.3f / %.3f", auc(roc_rf_f), acc_at(te_cls$hit, prf_full, prev_thr))))
  add_res("Random forest (reduced)", tag,
          data.frame(Metric = "AUC / Accuracy", Value = sprintf("%.3f / %.3f", auc(roc_rf_r), acc_at(te_cls$hit, prf_red,  prev_thr))))
  roc_list[[paste("Random forest (reduced)", tag)]] <<- roc_rf_r

  pimp <- ggplot(imp_df, aes(reorder(Variable, Importance), Importance)) +
    geom_col(fill = "#1DB954") + coord_flip() +
    labs(title = paste("Random forest variable importance - scenario", tag),
         x = "Variable", y = "Mean decrease in Gini impurity")
  ggsave(file.path(out_dir, paste0("07_rf_importance_", tag, ".png")), pimp, width = 7, height = 5, dpi = 150)

  # ---- >= 10 explicit predictions on unseen test rows -------------------------
  idx <- sample(seq_len(nrow(te_cls)), 15)
  pred_tbl <- data.frame(
    Row                = idx,
    Actual_popularity  = te_reg$popularity[idx],
    LM_predicted       = round(p_red[idx], 1),
    Actual_class       = te_cls$hit[idx],
    Logistic_P_High    = round(pr_red[idx], 3),
    Tree_P_High        = round(pt_red[idx], 3),
    RF_P_High          = round(prf_red[idx], 3)
  )
  cat("\n--- 15 predictions on unseen test data (scenario", tag, ") ---\n")
  print(pred_tbl, row.names = FALSE)
  write.csv(pred_tbl, file.path(out_dir, paste0("predictions_", tag, ".csv")), row.names = FALSE)
}

run_scenario(feats_A, "A")   # A: all variables incl. log_streams
run_scenario(feats_B, "B")   # B: audio features + metadata only (no streams)

# ---- 5. Summary table & ROC curves ------------------------------------------
res_tbl <- do.call(rbind, results)
cat("\n================ SUMMARY OF TEST-SET PERFORMANCE ================\n")
cat("Scenario A = all variables (incl. log streams); B = without streams\n")
print(res_tbl, row.names = FALSE)
write.csv(res_tbl, file.path(out_dir, "results_summary.csv"), row.names = FALSE)

roc_df <- do.call(rbind, lapply(names(roc_list), function(nm) {
  r <- roc_list[[nm]]
  data.frame(Model = sprintf("%s (AUC=%.3f)", nm, auc(r)),
             FPR = 1 - r$specificities, TPR = r$sensitivities)
}))
p_roc <- ggplot(roc_df, aes(FPR, TPR, colour = Model)) +
  geom_line(linewidth = 0.9) +
  geom_abline(linetype = "dashed") +
  labs(title = "ROC curves on the held-out test set",
       x = "False positive rate (1 - specificity)", y = "True positive rate (sensitivity)")
ggsave(file.path(out_dir, "08_roc_curves.png"), p_roc, width = 9, height = 5.5, dpi = 150)

cat("\nDone. Plots, predictions and summary saved in ./", out_dir, "\n", sep = "")
