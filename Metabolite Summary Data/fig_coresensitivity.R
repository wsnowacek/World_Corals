################################################################################
# Core coral metabolome sensitivity analysis
#
# Repeats the core-metabolome analysis using Scleractinian ubiquity thresholds
# of 80%, 85%, 90%, and 95%. At each threshold, the sensitivity core is:
#
#   family core (present in every sampled Scleractinian family)
#     INTERSECT Scleractinian ubiquity set
#     INTERSECT ML set (important in XGBoost OR random forest)
#
# The script also estimates the expected three-way overlap under independence
# while preserving the observed size of each criterion set.
################################################################################

library(tidyverse)
library(here)
set.seed(123)

ubiquity_thresholds <- c(80, 85, 90, 95)
n_null_simulations <- 10000L

xgb_importance_threshold <- 0
rf_importance_threshold <- 0.001 # to reduce complexity

output_dir <- here("misc", "core_sensitivity_analysis")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

################################################################################
# Read and standardize data

df <- read.csv(here("Cleaned data CSVs", "qc_data_PQN.csv"))
df$X <- NULL

met_df <- read.csv(
  here("Cleaned data CSVs", "merged_met_plot_df.csv"),
  stringsAsFactors = FALSE
) %>%
  mutate(
    compound_class = recode(
      compound_class,
      "Carotenoids (C40, Î²-Î²)" = "Carotenoids",
      "Oxidized glycerophospholipids" = "OxPL",
      "Glycerophosphoethanolamines" = "GPEtn",
      "Neutral glycosphingolipids" = "Neutral GSL",
      "Triacylglycerols" = "TAG",
      "Diacylglycerols" = "DAG",
      "Prenyl quinone meroterpenoids" = "TQ/THQs"
    ),
    compound_class = replace_na(trimws(compound_class), "Unknown")
  )

glycero_df <- read.csv(
  here("Cleaned data CSVs", "glycerolipids_fa_TyCOTW.csv"),
  stringsAsFactors = FALSE
)

required_met_columns <- c(
  "metabolite",
  "compound_class",
  "scler_ubiquity",
  "XGBoost_Importance",
  "RandomForest_Importance"
)

missing_met_columns <- setdiff(required_met_columns, names(met_df))
if (length(missing_met_columns) > 0) {
  stop(
    "Missing required columns in met_df: ",
    paste(missing_met_columns, collapse = ", ")
  )
}

if (anyDuplicated(met_df$metabolite)) {
  stop("met_df$metabolite must contain one row per metabolite.")
}

feature_ids <- names(df)[startsWith(names(df), "x")]
universe_ids <- intersect(feature_ids, met_df$metabolite)

if (length(universe_ids) == 0) {
  stop("No metabolite features were shared between df and met_df.")
}

if (length(setdiff(feature_ids, universe_ids)) > 0) {
  warning(
    length(setdiff(feature_ids, universe_ids)),
    " feature columns in df were absent from met_df and excluded from the universe."
  )
}

met_universe <- met_df %>%
  filter(metabolite %in% universe_ids)

df_scler <- df %>%
  filter(host_order == "Scleractinia", !is.na(host_family))

total_families <- n_distinct(df_scler$host_family)

family_presence <- df_scler %>%
  select(host_family, all_of(universe_ids)) %>%
  pivot_longer(
    cols = all_of(universe_ids),
    names_to = "metabolite",
    values_to = "value"
  ) %>%
  group_by(host_family, metabolite) %>%
  summarise(
    present = any(value > 0, na.rm = TRUE),
    .groups = "drop"
  )

family_core_ids <- family_presence %>%
  group_by(metabolite) %>%
  summarise(n_families = sum(present), .groups = "drop") %>%
  filter(n_families == total_families) %>%
  pull(metabolite)

xgb_ids <- met_universe %>%
  filter(!is.na(XGBoost_Importance),
         XGBoost_Importance > xgb_importance_threshold) %>%
  pull(metabolite)

rf_ids <- met_universe %>%
  filter(!is.na(RandomForest_Importance),
         RandomForest_Importance > rf_importance_threshold) %>%
  pull(metabolite)

ml_union_ids <- union(xgb_ids, rf_ids)

ml_set_summary <- tibble(
  criterion = c("XGBoost", "Random forest", "Either ML model"),
  threshold = c(
    paste0("> ", xgb_importance_threshold),
    paste0("> ", rf_importance_threshold),
    "XGBoost OR random forest"
  ),
  n_metabolites = c(length(xgb_ids), length(rf_ids), length(ml_union_ids))
)

print(ml_set_summary)

################################################################################
# Null model for the overlap of three sets

# Given a universe of N metabolites and observed set sizes a, b, and c:
#   E[|A intersection B intersection C|] = a * b * c / N^2.
#
# The simulation below draws the A-B overlap from its exact hypergeometric
# distribution and then draws the overlap of A-B with C. This is equivalent to
# independently permuting set membership while preserving all three set sizes,
# but is substantially faster than explicitly shuffling N labels 10,000 times.

simulate_independent_overlap <- function(
    universe_n,
    family_n,
    ubiquity_n,
    ml_n,
    observed_n,
    n_sim = 10000L
) {
  overlap_family_ubiquity <- rhyper(
    nn = n_sim,
    m = family_n,
    n = universe_n - family_n,
    k = ubiquity_n
  )
  
  simulated_three_way <- rhyper(
    nn = n_sim,
    m = overlap_family_ubiquity,
    n = universe_n - overlap_family_ubiquity,
    k = ml_n
  )
  
  expected_analytic <- family_n * ubiquity_n * ml_n / universe_n^2
  null_sd <- sd(simulated_three_way)
  
  tibble(
    universe_n = universe_n,
    family_core_n = family_n,
    ubiquity_set_n = ubiquity_n,
    ml_union_n = ml_n,
    observed_overlap_n = observed_n,
    expected_overlap_analytic = expected_analytic,
    expected_overlap_simulated = mean(simulated_three_way),
    null_sd = null_sd,
    null_median = median(simulated_three_way),
    null_ci_low = unname(quantile(simulated_three_way, 0.025)),
    null_ci_high = unname(quantile(simulated_three_way, 0.975)),
    fold_enrichment = if_else(
      expected_analytic > 0,
      observed_n / expected_analytic,
      NA_real_
    ),
    z_score = if_else(
      null_sd > 0,
      (observed_n - mean(simulated_three_way)) / null_sd,
      NA_real_
    ),
    empirical_p_enrichment =
      (sum(simulated_three_way >= observed_n) + 1) / (n_sim + 1)
  )
}

################################################################################

# calculate null expectation

sensitivity_sets <- set_names(
  map(ubiquity_thresholds, function(cutoff) {
    ubiquity_ids <- met_universe %>%
      filter(!is.na(scler_ubiquity), scler_ubiquity >= cutoff) %>%
      pull(metabolite)
    
    sensitivity_core_ids <- Reduce(
      intersect,
      list(family_core_ids, ubiquity_ids, ml_union_ids)
    )
    
    list(
      cutoff = cutoff,
      ubiquity_ids = ubiquity_ids,
      core_ids = sensitivity_core_ids
    )
  }),
  paste0("Scler_", ubiquity_thresholds)
)

sensitivity_summary <- imap_dfr(sensitivity_sets, function(set_data, set_name) {
  core_metadata <- met_universe %>%
    filter(metabolite %in% set_data$core_ids)
  
  null_summary <- simulate_independent_overlap(
    universe_n = length(universe_ids),
    family_n = length(family_core_ids),
    ubiquity_n = length(set_data$ubiquity_ids),
    ml_n = length(ml_union_ids),
    observed_n = length(set_data$core_ids),
    n_sim = n_null_simulations
  )
  
  null_summary %>%
    mutate(
      threshold = set_data$cutoff,
      sensitivity_set = set_name,
      annotated_n = sum(core_metadata$compound_class != "Unknown", na.rm = TRUE),
      xgb_important_n = sum(set_data$core_ids %in% xgb_ids),
      rf_important_n = sum(set_data$core_ids %in% rf_ids),
      important_in_both_models_n =
        sum(set_data$core_ids %in% intersect(xgb_ids, rf_ids)),
      .before = 1
    )
})

print(sensitivity_summary, width = Inf)

################################################################################
# Membership and compound-class composition at each threshold

core_membership <- imap_dfr(sensitivity_sets, function(set_data, set_name) {
  met_universe %>%
    filter(metabolite %in% set_data$core_ids) %>%
    mutate(
      threshold = set_data$cutoff,
      sensitivity_set = set_name,
      selected_by_xgboost = metabolite %in% xgb_ids,
      selected_by_rf = metabolite %in% rf_ids,
      .before = 1
    )
})

core_class_composition <- core_membership %>%
  count(threshold, sensitivity_set, compound_class, name = "n_metabolites") %>%
  group_by(threshold, sensitivity_set) %>%
  mutate(
    proportion = n_metabolites / sum(n_metabolites),
    percentage = 100 * proportion
  ) %>%
  ungroup() %>%
  arrange(threshold, desc(n_metabolites))

################################################################################
# Fisher's exact tests for compound-class enrichment

fisher_class_enrichment <- function(target_ids, background_df, set_name, cutoff) {
  background_ids <- unique(background_df$metabolite)
  target_ids <- intersect(unique(target_ids), background_ids)
  remainder_ids <- setdiff(background_ids, target_ids)
  
  target_df <- background_df %>% filter(metabolite %in% target_ids)
  remainder_df <- background_df %>% filter(metabolite %in% remainder_ids)
  
  all_classes <- sort(unique(background_df$compound_class))
  
  map_dfr(all_classes, function(current_class) {
    core_class_n <- sum(target_df$compound_class == current_class, na.rm = TRUE)
    core_other_n <- nrow(target_df) - core_class_n
    remainder_class_n <- sum(
      remainder_df$compound_class == current_class,
      na.rm = TRUE
    )
    remainder_other_n <- nrow(remainder_df) - remainder_class_n
    
    contingency_table <- matrix(
      c(
        core_class_n,
        core_other_n,
        remainder_class_n,
        remainder_other_n
      ),
      nrow = 2,
      byrow = TRUE,
      dimnames = list(
        Dataset = c("Sensitivity core", "Dataset remainder"),
        Class = c("Target class", "Other classes")
      )
    )
    
    test <- fisher.test(contingency_table)
    
    tibble(
      threshold = cutoff,
      sensitivity_set = set_name,
      compound_class = current_class,
      count_in_core = core_class_n,
      core_total = length(target_ids),
      count_in_remainder = remainder_class_n,
      remainder_total = length(remainder_ids),
      odds_ratio = unname(test$estimate),
      conf_low = test$conf.int[1],
      conf_high = test$conf.int[2],
      p_value = test$p.value
    )
  }) %>%
    mutate(
      p_adj = p.adjust(p_value, method = "BH"),
      enriched = odds_ratio > 1 & p_adj < 0.05,
      depleted = odds_ratio < 1 & p_adj < 0.05
    ) %>%
    arrange(p_adj, desc(odds_ratio))
}

class_enrichment_results <- imap_dfr(
  sensitivity_sets,
  ~ fisher_class_enrichment(
    target_ids = .x$core_ids,
    background_df = met_universe,
    set_name = .y,
    cutoff = .x$cutoff
  )
)

significant_class_enrichment <- class_enrichment_results %>%
  filter(enriched)

print(significant_class_enrichment, n = Inf)

################################################################################
# FA chain length enrichment

### need to try and redo using Nina methodology

extract_lengths <- function(x) {
  map(x, function(value) {
    if (is.na(value) || value == "") {
      return(numeric())
    }
    unique(as.numeric(str_extract_all(value, "\\d+(?=:)")[[1]]))
  })
}

glycerolipid_classes <- c("TAG", "DAG", "MADAG")
target_fa_lengths <- c(12, 14, 16, 18, 20, 22, 24, 26, 28)

glycerolipid_background_ids <- met_universe %>%
  filter(compound_class %in% glycerolipid_classes) %>%
  pull(metabolite) %>%
  intersect(unique(glycero_df$metabolite))

fa_by_metabolite <- glycero_df %>%
  filter(metabolite %in% glycerolipid_background_ids) %>%
  mutate(FA_Length = extract_lengths(fatty_acid)) %>%
  unnest(FA_Length) %>%
  filter(FA_Length %in% target_fa_lengths) %>%
  distinct(metabolite, FA_Length)

glycerolipid_summary <- imap_dfr(sensitivity_sets, function(set_data, set_name) {
  core_glycerolipid_ids <- intersect(
    set_data$core_ids,
    met_universe$metabolite[
      met_universe$compound_class %in% glycerolipid_classes
    ]
  )
  
  annotated_glycerolipid_ids <- intersect(
    core_glycerolipid_ids,
    unique(glycero_df$metabolite)
  )
  
  tibble(
    threshold = set_data$cutoff,
    sensitivity_set = set_name,
    core_glycerolipids_n = length(core_glycerolipid_ids),
    manually_annotated_glycerolipids_n = length(annotated_glycerolipid_ids),
    TAG_n = sum(
      met_universe$metabolite %in% core_glycerolipid_ids &
        met_universe$compound_class == "TAG"
    ),
    DAG_n = sum(
      met_universe$metabolite %in% core_glycerolipid_ids &
        met_universe$compound_class == "DAG"
    ),
    MADAG_n = sum(
      met_universe$metabolite %in% core_glycerolipid_ids &
        met_universe$compound_class == "MADAG"
    )
  )
})

test_fa_lengths <- function(core_ids, cutoff, set_name) {
  relevant_core <- intersect(core_ids, glycerolipid_background_ids)
  relevant_noncore <- setdiff(glycerolipid_background_ids, relevant_core)
  
  map_dfr(target_fa_lengths, function(fa_length) {
    has_length <- fa_by_metabolite %>%
      filter(FA_Length == fa_length) %>%
      pull(metabolite)
    
    core_with_length_n <- sum(relevant_core %in% has_length)
    core_without_length_n <- sum(!(relevant_core %in% has_length))
    background_with_length_n <- sum(relevant_noncore %in% has_length)
    background_without_length_n <- sum(!(relevant_noncore %in% has_length))
    
    contingency_table <- matrix(
      c(
        core_with_length_n,
        core_without_length_n,
        background_with_length_n,
        background_without_length_n
      ),
      nrow = 2,
      byrow = TRUE,
      dimnames = list(
        Dataset = c("Core glycerolipids", "Background glycerolipids"),
        FA = c("Contains chain length", "Does not contain chain length")
      )
    )
    
    test <- fisher.test(contingency_table)
    
    tibble(
      threshold = cutoff,
      sensitivity_set = set_name,
      FA_Length = fa_length,
      core_with_length = core_with_length_n,
      core_without_length = core_without_length_n,
      background_with_length = background_with_length_n,
      background_without_length = background_without_length_n,
      odds_ratio = unname(test$estimate),
      conf_low = test$conf.int[1],
      conf_high = test$conf.int[2],
      p_value = test$p.value
    )
  }) %>%
    mutate(
      p_adj = p.adjust(p_value, method = "BH"),
      enriched = odds_ratio > 1 & p_adj < 0.05,
      depleted = odds_ratio < 1 & p_adj < 0.05
    ) %>%
    arrange(p_adj, FA_Length)
}

fa_length_enrichment <- imap_dfr(
  sensitivity_sets,
  ~ test_fa_lengths(
    core_ids = .x$core_ids,
    cutoff = .x$cutoff,
    set_name = .y
  )
)

################################################################################
# Plot observed overlap against the independence null

overlap_null_plot <- ggplot(
  sensitivity_summary,
  aes(x = factor(threshold), y = observed_overlap_n)
) +
  geom_linerange(
    aes(ymin = null_ci_low, ymax = null_ci_high),
    linewidth = 0.9,
    color = "grey45"
  ) +
  geom_point(
    aes(y = expected_overlap_simulated),
    size = 3,
    shape = 21,
    fill = "white",
    color = "black",
    stroke = 0.9
  ) +
  geom_point(
    size = 3.5,
    color = "#D62728"
  ) +
  geom_line(
    aes(group = 1),
    linewidth = 0.8,
    color = "#D62728"
  ) +
  labs(
    x = "Minimum Scleractinian ubiquity (%)",
    y = "Intersection Size (Metabolite Features)"
  ) +
  theme_classic(base_size = 12)

print(overlap_null_plot)

################################################################################
# Save results

write.csv(
  sensitivity_summary,
  file.path(output_dir, "sensitivity_summary_and_null_model.csv"),
  row.names = FALSE
)

write.csv(
  core_membership,
  file.path(output_dir, "sensitivity_core_membership.csv"),
  row.names = FALSE
)

write.csv(
  core_class_composition,
  file.path(output_dir, "sensitivity_core_class_composition.csv"),
  row.names = FALSE
)

write.csv(
  class_enrichment_results,
  file.path(output_dir, "sensitivity_core_class_enrichment.csv"),
  row.names = FALSE
)

write.csv(
  glycerolipid_summary,
  file.path(output_dir, "sensitivity_glycerolipid_summary.csv"),
  row.names = FALSE
)

write.csv(
  fa_length_enrichment,
  file.path(output_dir, "sensitivity_fatty_acid_length_enrichment.csv"),
  row.names = FALSE
)

write.csv(
  ml_set_summary,
  file.path(output_dir, "machine_learning_set_summary.csv"),
  row.names = FALSE
)

ggsave(
  filename = file.path(output_dir, "observed_vs_null_overlap.pdf"),
  plot = overlap_null_plot,
  width = 7,
  height = 5
)

ggsave(
  filename = file.path(output_dir, "observed_vs_null_overlap.png"),
  plot = overlap_null_plot,
  width = 7,
  height = 5,
  dpi = 300
)

saveRDS(
  list(
    settings = list(
      ubiquity_thresholds = ubiquity_thresholds,
      xgb_importance_threshold = xgb_importance_threshold,
      rf_importance_threshold = rf_importance_threshold,
      n_null_simulations = n_null_simulations,
      random_seed = 123
    ),
    universe_ids = universe_ids,
    family_core_ids = family_core_ids,
    xgb_ids = xgb_ids,
    rf_ids = rf_ids,
    ml_union_ids = ml_union_ids,
    sensitivity_sets = sensitivity_sets,
    sensitivity_summary = sensitivity_summary,
    class_enrichment_results = class_enrichment_results,
    fa_length_enrichment = fa_length_enrichment
  ),
  file.path(output_dir, "core_sensitivity_analysis_objects.rds")
)

message("Sensitivity analysis complete. Results saved to: ", output_dir)