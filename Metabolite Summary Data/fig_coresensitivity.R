################################################################################
# find core interesection set  using Scleractinian ubiquity thresholds
# of 80%, 85%, 90%, and 95%. At each threshold, the sensitivity core is:
#   family core (present in every sampled Scleractinian family)
#     INTERSECT Scleractinian ubiquity set
#     INTERSECT ML set (important in XGBoost OR random forest)
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

feature_ids <- names(df)[startsWith(names(df), "x")]
universe_ids <- intersect(feature_ids, met_df$metabolite)

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
# distribution and then draws the overlap of A-B with C

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
# Plot observed overlap against the independence null

overlap_null_plot <- ggplot(
  sensitivity_summary,
  aes(x = factor(threshold))
) +
  geom_linerange(
    aes(
      ymin = null_ci_low,
      ymax = null_ci_high
    ),
    linewidth = 0.9,
    color = "grey45",
    show.legend = FALSE
  ) +
  geom_point(
    aes(
      y = expected_overlap_simulated,
      color = "Null model"
    ),
    size = 3,
    shape = 21,
    fill = "white",
    stroke = 0.9
  ) +
  geom_line(
    aes(
      y = observed_overlap_n,
      group = 1,
      color = "Observed core"
    ),
    linewidth = 0.8
  ) +
  geom_point(
    aes(
      y = observed_overlap_n,
      color = "Observed core"
    ),
    size = 3.5
  ) +
  scale_color_manual(
    name = NULL,
    breaks = c("Observed core", "Null model"),
    values = c(
      "Observed core" = "#D62728",
      "Null model" = "black"
    )
  ) +
  guides(
    color = guide_legend(
      override.aes = list(
        shape = c(NA, 21),
        linetype = c(1, 0),
        fill = c(NA, "white"),
        linewidth = c(0.8, 0),
        size = c(3, 3)
      )
    )
  ) +
  labs(
    x = "Minimum Scleractinian Ubiquity (%)",
    y = "Intersection Size (Metabolite Features)"
  ) +
  theme_classic(base_size = 12) +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.78, 0.82),
    legend.background = element_rect(
      fill = scales::alpha("white", 0.85),
      color = "black",
      linewidth = 0.4
    ),
    legend.margin = margin(5, 7, 5, 7)
  )

print(overlap_null_plot)
ggsave(here("misc", "figs/pqn", "null_core.jpg"),
       overlap_null_plot, width=8, height=7, dpi=300)


################################################################################

### additional null comparison

# Conditional null model:
# Hold the combined conservation set fixed and randomize ML membership
#
# C = family-core metabolites intersected with the threshold-specific
#     Scleractinian ubiquity set
# M = metabolites important in XGBoost OR RF
#
# Under the null:
#   E[|C intersection M|] = |C| * |M| / N

simulate_conditional_ml_overlap <- function(
    universe_n,
    conservation_n,
    ml_n,
    observed_n,
    n_sim = 10000L
) {
  # Exact null distribution generated by randomly assigning ML membership
  # while preserving the observed sizes of the universe, conservation set,
  # and ML set
  simulated_overlap <- rhyper(
    nn = n_sim,
    m = conservation_n,
    n = universe_n - conservation_n,
    k = ml_n
  )
  
  expected_analytic <- conservation_n * ml_n / universe_n
  null_mean <- mean(simulated_overlap)
  null_sd <- sd(simulated_overlap)
  
  # Exact upper-tail hypergeometric probability
  exact_p_enrichment <- phyper(
    q = observed_n - 1,
    m = conservation_n,
    n = universe_n - conservation_n,
    k = ml_n,
    lower.tail = FALSE
  )
  
  tibble(
    universe_n = universe_n,
    conservation_set_n = conservation_n,
    ml_union_n = ml_n,
    observed_overlap_n = observed_n,
    expected_overlap_analytic = expected_analytic,
    expected_overlap_simulated = null_mean,
    null_sd = null_sd,
    null_median = median(simulated_overlap),
    null_ci_low = unname(quantile(simulated_overlap, 0.025)),
    null_ci_high = unname(quantile(simulated_overlap, 0.975)),
    fold_enrichment = if_else(
      expected_analytic > 0,
      observed_n / expected_analytic,
      NA_real_
    ),
    z_score = if_else(
      null_sd > 0,
      (observed_n - null_mean) / null_sd,
      NA_real_
    ),
    exact_p_enrichment = exact_p_enrichment,
    empirical_p_enrichment =
      (sum(simulated_overlap >= observed_n) + 1) / (n_sim + 1)
  )
}

################################################################################
# Run conditional null comparison at each ubiquity threshold

conditional_null_summary <- imap_dfr(
  sensitivity_sets,
  function(set_data, set_name) {
    
    # Hold this observed conservation set fixed
    conservation_ids <- intersect(
      family_core_ids,
      set_data$ubiquity_ids
    )
    
    # Observed overlap between conservation and ML importance
    observed_ids <- intersect(
      conservation_ids,
      ml_union_ids
    )
    
    conditional_result <- simulate_conditional_ml_overlap(
      universe_n = length(universe_ids),
      conservation_n = length(conservation_ids),
      ml_n = length(ml_union_ids),
      observed_n = length(observed_ids),
      n_sim = n_null_simulations
    )
    
    conditional_result %>%
      mutate(
        threshold = set_data$cutoff,
        sensitivity_set = set_name,
        .before = 1
      )
  }
)

print(conditional_null_summary, width = Inf)

stopifnot(
  identical(
    conditional_null_summary$observed_overlap_n,
    sensitivity_summary$observed_overlap_n
  )
)

################################################################################

# new plot with both null models

null_plot_data <- bind_rows(
  sensitivity_summary %>%
    transmute(
      threshold,
      null_model = "Independent set null",
      expected_overlap = expected_overlap_simulated,
      null_ci_low,
      null_ci_high
    ),
  
  conditional_null_summary %>%
    transmute(
      threshold,
      null_model = "Conditional ML null",
      expected_overlap = expected_overlap_simulated,
      null_ci_low,
      null_ci_high
    )
) %>%
  mutate(
    threshold = factor(
      threshold,
      levels = ubiquity_thresholds
    ),
    null_model = factor(
      null_model,
      levels = c(
        "Independent set null",
        "Conditional ML null"
      )
    )
  )

observed_plot_data <- sensitivity_summary %>%
  mutate(
    threshold = factor(
      threshold,
      levels = ubiquity_thresholds
    )
  )

null_dodge <- position_dodge(width = 0.35)

overlap_null_plot <- ggplot(
  observed_plot_data,
  aes(x = threshold)
) +
  # Null-model 95% intervals
  geom_linerange(
    data = null_plot_data,
    aes(
      ymin = null_ci_low,
      ymax = null_ci_high,
      color = null_model,
      group = null_model
    ),
    position = null_dodge,
    linewidth = 0.9,
    alpha = 0.7,
    show.legend = FALSE
  ) +
  
  # Null-model expected overlaps
  geom_point(
    data = null_plot_data,
    aes(
      y = expected_overlap,
      color = null_model,
      shape = null_model,
      group = null_model
    ),
    position = null_dodge,
    size = 3.2,
    fill = "white",
    stroke = 1
  ) +
  
  # Observed core
  geom_line(
    aes(
      y = observed_overlap_n,
      group = 1,
      color = "Observed core"
    ),
    linewidth = 0.8
  ) +
  geom_point(
    aes(
      y = observed_overlap_n,
      color = "Observed core",
      shape = "Observed core"
    ),
    size = 3.5
  ) +
  
  scale_color_manual(
    name = NULL,
    breaks = c(
      "Observed core",
      "Independent set null",
      "Conditional ML null"
    ),
    values = c(
      "Observed core" = "#D62728",
      "Independent set null" = "black",
      "Conditional ML null" = "#0072B2"
    )
  ) +
  scale_shape_manual(
    name = NULL,
    breaks = c(
      "Observed core",
      "Independent set null",
      "Conditional ML null"
    ),
    values = c(
      "Observed core" = 16,
      "Independent set null" = 21,
      "Conditional ML null" = 24
    )
  ) +
  
  # Construct a single customized legend
  guides(
    shape = "none",
    color = guide_legend(
      override.aes = list(
        shape = c(16, 21, 24),
        linetype = c(1, 0, 0),
        fill = c(NA, "white", "white"),
        linewidth = c(0.8, 0, 0),
        size = c(3, 3, 3)
      )
    )
  ) +
  
  labs(
    x = "Minimum Scleractinian Ubiquity (%)",
    y = "Intersection Size (Metabolite Features)"
  ) +
  theme_classic(base_size = 12) +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.76, 0.82),
    legend.background = element_rect(
      fill = scales::alpha("white", 0.9),
      color = "black",
      linewidth = 0.4
    ),
    legend.margin = margin(5, 7, 5, 7)
  )

print(overlap_null_plot)

ggsave(
  here("misc", "figs/pqn", "null_core.jpg"),
  overlap_null_plot,
  width = 8,
  height = 7,
  dpi = 300
)

################################################################################

# make one final DF for publication table
# contains one row for each: 80, 85, 90, 95
# include expected intersection size + CI for both null models
# include p-value enrichment for both null
# include any significant compound class enrichments (size in core + p-value)
################################################################################
# Final publication table

# Format p-values consistently
format_p_value <- function(p) {
  case_when(
    is.na(p)   ~ NA_character_,
    p < 0.001  ~ formatC(p, format = "e", digits = 2),
    TRUE       ~ formatC(p, format = "f", digits = 3)
  )
}

################################################################################
# Summarize significant compound-class enrichments

format_p_value <- function(p) {
  case_when(
    is.na(p)   ~ NA_character_,
    p < 0.001  ~ formatC(p, format = "e", digits = 2),
    TRUE       ~ formatC(p, format = "f", digits = 3)
  )
}

class_enrichment_table <- class_enrichment_results %>%
  filter(
    enriched,
    p_adj < 0.05
  ) %>%
  arrange(threshold, p_adj) %>%
  group_by(threshold) %>%
  summarise(
    significant_compound_class_enrichments = paste0(
      compound_class,
      ": ",
      count_in_core,
      "/",
      core_total,
      " core features",
      " (OR = ",
      sprintf("%.2f", odds_ratio),
      "; FDR p = ",
      format_p_value(p_adj),
      ")",
      collapse = "; "
    ),
    .groups = "drop"
  )

################################################################################
# Format the independent-set null results

independent_null_table <- sensitivity_summary %>%
  mutate(
    independent_p_adj = p.adjust(
      empirical_p_enrichment,
      method = "BH"
    )
  ) %>%
  transmute(
    threshold,
    ubiquity_set_n,
    family_core_n,
    ml_union_n,
    observed_core_n = observed_overlap_n,
    
    independent_expected_n = expected_overlap_simulated,
    independent_ci_low = null_ci_low,
    independent_ci_high = null_ci_high,
    independent_fold_enrichment = fold_enrichment,
    independent_empirical_p = empirical_p_enrichment,
    independent_p_adj
  )

################################################################################
# Format the conditional ML null results

conditional_null_table <- conditional_null_summary %>%
  mutate(
    conditional_p_adj = p.adjust(
      exact_p_enrichment,
      method = "BH"
    )
  ) %>%
  transmute(
    threshold,
    conservation_set_n,
    
    conditional_expected_n = expected_overlap_simulated,
    conditional_ci_low = null_ci_low,
    conditional_ci_high = null_ci_high,
    conditional_fold_enrichment = fold_enrichment,
    conditional_exact_p = exact_p_enrichment,
    conditional_empirical_p = empirical_p_enrichment,
    conditional_p_adj
  )

################################################################################
# try later - combine results into a table

# publication_table <- independent_null_table %>%
#   left_join(
#     conditional_null_table,
#     by = "threshold"
#   ) %>%
#   left_join(
#     class_enrichment_table,
#     by = "threshold"
#   ) %>%
#   mutate(
#     significant_compound_class_enrichments = replace_na(
#       significant_compound_class_enrichments,
#       "None"
#     ),
#     
#     # Compact formatted fields for publication
#     independent_null_expected_95_CI = paste0(
#       sprintf("%.2f", independent_expected_n),
#       " (",
#       independent_ci_low,
#       "–",
#       independent_ci_high,
#       ")"
#     ),
#     
#     conditional_null_expected_95_CI = paste0(
#       sprintf("%.2f", conditional_expected_n),
#       " (",
#       conditional_ci_low,
#       "–",
#       conditional_ci_high,
#       ")"
#     ),
#     
#     independent_null_p = format_p_value(
#       independent_empirical_p
#     ),
#     
#     conditional_null_p = format_p_value(
#       conditional_exact_p
#     )
#   ) %>%
#   select(
#     `Ubiquity threshold (%)` = threshold,
#     `Ubiquity set (n)` = ubiquity_set_n,
#     `Family-core set (n)` = family_core_n,
#     `Combined conservation set (n)` = conservation_set_n,
#     `ML importance set (n)` = ml_union_n,
#     `Observed core (n)` = observed_core_n,
#     `Independent null: expected overlap (95% CI)` =
#       independent_null_expected_95_CI,
#     `Independent null: empirical p` =
#       independent_null_p,
#     `Conditional null: expected overlap (95% CI)` =
#       conditional_null_expected_95_CI,
#     `Conditional null: exact p` =
#       conditional_null_p,
#     `Significantly enriched compound classes` =
#       significant_compound_class_enrichments
#   ) %>%
#   arrange(`Ubiquity threshold (%)`)
# 
# print(publication_table, width = Inf)
# 
# write.csv(publication_table, file = "/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/core_null.csv", 
#           row.names = FALSE, fileEncoding = "UTF-8")
