library(tidyverse)
library(knitr)
library(readxl)
library(data.table)
library(vegan)
library(scales)
library(cowplot)
library(RColorBrewer)
library(ggpubr)
library(forcats)
library(ggvenn)
library(ggrepel)
library(ggforce)
library(here)

df <- read.csv(here("Cleaned data CSVs", "qc_data.csv"))

## round all numeric columns to 3 decimal points

df <- df %>%
  mutate(
    bleaching = case_when(
      bleaching == "B"  ~ "Bleached",
      bleaching == "NB" ~ "Non-Bleached",
      is.na(bleaching)  ~ "Not Applicable",
      TRUE              ~ as.character(bleaching)
    ),
    bleaching = factor(bleaching, levels = c("Bleached", "Non-Bleached", "Not Applicable")),
    
    scleractinia = if_else(host_order == "Scleractinia", "1", "0"),
    scleractinia = factor(scleractinia, levels = c("1", "0")),
    location = factor(location),
    symbiont.potential = factor(symbiont.potential),
    host_order = fct_relevel(factor(host_order), "Scleractinia"),
    host_family = factor(host_family),
    host_phylum = factor(host_phylum)
  )

## defining things
met_df <- read.csv(here("Cleaned data CSVs", "merged_met_plot_df.csv"))
met_df <- met_df %>%
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
    )
  )

### take top 20 compound_class by count, with Unknown forced to end
target_classes <- met_df %>%
  count(compound_class, sort = TRUE) %>%
  slice_head(n = 20) %>%
  pull(compound_class) %>%
  trimws()

target_classes <- c(
  setdiff(target_classes, "Unknown"),
  intersect(target_classes, "Unknown")
)

### colors
provided_hex <- c(
  "#1F77B4FF", "#FF7F0EFF", "#2CA02CFF", "#D62728FF",
  "#9467BDFF", "#8C564BFF", "#E377C2FF", "deepskyblue4", "#BCBD22FF",
  "#17BECFFF", "#AEC7E8FF", "#FFBB78FF", "#98DF8AFF", "#FF9896FF",
  "#C5B0D5FF", "#C49C94FF", "#F7B6D2FF", "#9EDAE5FF", "#DBDB8DFF",
  "#C7C7C7FF"
)

spec_colors <- setNames(provided_hex[seq_along(target_classes)], target_classes)

final_palette <- c(spec_colors, "Other" = "gray30")
ordered_levels <- c(target_classes, "Other")

origin_shapes <- c("Host" = 16, "Symbiont" = 3, "Both" = 17, "Unknown" = 8)


glycero_df <- read.csv(here("Cleaned data CSVs", "glycerolipids_fa_TyCOTW.csv"))

################################################################################
## datasets to compare

### define 59 df 
met_df_scler_all <- met_df %>%
  filter(scler_ubiquity == 100)

# define 95 %ile df 
met_df_scler_95 <- met_df %>%
  filter(scler_ubiquity >= 95)

# define 90 %ile df 
met_df_scler_90 <- met_df %>%
  filter(scler_ubiquity >= 90)

# define df of non scler
met_df_nonscler <- met_df %>%
  filter(non_scler_ubiquity >= 95)

# define ML dfs
xgb_df <- met_df %>%
  filter(XGBoost_Importance > 0) %>%
  arrange(desc(XGBoost_Importance)) %>%
  mutate(metabolite = fct_reorder(metabolite, XGBoost_Importance, .desc = TRUE))

rf_df <- met_df %>%
  filter(RandomForest_Importance > 0.001) %>%
  arrange(desc(RandomForest_Importance)) %>%
  mutate(metabolite = fct_reorder(metabolite, RandomForest_Importance, .desc = TRUE))

# define core
df_scler <- df %>% filter(host_order == "Scleractinia", !is.na(host_family))

group_summary_core <- df_scler %>%
  pivot_longer(cols = starts_with("x"), names_to = "metabolite", values_to = "val") %>%
  group_by(host_family, metabolite) %>%
  summarise(present = any(val > 0, na.rm = TRUE), .groups = "drop")

total_families <- length(unique(df_scler$host_family))

core_metabolite_ids <- group_summary_core %>%
  group_by(metabolite) %>%
  summarise(n_families = sum(present)) %>%
  filter(n_families == total_families) %>%
  pull(metabolite)

core_df <- met_df %>%
  filter(metabolite %in% core_metabolite_ids)

# define perm importance df
perm_df <- read.csv(here("machine_learning/perm_importance", "perm_importance_results_kbest.csv"))
important_metabolites_joined <- perm_df %>%
  # Keep only features that contributed positively to the model
  filter(importance_mean > 0) %>%
  left_join(met_df, by = "metabolite")

feature_set_list <- important_metabolites_joined %>%
  group_split(feature_set) %>%
  set_names(map_chr(., ~ first(.x$feature_set)))

all_kbest_df <- feature_set_list[["All_KBest"]] %>%
  select(-X.x)

host_kbest_df <- feature_set_list[["Host_KBest"]] %>%
  select(-X.x)
## x31046_620_59826_13_466 --> perm importance 18.3

################################################################################

# generate contingency table for different compound classes

fisher <- function(target_df, background_df, class_col = "compound_class") {
  all_classes <- unique(background_df[[class_col]])
  
  results <- lapply(all_classes, function(current_class) {
    in_target_is_class <- sum(target_df[[class_col]] == current_class, na.rm = TRUE)
    in_target_not_class <- nrow(target_df) - in_target_is_class
    remainder_df <- background_df %>% filter(!(metabolite %in% target_df$metabolite))
    in_rem_is_class <- sum(remainder_df[[class_col]] == current_class, na.rm = TRUE)
    in_rem_not_class <- nrow(remainder_df) - in_rem_is_class
    contingency_matrix <- matrix(c(in_target_is_class, in_target_not_class, 
                                   in_rem_is_class, in_rem_not_class), 
                                 nrow = 2, byrow = TRUE)
    test <- fisher.test(contingency_matrix)
    data.frame(
      compound_class = current_class,
      count_in_target = in_target_is_class,
      count_in_total = in_rem_is_class,
      not_in_target = in_target_not_class,
      not_in_total = in_rem_not_class,
      odds_ratio = test$estimate,
      p_value = test$p.value,
      stringsAsFactors = FALSE
    )
  })
  
  bind_rows(results) %>%
    mutate(p_adj = p.adjust(p_value, method = "fdr")) %>%
    arrange(p_value)
}

enrichment_results <- list(
  xgb   = fisher(xgb_df, met_df),
  xgb_origin = fisher(xgb_df, met_df, "refined_origin"),
  rf    = fisher(rf_df, met_df),
  rf_origin = fisher(rf_df, met_df, "refined_origin"),
  core  = fisher(core_df, met_df),
  scl90 = fisher(met_df_scler_90, met_df)
)

significant_xgb_origin <- enrichment_results$xgb_origin %>% 
  filter(p_value < 0.05, odds_ratio > 1)
significant_rf_origin <- enrichment_results$rf_origin %>% 
  filter(p_value < 0.05, odds_ratio > 1)

significant_xgb <- enrichment_results$xgb %>% 
  filter(p_value < 0.05, odds_ratio > 1)
significant_xgb$dataset <- "xgboost"
significant_rf <- enrichment_results$rf %>% 
  filter(p_value < 0.05, odds_ratio > 1)
significant_rf$dataset <- "rf"

significant_core <- enrichment_results$core %>% 
  filter(p_value < 0.05, odds_ratio > 1)
significant_core$dataset <- "host_family"

significant_scler90 <- enrichment_results$scl90 %>% 
  filter(p_value < 0.05, odds_ratio > 1)
significant_scler90$dataset <- "scler_90"


################################################################################

## check for overlaps between dfs
list_of_metabolites_xg <- list(
  # Scler_100 = met_df_scler_all$metabolite,
  # Scler_95  = met_df_scler_95$metabolite,
  Scler_90  = met_df_scler_90$metabolite,
  # NonScler_95 = met_df_nonscler$metabolite,
  XGBoost   = xgb_df$metabolite,
  # RandomForest = rf_df$metabolite,
  Core_Family = core_df$metabolite
)

common_metabolites_xg <- Reduce(intersect, list_of_metabolites_xg)

super_core_table_xg <- met_df %>%
  filter(metabolite %in% common_metabolites_xg) %>%
  select(metabolite, compound_class, refined_origin, XGBoost_Importance, RandomForest_Importance,
         scler_ubiquity, non_scler_ubiquity, total_ubiquity) %>%
  mutate(ubiquity_diff = scler_ubiquity - non_scler_ubiquity) %>%
  arrange(desc(XGBoost_Importance))

###########################################

list_of_metabolites_rf <- list(
  # Scler_100 = met_df_scler_all$metabolite,
  # Scler_95  = met_df_scler_95$metabolite,
  Scler_90  = met_df_scler_90$metabolite,
  # NonScler_95 = met_df_nonscler$metabolite,
  # XGBoost   = xgb_df$metabolite,
  RandomForest = rf_df$metabolite,
  Core_Family = core_df$metabolite
)

common_metabolites_rf <- Reduce(intersect, list_of_metabolites_rf)

# Create the summary table with metadata
super_core_table_rf <- met_df %>%
  filter(metabolite %in% common_metabolites_rf) %>%
  select(metabolite, compound_class, refined_origin, XGBoost_Importance, RandomForest_Importance,
         scler_ubiquity, non_scler_ubiquity, total_ubiquity) %>%
  mutate(ubiquity_diff = scler_ubiquity - non_scler_ubiquity) %>%
  arrange(desc(XGBoost_Importance))

super_core_table <- rbind(super_core_table_rf, super_core_table_xg)
super_core_table <- unique(super_core_table)

class_counts <- super_core_table %>%
  summarise(
    DAG = sum(compound_class == "DAG"),
    MADAG = sum(compound_class == "MADAG"),
    TAG = sum(compound_class == "TAG"),
    Ceramide            = sum(compound_class == "Ceramides"),
    Unknown             = sum(compound_class == "Unknown")
  )
class_counts

# write.csv(super_core_table, here("Cleaned data CSVs", "core_df.csv"))

significant_supercore <- fisher(super_core_table, met_df )%>% 
  filter(p_value < 0.05, odds_ratio > 1)
significant_supercore$dataset <- "supercore"

# combined dataset of compound classes enriched in Scleractinia across different comparisons
fishers_csv <- rbind(significant_core, significant_rf, significant_xgb, 
                     significant_scler90, significant_supercore)
# write.csv(fishers_csv,"Metabolite Summary Data/fishers_results.csv")



#########################################
# TAG/DAG/MADAG of interest 
tag_core <- super_core_table %>%
  filter(compound_class == "TAG" | compound_class == "DAG" | compound_class == "MADAG")

mean(tag_core$ubiquity_diff)
# on average these compounds 70.3% more ubiquitous in Scleractinia than outgroups
tag_core_vec <- tag_core$metabolite

# ceramide of interest 
ceramide_core <- super_core_table %>%
  filter(compound_class == "Ceramides")

mean(ceramide_core$ubiquity_diff)
# on average these compounds 37.91% more ubiquitous in Scleractinia than outgroups
ceramide_core_vec <- ceramide_core$metabolite


################################################################################

## ubq abundance 
met_presence_long <- df %>%
  pivot_longer(
    cols = starts_with("x"),
    names_to = "metabolite",
    values_to = "value"
  ) %>%
  mutate(present = value > 0)

ubiquity_overall <- met_presence_long %>%
  group_by(metabolite) %>%
  summarise(ubiquity_all = mean(present) * 100, .groups = "drop")

ubiquity_corals <- met_presence_long %>%
  filter(host_order == "Scleractinia") %>%
  group_by(metabolite) %>%
  summarise(ubiquity_coral = mean(present) * 100, .groups = "drop")

met_summary <- ubiquity_overall %>%
  left_join(ubiquity_corals, by = "metabolite") %>%
  mutate(ubiquity_coral = ifelse(is.na(ubiquity_coral), 0, ubiquity_coral))

coral_present <- met_presence_long %>%
  filter(host_order == "Scleractinia", present) %>%
  distinct(metabolite)

noncoral_present <- met_presence_long %>%
  filter(host_order != "Scleractinia", present) %>%
  distinct(metabolite)

coral_only <- setdiff(coral_present$metabolite, noncoral_present$metabolite)

met_summary <- met_summary %>%
  left_join(
    met_presence_long %>%
      group_by(metabolite) %>%
      summarise(avg_abundance = mean(value, na.rm = TRUE), .groups = "drop"),
    by = "metabolite"
  ) %>%
  mutate(category = ifelse(metabolite %in% coral_only, "Coral-only", "Other"))

class_mapping <- met_df %>%
  select(metabolite, compound_class) %>%
  distinct()

met_df$compound_class <- factor(met_df$compound_class, levels = ordered_levels)
class_mapping <- met_df %>%
  select(metabolite, compound_class, compound_class) %>%
  distinct()

met_plot_comparison <- met_df %>%
  filter(compound_class %in% c("TAG", "DAG", "MADAG")) %>%
  left_join(
    met_summary %>% select(metabolite, avg_abundance), 
    by = "metabolite"
  )

pa_scler_ubq <- ggplot(met_plot_comparison, aes(x = scler_ubiquity, y = avg_abundance)) +
  geom_point(
    aes(color = compound_class, shape = refined_origin),
    size = 4, stroke = 1, alpha = 0.8
  ) +
  facet_wrap(~compound_class) +
  scale_color_manual(values = final_palette, name = "Lipid Class") +
  scale_shape_manual(values = origin_shapes, name = "Metabolite Origin") +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale())) + 
  scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 20)) +
  labs(
    x = "Scleractinian Ubiquity (%)",
    y = "Average Abundance"
  ) +
  geom_text_repel(
    data = . %>% filter(metabolite %in% tag_core_vec),
    aes(label = metabolite),
    box.padding = 0.5, 
    point.padding = 0.3,
    size = 3, 
    max.overlaps = Inf,
    fontface = "bold", 
    color = "black",
    segment.color = "grey30"
  ) +
  theme_pubr() +
  theme(legend.position = "right")
pa_scler_ubq

ggsave(here("misc", "figs", "ubqtag.jpg"), 
       pa_scler_ubq, width=15, height=7, dpi=300)

# pa_total_ubq <- ggplot(met_plot_comparison, aes(x = total_ubiquity, y = avg_abundance)) +
#   geom_point(
#     aes(color = compound_class, shape = refined_origin),
#     size = 4, stroke = 1, alpha = 0.8
#   ) +
#   scale_color_manual(values = final_palette, name = "Lipid Class") +
#   scale_shape_manual(values = origin_shapes, name = "Metabolite Origin") +
#   scale_y_continuous(labels = label_number(scale_cut = cut_short_scale())) + 
#   facet_wrap(~compound_class) +
#   scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 20)) +
#   labs(
#     x = "Total Ubiquity (%)",
#     y = "Average Abundance"
#   ) +
#   theme_pubr() +
#   theme(legend.position = "right")
# pa_total_ubq

met_plot_comparison_2 <- met_df %>%
  filter(compound_class == "Ceramides") %>%
  left_join(
    met_summary %>% select(metabolite, avg_abundance), 
    by = "metabolite"
  )

pb_scler_ubq <- ggplot(met_plot_comparison_2, aes(x = scler_ubiquity, y = avg_abundance)) +
  geom_point(
    aes(color = compound_class, shape = refined_origin),
    size = 4, stroke = 1, alpha = 0.8
  ) +
  facet_wrap(~compound_class) +
  scale_color_manual(values = final_palette) +
  guides(color = "none") +
  scale_shape_manual(values = origin_shapes, name = "Metabolite Origin") +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale())) + 
  scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 20)) +
  labs(
    x = "Scleractinian Ubiquity (%)",
    y = "Average Abundance"
  ) +
  geom_text_repel(
    data = . %>% filter(metabolite %in% ceramide_core_vec),
    aes(label = metabolite),
    box.padding = 0.5, 
    point.padding = 0.3,
    size = 3, 
    max.overlaps = Inf,
    fontface = "bold", 
    color = "black",
    segment.color = "grey30"
  ) +
  theme_pubr() +
  theme(legend.position = "right")
pb_scler_ubq

ggsave(here("misc", "figs", "ubqtag.jpg"), 
       pa_scler_ubq, width=15, height=7, dpi=300)


################################################################################

# subvolcanoes with tag_core highlighted

comm_long <- df %>%
  pivot_longer(
    cols = starts_with("x"), 
    names_to  = "metabolite",
    values_to = "abundance"
  )

comm_long <- comm_long %>%
  mutate(abundance = as.numeric(as.character(abundance)))

stats_data <- comm_long %>%
  select(-scleractinia) %>%                    
  left_join(df %>% select(sample, scleractinia), by = "sample") %>%
  filter(!is.na(scleractinia)) %>%
  mutate(group = if_else(as.character(scleractinia) == "1", "Scleractinia", "Other"))

# compute L2FC and p-values per metabolite
volcano_results <- stats_data %>%
  group_by(metabolite) %>%
  summarise(
    mean_scler = mean(abundance[group == "Scleractinia"], na.rm = TRUE),
    mean_other = mean(abundance[group == "Other"], na.rm = TRUE),
    p_val_raw = wilcox.test(abundance ~ group)$p.value,
    .groups = "drop"
  ) %>%
  mutate(
    p_adj = p.adjust(p_val_raw, method = "bonferroni"),
    log2FC = log2((mean_scler + 1) / (mean_other + 1)),
    # Use adjusted p-value for the y-axis
    neg_log_p_adj = -log10(p_adj)
  )

plot_data_volcano <- volcano_results %>%
  inner_join(met_df %>% select(metabolite, refined_origin, compound_class, display_class), by = "metabolite")

m <- nrow(volcano_results)   
sig_threshold <- -log10(0.05 / m)

################################################################################

p_tag <- plot_data_volcano %>%
  filter(compound_class == "TAG") %>%
  ggplot(aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70") +
  geom_point(aes(color = compound_class), size = 3, alpha = 0.8) +
  geom_text_repel(
    data = . %>% filter(metabolite %in% tag_core_vec),
    aes(label = metabolite),
    box.padding = 0.5, 
    point.padding = 0.3,
    size = 3, 
    max.overlaps = Inf, # Ensures all core metabolites get a label
    fontface = "bold", 
    color = "black",
    segment.color = "grey30"
  ) +
  xlim(-25,25) +
  ylim(0,75) + 
  scale_color_manual(values = final_palette) +
  labs(title = "TAG", x = "log2 Fold Change", y = "-log10(adj. p-value)") +
  theme_pubr() +
  theme(legend.position = "none", plot.title = element_text(size = 18, face = "bold"))

p_dag <- plot_data_volcano %>%
  filter(compound_class == "DAG") %>%
  ggplot(aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70") +
  geom_point(aes(color = compound_class), size = 3, alpha = 0.8) +
  geom_text_repel(
    data = . %>% filter(metabolite %in% tag_core_vec),
    aes(label = metabolite),
    box.padding = 0.5, 
    point.padding = 0.3,
    size = 3, 
    max.overlaps = Inf,
    fontface = "bold", 
    color = "black",
    segment.color = "grey30"
  ) +
  xlim(-25,25) +
  ylim(0,75) + 
  scale_color_manual(values = final_palette) +
  labs(title = "DAG", x = "log2 Fold Change", y = "-log10(adj. p-value)") +
  theme_pubr() +
  theme(legend.position = "none", plot.title = element_text(size = 18, face = "bold"))

p_madag <- plot_data_volcano %>%
  filter(compound_class == "MADAG") %>%
  ggplot(aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70") +
  geom_point(aes(color = compound_class), size = 3, alpha = 0.8) +
  geom_text_repel(
    data = . %>% filter(metabolite %in% tag_core_vec),
    aes(label = metabolite),
    box.padding = 0.5, 
    point.padding = 0.3,
    size = 3, 
    max.overlaps = Inf,
    fontface = "bold", 
    color = "black",
    segment.color = "grey30"
  ) +
  xlim(-25,25) +
  ylim(0,75) + 
  scale_color_manual(values = final_palette) +
  labs(title = "MADAG", x = "log2 Fold Change", y = "-log10(adj. p-value)") +
  theme_pubr() +
  theme(legend.position = "none", plot.title = element_text(size = 18, face = "bold"))

subcano <- plot_grid(p_tag, p_dag, p_madag, ncol = 3)

ggsave(here("misc", "figs", "subcano.jpg"), 
       subcano, width=14, height=7, dpi=300)


p_ceramide <- plot_data_volcano %>%
  filter(compound_class == "Ceramides") %>%
  ggplot(aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70") +
  geom_point(aes(color = compound_class), size = 3, alpha = 0.8) +
  geom_text_repel(
    data = . %>% filter(metabolite %in% ceramide_core_vec),
    aes(label = metabolite),
    box.padding = 0.5, 
    point.padding = 0.3,
    size = 3, 
    max.overlaps = Inf,
    fontface = "bold", 
    color = "black",
    segment.color = "grey30"
  ) +
  xlim(-25,25) +
  ylim(0,75) + 
  scale_color_manual(values = final_palette) +
  labs(title = "Ceramides", x = "log2 Fold Change", y = "-log10(adj. p-value)") +
  theme_pubr() +
  theme(legend.position = "none", plot.title = element_text(size = 18, face = "bold"))
p_ceramide

## important ceramide identified by perm importance
p_ceramide <- plot_data_volcano %>%
  filter(compound_class == "Ceramides") %>%
  ggplot(aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70") +
  geom_point(aes(color = compound_class), size = 3, alpha = 0.8) +
  geom_text_repel(
    data = . %>% filter(metabolite == "x31046_620_59826_13_466"),
    aes(label = metabolite),
    box.padding = 0.5, 
    point.padding = 0.3,
    size = 3, 
    max.overlaps = Inf,
    fontface = "bold", 
    color = "black",
    segment.color = "grey30"
  ) +
  xlim(-25,25) +
  ylim(0,75) + 
  scale_color_manual(values = final_palette) +
  labs(title = "Ceramides", x = "log2 Fold Change", y = "-log10(adj. p-value)") +
  theme_pubr() +
  theme(legend.position = "none", plot.title = element_text(size = 18, face = "bold"))
p_ceramide


################################################################################

# glycerolipid df lookup
glycero_df_core <- glycero_df %>%
  filter(glycero_df$metabolite %in% tag_core_vec)

glycero_df_core %>%
  group_by(metabolite) %>%
  group_walk(~ {
    # Print the metabolite ID as a header for clarity
    cat("\n", paste(rep("-", 30), collapse = ""), "\n")
    cat("Metabolite:", .y$metabolite, "\n")
    cat(paste(rep("-", 30), collapse = ""), "\n")
    
    sub_table <- .x %>%
      select(fatty_acid, compound_name, molecular_formula)
    
    print(as.data.frame(sub_table))
  })

glycero_df_summary <- glycero_df_core %>%
  left_join(
    met_df %>% select(metabolite, scler_ubiquity, non_scler_ubiquity), 
    by = "metabolite"
  ) %>%
  group_by(metabolite) %>%
  summarise(
    FA_Composition = paste(fatty_acid, collapse = " | "),
    Compound_Names = paste(unique(compound_name), collapse = " | "),
    Formulas       = paste(unique(molecular_formula), collapse = " | "),
      Scleractinian_Ubiquity     = paste(unique(scler_ubiquity), collapse = " | "),
    Non_Scleractinian_Ubiquity = paste(unique(non_scler_ubiquity), collapse = " | "),
    .groups = "drop"
  )

print(glycero_df_summary)
# write.csv(glycero_df_summary,"Metabolite Summary Data/glycero_summary.csv")

################################################################################
# fisher's exact tests to compare fatty acid chain lengths

test_df <- glycero_df %>%
  mutate(is_core = if_else(metabolite %in% tag_core_vec, "Core", "Non-Core"))

# extract FA chain lengths
extract_lengths <- function(x) {
  unique(as.numeric(unlist(str_extract_all(x, "\\d+(?=:)"))))
}

# create a long-format dataframe where every metabolite-FA combo is a row
fa_analysis <- test_df %>%
  mutate(lengths = map(fatty_acid, extract_lengths)) %>%
  unnest(lengths) %>%
  filter(lengths %in% c(12, 14, 16, 18, 20, 22, 24, 26, 28)) %>%
  select(metabolite, is_core, lengths) %>%
  distinct()

target_lengths <- c(12, 14, 16, 18, 20, 22, 24, 26, 28)

fisher_results <- map_df(target_lengths, function(len) {
  
  # Create contingency table
  tab <- matrix(c(
    sum(fa_analysis$is_core == "Core" & fa_analysis$lengths == len),
    sum(fa_analysis$is_core == "Non-Core" & fa_analysis$lengths == len),
    length(unique(test_df$metabolite[test_df$is_core == "Core"])) - 
      sum(fa_analysis$is_core == "Core" & fa_analysis$lengths == len),
    length(unique(test_df$metabolite[test_df$is_core == "Non-Core"])) - 
      sum(fa_analysis$is_core == "Non-Core" & fa_analysis$lengths == len)
  ), nrow = 2, byrow = TRUE)
  
  test <- fisher.test(tab)
  tibble(
    FA_Length = as.character(len),
    p_value = test$p.value,
    odds_ratio = test$estimate,
    conf_low = test$conf.int[1],
    conf_high = test$conf.int[2]
  )
})

fisher_results <- fisher_results %>%
  mutate(p_adj = p.adjust(p_value, method = "BH")) %>%
  arrange(p_value)
print(fisher_results)

#########################################
# General function to generate contingency table and run Fisher's test for Fatty acids
test_fa_length <- function(target_length, core_vec, full_df) {
  
  all_metabolites <- unique(full_df$metabolite)
  relevant_core <- intersect(core_vec, all_metabolites)
  relevant_non_core <- setdiff(all_metabolites, relevant_core)
  
  has_target <- full_df %>%
    mutate(lengths = map(fatty_acid, extract_lengths)) %>%
    unnest(lengths) %>%
    filter(lengths == target_length) %>%
    pull(metabolite) %>%
    unique()
  
  tab <- matrix(c(
    sum(relevant_core %in% has_target),      # Core with FA
    sum(relevant_non_core %in% has_target),  # Non-Core with FA
    sum(!(relevant_core %in% has_target)),   # Core without FA
    sum(!(relevant_non_core %in% has_target)) # Non-Core without FA
  ), nrow = 2, byrow = TRUE)
  
  rownames(tab) <- c(paste0("C", target_length, "_Present"), 
                     paste0("C", target_length, "_Absent"))
  colnames(tab) <- c("Core", "Non-Core")
  
  cat("\n--- Contingency Table for FA Length", target_length, "---\n")
  print(tab)
  
  res <- fisher.test(tab)
  cat("\nFisher's Exact Test p-value:", res$p.value, "\n")
  cat("Odds Ratio:", res$estimate, "\n")
  cat("--------------------------------------------\n")
  
  return(res)
}
test_fa_length(22, tag_core_vec, glycero_df)

######################################################

# make df for nina
met_metrics <- met_df %>%
  select(
    metabolite, 
    XGBoost_Importance, 
    RandomForest_Importance, 
    scler_ubiquity, 
    non_scler_ubiquity, 
    total_ubiquity
  )

glycero_df_enriched <- glycero_df %>%
  left_join(met_metrics, by = "metabolite") %>%
  mutate(
    is_super_core = metabolite %in% super_core_table$metabolite,
    ubiquity_diff = scler_ubiquity - non_scler_ubiquity
  )
write.csv(glycero_df_enriched, here("Cleaned data CSVs", "glycero_df_Nina.csv"))



