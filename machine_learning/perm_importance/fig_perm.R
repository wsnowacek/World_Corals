library(tidyverse)
library(readr)
library(broom)
library(ggpubr)
library(cowplot)

met_df <- read.csv(here("Cleaned data CSVs", "merged_met_plot_df.csv"))
perm_df <- read.csv(here("machine_learning/perm_importance", "perm_importance_results_kbest.csv"))

###########################################################
## 1: make dfs of each column in reduced_feature_sets_multicollinear
# df_reduced <- read.csv("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/machine_learning/ftsets/reduced_feature_sets_multicollinear.csv")
# 
# mets_coral_gb <- df_reduced$coralonly_gb
# mets_coral_rf <- df_reduced$coralonly_rf
# mets_all_rfgb  <- df_reduced$all_rfgb
# 
# df_met_coral_gb <- met_df %>% 
#   filter(metabolite %in% mets_coral_gb)
# 
# df_met_coral_rf <- met_df %>% 
#   filter(metabolite %in% mets_coral_rf)
# 
# df_met_all_rfgb <- met_df %>% 
#   filter(metabolite %in% mets_all_rfgb)
# 
# write.csv(df_met_all_rfgb, "/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/machine_learning/ftsets/met_all_rfgb_reduced.csv")

###########################################################

## permutation importance plots

cols_origin <- c("Host" = "#97B9CBFF", "Symbiont" = "#9057C6FF", 
                 "Both" = "#FFE1BDFF", "Unknown" = "#8DC657FF")

# target_classes <- trimws(c(
#   "Glycerophospholipids", 
#   "Sphingolipids", 
#   "Oligopeptides", 
#   "Glycerolipids", 
#   "Triacylglycerols", 
#   "Steroids", 
#   "Carotenoids (C40)", 
#   "Fatty esters", 
#   "Diacylglyceryl-carboxyhydroxymethylcholines", 
#   "Triterpenoids", 
#   "Fatty amides", 
#   "Phosphatidylglycerocholines", 
#   "Monogalactosyldiacylglycerol", 
#   "Phosphatidylglyceroethanolamines", 
#   "Monoalkyldiacylglycerols", 
#   "Meroterpenoids",
#   "Unknown"
# ))
# 
# provided_hex <- c(
#   "#BEAED4", "#FDC086", "#FFFF99", "#386CB0", "#F0027F", "#BF5B17", "#1B9E77",
#   "#D95F02", "#7570B3", "#984EA3", "#66A61E", "#E6AB02", "#666666", "#A6CEE3", "#B2DF8A",
#   "#FB9A99", "#CBD5E8")
# # "#E5D8BD" "#FDDAEC"
# spec_colors <- setNames(provided_hex, target_classes)
# 
# final_palette <- c(spec_colors, "Other" = "gray60")

#############################################################

# custom compound class palette 

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

target_classes <- met_df %>%
  count(compound_class, sort = TRUE) %>%
  slice_head(n = 20) %>%
  pull(compound_class) %>%
  trimws()

target_classes <- c(
  setdiff(target_classes, "Unknown"),
  intersect(target_classes, "Unknown")
)

provided_hex <- c("#1F77B4FF", "#FF7F0EFF", "#2CA02CFF", "#D62728FF", 
                  "#9467BDFF", "#8C564BFF", "#E377C2FF", "deepskyblue4", "#BCBD22FF", 
                  "#17BECFFF", "#AEC7E8FF", "#FFBB78FF", "#98DF8AFF", "#FF9896FF", 
                  "#C5B0D5FF", "#C49C94FF", "#F7B6D2FF", "#9EDAE5FF", "#DBDB8DFF", 
                  "#C7C7C7FF")

spec_colors <- setNames(provided_hex, target_classes)

final_palette <- c(spec_colors, "Other" = "gray30")

ordered_levels <- c(target_classes, "Other")

process_importance_data <- function(df) {
  df %>%
    mutate(compound_class = trimws(as.character(compound_class))) %>%
    mutate(display_class = if_else(compound_class %in% names(final_palette), 
                                   compound_class, 
                                   "Other")) %>%
    mutate(display_class = fct_relevel(factor(display_class), "Other", after = Inf)) 
}

met_df <- process_importance_data(met_df)

ordered_levels <- c(target_classes, "Other")
met_df$display_class <- factor(met_df$display_class, levels = ordered_levels)

perm_df_clean <- perm_df %>%
  filter(importance_mean > 0) %>%
  left_join(met_df %>% select(metabolite, refined_origin), by = "metabolite") %>%
  mutate(
    feature_set = recode(feature_set,
                         "coral_only_pruned_rf" = "Host RF",
                         "coral_only_pruned_gb" = "Host XGB",
                         "all_pruned_rfgb"      = "All RF/XGB"),
    # Reorder metabolite for the lollipops
    metabolite = reorder(metabolite, importance_mean)
  )

# perm_df_clean <- perm_df %>%
#   filter(importance_mean > 0) %>%
#   left_join(met_df %>% select(metabolite, refined_origin), by = "metabolite") %>%
#   mutate(
#     feature_set = recode(feature_set, 
#                          "all_500" = "Host Selected",
#                          "coralonly_500" = "Coral-Only Selected"),
#     # Reorder metabolite for the lollipops
#     metabolite = reorder(metabolite, importance_mean)
#   )

p_perm <- ggplot(perm_df_clean, aes(x = importance_mean, y = metabolite, color = refined_origin)) +
  geom_errorbarh(aes(xmin = importance_mean - importance_std, 
                     xmax = importance_mean + importance_std),
                 height = 0.6, color = "gray70", alpha = 0.8) +
  geom_point(size = 4) +
  
  scale_color_manual(values = cols_origin, name = "Metabolite Origin") +
  
  facet_grid(feature_set ~ model, scales = "free_y", space = "free_y") +
  
  theme_pubr() +
  labs(x = "Permutation Importance",
       y = "Metabolite") +
  theme(
    strip.background = element_rect(fill = "gray90", color = "black"),
    strip.text = element_text(size = 10, face = "bold"),
    axis.text.y = element_text(size = 8),
    panel.spacing = unit(1, "lines"),
    legend.position = "top"
  )
print(p_perm)

ggsave("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/perm_importance.jpg", 
       p_perm, width = 12, height = 8, dpi = 300)

## 3: by display class

perm_df_display <- perm_df %>%
  filter(importance_mean > 0) %>%
  left_join(met_df %>% select(metabolite, display_class, refined_origin), by = "metabolite") %>%
  mutate(
    feature_set = recode(feature_set, 
                         "coral_only_pruned_rf" = "Host RF",
                         "coral_only_pruned_gb" = "Host XGB",
                         "all_pruned_rfgb"      = "All RF/XGB"),
    # Reorder metabolite for the lollipops
    metabolite = reorder(metabolite, importance_mean)
  )

# 2. Generate the Plot
p_perm_class <- ggplot(perm_df_display, aes(x = importance_mean, y = metabolite, color = display_class)) +
  geom_errorbarh(aes(xmin = importance_mean - importance_std, 
                     xmax = importance_mean + importance_std),
                 height = 0.6, color = "gray70", alpha = 0.8) +
  geom_point(size = 4) +
  scale_color_manual(values = final_palette, name = "Compound Class") +
  facet_grid(feature_set ~ model, scales = "free_y", space = "free_y") +
  
  # Formatting
  theme_pubr() +
  labs(x = "Permutation Importance",
       y = "Metabolite") +
  theme(
    strip.background = element_rect(fill = "gray90", color = "black"),
    strip.text       = element_text(size = 10, face = "bold"),
    axis.text.y      = element_text(size = 7),
    panel.spacing    = unit(1, "lines"),
    # Move legend to top and set direction
    legend.position  = "top",
    legend.direction = "horizontal",
    legend.box       = "horizontal",
    legend.text      = element_text(size = 10), # Slightly smaller to fit 1 row
    legend.title     = element_text(size = 12)
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE, title.position = "top", title.hjust = 0.5))
ggsave("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/perm_importance_class.jpg", 
       p_perm_class, width = 14, height = 8, dpi = 300)

###########################################################

final_plot <- plot_grid(
  p_perm, p_perm_class,
  nrow=2,
  rel_heights=c(1,1.2),
  labels=c('A', 'B'),
  label_size=24
)
ggsave("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/perm_importance.jpg", 
       final_plot, width = 16, height = 14, dpi = 300)
