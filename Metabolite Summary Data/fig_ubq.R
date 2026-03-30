library(tidyverse)
library(knitr)
library(readxl)
library(data.table)
library(vegan)
library(scales)
library(ggraph)
library(cowplot)
library(ggdendro)
library(ggridges)
library(dendextend)
library(RColorBrewer)
library(ggpubr)
library(forcats)
library(caret)
library(tibble)
library(GGally)
library(stringr)
library(RColorBrewer)
library(ggrepel)
library(Polychrome)
library(here)

# read in data
df <- read.csv(here("Metabolite Summary Data", "qc_data.csv"))
met_df <- read.csv(here("Cleaned data CSVs", "merged_met_plot_df.csv"))

present_metabolites <- df %>% 
  select(starts_with("x")) %>% 
  colnames()

met_df <- met_df %>%
  filter(met_df$metabolite %in% present_metabolites)

cols_bleaching <- c(
  "Bleached" = "#FF847CFF", 
  "Non-Bleached" = "#019875FF", 
  "Not Applicable" = "#D3D3D3")

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

# color palettes
cols_location <-c("#002594FF", "#E0B2CDFF", "#54C4E3FF", "#F3AA4FFF")
cols_symbiont  <- c("#D84D16FF", "#FFF800FF", "#8FDA04FF")
cols_phylum <- c("#24492EFF", "#015B58FF", "#2C6184FF", "#59629BFF", "#89689DFF", "#BA7999FF", "#E69B99FF")
cols_sclero    <- c("1" = "#DE7862FF", "0" = "#D8AF39FF")


##################################################
# for Compound Class - ClassyFire annotation

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
# "#D95F02", "#7570B3", "#984EA3", "#66A61E", "#E6AB02", "#666666", "#A6CEE3", "#B2DF8A",
# "#FB9A99", "#CBD5E8")
# # "#E5D8BD" "#FDDAEC"
# spec_colors <- setNames(provided_hex, target_classes)
# 
# final_palette <- c(spec_colors, "Other" = "gray60")
# ordered_levels <- c(target_classes, "Other")
# 
# process_importance_data <- function(df) {
#   df %>%
#     mutate(compound_superclass = trimws(as.character(compound_superclass))) %>%
#     mutate(display_class = if_else(compound_superclass %in% names(final_palette), 
#                                    compound_superclass, 
#                                    "Other")) %>%
#     mutate(display_class = fct_relevel(factor(display_class), "Other", after = Inf)) 
# }
# 
# met_plot_df <- process_importance_data(met_df)
# 
# ordered_levels <- c(target_classes, "Other")
# met_plot_df$display_class <- factor(met_plot_df$display_class, levels = ordered_levels)

##################################################

# for compound class - custom spectral library

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

met_plot_df <- process_importance_data(met_df)

ordered_levels <- c(target_classes, "Other")
met_plot_df$display_class <- factor(met_plot_df$display_class, levels = ordered_levels)


##################################################

origin_shapes <- c("Host" = 16, "Symbiont" = 3, "Both" = 17, "Unknown" = 8)

#################################################################################

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
met_plot_df$display_class <- factor(met_plot_df$display_class, levels = ordered_levels)

x_vline_pos <- met_summary %>%
  filter(category == "Coral-only") %>%
  pull(ubiquity_all) %>%
  { if(length(.) == 0) NA_real_ else max(., na.rm = TRUE) }

class_mapping <- met_plot_df %>%
  select(metabolite, display_class) %>%
  distinct()

met_plot_df$display_class <- factor(met_plot_df$display_class, levels = ordered_levels)
class_mapping <- met_plot_df %>%
  select(metabolite, display_class) %>%
  distinct()

met_summary_classed <- met_summary %>%
  left_join(class_mapping, by = "metabolite") %>%
  mutate(display_class = replace_na(display_class, "Other"))

met_summary_classed <- met_summary_classed %>%
  left_join(met_df %>% select(metabolite, refined_origin), by = "metabolite") 

met_summary_classed$refined_origin <- factor(
  met_summary_classed$refined_origin, 
  levels = c("Host", "Symbiont", "Both", "Unknown")
)

pa <- ggplot(met_summary_classed, aes(x = ubiquity_all, y = avg_abundance)) +
  geom_point(
    aes(color = display_class, shape = refined_origin), # Changed fill to color
    size = 3,      
    stroke = 0.8,  # Increased slightly for better visibility of shapes 3 and 8
    alpha = 0.85
  ) +
  
  # Use scale_color_manual to match the scatter plots H and I
  scale_color_manual(values = final_palette) +
  scale_shape_manual(values = origin_shapes) +
  
  scale_y_continuous(
    labels = label_number(scale_cut = cut_short_scale())
  ) + 
  scale_x_continuous(
    limits = c(0, 100), 
    breaks = seq(0, 100, by = 20)
  ) +
  labs(
    x = "Ubiquity",
    y = "Abundance",
    color = "Compound Class", # Updated label
    shape = "Metabolite Origin"
  ) + 
  theme_pubr() + 
  theme(
    plot.background   = element_rect(fill = "white", color = NA),
    panel.background  = element_rect(fill = "white", color = NA),
    axis.line         = element_line(color = "black"),
    legend.position   = "right",
    legend.text       = element_text(size = 9),
    plot.title        = element_text(hjust = 0.5, face = "bold", size = 16)
  )

pa

#################################################################################

## remake this plot using only Scleractinian samples (e.g. df$scleractinia ==1) 
## and host-only metabolites (e.g. met_df$refined_origin == 'Host')
host_only_mets <- met_df %>%
  filter(refined_origin == "Host") %>%
  pull(metabolite)

# 2. Pivot data using only Scleractinian samples and Host-only metabolites
met_presence_scler_host <- df %>%
  filter(scleractinia == 1) %>% # Only Scleractinian samples
  select(sample, all_of(intersect(names(.), host_only_mets))) %>% # Only Host metabolites
  pivot_longer(
    cols = -sample,
    names_to = "metabolite",
    values_to = "value"
  ) %>%
  mutate(present = value > 0)

# 3. Calculate Ubiquity and Avg Abundance for this subset
met_summary_scler <- met_presence_scler_host %>%
  group_by(metabolite) %>%
  summarise(
    ubiquity_all = mean(present) * 100,
    avg_abundance = mean(value, na.rm = TRUE),
    .groups = "drop"
  )

# 4. Join with chemical class metadata
met_summary_classed_scler <- met_summary_scler %>%
  left_join(class_mapping, by = "metabolite") %>%
  mutate(display_class = replace_na(display_class, "Other"))

pb <- ggplot(met_summary_classed_scler, aes(x = ubiquity_all, y = avg_abundance)) +
  geom_point(
    aes(fill = display_class),
    shape = 21,
    size = 3,
    stroke = 0.4,
    color = "black",
    alpha = 0.85
  ) +
  scale_fill_manual(values = final_palette) +
  scale_y_continuous(
    labels = label_number(scale_cut = cut_short_scale())
  ) + 
  scale_x_continuous(
    limits = c(0, 100), 
    breaks = seq(0, 100, by = 20)
  ) +
  labs(
    x = "Scleractinian Ubiquity",
    y = "Abundance",
    fill = "Compound Class",
  ) + 
  theme_pubr() + 
  theme(
    plot.background   = element_rect(fill = "white", color = NA),
    panel.background  = element_rect(fill = "white", color = NA),
    axis.line         = element_line(color = "black"),
    legend.position   = "right",
    legend.text       = element_text(size = 9),
  ) +
  guides(
    fill = guide_legend(override.aes = list(size = 4, shape = 21))
  )
pb

#################################################################################
# 
# feature_importance_comparison_all <- read.csv("/work/hs325/World_Corals/machine_learning/all_mets/featureimportanceallmets.csv")
# 
# feature_importance_comparison_all <- feature_importance_comparison_all %>%
#   dplyr::rename(metabolite = Feature)
# merged_df_all <- feature_importance_comparison_all %>%
#   inner_join(met_df, by = "metabolite")
# 
# importance_scores <- merged_df_all %>%
#   select(metabolite, XGBoost_Importance, RandomForest_Importance)
# 
# met_plot_df <- met_plot_df %>%
#   left_join(importance_scores, by = "metabolite") %>%
#   mutate(
#     XGBoost_Importance = replace_na(XGBoost_Importance, 0),
#     RandomForest_Importance = replace_na(RandomForest_Importance, 0)
#   )
# 
# ordered_levels <- c(target_classes, "Other")
# met_plot_df$display_class <- factor(met_plot_df$display_class, levels = ordered_levels)

## change # of metabolites to plot here
xgb_plot_data <- met_plot_df %>%
  arrange(desc(XGBoost_Importance)) %>%
  slice_head(n = 60) %>%
  mutate(metabolite = fct_reorder(metabolite, XGBoost_Importance, .desc = TRUE))

rf_plot_data <- met_plot_df %>%
  arrange(desc(RandomForest_Importance)) %>%
  slice_head(n = 120) %>%
  mutate(metabolite = fct_reorder(metabolite, RandomForest_Importance, .desc = TRUE))

 
# xgb_plot_df$display_class <- factor(xgb_plot_df$display_class, levels = ordered_levels)
# rf_plot_df$display_class  <- factor(rf_plot_df$display_class,  levels = ordered_levels)
# plot_df_all$display_class <- factor(plot_df_all$display_class, levels = ordered_levels)

#################### make CDE plots ###########################
p1 <- ggbarplot(xgb_plot_data, x = "metabolite", y = "XGBoost_Importance",
                fill = "display_class", color = "transparent",
                xlab = "Metabolite", ylab = "XGBoost Feature Importance") +
  theme_pubr() +
  scale_fill_manual(values = final_palette) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) + 
  scale_x_discrete(expand = expansion(add = c(1, 0.5))) + 
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank(),
        legend.position = "none")
p1

p2 <- ggbarplot(rf_plot_data, x = "metabolite", y = "RandomForest_Importance",
                fill = "display_class", color = "transparent",
                xlab = "Metabolite", ylab = "RF Feature Importance") +
  theme_pubr() +
  scale_fill_manual(values = final_palette) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  scale_x_discrete(expand = expansion(add = c(1, 0.5))) +
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank(),
        legend.position = "none")
p2

top_labels <- met_plot_df %>%
  mutate(dist = sqrt(XGBoost_Importance^2 + RandomForest_Importance^2)) %>%
  arrange(desc(dist)) %>%
  slice_head(n = 5) %>%
  pull(metabolite)

origin_shapes <- c("Host" = 16, "Symbiont" = 3, "Both" = 17, "Unknown" = 8)

p3 <- ggscatter(met_plot_df, 
                x = "XGBoost_Importance", 
                y = "RandomForest_Importance",
                color = "display_class", 
                shape = "refined_origin", 
                palette = final_palette,
                label = "metabolite", 
                size = 5,
                font.x = c(22, "bold"),          # Increases x-axis label size
                font.y = c(16, "bold"),          # Increases y-axis label size
                label.select = top_labels,
                repel = TRUE,                      
                font.label = c(10, "italic"),      
                cor.coeff = TRUE, 
                cor.method = "pearson",
                xlab = "XGBoost Feature Importance", 
                ylab = "RF Feature Importance") +
  scale_shape_manual(values = origin_shapes) +
  theme_pubr() + 
  theme(legend.position = "right",
        axis.title = element_text(size = 16)) +
  guides(
    color = "none",
    fill = "none",
    shape = guide_legend(title = "Metabolite Origin", override.aes = list(size = 5)))
p3

ggsave("/work/hs325/World_Corals/misc/figs/p3alone.jpg", p3, width = 12, height = 8, dpi = 300)

#################################################################################
# 
# ubiquity_pal <- c(
#   "[0,20]"   = "#440154FF", # Dark Purple
#   "(20,40]"  = "#3B528BFF", # Blue
#   "(40,60]"  = "#21908CFF", # Teal
#   "(60,80]"  = "#5DC863FF", # Green
#   "(80,100]" = "#FDE725FF"  # Yellow
# )
# 
# # Helper to prepare data for ubiquity-colored barplots
# prep_ubiq_plot <- function(df, importance_col, n_slice) {
#   df %>%
#     arrange(desc(!!sym(importance_col))) %>%
#     slice_head(n = n_slice) %>%
#     mutate(
#       ubiquity_bin = cut(scler_ubiquity, 
#                          breaks = c(-Inf, 20, 40, 60, 80, 100), 
#                          labels = names(ubiquity_pal), 
#                          include.lowest = TRUE),
#       metabolite = fct_reorder(metabolite, !!sym(importance_col), .desc = TRUE)
#     )
# }
# 
# xgb_ubiq_data <- prep_ubiq_plot(met_plot_df, "XGBoost_Importance", 60)
# rf_ubiq_data  <- prep_ubiq_plot(met_plot_df, "RandomForest_Importance", 120)
# 
# p_xgb_ubiq <- ggplot(xgb_ubiq_data, aes(x = metabolite, y = XGBoost_Importance, fill = ubiquity_bin)) +
#   geom_bar(stat = "identity", color = "black", linewidth = 0.1) +
#   scale_fill_manual(values = ubiquity_pal, drop = FALSE) +
#   scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
#   theme_pubr() +
#   labs(
#     x = "Metabolite",
#     y = "XGBoost Feature Importance",
#     fill = "Ubiquity Percentage"
#   ) +
#   theme(
#     axis.text.x = element_blank(),
#     axis.ticks.x = element_blank(),
#     legend.position = "right"
#   )
# 
# # --- Random Forest Importance colored by Scleractinian Ubiquity ---
# p_rf_ubiq <- ggplot(rf_ubiq_data, aes(x = metabolite, y = RandomForest_Importance, fill = ubiquity_bin)) +
#   geom_bar(stat = "identity", color = "black", linewidth = 0.1) +
#   scale_fill_manual(values = ubiquity_pal, drop = FALSE) +
#   scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
#   theme_pubr() +
#   labs(
#     x = "Metabolite",
#     y = "RF Feature Importance",
#     fill = "Scleractinian Ubiquity Percentage"
#   ) +
#   theme(
#     axis.text.x = element_blank(),
#     axis.ticks.x = element_blank(),
#     legend.position = "right"
#   )
# 
# print(p_xgb_ubiq)
# print(p_rf_ubiq)

#################################################################################

## final plot: two ubiquity-abundance plots with only the most important metabolites from 
# XGBoost Importance and RF importance

top_xgb_mets <- met_plot_df %>% arrange(desc(XGBoost_Importance)) %>% head(43) %>% pull(metabolite)
top_rf_mets  <- met_plot_df %>% arrange(desc(RandomForest_Importance)) %>% head(50) %>% pull(metabolite)
## 268: #of mets with importance > 0.001

met_summary_xgb <- met_plot_df %>% filter(metabolite %in% top_xgb_mets)
met_summary_rf  <- met_plot_df %>% filter(metabolite %in% top_rf_mets)

p_xgb_ubiq_scatter <- ggplot(met_summary_xgb, aes(x = non_scler_ubiquity, y = scler_ubiquity)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray60") +
  geom_point(
    aes(color = display_class, shape = refined_origin), 
    size = 4, stroke = 1, alpha = 0.9
  ) +
  # Using scale_color_manual because shapes 3 and 8 only take color
  scale_color_manual(values = final_palette) +
  scale_shape_manual(values = origin_shapes) + 
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, by = 20)) +
  scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, by = 20)) +
  labs(
    x = "Non-Scleractinian Ubiquity (%)", 
    y = "Scleractinian Ubiquity (%)", 
    color = "Compound Class",
    shape = "Metabolite Origin"
  ) +
  theme_pubr() +
  theme(
    legend.position = "none", 
    plot.title = element_text(size = 12, face = "bold")
  )
p_xgb_ubiq_scatter

p_rf_ubiq_scatter <- ggplot(met_summary_rf, aes(x = non_scler_ubiquity, y = scler_ubiquity)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray60") +
  geom_point(
    aes(color = display_class, shape = refined_origin), 
    size = 4, stroke = 1, alpha = 0.9
  ) +
  scale_color_manual(values = final_palette) +
  scale_shape_manual(values = origin_shapes) + 
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, by = 20)) +
  scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, by = 20)) +
  labs(
    x = "Non-Scleractinian Ubiquity (%)", 
    y = "Scleractinian Ubiquity (%)", 
    color = "Compound Class",
    shape = "Metabolite Origin"
  ) +
  theme_pubr() +
  theme(
    legend.position = "none", 
    plot.title = element_text(size = 12, face = "bold")
  )
p_rf_ubiq_scatter


#################################################################################
levels(met_plot_df$display_class) <- str_wrap(levels(met_plot_df$display_class), width = 20)
names(final_palette) <- str_wrap(names(final_palette), width = 20)

legend_dummy <- ggplot(met_plot_df, aes(x = XGBoost_Importance, y = RandomForest_Importance)) +
  geom_point(aes(fill = display_class, shape = refined_origin)) +
  scale_fill_manual(
    values = final_palette, 
    name = "Compound Class", 
    drop = FALSE
  ) +
  scale_shape_manual(
    values = origin_shapes, 
    name = "Metabolite Origin"
  ) +
  theme_pubr() +
  theme(
    legend.position = "top",
    legend.direction = "horizontal",
    legend.box = "vertical",
    legend.text = element_text(size = 12, lineheight = 0.8), # lineheight handles wrapped text
    legend.title = element_text(size = 14, face = "bold"),
    legend.key.height = unit(0.8, "cm"), # Increase height to fit wrapped text
    legend.spacing.x = unit(0.2, "cm")
  ) +
  guides(
    fill = guide_legend(
      ncol = 6,           # Set columns to 6; with 18 items, it MUST show 3 rows
      byrow = TRUE,
      order = 1,
      override.aes = list(shape = 21, size = 5, stroke = 0.5)
    ),
    shape = guide_legend(
      nrow = 1,
      order = 2
    )
  )

unified_legend <- get_legend(legend_dummy)

row_ab <- plot_grid(
  pa + theme(legend.position = "none"), 
  pb + theme(legend.position = "none"), 
  labels = c("A", "B"), label_size = 18, ncol = 2
)

# Row 2: XGB and RF Barplots (C, D)
row_cd <- plot_grid(
  p1 + theme(legend.position = "none") + labs(subtitle = "XGBoost"), 
  p2 + theme(legend.position = "none") + labs(subtitle = "Random Forest"), 
  labels = c("C", "D"),
  label_size = 18, ncol = 2
)

# Row 3: Scatter Importance (E) - Centered or full width
row_e <- plot_grid(
  p3 + theme(legend.position = "none"), 
  labels = c("E"),
  label_size = 18, ncol = 1
)
# 
# row_fg <- plot_grid(
#   p_xgb_ubiq + theme(legend.position = "none"), 
#   p_rf_ubiq, # Keep one ubiquity legend here as it's a different scale/meaning
#   labels = c("F", "G"), label_size = 18, ncol = 2, rel_widths = c(1, 1.2)
# )

row_hi <- plot_grid(
  p_xgb_ubiq_scatter + theme(legend.position = "none") + labs(subtitle = "XGBoost"), 
  p_rf_ubiq_scatter 
  + theme(legend.position = "none")
  + labs(subtitle = "Random Forest"), 
  labels = c("F", "G"),
  label_size = 18, ncol = 2
)

row_cde <- plot_grid(
  row_cd, row_e,
  nrow = 2)
row_cde <- plot_grid(
  unified_legend, row_e,
  nrow = 2, rel_heights = c(0.4,1))

ggsave("/work/hs325/World_Corals/misc/figs/fig5cde_ppt.jpg", 
       row_cde, width = 18, height = 8, dpi = 300)

true_final_figure <- plot_grid(
  unified_legend,
  row_ab,
  row_cd,
  row_e,
  # row_fg,
  row_hi,
  ncol = 1,
  rel_heights = c(0.4, 1, 1, 1.2, 1, 1) # Adjust heights based on content density
)

# Save high-resolution for publication
ggsave("/work/hs325/World_Corals/misc/figs/fig5.png", 
       true_final_figure, width = 18, height = 24, dpi = 600, bg = "white")


################################################################################

#save each row
temp_save <- plot_grid(
  unified_legend, row_hi,
  ncol = 1,
  rel_heights = c(0.5, 1) 
)
ggsave("/work/hs325/World_Corals/misc/figs/fig5_ppt.jpg", 
       temp_save, width = 14, height = 8, dpi = 600, bg = "white")


################################################################################
target_mets <- c("x39055_948_80202_15_826", 
                 "x15256_518_49365_7_407", 
                 "x23838_655_56593_11_538")

p_xgb_ubiq_scatter <- ggplot(met_summary_xgb, aes(x = non_scler_ubiquity, y = scler_ubiquity)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray60") +
  geom_point(
    aes(color = display_class, shape = refined_origin), 
    size = 4, stroke = 1, alpha = 0.9
  ) +
  geom_text_repel(
    data = subset(met_summary_xgb, metabolite %in% target_mets),
    aes(label = metabolite),
    size = 3.5,
    fontface = "bold",
    box.padding = 2.5,        # Distance between label and other objects
    point.padding = 1.5,      # Distance between label and the data point
    force = 50,               # Strength of the repulsion
    force_pull = 0.5,         
    min.segment.length = 0,   # Always draw the line, no matter how short
    segment.color = "grey30",
    segment.curvature = -0.1, # Adds a slight curve to the leader lines for better aesthetics
    segment.ncp = 3,
    arrow = arrow(length = unit(0.02, "npc")) 
  ) +
  scale_color_manual(values = final_palette) +
  scale_shape_manual(values = origin_shapes) + 
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, by = 20)) +
  scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, by = 20)) +
  labs(
    x = "Non-Scleractinian Ubiquity (%)", 
    y = "Scleractinian Ubiquity (%)", 
    color = "Compound Class",
    shape = "Metabolite Origin"
  ) +
  theme_pubr() +
  theme(legend.position = "none", plot.title = element_text(size = 12, face = "bold"))

p_rf_ubiq_scatter <- ggplot(met_summary_rf, aes(x = non_scler_ubiquity, y = scler_ubiquity)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray60") +
  geom_point(
    aes(color = display_class, shape = refined_origin), 
    size = 4, stroke = 1, alpha = 0.9
  ) +
  geom_text_repel(
    data = subset(met_summary_rf, metabolite %in% target_mets),
    aes(label = metabolite),
    size = 3.5,
    fontface = "bold",
    box.padding = 2.5,
    point.padding = 1.5,
    force = 50,
    force_pull = 0.5,
    min.segment.length = 0,
    segment.color = "grey30",
    segment.curvature = -0.1,
    arrow = arrow(length = unit(0.02, "npc"))
  ) +
  scale_color_manual(values = final_palette) +
  scale_shape_manual(values = origin_shapes) + 
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, by = 20)) +
  scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, by = 20)) +
  labs(
    x = "Non-Scleractinian Ubiquity (%)", 
    y = "Scleractinian Ubiquity (%)", 
    color = "Compound Class",
    shape = "Metabolite Origin"
  ) +
  theme_pubr() +
  theme(legend.position = "none", plot.title = element_text(size = 12, face = "bold"))

print(p_xgb_ubiq_scatter)
print(p_rf_ubiq_scatter)

row_hi <- plot_grid(
  p_xgb_ubiq_scatter + theme(legend.position = "none") + labs(subtitle = "XGBoost"), 
  p_rf_ubiq_scatter 
  + theme(legend.position = "none")
  + labs(subtitle = "Random Forest"), 
  # labels = c("H", "I"), 
  label_size = 18, ncol = 2
)
# row_hi <- plot_grid(
#   unified_legend, row_hi,
#   nrow = 2, rel_heights = c(0.4,1))

ggsave("/work/hs325/World_Corals/misc/figs/fig5_hi.jpg", 
       row_hi, width = 14, height = 6, dpi = 300, bg = "white")

################################################################################
# p1 <- ggbarplot(xgb_plot_data, x = "metabolite", y = "XGBoost_Importance",
#                 fill = "display_class", color = "transparent",
#                 xlab = "Metabolite", ylab = "XGBoost Feature Importance") +
#   theme_pubr() +
#   scale_fill_manual(values = final_palette) +
#   scale_y_continuous(expand = expansion(mult = c(0, 0.3))) + # More top space for callouts
#   scale_x_discrete(expand = expansion(add = c(1, 10))) +     # More right space for labels
#   geom_text_repel(
#     data = subset(xgb_plot_data, metabolite %in% target_mets),
#     aes(label = metabolite),
#     # Push labels up and to the right
#     nudge_x = 20, 
#     nudge_y = 0.01,
#     direction = "both",
#     angle = 0,                # Horizontal text as requested
#     segment.size = 0.5,
#     segment.color = "grey30",
#     segment.curvature = -0.2, # Curved lines look better for long distances
#     box.padding = 2,          # Forces label further from the bar
#     point.padding = 0.5,
#     min.segment.length = 0,   # Always show the line
#     size = 3.5,
#     fontface = "bold"
#   ) +
#   theme(axis.text.x = element_blank(), 
#         axis.ticks.x = element_blank(),
#         legend.position = "none")
# 
# p2 <- ggbarplot(rf_plot_data, x = "metabolite", y = "RandomForest_Importance",
#                 fill = "display_class", color = "transparent",
#                 xlab = "Metabolite", ylab = "RF Feature Importance") +
#   theme_pubr() +
#   scale_fill_manual(values = final_palette) +
#   scale_y_continuous(expand = expansion(mult = c(0, 0.3))) +
#   scale_x_discrete(expand = expansion(add = c(1, 10))) +
#   geom_text_repel(
#     data = subset(rf_plot_data, metabolite %in% target_mets),
#     aes(label = metabolite),
#     nudge_x = 20, 
#     nudge_y = 0.01,
#     direction = "both",
#     angle = 0,
#     segment.size = 0.5,
#     segment.color = "grey30",
#     segment.curvature = -0.2,
#     box.padding = 2,
#     point.padding = 0.5,
#     min.segment.length = 0,
#     size = 3.5,
#     fontface = "bold"
#   ) +
#   theme(axis.text.x = element_blank(), 
#         axis.ticks.x = element_blank(),
#         legend.position = "none")
# 
# print(p1)
# print(p2)
# 
# p1p2 <- plot_grid(p1, p2)
# row_ab <- plot_grid(
#   unified_legend, p1p2,
#   nrow = 2, rel_heights = c(0.4,1))
# ggsave("/work/hs325/World_Corals/misc/figs/fig5_hi.jpg", 
#        p1p2, width = 12, height = 6, dpi = 600, bg = "white")