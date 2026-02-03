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

setwd("/work/hs325/World_Corals/Metabolite Summary Data")
df<- read.csv("qc_data.csv")
met_df<- read.csv("/work/hs325/World_Corals/Cleaned data CSVs/metabolite_clean.csv")

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

target_classes <- trimws(c(
  "Diacylglycerols", "Fatty amides", "Fatty esters", "Glycerolipids", 
  "Glycerophospholipids", "Monoalkyldiacylglycerols", "Phosphatidylglycerocholines", 
  "Sphingolipids", "Steroids", "Triacylglycerols", "Unknown"))

spec_colors <- c("#FFBB78FF", "#D62728FF", 
                 "#9467BDFF", "#8C564BFF", "#E377C2FF", "#BCBD22FF", 
                 "#17BECFFF", "#2CA02CFF", "#FF9896FF", "#98DF8AFF", "#1F77B4FF")

names(spec_colors) <- target_classes
final_palette <- c(spec_colors, "Other" = "#D3D3D3")

process_importance_data <- function(df) {
  df %>%
    mutate(compound_superclass = trimws(as.character(compound_superclass))) %>%
    mutate(display_class = if_else(compound_superclass %in% names(final_palette), 
                                   compound_superclass, 
                                   "Other")) %>%
    mutate(display_class = fct_relevel(factor(display_class), "Other", after = Inf)) 
}

met_plot_df <- process_importance_data(met_df)

ordered_levels <- c(target_classes, "Other")
met_plot_df$display_class <- factor(met_plot_df$display_class, levels = ordered_levels)

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

x_vline_pos <- met_summary %>%
  filter(category == "Coral-only") %>%
  pull(ubiquity_all) %>%
  { if(length(.) == 0) NA_real_ else max(., na.rm = TRUE) }

met_summary <- met_summary %>%
  left_join(
    met_presence_long %>%
      group_by(metabolite) %>%
      summarise(avg_abundance = mean(value, na.rm = TRUE), .groups = "drop"),
    by = "metabolite"
  ) %>%
  mutate(category = ifelse(metabolite %in% coral_only, "Coral-only", "Other"))
met_plot_df$display_class <- factor(met_plot_df$display_class, levels = ordered_levels)

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

pa <- ggplot(met_summary_classed, aes(x = ubiquity_all, y = avg_abundance)) +
  geom_point(
    aes(fill = display_class),
    shape = 21,    # Filled circle with border
    size = 3,      # Uniform size for all points
    stroke = 0.4,  # Thin black outline
    color = "black",
    alpha = 0.85
  ) +
  
  scale_fill_manual(values = final_palette) +
  scale_y_continuous(
    labels = label_number(scale_cut = cut_short_scale())
  ) + scale_x_continuous(
    limits = c(0, 100), 
    breaks = seq(0, 100, by = 20)
  ) +
  labs(
    x = "Ubiquity",
    y = "Average Abundance",
    fill = "Compound Superclass",
  ) + theme_pubr() + 
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
    y = "Average Abundance",
    fill = "Compound Superclass",
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

feature_importance_comparison_all <- read.csv("/work/hs325/World_Corals/machine_learning/all_mets/featureimportanceallmets.csv")

feature_importance_comparison_all <- feature_importance_comparison_all %>%
  dplyr::rename(metabolite = Feature)
merged_df_all <- feature_importance_comparison_all %>%
  inner_join(met_df, by = "metabolite")


process_importance_data <- function(df, importance_col) {
  df %>%
    mutate(compound_superclass = trimws(as.character(compound_superclass))) %>%
    mutate(display_class = if_else(compound_superclass %in% names(final_palette), 
                                   compound_superclass, 
                                   "Other")) %>%
    mutate(display_class = fct_relevel(factor(display_class), "Other", after = Inf)) %>%
    mutate(metabolite = fct_reorder(metabolite, !!sym(importance_col), .desc = TRUE))
}

xgb_plot_df <- process_importance_data(merged_df_all[1:40,], "XGBoost_Importance")
rf_plot_df  <- process_importance_data(merged_df_all[1:120,], "RandomForest_Importance")
plot_df_all <- process_importance_data(merged_df_all, "XGBoost_Importance")

ordered_levels <- c(target_classes, "Other")

xgb_plot_df$display_class <- factor(xgb_plot_df$display_class, levels = ordered_levels)
rf_plot_df$display_class  <- factor(rf_plot_df$display_class,  levels = ordered_levels)
plot_df_all$display_class <- factor(plot_df_all$display_class, levels = ordered_levels)

#################### make CDE plots ###########################

#xgb importance
p1 <- ggbarplot(xgb_plot_df, x = "metabolite", y = "XGBoost_Importance",
                fill = "display_class", color = "transparent",
                xlab = "Metabolite", ylab = "XGBoost Importance") +
  theme_pubr() +
  #  how to remove spaces between or add spaces between axes and plot!
  scale_fill_manual(values = final_palette) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) + 
  scale_x_discrete(expand = expansion(add = c(1, 0.5))) + 
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank(),
        legend.position = "none")

#rf importance
p2 <- ggbarplot(rf_plot_df, x = "metabolite", y = "RandomForest_Importance",
                fill = "display_class", color = "transparent",
                xlab = "Metabolite", ylab = "RF Importance") +
  theme_pubr() +
  scale_fill_manual(values = final_palette) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  scale_x_discrete(expand = expansion(add = c(1, 0.5))) +
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank()
        ,legend.position = "none"
        )
p2

# for p3 - only label top text
ordered_levels <- c(target_classes, "Other")
top <- plot_df_all %>%
  mutate(dist = sqrt(XGBoost_Importance^2 + RandomForest_Importance^2)) %>%
  arrange(desc(dist)) %>%
  slice_head(n = 5) %>%
  pull(metabolite)

active_classes <- ordered_levels[ordered_levels %in% unique(c(
  as.character(xgb_plot_df$display_class), 
  as.character(rf_plot_df$display_class)
))]

origin_shapes <- c("Host" = 16, "Symbiont" = 3, "Both" = 17, "Unknown" = 8)
p3 <- ggscatter(plot_df_all, 
                x = "XGBoost_Importance", 
                y = "RandomForest_Importance",
                color = "display_class", 
                shape = "refined_origin", 
                palette = final_palette,
                label = "metabolite", 
                label.select = top,
                repel = TRUE,                     
                font.label = c(10, "italic"),      
                cor.coeff = TRUE, 
                cor.method = "pearson",
                xlab = "XGBoost Feature Importance", 
                ylab = "RF Feature Importance") +
  scale_shape_manual(values = origin_shapes) +
  theme_pubr() +
  theme(legend.position = "none")

#################################################################################

ubiquity_pal <- c(
  "[0,20]"   = "#440154FF", # Dark Purple
  "(20,40]"  = "#3B528BFF", # Blue
  "(40,60]"  = "#21908CFF", # Teal
  "(60,80]"  = "#5DC863FF", # Green
  "(80,100]" = "#FDE725FF"  # Yellow
)
target_mets <- unique(c(as.character(xgb_plot_df$metabolite), 
                        as.character(rf_plot_df$metabolite)))

# Ubiquity = (count of samples where value > 0) / (total samples) * 100
global_ubiquity <- df %>%
  select(all_of(intersect(names(.), target_mets))) %>%
  summarise(across(everything(), ~ mean(.x > 0, na.rm = TRUE) * 100)) %>%
  pivot_longer(everything(), names_to = "metabolite", values_to = "ubiquity_all")

xgb_binned_df <- xgb_plot_df %>%
  select(-any_of("ubiquity_all")) %>% 
  left_join(global_ubiquity, by = "metabolite") %>%
  bin_ubiquity()

rf_binned_df <- rf_plot_df %>%
  select(-any_of("ubiquity_all")) %>% 
  left_join(global_ubiquity, by = "metabolite") %>%
  bin_ubiquity()

xgb_binned_df <- xgb_binned_df %>%
  mutate(metabolite = fct_reorder(metabolite, XGBoost_Importance, .desc = TRUE))

rf_binned_df <- rf_binned_df %>%
  mutate(metabolite = fct_reorder(metabolite, RandomForest_Importance, .desc = TRUE))

p_xgb_ubiq <- ggplot(xgb_binned_df, aes(x = metabolite, y = XGBoost_Importance, fill = ubiquity_bin)) +
  geom_bar(stat = "identity", color = "black", linewidth = 0.1) +
  scale_fill_manual(values = ubiquity_pal) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  theme_pubr() +
  labs(
    x = "Metabolite",
    y = "XGBoost Importance Score",
    fill = "Ubiquity Percentage"
  ) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    legend.position = "right"
  )

p_rf_ubiq <- ggplot(rf_binned_df, aes(x = metabolite, y = RandomForest_Importance, fill = ubiquity_bin)) +
  geom_bar(stat = "identity", color = "black", linewidth = 0.1) +
  scale_fill_manual(values = ubiquity_pal) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  theme_pubr() +
  labs(
    x = "Metabolite",
    y = "RF Importance",
    fill = "Ubiquity Percentage"
  ) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    legend.position = "right"
  )

# Display separately or as a grid
print(p_xgb_ubiq)
print(p_rf_ubiq)

#################################################################################

## final plot: two ubiquity-abundance plots with only the most important metabolites from 
# XGBoost Importance and RF importance

top_xgb_mets <- xgb_plot_df %>% head(40) %>% pull(metabolite)
top_rf_mets  <- rf_plot_df  %>% head(100) %>% pull(metabolite)
met_summary_xgb <- met_summary_classed_scler %>% filter(metabolite %in% top_xgb_mets)
met_summary_rf  <- met_summary_classed_scler %>% filter(metabolite %in% top_rf_mets)

# Helper function to generate the plot to ensure consistency
plot_important_ubiquity <- function(data) {
  ggplot(data, aes(x = ubiquity_all, y = avg_abundance)) +
    geom_point(
      aes(fill = display_class),
      shape = 21, size = 4, stroke = 0.5, alpha = 0.9, color = "black"
    ) +
    geom_text_repel(
      data = data %>% arrange(desc(ubiquity_all)) %>% head(5),
      aes(label = metabolite),
      size = 3, fontface = "italic", max.overlaps = 15
    ) +
    scale_fill_manual(values = final_palette) +
    scale_y_continuous(labels = label_number(scale_cut = cut_short_scale())) +
    scale_x_continuous(limits = c(60, 100), breaks = seq(60, 100, by = 20)) +
    labs(x = "Scleractinian Ubiquity", 
         y = "Average Abundance", 
         fill = "Superclass") +
    theme_pubr() +
    theme(legend.position = "right", legend.text = element_text(size = 8))
}

p_xgb_ubiq_scatter <- plot_important_ubiquity(met_summary_xgb)
p_rf_ubiq_scatter  <- plot_important_ubiquity(met_summary_rf)
p_rf_ubiq_scatter


#################################################################################
unified_legend <- get_legend(
  p2 + 
    # Use scale_fill_manual because barplots use the 'fill' aesthetic
    scale_fill_manual(values = final_palette, name = "Compound Superclass") +
    scale_shape_manual(values = origin_shapes, name = "Metabolite Origin") +
    theme(legend.position = "top", 
          legend.direction = "horizontal",
          legend.box = "vertical",
          legend.text = element_text(size = 12),
          legend.title = element_text(size = 15)) +
    guides(
      # Match 'fill' here to the scale above
      fill = guide_legend(nrow=2, order = 1, override.aes = list(shape = 22, size = 6))
    )
)


row_ab <- plot_grid(
  pa + theme(legend.position = "none"), 
  pb + theme(legend.position = "none"), 
  labels = c("A", "B"), label_size = 18, ncol = 2
)

# Row 2: XGB and RF Barplots (C, D)
row_cd <- plot_grid(
  p1 + theme(legend.position = "none"), 
  p2 + theme(legend.position = "none"), 
  labels = c("C", "D"), label_size = 18, ncol = 2
)

# Row 3: Scatter Importance (E) - Centered or full width
row_e <- plot_grid(
  p3 + theme(legend.position = "none"), 
  labels = c("E"), label_size = 18, ncol = 1
)

# Row 4: Importance Binned by Ubiquity (F, G)
row_fg <- plot_grid(
  p_xgb_ubiq + theme(legend.position = "none"), 
  p_rf_ubiq, # Keep one ubiquity legend here as it's a different scale/meaning
  labels = c("F", "G"), label_size = 18, ncol = 2, rel_widths = c(1, 1.2)
)

# Row 5: Ubiquity-Abundance Scatters (H, I)
row_hi <- plot_grid(
  p_xgb_ubiq_scatter + theme(legend.position = "none"), 
  p_rf_ubiq_scatter + theme(legend.position = "none"), 
  labels = c("H", "I"), label_size = 18, ncol = 2
)

true_final_figure <- plot_grid(
  unified_legend,
  row_ab,
  row_cd,
  row_e,
  row_fg,
  row_hi,
  ncol = 1,
  rel_heights = c(0.2, 1, 1, 1.2, 1, 1) # Adjust heights based on content density
)

# Save high-resolution for publication
ggsave("/work/hs325/World_Corals/misc/figs/crazyfig5.jpg", 
       true_final_figure, width = 16, height = 24, dpi = 300, bg = "white")