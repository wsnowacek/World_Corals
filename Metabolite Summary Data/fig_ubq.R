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
library(Polychrome)

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
  "Glycerophospholipids", 
  "Sphingolipids", 
  "Oligopeptides", 
  "Glycerolipids", 
  "Triacylglycerols", 
  "Steroids", 
  "Carotenoids (C40)", 
  "Fatty esters", 
  "Diacylglyceryl-carboxyhydroxymethylcholines", 
  "Triterpenoids", 
  "Fatty amides", 
  "Phosphatidylglycerocholines", 
  "Monogalactosyldiacylglycerol", 
  "Phosphatidylglyceroethanolamines", 
  "Monoalkyldiacylglycerols", 
  "Meroterpenoids",
  "Unknown"
))

# 2. Provided Hex Colors
provided_hex <- c(
  "#BEAED4", "#FDC086", "#FFFF99", "#386CB0", "#F0027F", "#BF5B17", "#1B9E77",
"#D95F02", "#7570B3", "#984EA3", "#66A61E", "#E6AB02", "#666666", "#A6CEE3", "#B2DF8A",
"#FB9A99", "#CBD5E8")
# "#E5D8BD" "#FDDAEC"
spec_colors <- setNames(provided_hex, target_classes)

final_palette <- c(spec_colors, "Other" = "gray60")
ordered_levels <- c(target_classes, "Other")

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
    y = "Average Abundance",
    color = "Compound Superclass", # Updated label
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

importance_scores <- merged_df_all %>%
  select(metabolite, XGBoost_Importance, RandomForest_Importance)

met_plot_df <- met_plot_df %>%
  left_join(importance_scores, by = "metabolite") %>%
  mutate(
    XGBoost_Importance = replace_na(XGBoost_Importance, 0),
    RandomForest_Importance = replace_na(RandomForest_Importance, 0)
  )

ordered_levels <- c(target_classes, "Other")
met_plot_df$display_class <- factor(met_plot_df$display_class, levels = ordered_levels)

## change # of metabolites to plot here
xgb_plot_data <- met_plot_df %>%
  arrange(desc(XGBoost_Importance)) %>%
  slice_head(n = 60) %>%
  mutate(metabolite = fct_reorder(metabolite, XGBoost_Importance, .desc = TRUE))

rf_plot_data <- met_plot_df %>%
  arrange(desc(RandomForest_Importance)) %>%
  slice_head(n = 120) %>%
  mutate(metabolite = fct_reorder(metabolite, RandomForest_Importance, .desc = TRUE))

xgb_plot_df$display_class <- factor(xgb_plot_df$display_class, levels = ordered_levels)
rf_plot_df$display_class  <- factor(rf_plot_df$display_class,  levels = ordered_levels)
plot_df_all$display_class <- factor(plot_df_all$display_class, levels = ordered_levels)

#################### make CDE plots ###########################
p1 <- ggbarplot(xgb_plot_data, x = "metabolite", y = "XGBoost_Importance",
                fill = "display_class", color = "transparent",
                xlab = "Metabolite", ylab = "XGBoost Importance") +
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
                xlab = "Metabolite", ylab = "RF Importance") +
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

p3 <- ggscatter(met_plot_df, 
                x = "XGBoost_Importance", 
                y = "RandomForest_Importance",
                color = "display_class", 
                shape = "refined_origin", 
                palette = final_palette,
                label = "metabolite", 
                label.select = top_labels,
                repel = TRUE,                      
                font.label = c(10, "italic"),      
                cor.coeff = TRUE, 
                cor.method = "pearson",
                xlab = "XGBoost Feature Importance", 
                ylab = "RF Feature Importance") +
  scale_shape_manual(values = origin_shapes) +
  theme_pubr() + 
  theme(legend.position = "right") +
  guides(
    color = "none",
    fill = "none",
    shape = guide_legend(title = "Metabolite Origin", override.aes = list(size = 4))
  )
p3
#################################################################################

## add a column to the met_plot_df that has the ubiquity 
# of the metabolite in Scleractinia in the entire dataset
# and of the metabolite in non-Scleractinia in the entire dataset

target_mets <- unique(as.character(met_plot_df$metabolite))

scler_ubiquity_df <- df %>%
  filter(scleractinia == 1) %>%
  select(all_of(intersect(names(.), target_mets))) %>%
  summarise(across(everything(), ~ mean(.x > 0, na.rm = TRUE) * 100)) %>%
  # Reshape for joining
  pivot_longer(everything(), names_to = "metabolite", values_to = "scler_ubiquity")

met_plot_df <- met_plot_df %>%
  left_join(scler_ubiquity_df, by = "metabolite") %>%
  # Handle any metabolites that might have 0 presence in the Scleractinian subset
  mutate(scler_ubiquity = replace_na(scler_ubiquity, 0))

non_scler_ubiquity_df <- df %>%
  filter(scleractinia != 1) %>%
  select(all_of(intersect(names(.), target_mets))) %>%
  summarise(across(everything(), ~ mean(.x > 0, na.rm = TRUE) * 100)) %>%
  # Reshape for joining
  pivot_longer(everything(), names_to = "metabolite", values_to = "non_scler_ubiquity")

met_plot_df <- met_plot_df %>%
  left_join(non_scler_ubiquity_df, by = "metabolite") %>%
  # Handle any metabolites that might have 0 presence in the Scleractinian subset
  mutate(non_scler_ubiquity = replace_na(non_scler_ubiquity, 0))

met_plot_df %>%
  select(metabolite, scler_ubiquity, non_scler_ubiquity) %>%
  arrange(desc(scler_ubiquity)) %>%
  head(10)

#################################################################################

ubiquity_pal <- c(
  "[0,20]"   = "#440154FF", # Dark Purple
  "(20,40]"  = "#3B528BFF", # Blue
  "(40,60]"  = "#21908CFF", # Teal
  "(60,80]"  = "#5DC863FF", # Green
  "(80,100]" = "#FDE725FF"  # Yellow
)

# Helper to prepare data for ubiquity-colored barplots
prep_ubiq_plot <- function(df, importance_col, n_slice) {
  df %>%
    arrange(desc(!!sym(importance_col))) %>%
    slice_head(n = n_slice) %>%
    mutate(
      ubiquity_bin = cut(scler_ubiquity, 
                         breaks = c(-Inf, 20, 40, 60, 80, 100), 
                         labels = names(ubiquity_pal), 
                         include.lowest = TRUE),
      metabolite = fct_reorder(metabolite, !!sym(importance_col), .desc = TRUE)
    )
}

xgb_ubiq_data <- prep_ubiq_plot(met_plot_df, "XGBoost_Importance", 60)
rf_ubiq_data  <- prep_ubiq_plot(met_plot_df, "RandomForest_Importance", 120)

p_xgb_ubiq <- ggplot(xgb_ubiq_data, aes(x = metabolite, y = XGBoost_Importance, fill = ubiquity_bin)) +
  geom_bar(stat = "identity", color = "black", linewidth = 0.1) +
  scale_fill_manual(values = ubiquity_pal, drop = FALSE) +
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

# --- Random Forest Importance colored by Scleractinian Ubiquity ---
p_rf_ubiq <- ggplot(rf_ubiq_data, aes(x = metabolite, y = RandomForest_Importance, fill = ubiquity_bin)) +
  geom_bar(stat = "identity", color = "black", linewidth = 0.1) +
  scale_fill_manual(values = ubiquity_pal, drop = FALSE) +
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

print(p_xgb_ubiq)
print(p_rf_ubiq)

#################################################################################

## final plot: two ubiquity-abundance plots with only the most important metabolites from 
# XGBoost Importance and RF importance

top_xgb_mets <- met_plot_df %>% arrange(desc(XGBoost_Importance)) %>% head(60) %>% pull(metabolite)
top_rf_mets  <- met_plot_df %>% arrange(desc(RandomForest_Importance)) %>% head(120) %>% pull(metabolite)

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
    color = "Compound Superclass",
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
    color = "Compound Superclass",
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

# 2. Re-run the dummy plot with a more flexible guide
legend_dummy <- ggplot(met_plot_df, aes(x = XGBoost_Importance, y = RandomForest_Importance)) +
  geom_point(aes(fill = display_class, shape = refined_origin)) +
  scale_fill_manual(
    values = final_palette, 
    name = "Compound Superclass", 
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
    legend.text = element_text(size = 9, lineheight = 0.8), # lineheight handles wrapped text
    legend.title = element_text(size = 12, face = "bold"),
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
  rel_heights = c(0.4, 1, 1, 1.2, 1, 1) # Adjust heights based on content density
)

# Save high-resolution for publication
ggsave("/work/hs325/World_Corals/misc/figs/crazyfig5.jpg", 
       true_final_figure, width = 16, height = 24, dpi = 300, bg = "white")