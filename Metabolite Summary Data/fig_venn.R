library(tidyverse)
library(knitr)
library(readxl)
library(data.table)
library(vegan)
library(scales)
library(cowplot)
library(ggdendro)
library(ggridges)
library(dendextend)
library(RColorBrewer)
library(corrplot)  
library(reshape2)
library(pheatmap)
library(ggpubr)
library(forcats)
library(ggvenn)
library(ggrepel)
library(ggforce)
library(ComplexUpset)
library(UpSetR)
library(here)

# read in data
df <- read.csv(here("Cleaned data CSVs", "qc_data.csv"))
met_df <- read.csv(here("Cleaned data CSVs", "merged_met_plot_df.csv"))

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
cols_bleaching <- c(
  "Bleached" = "#FF847CFF", 
  "Non-Bleached" = "#019875FF", 
  "Not Applicable" = "#D3D3D3")
cols_location <-c("#002594FF", "#E0B2CDFF", "#54C4E3FF", "#F3AA4FFF")
# cols_location  <- c("#449DB3FF", "#A3BAC2FF", "#60BFAEFF", "#8C6E5DFF")
cols_symbiont  <- c("#D84D16FF", "#FFF800FF", "#8FDA04FF")
cols_phylum <- c("#24492EFF", "#015B58FF", "#2C6184FF", "#59629BFF", "#89689DFF", "#BA7999FF", "#E69B99FF")
cols_sclero    <- c("1" = "#DE7862FF", "0" = "#D8AF39FF")

################################################################################

### for custom compound_class
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
class_order <- c(target_classes, "Other")

origin_shapes <- c("Host" = 16, "Symbiont" = 3, "Both" = 17, "Unknown" = 8)
###########################################################


## categories for venns

# flowers

draw_flower <- function(data, group_var) {
  group_summary <- data %>%
    pivot_longer(cols = starts_with("x"), names_to = "metabolite", values_to = "val") %>%
    group_by(!!sym(group_var), metabolite) %>%
    summarise(present = any(val > 0, na.rm = TRUE), .groups = "drop")
  
  # core
  total_groups <- length(unique(data[[group_var]]))
  core_mets <- group_summary %>%
    group_by(metabolite) %>%
    summarise(n_groups = sum(present)) %>%
    filter(n_groups == total_groups) %>%
    pull(metabolite)
  
  core_count <- length(core_mets)
  
  petal_data <- group_summary %>%
    filter(present == TRUE) %>%
    group_by(!!sym(group_var)) %>%
    tally() %>%
    mutate(label = paste0(!!sym(group_var), "\n(n=", n, ")"))
  
  # Calculate petal positions
  n_petals <- nrow(petal_data)
  angle <- seq(0, 2 * pi, length.out = n_petals + 1)[1:n_petals]
  
  petal_data$x <- sin(angle) * 2
  petal_data$y <- cos(angle) * 2
  
  ggplot(petal_data) +
    geom_ellipse(aes(x0 = x, y0 = y, a = 0.8, b = 1.5, angle = -angle), 
                 fill = "lightblue", alpha = 0.3, color = "steelblue") +
    
    # Center Circle
    annotate("point", x = 0, y = 0, size = 45, color = "gold", 
             fill = "white", shape = 21, stroke = 2) +
    annotate("text", x = 0, y = 0, label = paste0(core_count), 
             fontface = "bold", size = 7) +
    
    geom_text(aes(x = x * 3, y = y * 3, label = label), 
              size = 5, fontface = "bold",
              hjust = "middle", vjust = "middle") +
    
    theme_void() +
    coord_cartesian(xlim = c(-8,8), ylim = c(-7, 7)) 
}

df_scler <- df %>% filter(host_order == "Scleractinia", !is.na(host_family))
p_flower_family <- draw_flower(df_scler, "host_family")
ggsave("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/flower_plot_family.jpg", 
       p_flower_family, width = 10, height = 8, dpi = 300)

################################################################################

# richness by host_family 

scler_df <- df %>% filter (df$scleractinia == 1)
scler_df$richness <- rowSums(scler_df %>% select(starts_with("x")) > 0, na.rm = TRUE)
scler_df$host_family <- reorder(scler_df$host_family, scler_df$richness, FUN = mean)

family_palette <- c("#E29191FF", "#99DD92FF", "#93D8B9FF", "#94C4D3FF", "#949ACEFF", 
                    "#B394CCFF", "#CC96B1FF", "#CCA499FF", "#DFE592FF", "#FFA560FF", 
                    "#6BFF63FF", "#65FFCCFF", "#65C4FFFF", "#656BFFFF", "#AD65FFFF", 
                    "#FF65F4FF", "#FF6584FF", "#FF6565FF")

family_counts <- scler_df %>%
  group_by(host_family) %>%
  summarise(n = n(), .groups = "drop")

scler_df <- scler_df %>%
  left_join(family_counts, by = "host_family") %>%
  mutate(family_label = paste0(host_family, " (n=", n, ")"))

scler_df$family_label <- reorder(scler_df$family_label, scler_df$richness, FUN = mean)

p_family_richness <- ggbarplot(
  scler_df, 
  x = "family_label", 
  y = "richness",
  fill = "family_label",      # Match fill to the new label
  color = "black",
  add = "mean_sd",
  error.plot = "pointrange",
  orientation = "horizontal",
  palette = family_palette,   # The 18 colors will map to the new labels
  label = FALSE
) +
  labs(
    y = "Metabolite Richness",
    x = "Host Family"
  ) +
  theme_pubr(base_size = 14) +
  theme(
    legend.position = "none",
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

print(p_family_richness)

# jaccard similarity index correlation plot by family 

family_pa_matrix <- scler_df %>%
  pivot_longer(cols = starts_with("x"), names_to = "metabolite", values_to = "val") %>%
  group_by(host_family, metabolite) %>%
  summarise(present = any(val > 0, na.rm = TRUE), .groups = "drop") %>%
  mutate(present = as.numeric(present)) %>%
  pivot_wider(names_from = metabolite, values_from = present, values_fill = 0) %>%
  column_to_rownames("host_family")

jaccard_dist <- vegdist(family_pa_matrix, method = "jaccard", binary = TRUE)
jaccard_sim_matrix <- as.matrix(1 - jaccard_dist)

plot_matrix <- jaccard_sim_matrix
diag(plot_matrix) <- 0  # Setting to NA or 0 removes the text/color 

# 2. Define a Red Color Palette
# Using a gradient from a very light pink/white to a deep coral red
red_palette <- colorRampPalette(c("#FFF5F0", "#FEE0D2", "#FC9272", "#FB6A4A", "#DE2D26", "#A50F15"))(100)

# 3. Create the Heatmap
p_jaccard <- pheatmap(
  plot_matrix,
  clustering_distance_rows = jaccard_dist, 
  clustering_distance_cols = jaccard_dist,
  clustering_method = "ward.D2",
  color = red_palette,
  display_numbers = TRUE,
  number_format = "%.2f",
  number_color = "white",     
  fontsize_number = 10,       
  fontsize_row = 14,        
  fontsize_col = 14,
  na_col = "white",            
  border_color = "white",
  filename = "/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/family_jaccard_heatmap.png",
  width = 12,                  # Inches
  height = 10                  # Inches
)

################################################################################

## flower plot modified to show number of unique metabolites in each family

draw_flower <- function(data, group_var) {
  group_summary <- data %>%
    pivot_longer(cols = starts_with("x"), names_to = "metabolite", values_to = "val") %>%
    group_by(!!sym(group_var), metabolite) %>%
    summarise(present = any(val > 0, na.rm = TRUE), .groups = "drop")
  
  total_groups <- length(unique(data[[group_var]]))
  metabolite_counts <- group_summary %>%
    group_by(metabolite) %>%
    summarise(n_groups = sum(present), .groups = "drop")
  
  core_count <- sum(metabolite_counts$n_groups == total_groups)
  unique_mets <- metabolite_counts %>% filter(n_groups == 1) %>% pull(metabolite)
  
  petal_data <- group_summary %>%
    filter(present == TRUE) %>%
    group_by(!!sym(group_var)) %>%
    summarise(
      total_n = n(),
      unique_n = sum(metabolite %in% unique_mets),
      .groups = "drop"
    ) %>%
    mutate(label = paste0(!!sym(group_var), "\nTotal: ", total_n, "\nUnique: ", unique_n))
  
  n_petals <- nrow(petal_data)
  angle <- seq(0, 2 * pi, length.out = n_petals + 1)[1:n_petals]
  
  petal_data$x <- sin(angle) * 1.5
  petal_data$y <- cos(angle) * 1.5
  
  ggplot(petal_data) +
    geom_ellipse(aes(x0 = x, y0 = y, a = 0.8, b = 1.5, angle = -angle), 
                 fill = "lightblue", alpha = 0.3, color = "steelblue") +
    
    annotate("point", x = 0, y = 0, size = 45, color = "gold", 
             fill = "white", shape = 21, stroke = 2) +
    annotate("text", x = 0, y = 0, label = paste0(core_count), 
             fontface = "bold", size = 7) +
    
    geom_text(aes(x = sin(angle) * 5, y = cos(angle) * 5, label = label), 
              size = 4, lineheight = 0.9) +
    
    theme_void() +
    coord_cartesian(xlim = c(-7, 7), ylim = c(-5, 5)) 
}
df_scler <- df %>% filter(host_order == "Scleractinia", !is.na(host_family))
p_flower_family <- draw_flower(df_scler, "host_family")
ggsave("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/flower_plot_family_unique.jpg", 
       p_flower_family, width = 13, height = 9, dpi = 300)


################################################################################

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

#######################
# venn by host_origin TBA

core_filtered <- core_df %>% 
  filter(refined_origin != "Unknown")

list_core <- get_venn_list(core_filtered)

venn_fill <- c(cols_origin["Host"], cols_origin["Symbiont"])

# 4. Create the plot
p_core_venn <- ggvenn(
  list_core, 
  fill_color = venn_fill, 
  stroke_size = 0.5, 
  set_name_size = 5,
  text_size = 4,
  show_percentage = TRUE 
) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16))

print(p_core_venn)
# ggsave("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/venn_core_origin.jpg", 
#        p_core_venn, width = 6, height = 5, dpi = 300)

################################################################################
## categories for venns

met_df_filtered <- met_df %>%
  filter(refined_origin != "Unknown")

get_venn_list <- function(data) {
  list(
    Host = data %>% filter(refined_origin %in% c("Host", "Both")) %>% pull(metabolite),
    Symbiont = data %>% filter(refined_origin %in% c("Symbiont", "Both")) %>% pull(metabolite)
  )
}

list_all <- get_venn_list(met_df_filtered)

### plot
cols_origin <- c("Host" = "#97B9CBFF", "Symbiont" = "#9057C6FF",
                 "Both" = "#FFE1BDFF", "Unknown" = "#8DC657FF")
venn_fill <- c(cols_origin["Host"], cols_origin["Symbiont"])

p_a <- ggvenn(list_all, fill_color = venn_fill, stroke_size = 0.5, set_name_size = 4) +
  labs() + theme(plot.title = element_text(hjust = 0.5, face = "bold"))


################################################################################

# volcano
# choose only metabolites in core_df
core_abundance_df <- df %>%
  select(sample, all_of(core_metabolite_ids), scleractinia)

comm_long <- core_abundance_df %>%
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

### volcano plotting data
plot_data_volcano <- volcano_results %>%
  inner_join(
    met_df %>% select(metabolite, compound_class, refined_origin),
    by = "metabolite"
  ) %>%
  mutate(
    compound_class = trimws(as.character(compound_class)),
    refined_origin = as.character(refined_origin),
    display_class = if_else(
      is.na(compound_class) | !(compound_class %in% names(final_palette)),
      "Other",
      compound_class
    ),
    refined_origin = if_else(is.na(refined_origin), "Unknown", refined_origin),
    display_class = factor(display_class, levels = class_order)
  )

classes <- levels(droplevels(plot_data_volcano$display_class))
class_colors <- final_palette[classes]

###########################################################

sig_threshold <- -log10(0.05)

# build plot
p_volcano2 <- ggplot(plot_data_volcano, aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70", linewidth = 0.8) +
  geom_point(aes(color = display_class), alpha = 0.75, size = 3.5) +
  scale_color_manual(
    name = "Compound Class",
    values = class_colors,
    breaks = classes,
    na.value = "gray60"
  ) +
  ylim(0,75) + 
  xlim(-20,20) +
  scale_shape_manual(
    name = "Metabolite Origin",
    values = origin_shapes,
    na.value = 16) +
  facet_wrap(~display_class, ncol = 4) +
  
  guides(
    color = guide_legend(ncol = 2, byrow = TRUE),
    shape = guide_legend(ncol = 1)
  ) +
  labs(
    x = "log2 Fold Change",
    y = "-log10(adj. p-value)",
  ) +
  theme_pubr() +
  theme(
    legend.position = "none",
    strip.text.x = element_text(size = 14),
    axis.title = element_text(size = 20),
    plot.title = element_text(face = "bold", hjust = 0.5)
  )
ggsave("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/volcano_core.jpg", p_volcano2, width=14,height=10,dpi=300)

################################################################################

## make core plot with cowplot
top_right_col <- plot_grid(
  p_core_venn, 
  labels = c("B"), 
  label_size = 30,
  align = "v"
)

top_row <- plot_grid(
  p_flower_family, 
  top_right_col, 
  ncol = 2, 
  labels = c("A", ""),
  label_size = 30,
  rel_widths = c(1.5, 0.5) 
)

final_layout <- plot_grid(
  top_row, 
  p_volcano2, 
  nrow = 2, 
  labels = c("", "C"), 
  label_size = 30,
  rel_heights = c(0.8, 1)
)
ggsave("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/ecological_core.jpg", final_layout, width=20,height=16,dpi=300)


################################################################################

### filters for subsequent analysis

met_df_total_all <- met_df %>%
  filter(total_ubiquity == 100)
# 10 metabolites
# all metabolites inferred_origin = both
# 5 unknown, 2 primary amides, 1 simple phenolic acid, 1 glycerophoshocholine, 1 TQ/THQ

#######################
met_df_scler_all <- met_df %>%
  filter(scler_ubiquity >= 95)
# 494 scler 95% ubq

met_df_non_scler_all <- met_df %>%
  filter(non_scler_ubiquity >= 95)
# 284 non-scler 95% ubq

# 59 scler 100% ubiquity and 146 non-scler 100% ubiquity

make_volcano <- function(selected_df, outpath = NULL) {
  
  selected_metabolites <- intersect(selected_df$metabolite, colnames(df))
  
  abundance_df <- df %>%
    select(sample, all_of(selected_metabolites), scleractinia)
  
  comm_long <- abundance_df %>%
    pivot_longer(
      cols = starts_with("x"),
      names_to = "metabolite",
      values_to = "abundance"
    ) %>%
    mutate(abundance = as.numeric(as.character(abundance)))
  
  stats_data <- comm_long %>%
    select(-scleractinia) %>%
    left_join(df %>% select(sample, scleractinia), by = "sample") %>%
    filter(!is.na(scleractinia)) %>%
    mutate(group = if_else(as.character(scleractinia) == "1", "Scleractinia", "Other"))
  
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
      neg_log_p_adj = -log10(p_adj)
    )
  
  plot_data_volcano <- volcano_results %>%
    inner_join(
      met_df %>% select(metabolite, compound_class, refined_origin),
      by = "metabolite"
    ) %>%
    mutate(
      compound_class = trimws(as.character(compound_class)),
      refined_origin = as.character(refined_origin),
      display_class = if_else(
        is.na(compound_class) | !(compound_class %in% names(final_palette)),
        "Other",
        compound_class
      ),
      refined_origin = if_else(is.na(refined_origin), "Unknown", refined_origin),
      display_class = factor(display_class, levels = class_order)
    )
  
  classes <- levels(droplevels(plot_data_volcano$display_class))
  class_colors <- final_palette[classes]
  sig_threshold <- -log10(0.05)
  
  p <- ggplot(plot_data_volcano, aes(x = log2FC, y = neg_log_p_adj)) +
    geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
    geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70", linewidth = 0.8) +
    geom_point(aes(color = display_class, shape = refined_origin), alpha = 0.75, size = 3.5) +
    scale_color_manual(
      name = "Compound Class",
      values = class_colors,
      breaks = classes,
      na.value = "gray60"
    ) +
    scale_shape_manual(
      name = "Metabolite Origin",
      values = origin_shapes,
      na.value = 16
    ) +
    ylim(0, 75) +
    xlim(-10, 10) +
    labs(
      x = "log2 Fold Change",
      y = "-log10(adj. p-value)"
    ) +
    theme_pubr() +
    theme(
      legend.position = "right",
      axis.title = element_text(size = 20),
      axis.text = element_text(size = 14)
    )
  
  if (!is.null(outpath)) {
    ggsave(outpath, p, width = 14, height = 10, dpi = 300)
  }
  
  return(p)
}

p_volcano_scler_all <- make_volcano(
  met_df_scler_all,
  "/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/volcano_scler_all.jpg"
)

p_volcano_non_scler_all <- make_volcano(
  met_df_non_scler_all,
  "/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/volcano_non_scler_all.jpg"
)

volcano_ubiquitous <- plot_grid(
  p_volcano_scler_all, 
  p_volcano_non_scler_all, 
  nrow = 2, 
  labels = c("A", "B"), 
  label_size = 18,
  rel_heights = c(1, 1)
)
ggsave("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/volcano_ubq.jpg", volcano_ubiquitous, width=12,height=14,dpi=300)

#######################

met_df_scler_none <- met_df %>%
  filter(scler_ubiquity == 0)

met_df_non_scler_none <- met_df %>%
  filter(non_scler_ubiquity == 0)

scler_none_mets <- met_df_scler_none$metabolite
non_scler_none_mets <- met_df_non_scler_none$metabolite

# calculate average abundances
group_abundances <- df %>%
  pivot_longer(
    cols = starts_with("x"), 
    names_to = "metabolite", 
    values_to = "value"
  ) %>%
  mutate(is_scler = ifelse(host_order == "Scleractinia", "scler_avg", "non_scler_avg")) %>%
  group_by(metabolite, is_scler) %>%
  summarise(avg_val = mean(value, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = is_scler, values_from = avg_val)

# update the summary dataframe to include both group-specific averages
met_summary_classed <- met_df %>%
  select(
    metabolite, 
    display_class, 
    compound_class, 
    refined_origin,
    scler_ubiquity, 
    non_scler_ubiquity
  ) %>%
  left_join(group_abundances, by = "metabolite") %>%
  mutate(
    category = case_when(
      metabolite %in% non_scler_none_mets ~ "Coral-only",
      metabolite %in% scler_none_mets     ~ "Non-coral-only",
      TRUE                                ~ "Shared/Other"
    ),
    refined_origin = factor(refined_origin, levels = c("Host", "Symbiont", "Both", "Unknown")),
    display_class = factor(display_class, levels = class_order)
  )

met_summary_classed <- met_summary_classed %>%
  mutate(
    display_class = if_else(is.na(display_class), as.character(compound_class), as.character(display_class)),
    # Re-apply the factor levels (adding "Fatty acyl carnitines" if it wasn't in class_order)
    display_class = factor(display_class, levels = unique(c(class_order, "Fatty acyl carnitines")))
  )

make_ubiquity_plot <- function(selected_df, x_var, y_var, x_label) {
  
  plot_df <- met_summary_classed %>%
    filter(metabolite %in% selected_df$metabolite)
  
  ggplot(plot_df, aes(x = .data[[x_var]], y = .data[[y_var]])) +
    geom_point(
      aes(color = display_class, shape = refined_origin),
      size = 3,
      stroke = 0.8,
      alpha = 0.85
    ) +
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
      x = x_label,
      y = "Average Abundance",
      color = "Compound Class",
      shape = "Metabolite Origin"
    ) +
    theme_pubr() +
    theme(
      legend.position  = "right",
      axis.title       = element_text(size = 14),
      axis.text        = element_text(size = 12)
    )
}

# non-Scleractinia ubiquity of metabolites never found in Scleractinia
p_scler_none <- make_ubiquity_plot(
  met_df_scler_none,
  x_var = "non_scler_ubiquity",
  y_var = "non_scler_avg",
  x_label = "Non-Scleractinian Ubiquity"
)


# opposite - metabolites absent in Scleractinia
p_non_scler_none <- make_ubiquity_plot(
  met_df_non_scler_none,
  x_var = "scler_ubiquity",
  y_var = "scler_avg",
  x_label = "Scleractinian Ubiquity"
)

combined_ubiquity_plot <- plot_grid(
  p_scler_none, 
  p_non_scler_none, 
  labels = c("A", "B"), 
  label_size = 18,
  ncol = 2,
  align = "h",         # Aligns the axes horizontally
  axis = "bt"          # Ensures bottom and top axes stay in line
)

ggsave(
  "/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/combined_ubiquity_plots.jpg", 
  combined_ubiquity_plot, 
  width = 16, 
  height = 8, 
  dpi = 300
)

# non-scler ubiquity abundance of compounds never found in scleractinia
# scler ubiquity abundance of compounds never found in outgroups

################################################################################

## scler ubiquity abundance of compounds shared amongst all families

plot_df_core <- core_df %>%
  left_join(
    met_summary_classed %>% 
      select(metabolite, scler_ubiquity, scler_avg), 
    by = "metabolite"
  )

p_core <- make_ubiquity_plot(
  plot_df_core,
  x_var = "scler_ubiquity",
  y_var = "scler_avg",
  x_label = "Scleractinia Ubiquity (Core Metabolites)"
) 

# ggsave(
#   "/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/ecological_core_ubiquity.jpg", 
#   p_core, 
#   width = 12, 
#   height = 8, 
#   dpi = 300
# )

################################################################################

## scatterplot of met_df$scler_ubiquity vs met_df$non_scler_ubiquity
met_df_scatter <- met_df %>%
  mutate(
    exclusivity = case_when(
      scler_ubiquity == 0 & non_scler_ubiquity > 0 ~ "Outgroups only",
      non_scler_ubiquity == 0 & scler_ubiquity > 0 ~ "Scleractinia only",
      TRUE ~ "Shared"
    ) 
  ) %>%
  filter(display_class != "Unknown")

ps <- ggplot(met_df_scatter, aes(x = scler_ubiquity, y = non_scler_ubiquity)) +
  geom_point(
    aes(color = display_class),
    alpha = 0.8,
    size = 3
  ) +
  scale_color_manual(values = final_palette) +
  scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 20)) +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 20)) +
  labs(
    x = "Scleractinian Ubiquity (%)",
    y = "Non-Scleractinian Ubiquity (%)",
    color = "Compound Class"
  ) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  geom_vline(xintercept = 0, linetype = "dotted", color = "grey70") +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey70") +
  theme_pubr() +
  theme(
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 12),
    legend.position = "right"
  )
ggsave(
  "/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/ubqplot.jpg",
  ps,
  width = 12,
  height = 8,
  dpi = 300
)

################################################################################

## venn diagrams for each combination of metadata variables 

draw_venn_comparison <- function(data, group_var, custom_palette) {
  venn_list <- data %>%
    mutate(!!sym(group_var) := as.character(!!sym(group_var))) %>%
    pivot_longer(cols = starts_with("x"), names_to = "metabolite", values_to = "val") %>%
    filter(val > 0) %>%
    group_by(!!sym(group_var)) %>%
    summarise(mets = list(unique(metabolite)), .groups = "drop") %>%
    # Remove any groups that ended up with 0 metabolites after filtering
    filter(lengths(mets) > 0) %>% 
    deframe()
  plot_colors <- if(!is.null(names(custom_palette))) {
    custom_palette[names(venn_list)]
  } else {
    custom_palette[1:length(venn_list)]
  }
  
  ggvenn(
    venn_list, 
    fill_color = plot_colors,
    stroke_size = 0.5, 
    set_name_size = 5,
    text_size = 4
  ) 
}

df_bleach <- df %>% filter(bleaching != "Not Applicable", !is.na(bleaching)) %>%
  filter(scleractinia == 1)

venn_bleach <- draw_venn_comparison(data = df_bleach, 
                                    group_var = "bleaching", 
                                    custom_palette = cols_bleaching)

ggsave(
  "/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/venntest.jpg",
  venn_bleach,
  width = 12,
  height = 8,
  dpi = 300
)

#scler only 
df_sym <- df %>% filter(scleractinia == 1 & !is.na(symbiont.potential)) 
venn_sym <- draw_venn_comparison(df_sym, "symbiont.potential", cols_symbiont)

# with outgroups
df_sym_2 <- df %>% filter(!is.na(symbiont.potential))
venn_sym2 <- draw_venn_comparison(df_sym_2, "symbiont.potential", cols_symbiont)

venn_sym <- plot_grid(
  venn_sym, venn_sym2,
  labels = c("A", "B"),
  label_size = 20, nrow = 2
)

ggsave(
  "/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/vennsym.jpg",
  venn_sym,
  width = 8,
  height = 12,
  dpi = 300
)

# scleractinian metabolites by location
df_loc <- df %>% filter(!is.na(location)) %>% filter(scleractinia == 1)
venn_loc <- draw_venn_comparison(df_loc, "location", cols_location)

ggsave(
  "/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/vennloc.jpg",
  venn_loc,
  width = 12,
  height = 8,
  dpi = 300
)

################################################################################

# Upset plot of non Scleractinians by phylum

upset_data <- df %>%
  filter(scleractinia == 0, !is.na(host_phylum)) %>%
  pivot_longer(
    cols = starts_with("x"),
    names_to = "metabolite",
    values_to = "val"
  ) %>%
  group_by(host_phylum, metabolite) %>%
  summarise(
    present = as.integer(any(val > 0, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = host_phylum,
    values_from = present,
    values_fill = 0
  )

# Keep only membership columns
upset_mat <- upset_data %>%
  select(-metabolite) %>%
  as.data.frame()

png(
  filename = "/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/upset_phylum.jpg",
  width = 16,
  height = 9,
  units = "in",
  res = 300
)

set_cols <- cols_phylum[colnames(upset_mat)]
set_cols[is.na(set_cols)] <- "grey25"

upset(
  upset_mat,
  nsets = ncol(upset_mat),
  sets = colnames(upset_mat),
  keep.order = TRUE,
  order.by = "freq",
  main.bar.color = "grey25",
  sets.bar.color = set_cols,
  mb.ratio = c(0.7, 0.3),
  text.scale = c(2, 1.8, 1.5, 1.5, 1.5, 1.5)
)

dev.off()
