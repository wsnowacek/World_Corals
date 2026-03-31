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
library(ggpubr)
library(forcats)
library(ggvenn)
library(ggrepel)
library(ggforce)
library(here)

# read in data
df <- read.csv(here("Cleaned data CSVs", "qc_data.csv"))
met_df <- read.csv(here("Cleaned data CSVs", "merged_met_plot_df.csv"))

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
# cols_location  <- c("#449DB3FF", "#A3BAC2FF", "#60BFAEFF", "#8C6E5DFF")
cols_symbiont  <- c("#D84D16FF", "#FFF800FF", "#8FDA04FF")
cols_phylum <- c("#24492EFF", "#015B58FF", "#2C6184FF", "#59629BFF", "#89689DFF", "#BA7999FF", "#E69B99FF")
cols_sclero    <- c("1" = "#DE7862FF", "0" = "#D8AF39FF")

# 
# feature_importance_comparison_all <- read.csv("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/machine_learning/all_mets/featureimportanceallmets.csv")
# 
# feature_importance_comparison_all <- feature_importance_comparison_all %>%
#   dplyr::rename(metabolite = Feature)
# 
# present_metabolites <- df %>% 
#   select(starts_with("x")) %>% 
#   colnames()
# 
# met_df <- met_df %>%
#   filter(met_df$metabolite %in% present_metabolites)
# 
# met_df <- feature_importance_comparison_all %>%
#   inner_join(met_df, by = "metabolite")

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

# Define the subsets
list_all <- get_venn_list(met_df_filtered)
list_xgb <- get_venn_list(met_df_filtered %>% arrange(desc(XGBoost_Importance)) %>% head(43))
list_rf  <- get_venn_list(met_df_filtered %>% arrange(desc(RandomForest_Importance)) %>% head(1541))

### plot
cols_origin <- c("Host" = "#97B9CBFF", "Symbiont" = "#9057C6FF", 
                 "Both" = "#FFE1BDFF", "Unknown" = "#8DC657FF")
venn_fill <- c(cols_origin["Host"], cols_origin["Symbiont"])

p_a <- ggvenn(list_all, fill_color = venn_fill, stroke_size = 0.5, set_name_size = 4) +
  labs() + theme(plot.title = element_text(hjust = 0.5, face = "bold"))

#xgb 50
p_b <- ggvenn(list_xgb, fill_color = venn_fill, stroke_size = 0.5, set_name_size = 4) +
  labs() + theme(plot.title = element_text(hjust = 0.5, face = "bold"))

#Rf 500
p_c <- ggvenn(list_rf, fill_color = venn_fill, stroke_size = 0.5, set_name_size = 4) +
  labs() + theme(plot.title = element_text(hjust = 0.5, face = "bold"))

venn_grid <- plot_grid(p_a, p_b, p_c, ncol = 3, labels = c("A", "B", "C"), label_size = 20)
print(venn_grid)
ggsave("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/venn.jpg", 
       venn_grid, width = 15, height = 5, dpi = 300)

################################################################################

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
  show_percentage = TRUE # Optional: helpful to see proportion of the core
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

# Define the subsets
list_all <- get_venn_list(met_df_filtered)
list_xgb <- get_venn_list(met_df_filtered %>% arrange(desc(XGBoost_Importance)) %>% head(43))
list_rf  <- get_venn_list(met_df_filtered %>% arrange(desc(RandomForest_Importance)) %>% head(1541))

### plot
cols_origin <- c("Host" = "#97B9CBFF", "Symbiont" = "#9057C6FF", 
                 "Both" = "#FFE1BDFF", "Unknown" = "#8DC657FF")
venn_fill <- c(cols_origin["Host"], cols_origin["Symbiont"])

p_a <- ggvenn(list_all, fill_color = venn_fill, stroke_size = 0.5, set_name_size = 4) +
  labs() + theme(plot.title = element_text(hjust = 0.5, face = "bold"))


#######################
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

###########################################################

#### Compound Superclass
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

provided_hex <- c(
  "#BEAED4", "#FDC086", "#FFFF99", "#386CB0", "#F0027F", "#BF5B17", "#1B9E77",
  "#D95F02", "#7570B3", "#984EA3", "#66A61E", "#E6AB02", "#666666", "#A6CEE3", "#B2DF8A",
  "#FB9A99", "#CBD5E8")
# "#E5D8BD" "#FDDAEC"
spec_colors <- setNames(provided_hex, target_classes)

final_palette <- c(spec_colors, "Other" = "gray60")
origin_shapes <- c("Host" = 16, "Symbiont" = 3, "Both" = 17, "Unknown" = 8)

plot_data_volcano <- volcano_results %>%
  inner_join(
    met_df %>% select(metabolite, display_class, refined_origin),
    by = "metabolite"
  ) %>%
  mutate(
    # coerce to character
    display_class = as.character(display_class),
    refined_origin = as.character(refined_origin),
    display_class = if_else(
      is.na(display_class) | !(display_class %in% names(final_palette)),
      "Other",
      display_class
    ),
    refined_origin = if_else(is.na(refined_origin), "Unknown", refined_origin)
  )

classes <- sort(unique(plot_data_volcano$display_class))
class_colors <- final_palette[classes]
class_order <- c(target_classes, "Other")

plot_data_volcano <- volcano_results %>%
  inner_join(
    met_df %>% select(metabolite, display_class, refined_origin),
    by = "metabolite"
  ) %>%
  mutate(
    display_class = as.character(display_class),
    refined_origin = as.character(refined_origin),
    display_class = if_else(
      is.na(display_class) | !(display_class %in% names(final_palette)),
      "Other",
      display_class
    ),
    display_class = factor(display_class, levels = class_order),
    refined_origin = if_else(is.na(refined_origin), "Unknown", refined_origin)
  )

classes <- levels(plot_data_volcano$display_class)
class_colors <- final_palette[classes]

###########################################################


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

