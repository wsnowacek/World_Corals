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
library(ggvenn)
library(ggrepel)
library(ggforce)

setwd("/work/hs325/World_Corals/Metabolite Summary Data")
df<- read.csv("qc_data.csv")
met_df<- read.csv("/work/hs325/World_Corals/Cleaned data CSVs/metabolite_plot_df.csv")

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
# feature_importance_comparison_all <- read.csv("/work/hs325/World_Corals/machine_learning/all_mets/featureimportanceallmets.csv")
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
ggsave("/work/hs325/World_Corals/misc/figs/venn.jpg", 
       venn_grid, width = 15, height = 5, dpi = 300)

################################################################################

# flowers
draw_flower <- function(data, group_var) {
  # Determine presence/absence per group
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
  
  n_petals <- nrow(petal_data)
  angle <- seq(0, 2 * pi, length.out = n_petals + 1)[1:n_petals]
  
  petal_data$x <- sin(angle) * 2
  petal_data$y <- cos(angle) * 2
  
  ggplot(petal_data) +
    # Draw Petals
    geom_ellipse(aes(x0 = x, y0 = y, a = 0.8, b = 1.5, angle = -angle), 
                 fill = "lightblue", alpha = 0.3, color = "steelblue") +
    # Center Circle
    annotate("point", x = 0, y = 0, size = 45, color = "gold", fill = "white", shape = 21, stroke = 2) +
    annotate("text", x = 0, y = 0, label = paste0(core_count), 
             fontface = "bold", size = 7) +
###group labels shift
    geom_text(aes(x = x * 2.9, y = y * 3, label = label), 
              size = 6, fontface = "bold") +
    theme_void() +
    coord_fixed(xlim = c(-6, 6), ylim = c(-6, 6)) 
}

df_scler <- df %>% filter(host_order == "Scleractinia", !is.na(host_family))

p_flower_family <- draw_flower(df_scler, "host_family")
# p_flower_phylum <- draw_flower(df %>% filter(!is.na(host_phylum)), "host_phylum") 
# p_flower_loc    <- draw_flower(df_scler, "location")
# 
# flower_grid <- plot_grid(p_flower_family, p_flower_phylum, p_flower_loc,
#                          ncol = 3, labels = c("A", "B", "C"), 
#                          rel_widths = c(1, 0.6, 0.6), label_size = 20)

ggsave("/work/hs325/World_Corals/misc/figs/flower_plot_family.jpg", 
       p_flower_family, width = 10, height = 10, dpi = 300)

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

# ggsave("/work/hs325/World_Corals/misc/figs/venn_core_origin.jpg", 
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

sig_threshold <- -log10(0.05)

# build plot
p_volcano2 <- ggplot(plot_data_volcano, aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70", linewidth = 0.8) +
  geom_point(aes(color = display_class), alpha = 0.75, size = 3.5) +
  scale_color_manual(
    name = "Compound Superclass",
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
ggsave("/work/hs325/World_Corals/misc/figs/volcano_core.jpg", p_volcano2, width=14,height=10,dpi=300)


############## IN CORE: highlight our 3 metabolites
p_glycerolipids <- plot_data_volcano %>%
  filter(display_class == "Glycerolipids") %>%
  ggplot(aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70") +
  geom_point(aes(color = display_class), size = 3, alpha = 0.8) +
  # Label specific metabolite
  geom_text_repel(
    data = . %>% filter(metabolite == "x23838_655_56593_11_538"),
    aes(label = metabolite),
    box.padding = 1, point.padding = 0.5,
    size = 4, fontface = "bold", color = "black",
    segment.color = "grey30"
  ) +
  scale_color_manual(values = class_colors) +
  labs(title = "Glycerolipids", x = "log2 Fold Change", y = "-log10(adj. p-value)") +
  theme_pubr() +
  theme(legend.position = "none", plot.title = element_text(size = 18, face = "bold"))

p_triacylglycerols <- plot_data_volcano %>%
  filter(display_class == "Triacylglycerols") %>%
  ggplot(aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70") +
  geom_point(aes(color = display_class), size = 3, alpha = 0.8) +
  # Label specific metabolite
  geom_text_repel(
    data = . %>% filter(metabolite == "x39055_948_80202_15_826"),
    aes(label = metabolite),
    box.padding = 1, point.padding = 0.5,
    size = 4, fontface = "bold", color = "black",
    segment.color = "grey30"
  ) +
  scale_color_manual(values = class_colors) +
  labs(title = "Triacylglycerols", x = "log2 Fold Change", y = "-log10(adj. p-value)") +
  theme_pubr() +
  theme(legend.position = "none", plot.title = element_text(size = 18, face = "bold"))

p_sphingolipids <- plot_data_volcano %>%
  filter(display_class == "Sphingolipids") %>%
  ggplot(aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70") +
  geom_point(aes(color = display_class), size = 3, alpha = 0.8) +
  # Label specific metabolite
  geom_text_repel(
    data = . %>% filter(metabolite == "x15256_518_49365_7_407"),
    aes(label = metabolite),
    box.padding = 1, point.padding = 0.5,
    size = 4, fontface = "bold", color = "black",
    segment.color = "grey30"
  ) +
  scale_color_manual(values = class_colors) +
  labs(title = "Sphingolipids", x = "log2 Fold Change", y = "-log10(adj. p-value)") +
  theme_pubr() +
  theme(legend.position = "none", plot.title = element_text(size = 18, face = "bold"))

subcano <- plot_grid(p_glycerolipids, p_triacylglycerols, p_sphingolipids, ncol = 3)
ggsave("/work/hs325/World_Corals/misc/figs/volcano_glycerolipids.jpg", subcano, width=14, height=7, dpi=300)
