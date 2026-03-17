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
library(ggrepel)
library(forcats)
# library(ComplexHeatmap)
# library(circlize)

setwd("/work/hs325/World_Corals/Metabolite Summary Data")
df<- read.csv("qc_data.csv")
met_df<- read.csv("/work/hs325/World_Corals/Cleaned data CSVs/merged_met_plot_df.csv")

present_metabolites <- df %>% 
  select(starts_with("x")) %>% 
  colnames()

met_df <- met_df %>%
  filter(met_df$metabolite %in% present_metabolites)

unique(met_df$compound_superclass) #58
unique(met_df$npc_number_pathway) #8
unique(met_df$npc_number_superclass) #42
unique(met_df$npc_number_class) #83
unique(met_df$coraldb_compound_superclass) #17
unique(met_df$coraldb_compound_class) #34
unique(met_df$coraldb_compound_family) #10
unique(met_df$gnps_compound_class) #13
unique(met_df$classy_fire_number_most_specific_class) #231

class_counts <- met_df %>%
  group_by(coraldb_compound_class) %>%
  summarise(n = n()) %>%
  arrange(desc(n))
print(class_counts, n = 50)

## met_df_npc_number_pathway

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


#################################################################################
## panel A) ridgeline Scleractinia by coral compound class (or NPC number pathway)
# 
# comm_matrix = df |>
#   select(c(sample, grep("^x", names(df), value = TRUE))) |>
#   column_to_rownames(var = "sample")
# 
# comm_matrix = comm_matrix |>
#   mutate(across(where(is.numeric), ~ ifelse(.x > 0, 1, 0)))
# 
# top_20_classes <- met_df %>%
#   filter(!is.na(coraldb_compound_class)) %>%
#   group_by(coraldb_compound_class) %>%
#   summarise(n = n()) %>%
#   arrange(desc(n)) %>%
#   slice_head(n = 20) %>%
#   mutate(class_label = paste0(coraldb_compound_class, " (n=", n, ")"))
# top_20_classes # remove NA
# 
# comm_long <- as.data.frame(comm_matrix) %>%
#   mutate(sample = rownames(.)) %>%
#   pivot_longer(-sample, names_to = "metabolite", values_to = "abundance")
# 
# class_abundance <- comm_long %>%
#   inner_join(met_df %>% select(metabolite, coraldb_compound_class), by = "metabolite") %>%
#   filter(coraldb_compound_class %in% top_20_classes$coraldb_compound_class) %>%
#   group_by(sample, coraldb_compound_class) %>%
#   summarise(mean_abundance = mean(abundance, na.rm = TRUE), .groups = "drop")
# 
# plot_data_ridge <- class_abundance %>%
#   left_join(df %>% select(sample, scleractinia), by = "sample") %>%
#   left_join(top_20_classes %>% select(coraldb_compound_class, class_label), by = "coraldb_compound_class") %>%
#   mutate(scler_label = if_else(scleractinia == "1", "Scleractinia", "Other")) %>%
#   filter(!is.na(scler_label)) #
# 
# plot_data_ridge <- plot_data_ridge %>%
#   mutate(scler_label = factor(scler_label, levels = c("Scleractinia", "Other")))
# 
# ridge_cols <- c("Scleractinia" = "#DE7862FF", "Other" = "#D8AF39FF") 
# 
# ## no averaging
# plot_data_ridge_full <- comm_long %>%
#   inner_join(met_df %>% select(metabolite, coraldb_compound_class), by = "metabolite") %>%
#   filter(coraldb_compound_class %in% top_20_classes$coraldb_compound_class) %>%
#   left_join(df %>% select(sample, scleractinia), by = "sample") %>%
#   left_join(top_20_classes %>% select(coraldb_compound_class, class_label), by = "coraldb_compound_class") %>%
#   mutate(scler_label = factor(if_else(scleractinia == "1", "Scleractinia", "Other"), 
#                               levels = c("Scleractinia", "Other"))) %>%
#   filter(!is.na(scler_label))
# 
# ############################
# 
# p_ridge <- ggplot(plot_data_ridge, 
#                   aes(x = log10(mean_abundance + 1), 
#                       y = reorder(class_label, mean_abundance), 
#                       fill = scler_label, 
#                       color = scler_label)) +
# 
#   geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
#   
#   geom_density_ridges(alpha = 0.6, 
#                       scale = 1.2, 
#                       rel_min_height = 0.01,
#                       bandwidth = 0.1,
#                       show.legend = FALSE) + 
#   
#   scale_fill_manual(values = ridge_cols) +
#   scale_color_manual(values = ridge_cols) +
#   
#   facet_grid(scler_label ~ .) +
#   
#   labs(
#     x = "Metabolite Abundances (log10 Mean + 1)",
#     y = "Top 20 Compound Classes"
#   ) +
#   theme_pubr() + 
#   theme(
#     axis.text.y = element_text(size = 9),
#     axis.title.y = element_blank(),
#     strip.text = element_blank(),
#     strip.background = element_blank(),
#     panel.spacing = unit(1.5, "lines") 
#   )
# 
# print(p_ridge)
# ggsave("/work/hs325/World_Corals/misc/figs/ridgeline_avg.jpg", p_ridge, width = 5, height = 10, dpi=300)
# 
# 
# #################################################################################
# 
# p_ridge_full <- ggplot(plot_data_ridge_full, 
#                        aes(x = log10(abundance + 1), 
#                            y = reorder(class_label, abundance, FUN = median), 
#                            fill = scler_label, 
#                            color = scler_label)) +
#   
#   geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
#   
#     geom_density_ridges(alpha = 0.6, 
#                       scale = 1.2, 
#                       rel_min_height = 0.01, # lowered slightly to capture rare metabolites
#                       bandwidth = 0.1,
#                       show.legend = FALSE) + 
#   
#   scale_fill_manual(values = ridge_cols) +
#   scale_color_manual(values = ridge_cols) +
#   
#   facet_grid(scler_label ~ .) +
#   
#   labs(
#     x = "Metabolite Abundances (log10 + 1)",
#     y = ""
#   ) +
#   theme_pubr() + 
#   theme(
#     axis.text.y = element_text(size = 9),
#     strip.text = element_blank(),
#     strip.background = element_blank(),
#     panel.spacing.y = unit(1, "lines")
#   ) +
#   coord_cartesian(xlim = c(0, NA))
# print(p_ridge_full)
# ggsave("/work/hs325/World_Corals/misc/figs/ridgeline.jpg", p_ridge_full, width = 5, height = 10, dpi=300)

#################################################################################

# npc number pathway
# 
# top_20_npc <- met_df %>%
#   filter(!is.na(npc_number_pathway), 
#          !npc_number_pathway %in% c("null", "Unknown", "unclassified")) %>%
#   group_by(npc_number_pathway) %>%
#   summarise(n = n()) %>%
#   arrange(desc(n)) %>%
#   slice_head(n = 20) %>%
#   mutate(npc_label = paste0(npc_number_pathway, " (n=", n, ")"))
# 
# # 2. Build the full distribution dataset (no averaging)
# plot_data_npc_full <- comm_long %>%
#   inner_join(met_df %>% select(metabolite, npc_number_pathway), by = "metabolite") %>%
#   filter(npc_number_pathway %in% top_20_npc$npc_number_pathway) %>%
#   # Join with sample metadata
#   left_join(df %>% select(sample, scleractinia), by = "sample") %>%
#   # Join with our new labels
#   left_join(top_20_npc %>% select(npc_number_pathway, npc_label), by = "npc_number_pathway") %>%
#   mutate(scler_label = factor(if_else(scleractinia == "1", "Scleractinia", "Other"), 
#                               levels = c("Scleractinia", "Other"))) %>%
#   filter(!is.na(scler_label))
# 
# p_ridge_npc <- ggplot(plot_data_npc_full, 
#                       aes(x = log10(abundance + 1), 
#                           y = reorder(npc_label, abundance, FUN = median), 
#                           fill = scler_label, 
#                           color = scler_label)) +
#   
#   geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
#   
#   geom_density_ridges(alpha = 0.6, 
#                       scale = 1.2, 
#                       rel_min_height = 0.005, 
#                       bandwidth = 0.1,
#                       show.legend = TRUE) + 
#   
#   scale_fill_manual(values = ridge_cols) +
#   scale_color_manual(values = ridge_cols) +
#     coord_cartesian(xlim = c(0, NA)) +
#   
#   facet_grid(scler_label ~ .) +
#   
#   labs(
#     x = "Metabolite Abundances (log10 Intensity + 1)",
#     y = "",
#     fill = "Order",
#     color = "Order"
#   ) +
#   theme_pubr() + 
#   theme(
#     axis.text.y = element_text(size = 9),
#     strip.background = element_blank(),
#     strip.text = element_blank(),
#     panel.spacing.y = unit(0.5, "lines"),
#     legend.position = "none"
#   )
# 
# print(p_ridge_npc)

#################################################################################

# top_20_superclass <- met_df %>%
#   filter(!is.na(compound_superclass), 
#          !compound_superclass %in% c("null", "Unknown", "unclassified", "N/A")) %>%
#   group_by(compound_superclass) %>%
#   summarise(n = n()) %>%
#   arrange(desc(n)) %>%
#   slice_head(n = 20) %>%
#   mutate(superclass_label = paste0(compound_superclass, " (n=", n, ")"))
# 
# plot_data_superclass_full <- comm_long %>%
#   inner_join(met_df %>% select(metabolite, compound_superclass), by = "metabolite") %>%
#   filter(compound_superclass %in% top_20_superclass$compound_superclass) %>%
#   left_join(df %>% select(sample, scleractinia), by = "sample") %>%
#   left_join(top_20_superclass %>% select(compound_superclass, superclass_label), by = "compound_superclass") %>%
#   mutate(scler_label = factor(if_else(scleractinia == "1", "Scleractinia", "Other"), 
#                               levels = c("Scleractinia", "Other"))) %>%
#   filter(!is.na(scler_label))
# 
# p_ridge_superclass <- ggplot(plot_data_superclass_full, 
#                              aes(x = log10(abundance + 1), 
#                                  y = reorder(superclass_label, abundance, FUN = median), 
#                                  fill = scler_label, 
#                                  color = scler_label)) +
#   
#   geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
#   
#   geom_density_ridges(alpha = 0.6, 
#                       scale = 1.2, 
#                       rel_min_height = 0.005, 
#                       bandwidth = 0.1) + 
#   
#   scale_fill_manual(values = ridge_cols) +
#   scale_color_manual(values = ridge_cols) +
#   
#   coord_cartesian(xlim = c(0, NA)) + # Strictly clips the x-axis at 0
#   
#   facet_grid(scler_label ~ .) +
#   
#   labs(
#     x = "Metabolite Abundances (log10 Intensity + 1)",
#     y = ""
#   ) +
#   theme_pubr() + 
#   theme(
#     axis.text.y = element_text(size = 9),
#     strip.background = element_blank(),
#     strip.text = element_blank(),
#     panel.spacing.y = unit(0.5, "lines")
#   )
# print(p_ridge_superclass)
# ggsave("/work/hs325/World_Corals/misc/figs/superclass.jpg", p_ridge_superclass, width=8,height=8)


#################################################################################
### Volcano plot

cols_origin <- c("Host" = "#97B9CBFF", "Symbiont" = "#9057C6FF", 
                 "Both" = "#FFE1BDFF", "Unknown" = "#8DC657FF")

met_df$refined_origin <- factor(met_df$refined_origin, 
                                levels = c("Host", "Symbiont", "Both", "Unknown"))

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

########################################
# just refined origin color

plot_data_volcano <- volcano_results %>%
  inner_join(met_df %>% select(metabolite, refined_origin), by = "metabolite")

m <- nrow(volcano_results)   
sig_threshold <- -log10(0.05 / m)

p_volcano <- ggplot(plot_data_volcano, aes(x = log2FC, y = neg_log_p_adj, color = refined_origin)) +
  # Vertical lines for 2-fold change
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70", linewidth = 0.8) +
  geom_point(alpha = 0.6, size = 2.5) +
  facet_wrap(~refined_origin, ncol = 2) +
  xlim(-20,20) +
  ylim(0,75) + 
  scale_color_manual(values = cols_origin) +
  labs(
    x = "log2 Fold Change",
    y = "-log10(adj. p-value)",
    color = "Metabolite Origin",
  ) +
  theme_pubr() +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", hjust = 0.5),
    strip.text = element_text(size = 16),
    # Increase Axis Title Text (labels)
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 14)
  )
print(p_volcano)
ggsave("/work/hs325/World_Corals/misc/figs/volcano_origin.jpg", p_volcano, width=15,height=12,dpi=300)
## 110 rows outside

################################################################################

## color the points using this met_df's "display_class" and shape by origin
met_df<- read.csv("/work/hs325/World_Corals/Cleaned data CSVs/merged_met_plot_df.csv")

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
  geom_point(aes(color = display_class), alpha = 0.75, size = 2.5) +
  scale_color_manual(
    name = "Compound Superclass",
    values = class_colors,
    breaks = classes,
    na.value = "gray60"
  ) +
  ylim(0,75) + 
  xlim(-20,20) +
  # scale_shape_manual(
  #   name = "Metabolite Origin",
  #   values = origin_shapes,
  #   na.value = 16) +
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
ggsave("/work/hs325/World_Corals/misc/figs/volcano.jpg", p_volcano2, width=14,height=10,dpi=300)

#############################################

combined_volcano <- plot_grid(p_volcano, p_volcano2, ncol = 1, labels = c("A", "B"), label_size = 24, align = "hv")
ggsave("/work/hs325/World_Corals/misc/figs/combined_volcano.jpg", combined_volcano, width=14,height=20,dpi=300)


################################################################################
## subvolcanos 

# 1. Glycerolipids Plot
p_glycerolipids <- plot_data_volcano %>%
  filter(display_class == "Glycerolipids") %>%
  ggplot(aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70") +
  geom_point(aes(color = display_class), size = 3, alpha = 0.8) +
  # Label specific metabolite
  # geom_text_repel(
  #   data = . %>% filter(metabolite == "x23838_655_56593_11_538"),
  #   aes(label = metabolite),
  #   box.padding = 1, point.padding = 0.5,
  #   size = 4, fontface = "bold", color = "black",
  #   segment.color = "grey30"
  # ) 
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
  # geom_text_repel(
  #   data = . %>% filter(metabolite == "x39055_948_80202_15_826"),
  #   aes(label = metabolite),
  #   box.padding = 1, point.padding = 0.5,
  #   size = 4, fontface = "bold", color = "black",
  #   segment.color = "grey30"
  # ) 
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
  # # Label specific metabolite
  # geom_text_repel(
  #   data = . %>% filter(metabolite == "x15256_518_49365_7_407"),
  #   aes(label = metabolite),
  #   box.padding = 1, point.padding = 0.5,
  #   size = 4, fontface = "bold", color = "black",
  #   segment.color = "grey30"
  # ) 
  scale_color_manual(values = class_colors) +
  labs(title = "Sphingolipids", x = "log2 Fold Change", y = "-log10(adj. p-value)") +
  theme_pubr() +
  theme(legend.position = "none", plot.title = element_text(size = 18, face = "bold"))

subcano <- plot_grid(p_glycerolipids, p_triacylglycerols, p_sphingolipids, ncol = 3)
ggsave("/work/hs325/World_Corals/misc/figs/subcano.jpg", subcano, width=14, height=7, dpi=300)

################################################################################

################################################################################

## TBA: boxplots for specific categories of coraldb compound class?
# triacylglycerols
# MADAG
# DGCC
# MGDG
# DGDG
# LysoPC acyl
# Diacylglycerols
