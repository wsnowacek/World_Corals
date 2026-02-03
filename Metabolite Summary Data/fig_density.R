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
met_df<- read.csv("/work/hs325/World_Corals/Cleaned data CSVs/metabolite_clean.csv")

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

comm_matrix = df |>
  select(c(sample, grep("^x", names(df), value = TRUE))) |>
  column_to_rownames(var = "sample")

comm_matrix = comm_matrix |>
  mutate(across(where(is.numeric), ~ ifelse(.x > 0, 1, 0)))

top_20_classes <- met_df %>%
  filter(!is.na(coraldb_compound_class)) %>%
  group_by(coraldb_compound_class) %>%
  summarise(n = n()) %>%
  arrange(desc(n)) %>%
  slice_head(n = 20) %>%
  mutate(class_label = paste0(coraldb_compound_class, " (n=", n, ")"))
top_20_classes # remove NA

class_abundance <- comm_long %>%
  inner_join(met_df %>% select(metabolite, coraldb_compound_class), by = "metabolite") %>%
  filter(coraldb_compound_class %in% top_20_classes$coraldb_compound_class) %>%
  group_by(sample, coraldb_compound_class) %>%
  summarise(mean_abundance = mean(abundance, na.rm = TRUE), .groups = "drop")

plot_data_ridge <- class_abundance %>%
  left_join(df %>% select(sample, scleractinia), by = "sample") %>%
  left_join(top_20_classes %>% select(coraldb_compound_class, class_label), by = "coraldb_compound_class") %>%
  mutate(scler_label = if_else(scleractinia == "1", "Scleractinia", "Other")) %>%
  filter(!is.na(scler_label)) #

plot_data_ridge <- plot_data_ridge %>%
  mutate(scler_label = factor(scler_label, levels = c("Scleractinia", "Other")))

ridge_cols <- c("Scleractinia" = "#DE7862FF", "Other" = "#D8AF39FF") 

## no averaging
plot_data_ridge_full <- comm_long %>%
  inner_join(met_df %>% select(metabolite, coraldb_compound_class), by = "metabolite") %>%
  filter(coraldb_compound_class %in% top_20_classes$coraldb_compound_class) %>%
  left_join(df %>% select(sample, scleractinia), by = "sample") %>%
  left_join(top_20_classes %>% select(coraldb_compound_class, class_label), by = "coraldb_compound_class") %>%
  mutate(scler_label = factor(if_else(scleractinia == "1", "Scleractinia", "Other"), 
                              levels = c("Scleractinia", "Other"))) %>%
  filter(!is.na(scler_label))

#################################################################################

p_ridge <- ggplot(plot_data_ridge, 
                  aes(x = log10(mean_abundance + 1), 
                      y = reorder(class_label, mean_abundance), 
                      fill = scler_label, 
                      color = scler_label)) +

  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
  
  geom_density_ridges(alpha = 0.6, 
                      scale = 1.2, 
                      rel_min_height = 0.01,
                      bandwidth = 0.1,
                      show.legend = FALSE) + 
  
  scale_fill_manual(values = ridge_cols) +
  scale_color_manual(values = ridge_cols) +
  
  facet_grid(scler_label ~ .) +
  
  labs(
    x = "Metabolite Abundances (log10 Mean + 1)",
    y = "Top 20 Compound Classes"
  ) +
  theme_pubr() + 
  theme(
    axis.text.y = element_text(size = 9),
    axis.title.y = element_blank(),
    strip.text = element_blank(),
    strip.background = element_blank(),
    panel.spacing = unit(1.5, "lines") 
  )

print(p_ridge)
ggsave("/work/hs325/World_Corals/misc/figs/ridgeline_avg.jpg", p_ridge, width = 5, height = 10, dpi=300)


#################################################################################

p_ridge_full <- ggplot(plot_data_ridge_full, 
                       aes(x = log10(abundance + 1), 
                           y = reorder(class_label, abundance, FUN = median), 
                           fill = scler_label, 
                           color = scler_label)) +
  
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
  
    geom_density_ridges(alpha = 0.6, 
                      scale = 1.2, 
                      rel_min_height = 0.01, # lowered slightly to capture rare metabolites
                      bandwidth = 0.1,
                      show.legend = FALSE,
                      trim = TRUE) + 
  
  scale_fill_manual(values = ridge_cols) +
  scale_color_manual(values = ridge_cols) +
  
  facet_grid(scler_label ~ .) +
  
  labs(
    x = "Metabolite Abundances (log10 + 1)",
    y = ""
  ) +
  theme_pubr() + 
  theme(
    axis.text.y = element_text(size = 9),
    strip.text = element_blank(),
    strip.background = element_blank(),
    panel.spacing.y = unit(1, "lines")
  ) +
  coord_cartesian(xlim = c(0, NA))
print(p_ridge_full)
ggsave("/work/hs325/World_Corals/misc/figs/ridgeline.jpg", p_ridge_full, width = 5, height = 10, dpi=300)

#################################################################################

# npc number pathway

top_20_npc <- met_df %>%
  filter(!is.na(npc_number_pathway), 
         !npc_number_pathway %in% c("null", "Unknown", "unclassified")) %>%
  group_by(npc_number_pathway) %>%
  summarise(n = n()) %>%
  arrange(desc(n)) %>%
  slice_head(n = 20) %>%
  mutate(npc_label = paste0(npc_number_pathway, " (n=", n, ")"))

# 2. Build the full distribution dataset (no averaging)
plot_data_npc_full <- comm_long %>%
  inner_join(met_df %>% select(metabolite, npc_number_pathway), by = "metabolite") %>%
  filter(npc_number_pathway %in% top_20_npc$npc_number_pathway) %>%
  # Join with sample metadata
  left_join(df %>% select(sample, scleractinia), by = "sample") %>%
  # Join with our new labels
  left_join(top_20_npc %>% select(npc_number_pathway, npc_label), by = "npc_number_pathway") %>%
  mutate(scler_label = factor(if_else(scleractinia == "1", "Scleractinia", "Other"), 
                              levels = c("Scleractinia", "Other"))) %>%
  filter(!is.na(scler_label))

p_ridge_npc <- ggplot(plot_data_npc_full, 
                      aes(x = log10(abundance + 1), 
                          y = reorder(npc_label, abundance, FUN = median), 
                          fill = scler_label, 
                          color = scler_label)) +
  
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
  
  geom_density_ridges(alpha = 0.6, 
                      scale = 1.2, 
                      rel_min_height = 0.005, 
                      bandwidth = 0.1,
                      show.legend = TRUE) + 
  
  scale_fill_manual(values = ridge_cols) +
  scale_color_manual(values = ridge_cols) +
    coord_cartesian(xlim = c(0, NA)) +
  
  facet_grid(scler_label ~ .) +
  
  labs(
    x = "Metabolite Abundances (log10 Intensity + 1)",
    y = "",
    fill = "Order",
    color = "Order"
  ) +
  theme_pubr() + 
  theme(
    axis.text.y = element_text(size = 9),
    strip.background = element_blank(),
    strip.text = element_blank(),
    panel.spacing.y = unit(0.5, "lines"),
    legend.position = "none"
  )

print(p_ridge_npc)

#################################################################################

top_20_superclass <- met_df %>%
  filter(!is.na(compound_superclass), 
         !compound_superclass %in% c("null", "Unknown", "unclassified", "N/A")) %>%
  group_by(compound_superclass) %>%
  summarise(n = n()) %>%
  arrange(desc(n)) %>%
  slice_head(n = 20) %>%
  mutate(superclass_label = paste0(compound_superclass, " (n=", n, ")"))

plot_data_superclass_full <- comm_long %>%
  inner_join(met_df %>% select(metabolite, compound_superclass), by = "metabolite") %>%
  filter(compound_superclass %in% top_20_superclass$compound_superclass) %>%
  left_join(df %>% select(sample, scleractinia), by = "sample") %>%
  left_join(top_20_superclass %>% select(compound_superclass, superclass_label), by = "compound_superclass") %>%
  mutate(scler_label = factor(if_else(scleractinia == "1", "Scleractinia", "Other"), 
                              levels = c("Scleractinia", "Other"))) %>%
  filter(!is.na(scler_label))

p_ridge_superclass <- ggplot(plot_data_superclass_full, 
                             aes(x = log10(abundance + 1), 
                                 y = reorder(superclass_label, abundance, FUN = median), 
                                 fill = scler_label, 
                                 color = scler_label)) +
  
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
  
  geom_density_ridges(alpha = 0.6, 
                      scale = 1.2, 
                      rel_min_height = 0.005, 
                      bandwidth = 0.1) + 
  
  scale_fill_manual(values = ridge_cols) +
  scale_color_manual(values = ridge_cols) +
  
  coord_cartesian(xlim = c(0, NA)) + # Strictly clips the x-axis at 0
  
  facet_grid(scler_label ~ .) +
  
  labs(
    x = "Metabolite Abundances (log10 Intensity + 1)",
    y = ""
  ) +
  theme_pubr() + 
  theme(
    axis.text.y = element_text(size = 9),
    strip.background = element_blank(),
    strip.text = element_blank(),
    panel.spacing.y = unit(0.5, "lines")
  )
print(p_ridge_superclass)
ggsave("/work/hs325/World_Corals/misc/figs/superclass.jpg", p_ridge_superclass, width=8,height=8)


#################################################################################
### Volcano plot

cols_origin <- c("Host" = "#97B9CBFF", "Symbiont" = "#9057C6FF", 
                 "Both" = "#FFE1BDFF", "Unknown" = "#8DC657FF")

met_df$refined_origin <- factor(met_df$refined_origin, 
                                levels = c("Host", "Symbiont", "Both", "Unknown"))

stats_data <- comm_long %>%
  left_join(df %>% select(sample, scleractinia), by = "sample") %>%
  filter(!is.na(scleractinia)) %>%
  mutate(group = if_else(scleractinia == "1", "Scleractinia", "Other"))

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
  inner_join(met_df %>% select(metabolite, refined_origin), by = "metabolite")

sig_threshold <- -log10(0.05)

p_volcano <- ggplot(plot_data_volcano, aes(x = log2FC, y = neg_log_p_adj, color = refined_origin)) +
  # Vertical lines for 2-fold change
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70", linewidth = 0.8) +
  geom_point(alpha = 0.6, size = 2.5) +
  xlim(-2,2) +
  scale_color_manual(values = cols_origin) +
  labs(
    x = "log2 Fold Change",
    y = "-log10(p-value)",
    color = "Origin",
  ) +
  theme_pubr() +
  theme(
    legend.position = "right",
    plot.title = element_text(face = "bold", hjust = 0.5)
  )
print(p_volcano)

################################################################################

top_row <- plot_grid(
  p_volcano, p_ridge_full, 
  labels = c("A", "B"), 
  label_size = 20,
  label_x=0,
  label_y=1,
  hjust = 0,   
  vjust = 1.5,
  ncol = 2,
  rel_widths = c(1,1)
)
p_ridge_superclass_noleg <- p_ridge_superclass + theme(legend.position = "none")
bottom_row <- plot_grid(
  p_ridge_npc, p_ridge_superclass, 
  labels = c("C", "D"), 
  label_size = 20,
  label_x=0,
  label_y=1,
  hjust = 0,   
  vjust = 1.5,
  ncol = 2,
  rel_widths = c(1,1)
)

p_for_legend <- p_ridge_superclass + 
  theme(legend.position = "bottom", 
        legend.justification = "center",
        legend.direction = "horizontal") +
  labs(fill = "Order:", color = "Order:")

# 2. Extract the legend
shared_legend <- get_legend(p_for_legend)
legend_row <- plot_grid(shared_legend)  
final_plot <- plot_grid(
  top_row, 
  bottom_row, 
  legend_row,
  ncol = 1, 
  rel_heights = c(1, 1, 0.1) 
)
ggsave("/work/hs325/World_Corals/misc/figs/fig4.jpg", final_plot, width=15,height=12,dpi=300)

################################################################################
## TBA: boxplots for specific categories of coraldb compound class?
# triacylglycerols
# MADAG
# DGCC
# MGDG
# DGDG
# LysoPC acyl
# Diacylglycerols
