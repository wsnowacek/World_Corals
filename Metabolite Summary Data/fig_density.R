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
library(ggrepel)
library(forcats)
library(here)
# library(ComplexHeatmap)
# library(circlize)

# read in data
df <- read.csv(here("Cleaned data CSVs", "qc_data.csv"))
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
  xlim(-25,25) +
  ylim(0,100) + 
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
ggsave("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/volcano_origin.jpg", p_volcano, width=15,height=12,dpi=300)
## 110 rows outside

################################################################################

### for ClassyFire compound_superclass
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
# origin_shapes <- c("Host" = 16, "Symbiont" = 3, "Both" = 17, "Unknown" = 8)
# 
# plot_data_volcano <- volcano_results %>%
#   inner_join(
#     met_df %>% select(metabolite, display_class, refined_origin),
#     by = "metabolite"
#   ) %>%
#   mutate(
#     # coerce to character
#     display_class = as.character(display_class),
#     refined_origin = as.character(refined_origin),
#     display_class = if_else(
#       is.na(display_class) | !(display_class %in% names(final_palette)),
#       "Other",
#       display_class
#     ),
#     refined_origin = if_else(is.na(refined_origin), "Unknown", refined_origin)
#   )
# 
# classes <- sort(unique(plot_data_volcano$display_class))
# class_colors <- final_palette[classes]
# class_order <- c(target_classes, "Other")


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

###############################################################

sig_threshold <- -log10(0.05)

# build plot
p_volcano2 <- ggplot(plot_data_volcano, aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70", linewidth = 0.8) +
  geom_point(aes(color = display_class), alpha = 0.75, size = 2.5) +
  scale_color_manual(
    name = "Compound Class",
    values = class_colors,
    breaks = classes,
    na.value = "gray60"
  ) +
  ylim(0,100) + 
  xlim(-25,25) +
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
ggsave("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/volcano.jpg", p_volcano2, width=14,height=10,dpi=300)

#############################################

combined_volcano <- plot_grid(p_volcano, p_volcano2, ncol = 1, labels = c("A", "B"), label_size = 24, align = "hv")
ggsave("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/combined_volcano.jpg", combined_volcano, width=14,height=20,dpi=300)


################################################################################
## subvolcanos 

# for compound superclass: display_class "Glycerolipids" "Sphingolipids" "Triacylglycerols"
# for compound class: "TAG" "DAG" "MADAG"

p_tag <- plot_data_volcano %>%
  filter(display_class == "TAG") %>%
  ggplot(aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70") +
  geom_point(aes(color = display_class), size = 3, alpha = 0.8) +
  xlim(-25,25) +
  ylim(0,75) + 
  # Label specific metabolite
  # geom_text_repel(
  #   data = . %>% filter(metabolite == "x23838_655_56593_11_538"),
  #   aes(label = metabolite),
  #   box.padding = 1, point.padding = 0.5,
  #   size = 4, fontface = "bold", color = "black",
  #   segment.color = "grey30"
  # ) 
  scale_color_manual(values = class_colors) +
  labs(title = "TAG", x = "log2 Fold Change", y = "-log10(adj. p-value)") +
  theme_pubr() +
  theme(legend.position = "none", plot.title = element_text(size = 18, face = "bold"))

p_dag <- plot_data_volcano %>%
  filter(display_class == "DAG") %>%
  ggplot(aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70") +
  geom_point(aes(color = display_class), size = 3, alpha = 0.8) +
  xlim(-25,25) +
  ylim(0,75) + 
  # Label specific metabolite
  # geom_text_repel(
  #   data = . %>% filter(metabolite == "x39055_948_80202_15_826"),
  #   aes(label = metabolite),
  #   box.padding = 1, point.padding = 0.5,
  #   size = 4, fontface = "bold", color = "black",
  #   segment.color = "grey30"
  # ) 
  scale_color_manual(values = class_colors) +
  labs(title = "DAG", x = "log2 Fold Change", y = "-log10(adj. p-value)") +
  theme_pubr() +
  theme(legend.position = "none", plot.title = element_text(size = 18, face = "bold"))

p_madag <- plot_data_volcano %>%
  filter(display_class == "MADAG") %>%
  ggplot(aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70") +
  geom_point(aes(color = display_class), size = 3, alpha = 0.8) +
  xlim(-25,25) +
  ylim(0,75) + 
  # # Label specific metabolite
  # geom_text_repel(
  #   data = . %>% filter(metabolite == "x15256_518_49365_7_407"),
  #   aes(label = metabolite),
  #   box.padding = 1, point.padding = 0.5,
  #   size = 4, fontface = "bold", color = "black",
  #   segment.color = "grey30"
  # ) 
  scale_color_manual(values = class_colors) +
  labs(title = "MADAG", x = "log2 Fold Change", y = "-log10(adj. p-value)") +
  theme_pubr() +
  theme(legend.position = "none", plot.title = element_text(size = 18, face = "bold"))

subcano <- plot_grid(p_tag, p_dag, p_madag, ncol = 3)
ggsave("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/subcano.jpg", subcano, width=14, height=7, dpi=300)

################################################################################

# for compound class: "TQ/THQ" "Neutral GSL" "Ceramide"

TAG <- plot_data_volcano %>%
  filter(display_class == "TQ/THQs") %>%
  ggplot(aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70") +
  geom_point(aes(color = display_class), size = 3, alpha = 0.8) +
  xlim(-25,25) +
  ylim(0,75) + 
  scale_color_manual(values = class_colors) +
  labs(title = "TQ/THQs", x = "log2 Fold Change", y = "-log10(adj. p-value)") +
  theme_pubr() +
  theme(legend.position = "none", plot.title = element_text(size = 18, face = "bold"))

DAG <- plot_data_volcano %>%
  filter(display_class == "Neutral GSL") %>%
  ggplot(aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70") +
  geom_point(aes(color = display_class), size = 3, alpha = 0.8) +
  xlim(-25,25) +
  ylim(0,75) + 
  scale_color_manual(values = class_colors) +
  labs(title = "Neutral GSL", x = "log2 Fold Change", y = "-log10(adj. p-value)") +
  theme_pubr() +
  theme(legend.position = "none", plot.title = element_text(size = 18, face = "bold"))

MADAG <- plot_data_volcano %>%
  filter(display_class == "Ceramides") %>%
  ggplot(aes(x = log2FC, y = neg_log_p_adj)) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "grey70") +
  geom_point(aes(color = display_class), size = 3, alpha = 0.8) +
  xlim(-25,25) +
  ylim(0,75) + 
  scale_color_manual(values = class_colors) +
  labs(title = "Ceramides", x = "log2 Fold Change", y = "-log10(adj. p-value)") +
  theme_pubr() +
  theme(legend.position = "none", plot.title = element_text(size = 18, face = "bold"))
MADAG

subcano2 <- plot_grid(TAG, DAG, MADAG, ncol = 3)
ggsave("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/subcano2.jpg", subcano2, width=14, height=7, dpi=300)


