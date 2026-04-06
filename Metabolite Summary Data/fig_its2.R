library(tidyverse)
library(ggplot2)
library(forcats)
library(tidyr)
library(dplyr)
library(stringr)
library(knitr)
library(readxl)
library(data.table)
library(vegan)
library(scales)
library(UpSetR)
library(ComplexUpset)
library(rstatix)
library(ggpubr)

# read data
df <- read.csv(here("Cleaned data CSVs", "ITS2full.csv"))

met_df <- read.csv(here("Cleaned data CSVs", "merged_met_plot_df.csv"))


# define color palettes
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

origin_shapes <- c("Host" = 16, "Symbiont" = 3, "Both" = 17, "Unknown" = 8)

######### define factor levels for its2

# recode ITS2.letter
# A = Symbiodinaceae
# B = Breviolum
# C = Cladocopium
# D = Durusdinium
# Keep mix and no seq the same

df <- df %>%
  mutate(
    ITS2.Letter = case_when(
      is.na(ITS2.Letter)           ~ NA_character_,
      str_detect(ITS2.Letter, ":") ~ "Mix",
      ITS2.Letter == "A" ~ "Symbiodinium",
      ITS2.Letter == "B" ~ "Breviolum",
      ITS2.Letter == "C" ~ "Cladocopium",
      ITS2.Letter == "D" ~ "Durusdinium",
      TRUE ~ ITS2.Letter   # keeps "Mix", "NoSeq", etc.
    )
  )

its2_palette <- c(
  "Breviolum"        = "#FF0000FF",
  "Cladocopium"      = "#00A08AFF",
  "Durusdinium"      = "#F2AD00FF",
  "Symbiodinium"  = "#5BBCD6FF",
  "Mix"              = "#F98400FF"
)

its2_levels <- c(
  "Symbiodinium",
  "Breviolum",
  "Cladocopium",
  "Durusdinium",
  "Mix"
)
################################################################################

its2_barplot <- df %>%
  filter(!is.na(ITS2.Letter), ITS2.Letter != "") %>%
  count(ITS2.Letter) %>%
  mutate(
    ITS2.Letter = factor(ITS2.Letter, 
                         levels = c("No Seq","Mix","Durusdinium","Cladocopium","Breviolum","Symbiodinium"))
  ) 

bar1 <- ggplot(its2_barplot, aes(x = ITS2.Letter, y = n, fill = ITS2.Letter)) +
  geom_col() +
  geom_text(aes(label = paste0("n=", n)), hjust = -0.2) +
  scale_fill_manual(values = its2_palette, breaks = its2_levels) +
  coord_flip(clip = "off") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(
    x = "Symbiont Genus",
  ) +
  theme_pubr()+
  ylim(0,150)+
  theme(legend.position = "none",
        axis.title.x = element_blank())
bar1

################################################################################

bar2_df <- df %>%
  filter(
    !is.na(host_family), host_family != "",
    !is.na(ITS2.Letter), ITS2.Letter != "", 
    ITS2.Letter != "No Seq"
  ) %>%
  count(host_family = host_family, ITS2.Letter, name = "n") %>%
  group_by(host_family) %>%
  mutate(
    total_n = sum(n),
    prop = n / total_n
  ) %>%
  ungroup() %>%
  mutate(
    host_family_label = paste0(host_family, " (n=", total_n, ")"),
    host_family_label = fct_reorder(host_family_label, total_n, .desc = TRUE)
  )

bar2 <- ggplot(bar2_df, aes(x = host_family_label, y = prop, fill = ITS2.Letter)) +
  geom_col() +
  scale_fill_manual(values = its2_palette, breaks = its2_levels) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    x = "Scleractinian Family",
    y = "Proportion of Samples",
    fill = "Symbiont Genus"
  ) + coord_flip() +
  theme_pubr() +
  theme(axis.text.y = element_text(size = 10))
bar2

################################################################################

bar3_df <- df %>%
  filter(
    !is.na(ITS2.Letter), ITS2.Letter != "",
    ITS2.Letter != "No Seq",
    !is.na(bleaching), bleaching != "Not Applicable"
  ) %>%
  count(bleaching, ITS2.Letter, name = "n") %>%
  group_by(bleaching) %>%
  mutate(
    total_n = sum(n),
    prop = n / total_n
  ) %>%
  ungroup() %>%
  mutate(
    bleaching_label = paste0(bleaching, " (n=", total_n, ")")
  )

bar3 <- ggplot(bar3_df, aes(x = bleaching_label, y = prop, fill = ITS2.Letter, alpha = bleaching)) +
  geom_col() +
  scale_fill_manual(values = its2_palette, breaks = its2_levels) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_alpha_manual(
    values = c("Bleached" = 0.6, "Non-Bleached" = 1)  # adjust names to match your data exactly
  ) +
  labs(
    y = "Proportion of Samples",
    fill = "Symbiont Genus",
  ) + guides(alpha = "none") +
  theme_pubr() +
  theme(axis.title.x = element_blank())
bar3

################################################################################

# richness 
metabolite_cols <- grep("^x", names(df), value = TRUE)

richness_df <- df %>%
  mutate(ITS2.Letter = factor(ITS2.Letter, levels = its2_levels)) |> 
  filter(
    !is.na(ITS2.Letter),
    ITS2.Letter != "No Seq"
  ) %>%
  select(sample_id, ITS2.Letter, all_of(metabolite_cols)) %>%
  rowwise() %>%
  mutate(MetabolomicRichness = sum(c_across(all_of(metabolite_cols)) > 0, na.rm = TRUE)) %>%
  ungroup()

kruskal_test_res <- richness_df %>%
  kruskal_test(MetabolomicRichness ~ ITS2.Letter)
# .y.                     n statistic    df         p method        
# * <chr>               <int>     <dbl> <int>     <dbl> <chr>         
#   1 MetabolomicRichness   296      26.4     4 0.0000258 Kruskal-Wallis

stat.test <- richness_df %>%
  dunn_test(MetabolomicRichness ~ ITS2.Letter, p.adjust.method = "BH") %>%
  filter(p.adj < 0.05) 

max_y <- max(richness_df$MetabolomicRichness, na.rm = TRUE)
n_comparisons <- nrow(stat.test)

stat.test <- stat.test %>%
  mutate(y.position = seq(from = max_y * 1.1, 
                          by = max_y * 0.05, 
                          length.out = n_comparisons))
# .y.                 group1          group2      n1    n2 statistic       p   p.adj p.adj.signif y.position
# <chr>               <chr>           <chr>    <int> <int>     <dbl>   <dbl>   <dbl> <chr>             <dbl>
#   1 MetabolomicRichness Symbiodinium Breviol…    35    50      3.62 2.92e-4 1.46e-3 **                9589.
# 2 MetabolomicRichness Symbiodinium Durusdi…    35    43      2.60 9.20e-3 1.84e-2 *                10025.
# 3 MetabolomicRichness Breviolum       Cladoco…    50   126     -4.27 1.94e-5 1.94e-4 ***              10460.
# 4 MetabolomicRichness Breviolum       Mix         50    42     -3.30 9.70e-4 3.23e-3 **               10896.
# 5 MetabolomicRichness Cladocopium     Durusdi…   126    43      2.88 3.98e-3 9.96e-3 **               11332.
# 6 MetabolomicRichness Durusdinium     Mix         43    42     -2.24 2.53e-2 4.22e-2 *                11768.

## % plot
richness <- ggplot(richness_df, aes(x = ITS2.Letter, y = MetabolomicRichness, fill = ITS2.Letter)) +
  geom_boxplot(alpha = 0.8, outlier.shape = NA, aes(color = ITS2.Letter), median.linewidth = 2) + 
  geom_jitter(aes(color = ITS2.Letter), width = 0.2, size = 1.8, alpha = 0.3) +
  scale_fill_manual(values = its2_palette) +
  scale_color_manual(values = its2_palette) +
  stat_pvalue_manual(stat.test, label = "p.adj.signif", tip.length = 0.01) +
  theme_pubr() + 
  theme(
    legend.position = "none",
    axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    axis.ticks.x = element_blank()
  )

################################################################################

# Shannon entropy
shannon_index <- function(counts) {
  counts <- counts[counts > 0]   # drop zeros
  total <- sum(counts)
  if (total == 0) return(0)      # if no metabolites detected
  p <- counts / total
  -sum(p * log(p))
}

metabolite_cols <- grep("^x", names(df), value = TRUE)

# calculate metabolic entropy per sample
entropy_df <- df %>%
  mutate(ITS2.Letter = factor(ITS2.Letter, levels = its2_levels)) |>
  filter(
    !is.na(ITS2.Letter),
    ITS2.Letter != "No Seq"
  ) |> 
  select(sample_id, ITS2.Letter, all_of(metabolite_cols)) %>%
  rowwise() %>%
  mutate(MetabolicEntropy = shannon_index(c_across(all_of(metabolite_cols)))) %>%
  ungroup()

entropy_df$ITS2.Letter <- droplevels(entropy_df$ITS2.Letter)
kruskal_entropy <- entropy_df %>%
  kruskal_test(MetabolicEntropy ~ ITS2.Letter)
# 1 MetabolicEntropy   296      10.1     4 0.0386 Kruskal-Wallis

stat.test_entropy <- entropy_df %>%
  dunn_test(MetabolicEntropy ~ ITS2.Letter, p.adjust.method = "BH") %>%
  arrange(p.adj)

stat.test_entropy <- stat.test_entropy %>%
  filter(p.adj < 0.05)


# Manual Y-positioning for brackets
max_y_ent <- max(entropy_df$MetabolicEntropy, na.rm = TRUE)
n_comp_ent <- nrow(stat.test_entropy)

stat.test_entropy <- stat.test_entropy %>%
  mutate(y.position = seq(from = max_y_ent * 1.05, 
                          by = max_y_ent * 0.07, 
                          length.out = n_comp_ent))
# .y.              group1      group2         n1    n2 statistic       p  p.adj p.adj.signif y.position
# <chr>            <chr>       <chr>       <int> <int>     <dbl>   <dbl>  <dbl> <chr>             <dbl>
#   1 MetabolicEntropy Breviolum   Durusdinium    50    43      2.73 0.00642 0.0370 *                  7.15
# 2 MetabolicEntropy Durusdinium Mix            43    42     -2.68 0.00740 0.0370 *                  7.62

entropy <- ggplot(entropy_df, aes(x = ITS2.Letter, y = MetabolicEntropy, fill = ITS2.Letter)) +
  geom_boxplot(
    aes(color = ITS2.Letter),
    alpha = 0.8, 
    outlier.shape = NA, 
    median.linewidth = 2
  ) + 
  geom_jitter(
    aes(color = ITS2.Letter), 
    width = 0.2, 
    size = 1.8, 
    alpha = 0.3
  ) +
  scale_fill_manual(values = its2_palette) +
  scale_color_manual(values = its2_palette) +
  stat_pvalue_manual(
    stat.test_entropy, 
    label = "p.adj.signif", 
    tip.length = 0.01
  ) +
  labs(
    y = "Shannon Entropy"
  ) +
  ylim(5, 8) +
  theme_pubr() + theme(
    legend.position = "none",
    axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    axis.ticks.x = element_blank()
  )

################################################################################

## pcoas
## filter to remove "mix" samples
keep_genera <- c("Symbiodinium", "Breviolum", "Cladocopium", "Durusdinium")

# Filter and ensure numeric data is clean
corals_4g <- df %>%
  filter(ITS2.Letter %in% keep_genera, !is.na(sample_id))

corals_4g <- corals_4g %>%
  select(-X.1)

permanova_numeric_data <- corals_4g %>% select(starts_with("x"))
keep_rows <- rowSums(is.na(permanova_numeric_data)) < ncol(permanova_numeric_data)

# Final clean subsets
permanova_numeric_data2 <- permanova_numeric_data[keep_rows, , drop = FALSE]
meta2 <- corals_4g[keep_rows, , drop = FALSE] %>%
  mutate(
    ITS2.Letter = factor(ITS2.Letter, levels = its2_levels),
    location = factor(location, levels = c("Sri Lanka", "Curaçao", "Hawaii"))
  )
permanova_numeric_data2[is.na(permanova_numeric_data2)] <- 0

numeric_only_data <- meta2 %>% 
  select(starts_with("x")) %>% 
  select(where(is.numeric)) 

permanova_matrix <- as.matrix(numeric_only_data)
bray_curtis_4g <- vegdist(permanova_matrix, method = "bray")

################################################################################
# nested permanova
permanova_result <- adonis2(
  bray_curtis_4g ~ ITS2.Letter / location,
  data = meta2,
  permutations = 999,
  by = "margin"
)
print(permanova_result)
# adonis2(formula = bray_curtis_4g ~ ITS2.Letter/location, data = meta2, permutations = 999, by = "margin")
# Df SumOfSqs      R2      F Pr(>F)    
# ITS2.Letter:location   6    8.086 0.20385 9.9613  0.001 ***
#   Residual             210   28.412 0.71625                  
# Total                218   39.668 1.00000                  
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

pcoa_result <- cmdscale(bray_curtis_4g, eig = TRUE, k = 2)
pcoa_points <- as.data.frame(pcoa_result$points)
colnames(pcoa_points) <- c("PCoA1", "PCoA2")
pcoa_points <- bind_cols(pcoa_points, meta2)

pos_eig <- pcoa_result$eig[pcoa_result$eig > 0]
var_explained <- round(100 * pcoa_result$eig[1:2] / sum(pos_eig), 1)

perm_p_val <- min(permanova_result$`Pr(>F)`, na.rm = TRUE)
perm_r2    <- round(sum(permanova_result$R2[1]), 3)
p_label    <- if(perm_p_val <= 0.001) "p < 0.001" else paste("p =", perm_p_val)
stats_annotation_combined <- paste0("PERMANOVA: R² = ", perm_r2, ", ", p_label)

p <- ggplot(pcoa_points, aes(x = PCoA1, y = PCoA2, color = ITS2.Letter, fill = ITS2.Letter)) +
  geom_point(size = 2, alpha = 0.8, aes(color = ITS2.Letter, shape = location)) +
  stat_ellipse(
    geom = "polygon",
    alpha = 0.15,
    level = 0.95,
    type = "t",
    colour = NA
  ) +
  annotate(
    "text",
    x = -Inf,  
    y = -Inf, 
    label = stats_annotation_combined,
    hjust = -0.05,
    vjust = -0.6,
    size = 4) +
  scale_color_manual(values = its2_palette) +
  scale_fill_manual(values = its2_palette) +
  scale_shape_manual(values = c("Sri Lanka" = 15, "Curaçao" = 11, "Hawaii" = 16)) + 
  labs(
    x = paste0("PCoA1: (", var_explained[1], "%)"),
    y = paste0("PCoA2: (", var_explained[2], "%)"),
    color = "Symbiont Genus",
    fill = "Symbiont Genus",
    shape = "Location"
  ) +
  theme_pubr() +
  theme(legend.position = "right")

################################################################################

its2_permanova_result <- adonis2(
  bray_curtis_4g ~ ITS2.Letter,
  data = meta2,
  permutations = 999,
  by = "margin"
)
print(its2_permanova_result)
# adonis2(formula = bray_curtis_4g ~ ITS2.Letter, data = meta2, permutations = 999, by = "margin")
# Df SumOfSqs     R2      F Pr(>F)    
# ITS2.Letter   2    3.169 0.0799 9.3779  0.001 ***
#   Residual    216   36.499 0.9201                  
# Total       218   39.668 1.0000                  
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

pcoa_result <- cmdscale(bray_curtis_4g, eig = TRUE, k = 2)
pcoa_points <- as.data.frame(pcoa_result$points)
colnames(pcoa_points) <- c("PCoA1", "PCoA2")
pcoa_points <- bind_cols(pcoa_points, meta2)

pos_eig <- pcoa_result$eig[pcoa_result$eig > 0]
var_explained <- round(100 * pcoa_result$eig[1:2] / sum(pos_eig), 1)

perm_p_val <- min(its2_permanova_result$`Pr(>F)`, na.rm = TRUE)
perm_r2    <- round(sum(its2_permanova_result$R2[1]), 3)
p_label    <- if(perm_p_val <= 0.001) "p < 0.001" else paste("p =", perm_p_val)
stats_annotation_combined <- paste0("PERMANOVA: R² = ", perm_r2, ", ", p_label)

p2 <- ggplot(pcoa_points, aes(x = PCoA1, y = PCoA2, color = ITS2.Letter, fill = ITS2.Letter)) +
  geom_point(size = 2, alpha = 0.8, aes(color = ITS2.Letter)) +
  stat_ellipse(
    geom = "polygon",
    alpha = 0.15,
    level = 0.95,
    type = "t",
    colour = NA
  ) +
  annotate(
    "text",
    x = -Inf,  
    y = -Inf, 
    label = stats_annotation_combined,
    hjust = -0.05,
    vjust = -0.6,
    size = 4) +
  scale_color_manual(values = its2_palette) +
  scale_fill_manual(values = its2_palette) +
  labs(
    x = paste0("PCoA1: (", var_explained[1], "%)"),
    y = paste0("PCoA2: (", var_explained[2], "%)"),
    color = "Symbiont Genus",
    fill = "Symbiont Genus"
  ) +
  theme_pubr() +
  theme(legend.position = "right")
p2

################################################################################

# upset_input <- df %>%
#   filter(!is.na(ITS2.Letter), ITS2.Letter != "No Seq") %>% 
#   filter(ITS2.Letter != "Mix") %>%
#   select(ITS2.Letter, all_of(metabolite_cols)) %>%
#   pivot_longer(cols = starts_with("x"), names_to = "metabolite", values_to = "abundance") %>%
#   group_by(ITS2.Letter, metabolite) %>%
#   summarise(present = as.numeric(any(abundance > 0, na.rm = TRUE)), .groups = "drop") %>%
#   pivot_wider(names_from = ITS2.Letter, values_from = present, values_fill = 0) %>%
#   as.data.frame()
######

upset_input <- df %>%
  filter(!is.na(ITS2.Letter), ITS2.Letter != "No Seq") %>%
  select(ITS2.Letter, all_of(metabolite_cols)) %>%
  pivot_longer(cols = starts_with("x"), names_to = "metabolite", values_to = "abundance") %>%
  group_by(ITS2.Letter, metabolite) %>%
  summarise(present = as.numeric(any(abundance > 0, na.rm = TRUE)), .groups = "drop") %>%
  pivot_wider(names_from = ITS2.Letter, values_from = present, values_fill = 0) %>%
  as.data.frame()

upset_data <- upset_input[, -1]
upset(
  upset_data,
  sets = colnames(upset_data), # Uses your ITS2.Letter groups
  main.bar.color = "steelblue",
  sets.bar.color = its2_palette[colnames(upset_data)], # Match your existing palette
  order.by = "freq", 
  decreasing = TRUE,
  point.size = 3.5, 
  line.size = 1.5,
  text.scale = c(1.3, 1.3, 1, 1, 1.5, 1) # Adjust text sizes for labels
)