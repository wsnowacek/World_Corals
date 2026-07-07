library(tidyverse)
library(knitr)
library(readxl)
library(data.table)
library(vegan)
library(scales)
library(cowplot)
library(RColorBrewer)
library(ggpubr)
library(forcats)
library(caret)
library(tibble)
library(stringr)
library(RColorBrewer)
library(ggrepel)
library(gt)
library(here)

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
    # host_order = fct_relevel(factor(host_order), "Scleractinia"),
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
cols_origin <- c("Host" = "#97B9CBFF", "Symbiont" = "#9057C6FF", 
                 "Both" = "#FFE1BDFF", "Unknown" = "#8DC657FF")

##################################################

# summary stats
sum(!is.na(df$host_order)) #542
sum(!is.na(df$host_family)) #542
sum(!is.na(df$host_species)) #479

sum(met_df$refined_origin == "Unknown") ##7789
sum(met_df$refined_origin != "Unknown") ##8579

sum(met_df$refined_origin == "Host") ##3623
sum(met_df$refined_origin == "Both") ##1977
sum(met_df$refined_origin == "Symbiont") ##2979

classification_table <- df %>%
  group_by(host_phylum, host_class, host_order, host_family, host_genus, host_species) %>%
  tally(name = "sample_count") %>%
  ungroup()
write.csv(classification_table, "misc/class_table.csv")

## produces a nested classification table
nested_gt_table <- classification_table %>%
  gt(groupname_col = "host_phylum") %>% # Groups by Phylum first
  tab_header(
    title = "Taxonomic Classification of Study Species",
    subtitle = "Hierarchical breakdown from Phylum to Species"
  ) %>%
  cols_label(
    host_class = "Class",
    host_order = "Order",
    host_family = "Family",
    host_genus = "Genus",
    host_species = "Species",
    sample_count = "N"
  ) %>%
  tab_options(
    row_group.font.weight = "bold",
    column_labels.font.weight = "bold"
  )
nested_gt_table

scler_df <- df %>% filter(df$scleractinia == 1, na.rm = TRUE)
out_df <- df %>% filter(df$scleractinia == 0, na.rm = TRUE)
  
taxa_summary <- out_df %>%
  summarise(
    Phyla = n_distinct(host_phylum, na.rm = TRUE),
    Classes = n_distinct(host_class, na.rm = TRUE),
    Orders = n_distinct(host_order, na.rm = TRUE),
    Families = n_distinct(host_family, na.rm = TRUE),
    Genera = n_distinct(host_genus, na.rm = TRUE),
    Species = n_distinct(host_species, na.rm = TRUE)
  ) %>%
  pivot_longer(
    cols = everything(), 
    names_to = "Taxonomic Level", 
    values_to = "Unique Count"
  )
print(taxa_summary)
## number of metabolites = 16368

################################################################################

# Bar plots and table of compounds by refined origin for Figure S1

class_counts <- met_plot_df %>%
  filter(compound_class != "Unknown") %>%
  count(compound_class, sort = TRUE) %>%
  mutate(compound_class = fct_inorder(compound_class))
kable(class_counts, col.names = c("Compound Class", "Count"))
#4655 metabolites with annotation
# |Compound Class                         | Count|
#   |:--------------------------------------|-----:|
#   |Glycerophosphocholines                 |   801|
#   |TAG                                    |   783|
#   |Neutral GSL                            |   336|
#   |Ceramides                              |   327|
#   |GPEtn                                  |   197|
#   |DAG                                    |   172|
#   |OxPL                                   |   141|
#   |Cyclic peptides                        |   120|
#   |PC(diacyl)                             |    86|
#   |TQ/THQs                                |    85|
#   |DGCC                                   |    83|
#   |PC(alkyl)                              |    76|
#   |Carotenoids                            |    75|
#   |MADAG                                  |    70|
#   |MGDG                                   |    68|
#   |Phosphosphingolipids                   |    66|
#   |N-acyl amines                          |    63|
#   |Fatty acyl carnitines                  |    60|
#   |Lyso-PC(acyl)                          |    55|
#   |Glycerophosphoserines                  |    48|
#   |Cholestane steroids                    |    45|
#   |Unsaturated fatty acids                |    45|
#   |DGDG                                   |    44|
#   |Depsipeptides                          |    39|
#   |HexCer                                 |    39|
#   |PE (mono-alkyl-mono-acyl)              |    38|
#   |Open-chain polyketides                 |    35|
#   |CAEP                                   |    31|
#   |Polyprenol derivatives                 |    30|
#   |Glycerophosphates                      |    29|
#   |Oxidized DGCC                          |    29|
#   |Cholane steroids                       |    28|
#   |Lyso-PC(alkyl)                         |    28|
#   |PS (diacyl)                            |    26|
#   |Linear peptides                        |    25|
#   |Lyso-DGCC                              |    24|
#   |Oleanane triterpenoids                 |    23|
#   |PI (diacyl)                            |    22|
#   |CerPE                                  |    20|
#   |Lipopeptides                           |    19|
#   |Glycerophosphoinositols                |    16|
#   |Lyso-PE(acyl)                          |    15|
#   |PE (diacyl)                            |    15|
#   |Apocarotenoids (Î²-)                   |    14|
#   |Prostaglandins                         |    14|
#   |Glycosyldiacylglycerols                |    11|
#   |Oxidized Lyso-PC(acyl)                 |    10|
#   |SQDG                                   |    10|
#   |Dammarane and Protostane triterpenoids |     9|
#   |Lyso-DGDG                              |     9|
#   |Oxidized PC(diacyl)                    |     9|
#   |Polyene macrolides                     |     9|
#   |Lyso-MGDG                              |     8|
#   |Macrotetrolides                        |     8|
#   |Monoacylglycerols                      |     8|
#   |PS (mono-alkyl-mono-acyl)              |     8|
#   |Pinane monoterpenoids                  |     7|
#   |Glycerophosphoglycerols                |     6|
#   |Purine nucleos(t)ides                  |     6|
#   |Hydrocarbons                           |     5|
#   |Microcystins                           |     5|
#   |Sphingoid bases                        |     5|
#   |Abietane diterpenoids                  |     4|
#   |Acetogenins                            |     4|
#   |CerPI                                  |     4|
#   |Flavanones                             |     4|
#   |Glycerophosphoinositol phosphates      |     4|
#   |Lyso-PE(alkyl)                         |     4|
#   |Oxidized DGDG                          |     4|
#   |Oxidized MGDG                          |     4|
#   |Primary amides                         |     4|
#   |Steroidal alkaloids                    |     4|
#   |Tripeptides                            |     4|
#   |Androstane steroids                    |     3|
#   |Labdane diterpenoids                   |     3|
#   |Lyso-PS (acyl)                         |     3|
#   |Oxidized PE (mono-alkyl-mono-acyl)     |     3|
#   |Polyamines                             |     3|
#   |Wax monoesters                         |     3|
#   |Avermectins                            |     2|
#   |Catechols with side chains             |     2|
#   |Epoxy fatty acids                      |     2|
#   |Ergostane steroids                     |     2|
#   |Estrane steroids                       |     2|
#   |Fatty acyl CoAs                        |     2|
#   |Halogenated hydrocarbons               |     2|
#   |Lupane triterpenoids                   |     2|
#   |Macrolide lactones                     |     2|
#   |Marine-bacterial DPEs                  |     2|
#   |Oxidized Lyso-PE(alkyl)                |     2|
#   |Pyrrole alkaloids                      |     2|
#   |RiPPs-Thiopeptides                     |     2|
#   |Simple phenolic acids                  |     2|
#   |Sphigomylein                           |     2|
#   |Vitamin D2 and derivatives             |     2|
#   |Anthraquinones and anthrones           |     1|
#   |Cembrane diterpenoids                  |     1|
#   |Chalcones                              |     1|
#   |Cyclitols                              |     1|
#   |Dipeptides                             |     1|
#   |Disaccharides                          |     1|
#   |Enediynes                              |     1|
#   |Erythromycins                          |     1|
#   |Glucosinolates                         |     1|
#   |Glycosylmonoacylglycerols              |     1|
#   |Guaiane sesquiterpenoids               |     1|
#   |Isoquinoline alkaloids                 |     1|
#   |Lyso-PS (alkyl)                        |     1|
#   |Methoxy fatty acids                    |     1|
#   |Microcolins and mirabimids             |     1|
#   |PI (mono-alkyl-mono-acyl)              |     1|
#   |Piperidine alkaloids                   |     1|
#   |Pyridine alkaloids                     |     1|
#   |Pyrimidine nucleos(t)ides              |     1|
#   |Taraxerane triterpenoids               |     1|
#   |Vitamin D3 and derivatives             |     1|
#   |ceramide                               |     1|
#   |oxidized PC(diacyl)                    |     1|

# plot bars per compound class
plot_data <- met_plot_df %>%
  filter(compound_class != "Unknown") %>%
  group_by(display_class) %>%
  mutate(class_count = n()) %>%
  ungroup() %>%
  mutate(axis_label = paste0(display_class, " (n = ", class_count, ")"))
label_levels <- plot_data %>%
  select(display_class, axis_label) %>%
  distinct() %>%
  arrange(display_class) %>%
  pull(axis_label)

plot_data$axis_label <- factor(plot_data$axis_label, levels = label_levels)
plot_data <- plot_data %>%
  mutate(refined_origin = factor(refined_origin, 
                                 levels = c("Host", "Symbiont", "Both", "Unknown")))

ordered_classes <- levels(plot_data$display_class)
ordered_classes <- ordered_classes[ordered_classes %in% plot_data$display_class]
axis_colors <- final_palette[ordered_classes]

### Figure S1
compound_bar_plot <- ggplot(plot_data, aes(x = axis_label, fill = refined_origin)) +
  geom_bar(position = "stack", width = 0.7) +
  scale_fill_manual(values = cols_origin) +
  coord_flip() +
  labs(
    x = "Compound Class",
    y = "Number of Metabolites",
    fill = "Metabolite Origin"
  ) +
  theme_pubr() +
  theme(
    axis.text.y = element_text(color = axis_colors, face = "bold"),
    panel.grid.minor = element_blank(),
    axis.label.x = element_text(size = 14),
    legend.position = "bottom"
  )

print(compound_bar_plot)
ggsave(here("misc", "figs", "compound_barplot.jpg"),
       compound_bar_plot,
       width = 10, height = 12, dpi = 300)


#########################
# host only metabolites
class_counts_host <- met_plot_df %>%
  filter(compound_class != "Unknown") %>%
  filter(refined_origin == "Host") %>%
  count(compound_class, sort = TRUE) %>%
  mutate(compound_class = fct_inorder(compound_class))
kable(class_counts_host, col.names = c("Compound Class", "Count"))
#1316 host mets with annotation
# |Compound Class                         | Count|
#   |:--------------------------------------|-----:|
#   |Glycerophosphocholines                 |   257|
#   |Ceramides                              |   166|
#   |PC(diacyl)                             |    86|
#   |PC(alkyl)                              |    76|
#   |MADAG                                  |    70|
#   |GPEtn                                  |    65|
#   |Neutral GSL                            |    49|
#   |Cyclic peptides                        |    38|
#   |PE (mono-alkyl-mono-acyl)              |    38|
#   |OxPL                                   |    36|
#   |Lyso-PC(acyl)                          |    32|
#   |Lyso-PC(alkyl)                         |    28|
#   |Glycerophosphoserines                  |    25|
#   |Phosphosphingolipids                   |    25|
#   |DAG                                    |    19|
#   |Cholestane steroids                    |    18|
#   |CerPE                                  |    17|
#   |Fatty acyl carnitines                  |    17|
#   |TAG                                    |    17|
#   |CAEP                                   |    15|
#   |Cholane steroids                       |    15|
# many others with n < 15

p_class <- ggbarplot(
  class_counts_host,
  x = "compound_class",
  y = "n",
  fill = "compound_class",
  palette = final_palette,
  sort.val = "desc",
  sort.by.groups = FALSE
) +
  geom_text(
    aes(label = paste0("n=", n)),
    hjust = -0.1,   # pushes text slightly outside bars
    size = 4,
    fontface = "bold"
  ) + 
  labs(
    x = "Compound Class",
    y = "# of Metabolites"
  ) +
  theme_pubr() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  ylim(0,300) +
  coord_flip()

p_class
ggsave(here("misc", "figs", "Fig1hostmetabolites.jpg"), p_class, width = 10, height = 14, dpi = 300)

#########################
# symbiont only metabolites
class_counts_sym <- met_plot_df %>%
  filter(compound_class != "Unknown") %>%
  filter(refined_origin == "Symbiont") %>%
  count(compound_class, sort = TRUE) %>%
  mutate(compound_class = fct_inorder(compound_class))
kable(class_counts_sym, col.names = c("Compound Class", "Count"))
# 989 symbiont metabolites with annotation
# |Compound Class                         | Count|
#   |:--------------------------------------|-----:|
#   |TAG                                    |   252|
#   |DGCC                                   |    83|
#   |Glycerophosphocholines                 |    76|
#   |MGDG                                   |    68|
#   |Neutral GSL                            |    68|
#   |Carotenoids                            |    44|
#   |DGDG                                   |    44|
#   |DAG                                    |    41|
#   |Cyclic peptides                        |    33|
#   |Oxidized DGCC                          |    29|
#   |Ceramides                              |    27|
#   |Lyso-DGCC                              |    24|
#   |TQ/THQs                                |    24|
#   |GPEtn                                  |    21|
#   |HexCer                                 |    10|
#   |SQDG                                   |    10|
#   |Apocarotenoids (Î²-)                   |     9|
#   |Lyso-DGDG                              |     9|
#   |Lyso-MGDG                              |     8|
#   |OxPL                                   |     8|
#   |N-acyl amines                          |     7|
#   |Linear peptides                        |     6|
#   |Polyprenol derivatives                 |     6|
#   |Glycerophosphates                      |     5|
#   |Lipopeptides                           |     5|
#   |Oleanane triterpenoids                 |     5|
#   |Polyene macrolides                     |     5|
#   |Fatty acyl carnitines                  |     4|
#   |Microcystins                           |     4|
#   |Open-chain polyketides                 |     4|
#   |Oxidized DGDG                          |     4|
#   |Oxidized MGDG                          |     4|
#   |Phosphosphingolipids                   |     4|
#   |Acetogenins                            |     3|
#   |Glycerophosphoinositols                |     3|
#   |Glycerophosphoserines                  |     3|
#   |Avermectins                            |     2|
#   |Cholestane steroids                    |     2|
#   |Depsipeptides                          |     2|
#   |Glycerophosphoinositol phosphates      |     2|
#   |Glycosyldiacylglycerols                |     2|
#   |PE (diacyl)                            |     2|
#   |PI (diacyl)                            |     2|
#   |Abietane diterpenoids                  |     1|
#   |Dammarane and Protostane triterpenoids |     1|
#   |Ergostane steroids                     |     1|
#   |Erythromycins                          |     1|
#   |Estrane steroids                       |     1|
#   |Flavanones                             |     1|
#   |Glycerophosphoglycerols                |     1|
#   |Guaiane sesquiterpenoids               |     1|
#   |Labdane diterpenoids                   |     1|
#   |Macrotetrolides                        |     1|
#   |Marine-bacterial DPEs                  |     1|
#   |Monoacylglycerols                      |     1|
#   |Pinane monoterpenoids                  |     1|
#   |Polyamines                             |     1|
#   |Wax monoesters                         |     1|

p_class_sym <- ggbarplot(
  class_counts_sym,
  x = "compound_class",
  y = "n",
  fill = "compound_class",
  palette = final_palette,
  sort.val = "desc",
  sort.by.groups = FALSE
) +
  geom_text(
    aes(label = paste0("n=", n)),
    hjust = -0.1,   # pushes text slightly outside bars
    size = 4,
    fontface = "bold"
  ) + 
  labs(
    x = "Compound Class",
    y = "# of Metabolites"
  ) +
  theme_pubr() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  ylim(0,300) +
  coord_flip()

p_class_sym
ggsave(here("misc", "figs", "Fig1hostmetabolites.jpg"), p_class_sym, width = 10, height = 14, dpi = 300)


#########################
# symbiont only metabolites
class_counts_both <- met_plot_df %>%
  filter(compound_class != "Unknown") %>%
  filter(refined_origin == "Both") %>%
  count(compound_class, sort = TRUE) %>%
  mutate(compound_class = fct_inorder(compound_class))
kable(class_counts_both, col.names = c("Compound Class", "Count"))
# 669 both metabolites with annotation
# |Compound Class                         | Count|
#   |:--------------------------------------|-----:|
#   |TAG                                    |   294|
#   |Glycerophosphocholines                 |   116|
#   |Ceramides                              |    33|
#   |Neutral GSL                            |    31|
#   |DAG                                    |    28|
#   |TQ/THQs                                |    22|
#   |GPEtn                                  |    14|
#   |HexCer                                 |    12|
#   |Polyprenol derivatives                 |    12|
#   |Lyso-PC(acyl)                          |     8|
#   |Carotenoids                            |     7|
#   |Fatty acyl carnitines                  |     7|
#   |N-acyl amines                          |     7|
#   |OxPL                                   |     7|
#   |Phosphosphingolipids                   |     7|
#   |Glycerophosphates                      |     5|
#   |Glycosyldiacylglycerols                |     4|
#   |Primary amides                         |     4|
#   |CAEP                                   |     3|
#   |CerPE                                  |     3|
#   |Cyclic peptides                        |     3|
#   |Depsipeptides                          |     3|
#   |PI (diacyl)                            |     3|
#   |PS (diacyl)                            |     3|
#   |Cholestane steroids                    |     2|
#   |Dammarane and Protostane triterpenoids |     2|
#   |Lipopeptides                           |     2|
#   |Monoacylglycerols                      |     2|
#   |Pinane monoterpenoids                  |     2|
#   |Sphingoid bases                        |     2|
#   |Cembrane diterpenoids                  |     1|
#   |CerPI                                  |     1|
#   |Cholane steroids                       |     1|
#   |Flavanones                             |     1|
#   |Glycerophosphoinositol phosphates      |     1|
#   |Glycerophosphoserines                  |     1|
#   |Hydrocarbons                           |     1|
#   |Labdane diterpenoids                   |     1|
#   |Lyso-PE(acyl)                          |     1|
#   |Macrotetrolides                        |     1|
#   |Marine-bacterial DPEs                  |     1|
#   |Microcolins and mirabimids             |     1|
#   |Oleanane triterpenoids                 |     1|
#   |Open-chain polyketides                 |     1|
#   |Oxidized Lyso-PC(acyl)                 |     1|
#   |Piperidine alkaloids                   |     1|
#   |Polyamines                             |     1|
#   |Prostaglandins                         |     1|
#   |Simple phenolic acids                  |     1|
#   |Unsaturated fatty acids                |     1|
#   |Wax monoesters                         |     1|

p_class_both <- ggbarplot(
  class_counts_both,
  x = "compound_class",
  y = "n",
  fill = "compound_class",
  palette = final_palette,
  sort.val = "desc",
  sort.by.groups = FALSE
) +
  geom_text(
    aes(label = paste0("n=", n)),
    hjust = -0.1,   # pushes text slightly outside bars
    size = 4,
    fontface = "bold"
  ) + 
  labs(
    x = "Compound Class",
    y = "# of Metabolites"
  ) +
  theme_pubr() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  ylim(0,350) +
  coord_flip()
p_class_both
ggsave(here("misc", "figs", "Fig1hostmetabolites.jpg"), p_class_both, width = 10, height = 14, dpi = 300)

################################################################################

# p_class <- p_class + (theme(axis.title.x = element_blank()))
# p_class_sym <- p_class_sym + theme(axis.title.x = element_blank(), axis.title.y = element_blank())
# p_class_both <- p_class_both + theme(axis.title.x = element_blank(), axis.title.y = element_blank())
# 
# combined <- plot_grid(p_class, p_class_sym, p_class_both,
#                       ncol = 3,
#                       labels = c("A","B","C"),
#                       label_size=30)
# ggsave(here("misc", "figs", "Fig1hostmetabolites.jpg"), combined, width = 24, height = 14, dpi = 300)


################################################################################
