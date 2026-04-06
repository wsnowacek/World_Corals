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
      ITS2.Letter == "A" ~ "Symbiodiniaceae",
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
  "Symbiodiniaceae"  = "#5BBCD6FF",
  "Mix"              = "#F98400FF"
)

its2_levels <- c(
  "Symbiodiniaceae",
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
                         levels = c("No Seq","Mix","Durusdinium","Cladocopium","Breviolum","Symbiodiniaceae"))
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
