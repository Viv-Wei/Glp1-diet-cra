suppressMessages({library(ggplot2); library(dplyr); library(tidyr); library(patchwork)})
d <- read.csv("data/results/v3/equity_cost_final/cov0.20/tableS_cost_v3.csv", check.names = FALSE) %>%
  filter(Outcomes == "IHD + IS", ISO3 != "GLB") %>%
  transmute(Country = ISO3, `Price scenario`,
            `Diet cost/death (USD)` = `Dietary policy expenditure per death averted (USD)`,
            `GLP-1 cost/death (USD)` = `GLP-1RA expenditure per death averted (USD)`) %>%
  filter(!is.na(`Diet cost/death (USD)`), !is.na(`GLP-1 cost/death (USD)`))
col_diet <- "#2d9b8a"; col_glp1 <- "#e85d5d"
ord <- d %>% filter(`Price scenario` == "Generic $140/yr") %>% arrange(`GLP-1 cost/death (USD)`) %>% pull(Country)
d_long <- d %>%
  transmute(Country = factor(Country, levels = ord), Price = `Price scenario`,
            `Dietary policy` = `Diet cost/death (USD)`, `GLP-1 treatment` = `GLP-1 cost/death (USD)`) %>%
  pivot_longer(c(`Dietary policy`, `GLP-1 treatment`), names_to = "Arm", values_to = "Cost") %>%
  mutate(Arm = factor(Arm, c("Dietary policy", "GLP-1 treatment")),
         y = log10(Cost / 1000))
brk <- c(1, 10, 100, 1000, 10000); lab <- c("$1k", "$10k", "$100k", "$1M", "$10M")
ymax <- log10(max(d_long$Cost) / 1000) * 1.02
mk <- function(price, show_leg) {
  ggplot(filter(d_long, Price == price), aes(Country, y, fill = Arm)) +
    geom_hline(yintercept = log10(brk), colour = "grey85", linewidth = 0.3) +
    geom_col(position = position_dodge(width = 0.8), width = 0.75) +
    annotate("text", x = 0.5, y = log10(brk), label = lab, size = 2.6, colour = "grey40", hjust = 1.1) +
    scale_fill_manual(values = c(col_diet, col_glp1), name = NULL) +
    scale_y_continuous(limits = c(-1.2, ymax)) +
    coord_polar(start = 0) +
    labs(title = price, x = NULL, y = NULL) +
    theme_minimal(base_size = 9) +
    theme(axis.text.y = element_blank(), axis.text.x = element_text(size = 8, face = "bold"),
          panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
          legend.position = if (show_leg) "bottom" else "none")
}
prices <- c("Branded", "Generic $140/yr", "Generic $28/yr")
p <- mk(prices[1], FALSE) + mk(prices[2], TRUE) + mk(prices[3], FALSE) +
  plot_layout(ncol = 3, guides = "collect") & theme(legend.position = "bottom")
ggsave("data/figures/figure6c_cost_log.pdf", p, width = 18, height = 7, device = cairo_pdf)
ggsave("data/figures/figure6c_cost_log.png", p, width = 18, height = 7, dpi = 300)
OUTDIR <- "data/figures"; dir.create(OUTDIR, showWarnings = FALSE)
tags <- c("branded", "generic140", "generic28")
for (i in seq_along(prices)) {
  pi <- mk(prices[i], FALSE) + theme(plot.background = element_rect(fill = "transparent", colour = NA),
                                     panel.background = element_rect(fill = "transparent", colour = NA))
  ggsave(file.path(OUTDIR, sprintf("fig6c_%s.pdf", tags[i])), pi, width = 6, height = 6.3,
         device = cairo_pdf, bg = "transparent")
  ggsave(file.path(OUTDIR, sprintf("fig6c_%s.png", tags[i])), pi, width = 6, height = 6.3,
         dpi = 600, bg = "transparent")
}
leg <- cowplot::get_legend(mk(prices[2], TRUE) + theme(legend.text = element_text(size = 11)))
ggsave(file.path(OUTDIR, "fig6c_legend.pdf"), leg, width = 4, height = 0.5, device = cairo_pdf, bg = "transparent")
cat("done: data/figures/figure6c_cost_log.*\n")
