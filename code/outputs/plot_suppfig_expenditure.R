suppressMessages({library(ggplot2); library(dplyr); library(tidyr); library(patchwork)})
# Monte Carlo medians and 95% uncertainty intervals (1000 simulations; expenditure at central values), 20% coverage
d <- read.csv("data/results/v3/expenditure_ui/cost_ui.csv", check.names = FALSE) %>%
  filter(outcomes == "IHD + IS", coverage == "20%", ISO3 != "GLB") %>%
  transmute(Country = ISO3, Price = price, diet = diet_pd, diet_lo = diet_pd_lo, diet_hi = diet_pd_hi,
            trt = trt_pd, trt_lo = trt_pd_lo, trt_hi = trt_pd_hi) %>%
  filter(is.finite(diet), is.finite(trt))
col_diet <- "#2d9b8a"; col_glp1 <- "#e85d5d"
ord <- d %>% filter(Price == "Generic $140/yr") %>% arrange(trt) %>% pull(Country)
d_long <- bind_rows(
  d %>% transmute(Country, Price, Arm = "Dietary policy", Cost = diet, lo = diet_lo, hi = diet_hi),
  d %>% transmute(Country, Price, Arm = "GLP-1RA treatment", Cost = trt, lo = trt_lo, hi = trt_hi)) %>%
  mutate(Country = factor(Country, levels = ord), Arm = factor(Arm, c("Dietary policy", "GLP-1RA treatment")),
         y = log10(Cost / 1000), ylo = log10(pmax(lo, 100) / 1000), yhi = log10(pmin(hi, 1e11) / 1000))
brk <- c(1, 10, 100, 1000, 10000); lab <- c("US$1k", "US$10k", "US$100k", "US$1M", "US$10M")
ymax <- max(d_long$yhi, na.rm = TRUE) * 1.02
TITLE <- c("Branded" = "Branded", "Generic $140/yr" = "Generic US$140", "Generic $28/yr" = "Generic US$28")
mk <- function(price, show_leg) {
  ggplot(filter(d_long, Price == price), aes(Country, y, fill = Arm)) +
    geom_hline(yintercept = log10(brk), colour = "grey85", linewidth = 0.3) +
    geom_col(position = position_dodge(width = 0.8), width = 0.75) +
    geom_errorbar(aes(ymin = ylo, ymax = yhi), position = position_dodge(width = 0.8), width = 0.3,
                  linewidth = 0.3, colour = "grey25") +
    annotate("text", x = 0.5, y = log10(brk), label = lab, size = 2.6, colour = "grey40", hjust = 1.1) +
    scale_fill_manual(values = c(col_diet, col_glp1), name = NULL) +
    scale_y_continuous(limits = c(-1.2, ymax)) +
    coord_polar(start = 0) +
    labs(title = TITLE[[price]], x = NULL, y = NULL) +
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
