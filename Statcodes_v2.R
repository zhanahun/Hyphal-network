## I. Proportional data [0, 1]=================================================================================
# 1. CV2 ======================================================================================================
## =========================
## 1. Load packages
## =========================
library(betareg)
library(emmeans)
library(multcomp)
library(ggplot2)

## =========================
## 2. Read data
## =========================
df <- read.csv("e2.csv")
df$Gradient <- factor(df$Gradient)

## =========================
## 3. Prepare response (0 < y < 1)
## =========================
n <- nrow(df)
df$CV2_beta <- (df$CV2 * (n - 1) + 0.5) / n   # Smithson & Verkuilen correction

## =========================
## 4. Fit beta regression model
## =========================
fit_beta <- betareg(CV2_beta ~ Gradient + ET, data = df)
summary(fit_beta)

library(lmtest)
fit_full   <- betareg(CV2_beta ~ Gradient + ET, data = df)
fit_noGrad <- betareg(CV2_beta ~ ET,            data = df)
fit_noET   <- betareg(CV2_beta ~ Gradient,      data = df)

lrtest(fit_noGrad, fit_full)
lrtest(fit_noET,   fit_full)

## =========================
## 5. Estimated marginal means (response scale)
## =========================
adj_means <- emmeans(fit_beta, ~ Gradient, type = "response")
pairs(adj_means, adjust = "tukey")

## =========================
## 6. Significance letters (CLD)
## =========================
cld_res <- cld(
  adj_means,
  adjust = "tukey",
  Letters = letters,
  reversed = TRUE
)
plot_df <- as.data.frame(cld_res)
plot_df$.group <- trimws(plot_df$.group)

## =========================
## 7. Compute SE-based error bar bounds       #### CHANGED ####
## =========================
plot_df$se_lower <- plot_df$emmean - plot_df$SE
plot_df$se_upper <- plot_df$emmean + plot_df$SE

fixed_y_pos <- 0.95

## =========================
## 8. Plot
## =========================
ggplot() +
  ## Raw data
  geom_jitter(
    data = df,
    aes(x = Gradient, y = CV2),
    width = 0.12,
    size = 1.5,
    alpha = 0.6,
    color = "grey40"
  ) +
  
  ## SE error bars                            #### CHANGED ####
geom_errorbar(
  data = plot_df,
  aes(x = Gradient, ymin = se_lower, ymax = se_upper),
  width = 0.12,
  linewidth = 0.8,
  color = "black"
) +
  
  ## Adjusted means
  geom_point(
    data = plot_df,
    aes(x = Gradient, y = emmean),
    size = 2.5,
    shape = 16,
    color = "red"
  ) +
  
  scale_y_continuous(
    limits = c(0, 1),
    expand = expansion(mult = c(0.05, 0.15))
  ) +
  
  labs(
    x = "N concentration (mM)",
    y = "Hyphal coverage"
  ) +
  
  theme_classic(base_size = 14) +
  theme(
    axis.text  = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 14),
    axis.line  = element_line(color = "black", linewidth = 0.6),
    legend.position = "none"
  )


ggsave("Fig_cv2_2.pdf",
       width = 85, height = 85, units = "mm")

