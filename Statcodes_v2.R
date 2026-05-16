##################################################################################################################################################
## Data analysis for "Arbuscular mycorrhizal fungi change foraging behavior by altering hyphal structures in response to nitrogen availability" ##
##################################################################################################################################################

### Data check ========================================================================================================================================
df<-read.csv("e2.csv")
##Homogeneity of variance check
library(car)
leveneTest(CV2 ~ Gradient, data = df)

##Normality assumption check
library(tidyverse)
library(rstatix)
library(ggpubr)

normality_results <- df %>%
  group_by(Gradient) %>%
  shapiro_test(CV2)

print("--- Shapiro-Wilk Test Results (p > 0.05 is good) ---")
print(normality_results)

qq_plots <- ggqqplot(df, x = "CV2", facet.by = "Gradient")
print(qq_plots)

### Pearson's correlation test=========================================================================================================================
cor.test(df$RBR, df$CV2, method = "pearson") #replace names with other variables

###Plotting ===========================================================================================================================================
## I. Proportional data [0, 1] =====================================================================================
# 1. CV2 =======================================================================
## =========================
## 1. Load packages
## =========================
library(betareg)
library(emmeans)
library(multcomp)
library(ggplot2)
library(lmtest)

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

# Fit full and nested models
fit_full   <- betareg(CV2_beta ~ Gradient + ET, data = df)
fit_noGrad <- betareg(CV2_beta ~ ET,            data = df)
fit_noET   <- betareg(CV2_beta ~ Gradient,      data = df)

# Test the overall effect of each factor
lrtest(fit_noGrad, fit_full)   # Overall effect of Gradient
lrtest(fit_noET,   fit_full)   # Overall effect of ET

## =========================
## 5. Estimated marginal means (response scale)
## =========================
adj_means <- emmeans(fit_beta, ~ Gradient, type = "response")
pairs(adj_means, adjust = "tukey")

plot_df <- as.data.frame(adj_means)

## =========================
## 6. Compute SE-based error bar bounds
## =========================
plot_df$se_lower <- plot_df$emmean - plot_df$SE
plot_df$se_upper <- plot_df$emmean + plot_df$SE

## =========================
## 7. Plot
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
  
  ## SE error bars
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


# 2. RH Branching Rate =========================================================
df <- read.csv("e2.csv")
df$Gradient <- factor(df$Gradient)

## =========================
## 3. Prepare response (0 < y < 1)
## =========================
n <- nrow(df)
df$RBR_beta <- (df$RBR * (n - 1) + 0.5) / n   # Smithson & Verkuilen correction

## =========================
## 4. Fit beta regression model
## =========================
fit_beta <- betareg(RBR_beta ~ Gradient + ET, data = df)
summary(fit_beta)

# Fit full and nested models
fit_full   <- betareg(RBR_beta ~ Gradient + ET, data = df)
fit_noGrad <- betareg(RBR_beta ~ ET,            data = df)
fit_noET   <- betareg(RBR_beta ~ Gradient,      data = df)

# Test the overall effect of each factor
lrtest(fit_noGrad, fit_full)   # Overall effect of Gradient
lrtest(fit_noET,   fit_full)   # Overall effect of ET

## =========================
## 5. Estimated marginal means (response scale)
## =========================
adj_means <- emmeans(fit_beta, ~ Gradient, type = "response")
pairs(adj_means, adjust = "tukey")

plot_df <- as.data.frame(adj_means)

## =========================
## 6. Compute SE-based error bar bounds
## =========================
plot_df$se_lower <- plot_df$emmean - plot_df$SE
plot_df$se_upper <- plot_df$emmean + plot_df$SE

## =========================
## 7. Plot
## =========================
ggplot() +
  ## Raw data
  geom_jitter(
    data = df,
    aes(x = Gradient, y = RBR),
    width = 0.12,
    size = 1.6,
    alpha = 0.6,
    color = "grey40"
  ) +
  
  ## SE error bars
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
    size = 3,
    shape = 16,
    color = "red"
  ) +
  
  scale_y_continuous(
    limits = c(0, 0.2),
    expand = expansion(mult = c(0.05, 0.1))
  ) +
  
  labs(
    x = "N concentration (mM)",
    y = "RH branching rate"
  ) +
  
  theme_classic(base_size = 14) +
  theme(
    axis.text  = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 14),
    axis.line  = element_line(color = "black", linewidth = 0.6),
    legend.position = "none"
  )

ggsave("Fig_RBR.pdf",
       width = 85, height = 85, units = "mm")


# 3. BAS Branching Rate (not used in the manuscript) ===========================
df <- read.csv("e2.csv")
df$Gradient <- factor(df$Gradient)
n <- nrow(df)
df$BBR_beta <- (df$BBR * (n - 1) + 0.5) / n   # Smithson & Verkuilen correction

## =========================
## 4. Fit beta regression model
## =========================
fit_beta <- betareg(BBR_beta ~ Gradient + ET, data = df)
summary(fit_beta)

# Fit full and nested models
fit_full   <- betareg(BBR_beta ~ Gradient + ET, data = df)
fit_noGrad <- betareg(BBR_beta ~ ET,            data = df)
fit_noET   <- betareg(BBR_beta ~ Gradient,      data = df)

# Test the overall effect of each factor
lrtest(fit_noGrad, fit_full)   # Overall effect of Gradient
lrtest(fit_noET,   fit_full)   # Overall effect of ET

## =========================
## 5. Estimated marginal means (response scale)
## =========================
adj_means <- emmeans(fit_beta, ~ Gradient, type = "response")
pairs(adj_means, adjust = "tukey")

plot_df <- as.data.frame(adj_means)

## =========================
## 6. Compute SE-based error bar bounds
## =========================
plot_df$se_lower <- plot_df$emmean - plot_df$SE
plot_df$se_upper <- plot_df$emmean + plot_df$SE

## =========================
## 7. Plot
## =========================
ggplot() +
  ## Raw data
  geom_jitter(
    data = df,
    aes(x = Gradient, y = BBR),
    width = 0.12,
    size = 1.6,
    alpha = 0.6,
    color = "grey40"
  ) +
  
  ## SE error bars
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
    size = 3,
    shape = 16,
    color = "red"
  ) +
  
  scale_y_continuous(
    limits = c(0, 0.8),
    expand = expansion(mult = c(0.05, 0.1))
  ) +
  
  labs(
    x = "N concentration (mM)",
    y = "BAS branching rate"
  ) +
  
  theme_classic(base_size = 14) +
  theme(
    axis.text  = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 14),
    axis.line  = element_line(color = "black", linewidth = 0.6),
    legend.position = "none"
  )

ggsave("Fig_BBR.pdf",
       width = 85, height = 85, units = "mm")


# 4. Meshedness ================================================================
## =========================
## 1. Load packages
## =========================
library(betareg)
library(emmeans)
library(multcomp)
library(ggplot2)
library(lmtest)

## =========================
## 2. Read data
## =========================
df <- read.csv("e2.csv")
df$Gradient <- factor(df$Gradient)

## =========================
## 3. Prepare response (0 < y < 1)
## =========================
n <- nrow(df)
df$MC2_beta <- (df$MC2 * (n - 1) + 0.5) / n   # Smithson & Verkuilen correction

## =========================
## 4. Fit beta regression model
## =========================
fit_beta <- betareg(MC2_beta ~ Gradient + ET, data = df)
summary(fit_beta)

# Fit full and nested models
fit_full   <- betareg(MC2_beta ~ Gradient + ET, data = df)
fit_noGrad <- betareg(MC2_beta ~ ET,            data = df)
fit_noET   <- betareg(MC2_beta ~ Gradient,      data = df)

# Test the overall effect of each factor
lrtest(fit_noGrad, fit_full)   # Overall effect of Gradient
lrtest(fit_noET,   fit_full)   # Overall effect of ET

## =========================
## 5. Estimated marginal means (response scale)
## =========================
adj_means <- emmeans(fit_beta, ~ Gradient, type = "response")
pairs(adj_means, adjust = "tukey")

plot_df <- as.data.frame(adj_means)

## =========================
## 6. Compute SE-based error bar bounds
## =========================
plot_df$se_lower <- plot_df$emmean - plot_df$SE
plot_df$se_upper <- plot_df$emmean + plot_df$SE

## =========================
## 7. Plot
## =========================
ggplot() +
  ## Raw data
  geom_jitter(
    data = df,
    aes(x = Gradient, y = MC2),
    width = 0.12,
    size = 1.6,
    alpha = 0.6,
    color = "grey40"
  ) +
  
  ## SE error bars
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
    size = 3,
    shape = 16,
    color = "red"
  ) +
  
  scale_y_continuous(
    limits = c(0, 0.6),
    breaks = seq(0, 0.6, by = 0.1),
    expand = expansion(mult = c(0.05, 0.15))
  ) +
  
  labs(
    x = "N concentration (mM)",
    y = "Meshedness coefficiency"
  ) +
  
  theme_classic(base_size = 14) +
  theme(
    axis.text  = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 14),
    axis.line  = element_line(color = "black", linewidth = 0.6),
    legend.position = "none"
  )

ggsave("Fig_MC2.pdf",
       width = 85, height = 85, units = "mm")


# 5. Network efficiency ========================================================
## =========================
## 1. Load packages
## =========================
library(betareg)
library(emmeans)
library(multcomp)
library(ggplot2)
library(lmtest)

## =========================
## 2. Read data
## =========================
df <- read.csv("e2.csv")
df$Gradient <- factor(df$Gradient)

## =========================
## 3. Prepare response (0 < y < 1)
## =========================
n <- nrow(df)
df$Reg2_beta <- (df$Reg2 * (n - 1) + 0.5) / n   # Smithson & Verkuilen correction

## =========================
## 4. Fit beta regression model
## =========================
fit_beta <- betareg(Reg2_beta ~ Gradient + ET, data = df)
summary(fit_beta)

# Fit full and nested models
fit_full   <- betareg(Reg2_beta ~ Gradient + ET, data = df)
fit_noGrad <- betareg(Reg2_beta ~ ET,            data = df)
fit_noET   <- betareg(Reg2_beta ~ Gradient,      data = df)

# Test the overall effect of each factor
lrtest(fit_noGrad, fit_full)   # Overall effect of Gradient
lrtest(fit_noET,   fit_full)   # Overall effect of ET

## =========================
## 5. Estimated marginal means (response scale)
## =========================
adj_means <- emmeans(fit_beta, ~ Gradient, type = "response")
pairs(adj_means, adjust = "tukey")

plot_df <- as.data.frame(adj_means)

## =========================
## 6. Compute SE-based error bar bounds
## =========================
plot_df$se_lower <- plot_df$emmean - plot_df$SE
plot_df$se_upper <- plot_df$emmean + plot_df$SE

## =========================
## 7. Plot
## =========================
ggplot() +
  ## Raw data
  geom_jitter(
    data = df,
    aes(x = Gradient, y = Reg2),
    width = 0.12,
    size = 1.6,
    alpha = 0.6,
    color = "grey40"
  ) +
  
  ## SE error bars
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
    size = 3,
    shape = 16,
    color = "red"
  ) +
  
  scale_y_continuous(
    limits = c(0, 0.6),
    breaks = seq(0, 0.6, by = 0.1),
    expand = expansion(mult = c(0.05, 0.15))
  ) +
  
  labs(
    x = "N concentration (mM)",
    y = "Relative network efficiency"
  ) +
  
  theme_classic(base_size = 14) +
  theme(
    axis.text  = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 14),
    axis.line  = element_line(color = "black", linewidth = 0.6),
    legend.position = "none"
  )

ggsave("Fig_Reg2.pdf",
       width = 85, height = 85, units = "mm")


# 6. BASr (not used in the manuscript)==========================================
## =========================
## 1. Load packages
## =========================
library(betareg)
library(emmeans)
library(multcomp)
library(ggplot2)
library(lmtest)

## =========================
## 2. Read data
## =========================
df <- read.csv("e2.csv")
df$Gradient <- factor(df$Gradient)

## =========================
## 3. Prepare response (0 < y < 1)
## =========================
n <- nrow(df)
df$BASr_beta <- (df$BASr * (n - 1) + 0.5) / n   # Smithson & Verkuilen correction

## =========================
## 4. Fit beta regression model
## =========================
fit_beta <- betareg(BASr_beta ~ Gradient + ET, data = df)
summary(fit_beta)

# Fit full and nested models
fit_full   <- betareg(BASr_beta ~ Gradient + ET, data = df)
fit_noGrad <- betareg(BASr_beta ~ ET,            data = df)
fit_noET   <- betareg(BASr_beta ~ Gradient,      data = df)

# Test the overall effect of each factor
lrtest(fit_noGrad, fit_full)   # Overall effect of Gradient
lrtest(fit_noET,   fit_full)   # Overall effect of ET

## =========================
## 5. Estimated marginal means (response scale)
## =========================
adj_means <- emmeans(fit_beta, ~ Gradient, type = "response")
pairs(adj_means, adjust = "tukey")

plot_df <- as.data.frame(adj_means)

## =========================
## 6. Compute SE-based error bar bounds
## =========================
plot_df$se_lower <- plot_df$emmean - plot_df$SE
plot_df$se_upper <- plot_df$emmean + plot_df$SE

## =========================
## 7. Plot
## =========================
ggplot() +
  ## Raw data
  geom_jitter(
    data = df,
    aes(x = Gradient, y = BASr),
    width = 0.12,
    size = 1.6,
    alpha = 0.6,
    color = "grey40"
  ) +
  
  ## SE error bars
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
    size = 3,
    shape = 16,
    color = "red"
  ) +
  
  scale_y_continuous(
    limits = c(0, 1),
    expand = expansion(mult = c(0.05, 0.15))
  ) +
  
  labs(
    x = "N concentration (mM)",
    y = "BAS rate"
  ) +
  
  theme_classic(base_size = 14) +
  theme(
    axis.text  = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 14),
    axis.line  = element_line(color = "black", linewidth = 0.6),
    legend.position = "none"
  )

ggsave("Fig_BASr.pdf",
       width = 85, height = 85, units = "mm")


## II. Angles ==========================================================================================================================
# 7. RH Branching angle (30-100) ===============================================
df2 <- read.csv("e1ra.csv")
df2$Gradient <- as.factor(df2$Gradient)   # IV should be a factor
df2$ET       <- as.numeric(df2$ET)        # Covariate must be numeric
df2$BA       <- as.numeric(df2$BA)        # DV must be numeric

# Remove missing values
df_clean <- na.omit(df2[, c("BA", "Gradient", "ET")])

# GLM (beta regression) - robust for N > 10,000 and angular data
library(glmmTMB)
library(emmeans)
library(multcomp)
library(ggplot2)

# Squeeze data to (0,1) to avoid boundary errors
n <- nrow(df_clean)
df_clean$BA_scaled <- ( (df_clean$BA - 29.9) / (100.1 - 29.9) * (n - 1) + 0.5 ) / n

## =========================
## Fit beta regression model
## =========================
model_beta <- glmmTMB(BA_scaled ~ Gradient + ET,
                      family = beta_family(link = "logit"),
                      data = df_clean)
summary(model_beta)

anova_results <- car::Anova(model_beta, type = "II")
print(anova_results)

## =========================
## Estimated marginal means (response scale)
## =========================
adj_means <- emmeans(model_beta, ~ Gradient, type = "response")
pairs(adj_means, adjust = "tukey")

plot_df <- as.data.frame(adj_means)

# Back-transformation function to degrees (30-100)
to_deg <- function(x) x * (100.1 - 29.9) + 29.9

# Compute SE bounds on the response (0,1) scale, then back-transform
plot_df$se_lower_resp <- plot_df$response - plot_df$SE
plot_df$se_upper_resp <- plot_df$response + plot_df$SE

plot_df$emmean_deg <- to_deg(plot_df$response)
plot_df$lower_deg  <- to_deg(plot_df$se_lower_resp)
plot_df$upper_deg  <- to_deg(plot_df$se_upper_resp)

## =========================
## Plot
## =========================
ggplot() +
  ## SE error bars
  geom_errorbar(
    data = plot_df,
    aes(x = Gradient, ymin = lower_deg, ymax = upper_deg),
    width = 0.1,
    linewidth = 1,
    color = "black"
  ) +
  
  ## Adjusted means
  geom_point(
    data = plot_df,
    aes(x = Gradient, y = emmean_deg),
    size = 3,
    shape = 16,
    color = "red"
  ) +
  
  scale_y_continuous(limits = c(65, 75), breaks = seq(65, 75, by = 5)) +
  labs(
    x = "N concentration (mM)",
    y = "RH branching angle (°)"
  ) +
  theme_classic(base_size = 14) +
  theme(
    axis.text  = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 14),
    axis.line  = element_line(color = "black", linewidth = 0.6),
    legend.position = "none"
  )

ggsave("Fig_RBA1.pdf",
       width = 85, height = 85, units = "mm")


# 8. BAS Branching angle =======================================================
df2 <- read.csv("e1ba.csv")
df2$Gradient <- as.factor(df2$Gradient)
df2$ET       <- as.numeric(df2$ET)
df2$BA       <- as.numeric(df2$BA)

# Remove missing values
df_clean <- na.omit(df2[, c("BA", "Gradient", "ET")])

library(glmmTMB)
library(emmeans)
library(multcomp)
library(ggplot2)

# Squeeze data to (0,1) to avoid boundary errors
n <- nrow(df_clean)
df_clean$BA_scaled <- ( (df_clean$BA - 29.9) / (100.1 - 29.9) * (n - 1) + 0.5 ) / n

## =========================
## Fit beta regression model
## =========================
model_beta <- glmmTMB(BA_scaled ~ Gradient + ET,
                      family = beta_family(link = "logit"),
                      data = df_clean)
summary(model_beta)

anova_results <- car::Anova(model_beta, type = "II")
print(anova_results)

## =========================
## Estimated marginal means (response scale)
## =========================
adj_means <- emmeans(model_beta, ~ Gradient, type = "response")
pairs(adj_means, adjust = "tukey")

plot_df <- as.data.frame(adj_means)

# Back-transformation function to degrees (30-100)
to_deg <- function(x) x * (100.1 - 29.9) + 29.9

# Compute SE bounds on the response (0,1) scale, then back-transform
plot_df$se_lower_resp <- plot_df$response - plot_df$SE
plot_df$se_upper_resp <- plot_df$response + plot_df$SE

plot_df$emmean_deg <- to_deg(plot_df$response)
plot_df$lower_deg  <- to_deg(plot_df$se_lower_resp)
plot_df$upper_deg  <- to_deg(plot_df$se_upper_resp)

## =========================
## Plot
## =========================
ggplot() +
  ## SE error bars
  geom_errorbar(
    data = plot_df,
    aes(x = Gradient, ymin = lower_deg, ymax = upper_deg),
    width = 0.1,
    linewidth = 1,
    color = "black"
  ) +
  
  ## Adjusted means
  geom_point(
    data = plot_df,
    aes(x = Gradient, y = emmean_deg),
    size = 3,
    shape = 16,
    color = "red"
  ) +
  
  scale_y_continuous(limits = c(65, 75), breaks = seq(65, 75, by = 5)) +
  labs(
    x = "N concentration (mM)",
    y = "BAS branching angle (°)"
  ) +
  theme_classic(base_size = 14) +
  theme(
    axis.text  = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 14),
    axis.line  = element_line(color = "black", linewidth = 0.6),
    legend.position = "none"
  )

ggsave("Fig_BBA1.pdf",
       width = 85, height = 85, units = "mm")


# 9. First ERH Branching angle (not used in the manuscript)=====================
df2 <- read.csv("e2.csv")
df2$Gradient <- as.factor(df2$Gradient)
df2$ET       <- as.numeric(df2$ET)
df2$FBA      <- as.numeric(df2$BA)

# Remove missing values
df_clean <- na.omit(df2[, c("FBA", "Gradient", "ET")])

library(glmmTMB)
library(emmeans)
library(multcomp)
library(ggplot2)

# Squeeze data to (0,1) to avoid boundary errors
n <- nrow(df_clean)
df_clean$FBA_scaled <- ( (df_clean$FBA - 29.9) / (100.1 - 29.9) * (n - 1) + 0.5 ) / n

## =========================
## Fit beta regression model
## =========================
model_beta <- glmmTMB(FBA_scaled ~ Gradient + ET,
                      family = beta_family(link = "logit"),
                      data = df_clean)
summary(model_beta)

anova_results <- car::Anova(model_beta, type = "II")
print(anova_results)

## =========================
## Estimated marginal means (response scale)
## =========================
adj_means <- emmeans(model_beta, ~ Gradient, type = "response")
pairs(adj_means, adjust = "tukey")

plot_df <- as.data.frame(adj_means)

# Back-transformation function to degrees (30-100)
to_deg <- function(x) x * (100.1 - 29.9) + 29.9

# Compute SE bounds on the response (0,1) scale, then back-transform
plot_df$se_lower_resp <- plot_df$response - plot_df$SE
plot_df$se_upper_resp <- plot_df$response + plot_df$SE

plot_df$emmean_deg <- to_deg(plot_df$response)
plot_df$lower_deg  <- to_deg(plot_df$se_lower_resp)
plot_df$upper_deg  <- to_deg(plot_df$se_upper_resp)

## =========================
## Plot with density shading (violin)
## =========================
ggplot() +
  ## Density shading
  geom_violin(
    data = df_clean,
    aes(x = Gradient, y = FBA, fill = Gradient),
    color = NA,
    alpha = 0.2,
    scale = "width"
  ) +
  
  ## SE error bars
  geom_errorbar(
    data = plot_df,
    aes(x = Gradient, ymin = lower_deg, ymax = upper_deg),
    width = 0.1,
    linewidth = 1,
    color = "black"
  ) +
  
  ## Adjusted means
  geom_point(
    data = plot_df,
    aes(x = Gradient, y = emmean_deg),
    size = 3,
    shape = 16,
    color = "red"
  ) +
  
  scale_fill_viridis_d(option = "mako", begin = 0.3, end = 0.9) +
  scale_y_continuous(limits = c(30, 110), breaks = seq(30, 100, by = 10)) +
  labs(
    x = "N concentration (mM)",
    y = "RH branching angle (°)"
  ) +
  theme_classic(base_size = 14) +
  theme(legend.position = "none")

ggsave("Fig_RBA.pdf",
       width = 85, height = 85, units = "mm")


## III. Continuous data ========================================================================================================
# 10. Distance of longest ERH ==================================================
library(emmeans)
library(multcomp)
library(ggplot2)

df <- read.csv("e1.csv")
df$Gradient <- factor(df$Gradient)

## =========================
## Fit linear model (ANCOVA)
## =========================
fit_iii <- lm(BD ~ Gradient + ET, data = df)
summary(fit_iii)
anova(fit_iii)

## =========================
## Estimated marginal means
## =========================
adj_means <- emmeans(fit_iii, specs = ~ Gradient)
pairs(adj_means, adjust = "tukey")

plot_df <- as.data.frame(adj_means)

## =========================
## Compute SE-based error bar bounds
## =========================
plot_df$se_lower <- plot_df$emmean - plot_df$SE
plot_df$se_upper <- plot_df$emmean + plot_df$SE

## =========================
## Plot
## =========================
df$Gradient       <- as.factor(df$Gradient)
plot_df$Gradient  <- as.factor(plot_df$Gradient)

ggplot() +
  ## Raw data
  geom_jitter(
    data = df,
    aes(x = Gradient, y = BD),
    width = 0.12,
    size = 1.6,
    alpha = 0.6,
    color = "grey40"
  ) +
  
  ## SE error bars
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
    size = 3,
    shape = 16,
    color = "red"
  ) +
  
  scale_y_continuous(
    breaks = seq(0, 40, by = 10),
    expand = expansion(mult = c(0.05, 0.15))
  ) +
  
  labs(
    x = "N concentration (mM)",
    y = "LERH branching distance (mm)"
  ) +
  
  theme_classic(base_size = 14) +
  theme(
    axis.text  = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 14),
    axis.line  = element_line(color = "black", linewidth = 0.6),
    legend.position = "none"
  )

ggsave("Fig_DLER.pdf",
       width = 85, height = 85, units = "mm")


# 11. Distance of first ERH ====================================================
library(emmeans)
library(multcomp)
library(ggplot2)

df <- read.csv("e1.csv")
df$Gradient <- factor(df$Gradient)

## =========================
## Fit linear model (ANCOVA)
## =========================
fit_iii <- lm(FBD ~ Gradient + ET, data = df)
summary(fit_iii)
anova(fit_iii)

## =========================
## Estimated marginal means
## =========================
adj_means <- emmeans(fit_iii, specs = ~ Gradient)
pairs(adj_means, adjust = "tukey")

plot_df <- as.data.frame(adj_means)

## =========================
## Compute SE-based error bar bounds
## =========================
plot_df$se_lower <- plot_df$emmean - plot_df$SE
plot_df$se_upper <- plot_df$emmean + plot_df$SE

## =========================
## Plot
## =========================
df$Gradient       <- as.factor(df$Gradient)
plot_df$Gradient  <- as.factor(plot_df$Gradient)

ggplot() +
  ## Raw data
  geom_jitter(
    data = df,
    aes(x = Gradient, y = FBD),
    width = 0.12,
    size = 1.6,
    alpha = 0.6,
    color = "grey40"
  ) +
  
  ## SE error bars
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
    size = 3,
    shape = 16,
    color = "red"
  ) +
  
  scale_y_continuous(
    breaks = seq(0, 20, by = 5),
    expand = expansion(mult = c(0.05, 0.15))
  ) +
  
  labs(
    x = "N concentration (mM)",
    y = "FERH branching distance (mm)"
  ) +
  
  theme_classic(base_size = 14) +
  theme(
    axis.text  = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 14),
    axis.line  = element_line(color = "black", linewidth = 0.6),
    legend.position = "none"
  )

ggsave("Fig_DFER.pdf",
       width = 85, height = 85, units = "mm")
