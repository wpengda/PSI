library(psych)
library(lavaan)
library(Metrics)


# Vector of file paths
file_paths <- c(
  "../All_Data/human_normal.csv",
  "../All_Data//persona_Gemma2_9b_zero_shot.csv",
  "../All_Data//persona_Gemma2_27b_zero_shot.csv",
  "../All_Data//persona_GPT4o_mini_zero_shot.csv",
  "../All_Data//persona_GPT4o_zero_shot.csv",
  "../All_Data//persona_Llama3_8b_zero_shot.csv",
  "../All_Data//persona_Llama3_70b_zero_shot.csv",
  "../All_Data//persona_Mistral_7b_zero_shot.csv",
  "../All_Data//psi_Gemma2_9b_zero_shot.csv",
  "../All_Data//psi_Gemma2_27b_zero_shot.csv",
  "../All_Data//psi_GPT4o_mini_zero_shot.csv",
  "../All_Data//psi_GPT4o_zero_shot.csv",
  "../All_Data//psi_Llama3_8b_zero_shot.csv",
  "../All_Data//psi_Llama3_70b_zero_shot.csv",
  "../All_Data//psi_Mistral_7b_zero_shot.csv",
  "../All_Data//shape_Gemma2_9b_zero_shot.csv",
  "../All_Data//shape_Gemma2_27b_zero_shot.csv",
  "../All_Data//shape_GPT4o_mini_zero_shot.csv",
  "../All_Data//shape_GPT4o_zero_shot.csv",
  "../All_Data//shape_Llama3_8b_zero_shot.csv",
  "../All_Data//shape_Llama3_70b_zero_shot.csv",
  "../All_Data//shape_Mistral_7b_zero_shot.csv"
)

# Read each CSV into a named list of data frames
expanded_paths <- path.expand(file_paths)
data_list <- setNames(
  lapply(file_paths, read.csv),
  nm = tools::file_path_sans_ext(basename(file_paths))
)

# Check the names of the dataset
names(data_list)

data_list <- lapply(data_list, function(df) df[, 2:61])

# Rename columns to item1 to item60
data_list <- lapply(data_list, function(df) {
  colnames(df) <- paste0("item", 1:60)
  return(df)
})



# Reverse code
reverse_code_columns <- c(11, 16, 26, 31, 36, 51, 12, 17, 22, 37, 42, 47, 3, 8, 23, 28, 48, 58, 4, 9, 
                          24, 29, 44, 49, 5, 25, 30, 45, 50, 55)

data_list <- lapply(data_list, function(df) {
  df[ , reverse_code_columns] <- 6 - df[ , reverse_code_columns]
  return(df)
})






facets <- list(
  Sociability = c(1, 16, 31, 46),
  Assertiveness = c(6, 21, 36, 51),
  Energy_Level = c(11, 26, 41, 56),
  Compassion = c(2, 17, 32, 47),
  Respectfulness = c(7, 22, 37, 52),
  Trust = c(12, 27, 42, 57),
  Organization = c(3, 18, 33, 48), 
  Productiveness = c(8, 23, 38, 53),
  Responsibility = c(13, 28, 43, 58),
  Anxiety = c(4, 19, 34, 49),
  Depression = c(9, 24, 39, 54),
  Emotional_Volatility = c(14, 29, 44, 59),
  Intellectual_Curiosity = c(10, 25, 40, 55),
  Aesthetic_Sensitivity = c(5, 20, 35, 50), 
  Creative_Imagination = c(15, 30, 45, 60)
)

domains <- list(
  Extraversion = c(1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56),
  Agreeableness = c(2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57),
  Conscientiousness = c(3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58),
  Neuroticism = c(4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59),
  Openness = c(5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
)








########################################################### Cronbach's alpha    ###########################################################
calculate_alpha <- function(data) {
  # check variance
  if (any(apply(data, 2, var, na.rm = TRUE) == 0)) {
    return("NA - no variance")
  }
  
  # calculate Cronbach's alpha without checking for warnings
  alpha_result <- tryCatch({
    a_result <- psych::alpha(data)
    return(as.character(a_result$total$raw_alpha))
  }, error = function(e) {
    # deal with error
    return("NA - calculation error")
  })
  
  return(alpha_result)
}


# Calculate results
results <- data.frame(
  file = character(),
  level = character(),
  scale = character(),
  alpha = numeric(),
  stringsAsFactors = FALSE
)

for (file_name in names(data_list)) {
  df <- data_list[[file_name]]
  
  # Facet level
  for (facet in names(facets)) {
    items <- facets[[facet]]
    alpha_val <- calculate_alpha(df[, items])
    results <- rbind(results, data.frame(
      file = file_name,
      level = "facet",
      scale = facet,
      alpha = alpha_val
    ))
  }
  
  # Domain level
  for (domain in names(domains)) {
    items <- domains[[domain]]
    alpha_val <- calculate_alpha(df[, items])
    results <- rbind(results, data.frame(
      file = file_name,
      level = "domain",
      scale = domain,
      alpha = alpha_val
    ))
  }
}

# cronbach_alpha Save to CSV 
write.csv(results, "Output/cronbach_alpha_results.csv", row.names = FALSE)






############################################################ Model ###########################################################

tfm_models <- list(
  ext = "Sociability =~ item1+item16+item31+item46
         Assertiveness =~ item6+item21+item36+item51
         Energy_Level =~ item11+item26+item41+item56",
  agr = "Compassion =~ item2+item17+item32+item47
         Respectfulness =~ item7+item22+item37+item52
         Trust =~ item12+item27+item42+item57",
  con = "Organization =~ item3+item18+item33+item48
         Productiveness =~ item8+item23+item38+item53
         Responsibility =~ item13+item28+item43+item58",
  neu = "Anxiety =~ item4+item19+item34+item49
         Depression =~ item9+item24+item39+item54
         Emotional_Volatility =~ item14+item29+item44+item59",
  ope = "Intellectual_Curiosity =~ item10+item25+item40+item55
         Aesthetic_Sensitivity =~ item5+item20+item35+item50
         Creative_Imagination =~ item15+item30+item45+item60"
)

fit_indices <- c("chisq","df", "cfi", "tli", "rmsea","srmr_bentler_nomean")
fit_results <- loadings_results <- correlation_results <- data.frame()

for (file_name in names(data_list)) {
  df <- data_list[[file_name]]
  for (model_name in names(tfm_models)) {
    model_def <- tfm_models[[model_name]]
    fit <- tryCatch(lavaan::cfa(model_def, data = df, estimator = "ML", std.lv = TRUE), error = function(e) NULL)
    if (!is.null(fit) && lavInspect(fit, "converged")) {
      clean_file <- gsub("-", "_", file_name)
      assign(paste0("fit.", clean_file, ".", model_name), fit, envir = .GlobalEnv)
      measures <- lavaan::fitMeasures(fit, fit.measures = fit_indices)
      fit_results <- rbind(fit_results, data.frame(file = file_name, model = model_name, t(measures[fit_indices])))
      std <- standardizedsolution(fit)
      loadings_results <- rbind(loadings_results, data.frame(file = file_name, model = model_name,
                                                             item = std$rhs[std$op == "=~"],
                                                             loading = std$est.std[std$op == "=~"]))
      corrs <- std[std$op == "~~" & std$lhs != std$rhs, ]
      correlation_results <- rbind(correlation_results, data.frame(file = file_name, model = model_name,
                                                                   factor1 = corrs$lhs, factor2 = corrs$rhs,
                                                                   correlation = corrs$est.std))
    }
  }
}
write.csv(fit_results, "Output/TFM_factor_analysis_model_fit_results.csv", row.names = FALSE)
write.csv(loadings_results, "Output/TFM_factor_analysis_factor_loadings.csv", row.names = FALSE)
write.csv(correlation_results, "Output/TFM_factor_analysis_factor_correlations.csv", row.names = FALSE)






# =========  FFM CFA ========= #
ffm_model <- "Extraversion =~ Sociability + Assertiveness + Energy_Level
              Agreeableness =~ Compassion + Respectfulness + Trust
              Conscientiousness =~ Organization + Productiveness + Responsibility
              Neuroticism =~ Anxiety + Depression + Emotional_Volatility
              Openness =~ Intellectual_Curiosity + Aesthetic_Sensitivity + Creative_Imagination"

fit_results_ffm <- loadings_results_ffm <- correlation_results_ffm <- data.frame()

for (file_name in names(data_list)) {
  df <- data_list[[file_name]]
  facet_scores <- as.data.frame(sapply(facets, function(items) rowMeans(df[, items])))
  fit <- tryCatch(lavaan::cfa(ffm_model, data = facet_scores, estimator = "ML", std.lv = TRUE), error = function(e) NULL)
  if (!is.null(fit) && lavInspect(fit, "converged")) {
    clean_file <- gsub("-", "_", file_name)
    assign(paste0("fit.", clean_file, ".FFM"), fit, envir = .GlobalEnv)
    measures <- lavaan::fitMeasures(fit, fit.measures = fit_indices)
    fit_results_ffm <- rbind(fit_results_ffm, data.frame(file = file_name, model = "FFM", t(measures[fit_indices])))
    std <- standardizedsolution(fit)
    loadings_results_ffm <- rbind(loadings_results_ffm, data.frame(file = file_name, model = "FFM",
                                                                   item = std$rhs[std$op == "=~"],
                                                                   loading = std$est.std[std$op == "=~"]))
    corrs <- std[std$op == "~~" & std$lhs != std$rhs, ]
    correlation_results_ffm <- rbind(correlation_results_ffm, data.frame(file = file_name, model = "FFM",
                                                                         factor1 = corrs$lhs, factor2 = corrs$rhs,
                                                                         correlation = corrs$est.std))
  }
}
write.csv(fit_results_ffm, "Output/FFM_factor_analysis_model_fit_results.csv", row.names = FALSE)
write.csv(loadings_results_ffm, "Output/FFM_factor_analysis_factor_loadings.csv", row.names = FALSE)
write.csv(correlation_results_ffm, "Output/FFM_factor_analysis_factor_correlations.csv", row.names = FALSE)





########################################### TCC and MAE###########################################

# define each TFM facets
tfm_dimensions <- list(
  ext = c("Sociability", "Assertiveness", "Energy_Level"),
  agr = c("Compassion", "Respectfulness", "Trust"),
  con = c("Organization", "Productiveness", "Responsibility"),
  neu = c("Anxiety", "Depression", "Emotional_Volatility"),
  ope = c("Intellectual_Curiosity", "Aesthetic_Sensitivity", "Creative_Imagination")
)

# Set human as baseline
baseline_file <- "human_normal"
tfm_files <- setdiff(names(data_list), baseline_file)  # other models

# Get baseline factor loadings
tfm_baseline_loadings <- list()
for (m in names(tfm_dimensions)) {
  fit_var <- paste0("fit.", baseline_file, ".", m)
  if (exists(fit_var)) {
    baseline_fit <- get(fit_var)
    base_loadings <- lavInspect(baseline_fit, "std")$lambda
    tfm_baseline_loadings[[m]] <- base_loadings
  }
}

# Initialize result container
tfm_results <- data.frame(file = character(),
                          model = character(),
                          factor = character(),
                          tcc = numeric(),
                          mae = numeric(),
                          stringsAsFactors = FALSE)

# Calculate TCC and MAE
for (m in names(tfm_dimensions)) {
  dims <- tfm_dimensions[[m]]
  baseline_mat <- tfm_baseline_loadings[[m]]
  
  for (f in tfm_files) {
    fit_var_current <- paste0("fit.", f, ".", m)
    if (exists(fit_var_current)) {
      current_fit <- get(fit_var_current)
      current_loadings <- lavInspect(current_fit, "std")$lambda
      
      # TCC (extract diagonal)
      tcc_matrix <- factor.congruence(baseline_mat, current_loadings)
      tcc_diag <- diag(tcc_matrix)
      
      # MAE
      mae_values <- sapply(1:length(dims), function(i) {
        mae(baseline_mat[, i], current_loadings[, i])
      })
      
      # Add results per facet
      for (i in 1:length(dims)) {
        tfm_results <- rbind(tfm_results, data.frame(file = f,
                                                     model = m,
                                                     factor = dims[i],
                                                     tcc = tcc_diag[i],
                                                     mae = mae_values[i],
                                                     stringsAsFactors = FALSE))
      }
    }
  }
}

# Save to CSV
write.csv(tfm_results, "Output/TFM_TCC_MAE_results.csv", row.names = FALSE)






########################################### FFM TCC and MAE ###########################################

# FFM
ffm_dimensions <- c("Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness")

# FFM factor loading
ffm_baseline_var <- paste0("fit.", baseline_file, ".FFM")
if (exists(ffm_baseline_var)) {
  ffm_baseline_fit <- get(ffm_baseline_var)
  ffm_baseline_loadings <- lavInspect(ffm_baseline_fit, "std")$lambda
}

# initialize FFM
ffm_results <- data.frame(file = character(),
                          factor = character(),
                          tcc = numeric(),
                          mae = numeric(),
                          stringsAsFactors = FALSE)

# calculate TCC and MAE
ffm_files <- setdiff(names(data_list), baseline_file)
for (f in ffm_files) {
  ffm_fit_var <- paste0("fit.", f, ".FFM")
  if (exists(ffm_fit_var)) {
    current_ffm_fit <- get(ffm_fit_var)
    current_ffm_loadings <- lavInspect(current_ffm_fit, "std")$lambda
    
    # TCC
    tcc_matrix <- factor.congruence(ffm_baseline_loadings, current_ffm_loadings)
    tcc_diag <- diag(tcc_matrix)
    
    # MAE
    mae_ffm <- sapply(1:length(ffm_dimensions), function(i) {
      mae(ffm_baseline_loadings[, i], current_ffm_loadings[, i])
    })
    
    # results
    for (i in 1:length(ffm_dimensions)) {
      ffm_results <- rbind(ffm_results, data.frame(file = f,
                                                   factor = ffm_dimensions[i],
                                                   tcc = tcc_diag[i],
                                                   mae = mae_ffm[i],
                                                   stringsAsFactors = FALSE))
    }
  }
}

# save to CSV
write.csv(ffm_results, "Output/FFM_TCC_MAE_results.csv", row.names = FALSE)



















