## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
    collapse = TRUE,
    comment = "#>"
)
options(rmarkdown.html_vignette.check_title = FALSE)

## ---- include = FALSE---------------------------------------------------------
### Don't overload CRAN servers
### https://stackoverflow.com/questions/28961431/computationally-heavy-r-vignettes
is_check <- ("CheckExEnv" %in% search()) || any(c("_R_CHECK_TIMINGS_",
             "_R_CHECK_LICENSE_") %in% names(Sys.getenv()))

## ---- message=FALSE-----------------------------------------------------------
library(Matrix)
library(MatrixExtra)
library(data.table)
library(kableExtra)
library(recommenderlab)
library(cmfrec)
library(recometrics)

data(MovieLense)
X_raw <- MovieLense@data

### Converting it to implicit-feedback
X_implicit <- as.coo.matrix(filterSparse(X_raw, function(x) x >= 4))
str(X_implicit)

## -----------------------------------------------------------------------------
reco_split <- create.reco.train.test(
    X_implicit,
    users_test_fraction = NULL,
    max_test_users = 100,
    items_test_fraction = 0.3,
    seed = 123
)
X_train <- reco_split$X_train ## Train data for test users
X_test <- reco_split$X_test ## Test data for test users
X_rem <- reco_split$X_rem ## Data to fit the model
users_test <- reco_split$users_test ## IDs of the test users

## -----------------------------------------------------------------------------
### Random recommendations (random latent factors)
set.seed(123)
UserFactors_random <- matrix(rnorm(nrow(X_test) * 5), nrow=5)
ItemFactors_random <- matrix(rnorm(ncol(X_test) * 5), nrow=5)

### Non-personalized recommendations
model_baseline <- cmfrec::MostPopular(as.coo.matrix(X_rem), implicit=TRUE)
item_biases <- model_baseline$matrices$item_bias

## ---- eval=FALSE--------------------------------------------------------------
#  ### Typical implicit-feedback ALS model
#  ### a.k.a. "WRMF" (weighted regularized matrix factorization)
#  model_wrmf <- cmfrec::CMF_implicit(as.coo.matrix(X_rem), k=10, verbose=FALSE)
#  UserFactors_wrmf <- t(cmfrec::factors(model_wrmf, X_train))
#  
#  ### As a comparison, this is the typical explicit-feedback model,
#  ### implemented by software such as Spark,
#  ### and called "Weighted-Lambda-Regularized Matrix Factorization".
#  ### Note that it determines the user factors using the train+test data.
#  model_wlr <- cmfrec::CMF(as.coo.matrix(X_raw[-users_test, ]),
#                           lambda=0.1, scale_lam=TRUE,
#                           user_bias=FALSE, item_bias=FALSE,
#                           k=10, verbose=FALSE)
#  UserFactors_wlr <- t(cmfrec::factors(model_wlr, as.csr.matrix(X_raw)[users_test,]))
#  
#  ### This is a different explicit-feedback model which
#  ### uses the same regularization for each user and item
#  ### (as opposed to the "weighted-lambda" model) and which
#  ### adds "implicit features", which are a binarized version
#  ### of the input data, but without weights.
#  ### Note that it determines the user factors using the train+test data.
#  model_hybrid <- cmfrec::CMF(as.coo.matrix(X_raw[-users_test, ]),
#                              lambda=20, scale_lam=FALSE,
#                              user_bias=FALSE, item_bias=FALSE,
#                              add_implicit_features=TRUE,
#                              k=10, verbose=FALSE)
#  UserFactors_hybrid <- t(cmfrec::factors(model_hybrid, as.csr.matrix(X_raw)[users_test,]))

## ---- echo=FALSE--------------------------------------------------------------
### Don't overload CRAN servers
if (!is_check) {
    model_wrmf <- cmfrec::CMF_implicit(as.coo.matrix(X_rem), k=10, verbose=FALSE)
    UserFactors_wrmf <- t(cmfrec::factors(model_wrmf, X_train))
    
    model_wlr <- cmfrec::CMF(as.coo.matrix(X_raw[-users_test, ]),
                             lambda=0.1, scale_lam=TRUE,
                             user_bias=FALSE, item_bias=FALSE,
                             k=10, verbose=FALSE)
    UserFactors_wlr <- t(cmfrec::factors(model_wlr, as.csr.matrix(X_raw)[users_test,]))
    
    model_hybrid <- cmfrec::CMF(as.coo.matrix(X_raw[-users_test, ]),
                                lambda=20, scale_lam=FALSE,
                                user_bias=FALSE, item_bias=FALSE,
                                add_implicit_features=TRUE,
                                k=10, verbose=FALSE)
    UserFactors_hybrid <- t(cmfrec::factors(model_hybrid, as.csr.matrix(X_raw)[users_test,]))
} else {
    model_wrmf <- cmfrec::CMF_implicit(as.coo.matrix(X_rem), k=3,
                                       verbose=FALSE, niter=2, nthreads=1)
    UserFactors_wrmf <- t(cmfrec::factors(model_wrmf, X_train))
}

## ---- eval=!is_check----------------------------------------------------------
### Processing user side information
U <- as.data.table(MovieLenseUser)[-users_test, ]
mean_age <- mean(U$age)
sd_age <- sd(U$age)
levels_occ <- levels(U$occupation)
MatrixExtra::restore_old_matrix_behavior()
process.U <- function(U, mean_age,sd_age, levels_occ) {
    U[, `:=`(
        id = NULL,
        age = (age - mean_age) / sd_age,
        sex = as.numeric(sex == "M"),
        occupation =  factor(occupation, levels_occ),
        zipcode = NULL
    )]
    U <- Matrix::sparse.model.matrix(~.-1, data=U)
    U <- as.coo.matrix(U)
    return(U)
}
U <- process.U(U, mean_age,sd_age, levels_occ)
U_train <- as.data.table(MovieLenseUser)[users_test, ]
U_train <- process.U(U_train, mean_age,sd_age, levels_occ)

### Processing item side information
I <- as.data.table(MovieLenseMeta)
mean_year <- mean(I$year, na.rm=TRUE)
sd_year <- sd(I$year, na.rm=TRUE)
I[
    is.na(year), year := mean_year
][, `:=`(
    title = NULL,
    year = (year - mean_year) / sd_year,
    url = NULL
)]
I <- as.coo.matrix(I)

### Manually re-creating a binarized matrix and weights
### that will mimic the WRMF model
X_rem_ones <- as.coo.matrix(mapSparse(X_rem, function(x) rep(1, length(x))))
W_rem <- as.coo.matrix(mapSparse(X_rem, function(x) x+1))
X_train_ones <- as.coo.matrix(mapSparse(X_train, function(x) rep(1, length(x))))
W_train <- as.coo.matrix(mapSparse(X_train, function(x) x+1))

## ---- eval=!is_check----------------------------------------------------------
### WRMF model, but with item biases/intercepts
model_bwrmf <- cmfrec::CMF(X_rem_ones, weight=W_rem, NA_as_zero=TRUE,
                           lambda=1, scale_lam=FALSE,
                           center=FALSE, user_bias=FALSE, item_bias=TRUE,
                           k=10, verbose=FALSE)
UserFactors_bwrmf <- t(cmfrec::factors(model_bwrmf, X_train_ones, weight=W_train))


### Collective WRMF model (taking user and item attributes)
model_cwrmf <- cmfrec::CMF_implicit(as.coo.matrix(X_rem), U=U, I=I,
                                    NA_as_zero_user=TRUE, NA_as_zero_item=TRUE,
                                    center_U=TRUE, center_I=TRUE,
                                    lambda=0.1,
                                    k=10, verbose=FALSE)
UserFactors_cwrmf <- t(cmfrec::factors(model_cwrmf, X_train, U=U_train))

### Collective WRMF plus item biases/intercepts
model_bcwrmf <- cmfrec::CMF(X_rem_ones, weight=W_rem, NA_as_zero=TRUE,
                            U=U, I=I, center_U=FALSE, center_I=FALSE,
                            NA_as_zero_user=TRUE, NA_as_zero_item=TRUE,
                            lambda=0.1, scale_lam=FALSE,
                            center=FALSE, user_bias=FALSE, item_bias=TRUE,
                            k=10, verbose=FALSE)
UserFactors_bcwrmf <- t(cmfrec::factors(model_bcwrmf, X_train_ones,
                                        weight=W_train, U=U_train))

## ---- eval=FALSE--------------------------------------------------------------
#  k <- 5 ## Top-K recommendations to evaluate
#  
#  ### Baselines
#  metrics_random <- calc.reco.metrics(X_train, X_test,
#                                      A=UserFactors_random,
#                                      B=ItemFactors_random,
#                                      k=k, all_metrics=TRUE)
#  metrics_baseline <- calc.reco.metrics(X_train, X_test,
#                                        A=NULL, B=NULL,
#                                        item_biases=item_biases,
#                                        k=k, all_metrics=TRUE)
#  
#  ### Simple models
#  metrics_wrmf <- calc.reco.metrics(X_train, X_test,
#                                    A=UserFactors_wrmf,
#                                    B=model_wrmf$matrices$B,
#                                    k=k, all_metrics=TRUE)
#  metrics_wlr <- calc.reco.metrics(X_train, X_test,
#                                   A=UserFactors_wlr,
#                                   B=model_wlr$matrices$B,
#                                   k=k, all_metrics=TRUE)
#  metrics_hybrid <- calc.reco.metrics(X_train, X_test,
#                                      A=UserFactors_hybrid,
#                                      B=model_hybrid$matrices$B,
#                                      k=k, all_metrics=TRUE)
#  
#  ### More complex models
#  metrics_bwrmf <- calc.reco.metrics(X_train, X_test,
#                                     A=UserFactors_bwrmf,
#                                     B=model_bwrmf$matrices$B,
#                                     item_biases=model_bwrmf$matrices$item_bias,
#                                     k=k, all_metrics=TRUE)
#  metrics_cwrmf <- calc.reco.metrics(X_train, X_test,
#                                     A=UserFactors_cwrmf,
#                                     B=model_cwrmf$matrices$B,
#                                     k=k, all_metrics=TRUE)
#  metrics_bcwrmf <- calc.reco.metrics(X_train, X_test,
#                                      A=UserFactors_bcwrmf,
#                                      B=model_bcwrmf$matrices$B,
#                                      item_biases=model_bcwrmf$matrices$item_bias,
#                                      k=k, all_metrics=TRUE)

## ---- echo=FALSE--------------------------------------------------------------
if (!is_check) {
    k <- 5 ## Top-K recommendations to evaluate

    ### Baselines
    metrics_random <- calc.reco.metrics(X_train, X_test,
                                        A=UserFactors_random,
                                        B=ItemFactors_random,
                                        k=k, all_metrics=TRUE)
    metrics_baseline <- calc.reco.metrics(X_train, X_test,
                                          A=NULL, B=NULL,
                                          item_biases=item_biases,
                                          k=k, all_metrics=TRUE)
    
    ### Simple models
    metrics_wrmf <- calc.reco.metrics(X_train, X_test,
                                      A=UserFactors_wrmf,
                                      B=model_wrmf$matrices$B,
                                      k=k, all_metrics=TRUE)
    metrics_wlr <- calc.reco.metrics(X_train, X_test,
                                     A=UserFactors_wlr,
                                     B=model_wlr$matrices$B,
                                     k=k, all_metrics=TRUE)
    metrics_hybrid <- calc.reco.metrics(X_train, X_test,
                                        A=UserFactors_hybrid,
                                        B=model_hybrid$matrices$B,
                                        k=k, all_metrics=TRUE)
    
    ### More complex models
    metrics_bwrmf <- calc.reco.metrics(X_train, X_test,
                                       A=UserFactors_bwrmf,
                                       B=model_bwrmf$matrices$B,
                                       item_biases=model_bwrmf$matrices$item_bias,
                                       k=k, all_metrics=TRUE)
    metrics_cwrmf <- calc.reco.metrics(X_train, X_test,
                                       A=UserFactors_cwrmf,
                                       B=model_cwrmf$matrices$B,
                                       k=k, all_metrics=TRUE)
    metrics_bcwrmf <- calc.reco.metrics(X_train, X_test,
                                        A=UserFactors_bcwrmf,
                                        B=model_bcwrmf$matrices$B,
                                        item_biases=model_bcwrmf$matrices$item_bias,
                                        k=k, all_metrics=TRUE)
} else {
    k <- 5 ## Top-K recommendations to evaluate

    ### Baselines
    metrics_random <- calc.reco.metrics(X_train, X_test,
                                        A=UserFactors_random,
                                        B=ItemFactors_random,
                                        k=k, all_metrics=TRUE,
                                        nthreads=1)
    metrics_baseline <- calc.reco.metrics(X_train, X_test,
                                          A=NULL, B=NULL,
                                          item_biases=item_biases,
                                          k=k, all_metrics=TRUE,
                                          nthreads=1)
    
    ### Simple models
    metrics_wrmf <- calc.reco.metrics(X_train, X_test,
                                      A=UserFactors_wrmf,
                                      B=model_wrmf$matrices$B,
                                      k=k, all_metrics=TRUE,
                                      nthreads=1)
}

## -----------------------------------------------------------------------------
metrics_baseline %>%
    head(5) %>%
    kable() %>%
    kable_styling()

## ---- eval=FALSE--------------------------------------------------------------
#  all_metrics <- list(
#      `Random` = metrics_random,
#      `Non-personalized` = metrics_baseline,
#      `Weighted-Lambda` = metrics_wlr,
#      `Hybrid-Explicit` = metrics_hybrid,
#      `WRMF (a.k.a. iALS)` = metrics_wrmf,
#      `bWRMF` = metrics_bwrmf,
#      `CWRMF` = metrics_cwrmf,
#      `bCWRMF` = metrics_bcwrmf
#  )
#  results <- all_metrics %>%
#      lapply(function(df) as.data.table(df)[, lapply(.SD, mean)]) %>%
#      data.table::rbindlist() %>%
#      as.data.frame()
#  row.names(results) <- names(all_metrics)
#  
#  results %>%
#      kable() %>%
#      kable_styling()

## ---- echo=FALSE--------------------------------------------------------------
if (!is_check) {
    all_metrics <- list(
        `Random` = metrics_random,
        `Non-personalized` = metrics_baseline,
        `Weighted-Lambda` = metrics_wlr,
        `Hybrid-Explicit` = metrics_hybrid,
        `WRMF (a.k.a. iALS)` = metrics_wrmf,
        `bWRMF` = metrics_bwrmf,
        `CWRMF` = metrics_cwrmf,
        `bCWRMF` = metrics_bcwrmf
    )
} else {
    all_metrics <- list(
        `Random` = metrics_random,
        `Non-personalized` = metrics_baseline,
        `WRMF (a.k.a. iALS)` = metrics_wrmf
    )
}
results <- all_metrics %>%
    lapply(function(df) as.data.table(df)[, lapply(.SD, mean)]) %>%
    data.table::rbindlist() %>%
    as.data.frame()
row.names(results) <- names(all_metrics)
results %>%
    kable() %>%
    kable_styling()

