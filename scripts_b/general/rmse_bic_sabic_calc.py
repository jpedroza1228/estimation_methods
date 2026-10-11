import pandas as pd
import numpy as np
from great_tables import GT as gt

def generate_fit_statistics(ppc_df, fitdf, df_responses, alpha):

    # 1. Calculate RMSE based on marginal item pass rates
    rmse = np.sqrt(((ppc_df['observed_p'] - ppc_df['predicted_p'])**2).mean())

    # 2. Extract maximized log-likelihood from Stan's L-BFGS output
    log_lik = fitdf['lp__'].iloc[0]

    # 3. Define sample size (n) and estimated parameters (k)
    # k = guessing (I) + slipping (I) + latent class proportions (C - 1)
    n = df_responses.shape[0]
    I = df_responses.shape[1]
    C = alpha.shape[0]
    k = (2 * I) + (C - 1)

    # 4. Compute Information Criteria
    bic = -2 * log_lik + k * np.log(n)
    sabic = -2 * log_lik + k * np.log((n + 2) / 24)

    # 5. Construct the metrics DataFrame
    fit_metrics = pd.DataFrame({
        'Fit Index': [
            'RMSE (Item Pass Rates)', 
            'BIC', 
            'Sample-Size Adjusted BIC (SABIC)'
        ],
        'Value': [rmse, bic, sabic]
    })

    # Force string formatting to maintain exactly 4 decimal places (prevents stripping trailing zeros)
    fit_metrics['Value'] = fit_metrics['Value'].apply(lambda x: f"{x:.4f}")

    # 6. Generate the formatted table
    return (
        gt(fit_metrics)
        .tab_header(
            title='Global Model Fit Statistics',
            subtitle='Calculated via maximized log-posterior and marginal item probabilities'
        )
        .opt_table_font(font="Arial")
    )

# Example execution if run as a standalone script:
if __name__ == "__main__":
    # Assuming ppc_df, fitdf, df_responses, and alpha are already loaded in memory
    # fit_table = generate_fit_statistics(ppc_df, fitdf, df_responses, alpha)
    # fit_table.show()
    pass
