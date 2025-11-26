def get_parameters():
    return {

        # General config
        "start_year": 2007,
        "end_year": 2023,
        "target": "CASES",

        "source_raw": "/data/raw",
        "source_curated": "/data/curated",
        "source_conformed": "/data/conformed",
        "source_model": "/data/model",
        "source_results": "/data/results",

        # APIs
        "api_temperature": "https://www.datos.gov.co/resource/sbwg-7ju4.json",
        "api_humidity": "https://www.datos.gov.co/resource/uext-mhny.json",
        "api_pressure": "https://www.datos.gov.co/resource/62tk-nxj5.json",
        "api_rainfall": "https://www.datos.gov.co/resource/s54a-sgyg.json",

        # Feature selection
        "corr_method": "spearman",
        "non_parametric": False,
        "max_lag": 52,
        "top_corr_cases": 10,
        "top_corr_exog": 100,
        "corr_threshold": 0.8,
        "reg_metric": "r2",
        "reg_runs": 3,
        "coeff_cut": 5,

        # Modeling
        "validation_year": '2016-12-25',
        "test_year": '2020-06-28',
        "steps": 52,
    }