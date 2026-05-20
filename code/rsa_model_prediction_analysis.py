#!/usr/bin/env python3
"""Write RSA stimulus-bin and model-RDM outputs."""

from rsa_model_predictions import run_rsa_model_predictions


if __name__ == "__main__":
    run_rsa_model_predictions(save_figures=False)
