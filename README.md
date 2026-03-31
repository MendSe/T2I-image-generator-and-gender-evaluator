# T2I Image Generator and Gender Evaluator

An automated experimentation pipeline for studying gender bias in text-to-image (T2I) models.  
This project generates portrait images for profession-based prompts using **DALL·E 2** and **SDXL**, then classifies the **visually perceived gender** of the main subject using a vision-capable LLM, and finally analyzes the results statistically and visually.

## Overview

The goal of this project is to measure whether T2I models reproduce or amplify gender stereotypes when asked to generate images for professions associated with:
- male-dominated fields
- female-dominated fields
- balanced or mixed professions
- a neutral baseline prompt

The pipeline:
1. generates images from profession prompts using multiple T2I models
2. classifies the perceived gender in each generated image as **Male**, **Female**, or **Ambiguous**
3. stores all experiment outputs in a structured CSV file
4. computes summary statistics and bias indicators
5. produces plots and deeper analysis reports

## Features

- Automated image generation with **DALL·E 2** and **SDXL**
- Profession-based prompt set grouped by stereotype category
- Vision-based gender classification of generated outputs
- Checkpointed experiment execution with CSV saving
- Statistical analysis with contingency tables and chi-square testing
- Visualization of gender distributions by model and profession category
- Extended reporting for deeper bias patterns across models

## Project Structure

```bash
.
├── run_experiment.py        # Main generation + evaluation pipeline
├── analyze_results.py       # Statistical analysis and plots
├── advanced_analysis.py     # Extended insights and report generation
├── requirements.txt         # Python dependencies
├── setup.sh                 # Environment setup helper
└── output/                  # Generated images, CSV results, plots
