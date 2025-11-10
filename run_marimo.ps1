# Run Marimo in app mode with health monitoring disabled
$env:MARIMO_HEALTH_CHECK_ENABLED = "false"
$env:MARIMO_DISABLE_GPU_STATS = "true"

# Run the notebook in app mode
uv run marimo run notebooks/10_adc_scenario_budget.py
