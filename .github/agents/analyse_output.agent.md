## Runtime Environment (Gadi)
For Python scripts in this repository, use:

```bash
module use /g/data/xp65/public/modules
module load conda/analysis3-25.08
```

## Execution Guidance
- For script debugging, first try reading files through the native vscode IDE rather than using sed in the terminal. 
- When running Python checks, plotting scripts, and analysis utilities in this module environment.
- Prefer matching the environment used by PBS job scripts in `new_run_analysis/`.
- If modules are unavailable in the current shell, report that limitation and still run syntax-only checks when possible.
