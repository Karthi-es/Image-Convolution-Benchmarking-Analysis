# Thread scalability evaluation (PowerShell)
# Mirrors thread_scalability_evaluation.sh but for Windows/PowerShell.
# It loops over thread counts and conv versions, sets env vars, runs the Python
# benchmark, and appends results to a CSV.

[CmdletBinding()]
param(
    [string]$PythonScript = "thread_scalability_evaluation.py",
    [string]$OutputFile = "thread_scalability_evaluation_result.csv",
    [int[]]$Threads = @(1, 2, 4, 8, 16, 32, 64),
    [string[]]$Versions = @(
        # "version1_0",
        # "version2_0",
        "version3_0",
        "version4_0_im2col",
        "version4_0_im2col_lib_baseline",
        "version4_0_winograd",
        "version4_0_fft"
    )
)

# Resolve paths relative to this script location
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

# Choose Python (prefer venv if present)
$venvPython = Join-Path $here ".venv/Scripts/python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
} else {
    $python = "python"
}

Write-Host "Python: $python"
Write-Host "Script: $PythonScript"
Write-Host "Output: $OutputFile"
Write-Host "Versions: $($Versions -join ', ')"
Write-Host "Threads: $($Threads -join ', ')"

# Initialize CSV header (the Python script prints: Version, Threads, Runtime_s)
"Version, Threads, Runtime_s" | Out-File -FilePath $OutputFile -Encoding UTF8

foreach ($t in $Threads) {
    # Set threading-related env vars for this process (inherited by child python)
    $env:NUMBA_NUM_THREADS = "$t"
    $env:MKL_NUM_THREADS = "$t"
    $env:OPENBLAS_NUM_THREADS = "$t"
    $env:OMP_NUM_THREADS = "$t"

    foreach ($v in $Versions) {
        Write-Host "Running $v with $t threads..." -ForegroundColor Cyan
        try {
            # Run the Python script: it prints a CSV line "version, threads, runtime"
            $result = & $python $PythonScript $v
            $exit = $LASTEXITCODE
            if ($exit -ne 0 -or [string]::IsNullOrWhiteSpace($result)) {
                Write-Warning "Run failed for $v with $t threads (exit $exit)."
                "$v, $t, ERROR" | Out-File -FilePath $OutputFile -Append -Encoding UTF8
            } else {
                # Append the exact output line from python (already formatted)
                $line = ($result -split "`n")[0].Trim()
                $line | Out-File -FilePath $OutputFile -Append -Encoding UTF8
                Write-Host "Result: $line"
            }
        }
        catch {
            Write-Warning "Exception running $v with $t threads: $($_.Exception.Message)"
            "$v, $t, ERROR" | Out-File -FilePath $OutputFile -Append -Encoding UTF8
        }
    }
}

Write-Host "-------------------------------------"
Write-Host "Scaling test complete. Results saved to $OutputFile"
Write-Host "The CSV format is: Version, Threads, Runtime_s"
