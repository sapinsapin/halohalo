<#
Run a resumable WSL job and start it again whenever WSL dies under it.

The workstation's WSL has been taken down repeatedly by CPU machine-check
exceptions (the kernel-panic logs in %LOCALAPPDATA%\Temp\wsl-crashes). Nothing
inside WSL survives a panic, so supervision lives on the Windows side. The job
must be resumable (every step skips finished work), which the local GPU queue
and the porting pipeline are.

  Start-Process -WindowStyle Hidden powershell -ArgumentList '-File','scripts\wsl_supervise.ps1','-Name','port1','-Command','cd /mnt/d/halohalo && bash scripts/port_phase1.sh'

A log of attempts goes to D:\halohalo\finetune_runs\supervise_<Name>.log.
#>
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Command,
    [int]$Tries = 8,
    [int]$WaitSeconds = 60
)
$log = "D:\halohalo\finetune_runs\supervise_$Name.log"
function Say($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content -Path $log -Encoding utf8 }

for ($i = 1; $i -le $Tries; $i++) {
    Say "attempt $i`: $Command"
    & wsl.exe -e bash -c $Command
    $rc = $LASTEXITCODE
    if ($rc -eq 0) { Say "finished (exit 0)"; exit 0 }
    $panic = Get-ChildItem "$env:LOCALAPPDATA\Temp\wsl-crashes\kernel-panic-*.txt" -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-10) } | Select-Object -First 1
    if ($panic) { Say "exit $rc after a WSL kernel panic ($($panic.Name)); restarting in $WaitSeconds s" }
    else { Say "exit $rc (no panic logged); restarting in $WaitSeconds s" }
    Start-Sleep -Seconds $WaitSeconds
}
Say "gave up after $Tries attempts"
exit 1
