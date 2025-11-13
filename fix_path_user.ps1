# fix_path_user.ps1
# Backup and sanitize machine+user PATH, then set User PATH permanently.
$backupDir = "D:\Uni\Fudecom\Trabalho Extra\path_backups"
if (!(Test-Path $backupDir)) { New-Item -ItemType Directory -Path $backupDir | Out-Null }

$machine = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment' -Name Path -ErrorAction SilentlyContinue).Path
$user = (Get-ItemProperty 'HKCU:\Environment' -Name Path -ErrorAction SilentlyContinue).Path

# Save backups
Set-Content -Path (Join-Path $backupDir 'machine_path_backup.txt') -Value $machine -Force
Set-Content -Path (Join-Path $backupDir 'user_path_backup.txt') -Value $user -Force

# Combine and sanitize
$combined = ($machine + ';' + $user)
$parts = $combined -split ';' | ForEach-Object { $_.Trim().Trim('"') } | Where-Object { $_ -and $_ -ne '%PATH%' -and $_ -ne 'True' } | Select-Object -Unique

# Ensure essential system entries exist
$essentials=@('C:\Windows\System32','C:\Windows','C:\Windows\System32\Wbem','C:\Windows\System32\WindowsPowerShell\v1.0','C:\Windows\System32\OpenSSH')
foreach($e in $essentials) { if (-not ($parts -contains $e)) { $parts = ,$e + $parts } }

# Prefer common Git path if present
if (Test-Path 'C:\Program Files\Git\cmd') { if (-not ($parts -contains 'C:\Program Files\Git\cmd')) { $parts += 'C:\Program Files\Git\cmd' } }

# Final cleanup: remove empties and duplicates
$parts = $parts | Where-Object { $_ } | Select-Object -Unique
$newPath = $parts -join ';'

# Write permanently to User PATH
[Environment]::SetEnvironmentVariable('Path', $newPath, 'User')

Write-Output "WROTE NEW USER PATH (backups in $backupDir):"
Write-Output $newPath
Write-Output "NOTE: Feche/abra terminais ou faça logoff/login para aplicar as mudanças."