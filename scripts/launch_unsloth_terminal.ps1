param()
$ErrorActionPreference = 'Stop'
$wt = Get-Command wt.exe -ErrorAction SilentlyContinue
$cmd = 'cd /mnt/s/AITraining/unsloth && source .venv/bin/activate && clear && echo "Unsloth env active at /mnt/s/AITraining/unsloth" && python --version && exec bash'
if ($wt) {
  wt.exe new-tab --title "Unsloth" wsl.exe -- bash -lc $cmd
} else {
  Start-Process wsl.exe -ArgumentList @('--','bash','-lc',$cmd)
}
