param(
  [int]$Port = 5196
)

$ErrorActionPreference = 'Stop'

$repoWsl = '/mnt/c/projects/ai-training-lab/unsloth'

function Invoke-WslText {
  param([Parameter(Mandatory = $true)][string]$Command)
  $output = & wsl.exe -- bash -lc $Command 2>&1
  return [string]::Join("`n", @($output))
}

function Get-WslIp {
  $ip = (Invoke-WslText "hostname -I | awk '{print `$1}'").Trim()
  if (-not $ip) {
    throw 'Unable to determine the WSL IP address.'
  }
  return ($ip -split '\s+')[0]
}

function Test-HttpOk {
  param([Parameter(Mandatory = $true)][string]$Url)
  try {
    $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
    return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500)
  } catch {
    return $false
  }
}

function Get-PanelListenerLine {
  return (Invoke-WslText "ss -ltnp 'sport = :$Port' 2>/dev/null | tail -n +2").Trim()
}

function Stop-LoopbackOnlyPanelIfNeeded {
  $ip = Get-WslIp
  if (Test-HttpOk "http://${ip}:$Port/api/status") {
    return
  }

  $listener = Get-PanelListenerLine
  if (-not $listener -or $listener -notmatch '127\.0\.0\.1:' -or $listener -notmatch 'pid=([0-9]+)') {
    return
  }

  $serverPid = $Matches[1]
  $cwd = (Invoke-WslText "readlink -f /proc/$serverPid/cwd 2>/dev/null || true").Trim()
  $cmdline = (Invoke-WslText "tr '\0' ' ' < /proc/$serverPid/cmdline 2>/dev/null || true").Trim()
  if ($cwd -eq $repoWsl -and $cmdline -match 'control_panel\.server') {
    Write-Host "Restarting loopback-only AI Training panel on port $Port..."
    Invoke-WslText "kill $serverPid 2>/dev/null || true" | Out-Null
    Start-Sleep -Seconds 1
  }
}

Stop-LoopbackOnlyPanelIfNeeded

$ip = Get-WslIp
$serverProcess = $null

if (-not (Test-HttpOk "http://${ip}:$Port/api/status")) {
  $bashCommand = "cd $repoWsl && AI_TRAINING_UI_HOST=0.0.0.0 AI_TRAINING_ALLOW_WSL_HOST_BIND=1 AI_TRAINING_UI_PORT=$Port ./scripts/run_control_panel.sh"
  $serverProcess = Start-Process -FilePath 'wsl.exe' -ArgumentList @('--', 'bash', '-lc', $bashCommand) -WindowStyle Hidden -PassThru

  for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 500
    $ip = Get-WslIp
    if (Test-HttpOk "http://${ip}:$Port/api/status") {
      break
    }
  }
}

$ip = Get-WslIp
$statusUrl = "http://${ip}:$Port/api/status"
$uiUrl = "http://${ip}:$Port/"

if (-not (Test-HttpOk $statusUrl)) {
  throw "AI Training control panel did not become reachable at $statusUrl."
}

Write-Host "Opening AI Training / Unsloth control panel at $uiUrl"
Start-Process $uiUrl

if ($serverProcess) {
  Wait-Process -Id $serverProcess.Id
}
