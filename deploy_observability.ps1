#Requires -Version 5.1
param(
  [string]$ProjectRoot = "C:\Users\Anant Sharma\Music\Linear_Regression",
  [int]$PrometheusPort = 9090,
  [int]$GrafanaPort = 3000,
  [int]$MetricsPort = 8000
)

Write-Host "Starting ML observability stack..." -ForegroundColor Cyan

# Paths
$PromConfig = Join-Path $ProjectRoot "prometheus.yml"
$GrafanaProv = Join-Path $ProjectRoot "grafana\provisioning"
$GrafanaDash = Join-Path $ProjectRoot "grafana\dashboards"
$ScriptPath = Join-Path $ProjectRoot "LinearRegression.py"

# Verify files
if (!(Test-Path $PromConfig)) { throw "prometheus.yml not found at $PromConfig" }
if (!(Test-Path $GrafanaProv)) { throw "Grafana provisioning folder missing at $GrafanaProv" }
if (!(Test-Path $GrafanaDash)) { throw "Grafana dashboards folder missing at $GrafanaDash" }
if (!(Test-Path $ScriptPath)) { throw "Training script not found at $ScriptPath" }

# Create network if missing
$netName = "ml-observability"
$existingNet = docker network ls --format '{{.Name}}' | Select-String -Pattern "^$netName$"
if (-not $existingNet) {
  docker network create $netName | Out-Null
}

# Stop and remove existing containers if present
foreach ($name in @('prometheus','grafana')) {
  $exists = docker ps -a --format '{{.Names}}' | Select-String -Pattern "^$name$"
  if ($exists) {
    docker rm -f $name | Out-Null
  }
}

# Start Prometheus
docker run -d --name prometheus --network $netName -p "${PrometheusPort}:9090" -v "${PromConfig}:/etc/prometheus/prometheus.yml" prom/prometheus:latest --config.file=/etc/prometheus/prometheus.yml | Out-Null

# Start Grafana
docker run -d --name grafana --network $netName -p "${GrafanaPort}:3000" -v "${GrafanaProv}:/etc/grafana/provisioning" -v "${GrafanaDash}:/var/lib/grafana/dashboards" -e GF_PATHS_PROVISIONING=/etc/grafana/provisioning grafana/grafana:latest | Out-Null

Write-Host "Prometheus UI: http://localhost:$PrometheusPort" -ForegroundColor Green
Write-Host "Grafana UI:    http://localhost:$GrafanaPort (admin/admin)" -ForegroundColor Green

# Run training script (make sure your venv is activated)
Push-Location $ProjectRoot
python .\LinearRegression.py
Pop-Location

Write-Host "Done. Verify metrics on Prometheus and dashboard on Grafana." -ForegroundColor Cyan