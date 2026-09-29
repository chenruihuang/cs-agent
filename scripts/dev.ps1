param([Parameter(Position=0)][string]$cmd = "help")

switch ($cmd) {
  "up"      { docker-compose up -d }
  "build"   { docker-compose up -d --build }
  "down"    { docker-compose down }
  "logs"    { docker-compose logs -f }
  "ps"      { docker-compose ps }
  "db"      { docker-compose run --rm build-db }
  "eval"    { docker-compose run --rm eval }
  "rebuild" {
    docker-compose down
    docker-compose up -d --build
    docker-compose run --rm build-db
  }
  default {
    Write-Host "用法: .\scripts\dev.ps1 <命令>"
    Write-Host "  up / build / down / logs / ps / db / eval / rebuild"
  }
}
