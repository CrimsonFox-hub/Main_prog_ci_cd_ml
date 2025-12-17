# setup_windows.ps1 - Скрипт настройки окружения Windows для MLOps проекта

Write-Host "=== Настройка MLOps окружения для Windows ===" -ForegroundColor Green
Write-Host ""

# Проверка прав администратора
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Запустите скрипт от имени администратора!" -ForegroundColor Red
    exit 1
}

# Функция проверки установки программы
function Test-Command {
    param($command)
    $null = Get-Command $command -ErrorAction SilentlyContinue
    return $?
}

# Проверка установленного ПО
Write-Host "Проверка установленного ПО..." -ForegroundColor Yellow

$requiredSoftware = @(
    @{Name="Python"; Command="python"; Version="--version"; MinVersion="3.9"},
    @{Name="Git"; Command="git"; Version="--version"},
    @{Name="Docker"; Command="docker"; Version="--version"},
    @{Name="Make"; Command="make"; Version="--version"}
)

foreach ($software in $requiredSoftware) {
    if (Test-Command $software.Command) {
        try {
            $version = Invoke-Expression "$($software.Command) $($software.Version)" 2>&1
            Write-Host "✓ $($software.Name): $version" -ForegroundColor Green
        } catch {
            Write-Host "✓ $($software.Name) установлен" -ForegroundColor Green
        }
    } else {
        Write-Host "✗ $($software.Name) не установлен" -ForegroundColor Red
    }
}

Write-Host ""

# Установка недостающего ПО
$installChoices = @()
if (-not (Test-Command "python")) {
    $installChoices += "Python"
}
if (-not (Test-Command "git")) {
    $installChoices += "Git"
}
if (-not (Test-Command "docker")) {
    $installChoices += "Docker"
}
if (-not (Test-Command "make")) {
    $installChoices += "Make"
}

if ($installChoices.Count -gt 0) {
    Write-Host "Необходимо установить:" -ForegroundColor Yellow
    $installChoices | ForEach-Object { Write-Host "  - $_" }
    
    $response = Read-Host "Установить автоматически? (y/n)"
    if ($response -eq 'y') {
        # Установка Chocolatey если не установлен
        if (-not (Test-Command "choco")) {
            Write-Host "Установка Chocolatey..." -ForegroundColor Yellow
            Set-ExecutionPolicy Bypass -Scope Process -Force
            [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
            Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        }
        
        # Установка ПО через Chocolatey
        foreach ($software in $installChoices) {
            switch ($software) {
                "Python" {
                    Write-Host "Установка Python..." -ForegroundColor Yellow
                    choco install python -y --version=3.11.0
                    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
                }
                "Git" {
                    Write-Host "Установка Git..." -ForegroundColor Yellow
                    choco install git -y
                    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
                }
                "Docker" {
                    Write-Host "Установка Docker Desktop..." -ForegroundColor Yellow
                    choco install docker-desktop -y
                }
                "Make" {
                    Write-Host "Установка Make..." -ForegroundColor Yellow
                    choco install make -y
                }
            }
        }
    } else {
        Write-Host "Установите ПО вручную:" -ForegroundColor Yellow
        Write-Host "  Python: https://python.org" -ForegroundColor Cyan
        Write-Host "  Git: https://git-scm.com" -ForegroundColor Cyan
        Write-Host "  Docker Desktop: https://docker.com" -ForegroundColor Cyan
        Write-Host "  Make: choco install make" -ForegroundColor Cyan
    }
}

Write-Host ""

# Создание виртуального окружения
Write-Host "Создание виртуального окружения..." -ForegroundColor Yellow
python -m venv venv
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Виртуальное окружение создано" -ForegroundColor Green
} else {
    Write-Host "✗ Ошибка при создании виртуального окружения" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Активация виртуального окружения и установка зависимостей
Write-Host "Установка зависимостей..." -ForegroundColor Yellow
& "venv\Scripts\activate.ps1"
if ($?) {
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    
    Write-Host "✓ Зависимости установлены" -ForegroundColor Green
} else {
    Write-Host "✗ Ошибка активации виртуального окружения" -ForegroundColor Red
}

Write-Host ""

# Настройка pre-commit и DVC
Write-Host "Настройка инструментов разработки..." -ForegroundColor Yellow
pre-commit install
dvc init

Write-Host "✓ Настройка завершена" -ForegroundColor Green

Write-Host ""
Write-Host "=== Настройка завершена ===" -ForegroundColor Green
Write-Host ""
Write-Host "Для активации виртуального окружения выполните:" -ForegroundColor Cyan
Write-Host "  venv\Scripts\activate" -ForegroundColor White
Write-Host ""
Write-Host "Для запуска проекта выполните:" -ForegroundColor Cyan
Write-Host "  docker-compose up -d" -ForegroundColor White
Write-Host "  make train" -ForegroundColor White