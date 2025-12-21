function Get-FolderTree {
    param(
        [string]$Path = ".",
        [int]$MaxDepth = 3,
        [string]$Indent = "",
        [int]$CurrentDepth = 0
    )
    
    # Останавливаем рекурсию, если достигли максимальной глубины
    if ($CurrentDepth -ge $MaxDepth) { return }
    
    # Получаем папки, исключая venv и .venv
    $folders = Get-ChildItem -Path $Path -Directory | 
               Where-Object { $_.Name -notmatch '^(venv|\.venv)$' }
    
    foreach ($folder in $folders) {
        # Определяем префикс для отображения
        if ($CurrentDepth -eq 0) {
            "$($folder.Name)/"
        } else {
            "$Indent+---$($folder.Name)/"
        }
        
        # Рекурсивный вызов для подпапок
        Get-FolderTree -Path $folder.FullName `
                      -MaxDepth $MaxDepth `
                      -Indent ("    " + $Indent) `
                      -CurrentDepth ($CurrentDepth + 1)
    }
}

# Вызов функции с записью в файл
Get-FolderTree -Path "." -MaxDepth 3 | Out-File -FilePath "project_structure.txt"